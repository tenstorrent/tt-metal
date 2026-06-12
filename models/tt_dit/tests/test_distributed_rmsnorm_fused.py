# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unified DistributedRMSNorm fused-op benchmarks + correctness/determinism.

ONE file covering BOTH production models that drive
``ttnn.experimental.wan_fused_distributed_rmsnorm`` (the single-program fused
device op): **Wan2.2 14B** and **LTX-2.3 AV**. The fused op is identical for
both; the models differ only in (a) config shapes, (b) the RoPE convention, and
(c) what the *baseline* decomposes into. Those three axes are factored into a
single ``Cfg`` + a handful of ``model``-dispatched helpers, so the two former
per-model files (~1100 lines) collapse to this.

Per-model differences captured here
------------------------------------
* **RoPE**: Wan is BROADCAST (cos/sin ``(1,1,N,hd)`` shared across heads, fp32,
  replicated). LTX is PER-HEAD (cos/sin ``(1,H,N,hd)`` with the head axis TP-
  sharded, bf16). The fused op auto-detects per-head mode from
  ``rope_cos.shape[1] == num_heads_per_device``.
* **Baseline** (``use_device_op=False``): Wan's composite C++ op *fuses*
  weight+RoPE in-op (the production composite), so the Wan baseline is just that
  one op. LTX uses *unfused* trailing ops today: composite RMSNorm (+static
  weight) followed by a standalone ``ttnn.addcmul`` (block adaLN) or standalone
  ``rotary_embedding_llama`` (RoPE).
* **Configs**: Wan = 7 attention call sites (self/cross x SP4/8/32 + cross_k).
  LTX = 14 shape x fusion-pattern sites per TP, including block adaLN norms.

Tests
-----
* ``test_bench``      — traced composite/baseline vs fused timing + table/CSV.
                        Params: wan TP4-line (BH 2x4), wan TP8-ring (BH 1x8),
                        wan TP4 galaxy, ltx TP2/TP4 galaxy.
* ``test_corr_det``   — fused vs fp32-PyTorch reference AND vs the on-device
                        composite baseline, plus 10x determinism (bit-exact).
                        Galaxy only: wan TP4, ltx TP2/TP4.

Env hooks: ``RMS_BENCH_ONLY=cid[,cid]`` / ``CORR_ONLY=cid[,cid]`` restrict the
sweep; ``RMS_BENCH_METHODS=baseline,fused`` picks methods; ``CORR_FRESH_POB=1``
allocates a fresh stats buffer per determinism run (surfaces uninitialized-DRAM
reads); ``CORR_LOCALIZE=1`` prints which tokens diverge; ``WAN_GALAXY_LINKS``
overrides the galaxy fabric link count (default 4).
"""

from __future__ import annotations

import csv
import os as _os
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

from ..parallel.manager import CCLManager
from ..utils.mochi import get_rot_transformation_mat, stack_cos_sin
from ..utils.tensor import bf16_tensor, from_torch
from ..utils.test import line_params, ring_params

WAN = "wan"
LTX = "ltx"

NORM_EPS = 1e-6
TP_AXIS = 1  # axis of the 1xTP submesh that holds the TP cluster
GALAXY_LINKS = int(_os.getenv("WAN_GALAXY_LINKS", "4"))
_ITERS = {WAN: 100, LTX: 50}  # bench iterations per model
_PINGPONG = 2  # (pob, AG-sem) sets alternated across traced fused iters (skew absorber)

# Wan2.2 14B: full feature dim 5120, 40 heads, head_dim 128, broadcast RoPE.
_WAN_RAW = [
    # (cid, seq_len[per-device rows], use_rope) — the 7 Wan2.2 720p attention sites.
    ("self_sp4_N18944", 18944, True),
    ("self_sp8_N9472", 9472, True),
    ("self_sp32_N2368", 2368, True),
    ("cross_q_sp4_N18944", 18944, False),
    ("cross_q_sp8_N9472", 9472, False),
    ("cross_q_sp32_N2368", 2368, False),
    ("cross_k_prompt_L512", 512, False),
]

# LTX-2.3 AV: video dim 4096 (hd 128), audio dim 2048 (hd 64), 32 heads, per-head RoPE.
_LTX_V, _LTX_A, _LTX_HV, _LTX_HA = 4096, 2048, 128, 64


@dataclass(frozen=True)
class Cfg:
    cid: str
    model: str
    tp: int
    rows: int  # per-device row count
    dim: int  # full feature dim (reduced over via the TP all-gather)
    head_dim: int | None  # None => block norm (adaLN affine, no head split)
    rope: bool
    full_heads: int  # model NUM_HEADS (per-head RoPE table + reference reshape)
    broadcast_rope: bool  # True => Wan (cos/sin shared across heads); False => LTX per-head

    @property
    def is_block(self) -> bool:
        return self.head_dim is None

    @property
    def feat_local(self) -> int:
        return self.dim // self.tp

    @property
    def heads(self) -> int:  # heads per device (num_heads_per_device fed to the op)
        return 1 if self.is_block else self.feat_local // self.head_dim

    @property
    def pattern(self) -> str:
        if self.is_block:
            return "block+addcmul"
        return "qk+rope" if self.rope else "qk"


def _ltx_rows(kind: str, tp: int, stage: int = 0) -> int:
    """Per-device row count. TP=2<->SP=4, TP=4<->SP=8 (distributed_rmsnorm_av.md §0)."""
    if kind == "video":
        return {(2, 1): 2432, (2, 2): 9696, (4, 1): 1216, (4, 2): 4864}[(tp, stage)]
    if kind == "audio":  # audio N_local (stage-independent)
        return 64 if tp == 2 else 32
    if kind == "text":  # Gemma prompt length L, replicated across SP
        return 1024
    if kind == "audio_full":  # A->V K: audio ctx SP-gathered to full
        return 256
    raise ValueError(kind)


def _make_cfgs(model: str, tp: int) -> list[Cfg]:
    if model == WAN:
        return [Cfg(cid, WAN, tp, seq, 5120, 128, rope, 40, True) for (cid, seq, rope) in _WAN_RAW]
    v, a, hv, ha = _LTX_V, _LTX_A, _LTX_HV, _LTX_HA
    mk = lambda cid, rows, dim, hd, rope: Cfg(cid, LTX, tp, rows, dim, hd, rope, 32, False)  # noqa: E731
    return [
        # block norms: RMSNorm + adaLN addcmul (weight=1+scale, bias=shift)
        mk(f"tp{tp}_v_block_s1", _ltx_rows("video", tp, 1), v, None, False),
        mk(f"tp{tp}_v_block_s2", _ltx_rows("video", tp, 2), v, None, False),
        mk(f"tp{tp}_a_block", _ltx_rows("audio", tp), a, None, False),
        # QK + per-head RoPE (self-attn + A<->V cross)
        mk(f"tp{tp}_v_selfattn_qk_s1", _ltx_rows("video", tp, 1), v, hv, True),
        mk(f"tp{tp}_v_selfattn_qk_s2", _ltx_rows("video", tp, 2), v, hv, True),
        mk(f"tp{tp}_a_selfattn_qk", _ltx_rows("audio", tp), a, ha, True),
        mk(f"tp{tp}_a2v_videoQ_s1", _ltx_rows("video", tp, 1), a, ha, True),
        mk(f"tp{tp}_a2v_videoQ_s2", _ltx_rows("video", tp, 2), a, ha, True),
        mk(f"tp{tp}_a2v_audioK", _ltx_rows("audio_full", tp), a, ha, True),
        # QK, no RoPE (text cross-attn)
        mk(f"tp{tp}_v_textcross_q_s1", _ltx_rows("video", tp, 1), v, hv, False),
        mk(f"tp{tp}_v_textcross_q_s2", _ltx_rows("video", tp, 2), v, hv, False),
        mk(f"tp{tp}_v_textcross_k", _ltx_rows("text", tp), v, hv, False),
        mk(f"tp{tp}_a_textcross_q", _ltx_rows("audio", tp), a, ha, False),
        mk(f"tp{tp}_a_textcross_k", _ltx_rows("text", tp), a, ha, False),
    ]


def _select(cfgs: list[Cfg], env: str) -> list[Cfg]:
    only = _os.getenv(env, "")
    if not only:
        return cfgs
    wanted = {s.strip() for s in only.split(",") if s.strip()}
    return [c for c in cfgs if c.cid in wanted]


# ---------------------------------------------------------------------------
# Input construction (seeded; _torch_ref mirrors the SAME draw order)
# ---------------------------------------------------------------------------


def _build(submesh: ttnn.MeshDevice, cfg: Cfg) -> dict:
    torch.manual_seed(0)
    x = torch.randn((1, 1, cfg.rows, cfg.dim), dtype=torch.bfloat16)
    out = {"x": bf16_tensor(x, device=submesh, mesh_axis=TP_AXIS, shard_dim=-1)}

    if cfg.is_block:
        # adaLN: out = normed*(1+scale) + shift. scale/shift per-channel; the fused
        # op takes them as weight=(1+scale) / bias=shift (1,D) TP-sharded, while the
        # baseline addcmul takes (1,1,1,D) broadcast activations.
        scale = torch.randn(cfg.dim, dtype=torch.bfloat16)
        shift = torch.randn(cfg.dim, dtype=torch.bfloat16)
        scale_p1 = (scale.float() + 1.0).to(torch.bfloat16)
        out["weight"] = bf16_tensor(scale_p1.reshape(1, cfg.dim), device=submesh, mesh_axis=TP_AXIS, shard_dim=-1)
        out["bias"] = bf16_tensor(shift.reshape(1, cfg.dim), device=submesh, mesh_axis=TP_AXIS, shard_dim=-1)
        out["scale_b"] = bf16_tensor(scale_p1.reshape(1, 1, 1, cfg.dim), device=submesh, mesh_axis=TP_AXIS, shard_dim=-1)
        out["shift_b"] = bf16_tensor(shift.reshape(1, 1, 1, cfg.dim), device=submesh, mesh_axis=TP_AXIS, shard_dim=-1)
        return out

    w = torch.randn(cfg.dim, dtype=torch.bfloat16)
    out["weight"] = bf16_tensor(w.reshape(1, cfg.dim), device=submesh, mesh_axis=TP_AXIS, shard_dim=-1)
    if cfg.rope:
        if cfg.broadcast_rope:  # Wan: shared across heads, fp32, replicated
            cos_raw = torch.randn(1, cfg.rows, 1, cfg.head_dim // 2)
            sin_raw = torch.randn(1, cfg.rows, 1, cfg.head_dim // 2)
            cos_f, sin_f = stack_cos_sin(cos_raw, sin_raw)  # (1, rows, 1, hd)
            out["cos"] = from_torch(cos_f.permute(0, 2, 1, 3), device=submesh, dtype=ttnn.float32)  # (1,1,rows,hd)
            out["sin"] = from_torch(sin_f.permute(0, 2, 1, 3), device=submesh, dtype=ttnn.float32)
        else:  # LTX: per-head, bf16, head axis TP-sharded (matches production rotary tables)
            cos_raw = torch.randn(1, cfg.full_heads, cfg.rows, cfg.head_dim // 2)
            sin_raw = torch.randn(1, cfg.full_heads, cfg.rows, cfg.head_dim // 2)
            cos_f, sin_f = stack_cos_sin(cos_raw, sin_raw)  # (1, H, rows, hd)
            out["cos"] = from_torch(cos_f, device=submesh, dtype=ttnn.bfloat16, mesh_axes=[None, TP_AXIS, None, None])
            out["sin"] = from_torch(sin_f, device=submesh, dtype=ttnn.bfloat16, mesh_axes=[None, TP_AXIS, None, None])
        out["trans"] = bf16_tensor(get_rot_transformation_mat(), device=submesh)
    return out


def _torch_ref(cfg: Cfg) -> torch.Tensor:
    """fp32 PyTorch reference: full-feature RMSNorm + weight(+bias), then RoPE.
    Reseeds and draws in the SAME order as _build so the random sources match."""
    torch.manual_seed(0)
    x = torch.randn((1, 1, cfg.rows, cfg.dim), dtype=torch.bfloat16)
    xf = x.float().reshape(cfg.rows, cfg.dim)
    y = xf * (xf.pow(2).mean(-1, keepdim=True) + NORM_EPS).rsqrt()

    if cfg.is_block:
        scale = torch.randn(cfg.dim, dtype=torch.bfloat16)
        shift = torch.randn(cfg.dim, dtype=torch.bfloat16)
        return y * (scale.float() + 1.0) + shift.float()

    w = torch.randn(cfg.dim, dtype=torch.bfloat16).float()
    y = y * w
    if cfg.rope:
        h = cfg.full_heads
        if cfg.broadcast_rope:
            cos_raw = torch.randn(1, cfg.rows, 1, cfg.head_dim // 2)
            sin_raw = torch.randn(1, cfg.rows, 1, cfg.head_dim // 2)
            cf, sf = stack_cos_sin(cos_raw, sin_raw)
            cos = cf[0, :, 0, :].to(torch.bfloat16).float()  # (rows, hd), broadcast over heads
            sin = sf[0, :, 0, :].to(torch.bfloat16).float()
            cos = cos[:, None, :]
            sin = sin[:, None, :]
        else:
            cos_raw = torch.randn(1, h, cfg.rows, cfg.head_dim // 2)
            sin_raw = torch.randn(1, h, cfg.rows, cfg.head_dim // 2)
            cf, sf = stack_cos_sin(cos_raw, sin_raw)
            cos = cf.to(torch.bfloat16).float()[0].permute(1, 0, 2)  # (rows, H, hd)
            sin = sf.to(torch.bfloat16).float()[0].permute(1, 0, 2)
        yh = y.reshape(cfg.rows, h, cfg.head_dim)
        x0, x1 = yh[..., 0::2], yh[..., 1::2]
        rot = torch.stack([-x1, x0], dim=-1).flatten(-2)  # (-x1, x0, -x3, x2, ...)
        y = (yh * cos + rot * sin).reshape(cfg.rows, cfg.dim)
    return y


def _gather(out) -> torch.Tensor:
    ts = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)]
    if ts[0].ndim == 4 and ts[0].shape[1] > 1:  # head-split [1, H_dev, N, hd] -> concat heads
        full = torch.cat(ts, dim=1)  # [1, H, N, hd]
        return full[0].permute(1, 0, 2).reshape(full.shape[2], -1)  # [N, H*hd] head-contiguous
    return torch.cat([t.reshape(-1, t.shape[-1]) for t in ts], dim=-1)  # flat [N, feat]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    af, bf = a.flatten().float(), b.flatten().float()
    if torch.allclose(af, bf):
        return 1.0
    cov = torch.cov(torch.stack([af, bf]))
    sa, sb = float(cov[0, 0].sqrt()), float(cov[1, 1].sqrt())
    return float(cov[0, 1]) / (sa * sb) if sa * sb > 0 else float("nan")


# ---------------------------------------------------------------------------
# Op wiring (the fused device op is identical for both models)
# ---------------------------------------------------------------------------


def _fused_links(op_override):
    return op_override if op_override is not None else 2  # BH bench fused = 2-link MUX default


def _call_op(inp, submesh, sem, cfg, topology, *, use_device_op, pob, num_links, weight, bias, rope):
    return ttnn.experimental.wan_fused_distributed_rmsnorm(
        inp["x"],
        TP_AXIS,
        submesh,
        sem,
        topology=topology,
        epsilon=NORM_EPS,
        num_heads_per_device=cfg.heads,
        weight=weight,
        bias=bias,
        transformation_mat=(inp["trans"] if rope else None),
        rope_cos=(inp["cos"] if rope else None),
        rope_sin=(inp["sin"] if rope else None),
        dtype=None,
        persistent_output_buffer=pob if use_device_op else None,
        num_preferred_links=num_links,
        use_device_op=use_device_op,
    )


def _run_fused(inp, submesh, sem, cfg, topology, pob, op_override):
    bias = inp.get("bias") if cfg.is_block else None
    return _call_op(
        inp, submesh, sem, cfg, topology,
        use_device_op=True, pob=pob, num_links=_fused_links(op_override),
        weight=inp["weight"], bias=bias, rope=cfg.rope,
    )


def _run_baseline(inp, submesh, sem, cfg, topology, op_override):
    # op_override = composite link count: None on BH (default 1), GALAXY_LINKS on galaxy.
    if cfg.model == WAN:
        # Wan's composite C++ op fuses weight+RoPE in-op (the production baseline).
        return _call_op(
            inp, submesh, sem, cfg, topology,
            use_device_op=False, pob=None, num_links=op_override,
            weight=inp["weight"], bias=None, rope=cfg.rope,
        )
    # LTX: composite RMSNorm (+static weight for non-block) then the *unfused* trailing op.
    weight = None if cfg.is_block else inp["weight"]
    normed = _call_op(
        inp, submesh, sem, cfg, topology,
        use_device_op=False, pob=None, num_links=op_override,
        weight=weight, bias=None, rope=False,
    )
    if cfg.is_block:  # unfused adaLN: shift + normed*(1+scale)
        return ttnn.addcmul(inp["shift_b"], normed, inp["scale_b"], value=1.0)
    if cfg.rope:  # unfused standalone RoPE on the BHNE output (all-bf16 op)
        return ttnn.experimental.rotary_embedding_llama(normed, inp["cos"], inp["sin"], inp["trans"])
    return normed


def _make_pob(inp, submesh, cfg, num_links):
    # Pass weight/RoPE + num_links so the stats-buffer geometry (chunk/window sizing
    # and the num_workers link-rounding) matches the program; a mismatch corrupts the AG.
    return ttnn.experimental.wan_fused_distributed_rmsnorm_create_stats_buffer(
        inp["x"],
        TP_AXIS,
        submesh,
        num_heads_per_device=cfg.heads,
        num_links=num_links,
        weight=inp.get("weight"),
        transformation_mat=inp.get("trans"),
        rope_cos=inp.get("cos"),
        rope_sin=inp.get("sin"),
    )


def _submesh(mesh_device, tp):
    return mesh_device if tuple(mesh_device.shape) == (1, tp) else mesh_device.create_submesh(ttnn.MeshShape(1, tp))


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def _trace_and_time(submesh, run_ops, *, num_iters: int) -> float:
    """Capture each program in ``run_ops`` as its own trace and replay round-robin.

    A LIST ping-pongs distinct resource sets (each binds its own pob + AG sem) so a
    lagging device's late atomic-inc can't be clobbered by the op's end-of-op sem
    reset under traced replay (the many-chunk selfattn_qk_s2 hang). A single callable
    keeps plain single-trace timing.
    """
    if callable(run_ops):
        run_ops = [run_ops]
    n = len(run_ops)
    for run_op in run_ops:  # warmup + cold-compile each program
        run_op()
    ttnn.synchronize_device(submesh)
    trace_ids = []
    for run_op in run_ops:
        trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
        run_op()
        ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
        trace_ids.append(trace_id)
    ttnn.synchronize_device(submesh)
    t0 = time.perf_counter()
    for i in range(num_iters):
        ttnn.execute_trace(submesh, trace_ids[i % n], cq_id=0, blocking=False)
    ttnn.synchronize_device(submesh)
    elapsed_us = (time.perf_counter() - t0) * 1e6
    for trace_id in trace_ids:
        ttnn.release_trace(submesh, trace_id)
    return elapsed_us / num_iters


def _bench_cfg(submesh, ccl, cfg: Cfg, topology, op_override) -> dict:
    inp = _build(submesh, cfg)
    methods = [m.strip() for m in _os.getenv("RMS_BENCH_METHODS", "baseline,fused").split(",") if m.strip()]
    iters = _ITERS[cfg.model]
    t: dict = {}
    # A config may exceed L1 (e.g. per-head RoPE at video-self-attn width). Record the
    # failure and keep sweeping so the table shows what fuses vs. what doesn't.
    if "baseline" in methods:
        try:
            sem = ccl.get_ag_ping_pong_semaphore(TP_AXIS)
            t["baseline"] = _trace_and_time(
                submesh, lambda: _run_baseline(inp, submesh, sem, cfg, topology, op_override), num_iters=iters
            )
        except Exception as e:  # noqa: BLE001
            t["baseline_err"] = type(e).__name__
            logger.warning(f"{cfg.cid} baseline FAILED: {str(e)[:160]}")
    if "fused" in methods:
        try:
            links = _fused_links(op_override)
            pobs = [_make_pob(inp, submesh, cfg, links) for _ in range(_PINGPONG)]
            sems = [ccl.get_ag_ping_pong_semaphore(TP_AXIS) for _ in range(_PINGPONG)]
            run_ops = [
                (lambda p=p, s=s: _run_fused(inp, submesh, s, cfg, topology, p, op_override)) for p, s in zip(pobs, sems)
            ]
            t["fused"] = _trace_and_time(submesh, run_ops, num_iters=iters)
        except Exception as e:  # noqa: BLE001
            t["fused_err"] = type(e).__name__
            logger.warning(f"{cfg.cid} fused FAILED: {str(e)[:160]}")
    return t


def _print_table(rows: list[dict], title: str) -> None:
    cid_w = max(len("config_id"), max(len(r["cid"]) for r in rows))
    header = (
        f"{'config_id':<{cid_w}}  {'tp':>2} {'rows':>6} {'feat':>5} {'heads':>5} {'hd':>4} "
        f"{'pattern':<14} {'baseline us':>11} {'fused us':>9} {'speedup':>8}"
    )
    box = "=" * max(len(header), len(title))
    print("\n" + box + f"\n{title}\n" + box + f"\n{header}\n" + "-" * len(header))
    for r in rows:
        b, f = r.get("baseline"), r.get("fused")
        b_s = f"{b:>11.2f}" if b is not None else f"{r.get('baseline_err', 'n/a'):>11}"
        f_s = f"{f:>9.2f}" if f is not None else f"{r.get('fused_err', 'n/a'):>9}"
        sp_s = f"{b / f:>7.2f}x" if (b is not None and f) else f"{'-':>8}"
        print(
            f"{r['cid']:<{cid_w}}  {r['tp']:>2} {r['rows']:>6} {r['feat']:>5} {r['heads']:>5} "
            f"{(r['hd'] or 0):>4} {r['pattern']:<14} {b_s} {f_s} {sp_s}"
        )
    print(box)


def _write_csv(rows: list[dict], filename: str) -> None:
    fields = ["cid", "tp", "rows", "feat", "heads", "hd", "pattern", "baseline", "fused"]
    path = Path.cwd() / filename
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info(f"Wrote CSV: {path}")


# mesh, device_params, model, tp, topology, op_override(=galaxy links; None on BH)
_DP_GAL = {**line_params, "trace_region_size": 131072}
_BENCH_PARAMS = [
    ((2, 4), {**line_params, "trace_region_size": 90112}, WAN, 4, ttnn.Topology.Linear, None),
    ((1, 8), {**ring_params, "trace_region_size": 90112}, WAN, 8, ttnn.Topology.Ring, None),
    ((4, 8), _DP_GAL, WAN, 4, ttnn.Topology.Linear, GALAXY_LINKS),
    ((4, 8), _DP_GAL, LTX, 2, ttnn.Topology.Linear, GALAXY_LINKS),
    ((4, 8), _DP_GAL, LTX, 4, ttnn.Topology.Linear, GALAXY_LINKS),
]
_BENCH_IDS = ["wan_tp4_line", "wan_tp8_ring", "wan_tp4_galaxy", "ltx_tp2_galaxy", "ltx_tp4_galaxy"]


@pytest.mark.parametrize(
    ("mesh_device", "device_params", "model", "tp", "topology", "op_override"),
    [pytest.param(*p, id=i) for p, i in zip(_BENCH_PARAMS, _BENCH_IDS)],
    indirect=["mesh_device", "device_params"],
)
def test_bench(mesh_device, model, tp, topology, op_override):
    """Traced baseline vs fused timing for every config of one (model, topology)."""
    submesh = _submesh(mesh_device, tp)
    ccl = CCLManager(mesh_device=submesh, num_links=(op_override or 1), topology=topology)
    cfgs = _select(_make_cfgs(model, tp), "RMS_BENCH_ONLY")
    rows: list[dict] = []
    for cfg in cfgs:
        logger.info(f"=== [{model}] {cfg.cid} rows={cfg.rows} feat={cfg.feat_local} heads={cfg.heads} hd={cfg.head_dim} ===")
        rows.append(
            {"cid": cfg.cid, "tp": cfg.tp, "rows": cfg.rows, "feat": cfg.feat_local,
             "heads": cfg.heads, "hd": cfg.head_dim, "pattern": cfg.pattern,
             **_bench_cfg(submesh, ccl, cfg, topology, op_override)}
        )
    topo = "ring" if topology == ttnn.Topology.Ring else "line"
    _write_csv(rows, f"rms_bench_{model}_tp{tp}_{topo}.csv")
    _print_table(rows, f"{model.upper()} DistributedRMSNorm: baseline vs fused (TP={tp}, {topo})")


# ---------------------------------------------------------------------------
# Correctness + determinism (galaxy only)
# ---------------------------------------------------------------------------

_CORR_PARAMS = [
    ((4, 8), _DP_GAL, WAN, 4, ttnn.Topology.Linear, GALAXY_LINKS),
    ((4, 8), _DP_GAL, LTX, 2, ttnn.Topology.Linear, GALAXY_LINKS),
    ((4, 8), _DP_GAL, LTX, 4, ttnn.Topology.Linear, GALAXY_LINKS),
]
_CORR_IDS = ["wan_tp4", "ltx_tp2", "ltx_tp4"]


@pytest.mark.parametrize(
    ("mesh_device", "device_params", "model", "tp", "topology", "op_override"),
    [pytest.param(*p, id=i) for p, i in zip(_CORR_PARAMS, _CORR_IDS)],
    indirect=["mesh_device", "device_params"],
)
def test_corr_det(mesh_device, model, tp, topology, op_override):
    """Fused vs fp32-PyTorch reference AND vs on-device composite baseline, plus a
    10x bit-exact determinism check. CORR_FRESH_POB=1 allocates a fresh stats buffer
    per run (surfaces uninitialized-DRAM reads); CORR_LOCALIZE=1 prints divergent tokens."""
    submesh = _submesh(mesh_device, tp)
    ccl = CCLManager(mesh_device=submesh, num_links=(op_override or 1), topology=topology)
    ag = ccl.get_ag_ping_pong_semaphore(TP_AXIS)
    fresh_pob = _os.getenv("CORR_FRESH_POB") == "1"
    links = _fused_links(op_override)
    flagged = []

    for cfg in _select(_make_cfgs(model, tp), "CORR_ONLY"):
        inp = _build(submesh, cfg)
        pob = _make_pob(inp, submesh, cfg, links)
        ref = _torch_ref(cfg)
        comp = _gather(_run_baseline(inp, submesh, ag, cfg, topology, op_override))
        out0 = _gather(_run_fused(inp, submesh, ag, cfg, topology, pob, op_override))

        ndiff, maxdelta, worst_oi = 0, 0.0, None
        for _ in range(9):  # 10 fused runs total, same input -> must be bit-exact
            p = _make_pob(inp, submesh, cfg, links) if fresh_pob else pob
            oi = _gather(_run_fused(inp, submesh, ag, cfg, topology, p, op_override))
            d = (oi - out0).abs().max().item()
            if d > 0.0:
                ndiff += 1
                if d > maxdelta:
                    maxdelta, worst_oi = d, oi.clone()
            del oi
        det = ndiff == 0

        if ndiff and _os.getenv("CORR_LOCALIZE") == "1":
            rd = (worst_oi - out0).abs().amax(dim=-1)  # [N] per-token max-abs-diff
            bad = (rd > 1e-3).nonzero().flatten().tolist()
            span = f"[{min(bad)},{max(bad)}]" if bad else "[]"
            logger.info(f"  LOCALIZE {cfg.cid}: {len(bad)}/{out0.shape[0]} tokens differ; first10={bad[:10]} span={span}")

        pcc_ft, pcc_ct, pcc_fc = _pcc(out0, ref), _pcc(comp, ref), _pcc(out0, comp)
        denom = ref.abs().mean().clamp_min(1e-6)
        maxabs = (out0 - ref).abs().max().item()
        ratio = (out0.abs().mean() / denom).item()
        worstrow = ((out0 - ref).abs().mean(-1) / ref.abs().mean(-1).clamp_min(1e-6)).max().item()
        susp = (pcc_ft < 0.999) or (worstrow > 0.10) or (abs(ratio - 1.0) > 0.05) or (not det)
        if susp:
            flagged.append(cfg.cid)
        logger.info(
            f"RMSCORR {cfg.cid:<22} det={'OK' if det else 'FAIL'} pcc(F:torch)={pcc_ft * 100:.4f}% "
            f"pcc(base:torch)={pcc_ct * 100:.4f}% pcc(F:base)={pcc_fc * 100:.4f}% maxabs={maxabs:.4f} "
            f"ratio={ratio:.4f} worstrow={worstrow * 100:.2f}% det_ndiff={ndiff}/9 det_maxdelta={maxdelta:.4f}"
            f"{'  <-- SUSPICIOUS' if susp else ''}"
        )
    logger.info(f"RMSCORR [{model} tp{tp}] flagged: {flagged if flagged else 'NONE'}")
