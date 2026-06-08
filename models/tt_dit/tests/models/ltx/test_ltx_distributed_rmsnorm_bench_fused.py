# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.3 AV DistributedRMSNorm benchmarks: unfused baseline vs fused device op.

Covers every distinct RMSNorm shape x fusion-pattern from the LTX-2.3 distilled
text->video+audio (AV) transformer block (see distributed_rmsnorm_av.md), at both
parallelism configs and both video stages:

  * TP=2 / SP=4  (writeup BH 2x4): video feat 2048, audio 1024, 16 heads/dev
  * TP=4 / SP=8  (writeup BH 4x8): video feat 1024, audio  512,  8 heads/dev

Three fusion patterns (all reduce over the FULL feature dim via the TP all-gather):

  * block norm  : RMSNorm (no static affine) + a separate adaLN
                  ``addcmul(shift, normed, 1+scale)`` -> FUSED as
                  ``weight=(1+scale)`` + ``bias=shift`` (both broadcast over rows),
                  num_heads_per_device=1, no RoPE.
  * QK + RoPE   : RMSNorm + static weight + create_heads + per-head RoPE -> FUSED
                  as weight + num_heads_per_device + per-head rope_cos/sin/trans_mat.
                  (self-attn Q/K, and both A<->V cross-attn Q/K.)
  * QK, no RoPE : RMSNorm + static weight + create_heads (text cross-attn). FUSED =
                  weight + num_heads_per_device. create_heads is already fused on
                  BOTH sides, so this just compares the norm op variants.

Baseline = composite RMSNorm (``use_device_op=False``) + the *unfused* trailing op
that LTX uses today (``ttnn.addcmul`` for block norms, standalone
``ttnn.experimental.rotary_embedding_llama`` for RoPE). Fused = the single device op
doing it in-kernel. Both traced + timed; speedup = baseline / fused.

LTX RoPE cos/sin are PER-HEAD: ``(1, num_heads, N, head_dim)`` with each freq
``repeat_interleave(2)``-duplicated (interleaved x0,x1) so the last dim is the full
head_dim. The fused op auto-detects per-head mode from ``rope_cos.shape[1] ==
num_heads_per_device``.

Run on the Wormhole Galaxy (4x8), carving a 1xTP LINE submesh (ring topology on a
sub-row of the 8-wide galaxy has no wrap link). SP only sets the per-device row
count, so benchmarking a single 1xTP group with rows=N_local is the exact per-group
op cost.
"""

from __future__ import annotations

import os as _os
import time
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn

from ....parallel.manager import CCLManager
from ....utils.mochi import get_rot_transformation_mat, stack_cos_sin
from ....utils.tensor import bf16_tensor, from_torch
from ....utils.test import line_params

# LTX-2.3 AV config (distributed_rmsnorm_av.md §0)
NUM_HEADS = 32
VIDEO_DIM = 4096
AUDIO_DIM = 2048
VIDEO_HEAD_DIM = 128  # 4096 / 32
AUDIO_HEAD_DIM = 64  # 2048 / 32  (audio path AND both A<->V cross-attns use dim=2048)
NORM_EPS = 1e-6
NUM_ITERS = 50
TP_AXIS = 1  # axis of the 1xTP submesh that holds the TP cluster
GALAXY_NUM_LINKS = int(_os.getenv("WAN_GALAXY_LINKS", "4"))


def _rows(kind: str, tp: int, stage: int) -> int:
    """Per-device row count. TP=2<->SP=4, TP=4<->SP=8 (writeup §0)."""
    if kind == "video":
        return {(2, 1): 2432, (2, 2): 9696, (4, 1): 1216, (4, 2): 4864}[(tp, stage)]
    if kind == "audio":  # audio N_local (stage-independent)
        return 64 if tp == 2 else 32
    if kind == "text":  # Gemma prompt length L, replicated across SP
        return 1024
    if kind == "audio_full":  # A->V K: audio ctx SP-gathered to full (151->256)
        return 256
    raise ValueError(kind)


@dataclass(frozen=True)
class LtxCfg:
    cid: str
    tp: int
    rows: int
    dim: int
    head_dim: int | None  # None => block norm (no head split, adaLN affine)
    rope: bool

    @property
    def is_block(self) -> bool:
        return self.head_dim is None

    @property
    def feat_local(self) -> int:
        return self.dim // self.tp

    @property
    def heads(self) -> int:
        return 1 if self.is_block else (self.dim // self.tp) // self.head_dim


def _make_configs() -> list[LtxCfg]:
    cfgs: list[LtxCfg] = []
    for tp in (2, 4):
        v, a, hv, ha = VIDEO_DIM, AUDIO_DIM, VIDEO_HEAD_DIM, AUDIO_HEAD_DIM
        cfgs += [
            # --- block norms: RMSNorm + adaLN addcmul (weight=1+scale, bias=shift) ---
            LtxCfg(f"tp{tp}_v_block_s1", tp, _rows("video", tp, 1), v, None, False),
            LtxCfg(f"tp{tp}_v_block_s2", tp, _rows("video", tp, 2), v, None, False),
            LtxCfg(f"tp{tp}_a_block", tp, _rows("audio", tp, 0), a, None, False),
            # --- QK + per-head RoPE (self-attn + A<->V cross) ---
            LtxCfg(f"tp{tp}_v_selfattn_qk_s1", tp, _rows("video", tp, 1), v, hv, True),
            LtxCfg(f"tp{tp}_v_selfattn_qk_s2", tp, _rows("video", tp, 2), v, hv, True),
            LtxCfg(f"tp{tp}_a_selfattn_qk", tp, _rows("audio", tp, 0), a, ha, True),
            LtxCfg(f"tp{tp}_a2v_videoQ_s1", tp, _rows("video", tp, 1), a, ha, True),
            LtxCfg(f"tp{tp}_a2v_videoQ_s2", tp, _rows("video", tp, 2), a, ha, True),
            LtxCfg(f"tp{tp}_a2v_audioK", tp, _rows("audio_full", tp, 0), a, ha, True),
            # --- QK, no RoPE (text cross-attn) ---
            LtxCfg(f"tp{tp}_v_textcross_q_s1", tp, _rows("video", tp, 1), v, hv, False),
            LtxCfg(f"tp{tp}_v_textcross_q_s2", tp, _rows("video", tp, 2), v, hv, False),
            LtxCfg(f"tp{tp}_v_textcross_k", tp, _rows("text", tp, 0), v, hv, False),
            LtxCfg(f"tp{tp}_a_textcross_q", tp, _rows("audio", tp, 0), a, ha, False),
            LtxCfg(f"tp{tp}_a_textcross_k", tp, _rows("text", tp, 0), a, ha, False),
        ]
    return cfgs


LTX_CONFIGS = _make_configs()


def _select(cfgs: list[LtxCfg]) -> list[LtxCfg]:
    """LTX_BENCH_ONLY=cid[,cid...] restricts the sweep (smoke hook)."""
    only = _os.getenv("LTX_BENCH_ONLY")
    if not only:
        return cfgs
    wanted = {s.strip() for s in only.split(",") if s.strip()}
    return [c for c in cfgs if c.cid in wanted]


def _build(submesh: ttnn.MeshDevice, cfg: LtxCfg) -> dict:
    torch.manual_seed(0)
    x = torch.randn((1, 1, cfg.rows, cfg.dim), dtype=torch.bfloat16)
    out = {"x": bf16_tensor(x, device=submesh, mesh_axis=TP_AXIS, shard_dim=-1)}

    if cfg.is_block:
        # adaLN modulation: out = normed*(1+scale) + shift. scale/shift are
        # per-channel, broadcast over the N tokens => weight/bias (1, D) sharded
        # on TP; baseline addcmul operands are (1,1,1,D) broadcast activations.
        scale = torch.randn(cfg.dim, dtype=torch.bfloat16)
        shift = torch.randn(cfg.dim, dtype=torch.bfloat16)
        scale_p1 = (scale.float() + 1.0).to(torch.bfloat16)
        out["weight"] = bf16_tensor(scale_p1.reshape(1, cfg.dim), device=submesh, mesh_axis=TP_AXIS, shard_dim=-1)
        out["bias"] = bf16_tensor(shift.reshape(1, cfg.dim), device=submesh, mesh_axis=TP_AXIS, shard_dim=-1)
        out["scale_b"] = bf16_tensor(
            scale_p1.reshape(1, 1, 1, cfg.dim), device=submesh, mesh_axis=TP_AXIS, shard_dim=-1
        )
        out["shift_b"] = bf16_tensor(shift.reshape(1, 1, 1, cfg.dim), device=submesh, mesh_axis=TP_AXIS, shard_dim=-1)
    else:
        w = torch.randn(cfg.dim, dtype=torch.bfloat16)
        out["weight"] = bf16_tensor(w.reshape(1, cfg.dim), device=submesh, mesh_axis=TP_AXIS, shard_dim=-1)
        if cfg.rope:
            # per-head cos/sin (1, NUM_HEADS, rows, head_dim), head axis sharded on TP
            cos_raw = torch.randn(1, NUM_HEADS, cfg.rows, cfg.head_dim // 2)
            sin_raw = torch.randn(1, NUM_HEADS, cfg.rows, cfg.head_dim // 2)
            cos_f, sin_f = stack_cos_sin(cos_raw, sin_raw)  # -> (1, NUM_HEADS, rows, head_dim)
            # Fused op consumes fp32 cos/sin (full precision); the standalone
            # rotary_embedding_llama baseline requires ALL inputs bf16, so keep a
            # bf16 copy for it.
            out["cos"] = from_torch(cos_f, device=submesh, dtype=ttnn.float32, mesh_axes=[None, TP_AXIS, None, None])
            out["sin"] = from_torch(sin_f, device=submesh, dtype=ttnn.float32, mesh_axes=[None, TP_AXIS, None, None])
            out["cos_bf16"] = from_torch(
                cos_f, device=submesh, dtype=ttnn.bfloat16, mesh_axes=[None, TP_AXIS, None, None]
            )
            out["sin_bf16"] = from_torch(
                sin_f, device=submesh, dtype=ttnn.bfloat16, mesh_axes=[None, TP_AXIS, None, None]
            )
            out["trans"] = bf16_tensor(get_rot_transformation_mat(), device=submesh)
    return out


def _norm_op(inp, submesh, ag_sem, cfg, *, use_device_op, pob=None):
    """The (composite or fused) distributed RMSNorm. The block-norm affine is fed
    only to the FUSED op (the baseline applies it via a separate addcmul); RoPE is
    fed only to the FUSED op (the baseline applies a standalone rotary op)."""
    if cfg.is_block:
        weight = inp["weight"] if use_device_op else None
        bias = inp["bias"] if use_device_op else None
    else:
        weight = inp["weight"]
        bias = None
    cos = inp.get("cos") if (cfg.rope and use_device_op) else None
    sin = inp.get("sin") if (cfg.rope and use_device_op) else None
    trans = inp.get("trans") if (cfg.rope and use_device_op) else None
    return ttnn.experimental.wan_fused_distributed_rmsnorm(
        inp["x"],
        TP_AXIS,
        submesh,
        ag_sem,
        topology=ttnn.Topology.Linear,
        epsilon=NORM_EPS,
        num_heads_per_device=cfg.heads,
        weight=weight,
        bias=bias,
        transformation_mat=trans,
        rope_cos=cos,
        rope_sin=sin,
        persistent_output_buffer=pob if use_device_op else None,
        num_preferred_links=GALAXY_NUM_LINKS,
        use_device_op=use_device_op,
    )


def _run_baseline(inp, submesh, ag_sem, cfg):
    normed = _norm_op(inp, submesh, ag_sem, cfg, use_device_op=False)
    if cfg.is_block:  # unfused adaLN: shift + normed*(1+scale)
        return ttnn.addcmul(inp["shift_b"], normed, inp["scale_b"], value=1.0)
    if cfg.rope:  # unfused standalone RoPE on the BHNE output (all-bf16 op)
        return ttnn.experimental.rotary_embedding_llama(normed, inp["cos_bf16"], inp["sin_bf16"], inp["trans"])
    return normed


def _run_fused(inp, submesh, ag_sem, cfg, pob):
    return _norm_op(inp, submesh, ag_sem, cfg, use_device_op=True, pob=pob)


def _trace_and_time(submesh, run_op, *, num_iters: int = NUM_ITERS) -> float:
    run_op()
    ttnn.synchronize_device(submesh)
    trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
    run_op()
    ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
    ttnn.synchronize_device(submesh)
    t0 = time.perf_counter()
    for _ in range(num_iters):
        ttnn.execute_trace(submesh, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(submesh)
    elapsed_us = (time.perf_counter() - t0) * 1e6
    ttnn.release_trace(submesh, trace_id)
    return elapsed_us / num_iters


def _bench_cfg(submesh, ag_sem, cfg: LtxCfg) -> dict:
    inp = _build(submesh, cfg)
    pob = ttnn.experimental.wan_fused_distributed_rmsnorm_create_stats_buffer(
        inp["x"], TP_AXIS, submesh, num_heads_per_device=cfg.heads
    )
    methods = [m.strip() for m in _os.getenv("LTX_BENCH_METHODS", "baseline,fused").split(",") if m.strip()]
    t = {}
    # A config may exceed L1 (e.g. per-head RoPE at video-self-attn width). Record
    # the failure and keep sweeping so the table shows what fuses vs. what doesn't.
    if "baseline" in methods:
        try:
            t["baseline"] = _trace_and_time(submesh, lambda: _run_baseline(inp, submesh, ag_sem, cfg))
        except Exception as e:  # noqa: BLE001
            t["baseline_err"] = type(e).__name__
            logger.warning(f"{cfg.cid} baseline FAILED: {str(e)[:160]}")
    if "fused" in methods:
        try:
            t["fused"] = _trace_and_time(submesh, lambda: _run_fused(inp, submesh, ag_sem, cfg, pob))
        except Exception as e:  # noqa: BLE001
            t["fused_err"] = type(e).__name__
            logger.warning(f"{cfg.cid} fused FAILED: {str(e)[:160]}")
    return t


def _pattern(cfg: LtxCfg) -> str:
    if cfg.is_block:
        return "block+addcmul"
    return "qk+rope" if cfg.rope else "qk"


def _print_table(rows: list[dict]) -> None:
    cid_w = max(len("config_id"), max(len(r["cid"]) for r in rows))
    header = (
        f"{'config_id':<{cid_w}}  {'tp':>2} {'rows':>5} {'feat':>5} {'heads':>5} {'hd':>4} "
        f"{'pattern':<14} {'baseline us':>11} {'fused us':>9} {'speedup':>8}"
    )
    print("\n" + "=" * len(header))
    print("LTX-2.3 AV DistributedRMSNorm: unfused baseline vs fused device op (WH Galaxy, LINE)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        b = r.get("baseline")
        f = r.get("fused")
        b_s = f"{b:>11.2f}" if b is not None else f"{r.get('baseline_err', 'n/a'):>11}"
        f_s = f"{f:>9.2f}" if f is not None else f"{r.get('fused_err', 'n/a'):>9}"
        sp_s = f"{b / f:>7.2f}x" if (b is not None and f) else f"{'-':>8}"
        print(
            f"{r['cid']:<{cid_w}}  {r['tp']:>2} {r['rows']:>5} {r['feat']:>5} {r['heads']:>5} "
            f"{(r['hd'] or 0):>4} {r['pattern']:<14} {b_s} {f_s} {sp_s}"
        )
    print("=" * len(header))


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [((4, 8), {**line_params, "trace_region_size": 131072})],
    indirect=True,
    ids=["wh_galaxy_4x8_line"],
)
@pytest.mark.parametrize("tp", [2, 4], ids=["tp2", "tp4"])
def test_ltx_rmsnorm_bench_galaxy(mesh_device: ttnn.MeshDevice, tp: int) -> None:
    """Sweep the distinct LTX-2.3 AV RMSNorm configs for one TP through the unfused
    baseline and the fused device op; print a comparison table. Parametrized by TP
    (separate process each) so we open one 1xTP submesh per run — mixing a 1x2 and
    1x4 submesh in one process wedges the galaxy fabric."""
    cfgs = [c for c in _select(LTX_CONFIGS) if c.tp == tp]
    if not cfgs:
        pytest.skip(f"no configs for TP={tp}")
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, tp))
    ccl = CCLManager(mesh_device=submesh, num_links=GALAXY_NUM_LINKS, topology=ttnn.Topology.Linear)
    ag_sem = ccl.get_ag_ping_pong_semaphore(TP_AXIS)
    rows: list[dict] = []
    for cfg in cfgs:
        logger.info(
            f"=== [LTX] {cfg.cid}  rows={cfg.rows} feat={cfg.feat_local} heads={cfg.heads} hd={cfg.head_dim} ==="
        )
        t = _bench_cfg(submesh, ag_sem, cfg)
        rows.append(
            {
                "cid": cfg.cid,
                "tp": cfg.tp,
                "rows": cfg.rows,
                "feat": cfg.feat_local,
                "heads": cfg.heads,
                "hd": cfg.head_dim,
                "pattern": _pattern(cfg),
                **t,
            }
        )
    _print_table(rows)
