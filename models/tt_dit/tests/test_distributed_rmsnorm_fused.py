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
from ..utils.tensor import bf16_tensor, float32_tensor, from_torch
from ..utils.test import line_params, ring_params

WAN = "wan"
LTX = "ltx"
FLUX = "flux"

# tp_axis = which MESH axis holds the TP cluster. Two strategies, both supported here:
#   * tp_axis=1: carve a 1xTP LINE submesh (one row of the parent) — used for line/BH benches.
#   * tp_axis=0: use the FULL 2D mesh; TP rides the 4-wide axis (a closed ring on the
#     galaxy torus) and everything replicates across the 8-wide axis 1. This is the
#     production "BH 4x8 ring" config (distributed_rmsnorm_av.md §0) and is the ONLY way
#     to get a real wrap link for Ring topology — a 1x4 sub-row of the 8-wide axis is open.
NORM_EPS = 1e-6
# Fabric links used by the op (= forwarder cores on the AG path). Wormhole galaxy = 4;
# Blackhole galaxy = 2 (torus, 2 links) — set WAN_GALAXY_LINKS=2 when running on BH.
GALAXY_LINKS = int(_os.getenv("WAN_GALAXY_LINKS", "4"))
_ITERS = {WAN: 100, LTX: 50, FLUX: 50}  # bench iterations per model
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

# FLUX: full dim 6144, 48 heads, head_dim 128, BROADCAST RoPE (rope_cos shape[1]==1),
# both per_head_norm=False (full-row norm + TP all-gather) and True (per-head RMSNorm
# over head_dim, no AG -> is_tp_1). Per-device rows given per (TP, SP).
_FLUX_DIM, _FLUX_HD, _FLUX_HEADS = 6144, 128, 48
_FLUX_ROWS = {
    8: (1024, 128, 4096, 16384),  # SP4/TP8
    4: (512, 64, 2048, 8192),  # SP8/TP4
    2: (512, 2048),  # TP=2 line (small + medium probe)
    1: (512, 2048),  # TP=1 line (local norm, no AG)
}


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
    per_head_norm: bool = False  # FLUX: RMSNorm over head_dim per head (no AG, is_tp_1)
    # Activation dtypes (model params stay bf16). Default bf16 = current behavior; the
    # stability sweep varies these to exercise the fp32 input/output codepaths.
    in_dtype: str = "bf16"  # "bf16" | "fp32"
    out_dtype: str = "bf16"  # "bf16" | "fp32"
    # Affine knobs for the sweep. "auto" = current behavior (qk->weight only,
    # block->adaLN weight+bias). "none"/"bcast" override for the qk path.
    # (per-token [N,D] weight/bias not yet built — see test_sweep notes.)
    weight_mode: str = "auto"  # "auto" | "none" | "bcast"
    bias_mode: str = "auto"  # "auto" | "none" | "bcast"

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
        if self.per_head_norm:
            return "perhead-norm+rope" if self.rope else "perhead-norm"
        return "qk+rope" if self.rope else "qk"


def _ltx_rows(kind: str, tp: int, stage: int = 0) -> int:
    """Per-device row count. TP=2<->SP=4, TP=4<->SP=8 (distributed_rmsnorm_av.md §0)."""
    if kind == "video":
        # TP=1 line mirrors the TP=2 per-device rows (same tile geometry, 2x feat).
        return {(1, 1): 2432, (1, 2): 9696, (2, 1): 2432, (2, 2): 9696, (4, 1): 1216, (4, 2): 4864}[(tp, stage)]
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
    if model == FLUX:
        # 4 per-device row counts. Broadcast RoPE, head_dim 128. Both per_head_norm
        # variants run: =False (whole-row RMSNorm) and =True (FLUX.2 QK-norm, per-head
        # reduce over head_dim, no AG). per_head_norm=True on ring_size>1 was fixed
        # (the compute/writer is_tp_1 mismatch — see ISSUE_per_head_norm_multidevice_
        # deadlock.md); the smallest shape (flux_tp4_N64_phn1) sits ~99.81% pcc(F:torch),
        # matching the composite baseline's own accuracy for that tiny per-head shape.
        return [
            Cfg(f"flux_tp{tp}_N{n}_phn{int(phn)}", FLUX, tp, n, _FLUX_DIM, _FLUX_HD, True, _FLUX_HEADS, True, phn)
            for n in _FLUX_ROWS[tp]
            for phn in (False, True)
        ]
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


def _build(submesh: ttnn.MeshDevice, cfg: Cfg, tp_axis: int) -> dict:
    torch.manual_seed(0)
    # Input activation dtype is cfg.in_dtype (model params below stay bf16). _torch_ref
    # draws x with the SAME dtype so the RNG streams stay in lock-step for determinism.
    in_fp32 = cfg.in_dtype == "fp32"
    x = torch.randn((1, 1, cfg.rows, cfg.dim), dtype=(torch.float32 if in_fp32 else torch.bfloat16))
    # Feature dim sharded across the TP mesh axis; the other mesh axis (if any) replicates.
    _act_tensor = float32_tensor if in_fp32 else bf16_tensor
    out = {"x": _act_tensor(x, device=submesh, mesh_axis=tp_axis, shard_dim=-1)}

    if cfg.is_block:
        # adaLN: out = normed*(1+scale) + shift. scale/shift per-channel; the fused
        # op takes them as weight=(1+scale) / bias=shift (1,D) TP-sharded, while the
        # baseline addcmul takes (1,1,1,D) broadcast activations.
        scale = torch.randn(cfg.dim, dtype=torch.bfloat16)
        shift = torch.randn(cfg.dim, dtype=torch.bfloat16)
        scale_p1 = (scale.float() + 1.0).to(torch.bfloat16)
        out["weight"] = bf16_tensor(scale_p1.reshape(1, cfg.dim), device=submesh, mesh_axis=tp_axis, shard_dim=-1)
        out["bias"] = bf16_tensor(shift.reshape(1, cfg.dim), device=submesh, mesh_axis=tp_axis, shard_dim=-1)
        out["scale_b"] = bf16_tensor(
            scale_p1.reshape(1, 1, 1, cfg.dim), device=submesh, mesh_axis=tp_axis, shard_dim=-1
        )
        out["shift_b"] = bf16_tensor(shift.reshape(1, 1, 1, cfg.dim), device=submesh, mesh_axis=tp_axis, shard_dim=-1)
        return out

    # qk path: weight on unless weight_mode="none"; optional broadcast bias when
    # bias_mode="bcast". Draw order (x, [w], [b], [cos/sin]) mirrors _torch_ref.
    if cfg.weight_mode != "none":
        w = torch.randn(cfg.dim, dtype=torch.bfloat16)
        out["weight"] = bf16_tensor(w.reshape(1, cfg.dim), device=submesh, mesh_axis=tp_axis, shard_dim=-1)
    if cfg.bias_mode == "bcast":
        b = torch.randn(cfg.dim, dtype=torch.bfloat16)
        out["bias"] = bf16_tensor(b.reshape(1, cfg.dim), device=submesh, mesh_axis=tp_axis, shard_dim=-1)
    if cfg.rope:
        if cfg.broadcast_rope:  # Wan: shared across heads, fp32, replicated
            cos_raw = torch.randn(1, cfg.rows, 1, cfg.head_dim // 2)
            sin_raw = torch.randn(1, cfg.rows, 1, cfg.head_dim // 2)
            cos_f, sin_f = stack_cos_sin(cos_raw, sin_raw)  # (1, rows, 1, hd)
            out["cos"] = from_torch(cos_f.permute(0, 2, 1, 3), device=submesh, dtype=ttnn.float32)  # (1,1,rows,hd)
            out["sin"] = from_torch(sin_f.permute(0, 2, 1, 3), device=submesh, dtype=ttnn.float32)
        else:  # LTX: per-head, bf16, head axis TP-sharded (matches production rotary tables)
            head_axes = [None, None, None, None]
            head_axes[1] = tp_axis  # shard tensor dim 1 (heads) onto the TP mesh axis
            cos_raw = torch.randn(1, cfg.full_heads, cfg.rows, cfg.head_dim // 2)
            sin_raw = torch.randn(1, cfg.full_heads, cfg.rows, cfg.head_dim // 2)
            cos_f, sin_f = stack_cos_sin(cos_raw, sin_raw)  # (1, H, rows, hd)
            out["cos"] = from_torch(cos_f, device=submesh, dtype=ttnn.bfloat16, mesh_axes=head_axes)
            out["sin"] = from_torch(sin_f, device=submesh, dtype=ttnn.bfloat16, mesh_axes=head_axes)
        out["trans"] = bf16_tensor(get_rot_transformation_mat(), device=submesh)
    return out


def _torch_ref(cfg: Cfg) -> torch.Tensor:
    """fp32 PyTorch reference: full-feature RMSNorm + weight(+bias), then RoPE.
    Reseeds and draws in the SAME order/dtype as _build so the random sources match."""
    torch.manual_seed(0)
    x = torch.randn((1, 1, cfg.rows, cfg.dim), dtype=(torch.float32 if cfg.in_dtype == "fp32" else torch.bfloat16))
    xf = x.float().reshape(cfg.rows, cfg.dim)
    if cfg.per_head_norm:
        # FLUX.2 per-head RMSNorm: reduce over head_dim per head (no cross-device AG),
        # not the full row. Each head normalized independently -> embarrassingly parallel.
        h, hd = cfg.full_heads, cfg.head_dim
        xh = xf.reshape(cfg.rows, h, hd)
        y = (xh * (xh.pow(2).mean(-1, keepdim=True) + NORM_EPS).rsqrt()).reshape(cfg.rows, cfg.dim)
    else:
        y = xf * (xf.pow(2).mean(-1, keepdim=True) + NORM_EPS).rsqrt()

    if cfg.is_block:
        scale = torch.randn(cfg.dim, dtype=torch.bfloat16)
        shift = torch.randn(cfg.dim, dtype=torch.bfloat16)
        return y * (scale.float() + 1.0) + shift.float()

    if cfg.weight_mode != "none":
        w = torch.randn(cfg.dim, dtype=torch.bfloat16).float()
        y = y * w
    if cfg.bias_mode == "bcast":
        b = torch.randn(cfg.dim, dtype=torch.bfloat16).float()
        y = y + b
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


def _gather(out, tp_axis: int) -> torch.Tensor:
    # On a 2D mesh the output has TP shards along tp_axis and (for the ring strategy)
    # identical replicas along the other axis. Pull ONE replica (other-axis coord 0),
    # ordered by TP position, then concat. For a 1xTP submesh the other axis is a
    # singleton so this selects every shard — backward compatible.
    coords = list(out.tensor_topology().mesh_coords())
    devs = ttnn.get_device_tensors(out)
    rep_axis = 1 - tp_axis
    sel = sorted(
        ((int(c[tp_axis]), d) for c, d in zip(coords, devs) if int(c[rep_axis]) == 0),
        key=lambda kv: kv[0],
    )
    ts = [ttnn.to_torch(d).float() for _, d in sel]
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


def _call_op(
    inp, submesh, sem, cfg, topology, tp_axis, *, use_device_op, pob, num_links, weight, bias, rope, per_head_norm=None
):
    if per_head_norm is None:
        per_head_norm = cfg.per_head_norm
    return ttnn.experimental.wan_fused_distributed_rmsnorm(
        inp["x"],
        tp_axis,
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
        dtype=(ttnn.float32 if cfg.out_dtype == "fp32" else None),  # None => bf16 (current default)
        persistent_output_buffer=pob if use_device_op else None,
        num_preferred_links=num_links,
        use_device_op=use_device_op,
        per_head_norm=per_head_norm,
    )


def _run_fused(inp, submesh, sem, cfg, topology, tp_axis, pob, op_override):
    bias = inp.get("bias")  # present for block adaLN and for qk bias_mode="bcast"; else None
    return _call_op(
        inp,
        submesh,
        sem,
        cfg,
        topology,
        tp_axis,
        use_device_op=True,
        pob=pob,
        num_links=_fused_links(op_override),
        weight=inp.get("weight"),  # None when weight_mode="none" (pure RMSNorm)
        bias=bias,
        rope=cfg.rope,
    )


def _run_baseline(inp, submesh, sem, cfg, topology, tp_axis, op_override):
    # op_override = composite link count: None on BH (default 1), GALAXY_LINKS on galaxy.
    if cfg.model in (WAN, FLUX):
        # Wan's composite C++ op fuses weight+RoPE in-op (the production baseline).
        # FLUX rides the same branch; the composite can't do per_head_norm, so PHN
        # rows benchmark against the full-row-norm composite of the SAME shape
        # (not a perfect match, but a decent relative-cost comparison).
        return _call_op(
            inp,
            submesh,
            sem,
            cfg,
            topology,
            tp_axis,
            use_device_op=False,
            pob=None,
            num_links=op_override,
            weight=inp.get("weight"),  # None when weight_mode="none" (pure RMSNorm)
            bias=None,
            rope=cfg.rope,
            per_head_norm=False,
        )
    # LTX: composite RMSNorm (+static weight for non-block) then the *unfused* trailing op.
    weight = None if cfg.is_block else inp["weight"]
    normed = _call_op(
        inp,
        submesh,
        sem,
        cfg,
        topology,
        tp_axis,
        use_device_op=False,
        pob=None,
        num_links=op_override,
        weight=weight,
        bias=None,
        rope=False,
    )
    if cfg.is_block:  # unfused adaLN: shift + normed*(1+scale)
        return ttnn.addcmul(inp["shift_b"], normed, inp["scale_b"], value=1.0)
    if cfg.rope:  # unfused standalone RoPE on the BHNE output (all-bf16 op)
        return ttnn.experimental.rotary_embedding_llama(normed, inp["cos"], inp["sin"], inp["trans"])
    return normed


def _make_pob(inp, submesh, cfg, num_links, tp_axis):
    # Pass weight/RoPE + num_links so the stats-buffer geometry (chunk/window sizing
    # and the num_workers link-rounding) matches the program; a mismatch corrupts the AG.
    return ttnn.experimental.wan_fused_distributed_rmsnorm_create_stats_buffer(
        inp["x"],
        tp_axis,
        submesh,
        num_heads_per_device=cfg.heads,
        per_head_norm=cfg.per_head_norm,
        num_links=num_links,
        weight=inp.get("weight"),
        transformation_mat=inp.get("trans"),
        rope_cos=inp.get("cos"),
        rope_sin=inp.get("sin"),
    )


def _resolve_submesh(mesh_device, tp, tp_axis, full_mesh=False):
    if tp_axis == 0:
        # TP rides the full mesh's axis-0 (the closed 4-wide ring); axis 1 replicates.
        assert (
            tuple(mesh_device.shape)[0] == tp
        ), f"tp_axis=0 needs mesh rows==tp, got {tuple(mesh_device.shape)} tp={tp}"
        return mesh_device
    if full_mesh:
        # TP rides the full mesh's axis-1 (the closed 8-wide ring); axis 0 replicates.
        assert (
            tuple(mesh_device.shape)[1] == tp
        ), f"full_mesh tp_axis=1 needs mesh cols==tp, got {tuple(mesh_device.shape)} tp={tp}"
        return mesh_device
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


def _bench_cfg(submesh, ccl, cfg: Cfg, topology, tp_axis, op_override) -> dict:
    inp = _build(submesh, cfg, tp_axis)
    methods = [m.strip() for m in _os.getenv("RMS_BENCH_METHODS", "baseline,fused").split(",") if m.strip()]
    iters = _ITERS[cfg.model]
    t: dict = {}
    # A config may exceed L1 (e.g. per-head RoPE at video-self-attn width). Record the
    # failure and keep sweeping so the table shows what fuses vs. what doesn't.
    if "baseline" in methods:
        try:
            sem = ccl.get_ag_ping_pong_semaphore(tp_axis)
            t["baseline"] = _trace_and_time(
                submesh, lambda: _run_baseline(inp, submesh, sem, cfg, topology, tp_axis, op_override), num_iters=iters
            )
        except Exception as e:  # noqa: BLE001
            t["baseline_err"] = type(e).__name__
            logger.warning(f"{cfg.cid} baseline FAILED: {str(e)[:160]}")
    if "fused" in methods:
        try:
            links = _fused_links(op_override)
            pobs = [_make_pob(inp, submesh, cfg, links, tp_axis) for _ in range(_PINGPONG)]
            sems = [ccl.get_ag_ping_pong_semaphore(tp_axis) for _ in range(_PINGPONG)]
            run_ops = [
                (lambda p=p, s=s: _run_fused(inp, submesh, s, cfg, topology, tp_axis, p, op_override))
                for p, s in zip(pobs, sems)
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


# mesh, device_params, model, tp, topology, op_override(=galaxy links; None on BH), tp_axis, full_mesh
# full_mesh=True keeps the whole 2D mesh with TP on axis 1 (the 8-wide closed ring) and
# axis 0 replicated — used for the FLUX TP=8 ring config (tp_axis=1 would otherwise carve
# a 1x8 LINE submesh).
_DP_GAL = {**line_params, "trace_region_size": 131072}
_DP_GAL_RING = {**ring_params, "trace_region_size": 131072}
_BENCH_PARAMS = [
    ((2, 4), {**line_params, "trace_region_size": 90112}, WAN, 4, ttnn.Topology.Linear, None, 1, False),
    ((1, 8), {**ring_params, "trace_region_size": 90112}, WAN, 8, ttnn.Topology.Ring, None, 1, False),
    ((4, 8), _DP_GAL, WAN, 4, ttnn.Topology.Linear, GALAXY_LINKS, 1, False),
    ((4, 8), _DP_GAL, LTX, 2, ttnn.Topology.Linear, GALAXY_LINKS, 1, False),
    ((4, 8), _DP_GAL, LTX, 4, ttnn.Topology.Linear, GALAXY_LINKS, 1, False),
    # TP=4 RING on the full-mesh 4-axis (replicate axis 1) — the production galaxy config.
    ((4, 8), _DP_GAL_RING, WAN, 4, ttnn.Topology.Ring, GALAXY_LINKS, 0, False),
    ((4, 8), _DP_GAL_RING, LTX, 4, ttnn.Topology.Ring, GALAXY_LINKS, 0, False),
    # FLUX: TP=4 ring on the 4-axis (replicate axis 1); TP=8 ring on the full-mesh 8-axis.
    ((4, 8), _DP_GAL_RING, FLUX, 4, ttnn.Topology.Ring, GALAXY_LINKS, 0, False),
    ((4, 8), _DP_GAL_RING, FLUX, 8, ttnn.Topology.Ring, GALAXY_LINKS, 1, True),
]
_BENCH_IDS = [
    "wan_tp4_line",
    "wan_tp8_ring",
    "wan_tp4_galaxy",
    "ltx_tp2_galaxy",
    "ltx_tp4_galaxy",
    "wan_tp4_ring",
    "ltx_tp4_ring",
    "flux_tp4_ring",
    "flux_tp8_ring",
]


@pytest.mark.parametrize(
    ("mesh_device", "device_params", "model", "tp", "topology", "op_override", "tp_axis", "full_mesh"),
    [pytest.param(*p, id=i) for p, i in zip(_BENCH_PARAMS, _BENCH_IDS)],
    indirect=["mesh_device", "device_params"],
)
def test_bench(mesh_device, model, tp, topology, op_override, tp_axis, full_mesh):
    """Traced baseline vs fused timing for every config of one (model, topology)."""
    submesh = _resolve_submesh(mesh_device, tp, tp_axis, full_mesh)
    ccl = CCLManager(mesh_device=submesh, num_links=(op_override or 1), topology=topology)
    cfgs = _select(_make_cfgs(model, tp), "RMS_BENCH_ONLY")
    rows: list[dict] = []
    for cfg in cfgs:
        logger.info(
            f"=== [{model}] {cfg.cid} rows={cfg.rows} feat={cfg.feat_local} heads={cfg.heads} hd={cfg.head_dim} ==="
        )
        row = {
            "cid": cfg.cid,
            "tp": cfg.tp,
            "rows": cfg.rows,
            "feat": cfg.feat_local,
            "heads": cfg.heads,
            "hd": cfg.head_dim,
            "pattern": cfg.pattern,
        }
        try:  # a per-config crash (e.g. input alloc / device hiccup) must not lose the table
            row.update(_bench_cfg(submesh, ccl, cfg, topology, tp_axis, op_override))
        except Exception as e:  # noqa: BLE001
            row["fused_err"] = row["baseline_err"] = type(e).__name__
            logger.warning(f"{cfg.cid} CONFIG FAILED: {str(e)[:160]}")
        rows.append(row)
    topo = "ring" if topology == ttnn.Topology.Ring else "line"
    _write_csv(rows, f"rms_bench_{model}_tp{tp}_{topo}.csv")
    _print_table(rows, f"{model.upper()} DistributedRMSNorm: baseline vs fused (TP={tp}, {topo})")


# ---------------------------------------------------------------------------
# Correctness + determinism (galaxy only)
# ---------------------------------------------------------------------------

_CORR_PARAMS = [
    ((4, 8), _DP_GAL, WAN, 4, ttnn.Topology.Linear, GALAXY_LINKS, 1, False),
    ((4, 8), _DP_GAL, LTX, 2, ttnn.Topology.Linear, GALAXY_LINKS, 1, False),
    ((4, 8), _DP_GAL, LTX, 4, ttnn.Topology.Linear, GALAXY_LINKS, 1, False),
    # TP=2 (1x2 LINE submesh): per-device feat is 2x the TP=4 case (Wan 2560, LTX video 2048).
    ((4, 8), _DP_GAL, WAN, 2, ttnn.Topology.Linear, GALAXY_LINKS, 1, False),
    ((4, 8), _DP_GAL, FLUX, 2, ttnn.Topology.Linear, GALAXY_LINKS, 1, False),
    # TP=1 (1x1 LINE submesh): local norm, no all-gather (ring_size==1). Exercises the
    # else-branch matmul reduce — the path the TP=1 JIT-compile fix unblocked.
    ((4, 8), _DP_GAL, WAN, 1, ttnn.Topology.Linear, GALAXY_LINKS, 1, False),
    ((4, 8), _DP_GAL, LTX, 1, ttnn.Topology.Linear, GALAXY_LINKS, 1, False),
    ((4, 8), _DP_GAL, FLUX, 1, ttnn.Topology.Linear, GALAXY_LINKS, 1, False),
    # TP=4 RING on the full-mesh 4-axis (replicate axis 1).
    ((4, 8), _DP_GAL_RING, WAN, 4, ttnn.Topology.Ring, GALAXY_LINKS, 0, False),
    ((4, 8), _DP_GAL_RING, LTX, 4, ttnn.Topology.Ring, GALAXY_LINKS, 0, False),
    # FLUX: TP=4 ring (4-axis) and TP=8 ring (full-mesh 8-axis); each runs PHN False+True.
    ((4, 8), _DP_GAL_RING, FLUX, 4, ttnn.Topology.Ring, GALAXY_LINKS, 0, False),
    ((4, 8), _DP_GAL_RING, FLUX, 8, ttnn.Topology.Ring, GALAXY_LINKS, 1, True),
]
_CORR_IDS = [
    "wan_tp4",
    "ltx_tp2",
    "ltx_tp4",
    "wan_tp2",
    "flux_tp2",
    "wan_tp1",
    "ltx_tp1",
    "flux_tp1",
    "wan_tp4_ring",
    "ltx_tp4_ring",
    "flux_tp4_ring",
    "flux_tp8_ring",
]


@pytest.mark.parametrize(
    ("mesh_device", "device_params", "model", "tp", "topology", "op_override", "tp_axis", "full_mesh"),
    [pytest.param(*p, id=i) for p, i in zip(_CORR_PARAMS, _CORR_IDS)],
    indirect=["mesh_device", "device_params"],
)
def test_corr_det(mesh_device, model, tp, topology, op_override, tp_axis, full_mesh):
    """Fused vs fp32-PyTorch reference AND vs on-device composite baseline, plus a
    10x bit-exact determinism check. CORR_FRESH_POB=1 allocates a fresh stats buffer
    per run (surfaces uninitialized-DRAM reads); CORR_LOCALIZE=1 prints divergent tokens."""
    submesh = _resolve_submesh(mesh_device, tp, tp_axis, full_mesh)
    fresh_pob = _os.getenv("CORR_FRESH_POB") == "1"
    links = _fused_links(op_override)
    flagged = []

    for cfg in _select(_make_cfgs(model, tp), "CORR_ONLY"):
        logger.info(
            f"=== [{model}] {cfg.cid} rows={cfg.rows} feat={cfg.feat_local} heads={cfg.heads} "
            f"hd={cfg.head_dim} rope={cfg.rope} ==="
        )
        try:  # isolate per-config so an OOM/hang on one shape characterizes it without losing the rest
            # Fresh CCL + AG semaphore per config: the fused op resets out_ready_sem at
            # end-of-op, so a config that throws mid-op can leave a shared semaphore in a
            # bad state and poison every later config in the sweep (the TP=2 nan cascade).
            # A per-config CCLManager/semaphore gives each shape independent AG state.
            ccl = CCLManager(mesh_device=submesh, num_links=(op_override or 1), topology=topology)
            ag = ccl.get_ag_ping_pong_semaphore(tp_axis)
            inp = _build(submesh, cfg, tp_axis)
            pob = _make_pob(inp, submesh, cfg, links, tp_axis)
            ref = _torch_ref(cfg)
            # Composite baseline is best-effort: the production distributed-rmsnorm op may
            # not support every shape (e.g. ring_size==1 at TP=1). A baseline that can't run
            # must not abort the fused-vs-torch correctness check, so swallow its failure.
            try:
                comp = _gather(_run_baseline(inp, submesh, ag, cfg, topology, tp_axis, op_override), tp_axis)
            except Exception as be:  # noqa: BLE001
                comp = None
                logger.warning(f"  baseline unavailable for {cfg.cid}: {type(be).__name__}: {str(be)[:120]}")
            out0 = _gather(_run_fused(inp, submesh, ag, cfg, topology, tp_axis, pob, op_override), tp_axis)

            ndiff, maxdelta, worst_oi = 0, 0.0, None
            for _ in range(9):  # 10 fused runs total, same input -> must be bit-exact
                p = _make_pob(inp, submesh, cfg, links, tp_axis) if fresh_pob else pob
                oi = _gather(_run_fused(inp, submesh, ag, cfg, topology, tp_axis, p, op_override), tp_axis)
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
                logger.info(
                    f"  LOCALIZE {cfg.cid}: {len(bad)}/{out0.shape[0]} tokens differ; first10={bad[:10]} span={span}"
                )

            pcc_ft = _pcc(out0, ref)
            pcc_ct = _pcc(comp, ref) if comp is not None else float("nan")
            pcc_fc = _pcc(out0, comp) if comp is not None else float("nan")
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
        except Exception as e:  # noqa: BLE001 — characterize (OOM / FATAL / hang-timeout), keep sweeping
            flagged.append(cfg.cid)
            logger.warning(f"RMSCORR {cfg.cid:<22} CONFIG FAILED: {type(e).__name__}: {str(e)[:220]}")
    logger.info(f"RMSCORR [{model} tp{tp}] flagged: {flagged if flagged else 'NONE'}")


@pytest.mark.parametrize(
    ("mesh_device", "device_params", "model", "tp", "topology", "op_override", "tp_axis", "full_mesh"),
    [pytest.param(*p, id=i) for p, i in zip(_CORR_PARAMS, _CORR_IDS)],
    indirect=["mesh_device", "device_params"],
)
def test_traced_corr(mesh_device, model, tp, topology, op_override, tp_axis, full_mesh):
    """Traced fused output must match the EAGER fused output (and torch ref).

    The perf path captures the op into a trace and replays it many times with one
    device sync. Trace replay does NOT re-run host-side semaphore init, so any
    op-managed semaphore not reset in-kernel accumulates across replays and the AG
    desyncs (fast-but-wrong or hang). This replays REPLAYS times before reading so
    a missing reset shows up as a low traced-vs-eager PCC, not just a lucky first
    replay. Eager correctness is covered by test_corr_det; this is the trace guard."""
    submesh = _resolve_submesh(mesh_device, tp, tp_axis, full_mesh)
    links = _fused_links(op_override)
    REPLAYS = int(_os.getenv("TRACED_CORR_REPLAYS", "20"))
    flagged = []

    for cfg in _select(_make_cfgs(model, tp), "CORR_ONLY"):
        logger.info(f"=== [{model}] {cfg.cid} rows={cfg.rows} feat={cfg.feat_local} ===")
        try:
            ccl = CCLManager(mesh_device=submesh, num_links=(op_override or 1), topology=topology)
            ag = ccl.get_ag_ping_pong_semaphore(tp_axis)
            inp = _build(submesh, cfg, tp_axis)
            ref = _torch_ref(cfg)
            # Eager reference output (this path is what test_corr_det validates).
            pob_e = _make_pob(inp, submesh, cfg, links, tp_axis)
            out_eager = _gather(_run_fused(inp, submesh, ag, cfg, topology, tp_axis, pob_e, op_override), tp_axis)
            # Traced output: warmup/compile, capture, then replay REPLAYS times.
            pob_t = _make_pob(inp, submesh, cfg, links, tp_axis)
            _run_fused(inp, submesh, ag, cfg, topology, tp_axis, pob_t, op_override)
            ttnn.synchronize_device(submesh)
            tid = ttnn.begin_trace_capture(submesh, cq_id=0)
            out_dev = _run_fused(inp, submesh, ag, cfg, topology, tp_axis, pob_t, op_override)
            ttnn.end_trace_capture(submesh, tid, cq_id=0)
            ttnn.synchronize_device(submesh)
            for _ in range(REPLAYS):
                ttnn.execute_trace(submesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(submesh)
            out_traced = _gather(out_dev, tp_axis)
            ttnn.release_trace(submesh, tid)

            pcc_te = _pcc(out_traced, out_eager)  # traced vs eager — the trace guard
            pcc_tr = _pcc(out_traced, ref)  # traced vs torch ref
            maxdelta = (out_traced - out_eager).abs().max().item()
            susp = (pcc_te < 0.9999) or (pcc_tr < 0.999)
            if susp:
                flagged.append(cfg.cid)
            logger.info(
                f"RMSTRACE {cfg.cid:<22} pcc(traced:eager)={pcc_te * 100:.4f}% "
                f"pcc(traced:torch)={pcc_tr * 100:.4f}% maxdelta(traced-eager)={maxdelta:.5f} "
                f"replays={REPLAYS}{'  <-- SUSPICIOUS' if susp else ''}"
            )
        except Exception as e:  # noqa: BLE001
            flagged.append(cfg.cid)
            logger.warning(f"RMSTRACE {cfg.cid:<22} CONFIG FAILED: {type(e).__name__}: {str(e)[:220]}")
    logger.info(f"RMSTRACE [{model} tp{tp}] flagged: {flagged if flagged else 'NONE'}")


# ===========================================================================
# Stability sweep — orthogonal option coverage (fused vs fp32-torch only)
# ===========================================================================
# Goal: ~15 min of broad coverage to confirm the op is stable. Sweeps seq / dim /
# heads×head_dim / weight / bias / rope / per_head_norm / in+out dtype, across
# tp1/2/4 LINE + tp4/tp8 RING. Each config: PCC(fused:fp32-torch) ≥ 0.999, sane
# maxabs/worstrow, N-run bit-exact determinism (SWEEP_DET_REPEATS, default 3), and
# no hang. SWEEP_ONLY=cid,... subsets. The composite baseline is NOT used (the fp32
# torch ref is the true oracle). KNOWN GAP: per-token [N,D] weight/bias (the compute
# supports it, but _build doesn't yet synthesize that layout) — broadcast/none only.

_SW_RING = ttnn.Topology.Ring
_SW_LINE = ttnn.Topology.Linear


def _swcfg(
    tp,
    cid,
    rows,
    feat_local,
    head_dim,
    *,
    rope=False,
    bcast_rope=True,
    phn=False,
    weight="auto",
    bias="auto",
    indt="bf16",
    outdt="bf16",
):  # noqa: E501
    dim = feat_local * tp
    full_heads = (feat_local // head_dim) * tp if head_dim else 1  # total heads across TP
    return Cfg(
        f"sw_{cid}_tp{tp}",
        "SWEEP",
        tp,
        rows,
        dim,
        head_dim,
        rope,
        full_heads,
        bcast_rope,
        phn,
        indt,
        outdt,
        weight,
        bias,
    )


def _sweep_cfgs(tp, topology):
    """Tiered config list for one (tp, topology). Dims scale as feat_local*tp so every
    config is valid at any tp (feat_local chosen divisible by head_dim)."""
    HD = 128
    is_ring = topology == _SW_RING
    c = []
    # ---- Tier 2: topology cross — 8 diverse configs on EVERY (tp, topology) ----
    c += [
        _swcfg(tp, "tiny_block", 32, 512, None),  # block adaLN, 1 worker
        _swcfg(tp, "small_qk", 512, 512, HD),  # qk + weight
        _swcfg(tp, "mid_rope", 2048, 1024, HD, rope=True),  # qk + broadcast rope
        _swcfg(tp, "large_qk", 8192, 768, HD),  # large, deep rounds
        _swcfg(tp, "adaln", 1024, 1024, None),  # block weight+bias
        _swcfg(tp, "perhead_norm", 512, 512, HD, rope=True, phn=True),  # FLUX per-head norm+rope
        _swcfg(tp, "perhead_rope", 1024, 512, 64, rope=True, bcast_rope=False),  # LTX per-head rope
        _swcfg(tp, "fp32_io", 512, 512, HD, indt="fp32", outdt="fp32"),  # fp32 in/out
    ]
    # ---- Tier 1: one-axis sweep — only on the base topology (tp4 ring) ----
    if tp == 4 and is_ring:
        for r in (32, 128, 512, 1216, 2048, 4096, 8192):  # seq
            c.append(_swcfg(tp, f"seq{r}", r, 1024, HD, rope=True))
        for fl in (512, 768, 1024, 1536, 2048):  # per-device width
            c.append(_swcfg(tp, f"fl{fl}", 2048, fl, HD, rope=True))
        for h, hd in ((4, 128), (8, 64), (8, 128), (12, 128), (16, 64)):  # heads × head_dim
            c.append(_swcfg(tp, f"h{h}x{hd}", 2048, h * hd, hd, rope=True))
        for wm in ("auto", "none"):  # weight × bias
            for bm in ("none", "bcast"):
                if wm == "none" and bm == "bcast":
                    continue  # op requires weight when bias is given (device_op validation)
                c.append(_swcfg(tp, f"w_{wm}_b_{bm}", 2048, 1024, HD, weight=wm, bias=bm))
        c += [
            _swcfg(tp, "rope_none", 2048, 1024, HD, rope=False),  # rope variants
            _swcfg(tp, "rope_bcast", 2048, 1024, HD, rope=True, bcast_rope=True),
            _swcfg(tp, "rope_perhead", 2048, 1024, HD, rope=True, bcast_rope=False),
            _swcfg(tp, "phn", 2048, 1024, HD, rope=True, phn=True),  # per_head_norm
            _swcfg(tp, "dt_fp32in", 2048, 1024, HD, indt="fp32"),  # dtype axis
            _swcfg(tp, "dt_fp32io", 2048, 1024, HD, indt="fp32", outdt="fp32"),
            _swcfg(tp, "dt_fp32out", 2048, 1024, HD, outdt="fp32"),
        ]
    # ---- Tier 3: interaction / stress, per (tp, topology) ----
    if is_ring:
        c.append(_swcfg(tp, "uneven1216", 1216, 1024, HD, rope=True))  # zero-present-round path
        c.append(_swcfg(tp, "uneven2368", 2368, 1024, HD, rope=True))
        if tp == 8:
            c.append(_swcfg(tp, "huge16384", 16384, 768, HD, rope=True))  # deep rounds, 8-wide ring
            c.append(_swcfg(tp, "wide_perhead", 2048, 2048, 128, rope=True, bcast_rope=False))  # wide L1
    c.append(_swcfg(tp, "fp32_large", 4096, 768, HD, rope=True, indt="fp32", outdt="fp32"))  # precision path
    return c


_SWEEP_PARAMS = [
    ((4, 8), _DP_GAL, 1, _SW_LINE, GALAXY_LINKS, 1, False),
    ((4, 8), _DP_GAL, 2, _SW_LINE, GALAXY_LINKS, 1, False),
    ((4, 8), _DP_GAL, 4, _SW_LINE, GALAXY_LINKS, 1, False),
    ((4, 8), _DP_GAL_RING, 4, _SW_RING, GALAXY_LINKS, 0, False),
    ((4, 8), _DP_GAL_RING, 8, _SW_RING, GALAXY_LINKS, 1, True),
]
_SWEEP_IDS = ["tp1_line", "tp2_line", "tp4_line", "tp4_ring", "tp8_ring"]


@pytest.mark.parametrize(
    ("mesh_device", "device_params", "tp", "topology", "op_override", "tp_axis", "full_mesh"),
    [pytest.param(*p, id=i) for p, i in zip(_SWEEP_PARAMS, _SWEEP_IDS)],
    indirect=["mesh_device", "device_params"],
)
def test_sweep(mesh_device, tp, topology, op_override, tp_axis, full_mesh):
    """Broad option-space stability sweep, fused vs fp32-torch ref. Run all 5 params
    for full coverage (~15 min); warm up first to dodge the cold-after-reset flake."""
    submesh = _resolve_submesh(mesh_device, tp, tp_axis, full_mesh)
    links = _fused_links(op_override)
    reps = int(_os.getenv("SWEEP_DET_REPEATS", "3"))
    topo = "ring" if topology == _SW_RING else "line"
    flagged = []
    cfgs = _select(_sweep_cfgs(tp, topology), "SWEEP_ONLY")
    for cfg in cfgs:
        try:
            ccl = CCLManager(mesh_device=submesh, num_links=(op_override or 1), topology=topology)
            ag = ccl.get_ag_ping_pong_semaphore(tp_axis)
            inp = _build(submesh, cfg, tp_axis)
            ref = _torch_ref(cfg)
            pob = _make_pob(inp, submesh, cfg, links, tp_axis)
            out0 = _gather(_run_fused(inp, submesh, ag, cfg, topology, tp_axis, pob, op_override), tp_axis)
            ndiff, maxdelta = 0, 0.0
            for _ in range(max(0, reps - 1)):  # bit-exact determinism across repeated launches
                oi = _gather(_run_fused(inp, submesh, ag, cfg, topology, tp_axis, pob, op_override), tp_axis)
                d = (oi - out0).abs().max().item()
                if d > 0.0:
                    ndiff += 1
                    maxdelta = max(maxdelta, d)
                del oi
            det = ndiff == 0
            pcc = _pcc(out0, ref)
            maxabs = (out0 - ref).abs().max().item()
            worstrow = ((out0 - ref).abs().mean(-1) / ref.abs().mean(-1).clamp_min(1e-6)).max().item()
            susp = (pcc < 0.999) or (worstrow > 0.10) or (not det)
            if susp:
                flagged.append(cfg.cid)
            logger.info(
                f"RMSSWEEP {cfg.cid:<26} det={'OK' if det else 'FAIL'} pcc={pcc * 100:.4f}% "
                f"maxabs={maxabs:.4f} worstrow={worstrow * 100:.2f}% in={cfg.in_dtype} out={cfg.out_dtype} "
                f"det_ndiff={ndiff}/{max(0, reps - 1)}{'  <-- SUSPICIOUS' if susp else ''}"
            )
        except Exception as e:  # noqa: BLE001 — characterize (OOM / hang-timeout) without losing the rest
            flagged.append(cfg.cid)
            logger.warning(f"RMSSWEEP {cfg.cid:<26} CONFIG FAILED: {type(e).__name__}: {str(e)[:200]}")
    logger.info(f"RMSSWEEP [{topo} tp{tp}] {len(cfgs)} cfgs — flagged: {flagged if flagged else 'NONE'}")
    assert not flagged, f"{len(flagged)} sweep config(s) flagged on {topo} tp{tp}: {flagged}"
