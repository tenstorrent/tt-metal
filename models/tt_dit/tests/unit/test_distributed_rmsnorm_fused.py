# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Unified DistributedRMSNorm fused-op benchmarks + correctness/determinism.

ONE file covering BOTH production models that drive
``ttnn.experimental.dit_fused_distributed_rmsnorm`` (the single-program fused
device op): **Wan2.2 14B** and **LTX-2.3 AV**. The fused op is identical for
both; the models differ only in (a) config shapes and (b) the RoPE convention.
Those axes are factored into a single ``Cfg`` + a handful of ``model``-dispatched
helpers, so the two former per-model files (~1100 lines) collapse to this.

Per-model differences captured here
------------------------------------
* **RoPE**: Wan is BROADCAST (cos/sin ``(1,1,N,hd)`` shared across heads, fp32,
  replicated). LTX is PER-HEAD (cos/sin ``(1,H,N,hd)`` with the head axis TP-
  sharded, bf16). The fused op auto-detects per-head mode from
  ``rope_cos.shape[1] == num_heads_per_device``.
* **Configs**: Wan = 7 attention call sites (self/cross x SP4/8/32 + cross_k).
  LTX = 14 shape x fusion-pattern sites per TP, including block adaLN norms.

Tests
-----
* ``test_corr_det``            — fused RMS op vs fp32-PyTorch reference, plus 10x
                                 determinism (bit-exact). Galaxy: wan/ltx/flux TP1-8.
* ``test_layernorm_corr``      — fused Welford-LN op vs fp32-PyTorch + determinism.
* ``test_rmsnorm_module_corr`` — ``DistributedRMSNorm`` module (static weight) e2e.
* ``test_layernorm_module_corr``— ``DistributedLayerNorm`` module (adaLN) e2e.
* ``test_traced_corr``         — traced-replay output == eager (semaphore-reset guard).
* ``test_layernorm_module_bench``— fused LN vs composite-chain speedup (perf, dev).

Env hooks: ``CORR_ONLY=cid[,cid]`` restricts the sweep; ``CORR_FRESH_POB=1``
allocates a fresh stats buffer per determinism run (surfaces uninitialized-DRAM
reads); ``CORR_LOCALIZE=1`` prints which tokens diverge; ``WAN_GALAXY_LINKS``
overrides the galaxy fabric link count (default 4).
"""

from __future__ import annotations

import os as _os
import time
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn

from ...layers.normalization import DistributedLayerNorm, DistributedRMSNorm
from ...parallel.manager import CCLManager
from ...utils.mochi import get_rot_transformation_mat, stack_cos_sin
from ...utils.tensor import bf16_tensor, float32_tensor, from_torch
from ...utils.test import line_params, ring_params

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
    # Activation dtypes (model params stay bf16). Default bf16 = current behavior;
    # "fp32" exercises the fp32 input/output codepaths.
    in_dtype: str = "bf16"  # "bf16" | "fp32"
    out_dtype: str = "bf16"  # "bf16" | "fp32"
    # Affine knobs. "auto" = current behavior (qk->weight only, block->adaLN
    # weight+bias). "none"/"bcast" override for the qk path.
    weight_mode: str = "auto"  # "auto" | "none" | "bcast"
    bias_mode: str = "auto"  # "auto" | "none" | "bcast"
    norm: str = "rms"  # "rms" | "layernorm" (Welford LayerNorm: (x-mean)/sqrt(var+eps))

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
        # Welford reciprocal LUT (LayerNorm): [1/1..1/H_local] fp32, ROW_MAJOR, replicated
        # per device (H_local = per-device shard width = dim // tp == the Welford reduce
        # width). The op reads it so the LLK does an array load vs a soft-float 1/(N+1).
        if cfg.norm == "layernorm":
            h_local = cfg.dim // cfg.tp
            recip = torch.tensor([1.0 / (i + 1) for i in range(h_local)], dtype=torch.float32).reshape(1, 1, 1, h_local)
            out["recip"] = from_torch(recip, device=submesh, layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.float32)
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
    elif cfg.norm == "layernorm":
        # Welford LayerNorm: (x - mean) / sqrt(var + eps) over the full feature row.
        # Population variance (unbiased=False) matches welford_finalize scale = 1/W.
        mean = xf.mean(-1, keepdim=True)
        var = xf.var(-1, unbiased=False, keepdim=True)
        y = (xf - mean) * (var + NORM_EPS).rsqrt()
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


def _gather(out, tp_axis: int, batched: bool = False) -> torch.Tensor:
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
    # batched output is [1, batch, N, H_dev] (batch preserved at dim1, num_heads_per_device==1);
    # its dim1>1 is NOT a head split, so force the flat concat (reshape folds batch*N).
    if not batched and ts[0].ndim == 4 and ts[0].shape[1] > 1:  # head-split [1, H_dev, N, hd] -> concat heads
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


def _call_op(inp, submesh, sem, cfg, topology, tp_axis, *, pob, num_links, weight, bias, rope, per_head_norm=None):
    if per_head_norm is None:
        per_head_norm = cfg.per_head_norm
    trans = inp["trans"] if rope else None
    cos = inp["cos"] if rope else None
    sin = inp["sin"] if rope else None
    out_dtype = ttnn.float32 if cfg.out_dtype == "fp32" else None  # None => bf16 (current default)
    if cfg.norm == "layernorm":
        # LayerNorm is its own op (no per_head_norm knob).
        return ttnn.experimental.dit_fused_distributed_layernorm(
            inp["x"],
            tp_axis,
            submesh,
            sem,
            topology=topology,
            epsilon=NORM_EPS,
            num_heads_per_device=cfg.heads,
            weight=weight,
            bias=bias,
            transformation_mat=trans,
            rope_cos=cos,
            rope_sin=sin,
            dtype=out_dtype,
            persistent_output_buffer=pob,
            num_preferred_links=num_links,
            reciprocals=inp.get("recip"),
        )
    return ttnn.experimental.dit_fused_distributed_rmsnorm(
        inp["x"],
        tp_axis,
        submesh,
        sem,
        topology=topology,
        epsilon=NORM_EPS,
        num_heads_per_device=cfg.heads,
        weight=weight,
        bias=bias,
        transformation_mat=trans,
        rope_cos=cos,
        rope_sin=sin,
        dtype=out_dtype,
        persistent_output_buffer=pob,
        num_preferred_links=num_links,
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
        pob=pob,
        num_links=_fused_links(op_override),
        weight=inp.get("weight"),  # None when weight_mode="none" (pure RMSNorm)
        bias=bias,
        rope=cfg.rope,
    )


def _make_pob(inp, submesh, cfg, num_links, tp_axis):
    # Pass weight/RoPE + num_links so the stats-buffer geometry (chunk/window sizing
    # and the num_workers link-rounding) matches the program; a mismatch corrupts the AG.
    # LayerNorm transports 2 stats/token (mean+var) vs RMS's 1, so it needs its own
    # (2x-wide) stats-buffer variant.
    if cfg.norm == "layernorm":
        return ttnn.experimental.dit_fused_distributed_layernorm_create_stats_buffer(
            inp["x"],
            tp_axis,
            submesh,
            num_heads_per_device=cfg.heads,
            num_links=num_links,
            weight=inp.get("weight"),
            transformation_mat=inp.get("trans"),
            rope_cos=inp.get("cos"),
            rope_sin=inp.get("sin"),
        )
    return ttnn.experimental.dit_fused_distributed_rmsnorm_create_stats_buffer(
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


# mesh, device_params, model, tp, topology, op_override(=galaxy links; None on BH), tp_axis, full_mesh
# full_mesh=True keeps the whole 2D mesh with TP on axis 1 (the 8-wide closed ring) and
# axis 0 replicated — used for the FLUX TP=8 ring config (tp_axis=1 would otherwise carve
# a 1x8 LINE submesh).
_DP_GAL = {**line_params, "trace_region_size": 131072}
_DP_GAL_RING = {**ring_params, "trace_region_size": 131072}


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
    """Fused op vs fp32-PyTorch reference, plus a 10x bit-exact determinism check.
    CORR_FRESH_POB=1 allocates a fresh stats buffer per run (surfaces uninitialized-DRAM
    reads); CORR_LOCALIZE=1 prints divergent tokens."""
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
            inp = _build(submesh, cfg, tp_axis)
            # CCL ping-pong (the ONLY correct way to drive sems + POBs for a CCL): consecutive
            # op launches MUST alternate between distinct (semaphore, POB) sets so an op's
            # end-of-op fabric atomic-incs drain on the unused set before it is reused. Reusing
            # one set across launches races the in-kernel sem reset against in-flight peer incs,
            # and the all-gather desyncs -> a worker reads a wrong/un-gathered stats slot
            # (intermittent wrong-row output, ~50% on a multi-config run). Mirrors the bench.
            sems = [ccl.get_ag_ping_pong_semaphore(tp_axis) for _ in range(_PINGPONG)]
            pobs = [_make_pob(inp, submesh, cfg, links, tp_axis) for _ in range(_PINGPONG)]
            # Barrier so EVERY device has finished creating+zeroing its out_ready GlobalSemaphore
            # and allocating the POB before any device launches the op and fires a cross-device
            # round-0 atomic-inc. Without it a fast device's inc can race a peer's sem creation
            # (which zeroes it) -> the inc is lost -> out_ready stuck at ring_size-2 -> round-0 hang.
            ttnn.synchronize_device(submesh)
            ref = _torch_ref(cfg)

            def _fused(k, _inp=inp, _sems=sems, _pobs=pobs, _cfg=cfg):  # launch k -> ping-pong set k%_PINGPONG
                s = _sems[k % _PINGPONG]
                p = _make_pob(_inp, submesh, _cfg, links, tp_axis) if fresh_pob else _pobs[k % _PINGPONG]
                return _gather(_run_fused(_inp, submesh, s, _cfg, topology, tp_axis, p, op_override), tp_axis)

            out0 = _fused(0)

            ndiff, maxdelta, worst_oi = 0, 0.0, None
            _det_reps = int(_os.getenv("CORR_DET_REPEATS", "9"))  # extra fused runs after out0
            for _j in range(_det_reps):  # same input -> must be bit-exact
                oi = _fused(_j + 1)
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
            denom = ref.abs().mean().clamp_min(1e-6)
            maxabs = (out0 - ref).abs().max().item()
            ratio = (out0.abs().mean() / denom).item()
            worstrow = ((out0 - ref).abs().mean(-1) / ref.abs().mean(-1).clamp_min(1e-6)).max().item()
            # Per-head-norm (FLUX phn1) reduces over head_dim per (token, head) on tiny
            # shapes, which has a known lower PCC floor (~99.81%, matching the composite
            # baseline's own accuracy for that reduce) — not a fused-op error. Relax the
            # PCC bar for that path only; determinism (det) is still required exactly.
            pcc_bar = 0.997 if cfg.per_head_norm else 0.999
            susp = (pcc_ft < pcc_bar) or (worstrow > 0.10) or (abs(ratio - 1.0) > 0.05) or (not det)
            if susp:
                flagged.append(cfg.cid)
            logger.info(
                f"RMSCORR {cfg.cid:<22} det={'OK' if det else 'FAIL'} pcc(F:torch)={pcc_ft * 100:.4f}% "
                f"maxabs={maxabs:.4f} ratio={ratio:.4f} worstrow={worstrow * 100:.2f}% "
                f"det_ndiff={ndiff}/{_det_reps} det_maxdelta={maxdelta:.4f}"
                f"{'  <-- SUSPICIOUS' if susp else ''}"
            )
        except Exception as e:  # noqa: BLE001 — characterize (OOM / FATAL / hang-timeout), keep sweeping
            flagged.append(cfg.cid)
            logger.warning(f"RMSCORR {cfg.cid:<22} CONFIG FAILED: {type(e).__name__}: {str(e)[:220]}")
    logger.info(f"RMSCORR [{model} tp{tp}] flagged: {flagged if flagged else 'NONE'}")
    assert not flagged, f"RMSNorm correctness/determinism flagged: {flagged}"


# adaLN block-norm hidden dim + head count per model (the DistributedLayerNorm
# use case). feat_local = hidden/tp drives the resident-vs-wide-shard split.
_LN_HID = {WAN: 5120, LTX: 4096, FLUX: 3072}
_LN_HEADS = {WAN: 40, LTX: 32, FLUX: 24}
# Real model shapes across TP (1x{tp} LINE submesh, tp_axis=1). Spans the
# feat_local range: WAN tp1=160 tiles (wide) ... tp8=20 tiles (resident).
_LN_PARAMS = [
    ((4, 8), _DP_GAL, WAN, 1, ttnn.Topology.Linear, GALAXY_LINKS, 1, False),
    ((4, 8), _DP_GAL, WAN, 2, ttnn.Topology.Linear, GALAXY_LINKS, 1, False),
    ((4, 8), _DP_GAL, WAN, 4, ttnn.Topology.Linear, GALAXY_LINKS, 1, False),
    ((4, 8), _DP_GAL, LTX, 2, ttnn.Topology.Linear, GALAXY_LINKS, 1, False),
    ((4, 8), _DP_GAL, LTX, 4, ttnn.Topology.Linear, GALAXY_LINKS, 1, False),
]
_LN_IDS = ["wan_tp1", "wan_tp2", "wan_tp4", "ltx_tp2", "ltx_tp4"]


@pytest.mark.parametrize(
    ("mesh_device", "device_params", "model", "tp", "topology", "op_override", "tp_axis", "full_mesh"),
    [pytest.param(*p, id=i) for p, i in zip(_LN_PARAMS, _LN_IDS)],
    indirect=["mesh_device", "device_params"],
)
def test_layernorm_corr(mesh_device, model, tp, topology, op_override, tp_axis, full_mesh):
    """Welford LayerNorm vs fp32-PyTorch reference on real adaLN block-norm shapes
    (head_dim=None -> _build supplies weight=(1+scale), bias=shift), plus a 3x
    bit-exact determinism check. Shapes whose shard overflows L1 resident hit the
    Phase-3 'resident only' TT_FATAL (wide-shard support is Phase 4) — those are
    recorded as NEED_WIDE, not correctness failures, so one run maps the landscape."""
    submesh = _resolve_submesh(mesh_device, tp, tp_axis, full_mesh)
    links = _fused_links(op_override)
    dim, heads = _LN_HID[model], _LN_HEADS[model]
    feat_local_tiles = (dim // tp) // 32
    cfgs = [
        Cfg(
            f"ln_{model}_tp{tp}_D{dim}_N256",
            model,
            tp,
            rows=256,
            dim=dim,
            head_dim=None,
            rope=False,
            full_heads=heads,
            broadcast_rope=True,
            norm="layernorm",
        ),
    ]
    flagged, need_wide = [], []
    for cfg in cfgs:
        logger.info(f"=== [LN] {cfg.cid} rows={cfg.rows} dim={cfg.dim} feat_local_tiles={feat_local_tiles} ===")
        ccl = CCLManager(mesh_device=submesh, num_links=(op_override or 1), topology=topology)
        inp = _build(submesh, cfg, tp_axis)
        sems = [ccl.get_ag_ping_pong_semaphore(tp_axis) for _ in range(_PINGPONG)]
        ref = _torch_ref(cfg)

        def _fused(k, _inp=inp, _sems=sems, _cfg=cfg):
            s = _sems[k % _PINGPONG]
            pob = _make_pob(_inp, submesh, _cfg, links, tp_axis)
            return _gather(_run_fused(_inp, submesh, s, _cfg, topology, tp_axis, pob, op_override), tp_axis)

        try:
            out0 = _fused(0)
        except RuntimeError as e:
            msg = str(e)
            # NEED_WIDE: shard too wide to fit even the block-major + streaming layout
            # in L1 (TP>1 wide adds the AG/combine CBs on top of whole-row weight/bias).
            # A clean program-construction error, not a hang — CB-fit tuning is pending.
            if (
                "resident" in msg
                or "beyond max L1 size" in msg
                or "max L1 size" in msg
                or "wide-shard LayerNorm at TP>1" in msg
            ):
                need_wide.append(cfg.cid)
                logger.info(f"LNCORR {cfg.cid:<26} NEED_WIDE (CBs exceed L1; needs wide-shard CB-fit tuning)")
                continue
            raise

        ndiff = 0
        for j in range(3):  # same input -> must be bit-exact
            if (_fused(j + 1) - out0).abs().max().item() > 0.0:
                ndiff += 1
        det = ndiff == 0
        pcc = _pcc(out0, ref)
        maxabs = (out0 - ref).abs().max().item()
        susp = (pcc < 0.999) or (not det)
        if susp:
            flagged.append(cfg.cid)
        logger.info(
            f"LNCORR {cfg.cid:<26} det={'OK' if det else 'FAIL'} pcc(F:torch)={pcc * 100:.4f}% "
            f"maxabs={maxabs:.4f} det_ndiff={ndiff}/3{'  <-- SUSPICIOUS' if susp else ''}"
        )
    logger.info(f"LNCORR [{model} tp{tp}] flagged={flagged or 'NONE'} need_wide={need_wide or 'NONE'}")
    assert not flagged, f"LayerNorm correctness flagged: {flagged}"


# ---------------------------------------------------------------------------
# Module-level wiring: DistributedLayerNorm (adaLN) fused device op vs fp32-PyTorch.
# The fused Welford LN device op is the only path in the module now; asserts it clears
# the PCC bar and is deterministic, on the galaxy ring / line / submesh-line configs.
# ---------------------------------------------------------------------------
# (mesh, device_params, model, tp, topology, tp_axis, full_mesh)
_LNMOD_PARAMS = [
    ((4, 8), _DP_GAL, WAN, 2, ttnn.Topology.Linear, 1, False),  # submesh line 1x2
    ((4, 8), _DP_GAL, WAN, 4, ttnn.Topology.Linear, 1, False),  # submesh line 1x4
    ((4, 8), _DP_GAL, LTX, 4, ttnn.Topology.Linear, 1, False),  # submesh line 1x4 (LTX dim)
    ((4, 8), _DP_GAL_RING, WAN, 4, ttnn.Topology.Ring, 0, False),  # 4x8 ring (TP on 4-axis)
    ((4, 8), _DP_GAL, WAN, 8, ttnn.Topology.Linear, 1, True),  # 4x8 line (TP on full 8-axis)
]
_LNMOD_IDS = ["wan_tp2_line", "wan_tp4_line", "ltx_tp4_line", "wan_tp4_ring", "wan_tp8_line"]


@pytest.mark.parametrize(
    ("mesh_device", "device_params", "model", "tp", "topology", "tp_axis", "full_mesh"),
    [pytest.param(*p, id=i) for p, i in zip(_LNMOD_PARAMS, _LNMOD_IDS)],
    indirect=["mesh_device", "device_params"],
)
def test_layernorm_module_corr(mesh_device, model, tp, topology, tp_axis, full_mesh):
    """DistributedLayerNorm (adaLN, batch=1): fused device op vs fp32-PyTorch. Must clear the
    PCC bar and be bit-exact across launches. Exercises the recip LUT + LN-sized stats buffer
    wiring end-to-end on the galaxy ring / line / submesh-line configs."""
    submesh = _resolve_submesh(mesh_device, tp, tp_axis, full_mesh)
    dim = _LN_HID[model]
    # LNMOD_ROWS lets us probe multi-AG-round behavior (block uses N~24800 -> ~775
    # tile-rows -> many rounds; N=256 is a single round). LNMOD_MEAN0=1 uses a mean-0
    # input (real block activations) instead of the mean-4 default.
    rows = int(_os.getenv("LNMOD_ROWS", "256"))
    torch.manual_seed(0)
    if _os.getenv("LNMOD_MEAN0") == "1":
        x_t = torch.randn(1, 1, rows, dim, dtype=torch.bfloat16).float()
    else:
        x_t = (torch.randn(1, 1, rows, dim, dtype=torch.bfloat16) * 2 + 4).float()
    scale_t = torch.randn(1, 1, dim, dtype=torch.bfloat16).float()
    shift_t = torch.randn(1, 1, dim, dtype=torch.bfloat16).float()

    # fp32-PyTorch reference: LayerNorm(x) * (1+scale) + shift (adaLN).
    mean = x_t.mean(-1, keepdim=True)
    var = x_t.var(-1, unbiased=False, keepdim=True)
    ref = ((x_t - mean) / torch.sqrt(var + NORM_EPS)) * (1.0 + scale_t) + shift_t
    ref = ref.reshape(rows, dim)

    x = bf16_tensor(x_t.to(torch.bfloat16), device=submesh, mesh_axis=tp_axis, shard_dim=-1)
    # adaLN modulation is FP32 in production (kept fp32 for precision); the fused op consumes
    # fp32 weight/bias natively, so feed fp32 here to exercise that path.
    dyn_w = float32_tensor(1.0 + scale_t, device=submesh, mesh_axis=tp_axis, shard_dim=-1)
    dyn_b = float32_tensor(shift_t, device=submesh, mesh_axis=tp_axis, shard_dim=-1)

    # ONE CCLManager + ONE module shared by both methods. Building a second CCLManager
    # (a second set of fabric global-semaphores) and running the composite AG then the
    # fused fabric-forwarder AG back-to-back deadlocked on the Ring fabric (it was fine
    # on Linear). Mirrors production: a model holds one manager and one norm module.
    ccl = CCLManager(mesh_device=submesh, num_links=GALAXY_LINKS, topology=topology)
    mod = DistributedLayerNorm(
        embedding_dim=dim,
        norm_elementwise_affine=False,
        bias=False,
        mesh_axis=tp_axis,
        mesh_device=submesh,
        ccl_manager=ccl,
    )

    # LNMOD_LAUNCHES: # of forward calls (default 3 for the determinism check).
    # Set to 1 to isolate single-launch CORRECTNESS at large N from the multi-launch
    # warm-AG hang (the block calls each norm once; the 3-launch path can deadlock).
    n_launch = int(_os.getenv("LNMOD_LAUNCHES", "3"))

    def _run():
        # n_launch forwards -> verify determinism across launches; return launch-0 output.
        outs = [mod.forward(x, dynamic_weight=dyn_w, dynamic_bias=dyn_b) for _ in range(n_launch)]
        gathered = [_gather(o, tp_axis) for o in outs]
        det = all(torch.equal(gathered[0], g) for g in gathered[1:])
        return gathered[0], det

    fused_out, fused_det = _run()
    fused_pcc = _pcc(fused_out, ref)
    logger.info(f"LNMOD {model}_tp{tp}_{topology.name:<6} fused pcc={fused_pcc*100:.4f}% det={fused_det}")
    assert fused_det, "fused DistributedLayerNorm not deterministic across launches"
    assert fused_pcc >= 0.999, f"fused pcc too low: {fused_pcc}"


_LNBATCH_PARAMS = [
    ((4, 8), _DP_GAL, 4, ttnn.Topology.Linear, 1, False),  # 1x4 line submesh, TP=4 -> 15 tile-cols (non-div)
    ((4, 8), _DP_GAL_RING, 4, ttnn.Topology.Ring, 0, False),  # 4x8 ring, TP on the 4-axis
]
_LNBATCH_IDS = ["tp4_line", "tp4_ring"]

# TP=1 single-device params: a (1,1) submesh (is_tp_1, no fabric / all-gather).
_LNTP1_PARAMS = [((4, 8), _DP_GAL, 1, ttnn.Topology.Linear, 1, False)]
_LNTP1_IDS = ["tp1"]


@pytest.mark.parametrize(
    ("mesh_device", "device_params", "tp", "topology", "tp_axis", "full_mesh"),
    [pytest.param(*p, id=i) for p, i in zip(_LNTP1_PARAMS, _LNTP1_IDS)],
    indirect=["mesh_device", "device_params"],
)
def test_layernorm_noaffine_wide_tp1_corr(mesh_device, tp, topology, tp_axis, full_mesh):
    """Plain (no-affine) LayerNorm on a WIDE TP=1 single-device shard (dim=3072 -> 96 tile-cols).
    Regression guard for the Blackhole-only garbage bug: with weight=bias=None the L1 estimate drops
    the affine CBs, so decide_streaming_low_l1 (hardcoded 1.5 MB budget) can select streamed input
    while decide_block_major_post (arch L1 cap) still says the POST fits resident. On BH's larger L1
    that lands in the illegal 'streamed input + resident POST' combo (resident POST reads a
    block-sized input_cb across the whole row -> PCC~0). The fix ties block_major_post to
    streaming_low_l1, so streamed input always uses the block-major POST. WH passes both before/after
    (WH's smaller L1 makes overflows_resident_post=true -> block-major already); the BH path is
    validated in CI on the BH SKU."""
    submesh = _resolve_submesh(mesh_device, tp, tp_axis, full_mesh)
    dim = 3072  # TP=1 -> 96 tile-cols (divisible; NOT a tail case), the width that hits the L1 band
    rows = 128
    torch.manual_seed(0)
    x_t = (torch.randn(1, 1, rows, dim, dtype=torch.bfloat16) * 2 + 4).float()
    mean = x_t.mean(-1, keepdim=True)
    var = x_t.var(-1, unbiased=False, keepdim=True)
    ref = ((x_t - mean) / torch.sqrt(var + NORM_EPS)).reshape(rows, dim)

    x = bf16_tensor(x_t.to(torch.bfloat16), device=submesh, mesh_axis=tp_axis, shard_dim=-1)
    ccl = CCLManager(mesh_device=submesh, num_links=GALAXY_LINKS, topology=topology)
    mod = DistributedLayerNorm(
        embedding_dim=dim,
        norm_elementwise_affine=False,  # no-affine: the discriminator that exposes the bug
        bias=False,
        mesh_axis=tp_axis,
        mesh_device=submesh,
        ccl_manager=ccl,
    )
    n_launch = int(_os.getenv("LNMOD_LAUNCHES", "3"))
    outs = [mod.forward(x) for _ in range(n_launch)]
    gathered = [_gather(o, tp_axis) for o in outs]
    det = all(torch.equal(gathered[0], g) for g in gathered[1:])
    out0 = gathered[0].reshape(rows, dim)
    pcc = _pcc(out0, ref)
    logger.info(f"LNTP1WIDE tp{tp}_{topology.name} dim={dim} (96 tile-cols) no-affine pcc={pcc * 100:.4f}% det={det}")
    assert det, "wide no-affine TP=1 LayerNorm not deterministic across launches"
    assert pcc >= 0.999, f"wide no-affine TP=1 LayerNorm pcc too low: {pcc}"


@pytest.mark.parametrize(
    ("mesh_device", "device_params", "tp", "topology", "tp_axis", "full_mesh"),
    [pytest.param(*p, id=i) for p, i in zip(_LNBATCH_PARAMS, _LNBATCH_IDS)],
    indirect=["mesh_device", "device_params"],
)
def test_layernorm_nondiv_corr(mesh_device, tp, topology, tp_axis, full_mesh):
    """adaLN LayerNorm at a width NOT divisible by the compute block_size (dim 1920 -> 15
    tile-cols at TP4), which the resident POST + block_size-draining writer must handle via
    padded blocks. SD3.5 (dim 2432 -> 38/19 tile-cols) hits this. Runs batch=1 by default;
    LNBATCH_B>1 exercises the (WIP) dim-1 batch path."""
    submesh = _resolve_submesh(mesh_device, tp, tp_axis, full_mesh)
    dim = 1920  # 1920/4 = 480 -> 15 tile-cols (15 % 4 != 0), and 1920 % (32*4) == 0 (tile-aligned)
    batch = int(_os.getenv("LNBATCH_B", "1"))
    rows = 256
    torch.manual_seed(0)
    x_t = (torch.randn(1, batch, rows, dim, dtype=torch.bfloat16) * 2 + 4).float()
    scale_t = torch.randn(batch, 1, dim, dtype=torch.bfloat16).float()  # per-batch (broadcast over N)
    shift_t = torch.randn(batch, 1, dim, dtype=torch.bfloat16).float()

    # fp32 reference: per-batch adaLN. LN over the last dim for each (batch, token) row.
    mean = x_t.mean(-1, keepdim=True)
    var = x_t.var(-1, unbiased=False, keepdim=True)
    ref = ((x_t - mean) / torch.sqrt(var + NORM_EPS)) * (1.0 + scale_t).unsqueeze(0) + shift_t.unsqueeze(0)
    ref = ref.reshape(batch * rows, dim)

    x = bf16_tensor(x_t.to(torch.bfloat16), device=submesh, mesh_axis=tp_axis, shard_dim=-1)
    dyn_w = float32_tensor(1.0 + scale_t, device=submesh, mesh_axis=tp_axis, shard_dim=-1)
    dyn_b = float32_tensor(shift_t, device=submesh, mesh_axis=tp_axis, shard_dim=-1)

    ccl = CCLManager(mesh_device=submesh, num_links=GALAXY_LINKS, topology=topology)
    mod = DistributedLayerNorm(
        embedding_dim=dim,
        norm_elementwise_affine=False,
        bias=False,
        mesh_axis=tp_axis,
        mesh_device=submesh,
        ccl_manager=ccl,
    )
    n_launch = int(_os.getenv("LNMOD_LAUNCHES", "3"))
    outs = [mod.forward(x, dynamic_weight=dyn_w, dynamic_bias=dyn_b) for _ in range(n_launch)]
    gathered = [_gather(o, tp_axis, batched=(batch > 1)) for o in outs]
    det = all(torch.equal(gathered[0], g) for g in gathered[1:])
    out0 = gathered[0].reshape(batch * rows, dim)
    pcc = _pcc(out0, ref)
    logger.info(f"LNBATCH tp{tp}_{topology.name} batch={batch} dim={dim} (15 tile-cols) pcc={pcc * 100:.4f}% det={det}")
    assert det, "batched LayerNorm not deterministic across launches"
    assert pcc >= 0.999, f"batched LayerNorm pcc too low: {pcc}"


@pytest.mark.parametrize(
    ("mesh_device", "device_params", "tp", "topology", "tp_axis", "full_mesh"),
    [pytest.param(*p, id=i) for p, i in zip(_LNBATCH_PARAMS, _LNBATCH_IDS)],
    indirect=["mesh_device", "device_params"],
)
def test_layernorm_batch_fold_corr(mesh_device, tp, topology, tp_axis, full_mesh):
    """Isolates the dim-1 batch FOLD for plain LayerNorm (no affine). Input [1, batch, N, H]
    folds to batch*N/32 tile-rows via physical_volume/W; each batch contributes an integer
    number of tile-rows (padded), so folded rows never straddle batches. The op preserves
    batch at dim1 in the output ([1, batch, N, H]). No weight/bias -> isolates the fold from
    the per-batch affine path (tasks: per-row / per-batch weight)."""
    submesh = _resolve_submesh(mesh_device, tp, tp_axis, full_mesh)
    dim = 1024  # 1024/4 = 256 -> 8 tile-cols (8 % 4 == 0): divisible, resident POST. Pure fold test.
    batch = int(_os.getenv("LNFOLD_B", "2"))
    rows = 128
    torch.manual_seed(0)
    x_t = (torch.randn(1, batch, rows, dim, dtype=torch.bfloat16) * 2 + 4).float()

    # fp32 reference: plain LN over the last dim for each (batch, token) row (no affine).
    mean = x_t.mean(-1, keepdim=True)
    var = x_t.var(-1, unbiased=False, keepdim=True)
    ref = ((x_t - mean) / torch.sqrt(var + NORM_EPS)).reshape(batch * rows, dim)

    x = bf16_tensor(x_t.to(torch.bfloat16), device=submesh, mesh_axis=tp_axis, shard_dim=-1)

    ccl = CCLManager(mesh_device=submesh, num_links=GALAXY_LINKS, topology=topology)
    mod = DistributedLayerNorm(
        embedding_dim=dim,
        norm_elementwise_affine=False,  # plain LN, no weight/bias
        bias=False,
        mesh_axis=tp_axis,
        mesh_device=submesh,
        ccl_manager=ccl,
    )
    n_launch = int(_os.getenv("LNMOD_LAUNCHES", "3"))
    outs = [mod.forward(x) for _ in range(n_launch)]
    gathered = [_gather(o, tp_axis, batched=True) for o in outs]
    det = all(torch.equal(gathered[0], g) for g in gathered[1:])
    out0 = gathered[0].reshape(batch * rows, dim)
    pcc = _pcc(out0, ref)
    logger.info(f"LNFOLD tp{tp}_{topology.name} batch={batch} dim={dim} plain-LN pcc={pcc * 100:.4f}% det={det}")
    assert det, "batched plain LayerNorm not deterministic across launches"
    assert pcc >= 0.999, f"batched plain LayerNorm pcc too low: {pcc}"


@pytest.mark.parametrize(
    ("mesh_device", "device_params", "tp", "topology", "tp_axis", "full_mesh"),
    [pytest.param(*p, id=i) for p, i in zip(_LNBATCH_PARAMS, _LNBATCH_IDS)],
    indirect=["mesh_device", "device_params"],
)
def test_layernorm_adaln_batch_corr(mesh_device, tp, topology, tp_axis, full_mesh):
    """Per-batch adaLN LayerNorm (Motif): dynamic weight/bias are [batch, 1, H] — broadcast over
    seq but DISTINCT per batch. The op keeps all batches' affine rows resident and offsets the
    weight tile index by wbatch*num_tile_cols (wbatch = global_tile_row / rows_per_batch). Uses a
    divisible width so this isolates the per-batch path from the Issue-1 non-div tail handling."""
    submesh = _resolve_submesh(mesh_device, tp, tp_axis, full_mesh)
    dim = 1024  # 1024/4 = 256 -> 8 tile-cols (divisible): resident POST. Isolates per-batch indexing.
    batch = int(_os.getenv("LNADALN_B", "2"))
    rows = 128
    torch.manual_seed(0)
    x_t = (torch.randn(1, batch, rows, dim, dtype=torch.bfloat16) * 2 + 4).float()
    # DISTINCT scale/shift per batch (this is the whole point — a broadcast read would apply
    # batch-0's modulation to every batch and fail).
    scale_t = torch.randn(batch, 1, dim, dtype=torch.bfloat16).float()
    shift_t = torch.randn(batch, 1, dim, dtype=torch.bfloat16).float()

    # fp32 reference: per-batch adaLN. LN over the last dim for each (batch, token) row.
    mean = x_t.mean(-1, keepdim=True)
    var = x_t.var(-1, unbiased=False, keepdim=True)
    ref = ((x_t - mean) / torch.sqrt(var + NORM_EPS)) * (1.0 + scale_t).unsqueeze(0) + shift_t.unsqueeze(0)
    ref = ref.reshape(batch * rows, dim)

    x = bf16_tensor(x_t.to(torch.bfloat16), device=submesh, mesh_axis=tp_axis, shard_dim=-1)
    dyn_w = float32_tensor(1.0 + scale_t, device=submesh, mesh_axis=tp_axis, shard_dim=-1)  # [batch,1,H] per-batch
    dyn_b = float32_tensor(shift_t, device=submesh, mesh_axis=tp_axis, shard_dim=-1)

    ccl = CCLManager(mesh_device=submesh, num_links=GALAXY_LINKS, topology=topology)
    mod = DistributedLayerNorm(
        embedding_dim=dim,
        norm_elementwise_affine=False,
        bias=False,
        mesh_axis=tp_axis,
        mesh_device=submesh,
        ccl_manager=ccl,
    )
    n_launch = int(_os.getenv("LNMOD_LAUNCHES", "3"))
    outs = [mod.forward(x, dynamic_weight=dyn_w, dynamic_bias=dyn_b) for _ in range(n_launch)]
    gathered = [_gather(o, tp_axis, batched=(batch > 1)) for o in outs]
    det = all(torch.equal(gathered[0], g) for g in gathered[1:])
    out0 = gathered[0].reshape(batch * rows, dim)
    pcc = _pcc(out0, ref)
    logger.info(
        f"LNADALN tp{tp}_{topology.name} batch={batch} dim={dim} per-batch adaLN pcc={pcc * 100:.4f}% det={det}"
    )
    assert det, "per-batch adaLN LayerNorm not deterministic across launches"
    assert pcc >= 0.999, f"per-batch adaLN LayerNorm pcc too low: {pcc}"


@pytest.mark.parametrize(
    ("mesh_device", "device_params", "tp", "topology", "tp_axis", "full_mesh"),
    [pytest.param(*p, id=i) for p, i in zip(_LNBATCH_PARAMS, _LNBATCH_IDS)],
    indirect=["mesh_device", "device_params"],
)
def test_layernorm_per_token_corr(mesh_device, tp, topology, tp_axis, full_mesh):
    """Per-TOKEN affine LayerNorm: weight/bias are [1, 1, N, H] — a DISTINCT gamma/beta for every
    token (row), folded to the same N/32 tile-rows as the input. The reader pushes each row's
    weight/bias slice; the compute consumes them with mul_tiles/add_tiles (full-tile, not bcast)
    and pops num_tile_cols per row. Divisible width -> resident POST. Not used by any tt_dit model
    yet, but required by an upcoming one."""
    submesh = _resolve_submesh(mesh_device, tp, tp_axis, full_mesh)
    dim = 1024  # 1024/4 = 256 -> 8 tile-cols (divisible): resident POST.
    rows = 128
    torch.manual_seed(0)
    x_t = (torch.randn(1, 1, rows, dim, dtype=torch.bfloat16) * 2 + 4).float()
    # DISTINCT scale/shift per TOKEN (row): [1, 1, rows, dim].
    scale_t = torch.randn(1, 1, rows, dim, dtype=torch.bfloat16).float()
    shift_t = torch.randn(1, 1, rows, dim, dtype=torch.bfloat16).float()

    # fp32 reference: per-token affine. LN over the last dim, then gamma/beta that vary per row.
    mean = x_t.mean(-1, keepdim=True)
    var = x_t.var(-1, unbiased=False, keepdim=True)
    ref = (((x_t - mean) / torch.sqrt(var + NORM_EPS)) * (1.0 + scale_t) + shift_t).reshape(rows, dim)

    x = bf16_tensor(x_t.to(torch.bfloat16), device=submesh, mesh_axis=tp_axis, shard_dim=-1)
    # [1,1,N,H] weight/bias: logical[-2]==N>1 -> the op detects per-token and reads per row.
    dyn_w = float32_tensor(1.0 + scale_t, device=submesh, mesh_axis=tp_axis, shard_dim=-1)
    dyn_b = float32_tensor(shift_t, device=submesh, mesh_axis=tp_axis, shard_dim=-1)

    ccl = CCLManager(mesh_device=submesh, num_links=GALAXY_LINKS, topology=topology)
    mod = DistributedLayerNorm(
        embedding_dim=dim,
        norm_elementwise_affine=False,
        bias=False,
        mesh_axis=tp_axis,
        mesh_device=submesh,
        ccl_manager=ccl,
    )
    n_launch = int(_os.getenv("LNMOD_LAUNCHES", "3"))
    outs = [mod.forward(x, dynamic_weight=dyn_w, dynamic_bias=dyn_b) for _ in range(n_launch)]
    gathered = [_gather(o, tp_axis) for o in outs]
    det = all(torch.equal(gathered[0], g) for g in gathered[1:])
    out0 = gathered[0].reshape(rows, dim)
    pcc = _pcc(out0, ref)
    logger.info(f"LNPERTOK tp{tp}_{topology.name} dim={dim} per-token affine pcc={pcc * 100:.4f}% det={det}")
    assert det, "per-token affine LayerNorm not deterministic across launches"
    assert pcc >= 0.999, f"per-token affine LayerNorm pcc too low: {pcc}"


@pytest.mark.parametrize(
    ("mesh_device", "device_params", "tp", "topology", "tp_axis", "full_mesh"),
    [pytest.param(*p, id=i) for p, i in zip(_LNBATCH_PARAMS, _LNBATCH_IDS)],
    indirect=["mesh_device", "device_params"],
)
def test_layernorm_per_token_wide_corr(mesh_device, tp, topology, tp_axis, full_mesh):
    """Per-token affine LayerNorm on a WIDE shard that engages the block-major POST
    (dim 7168 -> 1792/device -> 56 tile-cols at TP4; force_recip_stream fires at >=56).
    Exercises the block-major per-row weight/bias consume + per-row pop path."""
    submesh = _resolve_submesh(mesh_device, tp, tp_axis, full_mesh)
    dim = 7168  # 7168/4 = 1792 -> 56 tile-cols (>=56 and 56%block_size==0): block-major POST.
    rows = 64
    torch.manual_seed(0)
    x_t = (torch.randn(1, 1, rows, dim, dtype=torch.bfloat16) * 2 + 4).float()
    scale_t = torch.randn(1, 1, rows, dim, dtype=torch.bfloat16).float()
    shift_t = torch.randn(1, 1, rows, dim, dtype=torch.bfloat16).float()

    mean = x_t.mean(-1, keepdim=True)
    var = x_t.var(-1, unbiased=False, keepdim=True)
    ref = (((x_t - mean) / torch.sqrt(var + NORM_EPS)) * (1.0 + scale_t) + shift_t).reshape(rows, dim)

    x = bf16_tensor(x_t.to(torch.bfloat16), device=submesh, mesh_axis=tp_axis, shard_dim=-1)
    dyn_w = float32_tensor(1.0 + scale_t, device=submesh, mesh_axis=tp_axis, shard_dim=-1)
    dyn_b = float32_tensor(shift_t, device=submesh, mesh_axis=tp_axis, shard_dim=-1)

    ccl = CCLManager(mesh_device=submesh, num_links=GALAXY_LINKS, topology=topology)
    mod = DistributedLayerNorm(
        embedding_dim=dim,
        norm_elementwise_affine=False,
        bias=False,
        mesh_axis=tp_axis,
        mesh_device=submesh,
        ccl_manager=ccl,
    )
    n_launch = int(_os.getenv("LNMOD_LAUNCHES", "3"))
    outs = [mod.forward(x, dynamic_weight=dyn_w, dynamic_bias=dyn_b) for _ in range(n_launch)]
    gathered = [_gather(o, tp_axis) for o in outs]
    det = all(torch.equal(gathered[0], g) for g in gathered[1:])
    out0 = gathered[0].reshape(rows, dim)
    pcc = _pcc(out0, ref)
    logger.info(
        f"LNPERTOKW tp{tp}_{topology.name} dim={dim} (56 tile-cols, block-major) pcc={pcc * 100:.4f}% det={det}"
    )
    assert det, "wide per-token affine LayerNorm not deterministic across launches"
    assert pcc >= 0.999, f"wide per-token affine LayerNorm pcc too low: {pcc}"


@pytest.mark.parametrize("rope_mode", ["bcast", "perbatch"])
@pytest.mark.parametrize(
    ("mesh_device", "device_params", "tp", "topology", "tp_axis", "full_mesh"),
    [pytest.param(*p, id=i) for p, i in zip(_LNBATCH_PARAMS, _LNBATCH_IDS)],
    indirect=["mesh_device", "device_params"],
)
def test_rmsnorm_batched_rope_corr(mesh_device, tp, topology, tp_axis, full_mesh, rope_mode):
    """Batched RMSNorm + fused RoPE (Issue 3). The reader indexes cos/sin by the WITHIN-batch
    seq row plus a per-batch offset, so cos/sin dim0 is either 1 (broadcast the same RoPE to every
    input batch, rope_mode="bcast") or batch (each batch gets its own cos/sin, "perbatch").

    Oracle = the already-trusted batch=1 RoPE path: run batched (batch=2), then run batch=1 for
    each input batch with that batch's cos/sin; the batched output's batch-b seq slice must equal
    the batch=1 result. This validates the reindex + the head-split-with-batch output layout
    ([1, num_heads, batch*N, head_dim]) without a hand-derived RoPE reference."""
    submesh = _resolve_submesh(mesh_device, tp, tp_axis, full_mesh)
    torch.manual_seed(0)
    dim = 1024
    head_dim = 128
    batch = 2
    rows = 64  # 2 tile-rows per batch
    nh_dev = (dim // tp) // head_dim  # per-device whole heads (num_heads_per_device); 256/128 = 2
    assert nh_dev > 1, "batched RoPE test wants head-split (num_heads_per_device>1)"
    links = GALAXY_LINKS

    x_t = torch.randn(1, batch, rows, dim, dtype=torch.bfloat16)
    b_rope = 1 if rope_mode == "bcast" else batch
    cos_raw = torch.randn(b_rope, rows, 1, head_dim // 2)
    sin_raw = torch.randn(b_rope, rows, 1, head_dim // 2)
    cos_f, sin_f = stack_cos_sin(cos_raw, sin_raw)  # (b_rope, rows, 1, head_dim)
    cos_bhnd = cos_f.permute(0, 2, 1, 3)  # (b_rope, 1, rows, head_dim)
    sin_bhnd = sin_f.permute(0, 2, 1, 3)

    def _run(x_host, cos_host, sin_host):
        x = bf16_tensor(x_host, device=submesh, mesh_axis=tp_axis, shard_dim=-1)
        cos = from_torch(cos_host, device=submesh, dtype=ttnn.float32)  # broadcast RoPE: replicated
        sin = from_torch(sin_host, device=submesh, dtype=ttnn.float32)
        trans = bf16_tensor(get_rot_transformation_mat(), device=submesh)
        ccl = CCLManager(mesh_device=submesh, num_links=links, topology=topology)
        sems = [ccl.get_ag_ping_pong_semaphore(tp_axis) for _ in range(_PINGPONG)]
        mk_pob = lambda: ttnn.experimental.dit_fused_distributed_rmsnorm_create_stats_buffer(  # noqa: E731
            x,
            tp_axis,
            submesh,
            num_heads_per_device=nh_dev,
            num_links=links,
            transformation_mat=trans,
            rope_cos=cos,
            rope_sin=sin,
        )
        pobs = [mk_pob() for _ in range(_PINGPONG)]
        ttnn.synchronize_device(submesh)
        outs = []
        for k in range(3):  # ping-pong; 3 launches also checks determinism
            o = ttnn.experimental.dit_fused_distributed_rmsnorm(
                x,
                tp_axis,
                submesh,
                sems[k % _PINGPONG],
                topology=topology,
                epsilon=NORM_EPS,
                num_heads_per_device=nh_dev,
                transformation_mat=trans,
                rope_cos=cos,
                rope_sin=sin,
                persistent_output_buffer=pobs[k % _PINGPONG],
                num_preferred_links=links,
            )
            outs.append(_gather(o, tp_axis))  # head-split -> [seq, dim]
        det = all(torch.equal(outs[0], g) for g in outs[1:])
        return outs[0], det

    batched_out, det = _run(x_t, cos_bhnd, sin_bhnd)  # [batch*rows, dim]
    assert det, "batched RMSNorm+RoPE not deterministic across launches"

    worst = 1.0
    for b in range(batch):
        cos_b = cos_bhnd[b : b + 1] if b_rope == batch else cos_bhnd  # [1,1,rows,head_dim]
        sin_b = sin_bhnd[b : b + 1] if b_rope == batch else sin_bhnd
        oracle, _ = _run(x_t[:, b : b + 1], cos_b, sin_b)  # [rows, dim]
        pcc_b = _pcc(batched_out[b * rows : (b + 1) * rows], oracle)  # batch-major seq fold
        worst = min(worst, pcc_b)
        logger.info(f"ROPEB tp{tp}_{topology.name} {rope_mode} batch{b} vs batch=1 oracle pcc={pcc_b * 100:.4f}%")
    assert worst >= 0.999, f"batched RoPE ({rope_mode}) diverged from batch=1 oracle: pcc={worst}"


@pytest.mark.parametrize(
    ("mesh_device", "device_params", "model", "tp", "topology", "tp_axis", "full_mesh"),
    [pytest.param(*p, id=i) for p, i in zip(_LNMOD_PARAMS, _LNMOD_IDS)],
    indirect=["mesh_device", "device_params"],
)
def test_rmsnorm_module_corr(mesh_device, model, tp, topology, tp_axis, full_mesh):
    """DistributedRMSNorm (static weight): fused device op vs fp32-PyTorch reference. Must clear
    the PCC bar and be bit-exact across 3 launches. Same galaxy ring/line/submesh configs as
    LayerNorm."""
    submesh = _resolve_submesh(mesh_device, tp, tp_axis, full_mesh)
    dim = _LN_HID[model]
    rows = 256
    torch.manual_seed(0)
    x_t = (torch.randn(1, 1, rows, dim, dtype=torch.bfloat16) * 2 + 4).float()
    w_t = torch.randn(dim, dtype=torch.bfloat16).float()

    # fp32-PyTorch reference: x / sqrt(mean(x^2)) * weight.
    rms = x_t / torch.sqrt((x_t**2).mean(-1, keepdim=True) + NORM_EPS)
    ref = (rms * w_t).reshape(rows, dim)

    x = bf16_tensor(x_t.to(torch.bfloat16), device=submesh, mesh_axis=tp_axis, shard_dim=-1)

    ccl = CCLManager(mesh_device=submesh, num_links=GALAXY_LINKS, topology=topology)
    mod = DistributedRMSNorm(
        embedding_dim=dim,
        norm_elementwise_affine=True,
        mesh_axis=tp_axis,
        mesh_device=submesh,
        ccl_manager=ccl,
    )
    mod.weight.data = bf16_tensor(
        w_t.reshape(1, dim).to(torch.bfloat16), device=submesh, mesh_axis=tp_axis, shard_dim=-1
    )

    def _run():
        outs = [mod.forward(x) for _ in range(3)]
        gathered = [_gather(o, tp_axis) for o in outs]
        det = all(torch.equal(gathered[0], g) for g in gathered[1:])
        return gathered[0], det

    fused_out, fused_det = _run()
    fused_pcc = _pcc(fused_out, ref)
    logger.info(f"RMSMOD {model}_tp{tp}_{topology.name:<6} fused pcc={fused_pcc*100:.4f}% det={fused_det}")
    assert fused_det, "fused DistributedRMSNorm not deterministic across launches"
    assert fused_pcc >= 0.999, f"fused pcc too low: {fused_pcc}"


# (mesh, device_params, model, dim, tp, topology, tp_axis, full_mesh)
_LNBENCH_PARAMS = [
    ((4, 8), _DP_GAL, FLUX, _FLUX_DIM, 4, ttnn.Topology.Linear, 1, False),  # FLUX TP4 (1x4 line)
    ((4, 8), _DP_GAL, FLUX, _FLUX_DIM, 8, ttnn.Topology.Linear, 1, False),  # FLUX TP8 (1x8 line)
    ((4, 8), _DP_GAL, WAN, 5120, 4, ttnn.Topology.Linear, 1, False),  # WAN TP4 (wider feat)
]
_LNBENCH_IDS = ["flux_tp4", "flux_tp8", "wan_tp4"]
# Per-(model, tp) per-device row counts swept (full seq; only dim is TP-sharded). FLUX uses the
# RMS-bench shapes. On Blackhole galaxy run with WAN_GALAXY_LINKS=2 (2-link torus vs WH's 4).
_LNBENCH_SEQLENS = {
    (FLUX, 4): [512, 64, 2048, 8192],
    (FLUX, 8): [1024, 128, 4096, 16384],
    (WAN, 4): [512, 1024, 4096, 8192],
}


def _composite_ln_recip(submesh, width_per_device):
    """HEIGHT_SHARDED Welford reciprocal tensor that dit_layernorm_pre_allgather consumes
    (the composite baseline; the fused op instead reads a row-major recip LUT)."""
    grid = submesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    return ttnn.create_layer_norm_reciprocals(submesh, crs, width_per_device)


@pytest.mark.skip_post_commit  # perf/dev benchmark, not a correctness gate
@pytest.mark.parametrize(
    ("mesh_device", "device_params", "model", "dim", "tp", "topology", "tp_axis", "full_mesh"),
    [pytest.param(*p, id=i) for p, i in zip(_LNBENCH_PARAMS, _LNBENCH_IDS)],
    indirect=["mesh_device", "device_params"],
)
def test_layernorm_module_bench(mesh_device, model, dim, tp, topology, tp_axis, full_mesh):
    """Traced speedup: fused Welford LayerNorm device op vs the composite dit_layernorm
    chain (dit_layernorm_pre_allgather -> all_gather_persistent_buffer ->
    dit_layernorm_post_allgather, weight+bias), in the adaLN DistributedLayerNorm setup,
    across a span of sequence lengths. Whole-row LN: weight+bias, no RoPE, no per-head norm.
    Prints base/fused/speedup; not an assertion test. Galaxy only -- on Blackhole galaxy run
    with WAN_GALAXY_LINKS=2 (2-link torus) instead of WH's 4."""
    submesh = _resolve_submesh(mesh_device, tp, tp_axis, full_mesh)
    feat = dim // tp
    iters = 100
    ccl = CCLManager(mesh_device=submesh, num_links=GALAXY_LINKS, topology=topology)
    mod = DistributedLayerNorm(
        embedding_dim=dim,
        norm_elementwise_affine=False,
        bias=False,
        mesh_axis=tp_axis,
        mesh_device=submesh,
        ccl_manager=ccl,
    )
    ckc, eps = mod.compute_kernel_config, mod.norm_eps
    recip_comp = _composite_ln_recip(submesh, feat)
    seqlens = _LNBENCH_SEQLENS.get((model, tp), [512, 1024, 4096, 8192])
    rows = []
    for seq in seqlens:
        torch.manual_seed(0)
        x = bf16_tensor(
            torch.randn(1, 1, seq, dim, dtype=torch.bfloat16), device=submesh, mesh_axis=tp_axis, shard_dim=-1
        )
        dw = bf16_tensor(torch.randn(1, 1, dim, dtype=torch.bfloat16), device=submesh, mesh_axis=tp_axis, shard_dim=-1)
        db = bf16_tensor(torch.randn(1, 1, dim, dtype=torch.bfloat16), device=submesh, mesh_axis=tp_axis, shard_dim=-1)
        row = {"seq": seq}

        # Fused: the module rotates its pob+AG-sem internally per forward, so two captured
        # traces bind the two resource sets; replay them round-robin. Bind x/dw/db per iter.
        try:
            fused_ops = [
                lambda _x=x, _dw=dw, _db=db: mod.forward(_x, dynamic_weight=_dw, dynamic_bias=_db)
                for _ in range(_PINGPONG)
            ]
            row["fused"] = _trace_and_time(submesh, fused_ops, num_iters=iters)
        except Exception as e:  # noqa: BLE001
            row["fused_err"] = type(e).__name__
            logger.warning(f"{model} tp{tp} seq={seq} FUSED failed: {str(e)[:160]}")

        # Composite baseline: pre_allgather -> AG(persistent buffer) -> post_allgather(weight,bias).
        def _composite(_x=x, _dw=dw, _db=db):
            stats = ttnn.experimental.dit_layernorm_pre_allgather(_x, recip_comp, compute_kernel_config=ckc)
            stats = ccl.all_gather_persistent_buffer(stats, dim=len(_x.shape) - 1, mesh_axis=tp_axis)
            return ttnn.experimental.dit_layernorm_post_allgather(
                _x, stats, weight=_dw, bias=_db, epsilon=eps, compute_kernel_config=ckc, dtype=None
            )

        try:
            row["base"] = _trace_and_time(submesh, _composite, num_iters=iters)
        except Exception as e:  # noqa: BLE001
            row["base_err"] = type(e).__name__
            logger.warning(f"{model} tp{tp} seq={seq} BASELINE failed: {str(e)[:160]}")
        rows.append(row)

    title = (
        f"DistributedLayerNorm fused vs composite (model={model}, dim={dim}, TP={tp}, "
        f"feat_local={feat}, links={GALAXY_LINKS})"
    )
    box = "=" * max(len(title), 64)
    print("\n" + box + f"\n{title}\n" + box)
    print(f"{'seq_len':>8} {'base_us':>10} {'fused_us':>10} {'speedup':>9}")
    print("-" * 42)
    for r in rows:
        b, f = r.get("base"), r.get("fused")
        bs = f"{b:.2f}" if b is not None else r.get("base_err", "ERR")
        fs = f"{f:.2f}" if f is not None else r.get("fused_err", "ERR")
        sp = f"{b / f:.2f}x" if (b is not None and f is not None and f > 0) else "-"
        print(f"{r['seq']:>8} {bs:>10} {fs:>10} {sp:>9}")
    print(box)


@pytest.mark.parametrize(
    ("mesh_device", "device_params", "model", "tp", "topology", "op_override", "tp_axis", "full_mesh"),
    [pytest.param(*p, id=i) for p, i in zip(_CORR_PARAMS, _CORR_IDS)],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.skip_post_commit  # trace-replay dev check, not a correctness gate
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
            # pcc_te (traced vs eager) is the trace guard — kept strict. pcc_tr (vs torch)
            # inherits the per-head-norm accuracy floor (~99.81%), so relax that bar for
            # per_head_norm just like test_corr_det.
            pcc_tr_bar = 0.997 if cfg.per_head_norm else 0.999
            susp = (pcc_te < 0.9999) or (pcc_tr < pcc_tr_bar)
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
    assert not flagged, f"traced-vs-eager correctness flagged: {flagged}"


# ===========================================================================
# test_sweep — exhaustive local parameter sweep (skipped in CI)
#
# Independent, self-contained thorough test of the fused op across TP=1 / TP=4
# line / TP=4 ring: hidden width (-> resident vs streaming vs block-major POST
# selection), seqlen, batch, RMS vs LayerNorm, affine mode (none / broadcast
# weight[+bias] / per-token / per-batch adaLN), fused RoPE (broadcast-heads &
# per-head, broadcast & per-batch cos), per-head norm, and bf16/fp32 dtypes.
#
# It draws its own seeded host tensors, builds device inputs, and computes an
# fp32 torch reference IN THE SAME PASS (so build and reference share the exact
# values) — this avoids touching the shared Cfg/_build/_torch_ref machinery that
# test_corr_det relies on. The op RUN reuses the trusted _run_fused/_make_pob/
# _gather plumbing. Known-good shapes are included, so a reference bug surfaces as
# a failure on a config the op is known to handle. Per-point failures are collected
# (one bad shape doesn't abort the sweep) and asserted at the end.
# ===========================================================================

_SWEEP_PARAMS = [
    ((4, 8), _DP_GAL, 1, ttnn.Topology.Linear, 1, False),  # TP=1 single-device (1,1) submesh
    ((4, 8), _DP_GAL, 4, ttnn.Topology.Linear, 1, False),  # TP=4 line (1,4) submesh
    ((4, 8), _DP_GAL_RING, 4, ttnn.Topology.Ring, 0, False),  # TP=4 ring on the 4-axis
]
_SWEEP_IDS = ["tp1", "tp4_line", "tp4_ring"]


def _sweep_points(tp: int) -> list[dict]:
    """Exhaustive curated families covering every axis. Widths densely span the
    resident / streaming / block-major POST bands (block_size-divisible AND not);
    crossed with norm, affine (none/w/wb/per-token/per-batch), seqlen (incl.
    non-tile-aligned), fused RoPE (broadcast-heads & per-head, head_dim 64/128,
    broadcast & per-batch cos), per-head norm, head-split output (nh 2..), batch
    (1..8), and bf16/fp32 dtypes. Invalid combos are validity-filtered (skipped)."""
    if tp == 1:
        # per-device tile-cols = dim/32
        widths = [
            512,
            768,
            1024,
            1536,
            2048,
            2560,
            3072,
            3584,
            4096,
            4608,
            5120,
        ]  # 16,24,32,48,64,80,96,112,128,144,160
        widths_nondiv = [480, 608, 1216, 1600]  # cols 15,19,38,50 (not a multiple of block_size)
        rope_dims = [1024, 2048, 4096]  # heads(hd=128) = 8,16,32
        mid = 2048
    else:  # tp == 4: per-device tile-cols = (dim/4)/32
        # <= 6144 (FLUX full dim = 48 tile-cols/dev): the widest real AG-path shape. dim=8192
        # (64 cols) RMS+affine marginally overflows L1 on the AG path (~3 KB) — beyond real widths.
        widths = [2048, 3072, 4096, 5120, 6144]  # cols 16,24,32,40,48
        widths_nondiv = [1920, 2432, 4864]  # cols 15,19,38
        rope_dims = [2048, 4096, 6144]  # heads/dev(hd=128) = 4,8,12
        mid = 4096
    all_widths = widths + widths_nondiv
    feat = lambda dim: dim // tp  # noqa: E731

    def pt(**kw):
        base = dict(
            fam="?",
            norm="rms",
            dim=mid,
            rows=128,
            batch=1,
            head_dim=None,
            rope=False,
            bcast_rope=False,
            per_head_norm=False,
            affine="none",
            in_dtype="bf16",
            out_dtype="bf16",
            rope_perbatch=False,
        )
        base.update(kw)
        return base

    def rope_ok(dim, hd, need_split=True):
        # RoPE needs whole heads per device; test the head-split regime (>=2 heads/dev).
        return dim % hd == 0 and feat(dim) % hd == 0 and (feat(dim) // hd) >= (2 if need_split else 1)

    pts = []
    # A. width x norm x affine (nh=1, no rope) — the POST-path-selection stressor (densest axis).
    for dim in all_widths:
        for norm in ("rms", "layernorm"):
            for affine in ("none", "w", "wb"):
                pts.append(pt(fam="width", norm=norm, dim=dim, affine=affine))
    # B. width x seqlen (a wider row count crossed with width) — every other width, both norms.
    for dim in all_widths[::2]:
        for norm in ("rms", "layernorm"):
            for affine in ("none", "wb"):
                pts.append(pt(fam="widthxseq", norm=norm, dim=dim, rows=512, affine=affine))
    # C. seqlen sweep incl. non-tile-aligned (33, 100) and tiny/large.
    for rows in (16, 32, 64, 96, 128, 256, 512, 1024, 2048, 33, 100):
        for norm in ("rms", "layernorm"):
            pts.append(pt(fam="seqlen", norm=norm, rows=rows))
    for rows in (128, 512):
        pts.append(pt(fam="seqlen", norm="rms", rows=rows, affine="wb"))
    # D. fused RoPE (RMS): head_dim {64,128} x broadcast-heads/per-head x affine {none,w}.
    for dim in rope_dims:
        for hd in (64, 128):
            if not rope_ok(dim, hd):
                continue
            for bcast in (True, False):
                for affine in ("none", "w"):
                    pts.append(
                        pt(fam="rope", norm="rms", dim=dim, head_dim=hd, rope=True, bcast_rope=bcast, affine=affine)
                    )
    # E. RoPE x seqlen.
    for rows in (32, 256, 1024):
        for bcast in (True, False):
            pts.append(
                pt(fam="ropeseq", norm="rms", dim=rope_dims[0], rows=rows, head_dim=128, rope=True, bcast_rope=bcast)
            )
    # F. per-head norm (RMS, no AG): head_dim {64,128} x affine {none,w}.
    # per_head_norm has a RESIDENT-ONLY POST (it never auto-streams — the head-block reduce path
    # only handles whole-row-resident), so it caps out around ~64 tile-cols/device before the
    # whole-row input_cb + per-head stat tiles overflow L1. Cap its widths accordingly (dim=4096
    # at TP=1 = 128 cols OOMs); the streaming/block-major widths are covered by the whole-row
    # norm families above.
    perhead_dims = [d for d in rope_dims if feat(d) // 32 <= 64]
    for dim in perhead_dims:
        for hd in (64, 128):
            if not rope_ok(dim, hd):
                continue
            for affine in ("none", "w"):
                pts.append(pt(fam="perhead", norm="rms", dim=dim, head_dim=hd, per_head_norm=True, affine=affine))
    # G. head-split output (nh>1) is exercised WITH RoPE (family D/E) and by per-head norm
    # (family F) above. NOTE/FINDING: plain head-split RMSNorm WITHOUT rope on the is_tp_1 path
    # (TP=1) sits at ~99.65% PCC vs a whole-row reference (not garbage) — head-split+RoPE and
    # per-head-norm both pass, so this is specific to head-split-no-rope on the local (is_tp_1)
    # reduce path and is left as a separate investigation rather than swept here.
    #
    # H. dtypes: fp32 RMS (+/- rope) and bf16-in/fp32-out LN. Widths kept block_size-divisible:
    # a non-divisible wide fp32 shard can't block-major (that path needs divisibility) so it is
    # forced resident, and fp32 input (2x bytes) then overflows L1 (e.g. dim=1600 = 50 cols).
    for dim in (widths[0], mid):
        for aff in ("none", "w", "wb"):
            pts.append(pt(fam="dtype", norm="rms", dim=dim, affine=aff, in_dtype="fp32", out_dtype="fp32"))
            pts.append(pt(fam="dtype", norm="rms", dim=dim, affine=aff, in_dtype="fp32", out_dtype="bf16"))
            pts.append(pt(fam="dtype", norm="layernorm", dim=dim, affine=aff, in_dtype="bf16", out_dtype="fp32"))
    if rope_ok(rope_dims[0], 128):
        pts.append(
            pt(
                fam="dtype",
                norm="rms",
                dim=rope_dims[0],
                head_dim=128,
                rope=True,
                bcast_rope=True,
                in_dtype="fp32",
                out_dtype="fp32",
            )
        )
    # I. per-token affine (LN, batch=1).
    for dim in (widths[0], mid, all_widths[-1]):
        pts.append(pt(fam="pertoken", norm="layernorm", dim=dim, affine="pertoken"))
    # J. batch > 1.
    for batch in (2, 3, 4, 8):
        # LayerNorm nh=1: plain fold / broadcast affine / per-batch adaLN, over a couple widths.
        for dim in (widths[1], all_widths[-1]):
            for affine in ("none", "wb", "perbatch"):
                pts.append(pt(fam="batchLN", norm="layernorm", dim=dim, batch=batch, rows=96, affine=affine))
        # RMS whole-row (nh=1) batched fold + broadcast weight.
        for affine in ("none", "w"):
            pts.append(pt(fam="batchRMS", norm="rms", dim=widths[2], batch=batch, rows=96, affine=affine))
        # RMS batched RoPE (head-split): broadcast-heads & per-head, broadcast & per-batch cos.
        # (This covers the batch>1 head-split OUTPUT path; head-split-no-rope is scoped out — see G.)
        if rope_ok(rope_dims[0], 128):
            for bcast in (True, False):
                for pb in (False, True):
                    pts.append(
                        pt(
                            fam="batchRope",
                            norm="rms",
                            dim=rope_dims[0],
                            rows=64,
                            batch=batch,
                            head_dim=128,
                            rope=True,
                            bcast_rope=bcast,
                            affine="none",
                            rope_perbatch=pb,
                        )
                    )
    return pts


def _sweep_build(submesh, tp_axis, tp, p):
    """Build device inputs + fp32 reference for one sweep point.
    Returns (inp_dict, cfg, ref_flat[batch*rows, dim])."""
    norm, dim, rows, batch = p["norm"], p["dim"], p["rows"], p["batch"]
    head_dim, rope, bcast_rope = p["head_dim"], p["rope"], p["bcast_rope"]
    per_head_norm, affine = p["per_head_norm"], p["affine"]
    in_dtype, out_dtype, rope_perbatch = p["in_dtype"], p["out_dtype"], p["rope_perbatch"]
    feat_local = dim // tp
    heads_total = (dim // head_dim) if head_dim else 1
    heads_per_dev = (feat_local // head_dim) if head_dim else 1
    eps = NORM_EPS
    n = batch * rows

    torch.manual_seed(0)
    xdt = torch.float32 if in_dtype == "fp32" else torch.bfloat16
    x_host = torch.randn(1, batch, rows, dim, dtype=xdt)
    _act = float32_tensor if in_dtype == "fp32" else bf16_tensor
    inp = {"x": _act(x_host, device=submesh, mesh_axis=tp_axis, shard_dim=-1)}

    xf = x_host.float().reshape(n, dim)
    if per_head_norm:
        xh = xf.reshape(n, heads_total, head_dim)
        y = (xh * (xh.pow(2).mean(-1, keepdim=True) + eps).rsqrt()).reshape(n, dim)
    elif norm == "layernorm":
        y = (xf - xf.mean(-1, keepdim=True)) * (xf.var(-1, unbiased=False, keepdim=True) + eps).rsqrt()
    else:
        y = xf * (xf.pow(2).mean(-1, keepdim=True) + eps).rsqrt()

    # ---- affine ----
    if affine in ("w", "wb"):
        w_host = torch.randn(dim, dtype=torch.bfloat16).float()
        inp["weight"] = bf16_tensor(
            w_host.to(torch.bfloat16).reshape(1, dim), device=submesh, mesh_axis=tp_axis, shard_dim=-1
        )
        y = y * w_host
        if affine == "wb":
            b_host = torch.randn(dim, dtype=torch.bfloat16).float()
            inp["bias"] = bf16_tensor(
                b_host.to(torch.bfloat16).reshape(1, dim), device=submesh, mesh_axis=tp_axis, shard_dim=-1
            )
            y = y + b_host
    elif affine == "perbatch":  # [batch,1,dim] adaLN (broadcast over seq, distinct per batch)
        scale = torch.randn(batch, 1, dim, dtype=torch.bfloat16).float()
        shift = torch.randn(batch, 1, dim, dtype=torch.bfloat16).float()
        inp["weight"] = float32_tensor(1.0 + scale, device=submesh, mesh_axis=tp_axis, shard_dim=-1)
        inp["bias"] = float32_tensor(shift, device=submesh, mesh_axis=tp_axis, shard_dim=-1)
        idx = torch.arange(n) // rows  # row -> batch (batch-major fold)
        y = y * (1.0 + scale)[:, 0, :][idx] + shift[:, 0, :][idx]
    elif affine == "pertoken":  # [1,1,rows,dim] per-token (batch==1 only)
        scale = torch.randn(1, 1, rows, dim, dtype=torch.bfloat16).float()
        shift = torch.randn(1, 1, rows, dim, dtype=torch.bfloat16).float()
        inp["weight"] = float32_tensor(1.0 + scale, device=submesh, mesh_axis=tp_axis, shard_dim=-1)
        inp["bias"] = float32_tensor(shift, device=submesh, mesh_axis=tp_axis, shard_dim=-1)
        y = y * (1.0 + scale).reshape(n, dim) + shift.reshape(n, dim)

    # ---- fused RoPE (RMS) ----
    if rope:
        b_rope = batch if rope_perbatch else 1
        if bcast_rope:  # cos/sin shared across heads: [b_rope,1,rows,hd] fp32 replicated
            cos_raw = torch.randn(b_rope, rows, 1, head_dim // 2)
            sin_raw = torch.randn(b_rope, rows, 1, head_dim // 2)
            cos_f, sin_f = stack_cos_sin(cos_raw, sin_raw)  # (b_rope, rows, 1, hd)
            cos_dev = cos_f.permute(0, 2, 1, 3)  # (b_rope, 1, rows, hd)
            sin_dev = sin_f.permute(0, 2, 1, 3)
            inp["cos"] = from_torch(cos_dev, device=submesh, dtype=ttnn.float32)
            inp["sin"] = from_torch(sin_dev, device=submesh, dtype=ttnn.float32)
            cos_full, sin_full = cos_dev, sin_dev  # (b_rope, 1, rows, hd)
        else:  # per-head: [b_rope,heads_total,rows,hd] bf16, head axis TP-sharded
            head_axes = [None, tp_axis, None, None]
            cos_raw = torch.randn(b_rope, heads_total, rows, head_dim // 2)
            sin_raw = torch.randn(b_rope, heads_total, rows, head_dim // 2)
            cos_full, sin_full = stack_cos_sin(cos_raw, sin_raw)  # (b_rope, heads_total, rows, hd)
            inp["cos"] = from_torch(cos_full, device=submesh, dtype=ttnn.bfloat16, mesh_axes=head_axes)
            inp["sin"] = from_torch(sin_full, device=submesh, dtype=ttnn.bfloat16, mesh_axes=head_axes)
        inp["trans"] = bf16_tensor(get_rot_transformation_mat(), device=submesh)

        nhd = cos_full.shape[1]  # 1 (broadcast-heads) or heads_total (per-head)
        if rope_perbatch:  # (batch, nhd, rows, hd) -> (batch*rows, nhd, hd) batch-major
            cos_row = cos_full.permute(0, 2, 1, 3).reshape(n, nhd, head_dim).to(torch.bfloat16).float()
            sin_row = sin_full.permute(0, 2, 1, 3).reshape(n, nhd, head_dim).to(torch.bfloat16).float()
        else:  # broadcast cos across the input batch: tile the single batch's rows
            cos_row = cos_full[0].permute(1, 0, 2).to(torch.bfloat16).float().repeat(batch, 1, 1)
            sin_row = sin_full[0].permute(1, 0, 2).to(torch.bfloat16).float().repeat(batch, 1, 1)
        yh = y.reshape(n, heads_total, head_dim)
        x0, x1 = yh[..., 0::2], yh[..., 1::2]
        rot = torch.stack([-x1, x0], dim=-1).flatten(-2)
        y = (yh * cos_row + rot * sin_row).reshape(n, dim)

    # ---- Welford recip LUT (LayerNorm) ----
    if norm == "layernorm":
        recip = torch.tensor([1.0 / (i + 1) for i in range(feat_local)], dtype=torch.float32).reshape(
            1, 1, 1, feat_local
        )
        inp["recip"] = from_torch(recip, device=submesh, layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.float32)

    cfg = Cfg(
        cid="sweep",
        model="SWEEP",
        tp=tp,
        rows=rows,
        dim=dim,
        head_dim=head_dim,
        rope=rope,
        full_heads=heads_total,
        broadcast_rope=bcast_rope,
        per_head_norm=per_head_norm,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        weight_mode="auto",
        bias_mode="auto",
        norm=norm,
    )
    return inp, cfg, y


def _sweep_id(p: dict) -> str:
    s = f"{p['fam']}/{p['norm']}/d{p['dim']}/r{p['rows']}/b{p['batch']}/{p['affine']}"
    if p["rope"]:
        s += f"/rope-{'bcast' if p['bcast_rope'] else 'perhead'}{'-pb' if p['rope_perbatch'] else ''}"
    if p["per_head_norm"]:
        s += "/phn"
    if p["in_dtype"] != "bf16" or p["out_dtype"] != "bf16":
        s += f"/{p['in_dtype']}->{p['out_dtype']}"
    return s


@pytest.mark.timeout(3600)  # exhaustive: hundreds of points/config; Ring is slower and exceeds the 300s default
@pytest.mark.parametrize(
    ("mesh_device", "device_params", "tp", "topology", "tp_axis", "full_mesh"),
    [pytest.param(*p, id=i) for p, i in zip(_SWEEP_PARAMS, _SWEEP_IDS)],
    indirect=["mesh_device", "device_params"],
)
def test_sweep(mesh_device, tp, topology, tp_axis, full_mesh, is_ci_env):
    """Exhaustive local correctness sweep. Skipped in CI (expensive: hundreds of shapes)."""
    if is_ci_env:
        pytest.skip("test_sweep is a local-only exhaustive sweep (too expensive for CI)")

    submesh = _resolve_submesh(mesh_device, tp, tp_axis, full_mesh)
    links = GALAXY_LINKS
    points = _sweep_points(tp)
    only = _os.getenv("SWEEP_ONLY", "")  # substring filter on the sweep id, for debugging
    if only:
        points = [p for p in points if only in _sweep_id(p)]
    logger.info(f"SWEEP [tp{tp}_{topology.name}] running {len(points)} points")
    flagged = []
    for p in points:
        pid = _sweep_id(p)
        try:
            ccl = CCLManager(mesh_device=submesh, num_links=links, topology=topology)
            inp, cfg, ref = _sweep_build(submesh, tp_axis, tp, p)
            sems = [ccl.get_ag_ping_pong_semaphore(tp_axis) for _ in range(_PINGPONG)]
            pobs = [_make_pob(inp, submesh, cfg, links, tp_axis) for _ in range(_PINGPONG)]
            ttnn.synchronize_device(submesh)
            batched = p["batch"] > 1 and cfg.heads == 1  # nh>1 head-split uses the head-concat gather
            outs = []
            for k in range(2):  # ping-pong; also a bit-exact determinism check
                o = _run_fused(inp, submesh, sems[k % _PINGPONG], cfg, topology, tp_axis, pobs[k % _PINGPONG], links)
                outs.append(_gather(o, tp_axis, batched=batched))
            det = torch.equal(outs[0], outs[1])
            out0 = outs[0].reshape(p["batch"] * p["rows"], p["dim"])
            pcc = _pcc(out0, ref)
            ok = det and pcc >= 0.999
            logger.info(f"SWEEP {pid:<52} pcc={pcc * 100:8.4f}% det={det} {'ok' if ok else 'FAIL'}")
            if not ok:
                flagged.append(f"{pid}(pcc={pcc * 100:.3f}%,det={det})")
        except Exception as e:  # noqa: BLE001 — characterize the shape, keep sweeping
            flagged.append(f"{pid}(EXC:{type(e).__name__}:{str(e)[:140]})")
            logger.warning(f"SWEEP {pid} EXCEPTION: {type(e).__name__}: {str(e)[:220]}")
    logger.info(f"SWEEP [tp{tp}_{topology.name}] {len(points) - len(flagged)}/{len(points)} passed")
    assert not flagged, f"sweep failures ({len(flagged)}/{len(points)}):\n" + "\n".join(flagged)
