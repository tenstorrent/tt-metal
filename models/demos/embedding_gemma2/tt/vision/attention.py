# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VisionAttention (HF `Gemma4VisionAttention`, modeling_gemma4.py:903-969): bidirectional MHA.

    q, k, v = split_heads(x @ [Wq | Wk | Wv])                  12 heads x 64, no biases
    q = RMSNorm_64(q) * w_q;  k = RMSNorm_64(k) * w_k;  v = RMSNorm_64(v)   (v: no scale)
    q, k = axial_rope(q), axial_rope(k)                        per-patch (x, y) positions
    o = softmax(q k^T * 1.0 + mask) v                          scaling 1.0, no softcap
    out = concat_heads(o) @ Wo

Optimized path (E5b, doc/optimized_vision/README.md):
    projections   `vision_linear` program configs (batch folded into M); weight dtype, fidelity,
                  fp32 accumulation and output dtype from the precision policy (E6b,
                  `vision.weights.attn_qkv` / `attn_o`; SDPA from `vision.sdpa`, norms `vision.norms`)
    RoPE          one fused `rotary_embedding_llama` per tensor (AxialRope)
    SDPA          HiFi2 + fp32 accumulation, per-shape chunk/grid table (`vision_sdpa_program_config`),
                  bfp4 key-padding mask, heads concatenated by the SDPA writer (no concat op)
    placement     heads, norms, RoPE and SDPA outputs in L1 when they fit (`vision_act_memory_config`)

Contract
    input   x [B, 1, S, 768] bf16 TILE DRAM, S a patch bucket (1280 / 2560)
            mask [B, 1, S, S] additive TILE DRAM (bf16 or bfp4; 0 valid key, -1e9 padding key), or
                 PaddingWindows (windowed path, WINDOWED_SEQ_LENS / WINDOWED_MIN_BATCH)
            cos, sin: rope tables, [B, 1, S, 64] or the head-expanded form (AxialRope.expand);
                 [1, 1, B*S, 64] with PaddingWindows
    output  [B, 1, S, 768] bf16 TILE, `memory_config` (DRAM when None)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.embedding_gemma2.tt.config import RMSNormConfig
from models.demos.embedding_gemma2.tt.precision import VisionPrecision, resolve_precision
from models.demos.embedding_gemma2.tt.rms_norm import RMSNorm
from models.demos.embedding_gemma2.tt.vision.config import VisionAttentionConfig, vision_act_memory_config
from models.demos.embedding_gemma2.tt.vision.linear import vision_linear
from models.demos.embedding_gemma2.tt.vision.rope import AxialRope
from models.demos.embedding_gemma2.tt.weights import as_weight, linear_weight, substate

# (seq_len, batch >= key) -> (q_chunk, k_chunk, grid). E5b `sdpa_sweep.py` / `sdpa_windowed.py`
# (12 x 64 heads, HiFi2 + fp32 acc): S 2560 masked b1 0.98 ms (E2 config 2.01), b4 3.66 ms (E2
# 12.64); S 1280 windowed b8 1.99 ms (E2 masked 6.24), masked b1 0.27 ms.
_SDPA_TABLE = {
    2560: ((1, (128, 512, (13, 10))),),
    1280: ((8, (128, 256, (12, 10))), (1, (128, 256, (12, 10)))),
}


class VisionSdpaSpec(NamedTuple):
    """A tuned / searched SDPA config: chunk sizes, the op grid (None: the device grid) and the op's
    `max_cores_per_head_batch` (16 is the op default)."""

    q_chunk: int
    k_chunk: int
    grid: tuple[int, int] | None = None
    max_cores_per_head_batch: int = 16


# Tuned SDPA configs per compute grid (E10.2, doc/prefill_towers/02-attention/): VISION_SDPA_TUNED[grid][seq_len]
# replaces `_SDPA_TABLE` for that bucket on that grid only, at batch 1. Grids without an entry (the P150a's
# 13 x 10) keep the table. EG2_VISION_SDPA_TUNED=0 disables it (same-session control); VISION_SDPA_OVERRIDES
# [seq_len] (in-process, empty by default) takes precedence on every grid (the in-model search sets it).
# 12 x 10, 1 x 2560: q 256 / k 512 = 12 heads x 10 q chunks = 120 work units, one per core (q 128: 240 units,
# two waves); same k chunk, so bit-identical to the table entry; -3.0 % per forward (tt-ops e10/02-attention
# search vision_vs1 / vs3).
VISION_SDPA_TUNED: dict[tuple[int, int], dict[int, VisionSdpaSpec]] = {(12, 10): {2560: VisionSdpaSpec(256, 512)}}
VISION_SDPA_TUNED_ON = os.environ.get("EG2_VISION_SDPA_TUNED", "1") == "1"
VISION_SDPA_OVERRIDES: dict[int, VisionSdpaSpec] = {}


def vision_sdpa_spec(seq_len: int, batch: int, gx: int, gy: int) -> VisionSdpaSpec | None:
    """Tuned / searched SDPA spec for `seq_len` on grid (gx, gy), batch 1 only."""
    if batch != 1:
        return None
    spec = VISION_SDPA_OVERRIDES.get(seq_len)
    if spec is None and VISION_SDPA_TUNED_ON:
        spec = VISION_SDPA_TUNED.get((gx, gy), {}).get(seq_len)
    return spec


def vision_sdpa_program_config(
    seq_len: int, batch: int = 1, device_grid=None, exp_approx: bool = False
) -> ttnn.SDPAProgramConfig:
    """head_dim 64 SDPA config for a patch bucket. Unlisted buckets: q128 / k256 when they divide S.
    `exp_approx`: SDPA exp_approx_mode (precision policy `vision.sdpa.exp_approx`)."""
    gx, gy = (device_grid.x, device_grid.y) if device_grid is not None else (13, 10)
    q_chunk, k_chunk, grid = 128, 256, (gx, gy)
    max_cores = 16
    spec = vision_sdpa_spec(seq_len, batch, gx, gy)
    if spec is not None:
        q_chunk, k_chunk, grid, max_cores = (
            spec.q_chunk,
            spec.k_chunk,
            spec.grid or (gx, gy),
            spec.max_cores_per_head_batch,
        )
    else:
        for min_batch, cfg in _SDPA_TABLE.get(seq_len, ()):
            if batch >= min_batch:
                q_chunk, k_chunk, grid = cfg
                break
    grid = (min(grid[0], gx), min(grid[1], gy))
    while seq_len % k_chunk:
        k_chunk //= 2
    while seq_len % q_chunk:
        q_chunk //= 2
    if q_chunk < 32 or k_chunk < 32:
        raise ValueError(f"vision seq_len {seq_len} must be a multiple of 32")
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=exp_approx,
        max_cores_per_head_batch=max_cores,
    )


# Key padding as SDPA windows instead of a dense mask (E5b `sdpa_windowed.py`): the batch is
# flattened into one sequence [1, H, B*S, 64] and `cu_window_seqlens` = [0, n_0, S, S + n_1, 2S, ...,
# B*S] makes every image's valid patches attend only to themselves (block-diagonal attention;
# the padding rows form their own windows and are never read back). This needs the valid patches
# to be a prefix of each row (processor output always is). Measured per SDPA call: 8 x 1280
# 1.99 ms windowed vs 2.63 ms masked; it loses at 1 x 1280 (0.40 vs 0.27 ms) and 4 x 2560
# (3.77 vs 3.65 ms), so only these shapes use it:
WINDOWED_SEQ_LENS = (1280,)
WINDOWED_MIN_BATCH = 8


# (batch, seq_len) shapes forced onto the windowed path (in-process, empty by default; the in-model search
# compares it against the dense mask at batch 1).
WINDOWED_OVERRIDES: set[tuple[int, int]] = set()


def use_windowed_attention(batch: int, seq_len: int) -> bool:
    if (batch, seq_len) in WINDOWED_OVERRIDES:
        return True
    return seq_len in WINDOWED_SEQ_LENS and batch >= WINDOWED_MIN_BATCH


def padding_windows(num_valid: list[int], seq_len: int) -> torch.Tensor:
    """Host: valid-patch counts per row -> cu_window_seqlens [2B + 1] int32 (see WINDOWED_SEQ_LENS).

    Every n_b < seq_len for processor output (the buckets are not multiples of 9), so no window is empty.
    """
    cu = []
    for b, n in enumerate(num_valid):
        if not 0 < n < seq_len:
            raise ValueError(f"row {b}: {n} valid patches; windowed attention needs 0 < n < {seq_len}")
        cu += [b * seq_len, b * seq_len + n]
    cu.append(len(num_valid) * seq_len)
    return torch.tensor(cu, dtype=torch.int32)


class PaddingWindows:
    """Attention-mask argument for the windowed path: the device cu_window_seqlens tensor."""

    def __init__(self, cu: ttnn.Tensor):
        self.cu = cu


def sdpa_compute_kernel_config():
    """E5b default: HiFi2 + fp32 accumulation (per-op PCC 0.99994 vs fp32; HiFi4 is 5-7 % slower).
    The model takes its SDPA config from the precision policy (`vision.sdpa`)."""
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


class VisionAttention(LightweightModule):
    def __init__(
        self,
        config: VisionAttentionConfig,
        wqkv: ttnn.Tensor,
        wo: ttnn.Tensor,
        q_norm: RMSNorm,
        k_norm: RMSNorm,
        v_norm: RMSNorm,
        rope: AxialRope,
        *,
        layer_idx: int = 0,
        precision: VisionPrecision | None = None,
    ):
        super().__init__()
        precision = precision or resolve_precision(None).vision
        self.config = config
        self.wqkv = wqkv
        self.wo = wo
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.v_norm = v_norm
        self.rope = rope
        self.qkv_policy = precision.matmul("attn_qkv", layer_idx)
        self.o_policy = precision.matmul("attn_o", layer_idx)
        self.sdpa_policy = precision.sdpa_for(layer_idx)
        self.sdpa_compute_kernel_config = self.sdpa_policy.compute_kernel_config

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: VisionAttentionConfig,
        device,
        rope: AxialRope,
        *,
        cache_dir: Path | None = None,
        name: str = "vision.self_attn",
        layer_idx: int = 0,
        precision: VisionPrecision | None = None,
    ) -> "VisionAttention":
        """`state_dict` holds the HF `encoder.layers.<i>.self_attn.` sub-state (`.linear.` folded);
        `precision` the vision policy (None: the default precision file), applied as layer `layer_idx`."""
        precision = precision or resolve_precision(None).vision
        H, W = config.hidden_size, config.width
        ws = [state_dict[f"{p}_proj.weight"] for p in "qkvo"]
        expect = [(W, H), (W, H), (W, H), (H, W)]
        for p, w, shp in zip("qkvo", ws, expect):
            if tuple(w.shape) != shp:
                raise ValueError(f"{name}.{p}_proj: expected {shp}, got {tuple(w.shape)}")
        wqkv = as_weight(
            linear_weight(torch.cat(ws[:3], dim=0)),
            device,
            cache_dir=cache_dir,
            name=f"{name}.wqkv.weight",
            dtype=precision.matmul("attn_qkv", layer_idx).ttnn_dtype,
        )
        wo = as_weight(
            linear_weight(ws[3]),
            device,
            cache_dir=cache_dir,
            name=f"{name}.o_proj.weight",
            dtype=precision.matmul("attn_o", layer_idx).ttnn_dtype,
        )
        hd = RMSNormConfig(dim=config.head_dim, eps=config.rms_norm_eps)
        ckc_norm = precision.norms.compute_kernel_config
        q_norm = RMSNorm.from_state_dict(
            substate(state_dict, "q_norm"),
            hd,
            device,
            cache_dir=cache_dir,
            name=f"{name}.q_norm",
            compute_kernel_config=ckc_norm,
        )
        k_norm = RMSNorm.from_state_dict(
            substate(state_dict, "k_norm"),
            hd,
            device,
            cache_dir=cache_dir,
            name=f"{name}.k_norm",
            compute_kernel_config=ckc_norm,
        )
        v_norm = RMSNorm(RMSNormConfig(dim=config.head_dim, eps=config.rms_norm_eps, with_scale=False), None, ckc_norm)
        return cls(config, wqkv, wo, q_norm, k_norm, v_norm, rope, layer_idx=layer_idx, precision=precision)

    def forward(
        self, x: ttnn.Tensor, mask: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, memory_config=None
    ) -> ttnn.Tensor:
        """`memory_config`: output placement (DRAM when None)."""
        cfg = self.config
        B, S = x.shape[0], x.shape[2]
        dev = x.device()
        windows = mask.cu if isinstance(mask, PaddingWindows) else None
        # the [B, H, S, 64] heads and the [B, 1, S, 768] SDPA output have the bytes of a [B, 1, S, 768] tensor
        mem = vision_act_memory_config(dev, B, S, cfg.width)
        xqkv = vision_linear(
            x,
            self.wqkv,
            memory_config=vision_act_memory_config(dev, B, S, 3 * cfg.width, wide="qkv"),
            policy=self.qkv_policy,
            role="qkv",
        )
        if windows is not None:  # one [1, H, B*S, 64] sequence; cos/sin are [1, 1, B*S, 64]
            xqkv = ttnn.reshape(xqkv, (1, 1, B * S, 3 * cfg.width))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv, num_heads=cfg.num_heads, num_kv_heads=cfg.num_heads, transpose_k_heads=False, memory_config=mem
        )
        ttnn.deallocate(xqkv)

        q_n = self.q_norm(q, memory_config=mem)
        k_n = self.k_norm(k, memory_config=mem)
        v_n = self.v_norm(v, memory_config=mem)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        q_r = self.rope(q_n, cos, sin, memory_config=mem)
        k_r = self.rope(k_n, cos, sin, memory_config=mem)
        ttnn.deallocate(q_n)
        ttnn.deallocate(k_n)

        o = ttnn.transformer.scaled_dot_product_attention(
            q_r,
            k_r,
            v_n,
            is_causal=False,
            attn_mask=None if windows is not None else mask,
            cu_window_seqlens=windows,
            scale=cfg.scaling,
            program_config=vision_sdpa_program_config(
                S, B, dev.compute_with_storage_grid_size(), exp_approx=self.sdpa_policy.exp_approx
            ),
            compute_kernel_config=self.sdpa_compute_kernel_config,
            memory_config=mem,
            output_concat_heads=True,
        )
        ttnn.deallocate(q_r)
        ttnn.deallocate(k_r)
        ttnn.deallocate(v_n)
        if windows is not None:
            o = ttnn.reshape(o, (B, 1, S, cfg.width))

        out = vision_linear(
            o, self.wo, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG, policy=self.o_policy, role="o"
        )
        ttnn.deallocate(o)
        return out
