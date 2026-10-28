# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""VisionAttention (HF `Gemma4VisionAttention`, modeling_gemma4.py:903-969): bidirectional MHA.

    q, k, v = split_heads(x @ [Wq | Wk | Wv])                  12 heads x 64, no biases
    q = RMSNorm_64(q) * w_q;  k = RMSNorm_64(k) * w_k;  v = RMSNorm_64(v)   (v: no scale)
    q, k = axial_rope(q), axial_rope(k)                        per-patch (x, y) positions
    o = softmax(q k^T * 1.0 + mask) v                          scaling 1.0, no softcap
    out = concat_heads(o) @ Wo

Optimized path (E5b, doc/optimized_vision/README.md):
    projections   bf16 weights, HiFi2 + fp32 accumulation, `vision_linear` program configs
                  (batch folded into M)
    RoPE          one fused `rotary_embedding_llama` per tensor (AxialRope)
    SDPA          HiFi2 + fp32 accumulation, per-shape chunk/grid table (`vision_sdpa_program_config`),
                  bfp4 key-padding mask, heads concatenated by the SDPA writer (no concat op)
    placement     heads, norms, RoPE and SDPA outputs in L1 when they fit (`vision_act_memory_config`)

Contract
    input   x [B, 1, S, 768] bf16 TILE DRAM, S a patch bucket (1280 / 2560)
            mask [B, 1, S, S] additive TILE DRAM (bf16 or bfp4; 0 valid key, -1e9 padding key)
            cos, sin: rope tables, [B, 1, S, 64] or the head-expanded form (AxialRope.expand)
    output  [B, 1, S, 768] bf16 TILE, `memory_config` (DRAM when None)
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.config import RMSNormConfig
from models.demos.blackhole.embedding_gemma2.tt.rms_norm import RMSNorm
from models.demos.blackhole.embedding_gemma2.tt.vision.config import VisionAttentionConfig, vision_act_memory_config
from models.demos.blackhole.embedding_gemma2.tt.vision.linear import vision_linear
from models.demos.blackhole.embedding_gemma2.tt.vision.rope import AxialRope
from models.demos.blackhole.embedding_gemma2.tt.weights import as_weight, linear_weight, substate

# (seq_len, batch >= key) -> (q_chunk, k_chunk, grid). E5b `sdpa_sweep.py` (12 x 64 heads, bfp4 key
# padding mask, HiFi2 + fp32 acc): S 2560 b1 0.98 ms (E2 config 2.01), b4 3.66 ms (E2 12.64);
# S 1280 b8 2.67 ms (E2 6.24). q256 with k512 does not fit L1.
_SDPA_TABLE = {
    2560: ((1, (128, 512, (13, 10))),),
    1280: ((4, (256, 256, (12, 10))), (1, (128, 256, (13, 10)))),
}


def vision_sdpa_program_config(seq_len: int, batch: int = 1, device_grid=None) -> ttnn.SDPAProgramConfig:
    """head_dim 64 SDPA config for a patch bucket. Unlisted buckets: q128 / k256 when they divide S."""
    gx, gy = (device_grid.x, device_grid.y) if device_grid is not None else (13, 10)
    q_chunk, k_chunk, grid = 128, 256, (gx, gy)
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
        exp_approx_mode=False,
    )


def sdpa_compute_kernel_config():
    """HiFi2 + fp32 accumulation (per-op PCC 0.99994 vs fp32; HiFi4 is 5-7 % slower)."""
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
    ):
        super().__init__()
        self.config = config
        self.wqkv = wqkv
        self.wo = wo
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.v_norm = v_norm
        self.rope = rope
        self.sdpa_compute_kernel_config = sdpa_compute_kernel_config()

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
    ) -> "VisionAttention":
        """`state_dict` holds the HF `encoder.layers.<i>.self_attn.` sub-state (`.linear.` folded)."""
        H, W = config.hidden_size, config.width
        ws = [state_dict[f"{p}_proj.weight"] for p in "qkvo"]
        expect = [(W, H), (W, H), (W, H), (H, W)]
        for p, w, shp in zip("qkvo", ws, expect):
            if tuple(w.shape) != shp:
                raise ValueError(f"{name}.{p}_proj: expected {shp}, got {tuple(w.shape)}")
        wqkv = as_weight(
            linear_weight(torch.cat(ws[:3], dim=0)), device, cache_dir=cache_dir, name=f"{name}.wqkv.weight"
        )
        wo = as_weight(linear_weight(ws[3]), device, cache_dir=cache_dir, name=f"{name}.o_proj.weight")
        hd = RMSNormConfig(dim=config.head_dim, eps=config.rms_norm_eps)
        q_norm = RMSNorm.from_state_dict(
            substate(state_dict, "q_norm"), hd, device, cache_dir=cache_dir, name=f"{name}.q_norm"
        )
        k_norm = RMSNorm.from_state_dict(
            substate(state_dict, "k_norm"), hd, device, cache_dir=cache_dir, name=f"{name}.k_norm"
        )
        v_norm = RMSNorm(RMSNormConfig(dim=config.head_dim, eps=config.rms_norm_eps, with_scale=False), None)
        return cls(config, wqkv, wo, q_norm, k_norm, v_norm, rope)

    def forward(
        self, x: ttnn.Tensor, mask: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, memory_config=None
    ) -> ttnn.Tensor:
        """`memory_config`: output placement (DRAM when None)."""
        cfg = self.config
        B, S = x.shape[0], x.shape[2]
        dev = x.device()
        # the [B, H, S, 64] heads and the [B, 1, S, 768] SDPA output have the bytes of a [B, 1, S, 768] tensor
        mem = vision_act_memory_config(dev, B, S, cfg.width)
        xqkv = vision_linear(x, self.wqkv, memory_config=vision_act_memory_config(dev, B, S, 3 * cfg.width))
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
            attn_mask=mask,
            scale=cfg.scaling,
            program_config=vision_sdpa_program_config(S, B, dev.compute_with_storage_grid_size()),
            compute_kernel_config=self.sdpa_compute_kernel_config,
            memory_config=mem,
            output_concat_heads=True,
        )
        ttnn.deallocate(q_r)
        ttnn.deallocate(k_r)
        ttnn.deallocate(v_n)

        out = vision_linear(o, self.wo, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(o)
        return out
