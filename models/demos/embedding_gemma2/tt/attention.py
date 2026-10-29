# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Attention (HF `EmbeddingGemma2Attention`): bidirectional GQA/MQA, one class for both layer kinds.

    q, k, v = split_heads(x @ [Wq | Wk | Wv])              no biases
    q = RMSNorm_hd(q) * w_q;  k = RMSNorm_hd(k) * w_k;  v = RMSNorm_hd(v)   (v: no scale)
    q, k = rope(q), rope(k)                                  full rotation, HF rotate_half layout
    o = softmax(q k^T * 1.0 + mask) v                        scaling 1.0, no softcap, is_causal False
    out = concat_heads(o) @ Wo

Layer kinds (from config):
    sliding  head_dim 256, 4 Q heads, 2 KV heads (GQA 2), rope theta 1e4, mask = band(|q-k|<=512) + padding
    full     head_dim 512, 4 Q heads, 1 KV head  (MQA),   rope theta 1e6, mask = padding only

Contract
    input   x [B, 1, S, hidden] bf16 TILE DRAM; S a physical bucket (multiple of 32, <= rope table length)
            mask [B, 1, S, S] additive bf16 TILE DRAM (see masks.py)
    output  [B, 1, S, hidden] bf16 TILE DRAM
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.embedding_gemma2.tt.config import (
    P150_GRID,
    SLIDING,
    AttentionConfig,
    RMSNormConfig,
    act_memory_config,
    compute_grid,
    compute_kernel_config_hifi4,
)
from models.demos.embedding_gemma2.tt.linear import linear
from models.demos.embedding_gemma2.tt.precision import TextPrecision, resolve_precision
from models.demos.embedding_gemma2.tt.rms_norm import RMSNorm
from models.demos.embedding_gemma2.tt.rope import RotaryTables
from models.demos.embedding_gemma2.tt.weights import as_weight, linear_weight, substate

# Measured best (q_chunk, k_chunk, grid) per (head_dim, batch, bucket), HiFi2 + fp32 acc, dense mask
# (E5a sdpa_sweep: doc/optimized_text/README.md). Other shapes use `_sdpa_heuristic`.
_SDPA_TABLE = {
    (256, 1, 128): (32, 128, (8, 8)),
    (512, 1, 128): (64, 128, (8, 8)),
    (256, 1, 512): (32, 64, (8, 8)),
    (512, 1, 512): (32, 32, (8, 8)),
    (256, 1, 2048): (64, 128, None),
    (512, 1, 2048): (64, 64, None),
    (256, 8, 512): (128, 128, None),
    (512, 8, 512): (128, 128, None),
}


def _sdpa_heuristic(head_dim: int, batch: int, num_heads: int, seq_len: int, num_cores: int) -> tuple[int, int, bool]:
    """q_chunk: the smallest power of two that keeps (batch * heads * S / q_chunk) work units at or
    below the core count (one Q chunk per core), capped for L1 (hd 512: 128, else 256)."""
    units = batch * num_heads * seq_len
    q_chunk = 32
    cap = 128 if head_dim >= 512 else 256
    while q_chunk < cap and units / q_chunk > num_cores:
        q_chunk *= 2
    return q_chunk, 128, units / q_chunk > 64


def sdpa_program_config(
    head_dim: int, seq_len: int, batch: int = 1, num_heads: int = 4, device_grid=None, exp_approx: bool = False
) -> ttnn.SDPAProgramConfig:
    """Chunk sizes keep every core busy: the op parallelizes over batch x Q heads x Q chunks.

    Every bucket is a power of two >= 32, so S % chunk == 0 for the power-of-two chunks used here.
    """
    gx, gy = (device_grid.x, device_grid.y) if device_grid is not None else P150_GRID
    key = (head_dim, batch, seq_len)
    if key in _SDPA_TABLE:
        q_chunk, k_chunk, grid = _SDPA_TABLE[key]
        grid = grid or (gx, gy)
    else:
        q_chunk, k_chunk, full_grid = _sdpa_heuristic(head_dim, batch, num_heads, seq_len, gx * gy)
        grid = (gx, gy) if full_grid else (min(8, gx), min(8, gy))
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
        q_chunk_size=min(q_chunk, seq_len),
        k_chunk_size=min(k_chunk, seq_len),
        exp_approx_mode=exp_approx,
    )


class Attention(LightweightModule):
    def __init__(
        self,
        config: AttentionConfig,
        wqkv: ttnn.Tensor,
        wo: ttnn.Tensor,
        q_norm: RMSNorm,
        k_norm: RMSNorm,
        v_norm: RMSNorm,
        rope: RotaryTables,
        masks=None,
        precision: TextPrecision | None = None,
    ):
        super().__init__()
        self.config = config
        self.wqkv = wqkv
        self.wo = wo
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.v_norm = v_norm
        self.rope = rope
        self.masks = masks  # AttentionMasks (shared), source of the sliding-layer SDPA block map
        self.compute_kernel_config = compute_kernel_config_hifi4()
        precision = precision or resolve_precision(None).text
        i = config.layer_idx
        self.qkv_policy = precision.matmul("attn_qkv", i)
        self.o_policy = precision.matmul("attn_o", i)
        # E5a default: HiFi2 + fp32 accumulation, 2.0-2.3x faster than HiFi4 at the tuned chunks,
        # PCC vs an fp32 golden 0.99990+ (sdpa_sweep). Fidelity / fp32 acc / exp_approx: policy.
        self.sdpa_policy = precision.sdpa_for(i)
        self.sdpa_compute_kernel_config = self.sdpa_policy.compute_kernel_config

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: AttentionConfig,
        device,
        rope: RotaryTables,
        *,
        cache_dir: Path | None = None,
        name: str = "self_attn",
        masks=None,
        precision: TextPrecision | None = None,
    ) -> "Attention":
        """`state_dict` holds the HF `layers.<i>.self_attn.` sub-state; `precision` the text policy
        (None: the default precision file)."""
        precision = precision or resolve_precision(None).text
        i = config.layer_idx
        H = config.hidden_size
        wq, wk, wv, wo = (state_dict[f"{p}_proj.weight"] for p in ("q", "k", "v", "o"))
        expect = {
            "q": (config.q_width, H),
            "k": (config.kv_width, H),
            "v": (config.kv_width, H),
            "o": (H, config.q_width),
        }
        for k_, w in zip("qkvo", (wq, wk, wv, wo)):
            if tuple(w.shape) != expect[k_]:
                raise ValueError(f"{name}.{k_}_proj: expected {expect[k_]}, got {tuple(w.shape)}")
        wqkv = as_weight(
            linear_weight(torch.cat([wq, wk, wv], dim=0)),
            device,
            cache_dir=cache_dir,
            name=f"{name}.wqkv.weight",
            dtype=precision.matmul("attn_qkv", i).ttnn_dtype,
        )
        wo_t = as_weight(
            linear_weight(wo),
            device,
            cache_dir=cache_dir,
            name=f"{name}.o_proj.weight",
            dtype=precision.matmul("attn_o", i).ttnn_dtype,
        )
        hd_norm = RMSNormConfig(dim=config.head_dim, eps=config.rms_norm_eps)
        ckc_norm = precision.norms.compute_kernel_config
        q_norm = RMSNorm.from_state_dict(
            substate(state_dict, "q_norm"),
            hd_norm,
            device,
            cache_dir=cache_dir,
            name=f"{name}.q_norm",
            compute_kernel_config=ckc_norm,
        )
        k_norm = RMSNorm.from_state_dict(
            substate(state_dict, "k_norm"),
            hd_norm,
            device,
            cache_dir=cache_dir,
            name=f"{name}.k_norm",
            compute_kernel_config=ckc_norm,
        )
        v_norm = RMSNorm(RMSNormConfig(dim=config.head_dim, eps=config.rms_norm_eps, with_scale=False), None, ckc_norm)
        return cls(config, wqkv, wo_t, q_norm, k_norm, v_norm, rope, masks=masks, precision=precision)

    def forward(self, x: ttnn.Tensor, mask: ttnn.Tensor, out_memory_config=None) -> ttnn.Tensor:
        cfg = self.config
        dev = x.device()
        B, S = x.shape[0], x.shape[2]
        # intermediates in L1 when they fit (config.act_memory_config); the q tensor is the largest
        mem_qkv = act_memory_config(dev, B, S, cfg.q_width + 2 * cfg.kv_width)
        mem_q = mem_kv = act_memory_config(dev, B, S, cfg.q_width)  # one config for the q/k/v heads
        p = self.qkv_policy
        kind = "sliding" if cfg.layer_type == SLIDING else "full"  # tuned-config role suffix (tt/linear.py)
        xqkv = linear(x, self.wqkv, memory_config=mem_qkv, policy=p, dtype=p.out_ttnn_dtype, role=f"qkv_{kind}")
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            transpose_k_heads=False,
            memory_config=mem_q,
        )
        ttnn.deallocate(xqkv)

        q_n = self.q_norm(q, memory_config=mem_q)
        k_n = self.k_norm(k, memory_config=mem_kv)
        v_n = self.v_norm(v, memory_config=mem_kv)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        block_map = None
        if self.masks is not None and cfg.layer_type == SLIDING:
            block_map = self.masks.block_map(S, B)  # skips blocks outside the band (long buckets)

        q_r = self.rope(q_n, cfg.layer_type, memory_config=mem_q)
        k_r = self.rope(k_n, cfg.layer_type, memory_config=mem_kv)
        ttnn.deallocate(q_n)
        ttnn.deallocate(k_n)

        o_cat = ttnn.transformer.scaled_dot_product_attention(
            q_r,
            k_r,
            v_n,
            is_causal=False,
            attn_mask=mask,
            attn_mask_block_map=block_map,
            scale=cfg.scaling,
            program_config=sdpa_program_config(
                cfg.head_dim,
                S,
                x.shape[0],
                cfg.num_heads,
                compute_grid(x.device()),
                exp_approx=self.sdpa_policy.exp_approx,
            ),
            compute_kernel_config=self.sdpa_compute_kernel_config,
            memory_config=mem_q,
            output_concat_heads=True,  # [B,1,S,H*hd] directly: no nlp_concat_heads
        )
        ttnn.deallocate(q_r)
        ttnn.deallocate(k_r)
        ttnn.deallocate(v_n)

        if out_memory_config is None:
            out_memory_config = act_memory_config(dev, B, S, cfg.hidden_size)
        p = self.o_policy
        out = linear(
            o_cat, self.wo, memory_config=out_memory_config, policy=p, dtype=p.out_ttnn_dtype, role=f"o_{kind}"
        )
        ttnn.deallocate(o_cat)
        return out
