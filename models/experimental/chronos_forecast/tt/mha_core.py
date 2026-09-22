# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Shared single-chip TTNN MHA core for Chronos-2 encoder sublayers.

Device-to-device: callers upload host inputs, the core runs RMSNorm ->
fused-QKV MHA (+ optional RoPE) -> output projection, and callers own the
residual add and the host round-trip. Used by time attention (with RoPE) and
group attention (without RoPE, transposed layout).

Oracle math mirrors ``reference/chronos2/layers.py`` ``MHA`` (eval mode,
no QKV/output bias, attention scale 1.0)::

    x_norm = RMSNorm(x)
    Q/K/V = x_norm @ Wq/Wk/Wv            # one fused matmul + head split
    Q', K' = RoPE(Q, K) iff cos/sin given (V untouched)
    ctx = softmax(Q' @ K'.T * 1.0 + mask) @ V   # one SDPA call
    return merge(ctx) @ Wo
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TtMhaWeights:
    """Host-side weights using ``nn.Linear`` convention: (out_features, in_features)."""

    wqkv: torch.Tensor  # (3 * inner, d) — Wq/Wk/Wv concatenated along out dim
    wo: torch.Tensor  # (d, inner)
    rms_weight: torch.Tensor  # (d,)
    num_heads: int
    head_dim: int
    eps: float = 1e-6


class TtMhaCore:
    """TTNN MHA core. Weights move host -> device once in ``__init__``."""

    def __init__(self, device, weights: TtMhaWeights):
        self.device = device
        self.weights = weights
        self._tt = self._move_weights_to_device(device, weights)

    @staticmethod
    def _move_weights_to_device(device, weights: TtMhaWeights):
        import ttnn

        def _weight(out_in: torch.Tensor):
            # ttnn.linear expects (in, out); torch nn.Linear stores (out, in).
            t = out_in.detach().to(torch.float32).t().contiguous()
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        rms_w = ttnn.from_torch(
            weights.rms_weight.detach().to(torch.float32).reshape(1, -1).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return (_weight(weights.wqkv), _weight(weights.wo), rms_w)

    @staticmethod
    def _rotate_half(x):
        import ttnn

        last_dim = x.shape[-1]
        half = last_dim // 2
        b, h, s = x.shape[0], x.shape[1], x.shape[2]
        x1 = ttnn.slice(x, (0, 0, 0, 0), (b, h, s, half))
        x2 = ttnn.slice(x, (0, 0, 0, half), (b, h, s, last_dim))
        return ttnn.concat([ttnn.mul(x2, -1), x1], dim=-1)

    def forward(self, x, mask, cos=None, sin=None):
        """Device-to-device MHA (no residual).

        Args:
            x: device tensor (B, S, d), TILE.
            mask: device tensor (B|1, 1, S, S) additive, TILE + DRAM.
            cos / sin: device tensors (B, 1, S, Dh) or None to skip RoPE.

        Returns:
            Device tensor (B, S, d): merged heads @ Wo.
        """
        import ttnn

        wqkv, wo, rms_w = self._tt
        # 1. RMSNorm (T5-style: no mean subtraction, no bias).
        x_norm = ttnn.rms_norm(x, epsilon=self.weights.eps, weight=rms_w)
        # 2. Fused QKV + head split. transpose_key=False: SDPA needs K as [B,H,S,Dh].
        xqkv = ttnn.linear(x_norm, wqkv, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x_norm)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            xqkv,
            num_heads=self.weights.num_heads,
            transpose_key=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)
        # 3. Optional RoPE on Q/K (V untouched).
        if cos is not None and sin is not None:
            q_rot = ttnn.add(ttnn.mul(q, cos), ttnn.mul(self._rotate_half(q), sin))
            k_rot = ttnn.add(ttnn.mul(k, cos), ttnn.mul(self._rotate_half(k), sin))
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            q, k = q_rot, k_rot
        # 4-6. Scores + mask + softmax + context in one SDPA (scale=1.0, NOT 1/sqrt).
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=32,
            k_chunk_size=32,
            exp_approx_mode=True,
        )
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        ctx = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=False,
            scale=1.0,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        # 7. Merge heads + output projection (no residual; caller adds it).
        merged = ttnn.transformer.concatenate_heads(ctx, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(ctx)
        out = ttnn.linear(merged, wo, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(merged)
        return out

    __call__ = forward
