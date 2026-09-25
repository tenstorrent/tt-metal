# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Shared TTNN MHA core (time + group attention). Device-to-device, no residual.

reference : models/experimental/chronos_forecast/reference/chronos2/layers.py
    x_norm = RMSNorm(x)                            # T5-style, no bias
    Q/K/V = x_norm @ Wq/Wk/Wv                      # one fused matmul + head split
    Q', K' = RoPE(Q, K) iff cos/sin given          # V untouched
    ctx = softmax(Q' @ K'.T * 1.0 + mask) @ V      # scale 1.0, one SDPA call
    return merge(ctx) @ Wo
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.experimental.chronos_forecast.tt import program_configs


@dataclass(frozen=True)
class TtMhaWeights:
    """Host-side weights using ``nn.Linear`` convention: (out_features, in_features)."""

    wqkv: torch.Tensor  # (3 * inner, d) — Wq/Wk/Wv concatenated along out dim
    wo: torch.Tensor  # (d, inner)
    rms_weight: torch.Tensor  # (d,)
    num_heads: int
    head_dim: int
    eps: float = 1e-6


def maybe_upload_mask(device, mask_host: torch.Tensor, seq_len: int):
    """Upload an SDPA mask, or return None when it can be skipped.

    An all-zero mask is the identity, and the masked SDPA kernel loses a
    little accuracy at small seq lengths (and misbehaves badly at tiny ones),
    so zero masks are skipped when ``seq_len < 32``. Nonzero masks are always
    uploaded (correctness first), as are zero masks at production lengths
    where the masked path is the validated one. Callers must guard
    ``ttnn.deallocate`` against None.
    """
    import ttnn

    if bool((mask_host != 0).any().item()) or seq_len >= 32:
        return ttnn.from_torch(
            mask_host.detach().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return None


class TtMhaCore:
    """TTNN MHA core. Weights move host -> device once in ``__init__``."""

    def __init__(
        self,
        device,
        weights: TtMhaWeights,
        *,
        enable_diagonal_v_path: bool = False,
        precision: program_configs.TtChronosPrecision | None = None,
    ):
        self.device = device
        self.weights = weights
        self.precision = precision or program_configs.TtChronosPrecision()
        weight_dtype = self.precision.weight_dtype()
        self._tt = self._move_weights_to_device(device, weights, weight_dtype)
        self._diagonal_vo_weight = None
        if enable_diagonal_v_path:
            import ttnn

            inner = weights.num_heads * weights.head_dim
            v_weight = weights.wqkv[2 * inner : 3 * inner].detach().to(torch.float32).t()
            o_weight = weights.wo.detach().to(torch.float32).t()
            self._diagonal_vo_weight = ttnn.from_torch(
                (v_weight @ o_weight).contiguous(),
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    @staticmethod
    def _move_weights_to_device(device, weights: TtMhaWeights, weight_dtype):
        import ttnn

        def _weight(out_in: torch.Tensor):
            # ttnn.linear expects (in, out); torch nn.Linear stores (out, in).
            t = out_in.detach().to(torch.float32).t().contiguous()
            return ttnn.from_torch(
                t,
                dtype=weight_dtype,
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
    def _can_fuse_rope(cos, head_dim: int) -> bool:
        """``ttnn.experimental.rotary_embedding`` needs batch-shared (1,1,S,Dh)
        cos/sin and a rotate_half midpoint on a tile boundary."""
        return cos.shape[0] == 1 and cos.shape[1] == 1 and (head_dim == 32 or head_dim % 64 == 0)

    @staticmethod
    def _fused_rope(x, cos, sin, memory_config):
        """Rotate and deallocate ``x``; keeps x's logical seq length (the op
        returns the tile-padded one, which SDPA would treat as real keys)."""
        import ttnn

        out = ttnn.experimental.rotary_embedding(x, cos, sin, memory_config=memory_config)
        if tuple(out.shape) != tuple(x.shape):
            out = ttnn.reshape(out, x.shape, x.padded_shape, skip_padding_fill=True)
        ttnn.deallocate(x)
        return out

    @staticmethod
    def _rotate_half(x):
        import ttnn

        last_dim = x.shape[-1]
        half = last_dim // 2
        b, h, s = x.shape[0], x.shape[1], x.shape[2]
        x1 = ttnn.slice(x, (0, 0, 0, 0), (b, h, s, half))
        x2 = ttnn.slice(x, (0, 0, 0, half), (b, h, s, last_dim))
        return ttnn.concat([ttnn.mul(x2, -1), x1], dim=-1)

    def forward(self, x, mask=None, cos=None, sin=None, *, memory_config=None):
        """Device (B,S,d) + mask (or None) + cos/sin or None -> device (B,S,d).

        A None mask skips the SDPA mask path entirely; callers pass None for
        all-zero masks (v1 uses all-valid masks only). Borrowed inputs kept.
        Intermediates and the output use ``memory_config`` (default DRAM).
        """
        import ttnn

        mem = ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config
        wqkv, wo, rms_w = self._tt
        num_heads, head_dim = self.weights.num_heads, self.weights.head_dim
        batch, seq = x.shape[0], x.shape[1]
        # 1. RMSNorm (T5-style: no mean subtraction, no bias).
        x_norm = ttnn.rms_norm(x, epsilon=self.weights.eps, weight=rms_w, memory_config=mem)
        # 2. Fused QKV + head split. transpose_key=False: SDPA needs K as [B,H,S,Dh].
        xqkv = program_configs.linear(x_norm, wqkv, dtype=self.precision.attention_dtype(), memory_config=mem)
        ttnn.deallocate(x_norm)
        if head_dim % 32 == 0:
            q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                xqkv,
                num_heads=num_heads,
                transpose_key=False,
                memory_config=mem,
            )
            ttnn.deallocate(xqkv)
        else:
            # SDPA reject (TILE width). Split manually, apply RoPE at the true
            # will slow down the path, assuming data is passed in properly this path will never get hit
            inner = num_heads * head_dim

            def _split_head(i):
                part = ttnn.slice(xqkv, (0, 0, i * inner), (batch, seq, (i + 1) * inner))
                part = ttnn.reshape(part, (batch, seq, num_heads, head_dim))
                return ttnn.permute(part, (0, 2, 1, 3))

            q, k, v = _split_head(0), _split_head(1), _split_head(2)
            ttnn.deallocate(xqkv)
        # 3. Optional RoPE on Q/K (V untouched).
        if cos is not None and sin is not None and self._can_fuse_rope(cos, head_dim):
            q, k = self._fused_rope(q, cos, sin, mem), self._fused_rope(k, cos, sin, mem)
        elif cos is not None and sin is not None:
            q_rot = ttnn.add(ttnn.mul(q, cos), ttnn.mul(self._rotate_half(q), sin))
            k_rot = ttnn.add(ttnn.mul(k, cos), ttnn.mul(self._rotate_half(k), sin))
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            q, k = q_rot, k_rot
        if head_dim % 32 != 0:
            pad = 32 - head_dim

            def _pad_heads(t):
                padded = ttnn.pad(t, [(0, 0), (0, 0), (0, 0), (0, pad)], value=0.0)
                ttnn.deallocate(t)
                return padded

            q, k, v = _pad_heads(q), _pad_heads(k), _pad_heads(v)
        # 4-6. Scores + mask + softmax + context in one SDPA (scale=1.0, NOT 1/sqrt).
        q_chunk, k_chunk = program_configs.sdpa_chunk_sizes(q.padded_shape[-2], k.padded_shape[-2])
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=True,
        )
        compute_kernel_config = program_configs.compute_kernel_config(packer_l1_acc=False)
        ctx = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=False,
            scale=1.0,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=mem,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        if head_dim % 32 != 0:
            # Drop the SDPA head padding before merging.
            ctx_unpadded = ttnn.slice(ctx, (0, 0, 0, 0), (batch, num_heads, seq, head_dim))
            ttnn.deallocate(ctx)
            ctx = ctx_unpadded
        # 7. Merge heads + output projection (no residual; caller adds it).
        if head_dim % 32 == 0:
            merged = ttnn.transformer.concatenate_heads(ctx, memory_config=mem)
        else:
            # concatenate_heads also needs TILE-width heads; merge manually.
            ctx_t = ttnn.permute(ctx, (0, 2, 1, 3))
            merged = ttnn.reshape(ctx_t, (batch, seq, num_heads * head_dim))
            ttnn.deallocate(ctx_t)
        ttnn.deallocate(ctx)
        out = program_configs.linear(merged, wo, dtype=self.precision.sublayer_out_dtype(), memory_config=mem)
        ttnn.deallocate(merged)
        return out

    def forward_diagonal_group(self, x, *, memory_config=None):
        """Exact group-attention specialization when every group has size one.

        Softmax over one allowed key is one, so Q/K, scores, masking, softmax,
        head split, and head concat are unnecessary. The context is exactly V,
        so the sublayer is ``RMSNorm(x) @ (Wv @ Wo)``: per-token and therefore
        layout-agnostic, (B,T,d) and (T,B,d) give the same result.
        """
        import ttnn

        if self._diagonal_vo_weight is None:
            raise RuntimeError("diagonal group path was not enabled for this MHA core")
        mem = ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config
        _wqkv, _wo, rms_w = self._tt
        x_norm = ttnn.rms_norm(x, epsilon=self.weights.eps, weight=rms_w, memory_config=mem)
        out = program_configs.linear(
            x_norm, self._diagonal_vo_weight, dtype=self.precision.sublayer_out_dtype(), memory_config=mem
        )
        ttnn.deallocate(x_norm)
        return out

    __call__ = forward
