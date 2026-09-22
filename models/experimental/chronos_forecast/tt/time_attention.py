# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Single-chip TTNN TimeSelfAttention for Chronos-2 (encoder sublayer 1)
x_norm = RMSNorm(x)                        # (B, T, d), T5-style, no bias
Q/K/V = x_norm @ Wq/Wk/Wv (no bias)        # -> (B, H, T, Dh)
Q', K' = RoPE(Q, K, position_ids)          # V untouched
ctx = softmax(Q' @ K'.T * 1.0 + mask) @ V  # scale is 1.0, NOT 1/sqrt(Dh)
out = merge(ctx) @ Wo (no bias)            # (B, T, d)
return x + out
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TtTimeAttentionWeights:
    """Host-side weights using ``nn.Linear`` convention: (out_features, in_features)."""

    wqkv: torch.Tensor  # (3 * inner, d) — Wq/Wk/Wv concatenated along out dim
    wo: torch.Tensor  # (d, inner)
    rms_weight: torch.Tensor  # (d,)
    inv_freq: torch.Tensor  # (Dh // 2,) RoPE buffer (not trained)
    num_heads: int
    head_dim: int
    eps: float = 1e-6

    @classmethod
    def from_torch_layer(cls, layer) -> "TtTimeAttentionWeights":
        """Extract weights from a reference ``TimeSelfAttention`` (or matching module)."""
        mha = layer.self_attention
        wqkv = torch.cat(
            [mha.q.weight.detach(), mha.k.weight.detach(), mha.v.weight.detach()], dim=0
        ).clone()
        return cls(
            wqkv=wqkv,
            wo=mha.o.weight.detach().clone(),
            rms_weight=layer.layer_norm.weight.detach().clone(),
            inv_freq=mha.rope_embed.inv_freq.detach().clone(),
            num_heads=mha.n_heads,
            head_dim=mha.kv_proj_dim,
            eps=layer.layer_norm.variance_epsilon,
        )


def build_rope_cache(
    position_ids: torch.Tensor, inv_freq: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Host RoPE cos/sin replicating ``Chronos2RotaryEmbedding.forward`` (fp32).

    Args:
        position_ids: (B, T) long.
        inv_freq: (Dh // 2,) float.

    Returns:
        (cos, sin), each (B, T, Dh) float32.
    """
    with torch.no_grad():
        inv_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos_expanded = position_ids[:, None, :].float()
        freqs = (inv_expanded.float() @ pos_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class TtTimeAttention:
    """TTNN time self-attention. Weights move host -> device once in ``__init__``."""

    def __init__(self, device, weights: TtTimeAttentionWeights):
        self.device = device
        self.weights = weights
        self._tt = self._move_weights_to_device(device, weights)

    @staticmethod
    def _move_weights_to_device(device, weights: TtTimeAttentionWeights):
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

    def forward(
        self,
        x_host: torch.Tensor,
        cos_host: torch.Tensor,
        sin_host: torch.Tensor,
        mask_host: torch.Tensor,
    ) -> torch.Tensor:
        """Forward starting from host inputs. Returns host torch (float32) for PCC.

        Args:
            x_host: (B, T, d) float.
            cos_host / sin_host: (B, T, Dh) float32 from :func:`build_rope_cache`.
            mask_host: (1, 1, T, T) additive (0 valid / large-negative invalid).
        """
        import ttnn

        wqkv, wo, rms_w = self._tt
        b, t, _d = x_host.shape

        x = ttnn.from_torch(
            x_host.detach().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # 1. RMSNorm (T5-style: no mean subtraction, no bias).
        x_norm = ttnn.rms_norm(x, epsilon=self.weights.eps, weight=rms_w)
        # 2. Fused QKV + head split. transpose_key=False: SDPA needs K as [B,H,T,Dh].
        xqkv = ttnn.linear(x_norm, wqkv, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x_norm)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
            xqkv,
            num_heads=self.weights.num_heads,
            transpose_key=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)
        # 3. RoPE on Q/K (V untouched). cos/sin (B,T,Dh) -> (B,1,T,Dh) broadcast.
        cos = ttnn.from_torch(
            cos_host.detach().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin = ttnn.from_torch(
            sin_host.detach().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cos = ttnn.unsqueeze(cos, 1)
        sin = ttnn.unsqueeze(sin, 1)
        q_rot = ttnn.add(ttnn.mul(q, cos), ttnn.mul(self._rotate_half(q), sin))
        k_rot = ttnn.add(ttnn.mul(k, cos), ttnn.mul(self._rotate_half(k), sin))
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        # 4-6. Scores + mask + softmax + context in one SDPA (scale=1.0, NOT 1/sqrt).
        mask = ttnn.from_torch(
            mask_host.detach().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
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
            q_rot,
            k_rot,
            v,
            attn_mask=mask,
            is_causal=False,
            scale=1.0,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_rot)
        ttnn.deallocate(k_rot)
        ttnn.deallocate(v)
        ttnn.deallocate(mask)
        # 7. Merge heads, output projection, residual add.
        merged = ttnn.transformer.concatenate_heads(ctx, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(ctx)
        out = ttnn.linear(merged, wo, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(merged)
        if out.memory_config() != x.memory_config():
            out = ttnn.to_memory_config(out, x.memory_config())
        y = ttnn.add(x, out, memory_config=x.memory_config())
        ttnn.deallocate(x)
        ttnn.deallocate(out)
        # Drop seq tile padding on host; return float for PCC.
        return ttnn.to_torch(y).float()[:, :t, :]

    __call__ = forward
