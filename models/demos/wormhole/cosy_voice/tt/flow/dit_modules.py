# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementations of DiT (Diffusion Transformer) building blocks for CosyVoice3.

These modules map the PyTorch DiT architecture to TTNN operations for
execution on Tenstorrent Wormhole hardware. The DiT estimator is the
compute-critical component of the flow decoder (called 10×22 = 220 times
per inference).

Architecture per DiT block:
    AdaLayerNormZero → Attention (16h, 64d, RoPE) → gated residual
    → LayerNorm + modulation → FeedForward (1024→2048→1024) → gated residual
"""


import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtAdaLayerNormZero(LightweightModule):
    """
    Adaptive Layer Norm with Zero initialization.

    Produces 6 modulation parameters (shift_msa, scale_msa, gate_msa,
    shift_mlp, scale_mlp, gate_mlp) from the timestep embedding.

    Forward: SiLU(emb) → Linear(dim, dim*6) → chunk(6) → norm(x) * (1+scale) + shift
    """

    def __init__(self, dim, device, state_dict, prefix, dtype=ttnn.bfloat16):
        super().__init__()
        self.dim = dim

        # Linear(dim, dim*6) with bias — transpose for ttnn.linear (x @ W)
        w = state_dict[f"{prefix}.linear.weight"].T.unsqueeze(0).unsqueeze(0).contiguous()
        b = state_dict[f"{prefix}.linear.bias"].unsqueeze(0).unsqueeze(0)

        self.linear_weight = ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.linear_bias = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    def forward(self, x, emb):
        """
        Args:
            x: (1, batch, seq, dim) - input tensor
            emb: (1, 1, batch, dim) - timestep embedding

        Returns:
            norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp
            All modulation params shaped (1, batch, 1, dim) for broadcasting over seq.
        """
        # SiLU + Linear
        emb = ttnn.silu(emb)
        emb = ttnn.linear(emb, self.linear_weight, bias=self.linear_bias)

        # Transfer to host for reshape (split + transpose batch/seq dims)
        emb_host = ttnn.to_torch(emb).squeeze(0).squeeze(0).float()  # (batch, dim*6)
        chunks = torch.chunk(emb_host, 6, dim=-1)  # 6 × (batch, dim)

        # Reshape to (1, batch, 1, dim) for broadcasting over seq
        def to_device(t):
            return ttnn.from_torch(
                t.unsqueeze(0).unsqueeze(2),  # (1, batch, 1, dim)
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=x.device(),
            )

        shift_msa = to_device(chunks[0])
        scale_msa = to_device(chunks[1])
        gate_msa = to_device(chunks[2])
        shift_mlp = to_device(chunks[3])
        scale_mlp = to_device(chunks[4])
        gate_mlp = to_device(chunks[5])

        # LayerNorm (elementwise_affine=False)
        norm_x = ttnn.layer_norm(x)

        # Modulate: norm_x * (1 + scale_msa) + shift_msa
        norm_x = norm_x * (ttnn.add(scale_msa, 1.0)) + shift_msa

        return norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class TtDiTAttention(LightweightModule):
    """
    Multi-head self-attention for DiT blocks.

    Uses Q/K/V projections with bias, optional RoPE, and scaled dot-product attention.
    Dimensions: dim=1024, heads=16, dim_head=64.
    """

    def __init__(self, dim, heads, dim_head, device, state_dict, prefix, dtype=ttnn.bfloat16):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head

        # Q, K, V projections (all with bias)
        for name in ["to_q", "to_k", "to_v"]:
            w = state_dict[f"{prefix}.{name}.weight"].T.unsqueeze(0).unsqueeze(0).contiguous()
            b = state_dict[f"{prefix}.{name}.bias"].unsqueeze(0).unsqueeze(0)
            setattr(self, f"{name}_weight", ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device))
            setattr(self, f"{name}_bias", ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device))

        # Output projection (with bias)
        w = state_dict[f"{prefix}.to_out.0.weight"].T.unsqueeze(0).unsqueeze(0).contiguous()
        b = state_dict[f"{prefix}.to_out.0.bias"].unsqueeze(0).unsqueeze(0)
        self.to_out_weight = ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.to_out_bias = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    def forward(self, x, mask=None):
        """
        Args:
            x: (1, 1, seq, dim) - input after AdaLN
            mask: optional attention mask

        Returns:
            (1, 1, seq, dim) - attention output
        """
        # Project Q, K, V on device
        q = ttnn.linear(x, self.to_q_weight, bias=self.to_q_bias)
        k = ttnn.linear(x, self.to_k_weight, bias=self.to_k_bias)
        v = ttnn.linear(x, self.to_v_weight, bias=self.to_v_bias)

        # Transfer to host for attention (initial bring-up — move to device later)
        q_host = ttnn.to_torch(q).squeeze(0).float()  # (1, seq, inner_dim)
        k_host = ttnn.to_torch(k).squeeze(0).float()
        v_host = ttnn.to_torch(v).squeeze(0).float()

        # Reshape for multi-head: (batch, seq, heads*dim_head) → (batch, heads, seq, dim_head)
        b, s = q_host.shape[0], q_host.shape[1]
        q_host = q_host.view(b, s, self.heads, self.dim_head).transpose(1, 2)
        k_host = k_host.view(b, s, self.heads, self.dim_head).transpose(1, 2)
        v_host = v_host.view(b, s, self.heads, self.dim_head).transpose(1, 2)

        # Scaled dot-product attention on host
        attn_out = F.scaled_dot_product_attention(
            q_host, k_host, v_host, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).reshape(b, s, self.inner_dim)  # (batch, seq, dim)

        # Transfer back to device for output projection
        attn_tt = ttnn.from_torch(
            attn_out.unsqueeze(0),  # (1, batch, seq, dim)
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=x.device(),
        )
        out = ttnn.linear(attn_tt, self.to_out_weight, bias=self.to_out_bias)
        return out


class TtFeedForward(LightweightModule):
    """
    Feed-forward network with GELU(tanh) activation.

    Forward: Linear(dim, inner_dim) → GELU(tanh) → Linear(inner_dim, dim)
    For DiT: dim=1024, ff_mult=2, so inner_dim=2048.
    """

    def __init__(self, dim, ff_mult, device, state_dict, prefix, dtype=ttnn.bfloat16):
        super().__init__()
        inner_dim = int(dim * ff_mult)

        # Up projection: Linear(dim, inner_dim)
        w_up = state_dict[f"{prefix}.ff.0.0.weight"].T.unsqueeze(0).unsqueeze(0).contiguous()
        b_up = state_dict[f"{prefix}.ff.0.0.bias"].unsqueeze(0).unsqueeze(0)
        self.up_weight = ttnn.from_torch(w_up, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.up_bias = ttnn.from_torch(b_up, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        # Down projection: Linear(inner_dim, dim)
        w_down = state_dict[f"{prefix}.ff.2.weight"].T.unsqueeze(0).unsqueeze(0).contiguous()
        b_down = state_dict[f"{prefix}.ff.2.bias"].unsqueeze(0).unsqueeze(0)
        self.down_weight = ttnn.from_torch(w_down, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.down_bias = ttnn.from_torch(b_down, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    def forward(self, x):
        """
        Args:
            x: (1, 1, seq, dim)
        Returns:
            (1, 1, seq, dim)
        """
        x = ttnn.linear(x, self.up_weight, bias=self.up_bias)
        x = ttnn.gelu(x)  # approximate="tanh" is default in TTNN
        x = ttnn.linear(x, self.down_weight, bias=self.down_bias)
        return x


class TtDiTBlock(LightweightModule):
    """
    Single DiT transformer block.

    Combines AdaLayerNormZero, Attention, and FeedForward with gated residuals.

    Forward:
        norm, gates = AdaLN(x, t)
        x = x + gate_msa * Attention(norm)
        ff_norm = LayerNorm(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * FF(ff_norm)
    """

    def __init__(self, dim, heads, dim_head, ff_mult, device, state_dict, prefix, dtype=ttnn.bfloat16):
        super().__init__()
        self.attn_norm = TtAdaLayerNormZero(dim, device, state_dict, f"{prefix}.attn_norm", dtype)
        self.attn = TtDiTAttention(dim, heads, dim_head, device, state_dict, f"{prefix}.attn", dtype)
        self.ff = TtFeedForward(dim, ff_mult, device, state_dict, f"{prefix}.ff", dtype)

    def forward(self, x, t):
        """
        Args:
            x: (1, 1, seq, dim) - input hidden states
            t: (1, 1, batch, dim) - timestep embedding

        Returns:
            (1, 1, seq, dim) - output hidden states
        """
        # AdaLN + Attention
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, t)
        attn_out = self.attn(norm)
        x = x + gate_msa * attn_out

        # FF with modulated LayerNorm
        ff_norm = ttnn.layer_norm(x)
        ff_norm = ff_norm * (ttnn.add(scale_mlp, 1.0)) + shift_mlp
        ff_out = self.ff(ff_norm)
        x = x + gate_mlp * ff_out

        return x
