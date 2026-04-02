"""TTNN implementation of bidirectional MHA with fused QKV and RoPE.

Matches VocosBackbone Attention: non-causal, fused QKV Linear(d, 3d),
RoPE applied along heads axis (interleaved pairs), output projection Linear(d, d).
"""

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.inworld_tts.tt.model_config import (
    VOCOS_DIM,
    VOCOS_HEADS,
    VOCOS_POS_EMB_DIM,
    get_compute_kernel_config_hifi4,
)


def build_rope_cache(n_positions: int, dim: int, base: float = 10000.0) -> torch.Tensor:
    """Build RoPE cache matching torchtune RotaryPositionalEmbeddings.

    Returns cache of shape [n_positions, dim//2, 2] containing [cos, sin].
    """
    theta = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    seq_idx = torch.arange(n_positions, dtype=theta.dtype)
    idx_theta = torch.einsum("i, j -> ij", seq_idx, theta).float()
    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
    return cache


def apply_rope_bf16(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """Apply interleaved-pair RoPE IN BFLOAT16 precision to avoid requantization.

    The key insight: if input x is bf16 from device, applying RoPE in float32
    then converting back to bf16 adds quantization error. Instead, we stay in
    bf16 throughout so the output has the same precision as the input.

    Args:
        x: [B, H, T, D] tensor (bf16 from device)
        rope_cache: [n_positions, D//2, 2] precomputed cos/sin
    Returns:
        [B, H, T, D] with rotary embeddings applied (same dtype as input)
    """
    input_dtype = x.dtype
    seq_len = x.size(1)
    cache = rope_cache[:seq_len]

    # Stay in input dtype (bf16) to avoid requantization loss
    x_work = x
    cache_work = cache.to(input_dtype)

    # Reshape to interleaved pairs: [B, H, T, D//2, 2]
    xshaped = x_work.reshape(*x_work.shape[:-1], -1, 2)

    # Reshape cache for broadcasting: [1, H, 1, D//2, 2]
    cache_bc = cache_work.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

    # Apply rotation on interleaved pairs
    x_out = torch.stack(
        [
            xshaped[..., 0] * cache_bc[..., 0] - xshaped[..., 1] * cache_bc[..., 1],
            xshaped[..., 1] * cache_bc[..., 0] + xshaped[..., 0] * cache_bc[..., 1],
        ],
        -1,
    )

    return x_out.flatten(3).to(input_dtype)


class TtAttention(LightweightModule):
    """Bidirectional MHA with fused QKV and RoPE for VocosBackbone."""

    def __init__(
        self,
        device,
        state_dict,
        layer_num,
        n_heads=VOCOS_HEADS,
        dim=VOCOS_DIM,
        pos_emb_dim=VOCOS_POS_EMB_DIM,
        dtype=ttnn.bfloat16,
        state_dict_prefix="",
    ):
        super().__init__()
        self.device = device
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.pos_emb_dim = pos_emb_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Full compute grid for matmuls
        grid = device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)

        prefix = f"{state_dict_prefix}transformers.{layer_num}.att."

        c_attn_w = state_dict[prefix + "c_attn.weight"]
        self.c_attn = ttnn.from_torch(
            c_attn_w.T.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        c_proj_w = state_dict[prefix + "c_proj.weight"]
        self.c_proj = ttnn.from_torch(
            c_proj_w.T.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.rope_cache = build_rope_cache(n_heads, pos_emb_dim)
        self.compute_kernel_config = get_compute_kernel_config_hifi4()

    def forward(self, x):
        """Forward pass.

        Args:
            x: [1, 1, T, dim] in TILE_LAYOUT
        Returns:
            [1, 1, T, dim]
        """
        seq_len = x.shape[2]

        L1 = ttnn.L1_MEMORY_CONFIG

        # Fused QKV projection on device -> L1 (full core grid)
        qkv = ttnn.linear(
            x, self.c_attn, core_grid=self.core_grid, memory_config=L1, compute_kernel_config=self.compute_kernel_config
        )

        # Host roundtrip for QKV reshape + RoPE (unavoidable: interleaved pairs need 5D)
        qkv_torch = ttnn.to_torch(qkv)
        qkv_torch = qkv_torch.view(1, seq_len, 3, self.n_heads, self.head_dim)
        qkv_torch = qkv_torch.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        q, k, v = qkv_torch[0], qkv_torch[1], qkv_torch[2]

        # Apply RoPE in native dtype
        q = apply_rope_bf16(q, self.rope_cache)
        k = apply_rope_bf16(k, self.rope_cache)

        # Move Q,K,V to device L1 as bf16 (SDPA requires bf16)
        q = ttnn.from_torch(
            q.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=L1
        )
        k = ttnn.from_torch(
            k.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=L1
        )
        v = ttnn.from_torch(
            v.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=L1
        )

        # SDPA on device -> L1
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=self.scale,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Merge heads on device
        attn_output = ttnn.permute(attn_output, (0, 2, 1, 3))
        attn_output = ttnn.reshape(attn_output, (1, 1, seq_len, self.dim))

        # Output projection -> L1 (full core grid)
        return ttnn.linear(
            attn_output,
            self.c_proj,
            core_grid=self.core_grid,
            memory_config=L1,
            compute_kernel_config=self.compute_kernel_config,
        )
