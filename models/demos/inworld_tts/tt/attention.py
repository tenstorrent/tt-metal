"""TTNN implementation of bidirectional MHA with fused QKV and RoPE.

Matches VocosBackbone Attention: non-causal, fused QKV Linear(d, 3d),
RoPE applied along heads axis (interleaved pairs), output projection Linear(d, d).

All ops on device -- no host roundtrip. RoPE is computed via element-wise
multiply with precomputed cos/sin + matmul with a transformation matrix
for the pair-swap rotation.
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


def _build_rope_cos_sin_interleaved(n_heads: int, head_dim: int, base: float = 10000.0) -> tuple:
    """Build cos/sin in interleaved format for on-device RoPE.

    RoPE in VocosBackbone rotates along the HEAD axis (dim=1), not the
    sequence axis. Each of the n_heads gets a fixed rotation; all sequence
    positions within that head share the same rotation.

    Returns:
        cos_interleaved: [1, n_heads, 1, head_dim] torch.bfloat16
            Format: [c0, c0, c1, c1, ...] (interleaved duplicate)
        sin_interleaved: [1, n_heads, 1, head_dim] torch.bfloat16
    """
    half_dim = head_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2)[:half_dim].float() / head_dim))
    positions = torch.arange(n_heads, dtype=theta.dtype)
    freqs = torch.einsum("i, j -> ij", positions, theta).float()  # [n_heads, half_dim]

    cos_raw = torch.cos(freqs)  # [n_heads, half_dim]
    sin_raw = torch.sin(freqs)  # [n_heads, half_dim]

    # Interleave: [c0, c0, c1, c1, ...] -> [n_heads, head_dim]
    cos_interleaved = torch.stack([cos_raw, cos_raw], dim=-1).flatten(-2)  # [n_heads, head_dim]
    sin_interleaved = torch.stack([sin_raw, sin_raw], dim=-1).flatten(-2)

    # Add batch and seq dims for broadcasting: [1, n_heads, 1, head_dim]
    cos_interleaved = cos_interleaved.unsqueeze(0).unsqueeze(2)
    sin_interleaved = sin_interleaved.unsqueeze(0).unsqueeze(2)

    return cos_interleaved.to(torch.bfloat16), sin_interleaved.to(torch.bfloat16)


def _build_rope_transform_mat(head_dim: int) -> torch.Tensor:
    """Build the pair-swap-with-sign transformation matrix for interleaved RoPE.

    For interleaved pairs, the rotation is:
        out[2i]   = x[2i] * cos - x[2i+1] * sin
        out[2i+1] = x[2i+1] * cos + x[2i] * sin

    This equals: out = x * cos + (x @ T) * sin
    where T swaps adjacent pairs with a sign flip:
        (x @ T)[2i]   = -x[2i+1]
        (x @ T)[2i+1] =  x[2i]

    Returns:
        [1, 1, head_dim, head_dim] transformation matrix
    """
    T = torch.zeros(1, 1, head_dim, head_dim)
    for i in range(0, head_dim, 2):
        T[0, 0, i, i + 1] = 1  # x[2i] contributes positively to position 2i+1
        T[0, 0, i + 1, i] = -1  # x[2i+1] contributes negatively to position 2i
    return T


class TtAttention(LightweightModule):
    """Bidirectional MHA with fused QKV and RoPE for VocosBackbone.

    All ops on device -- QKV split, reshape, RoPE, SDPA, head merge, output projection.
    """

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

        # Precompute RoPE cos/sin on device [1, n_heads, 1, head_dim]
        cos_bf16, sin_bf16 = _build_rope_cos_sin_interleaved(n_heads, pos_emb_dim)
        self.rope_cos = ttnn.from_torch(
            cos_bf16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.rope_sin = ttnn.from_torch(
            sin_bf16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Transformation matrix for pair-swap rotation [1, 1, head_dim, head_dim]
        trans_mat = _build_rope_transform_mat(pos_emb_dim)
        self.rope_trans_mat = ttnn.from_torch(
            trans_mat,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.compute_kernel_config = get_compute_kernel_config_hifi4()

    def forward(self, x):
        """Forward pass -- all on device.

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

        # Split Q, K, V on device via slicing along last dim
        q = qkv[:, :, :, : self.dim]
        k = qkv[:, :, :, self.dim : 2 * self.dim]
        v = qkv[:, :, :, 2 * self.dim :]
        ttnn.deallocate(qkv)

        # Reshape to multi-head: [1, 1, T, dim] -> [1, T, H, D] -> [1, H, T, D]
        # The reshape splits the last dim: c = h * head_dim + d
        # The permute swaps heads and sequence dims for SDPA format
        q = ttnn.reshape(q, [1, seq_len, self.n_heads, self.head_dim])
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.reshape(k, [1, seq_len, self.n_heads, self.head_dim])
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.reshape(v, [1, seq_len, self.n_heads, self.head_dim])
        v = ttnn.permute(v, (0, 2, 1, 3))

        # Apply RoPE on device: out = x * cos + (x @ T) * sin
        # cos/sin broadcast over T (shape [1, n_heads, 1, head_dim])
        # trans_mat broadcasts over n_heads and T (shape [1, 1, head_dim, head_dim])
        q_rot = ttnn.matmul(q, self.rope_trans_mat)
        q = ttnn.add(ttnn.mul(q, self.rope_cos), ttnn.mul(q_rot, self.rope_sin))
        ttnn.deallocate(q_rot)

        k_rot = ttnn.matmul(k, self.rope_trans_mat)
        k = ttnn.add(ttnn.mul(k, self.rope_cos), ttnn.mul(k_rot, self.rope_sin))
        ttnn.deallocate(k_rot)

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
