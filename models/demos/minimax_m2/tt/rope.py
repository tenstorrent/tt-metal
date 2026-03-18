# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Partial RoPE for MiniMax-M2.5.

rotary_dim = 64  (first 64 of head_dim=128 get RoPE; remaining 64 are NoPE passthrough).

Implementation strategy:
  1. Build cos/sin cache on CPU for [max_seq_len, rotary_dim].
  2. Store as TTNN tensors [1, 1, max_seq_len, rotary_dim] on device or MeshDevice.
  3. apply_partial_rope():
       - Slice Q/K along last dim at rotary_dim boundary.
       - rotate_half(q_rot): split at rotary_dim//2, negate, swap halves.
       - q_rot_embed = q_rot * cos + rotate_half(q_rot) * sin
       - Concat [q_rot_embed, q_pass].

MeshDevice support:
  cos/sin matrices are replicated across all devices so each device can apply
  local RoPE independently to its TP-sharded Q/K heads.
"""

import torch

import ttnn


def _build_rope_cache(
    seq_len: int,
    rotary_dim: int,
    rope_theta: float,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin for partial RoPE.

    Returns:
        cos, sin: [1, 1, seq_len, rotary_dim]
    """
    half = rotary_dim // 2
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [seq_len, half]
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, rotary_dim]
    cos = emb.cos().to(dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, rotary_dim]
    sin = emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)
    return cos, sin


class PartialRoPESetup:
    """Holds cos/sin on device (or MeshDevice); provides get_cos_sin() and get_single_position()."""

    def __init__(
        self,
        device,
        rotary_dim: int = 64,
        rope_theta: float = 5_000_000.0,
        max_seq_len: int = 4096,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.rotary_dim = rotary_dim
        self.half = rotary_dim // 2

        cos_t, sin_t = _build_rope_cache(max_seq_len, rotary_dim, rope_theta)

        # On MeshDevice: replicate cos/sin across all chips so local RoPE works per device
        mesh_mapper = None
        if isinstance(device, ttnn.MeshDevice):
            mesh_mapper = ttnn.ReplicateTensorToMesh(device)

        self.cos_matrix = ttnn.from_torch(
            cos_t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        self.sin_matrix = ttnn.from_torch(
            sin_t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

    def get_cos_sin(self, seq_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Slice cos/sin to [1, 1, seq_len, rotary_dim]."""
        cos = ttnn.slice(self.cos_matrix, (0, 0, 0, 0), (1, 1, seq_len, self.rotary_dim))
        sin = ttnn.slice(self.sin_matrix, (0, 0, 0, 0), (1, 1, seq_len, self.rotary_dim))
        return cos, sin

    def get_single_position(self, pos: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Slice cos/sin for a single position → [1, 1, 1, rotary_dim].

        Used in decode to avoid growing tensor slicing.
        """
        cos = ttnn.slice(self.cos_matrix, (0, 0, pos, 0), (1, 1, pos + 1, self.rotary_dim))
        sin = ttnn.slice(self.sin_matrix, (0, 0, pos, 0), (1, 1, pos + 1, self.rotary_dim))
        return cos, sin


def apply_partial_rope(
    q: ttnn.Tensor,
    k: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    rotary_dim: int = 64,
    head_dim: int = 128,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Apply partial RoPE to Q and K tensors.

    Works on both single device and MeshDevice (per-device operation, no CCL needed).

    Args:
        q:   [B, num_q_heads,  S, head_dim]  (or local TP shard)
        k:   [B, num_kv_heads, S, head_dim]  (or local TP shard)
        cos: [1, 1, S, rotary_dim]
        sin: [1, 1, S, rotary_dim]

    Returns:
        q, k with RoPE applied to first rotary_dim dims; rest unchanged.
    """
    half = rotary_dim // 2
    B_q, nq, S, D = q.shape
    B_k, nk, _, _ = k.shape

    def _apply_rope_to(x, num_heads):
        B, nh, seq, _ = x.shape
        x_rot = ttnn.slice(x, (0, 0, 0, 0), (B, nh, seq, rotary_dim))
        x_pass = ttnn.slice(x, (0, 0, 0, rotary_dim), (B, nh, seq, head_dim))

        # rotate_half: split at half, negate second chunk, swap
        x1 = ttnn.slice(x_rot, (0, 0, 0, 0), (B, nh, seq, half))
        x2 = ttnn.slice(x_rot, (0, 0, 0, half), (B, nh, seq, rotary_dim))
        x2_neg = ttnn.neg(x2)
        x_rot_half = ttnn.concat([x2_neg, x1], dim=-1)

        x_rot_embed = ttnn.add(ttnn.mul(x_rot, cos), ttnn.mul(x_rot_half, sin))
        return ttnn.concat([x_rot_embed, x_pass], dim=-1)

    q_out = _apply_rope_to(q, nq)
    k_out = _apply_rope_to(k, nk)
    return q_out, k_out
