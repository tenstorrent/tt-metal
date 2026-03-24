# models/demos/blackhole/qwen3_5_9b/tt/qwen35_rope.py
"""RoPE setup for Qwen3.5-9B Gated Attention layers.

Qwen3.5 uses partial rotary embeddings: only 25% of the head dimensions
(64 out of 256) receive rotary position encoding. The remaining 192 dimensions
pass through unchanged. The gated attention TTNN op handles the partial
application internally — we just need to generate cos/sin for the rotary
portion (head_dim=64).
"""
import torch

import ttnn


def compute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10_000_000.0):
    """Compute RoPE frequency tensors (cos, sin) for given head_dim.

    Args:
        head_dim: Dimension of the rotary portion (64 for Qwen3.5).
        max_seq_len: Maximum sequence length to precompute.
        theta: RoPE base frequency.

    Returns:
        cos: torch.Tensor [max_seq_len, head_dim]
        sin: torch.Tensor [max_seq_len, head_dim]
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)  # [max_seq_len, head_dim // 2]
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1)  # [max_seq_len, head_dim]
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1)  # [max_seq_len, head_dim]
    return cos, sin


class Qwen35RoPESetup:
    """Precomputes and stores RoPE cos/sin tensors for Qwen3.5.

    Usage:
        rope = Qwen35RoPESetup(device, args)
        cos, sin = rope.get_rot_mats(position_ids)
    """

    def __init__(self, device, args):
        self.device = device
        self.head_dim = args.rope_head_dim  # 64
        self.max_seq_len = args.max_seq_len

        self.cos_cpu, self.sin_cpu = compute_rope_freqs(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            theta=args.rope_theta,
        )

        # Pre-compute full RoPE table on device for fast decode lookups
        # Shape: [1, max_seq_len, head_dim] on device
        self.cos_device = ttnn.from_torch(
            self.cos_cpu.unsqueeze(0),  # [1, max_seq_len, head_dim]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self.sin_device = ttnn.from_torch(
            self.sin_cpu.unsqueeze(0),  # [1, max_seq_len, head_dim]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    def get_rot_mats(self, position_ids: torch.Tensor):
        """Get cos/sin matrices for given positions.

        Args:
            position_ids: torch.Tensor [B, T] or [T] — position indices.

        Returns:
            cos_ttnn: ttnn.Tensor [B, T, head_dim] on device
            sin_ttnn: ttnn.Tensor [B, T, head_dim] on device
        """
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        B, T = position_ids.shape

        # Fast path for single-position decode: slice from pre-computed device table
        if T == 1 and B == 1:
            pos = position_ids.item()
            cos = self.cos_device[:, pos : pos + 1, :]
            sin = self.sin_device[:, pos : pos + 1, :]
            return cos, sin

        # General path for prefill (variable positions)
        flat_pos = position_ids.reshape(-1)
        cos = self.cos_cpu[flat_pos].reshape(B, T, self.head_dim)
        sin = self.sin_cpu[flat_pos].reshape(B, T, self.head_dim)

        cos_ttnn = ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        sin_ttnn = ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        return cos_ttnn, sin_ttnn
