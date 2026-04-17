"""RoPE2D positional encoding for CroCo-style encoder in Fast3R.

CroCo uses 2D RoPE split over head dim: first half encodes y-axis, second half encodes x-axis.
"""
from __future__ import annotations

import torch


def rope2d_freqs(head_dim: int, base: float = 100.0, device=None, dtype=torch.float32) -> torch.Tensor:
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
    half = head_dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half, 2, device=device, dtype=dtype) / half))
    return inv  # shape (half/2,)


def build_rope2d_cos_sin(H: int, W: int, head_dim: int, base: float = 100.0, device=None, dtype=torch.float32):
    """Return (cos, sin) tables of shape (H*W, head_dim) matching CroCo RoPE2D."""
    inv = rope2d_freqs(head_dim, base=base, device=device, dtype=dtype)
    ys = torch.arange(H, device=device, dtype=dtype)
    xs = torch.arange(W, device=device, dtype=dtype)
    fy = torch.outer(ys, inv)            # (H, head_dim/4)
    fx = torch.outer(xs, inv)            # (W, head_dim/4)
    fy = torch.cat([fy, fy], dim=-1)     # (H, head_dim/2)
    fx = torch.cat([fx, fx], dim=-1)     # (W, head_dim/2)
    fy = fy[:, None, :].expand(H, W, -1)
    fx = fx[None, :, :].expand(H, W, -1)
    full = torch.cat([fy, fx], dim=-1).reshape(H * W, head_dim)
    return full.cos(), full.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 4  # split each of y/x halves into its own pair
    # y-half
    y = x[..., : d // 2]
    xh = x[..., d // 2 :]
    y_rot = torch.cat([-y[..., half:], y[..., :half]], dim=-1)
    x_rot = torch.cat([-xh[..., half:], xh[..., :half]], dim=-1)
    return torch.cat([y_rot, x_rot], dim=-1)


def apply_rope2d(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q, k: (B, H, N, Dh). cos/sin: (N, Dh)."""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out, k_out
