# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
2D Axial Rotary Position Encoding (RoPE) for SAM3 ViT backbone.

The reference PyTorch implementation uses complex numbers (torch.polar, torch.view_as_complex).
Since ttnn does not support complex numbers, this module implements RoPE using real-valued
decomposition of complex multiplication:

    (a + bi)(c + di) = (ac - bd) + (ad + bc)i

where x[..., 0::2] = a (real parts) and x[..., 1::2] = b (imag parts).
The rotation applied is:
    x_rotated[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
    x_rotated[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
"""

from typing import Tuple, Optional, Dict

import torch
import ttnn


def compute_axial_cis_real(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 2D axial RoPE frequencies as separate cos and sin components.

    This is the real-valued decomposition of the complex-valued compute_axial_cis
    from sam3.model.vitdet. Produces identical results to:
        freqs_cis = compute_axial_cis(dim, end_x, end_y, theta)
        cos = freqs_cis.real.float()
        sin = freqs_cis.imag.float()

    The frequencies are split equally between x and y axes (dim//4 each), then
    concatenated, matching the reference axial decomposition.

    Args:
        dim: Head dimension. Each axis gets dim//4 frequency components.
        end_x: Number of positions along the x axis (width).
        end_y: Number of positions along the y axis (height).
        theta: Base for the geometric frequency sequence (default 10000.0).

    Returns:
        freqs_cos: Tensor of shape (end_x * end_y, dim // 2) with cosine values.
        freqs_sin: Tensor of shape (end_x * end_y, dim // 2) with sine values.
    """
    # Each axis gets dim//4 frequency components; together they fill dim//2 slots
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    # Build (x, y) position pairs for every grid cell, row-major order
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()

    # Outer product: (seq_len, dim//4) angle matrices
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)

    # Concatenate x and y frequency angles along feature dim -> (seq_len, dim//2)
    freqs = torch.cat([freqs_x, freqs_y], dim=-1)

    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)

    return freqs_cos, freqs_sin


def apply_rotary_enc_tt(
    xq: ttnn.Tensor,
    xk: ttnn.Tensor,
    freqs_cos: ttnn.Tensor,
    freqs_sin: ttnn.Tensor,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Apply 2D axial RoPE to query and key tensors using ttnn operations.

    Implements the real-valued decomposition of complex rotation:
        x_rotated[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
        x_rotated[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos

    Args:
        xq: Query tensor of shape (B, num_heads, seq_len, head_dim) in ttnn.
        xk: Key tensor of shape (B, num_heads, seq_len, head_dim) in ttnn.
        freqs_cos: Cosine frequencies of shape (1, 1, seq_len, head_dim // 2) in ttnn.
        freqs_sin: Sine frequencies of shape (1, 1, seq_len, head_dim // 2) in ttnn.

    Returns:
        Tuple of (xq_out, xk_out) rotated tensors with the same shape as inputs.

    Note:
        TODO: This implementation converts to PyTorch for the interleave/deinterleave
        reshape, then back to ttnn for the multiply/add. A future optimization should
        keep the entire operation in ttnn using ttnn.reshape and ttnn.concat to avoid
        device-to-host transfers. This is acceptable for an initial functional
        implementation while the optimization path is explored.
    """
    # Convert to torch for reshape operations (ttnn lacks fancy stride-based indexing)
    # TODO: Optimize by keeping fully in ttnn once ttnn.slice/gather support is confirmed
    device = xq.device()

    xq_torch = ttnn.to_torch(xq).float()
    xk_torch = ttnn.to_torch(xk).float()
    cos_torch = ttnn.to_torch(freqs_cos).float().squeeze(0).squeeze(0)  # (seq_len, head_dim//2)
    sin_torch = ttnn.to_torch(freqs_sin).float().squeeze(0).squeeze(0)  # (seq_len, head_dim//2)

    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (B, num_heads, seq_len, head_dim)
        # cos, sin: (seq_len, head_dim//2) -> broadcast to (1, 1, seq_len, head_dim//2)
        x_even = x[..., 0::2]  # (B, num_heads, seq_len, head_dim//2)
        x_odd = x[..., 1::2]   # (B, num_heads, seq_len, head_dim//2)

        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim//2)
        sin = sin.unsqueeze(0).unsqueeze(0)

        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        # Interleave back: stack along new last dim then flatten
        # rot_even, rot_odd: (B, num_heads, seq_len, head_dim//2)
        # -> stack -> (B, num_heads, seq_len, head_dim//2, 2) -> reshape
        out = torch.stack([rot_even, rot_odd], dim=-1)
        out = out.reshape(*x.shape)
        return out

    xq_rot = _rotate(xq_torch, cos_torch, sin_torch)
    xk_rot = _rotate(xk_torch, cos_torch, sin_torch)

    # Convert back to ttnn on same device
    dtype = ttnn.bfloat16
    xq_out = ttnn.from_torch(xq_rot.to(torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    xk_out = ttnn.from_torch(xk_rot.to(torch.bfloat16), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    return xq_out, xk_out


def precompute_freqs_cis(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    device: Optional[object] = None,
) -> Dict[str, ttnn.Tensor]:
    """Precompute RoPE cosine and sine frequencies and move them to a ttnn device.

    This is the preferred entry point for model initialization. Call once during
    setup and pass the returned dict to apply_rotary_enc_tt at each forward pass.

    Args:
        dim: Head dimension.
        end_x: Number of x-axis positions (e.g., window width).
        end_y: Number of y-axis positions (e.g., window height).
        theta: RoPE base frequency (default 10000.0).
        device: ttnn device handle. If None, tensors are returned without a device
                (useful for offline precomputation).

    Returns:
        Dictionary with keys:
            'cos': ttnn tensor of shape (1, 1, end_x*end_y, dim//2), bfloat16, TILE_LAYOUT.
            'sin': ttnn tensor of shape (1, 1, end_x*end_y, dim//2), bfloat16, TILE_LAYOUT.
    """
    freqs_cos, freqs_sin = compute_axial_cis_real(dim, end_x, end_y, theta)

    # Add batch and head dims for broadcasting: (1, 1, seq_len, head_dim//2)
    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(0)
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(0)

    tt_cos = ttnn.from_torch(
        freqs_cos.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_sin = ttnn.from_torch(
        freqs_sin.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    return {"cos": tt_cos, "sin": tt_sin}
