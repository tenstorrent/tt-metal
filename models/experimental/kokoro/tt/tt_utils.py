"""
Weight conversion helpers and tensor padding utilities for Kokoro TTNN port.

Convention:
  - All weights stored as ttnn.bfloat16, TILE_LAYOUT, DRAM_MEMORY_CONFIG
  - ttnn.linear expects weight in (in_features, out_features) — transpose from PyTorch
  - Helper functions take torch.Tensor in, run TTNN op, return torch.Tensor out
    so modules can stay clean and device-transfers are localised here
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import ttnn


# ─────────────────────────────────────────────
# Padding helpers
# ─────────────────────────────────────────────


def pad_to_tile_2d(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad last two dims to multiples of 32 for TILE_LAYOUT. Returns (padded, (orig_h, orig_w))."""
    assert x.dim() >= 2
    H, W = x.shape[-2], x.shape[-1]
    H_pad = ((H + 31) // 32) * 32
    W_pad = ((W + 31) // 32) * 32
    if H_pad != H or W_pad != W:
        x = F.pad(x, (0, W_pad - W, 0, H_pad - H))
    return x, (H, W)


def pad_seq_to_multiple(x: torch.Tensor, multiple: int = 32, dim: int = 1) -> Tuple[torch.Tensor, int]:
    """Pad a sequence dimension to a multiple. Returns (padded_tensor, original_length)."""
    orig = x.shape[dim]
    target = ((orig + multiple - 1) // multiple) * multiple
    if target == orig:
        return x, orig
    pad_amount = target - orig
    ndim = x.dim()
    actual_dim = dim if dim >= 0 else ndim + dim
    # F.pad works from last dim; compute index from end
    from_end = ndim - 1 - actual_dim
    padding = [0, 0] * ndim
    padding[2 * from_end + 1] = pad_amount
    return F.pad(x, padding), orig


# ─────────────────────────────────────────────
# Tensor conversion
# ─────────────────────────────────────────────


def to_tt(
    x: torch.Tensor,
    device,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    x, _ = pad_to_tile_2d(x.float().contiguous())
    return ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def from_tt(x: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(x).float()


# ─────────────────────────────────────────────
# Weight loaders
# ─────────────────────────────────────────────


def load_tt_weight(
    w: torch.Tensor,
    device,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
) -> ttnn.Tensor:
    """Generic weight load: pad to tile, store on device DRAM."""
    w = w.detach().float().contiguous()
    if w.dim() == 1:
        w = w.unsqueeze(0)  # (1, C) so it's 2-D for TILE_LAYOUT
    w, _ = pad_to_tile_2d(w)
    return ttnn.from_torch(w, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def load_tt_linear(linear: nn.Linear, device):
    """
    Load nn.Linear as TTNN weight + plain torch bias.

    PyTorch weight: (out, in)  →  TTNN linear expects (in, out) for x @ w

    Bias is returned as a plain torch.Tensor (not a TTNN tensor) because
    TTNN tile-padding of a 1D bias to (32, W) can cause incorrect broadcast
    when added inside ttnn.linear.  tt_linear adds the bias in torch instead.

    Returns (weight_tt, bias_torch_or_None, out_features_unpadded)
    """
    w = linear.weight.detach().float().T.contiguous()  # (in, out)
    weight_tt = load_tt_weight(w, device)
    bias_torch = None
    if linear.bias is not None:
        bias_torch = linear.bias.detach().float()  # (out,) — plain torch
    return weight_tt, bias_torch, linear.out_features


def load_tt_layernorm(ln: nn.LayerNorm, device):
    """
    Load LayerNorm gamma and beta as plain torch tensors.

    Not stored as TTNN tensors: passing (32, C_pad)-padded tensors to
    ttnn.layer_norm for the affine scale/shift causes incorrect results
    because TTNN sees the padded logical shape.  Instead, tt_layer_norm
    applies gamma/beta in torch after the TTNN normalization.
    """
    gamma = ln.weight.detach().float()  # (C,)
    beta = ln.bias.detach().float()  # (C,)
    return gamma, beta


# ─────────────────────────────────────────────
# TTNN operator wrappers
#
# Each wrapper:
#   - accepts torch.Tensor inputs
#   - converts to TTNN, runs the op, converts back
#   - handles dim-padding transparently
# ─────────────────────────────────────────────


def tt_linear(
    x: torch.Tensor,
    weight_tt: ttnn.Tensor,
    bias: Optional[torch.Tensor],  # plain torch Tensor (not TTNN)
    out_features: int,
    device,
) -> torch.Tensor:
    """
    ttnn.linear wrapper — maps torch.nn.Linear.

    Bias is a plain torch.Tensor added after the TTNN matmul to avoid
    TTNN tile-padding ambiguity with 1-D bias broadcast semantics.
    """
    orig_shape = x.shape[:-1]
    x_flat = x.float().reshape(-1, x.shape[-1])  # (N, in)
    N, K = x_flat.shape

    N_pad = ((N + 31) // 32) * 32
    K_pad = ((K + 31) // 32) * 32
    if N_pad != N or K_pad != K:
        buf = torch.zeros(N_pad, K_pad, dtype=torch.float32)
        buf[:N, :K] = x_flat
        x_flat = buf

    x_tt = ttnn.from_torch(
        x_flat,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Run without bias — we add it in torch to avoid TTNN broadcast issues
    out_tt = ttnn.linear(x_tt, weight_tt)
    out = from_tt(out_tt)  # (N_pad, out_pad)
    out = out[:N, :out_features]  # (N, out_features)
    if bias is not None:
        out = out + bias.to(out.device)  # broadcast over N
    return out.reshape(*orig_shape, out_features)


def tt_layer_norm(
    x: torch.Tensor,
    gamma: Optional[torch.Tensor],  # plain (C,) torch tensor
    beta: Optional[torch.Tensor],  # plain (C,) torch tensor
    eps: float,
    device,
) -> torch.Tensor:
    """
    ttnn.layer_norm wrapper — normalises over last dim.

    gamma/beta are applied in torch after TTNN normalisation to avoid
    TTNN tile-padding shape mismatches.  The expensive mean/var reduction
    runs in TTNN; the affine scale+shift is cheap in torch.
    """
    orig_shape = x.shape
    x_flat = x.float().reshape(-1, x.shape[-1])  # (N, C)
    N, C = x_flat.shape
    N_pad = ((N + 31) // 32) * 32
    C_pad = ((C + 31) // 32) * 32
    if N_pad != N or C_pad != C:
        buf = torch.zeros(N_pad, C_pad, dtype=torch.float32)
        buf[:N, :C] = x_flat
        x_flat = buf

    x_tt = ttnn.from_torch(
        x_flat,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Normalise without affine — apply gamma/beta separately in torch
    out_tt = ttnn.layer_norm(x_tt, epsilon=eps)
    out = from_tt(out_tt)[:N, :C]  # (N, C)
    if gamma is not None:
        out = out * gamma.to(out.device)
    if beta is not None:
        out = out + beta.to(out.device)
    return out.reshape(orig_shape)


def tt_leaky_relu(x: torch.Tensor, negative_slope: float, device) -> torch.Tensor:
    """ttnn.leaky_relu wrapper — maps F.leaky_relu."""
    orig = x.shape
    x_flat, (H, W) = pad_to_tile_2d(x.float().reshape(-1, x.shape[-1]).unsqueeze(0))
    x_tt = ttnn.from_torch(
        x_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tt = ttnn.leaky_relu(x_tt, negative_slope=negative_slope)
    out = from_tt(out_tt).squeeze(0)[:H, :W]
    return out.reshape(orig)


def tt_gelu(x: torch.Tensor, device) -> torch.Tensor:
    """ttnn.gelu wrapper — maps F.gelu."""
    orig = x.shape
    x_flat, (H, W) = pad_to_tile_2d(x.float().reshape(-1, x.shape[-1]).unsqueeze(0))
    x_tt = ttnn.from_torch(
        x_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tt = ttnn.gelu(x_tt)
    out = from_tt(out_tt).squeeze(0)[:H, :W]
    return out.reshape(orig)


def tt_tanh(x: torch.Tensor, device) -> torch.Tensor:
    """ttnn.tanh wrapper — maps torch.tanh."""
    orig = x.shape
    x_flat, (H, W) = pad_to_tile_2d(x.float().reshape(-1, x.shape[-1]).unsqueeze(0))
    x_tt = ttnn.from_torch(
        x_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tt = ttnn.tanh(x_tt)
    out = from_tt(out_tt).squeeze(0)[:H, :W]
    return out.reshape(orig)


def tt_sigmoid(x: torch.Tensor, device) -> torch.Tensor:
    """ttnn.sigmoid wrapper — maps torch.sigmoid."""
    orig = x.shape
    x_flat, (H, W) = pad_to_tile_2d(x.float().reshape(-1, x.shape[-1]).unsqueeze(0))
    x_tt = ttnn.from_torch(
        x_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tt = ttnn.sigmoid(x_tt)
    out = from_tt(out_tt).squeeze(0)[:H, :W]
    return out.reshape(orig)


def tt_sin(x: torch.Tensor, device) -> torch.Tensor:
    """ttnn.sin wrapper — maps torch.sin."""
    orig = x.shape
    x_flat, (H, W) = pad_to_tile_2d(x.float().reshape(-1, x.shape[-1]).unsqueeze(0))
    x_tt = ttnn.from_torch(
        x_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tt = ttnn.sin(x_tt)
    out = from_tt(out_tt).squeeze(0)[:H, :W]
    return out.reshape(orig)


def tt_cos(x: torch.Tensor, device) -> torch.Tensor:
    """ttnn.cos wrapper — maps torch.cos."""
    orig = x.shape
    x_flat, (H, W) = pad_to_tile_2d(x.float().reshape(-1, x.shape[-1]).unsqueeze(0))
    x_tt = ttnn.from_torch(
        x_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tt = ttnn.cos(x_tt)
    out = from_tt(out_tt).squeeze(0)[:H, :W]
    return out.reshape(orig)


def tt_exp(x: torch.Tensor, device) -> torch.Tensor:
    """ttnn.exp wrapper — maps torch.exp."""
    orig = x.shape
    x_flat, (H, W) = pad_to_tile_2d(x.float().reshape(-1, x.shape[-1]).unsqueeze(0))
    x_tt = ttnn.from_torch(
        x_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tt = ttnn.exp(x_tt)
    out = from_tt(out_tt).squeeze(0)[:H, :W]
    return out.reshape(orig)


def tt_rsqrt(x: torch.Tensor, device) -> torch.Tensor:
    """ttnn.rsqrt wrapper — maps torch.rsqrt."""
    orig = x.shape
    x_flat, (H, W) = pad_to_tile_2d(x.float().reshape(-1, x.shape[-1]).unsqueeze(0))
    x_tt = ttnn.from_torch(
        x_flat, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tt = ttnn.rsqrt(x_tt)
    out = from_tt(out_tt).squeeze(0)[:H, :W]
    return out.reshape(orig)


def tt_matmul(a: torch.Tensor, b: torch.Tensor, device) -> torch.Tensor:
    """ttnn.matmul wrapper — maps torch.matmul for 2-D / 3-D tensors."""
    # a: (..., M, K), b: (..., K, N)
    # Pad M, K, N to multiples of 32
    *batch, M, K = a.shape
    *_, K2, N = b.shape
    assert K == K2

    M_pad = ((M + 31) // 32) * 32
    K_pad = ((K + 31) // 32) * 32
    N_pad = ((N + 31) // 32) * 32

    a_p = torch.zeros(*batch, M_pad, K_pad, dtype=torch.float32)
    a_p[..., :M, :K] = a.float()
    b_p = torch.zeros(*batch, K_pad, N_pad, dtype=torch.float32)
    b_p[..., :K, :N] = b.float()

    a_tt = ttnn.from_torch(
        a_p, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    b_tt = ttnn.from_torch(
        b_p, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tt = ttnn.matmul(a_tt, b_tt)
    out = from_tt(out_tt)
    return out[..., :M, :N]
