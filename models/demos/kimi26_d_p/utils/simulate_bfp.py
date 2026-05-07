"""
BitSculpt - Block-float quantization simulator.

Simulates Tenstorrent hardware behavior: shared 8-bit exponent per 16-element
group, sign-magnitude mantissa rounding to specified bit width.

Format per element: 1 sign bit + (mantissa_bits - 1) mantissa bits.
  bfp8 = 1s + 7m, bfp4 = 1s + 3m, bfp2 = 1s + 1m.

Exponent uses IEEE 754 convention (same bias=127 as bfloat16/float32 when
stored as 8-bit unsigned on hardware; simulation uses unbiased float exponent).

Since we do offline (host-side) weight quantization, the shared exponent is
chosen optimally per group: both floor(log2(max_abs)) and floor(log2(max_abs))+1
are evaluated, and the exponent giving lower group MSE is selected.
"""

import torch

TILE_SIZE = 32
GROUP_SIZE = 16  # elements sharing one 8-bit exponent
GROUPS_PER_TILE = (TILE_SIZE * TILE_SIZE) // GROUP_SIZE  # 64


def compute_n_tiles(shape: tuple[int, int]) -> int:
    """Compute number of TILE_SIZE x TILE_SIZE tiles in a weight matrix."""
    return (shape[0] // TILE_SIZE) * (shape[1] // TILE_SIZE)


def make_homogeneous_assignment(n_tiles: int, fmt: str) -> dict:
    """Create a homogeneous precision assignment for all tiles."""
    return {i: fmt for i in range(n_tiles)}


# 8-bit exponent with IEEE 754 bias (same as bfloat16 / float32).
# Stored on hardware as uint8: stored_E = E_unbiased + EXPONENT_BIAS.
# Valid stored range [1, 254] → unbiased [MIN_EXPONENT, MAX_EXPONENT].
# (stored 0 and 255 are reserved for zero/subnormal and inf/NaN in IEEE 754;
# BFP hardware may use the full 0-255 range, but clamping to the normal range
# ensures float32 round-trip compatibility.)
EXPONENT_BIAS = 127
MIN_EXPONENT = -126  # stored exponent 1
MAX_EXPONENT = 127  # stored exponent 254


def quantize_bfp(tensor: torch.Tensor, mantissa_bits: int, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Simulate block-float quantization (offline-optimal).

    Sign-magnitude representation with symmetric clamp [-(2^(b-1)-1), 2^(b-1)-1].
    Shared exponent chosen per group to minimize MSE (tries E and E+1).
    Round-to-nearest-even (torch.round) is MSE-optimal for per-element rounding.

    Uses C++ backend when available (bit-exact, ~10x faster).

    Args:
        tensor: input values (any shape, total elements divisible by group_size)
        mantissa_bits: total bits per element including sign (2, 4, or 8)
        group_size: elements sharing one exponent (16 for TT hardware)
    Returns:
        Quantized tensor (same shape, float32 values after round-trip)
    """
    return _quantize_bfp_python(tensor, mantissa_bits, group_size)


def _quantize_bfp_python(tensor: torch.Tensor, mantissa_bits: int, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Pure-Python BFP quantization (reference implementation)."""
    orig_shape = tensor.shape
    flat = tensor.reshape(-1, group_size)

    # Shared exponent: base = floor(log2(max_abs)), clamped to hardware range.
    max_abs = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    base_exp = torch.floor(torch.log2(max_abs)).clamp(MIN_EXPONENT, MAX_EXPONENT)

    # Sign-magnitude: symmetric range [-(2^(b-1)-1), 2^(b-1)-1]
    n_levels = 2 ** (mantissa_bits - 1)
    max_mantissa = n_levels - 1

    # Offline-optimal exponent: try E and E+1, keep lower group MSE.
    # E (tighter scale) preserves precision for small values but clips the max.
    # E+1 (looser scale) avoids clipping but has 2x coarser quantization step.
    best_result = None
    best_mse = None

    for offset in (0, 1):
        exp = (base_exp + offset).clamp(max=MAX_EXPONENT)
        scale = (2.0**exp) / n_levels
        quantized = torch.round(flat / scale).clamp(-max_mantissa, max_mantissa)
        dequant = quantized * scale
        group_mse = (flat - dequant).pow(2).sum(dim=-1, keepdim=True)

        if best_result is None:
            best_result = dequant
            best_mse = group_mse
        else:
            improved = group_mse < best_mse
            best_result = torch.where(improved, dequant, best_result)
            best_mse = torch.where(improved, group_mse, best_mse)

    return best_result.reshape(orig_shape)


def quantize_bfp_with_error(
    tensor: torch.Tensor, mantissa_bits: int, group_size: int = GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize and return both quantized tensor and error vector.

    Returns:
        (quantized, error) where error = tensor - quantized (same shape).
    """
    q = quantize_bfp(tensor, mantissa_bits, group_size)
    return q, tensor.float() - q.float()


def tile_quantization_mse(tile: torch.Tensor, mantissa_bits: int) -> float:
    """MSE between original and quantized tile."""
    q = quantize_bfp(tile, mantissa_bits)
    return ((tile - q) ** 2).mean().item()


def tile_relative_error(tile: torch.Tensor, mantissa_bits: int) -> float:
    """Relative Frobenius error: ||W - Q(W)|| / ||W||."""
    q = quantize_bfp(tile, mantissa_bits)
    return torch.norm(tile - q).item() / (torch.norm(tile).item() + 1e-30)


def tile_bytes(format: str) -> int:
    """Bytes consumed by one 32x32 tile at given format."""
    return {"bfp8": 1088, "bfp4": 576, "bfp2": 320, "zero": 0}[format]


def bits_per_element(format: str) -> float:
    """Bits per element including exponent overhead."""
    return {"bfp8": 8.5, "bfp4": 4.5, "bfp2": 2.5, "zero": 0.0}[format]
