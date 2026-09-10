# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import struct
import torch
import ttnn
from tests.ttnn.utils_for_testing import generate_all_bfloat16_bitpatterns, flush_subnormal_values_to_zero


def generate_bfloat16_bits(dtype=torch.bfloat16, include_spl_values=False):
    """
    Generate all bfloat16 bit patterns, optionally with special values replaced by zero.

    Uses generate_all_bfloat16_bitpatterns to create the exhaustive 65,536-element tensor
    of shape (256, 256). When include_spl_values is False, replaces +/-0, +/-infinity,
    NaN values with zero. Subnormals are always replaced by zero.

    Args:
        dtype (torch.dtype, optional): The target dtype to cast the bit patterns to.
                                       Defaults to torch.bfloat16.
        include_spl_values (bool, optional): If True, keep all special values (-0, +/-inf,
                                            NaN) as-is. If False, replace them
                                            with zero. Defaults to False.

    Returns:
        torch.Tensor: A 2D tensor of shape (256, 256) containing all bfloat16 bit patterns.
                     When include_spl_values is False, special values are replaced by zero.
    """
    all_bf16 = generate_all_bfloat16_bitpatterns(dtype)
    # Remember where -0.0 lives before flushing (bit pattern 0x8000 in bf16)
    neg_zero_mask = (all_bf16 == 0) & (torch.signbit(all_bf16))
    all_bf16 = flush_subnormal_values_to_zero(all_bf16)
    if include_spl_values:
        # Restore -0.0 that was destroyed by subnormal flush
        all_bf16[neg_zero_mask] = torch.tensor(-0.0, dtype=all_bf16.dtype)
    else:
        # Replace -0 with +0
        all_bf16[all_bf16 == 0] = 0.0
        # Replace +/-infinity and NaN with zero
        all_bf16[~torch.isfinite(all_bf16)] = 0.0

    return all_bf16


SMALLEST_NORMAL_BF16 = 2.0 ** (-126)


def flush_to_zero(tensor):
    """Flush values at or below the smallest normal bfloat16 to zero."""
    tensor[torch.abs(tensor) <= SMALLEST_NORMAL_BF16] = 0.0
    return tensor


def generate_bfloat16_bits_in_range(low, high, dtype=torch.bfloat16, ftz=True):
    """
    Generate all bfloat16 bit patterns within a specified [low, high] range.

    Generates all 65,536 bfloat16 bit patterns, then keeps only values that
    fall within the given range. The result is padded to a tile-compatible 2D shape.

    Args:
        low (float): Lower bound of the range (inclusive).
        high (float): Upper bound of the range (inclusive).
        dtype (torch.dtype, optional): The target dtype to cast the bit patterns to.
                                       Defaults to torch.bfloat16.
        ftz (bool, optional): If True, flush subnormal values to zero before filtering.
                             Defaults to True.

    Returns:
        torch.Tensor: A 2D tensor of shape (N, 32) containing bfloat16 values in [low, high].
                     N is the smallest multiple of 32 that fits all values in range.
                     Padded with the first valid value for tile alignment.
    """
    all_bf16 = generate_all_bfloat16_bitpatterns(torch.float32)
    all_bf16 = all_bf16.flatten()

    if ftz:
        all_bf16 = flush_subnormal_values_to_zero(all_bf16)

    mask = (all_bf16 >= low) & (all_bf16 <= high)
    filtered = all_bf16[mask].to(dtype)

    num_elements = filtered.numel()
    cols = 32
    rows = (num_elements + cols - 1) // cols
    rows = ((rows + 31) // 32) * 32  # round up to multiple of 32 for tile compatibility

    total = rows * cols
    padded = torch.full((total,), filtered[0].item(), dtype=filtered.dtype)
    padded[:num_elements] = filtered

    return padded.reshape(rows, cols)


# Mantissa codes for the binary-op sweep grid (7-bit):
#   0000000 = exact power of 2 (1.0 × 2^e)  — includes 0.5 = 2^{-1}, 1.0 = 2^0, …
#   0000001 = next value after the power of 2
#   0001000 = 1.0625 × 2^e
#   1111111 = largest value in the binade
BF16_BINARY_GRID_MANTISSAS = (0b0000000, 0b0000001, 0b0001000, 0b1111111)
BF16_BINARY_GRID_SIZE = 2048
# Fill 2032 → 2048: extra mantissa 1000000 (1.5 × 2^e) near 1.0, both signs
_BF16_BINARY_GRID_EXTRA_MANTISSA = 0b1000000
_BF16_BINARY_GRID_EXTRA_EXPONENTS = range(-3, 5)  # 8 exponents × 2 signs = 16
# +0, -0, +inf, -inf, canonical qNaN (0x7FC0)
_BF16_SPECIAL_BITS = (0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0)


def _bf16_binary_grid_extra_bits():
    """16 unique 1.5 × 2^e encodings used to fill the grid to exactly 2048."""
    extra = []
    for sign in (0, 1):
        for exponent in _BF16_BINARY_GRID_EXTRA_EXPONENTS:
            stored_exp = exponent + 127
            extra.append((sign << 15) | (stored_exp << 7) | _BF16_BINARY_GRID_EXTRA_MANTISSA)
    return extra


def generate_bfloat16_binary_grid(dtype=torch.bfloat16, include_spl_values=False):
    """
    Generate a stratified bfloat16 grid for pairwise (binary) op testing.

    Every finite-normal exponent e ∈ [-126, 127] (stored exponents 1..254) is
    included with 4 mantissa codes and both signs:

        254 exponents × 4 mantissas × 2 signs = 2032 unique finite values.

    Mantissa 0000000 is 1.0 × 2^e, so every finite-normal power of 2
    (±2^{-126} … ±2^{127}) is present — including 0.5 = 2^{-1} and 1.0 = 2^0.
    Subnormal powers of 2 (2^{-133} … 2^{-127}) and 2^{128} (overflows to inf)
    are not.

    The remaining 16 slots (2032 → 2048) are 1.5 × 2^e at e ∈ [-3, 4], both
    signs. When include_spl_values is True, the last 5 of those extras are
    replaced by +0, -0, +inf, -inf, and one canonical qNaN, keeping the length
    at exactly 2048 unique encodings.

    Args:
        dtype (torch.dtype, optional): Target dtype. Defaults to torch.bfloat16.
        include_spl_values (bool, optional): If True, replace 5 fill values with
            ±0, ±inf, and one NaN. Defaults to False.

    Returns:
        torch.Tensor: 1D tensor of length 2048, all unique bit patterns.
    """
    bits = []
    for sign in (0, 1):
        for stored_exp in range(1, 255):  # unbiased e = -126 .. 127
            for mantissa in BF16_BINARY_GRID_MANTISSAS:
                bits.append((sign << 15) | (stored_exp << 7) | mantissa)

    extra = _bf16_binary_grid_extra_bits()
    if include_spl_values:
        bits.extend(extra[: len(extra) - len(_BF16_SPECIAL_BITS)])
        bits.extend(_BF16_SPECIAL_BITS)
    else:
        bits.extend(extra)

    assert len(bits) == BF16_BINARY_GRID_SIZE
    assert len(set(bits)) == BF16_BINARY_GRID_SIZE

    return torch.tensor(bits, dtype=torch.uint16).view(torch.bfloat16).to(dtype)


def to_tt_tensor(
    input_tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    """Convert a torch tensor to a ttnn tensor on device with TILE_LAYOUT and DRAM."""
    return ttnn.from_torch(
        input_tensor,
        dtype=dtype,
        device=device,
        layout=layout,
        memory_config=memory_config,
    )


def float_to_bf16_bits(f: float) -> int:
    """Convert float to BFloat16 bits by truncating the lower 16 FP32 mantissa bits."""
    f32_bits = struct.unpack(">I", struct.pack(">f", f))[0]
    return f32_bits >> 16


def bf16_bits_to_float(bits: int) -> float:
    """Convert BFloat16 bits to float."""
    f32_bits = bits << 16
    return struct.unpack(">f", struct.pack(">I", f32_bits))[0]


def is_bf16_denormal(bits: int) -> bool:
    """Check if BF16 bits represent a denormal (subnormal) value."""
    exp = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F
    return (exp == 0) and (mantissa != 0)


def bf16_daz_normalize(bits: int) -> int:
    """Apply DAZ (Denormals-Are-Zero) normalization to BF16 bits."""
    if is_bf16_denormal(bits):
        return 0x0000
    if bits == 0x8000:  # -0 -> +0
        return 0x0000
    return bits


def bf16_value_order_index_daz(bits: int) -> int:
    """Calculate the value order index for a BFloat16 value with DAZ."""
    bits = bf16_daz_normalize(bits)

    exp = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F
    if exp == 0xFF and mantissa != 0:
        return -1  # NaN
    if bits == 0x7F80:
        return 65281  # +inf
    if bits == 0xFF80:
        return -1  # -inf
    if bits == 0x0000:
        return 32640  # Zero

    if bits & 0x8000:
        magnitude = bits & 0x7FFF
        return 0x7F7F - magnitude
    else:
        return 32640 + bits - 0x007F


def ulp_distance_bf16_daz(a: float, b: float) -> int:
    """Calculate ULP distance with DAZ+FTZ model."""
    a_bits = bf16_daz_normalize(float_to_bf16_bits(a))
    b_bits = bf16_daz_normalize(float_to_bf16_bits(b))

    a_exp = (a_bits >> 7) & 0xFF
    b_exp = (b_bits >> 7) & 0xFF
    if (a_exp == 0xFF and (a_bits & 0x7F) != 0) or (b_exp == 0xFF and (b_bits & 0x7F) != 0):
        return -1

    idx_a = bf16_value_order_index_daz(a_bits)
    idx_b = bf16_value_order_index_daz(b_bits)

    if idx_a < 0 or idx_b < 0:
        return -1

    return abs(idx_a - idx_b)


def bf16_quantize_rne(x: float) -> float:
    """RNE-quantize a float to BF16 (matches torch's BFloat16 conversion).
    Required because the bit-level helpers above truncate, but torch — and therefore
    the device input — uses round-to-nearest-even. For test points that are not
    exact BF16 values (e.g., 2.9, 3.01), truncation and RNE diverge."""
    return float(torch.tensor([x], dtype=torch.bfloat16).item())
