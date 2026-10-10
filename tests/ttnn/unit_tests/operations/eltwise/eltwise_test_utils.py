# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import struct

import torch
import ttnn
from mpmath import cosh as mp_cosh
from mpmath import mp

from tests.ttnn.utils_for_testing import (
    flush_subnormal_values_to_zero,
    generate_all_bfloat16_bitpatterns,
)


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
MAX_BF16 = float(torch.finfo(torch.bfloat16).max)


def flush_to_zero(tensor):
    """Flush values at or below the smallest normal bfloat16 to zero. |x| ≤ 2^{-126}"""
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


def generate_float32_bits(include_spl_values=False):
    """
    Generate every bfloat16 bit pattern, stored as float32.

    A float32 accuracy sweep cannot enumerate all 2^32 encodings. Its input
    domain is the bfloat16 lattice: the 65,536 bfloat16 values, promoted to
    float32 without rounding. Each value is an exact float32 whose lower 16
    mantissa bits are zero (the bfloat16 encoding shifted left by 16).

    Subnormals are flushed to zero. Float32 and bfloat16 share the exponent
    range, and the hardware flushes both. When include_spl_values is False,
    +/-0, +/-infinity, and NaN are replaced by +0 as well.

    Args:
        include_spl_values (bool, optional): If True, keep -0, +/-inf, and NaN.
            If False, replace them with +0. Defaults to False.

    Returns:
        torch.Tensor: Shape (256, 256), dtype float32.
    """
    return generate_bfloat16_bits(dtype=torch.float32, include_spl_values=include_spl_values)


def generate_float32_bits_in_range(low, high, ftz=True):
    """
    Bfloat16 values inside [low, high], stored as float32.

    Same value set as generate_bfloat16_bits_in_range: every bfloat16 encoding
    that falls in range, promoted to float32 and padded to a tile-aligned
    (N, 32) shape. N is the smallest multiple of 32 that fits the filtered
    values; padding repeats the first in-range value.

    Args:
        low (float): Lower bound of the range (inclusive).
        high (float): Upper bound of the range (inclusive).
        ftz (bool, optional): If True, flush subnormal values to zero before
            filtering. Defaults to True.

    Returns:
        torch.Tensor: Shape (N, 32), dtype float32.
    """
    return generate_bfloat16_bits_in_range(low, high, dtype=torch.float32, ftz=ftz)


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


def generate_bfloat16_binary_grid(dtype=torch.bfloat16, include_spl_values=False, include_zero=False):
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
    at exactly 2048 unique encodings. When include_zero is True (and
    include_spl_values is False), only +0 replaces one fill value.

    Args:
        dtype (torch.dtype, optional): Target dtype. Defaults to torch.bfloat16.
        include_spl_values (bool, optional): If True, replace 5 fill values with
            ±0, ±inf, and one NaN. Defaults to False.
        include_zero (bool, optional): If True and include_spl_values is False,
            replace one fill value with +0. Defaults to False.

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
    elif include_zero:
        bits.extend(extra[:-1])
        bits.append(0x0000)  # +0
    else:
        bits.extend(extra)

    assert len(bits) == BF16_BINARY_GRID_SIZE
    assert len(set(bits)) == BF16_BINARY_GRID_SIZE

    return torch.tensor(bits, dtype=torch.uint16).view(torch.bfloat16).to(dtype)


def binary_grid_values(
    low=-float("inf"), high=float("inf"), min_magnitude=0.0, include_zero=True, dtype=torch.bfloat16
):
    """Binary-grid values restricted to a domain, for ops that are only defined
    (or only well-conditioned) on part of the bfloat16 range.

    Keeps the grid's stratification. An exponent that lies wholly inside
    [low, high] still carries all 4 mantissa codes and both signs. A bound
    that cuts a binade keeps only the codes that fall inside it — the
    mantissas are 1.0, 1.0078, 1.0625, 1.9922, so ±80, ±100, and ±1e19 each
    drop a code at the edge. The filter is by value, not by exponent. An
    outer product of two restricted sets stays a few million elements
    instead of the billions an exhaustive in-range sweep would need.

    Args:
        low, high (float, optional): Inclusive value bounds. Default unbounded.
        min_magnitude (float, optional): Drop values with |v| below this, e.g.
            operands whose square would underflow. Defaults to 0.0.
        include_zero (bool, optional): Keep +0 when it is inside [low, high],
            exempting it from min_magnitude. Defaults to True.
        dtype (torch.dtype, optional): Target dtype. Defaults to torch.bfloat16.

    Returns:
        torch.Tensor: 1D tensor of the surviving values, in grid order.
    """
    values = generate_bfloat16_binary_grid(dtype=dtype, include_zero=include_zero)
    in_range = (values >= low) & (values <= high)
    keep = in_range & (values.abs() >= min_magnitude)
    if include_zero:
        keep |= in_range & (values == 0)
    return values[keep].contiguous()


def pairwise_from_values(values_a, values_b=None):
    """Outer product of two value sets: A[i, j] = values_a[i], B[i, j] = values_b[j]."""
    if values_b is None:
        values_b = values_a
    a, b = torch.meshgrid(values_a, values_b, indexing="ij")
    return a.contiguous(), b.contiguous()


def pairwise_inputs(include_spl_values=False, include_zero=False, dtype=torch.bfloat16):
    """Outer product of the 2048-value binary grid: A[i, j] = v[i], B[i, j] = v[j]."""
    values = generate_bfloat16_binary_grid(
        dtype=dtype, include_spl_values=include_spl_values, include_zero=include_zero
    )
    return pairwise_from_values(values)


# The binary grid cubed is 8.6e9 triples (17 GB per bf16 operand), so the
# ternary grid trades exponent resolution — not mantissa resolution — for size:
#
#     2 signs × 32 exponents × 4 mantissas = 256 values → 256^3 = 16,777,216
#
# All 4 mantissa codes are kept because device/IEEE divergence lives in the
# rounding, not in the binade: dropping to 1 mantissa would make every operand
# an exact power of 2, where mul and div are exponent arithmetic and exact.
BF16_TERNARY_GRID_SIZE = 256
BF16_TERNARY_GRID_EXPONENT_COUNT = BF16_TERNARY_GRID_SIZE // (2 * len(BF16_BINARY_GRID_MANTISSAS))
# Exponents where the binary sweeps found device behaviour that a uniform
# stride would be free to miss: the min-normal edge (-126), the FPU mul/div
# underflow band (-121, -120), the unit binades, the 0x7AFF addend of the FPU
# add overflow pair (118), and the max binade (126, 127).
_BF16_TERNARY_PINNED_EXPONENTS = (-126, -125, -121, -120, -2, -1, 0, 1, 2, 118, 124, 125, 126, 127)


def _bf16_ternary_grid_exponents():
    """32 unbiased exponents: the pinned fences plus an even spread over the rest."""
    pinned = set(_BF16_TERNARY_PINNED_EXPONENTS)
    rest = [e for e in range(-126, 128) if e not in pinned]
    sample_count = BF16_TERNARY_GRID_EXPONENT_COUNT - len(pinned)
    step = len(rest) / sample_count
    sampled = [rest[int(i * step)] for i in range(sample_count)]

    exponents = sorted(pinned | set(sampled))
    assert len(exponents) == BF16_TERNARY_GRID_EXPONENT_COUNT
    return exponents


def _bf16_ternary_sacrificial_indices(bits, count):
    """Slots to overwrite with special values.

    The grid is exactly 256 wide with no fill, so a special value has to take a
    finite slot. Spend the 1.0625 × 2^e code at non-pinned exponents: it is the
    generic mid-binade sample, and the binade keeps its other three codes.
    """
    if count == 0:
        return []

    pinned = set(_BF16_TERNARY_PINNED_EXPONENTS)
    candidates = [
        i
        for i, b in enumerate(bits)
        if (b & 0x7F) == 0b0001000 and (((b >> 7) & 0xFF) - 127) not in pinned and b not in _BF16_SPECIAL_BITS
    ]
    assert len(candidates) >= count
    # Spread the sacrifices across the exponent range instead of clustering
    # them at the bottom, which is where the enumeration starts.
    step = len(candidates) / count
    return [candidates[int(i * step)] for i in range(count)]


def generate_bfloat16_ternary_grid(dtype=torch.bfloat16, include_spl_values=False, include_zero=False):
    """
    Generate a stratified bfloat16 grid for 3-way (ternary) op testing.

    32 of the 254 finite-normal exponents are carried with all 4 mantissa codes
    of the binary grid and both signs:

        2 signs × 32 exponents × 4 mantissas = 256 unique finite values.

    The exponents are the fences in _BF16_TERNARY_PINNED_EXPONENTS plus an even
    spread over the remainder, so the overflow and underflow boundaries the
    binary sweeps documented stay in the sweep. Mantissa 0000000 is 1.0 × 2^e,
    so the sampled exponents still contribute their exact powers of 2.

    Unlike the binary grid this one has no fill slots, so include_spl_values and
    include_zero overwrite 1.0625 × 2^e entries at non-pinned exponents, keeping
    the length at exactly 256 unique encodings.

    Args:
        dtype (torch.dtype, optional): Target dtype. Defaults to torch.bfloat16.
        include_spl_values (bool, optional): If True, overwrite 5 values with
            ±0, ±inf, and one NaN. Defaults to False.
        include_zero (bool, optional): If True and include_spl_values is False,
            overwrite one value with +0. Defaults to False.

    Returns:
        torch.Tensor: 1D tensor of length 256, all unique bit patterns.
    """
    bits = []
    for sign in (0, 1):
        for exponent in _bf16_ternary_grid_exponents():
            for mantissa in BF16_BINARY_GRID_MANTISSAS:
                bits.append((sign << 15) | ((exponent + 127) << 7) | mantissa)

    if include_spl_values:
        specials = _BF16_SPECIAL_BITS
    elif include_zero:
        specials = (0x0000,)
    else:
        specials = ()

    for index, special in zip(_bf16_ternary_sacrificial_indices(bits, len(specials)), specials):
        bits[index] = special

    assert len(bits) == BF16_TERNARY_GRID_SIZE
    assert len(set(bits)) == BF16_TERNARY_GRID_SIZE

    return torch.tensor(bits, dtype=torch.uint16).view(torch.bfloat16).to(dtype)


def triple_from_values(values_a, values_b=None, values_c=None):
    """Outer product of three value sets: A[i, j, k] = values_a[i], and so on."""
    if values_b is None:
        values_b = values_a
    if values_c is None:
        values_c = values_b
    a, b, c = torch.meshgrid(values_a, values_b, values_c, indexing="ij")
    return a.contiguous(), b.contiguous(), c.contiguous()


def ternary_inputs(include_spl_values=False, include_zero=False, dtype=torch.bfloat16):
    """Outer product of the 256-value ternary grid, shaped [256, 256, 256]."""
    values = generate_bfloat16_ternary_grid(
        dtype=dtype, include_spl_values=include_spl_values, include_zero=include_zero
    )
    return triple_from_values(values)


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


def run_binary(device, ttnn_op, input_a, input_b, *, golden_kwargs=None, **op_kwargs):
    """Run a tensor-tensor ttnn binary op and return ``(golden, result)``.

    Ops whose ``__name__`` ends in ``_`` (``ttnn.add_``, ``ttnn.multiply_``, …)
    are inplace: they write into lhs and the result is read back from that
    tensor rather than the return value.

    Inputs are uploaded as ``ttnn.bfloat16``. ``op_kwargs`` (e.g.
    ``fast_and_approximate_mode``) are forwarded only to the device op.
    Torch-valid golden arguments (e.g. isclose ``rtol``/``atol``) go in
    ``golden_kwargs`` and are also passed to the device op.
    """
    tt_a = to_tt_tensor(input_a, device)
    tt_b = to_tt_tensor(input_b, device)

    golden_kwargs = golden_kwargs or {}
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_a, input_b, **golden_kwargs)

    device_kwargs = {**golden_kwargs, **op_kwargs}
    if ttnn_op.__name__.endswith("_"):
        ttnn_op(tt_a, tt_b, **device_kwargs)
        result = ttnn.to_torch(tt_a)
    else:
        result = ttnn.to_torch(ttnn_op(tt_a, tt_b, **device_kwargs))
    return golden, result


def run_ternary(device, ttnn_op, input_a, input_b, input_c, *, golden_kwargs=None, **op_kwargs):
    """Run a tensor-tensor-tensor ttnn ternary op and return ``(golden, result)``.

    The 3-operand counterpart of ``run_binary`` — same inplace convention, same
    split between ``golden_kwargs`` (passed to both torch and the device, e.g.
    ``addcmul`` ``value``) and ``op_kwargs`` (device only).
    """
    tt_a = to_tt_tensor(input_a, device)
    tt_b = to_tt_tensor(input_b, device)
    tt_c = to_tt_tensor(input_c, device)

    golden_kwargs = golden_kwargs or {}
    golden_function = ttnn.get_golden_function(ttnn_op)
    golden = golden_function(input_a, input_b, input_c, **golden_kwargs)

    device_kwargs = {**golden_kwargs, **op_kwargs}
    if ttnn_op.__name__.endswith("_"):
        ttnn_op(tt_a, tt_b, tt_c, **device_kwargs)
        result = ttnn.to_torch(tt_a)
    else:
        result = ttnn.to_torch(ttnn_op(tt_a, tt_b, tt_c, **device_kwargs))
    return golden, result


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


def sech2_exact(x: float) -> float:
    """
    Exact tanh derivative using mpmath 256-bit precision.

    tanh'(x) = sech²(x) = 1 / cosh²(x)

    Uses 1/cosh²(x) form (not 1 - tanh²(x)) to avoid the catastrophic cancellation
    in the latter. Shared golden for test_tanh_bw_ulp.py and test_tanh_bw_fp32_ulp.py,
    which apply their own input rounding and flushing around it.
    """
    mp.prec = 256
    cosh_x = mp_cosh(mp.mpf(x))
    return float(1 / (cosh_x * cosh_x))
