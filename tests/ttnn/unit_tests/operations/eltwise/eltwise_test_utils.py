# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

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
