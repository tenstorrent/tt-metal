# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Block floating-point quantization utilities.

These functions simulate the round-trip precision loss that occurs when the
hardware packs values into BFP format and then unpacks them back.  Within
each 16-element block the shared exponent is the maximum element exponent,
so elements with smaller exponents lose mantissa bits via right-shifting.
"""

import torch

from .format_config import DataFormat
from .tilize_untilize import untilize_block


def bfp8b_to_float16b(operand: torch.Tensor, dimensions=None) -> torch.Tensor:
    """
    Simulate BFP8_b pack/unpack round-trip quantization.

    Processes values in blocks of 16, determines a shared exponent per block
    (the maximum element exponent), and right-shifts each element's mantissa
    by the delta between its exponent and the shared exponent, matching the
    hardware packer/unpacker behaviour.

    Args:
        operand: Input tensor (any shape).
        dimensions: If provided, untilize the result back to these dimensions.

    Returns:
        Quantized bfloat16 tensor (same number of elements as input).
    """
    BFP8_BLOCK = 16
    flat = operand.flatten().to(torch.float32)
    n = flat.numel()

    u32 = flat.view(torch.int32)
    bf16_bits = (u32 >> 16) & 0xFFFF

    signs = (bf16_bits >> 15) & 1
    exps = (bf16_bits >> 7) & 0xFF
    # Keep the FULL 8-bit normalized mantissa (implicit 1 + 7 explicit, range
    # 128..255) so we can apply the hw round-to-nearest rule below. The old
    # code pre-dropped the bf16 LSB via `(bf16_bits & 0x7F) >> 1 | 0x40` and
    # then did pure `>> delta` truncation, mismatching the hw packer's
    # `(m + (1 << shift)) >> (shift + 1)` rounding from ttsim PACR
    # (src/tensix.cpp:3032). Sibling fix to the bfp4b_to_float16b patch.
    mants_full = (bf16_bits & 0x7F) | 0x80

    exps_blocks = exps.view(-1, BFP8_BLOCK)
    mants_blocks_full = mants_full.view(-1, BFP8_BLOCK)
    signs_blocks = signs.view(-1, BFP8_BLOCK)

    shared_exps = exps_blocks.max(dim=1, keepdim=True).values

    # Per ttsim PACR for dst_element_size_bits=8: shift = delta + 0, then
    # m = (m + (1<<shift)) >> (shift+1) — i.e. always rounds at least 1 LSB.
    # Cap deltas at 16 so the shift remains in int32 range; values whose
    # mantissa shifts out entirely (delta > ~8) round to 0 naturally.
    deltas = shared_exps - exps_blocks
    deltas_capped = torch.clamp(deltas, max=16)
    rounding = 1 << deltas_capped
    shifted = (mants_blocks_full + rounding) >> (deltas_capped + 1)
    # Saturate at max_m - 1 = 127 per hw rule (ttsim PACR src/tensix.cpp:3035).
    shifted = torch.clamp(shifted, max=127)

    values = shifted.float() * torch.exp2((shared_exps - 133).float())
    values = torch.where(signs_blocks.bool(), -values, values)

    quantized = values.flatten()[:n].to(torch.bfloat16)

    if dimensions is not None:
        quantized = untilize_block(
            quantized,
            stimuli_format=DataFormat.Float16_b,
            dimensions=dimensions,
        ).flatten()

    return quantized


def bfp4b_to_float16b(operand: torch.Tensor, dimensions=None) -> torch.Tensor:
    """
    Simulate BFP4_b pack/unpack round-trip quantization.

    Processes values in blocks of 16, determines a shared exponent per block
    (the maximum element exponent), and right-shifts each element's 3-bit
    mantissa (1 implicit + 2 explicit bits) by the per-element delta, matching
    the hardware packer/unpacker behaviour.

    Args:
        operand: Input tensor (any shape).
        dimensions: If provided, untilize the result back to these dimensions.

    Returns:
        Quantized bfloat16 tensor (same number of elements as input).
    """
    BFP4_BLOCK = 16
    flat = operand.flatten().to(torch.float32)
    n = flat.numel()

    u32 = flat.view(torch.int32)

    signs = (u32 >> 31) & 1
    exps = (u32 >> 23) & 0xFF
    # bfp4_b has 3 mantissa bits: 1 implicit leading bit + 2 explicit bits.
    # Use the FULL 24-bit mantissa (implicit 1 prepended) so we can round-to-
    # nearest when we shift it down — matching the hw packer (ttsim PACR at
    # src/tensix.cpp:3032: `m = (m + (1 << shift)) >> (shift + 1)`). The
    # previous truncation drove 1-ULP divergences for any element whose top
    # discarded mantissa bit was 1.
    mants_full = (u32 & 0x7FFFFF) | (1 << 23)  # 24-bit, includes implicit leading 1

    exps_blocks = exps.view(-1, BFP4_BLOCK)
    mants_blocks_full = mants_full.view(-1, BFP4_BLOCK)
    signs_blocks = signs.view(-1, BFP4_BLOCK)

    shared_exps = exps_blocks.max(dim=1, keepdim=True).values

    deltas = shared_exps - exps_blocks
    # Total right-shift to reach the 3-bit BFP4 mantissa is 21 + delta:
    # 21 bits to strip from the fp32 explicit mantissa, plus delta bits to align
    # to the block's shared exponent.
    total_shift = 21 + deltas
    # Round-to-nearest: add 1 << (total_shift - 1) before shifting right by
    # total_shift. Guard total_shift == 0 (no discarded bits → no rounding).
    rounding = torch.where(
        total_shift > 0,
        1 << torch.clamp(total_shift - 1, min=0),
        torch.zeros_like(total_shift),
    )
    shifted = (mants_blocks_full + rounding) >> total_shift
    # Saturate at max_m = 8 (1<<(bfp4_mantissa_bits)) per hw rule
    # (ttsim src/tensix.cpp:3035 `if (m == max_m) m = max_m - 1`).
    shifted = torch.clamp(shifted, max=7)

    # Scale: 2^(shared_exp - 127 - 2) = 2^(shared_exp - 129)
    values = shifted.float() * torch.exp2((shared_exps - 129).float())
    values = torch.where(signs_blocks.bool(), -values, values)

    quantized = values.flatten()[:n].to(torch.bfloat16)

    if dimensions is not None:
        quantized = untilize_block(
            quantized,
            stimuli_format=DataFormat.Float16_b,
            dimensions=dimensions,
        ).flatten()

    return quantized
