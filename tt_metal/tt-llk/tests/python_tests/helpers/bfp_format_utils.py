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
    mants = ((bf16_bits & 0x7F) >> 1) | 0x40

    exps_blocks = exps.view(-1, BFP8_BLOCK)
    mants_blocks = mants.view(-1, BFP8_BLOCK)
    signs_blocks = signs.view(-1, BFP8_BLOCK)

    shared_exps = exps_blocks.max(dim=1, keepdim=True).values

    deltas = shared_exps - exps_blocks
    shifted = mants_blocks >> deltas

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
    # Hardware takes bits 23:21 of the 24-bit mantissa (with implicit leading 1).
    mants = ((u32 & 0x7FFFFF) >> 21) | 0x4

    exps_blocks = exps.view(-1, BFP4_BLOCK)
    mants_blocks = mants.view(-1, BFP4_BLOCK)
    signs_blocks = signs.view(-1, BFP4_BLOCK)

    shared_exps = exps_blocks.max(dim=1, keepdim=True).values

    deltas = shared_exps - exps_blocks
    shifted = mants_blocks >> deltas

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
