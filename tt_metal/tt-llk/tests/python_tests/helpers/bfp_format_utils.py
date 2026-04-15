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

BFP_BLOCK = 16


def _bf16_sign_exp_mantissa(flat: torch.Tensor):
    """Extract sign, exponent, and implicit 7-bit mantissa from BF16 (as uint16)."""
    bf16_bits = (flat.view(torch.int32) >> 16) & 0xFFFF
    signs = (bf16_bits >> 15) & 1
    exps = (bf16_bits >> 7) & 0xFF
    mants = ((bf16_bits & 0x7F) >> 1) | 0x40
    return signs, exps, mants


def _bfp8_mantissas_per_block(mants_blocks: torch.Tensor, exps_blocks: torch.Tensor):
    """
    Round to shared block exponent per ISA BFP8 packing: round-to-nearest,
    ties away from zero; guard bit when delta > 0.
    """
    shared_exps = exps_blocks.max(dim=1, keepdim=True).values
    deltas = shared_exps - exps_blocks
    guard = torch.where(
        deltas > 0, (mants_blocks >> (deltas - 1)) & 1, torch.zeros_like(mants_blocks)
    )
    bfp8 = ((mants_blocks >> deltas) + guard) & 0x7F
    return shared_exps, bfp8


def _finalize_bfp_quantized(
    values: torch.Tensor,
    signs_blocks: torch.Tensor,
    n: int,
    dimensions,
) -> torch.Tensor:
    values = torch.where(signs_blocks.bool(), -values, values)
    # Hardware runs with FTZ: flush near-zero values that arise from BFP
    # scale arithmetic with very small shared exponents (shared_exp <= 1).
    # The smallest meaningful BFP value has shared_exp=2, giving ~2.35e-38.
    FTZ_THRESHOLD = 1e-37
    values = torch.where(values.abs() < FTZ_THRESHOLD, torch.zeros_like(values), values)
    out = values.flatten()[:n].to(torch.bfloat16)
    if dimensions is not None:
        out = untilize_block(
            out,
            stimuli_format=DataFormat.Float16_b,
            dimensions=dimensions,
        ).flatten()
    return out


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
    flat = operand.flatten().to(torch.float32)
    n = flat.numel()
    signs, exps, mants = _bf16_sign_exp_mantissa(flat)

    signs_blocks = signs.view(-1, BFP_BLOCK)
    shared_exps, bfp8_mants = _bfp8_mantissas_per_block(
        mants.view(-1, BFP_BLOCK),
        exps.view(-1, BFP_BLOCK),
    )

    values = bfp8_mants.float() * torch.exp2((shared_exps - 133).float())
    return _finalize_bfp_quantized(values, signs_blocks, n, dimensions)


def bfp4b_to_float16b(operand: torch.Tensor, dimensions=None) -> torch.Tensor:
    """
    Simulate BFP4_b pack/unpack round-trip quantization.

    Per the ISA spec, BFP4 packing is a two-stage process:
      1. Convert to BFP8 with rounding (round-to-nearest, ties away from zero)
      2. Truncate BFP8 mantissa (7 bits) down to BFP4 mantissa (3 bits)

    Args:
        operand: Input tensor (any shape).
        dimensions: If provided, untilize the result back to these dimensions.

    Returns:
        Quantized bfloat16 tensor (same number of elements as input).
    """
    flat = operand.flatten().to(torch.float32)
    n = flat.numel()
    signs, exps, mants = _bf16_sign_exp_mantissa(flat)

    signs_blocks = signs.view(-1, BFP_BLOCK)
    shared_exps, bfp8_mants = _bfp8_mantissas_per_block(
        mants.view(-1, BFP_BLOCK),
        exps.view(-1, BFP_BLOCK),
    )

    # Stage 2: BFP8 -> BFP4 by truncating (drop the 4 LSBs of the 7-bit mantissa)
    bfp4_mants = bfp8_mants >> 4
    # Scale: 2^(shared_exp - 127 - 2) = 2^(shared_exp - 129)
    # BFP4 mantissa is 3 bits with implicit leading 1 worth 4 (0x4),
    # so divide by 4 -> exponent offset is 129
    values = bfp4_mants.float() * torch.exp2((shared_exps - 129).float())
    return _finalize_bfp_quantized(values, signs_blocks, n, dimensions)
