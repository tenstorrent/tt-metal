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
    signs = bf16_bits >> 15
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
    # Clamp the shift to avoid evaluating `mants_blocks >> -1` when delta == 0;
    # torch.where selects element-wise but both branches are evaluated eagerly.
    guard_shifts = torch.clamp(deltas - 1, min=0)
    guard_bits = (mants_blocks >> guard_shifts) & 1
    guard = torch.where(deltas > 0, guard_bits, torch.zeros_like(mants_blocks))
    bfp8 = ((mants_blocks >> deltas) + guard) & 0x7F
    return shared_exps, bfp8


def _finalize_bfp_quantized(
    values: torch.Tensor,
    signs_blocks: torch.Tensor,
    n: int,
    dimensions,
) -> torch.Tensor:
    # Note: hardware FTZ (flush of near-zero values that arise from BFP
    # scale arithmetic with very small shared exponents) is applied once,
    # centrally, at the end of the consuming golden's __call__ — so the
    # same FTZ pass covers FP outputs as well. Do not FTZ here.
    values = torch.where(signs_blocks.bool(), -values, values)
    out = values.flatten()[:n].to(torch.bfloat16)
    if dimensions is not None:
        out = untilize_block(
            out,
            stimuli_format=DataFormat.Float16_b,
            dimensions=dimensions,
        ).flatten()
    return out


def _quantize_bfp_b(
    operand: torch.Tensor, magnitude_bits: int, dimensions
) -> torch.Tensor:
    """Simulate BFP{8,4,2}_b pack/unpack round-trip quantization.

    Per the ISA spec the hardware packer always rounds to BFP8 first
    (round-to-nearest, ties away from zero; one shared 8-bit exponent per
    16 datums) and then truncates the 7-bit BFP8 magnitude down to
    ``magnitude_bits`` for the narrower formats (3 bits for BFP4, 1 bit
    for BFP2; 7 bits is the identity case for BFP8).

    After truncation the surviving leading 1 bit sits at position
    ``magnitude_bits - 1`` of the result, i.e. it is worth
    ``2^(magnitude_bits - 1)``. Combined with the standard FP32 bias of
    127 for the shared exponent, the dequantized magnitude is therefore
    ``trunc_mant * 2^(shared_exp - 126 - magnitude_bits)``.

    Args:
        operand: Input tensor (any shape).
        magnitude_bits: BFP magnitude width (7 for BFP8, 3 for BFP4, 1 for BFP2).
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

    trunc_mants = bfp8_mants >> (7 - magnitude_bits)
    exp_offset = 126 + magnitude_bits
    values = trunc_mants.float() * torch.exp2((shared_exps - exp_offset).float())
    return _finalize_bfp_quantized(values, signs_blocks, n, dimensions)


def bfp8b_to_float16b(operand: torch.Tensor, dimensions=None) -> torch.Tensor:
    """Simulate BFP8_b pack/unpack round-trip quantization.

    See :func:`_quantize_bfp_b` for the shared implementation. BFP8 keeps
    all 7 BFP8 magnitude bits and applies an exponent offset of 133.
    """
    return _quantize_bfp_b(operand, magnitude_bits=7, dimensions=dimensions)


def bfp4b_to_float16b(operand: torch.Tensor, dimensions=None) -> torch.Tensor:
    """Simulate BFP4_b pack/unpack round-trip quantization.

    Per the ISA spec, BFP4 packing is a two-stage process:
      1. Convert to BFP8 with rounding (round-to-nearest, ties away from zero)
      2. Truncate BFP8 mantissa (7 bits) down to BFP4 mantissa (3 bits)
    """
    return _quantize_bfp_b(operand, magnitude_bits=3, dimensions=dimensions)


def bfp2b_to_float16b(operand: torch.Tensor, dimensions=None) -> torch.Tensor:
    """Simulate BFP2_b pack/unpack round-trip quantization.

    Per the ISA spec, BFP2 packing is a two-stage process:
      1. Convert to BFP8 with rounding (round-to-nearest, ties away from zero)
      2. Truncate BFP8 mantissa (7 bits) down to BFP2 mantissa (1 bit)

    Representable per-element output values per block::

        { 0, +1 * 2^(shared_exp - 127), -1 * 2^(shared_exp - 127) }

    An element survives with magnitude 1 whenever the top bit of its 7-bit
    BFP8 magnitude (after round-to-nearest alignment to the shared exponent)
    is set; otherwise it collapses to 0.
    """
    return _quantize_bfp_b(operand, magnitude_bits=1, dimensions=dimensions)
