# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""bf16 round-to-nearest-even tie geometry, shared by the exact-tie tests.

test_eltwise_binary_sfpu, test_sfpu_binop_scalar and test_sfpu_ternary each build stimuli
whose exact fp32 result lies halfway between two bf16 neighbours, then pin the kernel's
narrowing bit for bit against torch. The constants and the host-side tie check live here
once, so the three builders cannot drift apart on what "a tie" means.

bf16 keeps 7 explicit fraction bits, so an integer significand runs over
[BF16_SIG_ONE, 2 * BF16_SIG_ONE), and an fp32 value is a bf16 tie exactly when the 16 bits
the narrowing drops are the half-ULP pattern 0x8000.
"""

import struct

BF16_FRAC_BITS = 7
BF16_SIG_ONE = 1 << BF16_FRAC_BITS
FP32_LOW_HALF_MASK = 0xFFFF
FP32_TIE_LOW_HALF = 0x8000


def fp32_bits(value: float) -> int:
    """The IEEE 754 single-precision bit pattern of *value*."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


def is_bf16_tie(value: float) -> bool:
    """True when *value*, as fp32, lies exactly halfway between two bf16 neighbours."""
    return fp32_bits(value) & FP32_LOW_HALF_MASK == FP32_TIE_LOW_HALF


def assert_is_bf16_tie(value: float, label: str) -> None:
    """Host self-check for a tie builder: a lane that is not a tie proves nothing."""
    assert is_bf16_tie(value), f"{label} is not a bf16 tie"
