# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the MX block scale at binade boundaries in ``helpers/pack.py``.

No kernel, no device. The packer derives each block's E8M0 scale from floor(log2(amax)),
and taking that floor with ``np.floor(np.log2())`` is wrong for the float32 just below a
power of two: ``log2`` rounds up to the exact integer and the whole block lands one scale
step too high. Generated stimuli essentially never hit those values, so the device sweep
would keep passing if the bug came back. These tests hit them on purpose.

Every expected byte is written out here rather than computed by the packer's own helpers:
the scale from a literal per-format max-exponent table, the payload from a literal
saturated-element table.
"""

import numpy as np
import pytest
import torch
from helpers.format_config import DataFormat
from helpers.pack import (
    pack_mxfp4,
    pack_mxfp8p,
    pack_mxfp8r,
    pack_mxint2,
    pack_mxint4,
    pack_mxint8,
)
from helpers.utils import floor_log2

# One 32-datum block: 1 face of 2 rows x 16 columns.
PACK_GEOMETRY = dict(num_faces=1, face_r_dim=2)
BLOCK_SIZE = 32
# The single scale byte is padded to the 16-byte L1 alignment, so elements start here.
FIRST_ELEMENT_BYTE = 16

# packer, element format's unbiased max exponent, mask selecting element 0 in its byte
FORMATS = {
    DataFormat.MxInt8: (pack_mxint8, 0, 0xFF),
    DataFormat.MxInt4: (pack_mxint4, 0, 0x0F),
    DataFormat.MxInt2: (pack_mxint2, 0, 0x03),
    DataFormat.MxFp4: (pack_mxfp4, 2, 0x0F),  # E2M1, max normal 6.0
    DataFormat.MxFp8P: (pack_mxfp8p, 8, 0xFF),  # E4M3, max normal 448
    DataFormat.MxFp8R: (pack_mxfp8r, 15, 0xFF),  # E5M2, max normal 57344
}

# Encoding of a datum with significand just below 2.0 at the block's own binade, i.e.
# above every format's max representable value, so it saturates: (positive, negative).
# With a scale one step too high it would instead land near half range.
SATURATED_ELEMENT = {
    DataFormat.MxInt8: (0x7F, 0x81),  # +-127, 2's complement
    DataFormat.MxInt4: (0x7, 0x9),  # +-7
    DataFormat.MxInt2: (0x1, 0x3),  # +-1
    DataFormat.MxFp4: (0x7, 0xF),  # +-6.0
    DataFormat.MxFp8P: (0x7E, 0xFE),  # +-448
    DataFormat.MxFp8R: (0x7B, 0xFB),  # +-57344
}

# Kept inside the range where no format's scale clamps at E8M0 0 or 254.
EXPONENTS = [-100, -64, -17, -3, -1, 0, 1, 2, 7, 31, 64, 100]


def _pack_block(fmt, amax):
    """Pack one block holding `amax` at index 0 and zeros elsewhere."""
    block = torch.zeros(BLOCK_SIZE, dtype=torch.float32)
    block[0] = float(amax)
    packer, _, _ = FORMATS[fmt]
    return packer(block, **PACK_GEOMETRY)


def _expected_scale(fmt, floor_log2_amax):
    _, elem_exp_max, _ = FORMATS[fmt]
    return floor_log2_amax - elem_exp_max + 127


@pytest.mark.parametrize("sign", [1, -1], ids=["pos", "neg"])
@pytest.mark.parametrize("k", EXPONENTS)
@pytest.mark.parametrize("fmt", list(FORMATS), ids=lambda f: f.name)
def test_float32_just_below_a_power_of_two_takes_the_lower_binade(fmt, k, sign):
    """amax = the float32 immediately below 2^k has floor(log2) = k - 1, not k."""
    amax = sign * np.nextafter(np.float32(2.0**k), np.float32(0.0))
    packed = _pack_block(fmt, amax)
    assert packed[0] == _expected_scale(fmt, k - 1)


@pytest.mark.parametrize("sign", [1, -1], ids=["pos", "neg"])
@pytest.mark.parametrize("k", EXPONENTS)
@pytest.mark.parametrize("fmt", list(FORMATS), ids=lambda f: f.name)
def test_an_exact_power_of_two_takes_its_own_binade(fmt, k, sign):
    """Control for the test above: one ULP up, the scale is one step higher."""
    packed = _pack_block(fmt, sign * np.float32(2.0**k))
    assert packed[0] == _expected_scale(fmt, k)


@pytest.mark.parametrize("sign", [1, -1], ids=["pos", "neg"])
@pytest.mark.parametrize("k", [-17, 0, 31])
@pytest.mark.parametrize("fmt", list(FORMATS), ids=lambda f: f.name)
def test_the_largest_element_saturates_at_the_lower_binade(fmt, k, sign):
    """The payload follows the scale: at the right binade amax saturates the element."""
    amax = sign * np.nextafter(np.float32(2.0**k), np.float32(0.0))
    packed = _pack_block(fmt, amax)
    _, _, mask = FORMATS[fmt]
    positive, negative = SATURATED_ELEMENT[fmt]
    assert packed[FIRST_ELEMENT_BYTE] & mask == (positive if sign > 0 else negative)


def test_mxint8_just_below_one_eighth():
    """The worked example from the fix, every byte spelled out."""
    amax = np.float32(0.12499999)
    assert amax == np.nextafter(np.float32(0.125), np.float32(0.0))

    block = torch.zeros(BLOCK_SIZE, dtype=torch.float32)
    block[0] = float(amax)
    block[1] = -0.0625
    packed = pack_mxint8(block, **PACK_GEOMETRY)

    # Shared exponent -4: 0x7B. The floor(log2()) bug gave -3: 0x7C.
    assert packed[0] == 0x7B
    # 0.12499999 / 2^-4 * 64 = 127.99999 -> rounds to 128 -> saturates to 127.
    # The bug's 0x7C scale would have given 64 (0x40).
    assert packed[FIRST_ELEMENT_BYTE] == 0x7F
    # -0.0625 / 2^-4 * 64 = -64.
    assert packed[FIRST_ELEMENT_BYTE + 1] == 0xC0
    assert packed[FIRST_ELEMENT_BYTE + 2 : FIRST_ELEMENT_BYTE + BLOCK_SIZE] == [0] * 30


# ---------------------------------------------------------------------------
# Special blocks: NaN, Inf and all-zero take reserved or derived scales. These
# bytes changed when MXFP8 moved onto the shared derivation, and nothing pinned
# them -- the device sweep cannot, because --compile-producer never reaches the
# packer and the compile-consumer sweep never generates a whole block of NaN.
# ---------------------------------------------------------------------------

E8M0_NAN = 0xFF
E8M0_INF = 0xFE


def _pack_values(fmt, values):
    """Pack one block holding `values` verbatim."""
    packer, _, _ = FORMATS[fmt]
    return packer(torch.tensor(values, dtype=torch.float32), **PACK_GEOMETRY)


@pytest.mark.parametrize("fmt", list(FORMATS), ids=lambda f: f.name)
def test_an_all_nan_block_takes_the_reserved_nan_scale(fmt):
    """Every datum NaN is the one case the 0xFF scale encodes."""
    assert _pack_values(fmt, [float("nan")] * BLOCK_SIZE)[0] == E8M0_NAN


@pytest.mark.parametrize(
    "values",
    [[float("inf")] * BLOCK_SIZE, [float("inf")] + [0.0] * (BLOCK_SIZE - 1)],
    ids=["all_inf", "inf_and_zeros"],
)
@pytest.mark.parametrize("fmt", list(FORMATS), ids=lambda f: f.name)
def test_a_block_of_only_inf_and_zero_takes_the_inf_scale(fmt, values):
    """0xFE is reserved for a block that holds an Inf and nothing finite."""
    assert _pack_values(fmt, values)[0] == E8M0_INF


@pytest.mark.parametrize("fmt", list(FORMATS), ids=lambda f: f.name)
def test_non_finite_datums_do_not_drive_the_scale(fmt):
    """A NaN or Inf alongside finite data is excluded from the max-exponent tree,
    so the scale follows the finite amax -- here 3.0, floor(log2) = 1."""
    values = [float("nan"), float("inf")] + [3.0] * (BLOCK_SIZE - 2)
    assert _pack_values(fmt, values)[0] == _expected_scale(fmt, 1)


@pytest.mark.parametrize("fmt", list(FORMATS), ids=lambda f: f.name)
def test_an_all_zero_block_takes_the_zero_exponent_scale(fmt):
    """amax of 0 gives shared exponent 0, not the reserved codes and not a
    neutral 127 -- the element format's max exponent is still subtracted."""
    assert _pack_values(fmt, [0.0] * BLOCK_SIZE)[0] == _expected_scale(fmt, 0)


# ---------------------------------------------------------------------------
# The comparator derives the same block scale, and has to agree with the packer
# or it judges a block against the wrong tolerance.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", EXPONENTS)
def test_the_comparator_takes_the_same_binade_just_below_a_power_of_two(k):
    """floor_log2 is what both sides use; np.log2 rounds up here and floor(log2)
    would return k, putting the comparator a binade above the packer."""
    amax = torch.tensor([float(np.nextafter(np.float32(2.0**k), np.float32(0.0)))])
    assert int(floor_log2(amax).item()) == k - 1


@pytest.mark.parametrize("k", EXPONENTS)
def test_the_comparator_takes_its_own_binade_mid_range(k):
    """Control: a value well inside a binade is unambiguous either way."""
    amax = torch.tensor([float(np.float32(1.9) * np.float32(2.0**k))])
    assert int(floor_log2(amax).item()) == k
