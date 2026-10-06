# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the MX decoders' NaN-block and range rules in ``helpers/unpack.py``.

No kernel, no device. Two register-level rules the generated stimuli never reach:

- A 0xFF block scale is the reserved NaN scale and NaNs every datum of the block, zeros
  included, for every MX format.
- The gasket lands every MX-float datum in an 8-bit-exponent register without subnormals:
  a decoded value saturates to +-Inf at unbiased exponent >= 128 and flushes to a signed
  zero at <= -127. The exponent is the decoded value's own, so a subnormal element is
  normalised first rather than summed field by field, which used to put it too high.

Inputs are raw packed bytes and every expected value is written out here, so nothing is
obtained from the packer.
"""

import math

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.unpack import (
    unpack_mxfp4,
    unpack_mxfp8p,
    unpack_mxfp8r,
    unpack_mxint2,
    unpack_mxint4,
    unpack_mxint8,
)

# One 32-datum block: 1 face of 2 rows x 16 columns.
UNPACK_GEOMETRY = dict(num_faces=1, face_r_dim=2)
BLOCK_SIZE = 32
SCALE_SECTION = 16  # one scale byte, padded to the 16-byte L1 alignment
NAN_SCALE = 0xFF
UNIT_SCALE = 127  # 2^0

# unpacker, payload bytes per block, sign bit of an element encoding
FORMATS = {
    DataFormat.MxFp8P: (unpack_mxfp8p, 32, 0x80),
    DataFormat.MxFp8R: (unpack_mxfp8r, 32, 0x80),
    DataFormat.MxFp4: (unpack_mxfp4, 16, 0x8),
    DataFormat.MxInt8: (unpack_mxint8, 32, None),
    DataFormat.MxInt4: (unpack_mxint4, 16, None),
    DataFormat.MxInt2: (unpack_mxint2, 8, None),
}


def _decode(fmt, scale, payload):
    """Decode one block from its scale byte and raw payload bytes."""
    unpacker, payload_len, _ = FORMATS[fmt]
    assert len(payload) == payload_len
    padded = payload + [0] * (-len(payload) % 16)
    packed = [scale] + [0] * (SCALE_SECTION - 1) + padded
    return unpacker(packed, **UNPACK_GEOMETRY).float()


def _decode_first(fmt, scale, element):
    """Decode a block whose element 0 has encoding `element` and the rest are zero.

    For MXFP4 element 0 is the low nibble of the first payload byte.
    """
    _, payload_len, _ = FORMATS[fmt]
    payload = [element] + [0] * (payload_len - 1)
    return _decode(fmt, scale, payload)[0].item()


# ---------------------------------------------------------------------------
# 0xFF block scale


@pytest.mark.parametrize(
    "payload_byte", [0x00, 0x11, 0xFF], ids=["zero", "0x11", "0xFF"]
)
@pytest.mark.parametrize("fmt", list(FORMATS), ids=lambda f: f.name)
def test_a_nan_scale_nans_the_whole_block(fmt, payload_byte):
    _, payload_len, _ = FORMATS[fmt]
    decoded = _decode(fmt, NAN_SCALE, [payload_byte] * payload_len)
    assert torch.isnan(decoded).all()


@pytest.mark.parametrize("fmt", list(FORMATS), ids=lambda f: f.name)
def test_the_same_payload_under_a_unit_scale_is_not_nan(fmt):
    """Control: with any other scale, 0x11 decodes to finite non-NaN datums."""
    _, payload_len, _ = FORMATS[fmt]
    decoded = _decode(fmt, UNIT_SCALE, [0x11] * payload_len)
    assert torch.isfinite(decoded).all()
    assert (decoded != 0).any()


# ---------------------------------------------------------------------------
# Range clamp at unbiased exponents 127/128 and -126/-127
#
# (format, scale byte, positive element encoding, decoded positive value).
# Element encodings: E4M3 1.0=0x38 2.0=0x40 448=0x7E min-subnormal 2^-9=0x01
# 1.75*2^-7=0x07; E5M2 1.0=0x3C 2.0=0x40 57344=0x7B min-subnormal 2^-16=0x01;
# E2M1 1.0=0x2 2.0=0x4 6.0=0x7 subnormal 0.5=0x1.
RANGE_CASES = [
    # Top: 2^127 holds, 2^128 is Inf.
    (DataFormat.MxFp8P, 0xFE, 0x38, 2.0**127),
    (DataFormat.MxFp8P, 0xFE, 0x40, math.inf),
    (DataFormat.MxFp8P, 246, 0x7E, 1.75 * 2.0**127),
    (DataFormat.MxFp8P, 247, 0x7E, math.inf),
    (DataFormat.MxFp8R, 0xFE, 0x3C, 2.0**127),
    (DataFormat.MxFp8R, 0xFE, 0x40, math.inf),
    (DataFormat.MxFp8R, 239, 0x7B, 1.75 * 2.0**127),
    (DataFormat.MxFp8R, 240, 0x7B, math.inf),
    (DataFormat.MxFp4, 0xFE, 0x2, 2.0**127),
    (DataFormat.MxFp4, 0xFE, 0x4, math.inf),
    (DataFormat.MxFp4, 252, 0x7, 1.5 * 2.0**127),
    (DataFormat.MxFp4, 253, 0x7, math.inf),
    # Bottom, normal elements: 2^-126 holds, 2^-127 flushes.
    (DataFormat.MxFp8P, 0x00, 0x40, 2.0**-126),
    (DataFormat.MxFp8P, 0x00, 0x38, 0.0),
    (DataFormat.MxFp8R, 0x00, 0x40, 2.0**-126),
    (DataFormat.MxFp8R, 0x00, 0x3C, 0.0),
    (DataFormat.MxFp4, 0x00, 0x4, 2.0**-126),
    (DataFormat.MxFp4, 0x00, 0x2, 0.0),
    # Bottom, subnormal elements. Summing the raw fields would read each
    # flushed case as finite: E4M3 and E5M2 by 5 and 14 binades, E2M1 by one.
    (DataFormat.MxFp8P, 10, 0x01, 2.0**-126),
    (DataFormat.MxFp8P, 9, 0x01, 0.0),
    (DataFormat.MxFp8P, 8, 0x07, 1.75 * 2.0**-126),
    (DataFormat.MxFp8P, 7, 0x07, 0.0),
    (DataFormat.MxFp8R, 17, 0x01, 2.0**-126),
    (DataFormat.MxFp8R, 16, 0x01, 0.0),
    (DataFormat.MxFp4, 2, 0x1, 2.0**-126),
    (DataFormat.MxFp4, 1, 0x1, 0.0),
]


def _case_id(case):
    fmt, scale, element, value = case
    return f"{fmt.name}-s{scale}-e{element:#04x}-{value:g}"


@pytest.mark.parametrize("negative", [False, True], ids=["pos", "neg"])
@pytest.mark.parametrize("case", RANGE_CASES, ids=[_case_id(c) for c in RANGE_CASES])
def test_the_decoded_value_is_clamped_to_the_register_range(case, negative):
    fmt, scale, element, value = case
    _, _, sign_bit = FORMATS[fmt]
    if negative:
        element |= sign_bit
        value = -value

    decoded = _decode_first(fmt, scale, element)

    if math.isinf(value):
        assert decoded == value
    elif value == 0.0:
        # A flush keeps the element's sign.
        assert decoded == 0.0
        assert math.copysign(1.0, decoded) == (-1.0 if negative else 1.0)
    else:
        assert decoded == value
