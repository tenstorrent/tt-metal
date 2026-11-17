# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host tests of the reference MX codec (mx_reference_codec.py), the oracle of the Quasar MX tests.

The truth for each conversion is tt-regen-spec section 4.2 and the MX block of section 4.3, then the Tensix Formats
spec, then the OCP MX spec. These tests take no device, so they run on every arch.

- test_encode_rules, test_decode_rules: rows worked by hand from the text of 4.2 (_ENCODE_ROWS, _DECODE_ROWS).
- test_block_shape: a block is a 2x16 patch of one face, and a tile holds its 32 scale bytes first.
- test_agree_encode, test_agree_decode: the reference and the ttnn host codec agree on every element code at each
  scale of _SCALES, and on the hand-worked blocks, except where 4.2 and the ttnn codec state different rules.
- test_ttnn_codec_differs: one strict expected failure per such rule. It asserts that the two codecs differ on each
  element that the rule predicts, then raises _WrongAsPredicted.

The ttnn codec is a second codec here and never the golden. The hand-worked rows hold the values of 4.2.
"""

import math
from typing import NamedTuple

import pytest
import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.quasar import mx_reference_codec as ref
from tests.ttnn.utils_for_testing import assert_equal


class _Fmt(NamedTuple):
    kind: str  # "int" (MXINT) or "float" (MX float)
    bits: int  # bits of one element code
    storage: int  # bits of one element in the tile
    exp_bits: int  # MX float element only
    man_bits: int  # MX float: mantissa bits; MXINT: magnitude bits w
    bias: int  # MX float element only
    emax: int  # the element's largest unbiased exponent (4.2 rule 1); 0 for MXINT


# The element formats of tt-regen-spec 4.1 and OCP MX v1.0. The test keeps its own table: it shares none with the
# reference.
_FORMATS = {
    "mxint8": _Fmt("int", 8, 8, 0, 7, 0, 0),
    "mxint4": _Fmt("int", 4, 4, 0, 3, 0, 0),
    "mxint2": _Fmt("int", 2, 2, 0, 1, 0, 0),
    "mxfp8_e4m3": _Fmt("float", 8, 8, 4, 3, 7, 8),
    "mxfp8_e5m2": _Fmt("float", 8, 8, 5, 2, 15, 15),
    "mxfp6_e2m3": _Fmt("float", 6, 8, 2, 3, 1, 2),
    "mxfp6_e3m2": _Fmt("float", 6, 8, 3, 2, 3, 4),
    "mxfp4": _Fmt("float", 4, 4, 2, 1, 1, 2),
}
_ALL = tuple(_FORMATS)
# The scales of the agreement sweep: the low scales (the decode flush), the middle, and the top (overflow, NaN).
_SCALES = (0, 1, 2, 126, 127, 128, 253, 254, 255)

# float32 bit patterns of the hand-worked rows.
_NAN, _NNAN, _INF, _NINF = 0x7FC00000, 0xFFC00000, 0x7F800000, 0xFF800000
_NZERO, _ONE, _TINY = 0x80000000, 0x3F800000, 0xB5800000  # -0, 1.0, -2^-20
_DEN_MAX, _NDEN_MAX, _DEN_MIN, _NDEN_MIN = 0x007FFFFF, 0x807FFFFF, 0x00000001, 0x80000001
_Z = "zero of either sign"  # 4.2: "a result below 2^-126 in magnitude becomes zero" states no sign
_N = "NaN of either sign"  # an element NaN code below S = 255: 4.2 states no sign


class _E(NamedTuple):
    """A hand-worked encode row: one block, the scale byte and codes of 4.2, and where the ttnn codec differs."""

    name: str
    fmt: str
    values: tuple  # float32 bit patterns of the first elements; the other elements of the block are +0
    scale: int
    codes: tuple  # the codes of 4.2 for the same elements; the code of each +0 element is 0
    ttnn: dict  # position ("S" or an element index) -> the rule by which the ttnn codec differs from 4.2


# Each row is worked from the text of 4.2 and cross-checked with the RTL-checked port of ttsim's MX model. The rows
# "g1" follow the text of 4.2 (one rounding in the element subnormal range); the port truncates first (design Q4).
# fmt: off
_ENCODE_ROWS = (
    # mxint8: w = 7, an element is code / 64 times X = 2^(S - 127).
    _E("zero", "mxint8", (), 0, (), {"S": "A7"}),
    _E("nan_zero", "mxint8", (_NAN,), 0, (0x00,), {"S": "A7"}),
    _E("all_nan", "mxint8", (_NAN, _NNAN) + (_NAN,) * 30, 255, (0x00,) * 32, {}),
    _E("inf_zero", "mxint8", (_INF, _NINF), 254, (0x7F, 0x81), {}),
    _E("inf_den", "mxint8", (_INF, _NINF, _DEN_MIN, _NDEN_MIN), 254, (0x7F, 0x81, 0x00, 0x00), {"S": "A6"}),
    _E("inf_nan", "mxint8", (_INF, _NAN, _NINF, _NNAN), 254, (0x7F, 0x00, 0x81, 0x00), {}),
    _E("den_max", "mxint8", (_DEN_MAX,), 0, (0x00,), {"S": "A4", 0: "B7"}),
    _E("basic", "mxint8", (0x3FC00000, 0x3EC00000, 0xBF400000, 0x3F810000, 0x3F830000, 0xBF830000), 127,
       (0x60, 0x18, 0xD0, 0x40, 0x42, 0xBE), {}),
    _E("trunc", "mxint8", (_ONE, 0x3F020001, 0xBF020001), 127, (0x40, 0x20, 0xE0), {1: "B1", 2: "B1"}),
    _E("saturate", "mxint8", (0x3FFFFFFF, 0xBFFFFFFF), 127, (0x7F, 0x81), {}),
    _E("wrap_s1", "mxint8", (0x00FFFFFF, 0x80FFFFFF, 0x00800000), 1, (0x00, 0x80, 0x40), {0: "B3", 1: "B3"}),
    _E("special", "mxint8", (_ONE, _INF, _NINF, _NAN, _NNAN, _NZERO, _TINY), 127,
       (0x40, 0x7F, 0x81, 0x00, 0x00, 0x00, 0x00), {}),
    _E("den_datum", "mxint8", (0x00800000, _DEN_MAX, _NDEN_MAX), 1, (0x40, 0x00, 0x00), {1: "B7", 2: "B7"}),
    # mxint4: w = 3, an element is code / 4 times X.
    _E("zero", "mxint4", (), 0, (), {"S": "A7"}),
    _E("nan_zero", "mxint4", (_NAN,), 0, (0x0,), {"S": "A7"}),
    _E("all_nan", "mxint4", (_NAN, _NNAN) + (_NAN,) * 30, 255, (0x0,) * 32, {}),
    _E("inf_zero", "mxint4", (_INF, _NINF), 254, (0x7, 0x9), {}),
    _E("inf_den", "mxint4", (_INF, _NINF, _DEN_MIN, _NDEN_MIN), 254, (0x7, 0x9, 0x0, 0x0), {"S": "A6"}),
    _E("inf_nan", "mxint4", (_INF, _NAN, _NINF, _NNAN), 254, (0x7, 0x0, 0x9, 0x0), {}),
    _E("den_max", "mxint4", (_DEN_MAX,), 0, (0x0,), {"S": "A4", 0: "B7"}),
    _E("basic", "mxint4", (0x3FC00000, 0x3EC00000, 0xBF400000, 0x3F810000, 0x3F830000, 0xBF830000), 127,
       (0x6, 0x2, 0xD, 0x4, 0x4, 0xC), {}),
    _E("trunc", "mxint4", (_ONE, 0x3F200001, 0xBF200001), 127, (0x4, 0x2, 0xE), {1: "B1", 2: "B1"}),
    _E("saturate", "mxint4", (0x3FFFFFFF, 0xBFFFFFFF), 127, (0x7, 0x9), {}),
    _E("wrap_s1", "mxint4", (0x00FFFFFF, 0x80FFFFFF, 0x00800000), 1, (0x0, 0x8, 0x4), {0: "B3", 1: "B3"}),
    _E("special", "mxint4", (_ONE, _INF, _NINF, _NAN, _NNAN, _NZERO, _TINY), 127,
       (0x4, 0x7, 0x9, 0x0, 0x0, 0x0, 0x0), {}),
    _E("den_datum", "mxint4", (0x00800000, _DEN_MAX, _NDEN_MAX), 1, (0x4, 0x0, 0x0), {1: "B7", 2: "B7"}),
    # mxint2: w = 1, an element is code times X (0b01 = 1, 0b11 = -1).
    _E("zero", "mxint2", (), 0, (), {"S": "A7"}),
    _E("nan_zero", "mxint2", (_NAN,), 0, (0x0,), {"S": "A7"}),
    _E("all_nan", "mxint2", (_NAN, _NNAN) + (_NAN,) * 30, 255, (0x0,) * 32, {}),
    _E("inf_zero", "mxint2", (_INF, _NINF), 254, (0x1, 0x3), {}),
    _E("inf_den", "mxint2", (_INF, _NINF, _DEN_MIN, _NDEN_MIN), 254, (0x1, 0x3, 0x0, 0x0), {"S": "A6"}),
    _E("inf_nan", "mxint2", (_INF, _NAN, _NINF, _NNAN), 254, (0x1, 0x0, 0x3, 0x0), {}),
    _E("den_max", "mxint2", (_DEN_MAX,), 0, (0x0,), {"S": "A4", 0: "B7"}),
    _E("basic", "mxint2", (0x3FC00000, 0x3EC00000, 0xBF400000, 0x3F810000, 0x3F830000, 0xBF830000), 127,
       (0x1, 0x0, 0x3, 0x1, 0x1, 0x3), {}),
    _E("trunc", "mxint2", (_ONE, 0x3F000001, 0xBF000001), 127, (0x1, 0x0, 0x0), {1: "B1", 2: "B1"}),
    _E("saturate", "mxint2", (0x3FFFFFFF, 0xBFFFFFFF), 127, (0x1, 0x3), {}),
    _E("wrap_s1", "mxint2", (0x00FFFFFF, 0x80FFFFFF, 0x00800000), 1, (0x0, 0x2, 0x1), {0: "B3", 1: "B3"}),
    _E("special", "mxint2", (_ONE, _INF, _NINF, _NAN, _NNAN, _NZERO, _TINY), 127,
       (0x1, 0x1, 0x3, 0x0, 0x0, 0x0, 0x0), {}),
    _E("den_datum", "mxint2", (0x00800000, _DEN_MAX, _NDEN_MAX), 1, (0x1, 0x0, 0x0), {1: "B7", 2: "B7"}),
    # mxfp8_e4m3: E4M3, bias 7, emax 8; no Inf; NaN 0x7F / 0xFF; the largest finite element 448 (0x7E).
    _E("zero", "mxfp8_e4m3", (), 0, (), {"S": "A7"}),
    _E("nan_zero", "mxfp8_e4m3", (_NAN,), 0, (0x7F,), {"S": "A7"}),
    _E("all_nan", "mxfp8_e4m3", (_NAN, _NNAN) + (_NAN,) * 30, 255, (0x7F, 0xFF) + (0x7F,) * 30, {1: "C4"}),
    _E("inf_zero", "mxfp8_e4m3", (_INF, _NINF), 254, (0x7E, 0xFE), {}),
    _E("inf_den", "mxfp8_e4m3", (_INF, _NINF, _DEN_MIN, _NDEN_MIN), 254, (0x7E, 0xFE, 0x00, 0x80), {"S": "A6"}),
    _E("inf_nan", "mxfp8_e4m3", (_INF, _NAN, _NINF, _NNAN), 254, (0x7E, 0x7F, 0xFE, 0xFF), {3: "C4"}),
    _E("den_max", "mxfp8_e4m3", (_DEN_MAX,), 0, (0x00,), {0: "C6"}),
    _E("clamp", "mxfp8_e4m3", (0x04000000, 0x00800000, 0x80800000), 0, (0x78, 0x40, 0xC0), {}),
    _E("clamp_lt", "mxfp8_e4m3", (0x03800000, 0x00800000, 0x80C00000), 0, (0x70, 0x40, 0xC4), {}),
    _E("basic", "mxfp8_e4m3", (0x43800000, _ONE, 0x3F880000, 0x3F980000, 0xBF8CCCCD), 127,
       (0x78, 0x38, 0x38, 0x3A, 0xB9), {}),
    _E("sub", "mxfp8_e4m3", (0x43800000, 0x3B000000, 0x3BC00000, 0x3A800000, 0x3B400000, 0xBBA00000, 0x3C600000,
                             0x3C800000), 127, (0x78, 0x01, 0x03, 0x00, 0x02, 0x82, 0x07, 0x08), {}),
    _E("g1", "mxfp8_e4m3", (0x43800000, 0x3A800001, 0x3BA00001), 127, (0x78, 0x01, 0x03), {}),
    _E("saturate", "mxfp8_e4m3", (0x43800000, 0x43FFFFFF, 0x43F00000, 0xC3E80000), 127, (0x78, 0x7E, 0x7E, 0xFE), {}),
    _E("special", "mxfp8_e4m3", (0x43800000, _INF, _NINF, _NAN, _NNAN, _NZERO, _TINY), 127,
       (0x78, 0x7E, 0xFE, 0x7F, 0xFF, 0x80, 0x80), {4: "C4"}),
    _E("den_datum", "mxfp8_e4m3", (0x04800000, _DEN_MAX, _NDEN_MAX), 1, (0x78, 0x00, 0x80), {1: "C6", 2: "C6"}),
    # mxfp8_e5m2: E5M2, bias 15, emax 15; Inf 0x7C / 0xFC; NaN 0x7D-0x7F / 0xFD-0xFF; the largest finite 57344 (0x7B).
    _E("zero", "mxfp8_e5m2", (), 0, (), {"S": "A7"}),
    _E("nan_zero", "mxfp8_e5m2", (_NAN,), 0, (0x7E,), {"S": "A7", 0: "C4"}),
    _E("all_nan", "mxfp8_e5m2", (_NAN, _NNAN) + (_NAN,) * 30, 255, (0x7E, 0xFE) + (0x7E,) * 30,
       {i: "C4" for i in range(32)}),
    _E("inf_zero", "mxfp8_e5m2", (_INF, _NINF), 254, (0x7B, 0xFB), {0: "C3", 1: "C3"}),
    _E("inf_den", "mxfp8_e5m2", (_INF, _NINF, _DEN_MIN, _NDEN_MIN), 254, (0x7B, 0xFB, 0x00, 0x80),
       {"S": "A6", 0: "C3", 1: "C3"}),
    _E("inf_nan", "mxfp8_e5m2", (_INF, _NAN, _NINF, _NNAN), 254, (0x7B, 0x7E, 0xFB, 0xFE),
       {0: "C3", 1: "C4", 2: "C3", 3: "C4"}),
    _E("den_max", "mxfp8_e5m2", (_DEN_MAX,), 0, (0x00,), {0: "C6"}),
    _E("clamp", "mxfp8_e5m2", (0x07800000, 0x00800000, 0x80800000), 0, (0x78, 0x40, 0xC0), {}),
    _E("clamp_lt", "mxfp8_e5m2", (0x07000000, 0x00800000, 0x80C00000), 0, (0x74, 0x40, 0xC2), {}),
    _E("basic", "mxfp8_e5m2", (0x47000000, _ONE, 0x3F900000, 0x3FB00000, 0xBF8CCCCD), 127,
       (0x78, 0x3C, 0x3C, 0x3E, 0xBC), {}),
    _E("sub", "mxfp8_e5m2", (0x47000000, 0x37800000, 0x38400000, 0x37000000, 0x37C00000, 0xB8200000, 0x38800000),
       127, (0x78, 0x01, 0x03, 0x00, 0x02, 0x82, 0x04), {}),
    _E("g1", "mxfp8_e5m2", (0x47000000, 0x37000001, 0x38200001), 127, (0x78, 0x01, 0x03), {}),
    _E("saturate", "mxfp8_e5m2", (0x47000000, 0x477FFFFF, 0xC7700000), 127, (0x78, 0x7B, 0xFB), {}),
    _E("special", "mxfp8_e5m2", (0x47000000, _INF, _NINF, _NAN, _NNAN, _NZERO, _TINY), 127,
       (0x78, 0x7B, 0xFB, 0x7E, 0xFE, 0x80, 0x80), {1: "C3", 2: "C3", 3: "C4", 4: "C4"}),
    _E("den_datum", "mxfp8_e5m2", (0x08000000, _DEN_MAX, _NDEN_MAX), 1, (0x78, 0x00, 0x80), {1: "C6", 2: "C6"}),
    # mxfp6_e2m3: E2M3, bias 1, emax 2; no Inf or NaN; the largest finite 7.5 (0x1F); the sign is bit 5.
    _E("zero", "mxfp6_e2m3", (), 0, (), {"S": "A7"}),
    _E("nan_zero", "mxfp6_e2m3", (_NAN,), 0, (0x00,), {"S": "A7"}),
    _E("all_nan", "mxfp6_e2m3", (_NAN, _NNAN) + (_NAN,) * 30, 255, (0x00,) * 32, {}),
    _E("inf_zero", "mxfp6_e2m3", (_INF, _NINF), 254, (0x1F, 0x3F), {}),
    _E("inf_den", "mxfp6_e2m3", (_INF, _NINF, _DEN_MIN, _NDEN_MIN), 254, (0x1F, 0x3F, 0x00, 0x20), {"S": "A6"}),
    _E("inf_nan", "mxfp6_e2m3", (_INF, _NAN, _NINF, _NNAN), 254, (0x1F, 0x00, 0x3F, 0x00), {}),
    _E("den_max", "mxfp6_e2m3", (_DEN_MAX,), 0, (0x00,), {0: "C6"}),
    _E("clamp", "mxfp6_e2m3", (0x01000000, 0x00800000, 0x80800000), 0, (0x18, 0x10, 0x30), {}),
    _E("clamp_lt", "mxfp6_e2m3", (0x00800000, 0x80C00000), 0, (0x10, 0x34), {}),
    _E("basic", "mxfp6_e2m3", (0x40800000, _ONE, 0x3F880000, 0x3F980000, 0xBF8CCCCD), 127,
       (0x18, 0x08, 0x08, 0x0A, 0x29), {}),
    _E("sub", "mxfp6_e2m3", (0x40800000, 0x3E000000, 0x3EC00000, 0x3D800000, 0x3E400000, 0xBEA00000, 0x3F600000),
       127, (0x18, 0x01, 0x03, 0x00, 0x02, 0x22, 0x07), {}),
    _E("g1", "mxfp6_e2m3", (0x40800000, 0x3D800001, 0x3F100001), 127, (0x18, 0x01, 0x05), {}),
    _E("saturate", "mxfp6_e2m3", (0x40800000, 0x40FFFFFF, 0xC0F80000), 127, (0x18, 0x1F, 0x3F), {}),
    _E("special", "mxfp6_e2m3", (0x40800000, _INF, _NINF, _NAN, _NNAN, _NZERO, _TINY), 127,
       (0x18, 0x1F, 0x3F, 0x00, 0x00, 0x20, 0x20), {}),
    _E("den_datum", "mxfp6_e2m3", (0x01800000, _DEN_MAX, _NDEN_MAX), 1, (0x18, 0x00, 0x20), {1: "C6", 2: "C6"}),
    # mxfp6_e3m2: E3M2, bias 3, emax 4; no Inf or NaN; the largest finite 28 (0x1F); the sign is bit 5.
    _E("zero", "mxfp6_e3m2", (), 0, (), {"S": "A7"}),
    _E("nan_zero", "mxfp6_e3m2", (_NAN,), 0, (0x00,), {"S": "A7"}),
    _E("all_nan", "mxfp6_e3m2", (_NAN, _NNAN) + (_NAN,) * 30, 255, (0x00,) * 32, {}),
    _E("inf_zero", "mxfp6_e3m2", (_INF, _NINF), 254, (0x1F, 0x3F), {}),
    _E("inf_den", "mxfp6_e3m2", (_INF, _NINF, _DEN_MIN, _NDEN_MIN), 254, (0x1F, 0x3F, 0x00, 0x20), {"S": "A6"}),
    _E("inf_nan", "mxfp6_e3m2", (_INF, _NAN, _NINF, _NNAN), 254, (0x1F, 0x00, 0x3F, 0x00), {}),
    _E("den_max", "mxfp6_e3m2", (_DEN_MAX,), 0, (0x00,), {0: "C6"}),
    _E("clamp", "mxfp6_e3m2", (0x02000000, 0x00800000, 0x80800000), 0, (0x1C, 0x10, 0x30), {}),
    _E("clamp_lt", "mxfp6_e3m2", (0x01800000, 0x00800000, 0x80C00000), 0, (0x18, 0x10, 0x32), {}),
    _E("basic", "mxfp6_e3m2", (0x41800000, _ONE, 0x3F900000, 0x3FB00000, 0xBF8CCCCD), 127,
       (0x1C, 0x0C, 0x0C, 0x0E, 0x2C), {}),
    _E("sub", "mxfp6_e3m2", (0x41800000, 0x3D800000, 0x3E400000, 0x3D000000, 0x3DC00000, 0xBE200000, 0x3E800000),
       127, (0x1C, 0x01, 0x03, 0x00, 0x02, 0x22, 0x04), {}),
    _E("g1", "mxfp6_e3m2", (0x41800000, 0x3D000001, 0x3E200001), 127, (0x1C, 0x01, 0x03), {}),
    _E("saturate", "mxfp6_e3m2", (0x41800000, 0x41FFFFFF, 0xC1F00000), 127, (0x1C, 0x1F, 0x3F), {}),
    _E("special", "mxfp6_e3m2", (0x41800000, _INF, _NINF, _NAN, _NNAN, _NZERO, _TINY), 127,
       (0x1C, 0x1F, 0x3F, 0x00, 0x00, 0x20, 0x20), {}),
    _E("den_datum", "mxfp6_e3m2", (0x02800000, _DEN_MAX, _NDEN_MAX), 1, (0x1C, 0x00, 0x20), {1: "C6", 2: "C6"}),
    # mxfp4: E2M1, bias 1, emax 2; no Inf or NaN; the values 0, 0.5, 1, 1.5, 2, 3, 4, 6; the sign is bit 3.
    _E("zero", "mxfp4", (), 0, (), {"S": "A7"}),
    _E("nan_zero", "mxfp4", (_NAN,), 0, (0x0,), {"S": "A7"}),
    _E("all_nan", "mxfp4", (_NAN, _NNAN) + (_NAN,) * 30, 255, (0x0,) * 32, {}),
    _E("inf_zero", "mxfp4", (_INF, _NINF), 254, (0x7, 0xF), {}),
    _E("inf_den", "mxfp4", (_INF, _NINF, _DEN_MIN, _NDEN_MIN), 254, (0x7, 0xF, 0x0, 0x8), {"S": "A6"}),
    _E("inf_nan", "mxfp4", (_INF, _NAN, _NINF, _NNAN), 254, (0x7, 0x0, 0xF, 0x0), {}),
    _E("den_max", "mxfp4", (_DEN_MAX,), 0, (0x0,), {0: "C6"}),
    _E("clamp", "mxfp4", (0x01000000, 0x00800000, 0x80800000), 0, (0x6, 0x4, 0xC), {}),
    _E("clamp_lt", "mxfp4", (0x00800000, 0x80C00000), 0, (0x4, 0xD), {}),
    _E("basic", "mxfp4", (0x40800000, _ONE, 0x3FA00000, 0x3FE00000, 0xC0200000, 0x40A00000), 127,
       (0x6, 0x2, 0x2, 0x4, 0xC, 0x6), {}),
    _E("sub", "mxfp4", (0x40800000, 0x3F000000, 0x3E800000, 0x3F400000, 0xBE800000), 127,
       (0x6, 0x1, 0x0, 0x2, 0x8), {}),
    _E("g1", "mxfp4", (0x40800000, 0x3E800001), 127, (0x6, 0x1), {}),
    _E("saturate", "mxfp4", (0x40800000, 0x40FFFFFF, 0xC0E00000), 127, (0x6, 0x7, 0xF), {}),
    _E("special", "mxfp4", (0x40800000, _INF, _NINF, _NAN, _NNAN, _NZERO, _TINY), 127,
       (0x6, 0x7, 0xF, 0x0, 0x0, 0x8, 0x8), {}),
    _E("den_datum", "mxfp4", (0x01800000, _DEN_MAX, _NDEN_MAX), 1, (0x6, 0x0, 0x8), {1: "C6", 2: "C6"}),
)
# fmt: on


class _D(NamedTuple):
    """A hand-worked decode row: one block's scale byte, some element codes, and the values of 4.2."""

    name: str
    fmt: str
    scale: int
    codes: tuple
    f32: tuple  # float32 bit patterns, _Z or _N; a NaN pattern compares the NaN and its sign, not the payload
    bf16: tuple  # bfloat16 bit patterns or _N


# A zero code decodes to a zero of its sign ("the value times X"; in bfloat16 "the product is exact"). The +0 of
# the bfloat16 flush applies to a nonzero product below 2^-126. The MXINT code -2^w is -0 (design Q3).
# fmt: off
_DECODE_ROWS = (
    _D("s127", "mxint8", 127, (0x40, 0x7F, 0x81, 0xFF, 0x01, 0x00, 0x80),
       (0x3F800000, 0x3FFE0000, 0xBFFE0000, 0xBC800000, 0x3C800000, 0x00000000, 0x80000000),
       (0x3F80, 0x3FFE, 0xBFFE, 0xBC80, 0x3C80, 0x0000, 0x8000)),
    _D("flush_s1", "mxint8", 1, (0x40, 0x3F, 0xC1, 0x7F, 0x01),
       (0x00800000, _Z, _Z, 0x00FE0000, _Z), (0x0080, 0x0000, 0x0000, 0x00FE, 0x0000)),
    _D("s254", "mxint8", 254, (0x7F, 0x81, 0x40), (0x7F7E0000, 0xFF7E0000, 0x7F000000), (0x7F7E, 0xFF7E, 0x7F00)),
    _D("s255", "mxint8", 255, (0x00, 0x40, 0xFF, 0x80), (_NAN, _NAN, _NNAN, _NNAN), (0x7FC0, 0x7FC0, 0xFFC0, 0xFFC0)),
    _D("s127", "mxint4", 127, (0x4, 0x7, 0x9, 0xF, 0x1, 0x0, 0x8),
       (0x3F800000, 0x3FE00000, 0xBFE00000, 0xBE800000, 0x3E800000, 0x00000000, 0x80000000),
       (0x3F80, 0x3FE0, 0xBFE0, 0xBE80, 0x3E80, 0x0000, 0x8000)),
    _D("flush_s1", "mxint4", 1, (0x4, 0x3, 0xD, 0x7), (0x00800000, _Z, _Z, 0x00E00000),
       (0x0080, 0x0000, 0x0000, 0x00E0)),
    _D("s254", "mxint4", 254, (0x7, 0x9, 0x4), (0x7F600000, 0xFF600000, 0x7F000000), (0x7F60, 0xFF60, 0x7F00)),
    _D("s255", "mxint4", 255, (0x0, 0x4, 0xF, 0x8), (_NAN, _NAN, _NNAN, _NNAN), (0x7FC0, 0x7FC0, 0xFFC0, 0xFFC0)),
    _D("s127", "mxint2", 127, (0x1, 0x3, 0x0, 0x2), (0x3F800000, 0xBF800000, 0x00000000, 0x80000000),
       (0x3F80, 0xBF80, 0x0000, 0x8000)),
    _D("flush_s0", "mxint2", 0, (0x1, 0x3), (_Z, _Z), (0x0000, 0x0000)),
    _D("normal_s1", "mxint2", 1, (0x1, 0x3), (0x00800000, 0x80800000), (0x0080, 0x8080)),
    _D("s254", "mxint2", 254, (0x1, 0x3), (0x7F000000, 0xFF000000), (0x7F00, 0xFF00)),
    _D("s255", "mxint2", 255, (0x0, 0x1, 0x3, 0x2), (_NAN, _NAN, _NNAN, _NNAN), (0x7FC0, 0x7FC0, 0xFFC0, 0xFFC0)),
    _D("s127", "mxfp8_e4m3", 127, (0x38, 0x7E, 0xFE, 0x01, 0x07, 0x08, 0x00, 0x80, 0x7F, 0xFF),
       (0x3F800000, 0x43E00000, 0xC3E00000, 0x3B000000, 0x3C600000, 0x3C800000, 0x00000000, 0x80000000, _N, _N),
       (0x3F80, 0x43E0, 0xC3E0, 0x3B00, 0x3C60, 0x3C80, 0x0000, 0x8000, _N, _N)),
    _D("flush_s1", "mxfp8_e4m3", 1, (0x38, 0x30, 0xB0, 0x01, 0x7E), (0x00800000, _Z, _Z, _Z, 0x04E00000),
       (0x0080, 0x0000, 0x0000, 0x0000, 0x04E0)),
    _D("s254", "mxfp8_e4m3", 254, (0x40, 0xC0, 0x3F, 0x7E), (0x7F800000, 0xFF800000, 0x7F700000, 0x7F800000),
       (0x7F80, 0xFF80, 0x7F70, 0x7F80)),
    _D("s255", "mxfp8_e4m3", 255, (0x00, 0x38, 0x80, 0xB8, 0xFF), (_NAN, _NAN, _NNAN, _NNAN, _NNAN),
       (0x7FC0, 0x7FC0, 0xFFC0, 0xFFC0, 0xFFC0)),
    _D("s127", "mxfp8_e5m2", 127, (0x3C, 0x7B, 0xFB, 0x01, 0x03, 0x04, 0x00, 0x80, 0x7C, 0xFC, 0x7D, 0xFF),
       (0x3F800000, 0x47600000, 0xC7600000, 0x37800000, 0x38400000, 0x38800000, 0x00000000, 0x80000000,
        0x7F800000, 0xFF800000, _N, _N),
       (0x3F80, 0x4760, 0xC760, 0x3780, 0x3840, 0x3880, 0x0000, 0x8000, 0x7F80, 0xFF80, _N, _N)),
    _D("flush_s1", "mxfp8_e5m2", 1, (0x3C, 0x38, 0xB8, 0x7B), (0x00800000, _Z, _Z, 0x08600000),
       (0x0080, 0x0000, 0x0000, 0x0860)),
    _D("s254", "mxfp8_e5m2", 254, (0x40, 0xC0, 0x3F, 0x7B), (0x7F800000, 0xFF800000, 0x7F600000, 0x7F800000),
       (0x7F80, 0xFF80, 0x7F60, 0x7F80)),
    _D("s255", "mxfp8_e5m2", 255, (0x00, 0x3C, 0x80, 0xFC), (_NAN, _NAN, _NNAN, _NNAN),
       (0x7FC0, 0x7FC0, 0xFFC0, 0xFFC0)),
    _D("s127", "mxfp6_e2m3", 127, (0x08, 0x1F, 0x3F, 0x01, 0x07, 0x00, 0x20),
       (0x3F800000, 0x40F00000, 0xC0F00000, 0x3E000000, 0x3F600000, 0x00000000, 0x80000000),
       (0x3F80, 0x40F0, 0xC0F0, 0x3E00, 0x3F60, 0x0000, 0x8000)),
    _D("flush_s1", "mxfp6_e2m3", 1, (0x08, 0x07, 0x27, 0x1F), (0x00800000, _Z, _Z, 0x01F00000),
       (0x0080, 0x0000, 0x0000, 0x01F0)),
    _D("s254", "mxfp6_e2m3", 254, (0x10, 0x30, 0x0F, 0x1F), (0x7F800000, 0xFF800000, 0x7F700000, 0x7F800000),
       (0x7F80, 0xFF80, 0x7F70, 0x7F80)),
    _D("s255", "mxfp6_e2m3", 255, (0x00, 0x08, 0x20, 0x3F), (_NAN, _NAN, _NNAN, _NNAN),
       (0x7FC0, 0x7FC0, 0xFFC0, 0xFFC0)),
    _D("s127", "mxfp6_e3m2", 127, (0x0C, 0x1F, 0x3F, 0x01, 0x03, 0x04, 0x00, 0x20),
       (0x3F800000, 0x41E00000, 0xC1E00000, 0x3D800000, 0x3E400000, 0x3E800000, 0x00000000, 0x80000000),
       (0x3F80, 0x41E0, 0xC1E0, 0x3D80, 0x3E40, 0x3E80, 0x0000, 0x8000)),
    _D("flush_s1", "mxfp6_e3m2", 1, (0x0C, 0x08, 0x28, 0x1F), (0x00800000, _Z, _Z, 0x02E00000),
       (0x0080, 0x0000, 0x0000, 0x02E0)),
    _D("s254", "mxfp6_e3m2", 254, (0x10, 0x30, 0x0F, 0x1F), (0x7F800000, 0xFF800000, 0x7F600000, 0x7F800000),
       (0x7F80, 0xFF80, 0x7F60, 0x7F80)),
    _D("s255", "mxfp6_e3m2", 255, (0x00, 0x0C, 0x20, 0x3F), (_NAN, _NAN, _NNAN, _NNAN),
       (0x7FC0, 0x7FC0, 0xFFC0, 0xFFC0)),
    _D("s127", "mxfp4", 127, (0x2, 0x7, 0xF, 0x1, 0x5, 0x0, 0x8),
       (0x3F800000, 0x40C00000, 0xC0C00000, 0x3F000000, 0x40400000, 0x00000000, 0x80000000),
       (0x3F80, 0x40C0, 0xC0C0, 0x3F00, 0x4040, 0x0000, 0x8000)),
    _D("flush_s1", "mxfp4", 1, (0x2, 0x1, 0x9, 0x7), (0x00800000, _Z, _Z, 0x01C00000),
       (0x0080, 0x0000, 0x0000, 0x01C0)),
    _D("s254", "mxfp4", 254, (0x4, 0xC, 0x3, 0x7), (0x7F800000, 0xFF800000, 0x7F400000, 0x7F800000),
       (0x7F80, 0xFF80, 0x7F40, 0x7F80)),
    _D("s255", "mxfp4", 255, (0x0, 0x2, 0x8, 0xF), (_NAN, _NAN, _NNAN, _NNAN), (0x7FC0, 0x7FC0, 0xFFC0, 0xFFC0)),
)
# fmt: on


# The rules by which the ttnn host codec differs from 4.2, from the stage 0 comparison of the two: rule -> (the
# number of elements that the rule predicts over the 8 dtypes, what differs). A, B, C: encode; D, Q3: decode.
_KNOWN_DIFFERENCES = {
    "A4": (3, "MXINT, a block whose largest datum is a float32 denormal: ttnn S = 1 (log2f rounds), 4.2 S = 0"),
    "A6": (8, "Inf with denormals and no normal datum: ttnn S = 0, 4.2 S = 254"),
    "A7": (27, "a block of zeros, or of zeros and NaNs: ttnn S = 127 - emax, 4.2 S = 0"),
    "B1": (6, "MXINT, a tie after the truncating alignment: ttnn rounds once, one code larger"),
    "B3": (6, "MXINT at S = 1, an overflowing magnitude: ttnn saturates, 4.2 wraps"),
    "B7": (9, "MXINT, a float32 denormal datum: ttnn rounds it, 4.2 counts it as zero"),
    "C3": (256, "mxfp8_e5m2 Inf: ttnn writes the Inf code 0x7C / 0xFC, 4.2 saturates to 0x7B / 0xFB"),
    "C4": (512, "NaN: mxfp8_e5m2 ttnn 0x7D, 4.2 0x7E / 0xFE; mxfp8_e4m3 -NaN ttnn 0x7F, 4.2 0xFF"),
    "C6": (15, "MX float, a float32 denormal datum: ttnn rounds it, 4.2 keeps a zero of its sign"),
    "D2": (1318, "float32 decode below 2^-126: ttnn gives a denormal, 4.2 zero"),
    "D4": (1312, "S = 255: MXINT ttnn 0, 4.2 NaN; MX float with the sign bit ttnn +NaN, 4.2 -NaN"),
    "D6": (1227, "bfloat16 decode below 2^-126: ttnn gives a denormal or -0, 4.2 +0"),
    "Q3": (176, "the MXINT code -2^w: ttnn -2 X, 4.2 -0 (the hardware's reading, design Q3)"),
}
_ENCODE_RULES = ("A4", "A6", "A7", "B1", "B3", "B7", "C3", "C4", "C6")


def _f32(bits):
    """float32 tensor with these bit patterns (int64 values in [0, 2^32))."""
    bits = torch.as_tensor(bits, dtype=torch.int64)
    return torch.where(bits >= 2**31, bits - 2**32, bits).to(torch.int32).view(torch.float32)


def _code_values(fmt):
    """float64 [2^bits]: the value of each element code before the scale. A NaN code gives NaN, an Inf code Inf,
    a negative zero code -0. The MXINT code -2^w is -0: the hardware's reading (design Q3)."""
    f = _FORMATS[fmt]
    out = []
    for c in range(1 << f.bits):
        if f.kind == "int":
            w = f.man_bits
            if c == 1 << w:
                out.append(-0.0)
            else:
                out.append((c - (1 << (w + 1)) if c >> w else c) / 2.0 ** (w - 1))
            continue
        sign, e, m = c >> (f.bits - 1), (c >> f.man_bits) & ((1 << f.exp_bits) - 1), c & ((1 << f.man_bits) - 1)
        top = e == (1 << f.exp_bits) - 1
        if fmt == "mxfp8_e5m2" and top:
            mag = math.inf if m == 0 else math.nan
        elif fmt == "mxfp8_e4m3" and top and m == (1 << f.man_bits) - 1:
            mag = math.nan
        elif e == 0:
            mag = m * 2.0 ** (1 - f.bias - f.man_bits)
        else:
            mag = (1 + m / 2**f.man_bits) * 2.0 ** (e - f.bias)
        out.append(-mag if sign else mag)
    return torch.tensor(out, dtype=torch.float64)


def _code_negative(fmt):
    """bool [2^bits]: the code has its sign bit set (MXINT: two's complement negative, or -2^w)."""
    f = _FORMATS[fmt]
    return (torch.arange(1 << f.bits) >> (f.bits - 1)) == 1


def _block_index():
    """(rows, cols) int64 [32, 32]: element i of block b of a 32x32 tile sits at (rows[b, i], cols[b, i]). 4.3: a
    block is two consecutive rows of one 16x16 face; block b is face b // 8, rows 2 (b % 8) and 2 (b % 8) + 1."""
    b = torch.arange(32).unsqueeze(1)
    i = torch.arange(32).unsqueeze(0)
    face, pair = b // 8, b % 8
    return (face // 2) * 16 + 2 * pair + i // 16, (face % 2) * 16 + i % 16


_ROWS, _COLS = _block_index()


def _tiles_to_blocks(x):
    """[32 T, 32] tiles stacked down -> [32 T, 32] blocks, tile by tile."""
    return x.reshape(-1, 32, 32)[:, _ROWS, _COLS].reshape(-1, 32)


def _blocks_to_tiles(blocks):
    """[32 T, 32] blocks -> [32 T, 32] tiles stacked down."""
    t = blocks.shape[0] // 32
    out = torch.zeros(t, 32, 32, dtype=blocks.dtype)
    out[:, _ROWS, _COLS] = blocks.reshape(t, 32, 32)
    return out.reshape(t * 32, 32)


def _pack_tiles(scales, codes, fmt):
    """uint8 bytes of the tiles that hold these blocks, 32 blocks a tile. Per tile: the 32 scale bytes, then element
    32 b + i at bit 256 + (32 b + i) * storage, low bits first; a 6-bit code sits in bits 7..2 of its byte."""
    f = _FORMATS[fmt]
    t = scales.shape[0] // 32
    per_byte = 8 // f.storage
    units = (codes.long() << (f.storage - f.bits)).reshape(t, 1024 // per_byte, per_byte)
    data = (units << (torch.arange(per_byte) * f.storage)).sum(-1)
    return torch.cat([scales.long().reshape(t, 32), data], dim=1).to(torch.uint8).reshape(-1)


def _unpack_tiles(raw, fmt):
    """The inverse of _pack_tiles: (scales [32 T], codes [32 T, 32])."""
    f = _FORMATS[fmt]
    per_byte = 8 // f.storage
    t = raw.long().reshape(-1, 32 + 1024 // per_byte)
    units = (t[:, 32:].unsqueeze(-1) >> (torch.arange(per_byte) * f.storage)) & ((1 << f.storage) - 1)
    return t[:, :32].reshape(-1), units.reshape(-1, 32) >> (f.storage - f.bits)


def _host_bytes(t):
    """A uint8 view of a host MX tensor's packed bytes, and the buffer that keeps the view alive."""
    buf = t.host_buffer().get_shard(ttnn.MeshCoordinate([0, 0]))
    return torch.from_dlpack(buf), buf


def _ttnn_encode(blocks, fmt):
    """The ttnn host codec's (scales, codes) for float32 blocks [n, 32], read from the tile bytes."""
    n = blocks.shape[0]
    bits = torch.zeros(-(-n // 32) * 32, 32, dtype=torch.int32)
    bits[:n] = blocks.view(torch.int32)
    t = ttnn.from_torch(_blocks_to_tiles(bits).view(torch.float32), dtype=getattr(ttnn, fmt), layout=ttnn.TILE_LAYOUT)
    view, _buf = _host_bytes(t)
    scales, codes = _unpack_tiles(view.clone(), fmt)
    return scales[:n], codes[:n]


def _ttnn_decode(scales, codes, fmt):
    """The ttnn host codec's float32 and bfloat16 values [n, 32] of these blocks, written as raw tile bytes."""
    n = scales.shape[0]
    m = -(-n // 32) * 32
    s = torch.full((m,), 127, dtype=torch.int64)
    c = torch.zeros(m, 32, dtype=torch.int64)
    s[:n], c[:n] = scales, codes
    raw = _pack_tiles(s, c, fmt)
    t = ttnn.from_torch(torch.zeros(m, 32), dtype=getattr(ttnn, fmt), layout=ttnn.TILE_LAYOUT)
    view, _buf = _host_bytes(t)
    assert view.numel() == raw.numel(), f"{fmt}: {view.numel()} host bytes, {raw.numel()} written"
    view.copy_(raw)
    f32 = ttnn.to_torch(t)
    bf16 = ttnn.to_torch(ttnn.to_dtype(t, ttnn.bfloat16))
    assert f32.dtype == torch.float32 and bf16.dtype == torch.bfloat16, (f32.dtype, bf16.dtype)
    return _tiles_to_blocks(f32)[:n], _tiles_to_blocks(bf16)[:n]


def _ttnn_bytes(tile, fmt):
    """The bytes that the ttnn host codec writes for a float32 32x32 tile."""
    view, _buf = _host_bytes(ttnn.from_torch(tile, dtype=getattr(ttnn, fmt), layout=ttnn.TILE_LAYOUT))
    return view.clone()


def _bits(x, keep_nan_sign):
    """Bit patterns (int64) of float32 or bfloat16 values. A NaN becomes the quiet NaN, with its sign only where
    keep_nan_sign is set: 4.2 states the NaN sign at S = 255 only. Zero signs stay."""
    if x.dtype == torch.float32:
        bits, sign, qnan = x.view(torch.int32).long() & 0xFFFFFFFF, 0x80000000, 0x7FC00000
    else:
        bits, sign, qnan = x.view(torch.int16).long() & 0xFFFF, 0x8000, 0x7FC0
    return torch.where(torch.isnan(x), qnan | (bits & sign) * keep_nan_sign.long(), bits)


def _encode_sweep(fmt):
    """float32 [n, 32]: per scale S of _SCALES, blocks of an anchor and 31 element codes, so that each code appears.
    An element is value(code) * 2^(S - 127), with these input rules:
    - a value below 2^-126 becomes a zero of its sign (the hand-worked rows hold the float32 denormal inputs);
    - a value of 2^128 or more becomes Inf of its sign (the float32 rounding);
    - a NaN code, and every code at S = 255, gives a NaN with the sign of the code.
    The anchor is the largest finite element of the scale, made positive, so that the block keeps the scale where
    it can. With no finite non-zero element (S = 255; MXINT at S = 0), the anchor is the element of code 0."""
    n = 1 << _FORMATS[fmt].bits
    v, neg = _code_values(fmt), _code_negative(fmt)
    blocks = []
    for s in _SCALES:
        x = v * 2.0 ** (s - 127)
        x = torch.where(torch.isfinite(x) & (x.abs() < 2.0**-126), torch.where(neg, -0.0, 0.0), x).float()
        bits = x.view(torch.int32).long() & 0xFFFFFFFF
        nan = torch.isnan(x) | (s == 255)
        bits = torch.where(nan, torch.where(neg, _NNAN, _NAN), bits)
        finite = ~nan & torch.isfinite(x) & (x != 0)
        if finite.any():
            anchor = int(bits[torch.argmax(torch.where(finite, x.abs(), -1.0))] & 0x7FFFFFFF)
        else:
            anchor = int(bits[0])
        for j in range(-(-n // 31)):
            blocks.append([anchor] + [int(bits[(31 * j + i) % n]) for i in range(31)])
    return _f32(blocks)


def _row_blocks(rows):
    """float32 [len(rows), 32]: the block of each hand-worked encode row, padded with +0."""
    bits = torch.zeros(len(rows), 32, dtype=torch.int64)
    for k, r in enumerate(rows):
        bits[k, : len(r.values)] = torch.tensor(r.values, dtype=torch.int64)
    return _f32(bits)


def _predict_encode(fmt, sweep, rows):
    """(block, position) -> rule, where the ttnn codec differs from 4.2 on the encode inputs (the sweep, then the
    rows). Position -1 is the scale byte. On the sweep:
    - A7: a block with no finite non-zero datum, not all NaN and with no Inf;
    - C3: an mxfp8_e5m2 Inf;
    - C4: an mxfp8_e5m2 NaN, and an mxfp8_e4m3 negative NaN.
    On the rows: the row's ttnn field."""
    bits = sweep.view(torch.int32).long() & 0xFFFFFFFF
    nan, inf = torch.isnan(sweep), torch.isinf(sweep)
    normal = ((bits & 0x7F800000) != 0) & ~nan & ~inf
    predicted = {}
    for b in torch.nonzero(~normal.any(1) & ~nan.all(1) & ~inf.any(1)).flatten().tolist():
        predicted[(b, -1)] = "A7"
    if fmt == "mxfp8_e5m2":
        rules = (("C3", inf), ("C4", nan))
    elif fmt == "mxfp8_e4m3":
        rules = (("C4", nan & (bits >= 0x80000000)),)
    else:
        rules = ()
    for rule, mask in rules:
        for b, i in torch.nonzero(mask).tolist():
            predicted[(b, i)] = rule
    for k, r in enumerate(rows):
        for pos, rule in r.ttnn.items():
            predicted[(sweep.shape[0] + k, -1 if pos == "S" else pos)] = rule
    return predicted


def _encode_comparison(fmt):
    """(predicted, measured) on the encode inputs of fmt: where the ttnn codec is predicted to differ from 4.2, and
    where the reference and the ttnn codec differ."""
    sweep = _encode_sweep(fmt)
    rows = [r for r in _ENCODE_ROWS if r.fmt == fmt]
    blocks = torch.cat([sweep, _row_blocks(rows)])
    ref_scales, ref_codes = ref.encode_block(blocks, fmt)
    tt_scales, tt_codes = _ttnn_encode(blocks, fmt)
    assert tuple(ref_scales.shape) == tuple(tt_scales.shape) and tuple(ref_codes.shape) == tuple(tt_codes.shape)
    measured = {(b, -1) for b in torch.nonzero(ref_scales.long() != tt_scales).flatten().tolist()}
    measured |= {(b, i) for b, i in torch.nonzero(ref_codes.long() != tt_codes).tolist()}
    return _predict_encode(fmt, sweep, rows), measured


def _decode_sweep(fmt):
    """(scales [n], codes [n, 32]): per scale S of _SCALES, blocks of the codes 32 j + i (mod 2^bits)."""
    n = 1 << _FORMATS[fmt].bits
    per_scale = max(1, n // 32)
    codes = (torch.arange(per_scale * 32) % n).reshape(per_scale, 32).repeat(len(_SCALES), 1)
    return torch.tensor(_SCALES).repeat_interleave(per_scale), codes


def _predict_decode(fmt, scales, codes):
    """(output, block, element) -> rule, where the ttnn decode differs from 4.2 on the decode sweep. X = 2^(S - 127).
    - D2: float32, 0 < |value * X| < 2^-126 (ttnn a denormal, 4.2 zero);
    - D6: bfloat16, the same elements except 0 < value * X <= 2^-134, which ttnn rounds to +0 as 4.2 flushes;
    - D4: S = 255, both outputs: MXINT every code; MX float a code with the sign bit;
    - Q3: below S = 255, both outputs: the MXINT code -2^w."""
    f = _FORMATS[fmt]
    v = _code_values(fmt)[codes]
    x = v * torch.pow(2.0, scales.double() - 127).unsqueeze(1)
    s255 = (scales == 255).unsqueeze(1).expand_as(x)
    tiny = torch.isfinite(x) & (x != 0) & (x.abs() < 2.0**-126) & ~s255
    flush_matches = tiny & (x > 0) & (x <= 2.0**-134)
    if f.kind == "int":
        d4, q3 = s255, (codes == 1 << f.man_bits) & ~s255
    else:
        d4, q3 = s255 & _code_negative(fmt)[codes], torch.zeros_like(s255)
    predicted = {}
    for out, rule, mask in (("f32", "D2", tiny), ("bf16", "D6", tiny & ~flush_matches)):
        predicted |= {(out, b, i): rule for b, i in torch.nonzero(mask).tolist()}
    for rule, mask in (("D4", d4), ("Q3", q3)):
        for out in ("f32", "bf16"):
            predicted |= {(out, b, i): rule for b, i in torch.nonzero(mask).tolist()}
    return predicted


def _decode_comparison(fmt):
    """(predicted, measured) on the decode sweep of fmt, as _encode_comparison. NaN signs compare at S = 255 only."""
    scales, codes = _decode_sweep(fmt)
    ref32 = ref.decode_block(scales, codes, fmt)
    ref16 = ref.to_bfloat16(scales, codes, fmt)
    tt32, tt16 = _ttnn_decode(scales, codes, fmt)
    assert ref32.dtype == torch.float32 and tuple(ref32.shape) == tuple(codes.shape), (ref32.dtype, ref32.shape)
    assert ref16.dtype == torch.bfloat16 and tuple(ref16.shape) == tuple(codes.shape), (ref16.dtype, ref16.shape)
    keep = (scales == 255).unsqueeze(1).expand(-1, 32)
    measured = set()
    for out, a, b in (("f32", ref32, tt32), ("bf16", ref16, tt16)):
        measured |= {(out, k, i) for k, i in torch.nonzero(_bits(a, keep) != _bits(b, keep)).tolist()}
    return _predict_decode(fmt, scales, codes), measured


def _matches(got, want, width):
    """One decoded value against a hand-worked expectation: _Z, _N, or a bit pattern (a NaN pattern compares the
    NaN and its sign)."""
    x = got.reshape(1)
    bits = int(x.view(torch.int32 if width == 32 else torch.int16)[0]) & ((1 << width) - 1)
    if want == _Z:
        return bits & ~(1 << (width - 1)) == 0
    if want == _N:
        return bool(torch.isnan(x)[0])
    exp = ((1 << 8) - 1) << (width - 9)
    if want & exp == exp and want & ((1 << (width - 9)) - 1):
        return bool(torch.isnan(x)[0]) and bits >> (width - 1) == want >> (width - 1)
    return bits == want


def _patch_tile(fmt):
    """A 32x32 tile whose block b has its own scale S = 100 + b, and exact element values: (tile, scales, codes).
    Element 0 of each block is the anchor (MXINT 1.0, MX float 2^emax) that sets S; the others cycle through the
    finite codes (the MXINT code -2^w left out, design Q3)."""
    f = _FORMATS[fmt]
    v = _code_values(fmt)
    anchor = int(torch.nonzero(v == (1.0 if f.kind == "int" else 2.0**f.emax))[0])
    safe = [c for c in range(1 << f.bits) if math.isfinite(v[c]) and not (f.kind == "int" and c == 1 << f.man_bits)]
    codes = torch.tensor([[anchor] + [safe[(5 * b + i) % len(safe)] for i in range(1, 32)] for b in range(32)])
    scales = torch.arange(100, 132)
    values = (v[codes] * torch.pow(2.0, scales.double() - 127).unsqueeze(1)).float()
    return _blocks_to_tiles(values), scales, codes


@pytest.mark.parametrize("fmt", _ALL)
def test_encode_rules(fmt):
    """S0-C1: the reference encodes each hand-worked block to the scale byte and codes of 4.2."""
    rows = [r for r in _ENCODE_ROWS if r.fmt == fmt]
    scales, codes = ref.encode_block(_row_blocks(rows), fmt)
    bad = []
    for k, r in enumerate(rows):
        want = list(r.codes) + [0] * (32 - len(r.codes))
        got = [int(c) for c in codes[k]]
        wrong = [f"[{i}] {got[i]:#x} (4.2: {want[i]:#x})" for i in range(32) if got[i] != want[i]]
        if int(scales[k]) != r.scale or wrong:
            bad.append(f"{r.name}: S {int(scales[k])} (4.2: {r.scale}); codes {', '.join(wrong) or 'equal'}")
    assert not bad, f"{fmt}: {len(bad)} of {len(rows)} hand-worked rows differ from 4.2:\n" + "\n".join(bad)


@pytest.mark.parametrize("fmt", _ALL)
def test_decode_rules(fmt):
    """S0-C2: the reference decodes each hand-worked block to the float32 and bfloat16 values of 4.2."""
    rows = [r for r in _DECODE_ROWS if r.fmt == fmt]
    scales = torch.tensor([r.scale for r in rows])
    codes = torch.zeros(len(rows), 32, dtype=torch.int64)
    for k, r in enumerate(rows):
        codes[k, : len(r.codes)] = torch.tensor(r.codes)
    f32 = ref.decode_block(scales, codes, fmt)
    bf16 = ref.to_bfloat16(scales, codes, fmt)
    assert f32.dtype == torch.float32 and bf16.dtype == torch.bfloat16, (f32.dtype, bf16.dtype)
    bad = []
    for k, r in enumerate(rows):
        for i, (code, w32, w16) in enumerate(zip(r.codes, r.f32, r.bf16)):
            for out, got, want, width in (("f32", f32, w32, 32), ("bf16", bf16, w16, 16)):
                if not _matches(got[k, i], want, width):
                    bits = int(got[k, i].reshape(1).view(torch.int32 if width == 32 else torch.int16)[0])
                    shown = want if isinstance(want, str) else f"{want:#x}"
                    bad.append(f"{r.name} code {code:#x} {out}: {bits & ((1 << width) - 1):#x} (4.2: {shown})")
    assert not bad, f"{fmt}: {len(bad)} hand-worked values differ from 4.2:\n" + "\n".join(bad)


@pytest.mark.parametrize("fmt", _ALL)
def test_block_shape(fmt):
    """S0-C3: a block is a 2x16 patch of one face, and the tile holds its 32 scale bytes first."""
    tile, scales, codes = _patch_tile(fmt)
    want = _pack_tiles(scales, codes, fmt)
    assert_equal(want, ref.encode_tile(tile, fmt))
    assert_equal(tile.view(torch.int32), ref.decode_tile(want, fmt).view(torch.int32))
    # A new maximum in block 5 moves the scale and the values of block 5 only.
    raised = tile.clone()
    raised[_ROWS[5, 0], _COLS[5, 0]] *= 2
    raw = ref.encode_tile(raised, fmt)
    assert int(raw[5]) == int(scales[5]) + 1, f"{fmt}: block 5 has S {int(raw[5])}, not {int(scales[5]) + 1}"
    moved = {tuple(p) for p in torch.nonzero(ref.decode_tile(raw, fmt) != ref.decode_tile(want, fmt)).tolist()}
    patch = set(zip(_ROWS[5].tolist(), _COLS[5].tolist()))
    assert moved and moved <= patch, f"{fmt}: the new maximum of block 5 moved {sorted(moved - patch)[:8]}"
    # The instrument of the agreement tests: the ttnn host codec writes the same bytes.
    assert_equal(want, _ttnn_bytes(tile, fmt))


@pytest.mark.parametrize("fmt", _ALL)
def test_agree_encode(fmt):
    """S0-C4, encode: the reference and the ttnn codec differ only where a rule of _KNOWN_DIFFERENCES predicts."""
    predicted, measured = _encode_comparison(fmt)
    unexpected = sorted(measured - predicted.keys())
    assert not unexpected, (
        f"{fmt}: the codecs differ at {len(unexpected)} places that no rule predicts (block, position; -1 is the "
        f"scale byte), first {unexpected[:8]}"
    )


@pytest.mark.parametrize("fmt", _ALL)
def test_agree_decode(fmt):
    """S0-C4, decode: the reference and the ttnn codec differ only where a rule of _KNOWN_DIFFERENCES predicts."""
    predicted, measured = _decode_comparison(fmt)
    unexpected = sorted(measured - predicted.keys())
    assert not unexpected, (
        f"{fmt}: the codecs differ at {len(unexpected)} places that no rule predicts (output, block, element), "
        f"first {unexpected[:8]}"
    )


class _WrongAsPredicted(AssertionError):
    """The reference and the ttnn host codec differ on each element that one rule predicts."""


@pytest.mark.xfail(
    raises=_WrongAsPredicted,
    strict=True,
    reason="design Q1: the ttnn host MX codec differs from tt-regen-spec 4.2 by this rule",
)
@pytest.mark.parametrize("rule", tuple(_KNOWN_DIFFERENCES))
def test_ttnn_codec_differs(rule):
    """S0-C4, the predicted differences: the codecs differ on each element that the rule predicts. A set that
    shrinks (a fixed or a moved difference) fails; test_agree_* fail on a difference that no rule predicts."""
    count, what = _KNOWN_DIFFERENCES[rule]
    compare = _encode_comparison if rule in _ENCODE_RULES else _decode_comparison
    predicted, gone = 0, []
    for fmt in _ALL:
        rules, measured = compare(fmt)
        mine = [k for k, r in rules.items() if r == rule]
        predicted += len(mine)
        gone += [(fmt,) + k for k in mine if k not in measured]
    assert predicted == count, f"{rule}: the rule predicts {predicted} elements, not {count}"
    assert not gone, f"{rule}: {len(gone)} of {count} predicted differences are gone, first {sorted(gone)[:8]}"
    raise _WrongAsPredicted(f"{rule}: the codecs differ at the {count} predicted elements: {what}")
