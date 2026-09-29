# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bit-exact check of the Blackhole integer comparison kernels on a crafted int32 domain.

Covers, on Int32 input with two's-complement stimuli (the ttnn convention on Blackhole,
where the datacopy leaves the int32 bits in Dst untouched):

* the six comparisons to zero -- metal ``calculate_comp_int`` (``eqz_tile_int32`` ...);
* the six comparisons against a scalar -- metal ``calculate_comp_unary_int`` for eq/ne
  (``unary_eq_tile_int32`` / ``unary_ne_tile_int32``) and tt-llk
  ``_calculate_comp_unary_int_`` for gt/lt/ge/le (``unary_gt_tile_int32`` ...), for every
  scalar in ``_SCALARS``: both int32 extremes, both neighbours of zero, zero itself and the
  kernel's default;
* eqz/nez on UInt32 (``calculate_eqz_uint32`` / ``calculate_nez_uint32``) with values at
  and above 2^31, and on UInt16 (``calculate_comp_uint16``).

Every kernel here maps an element to 0 or 1, so the golden is a one-line integer compare,
evaluated on the host in int64 and compared with ``==``: there is no tolerance under which
a wrong lane is acceptable. The domain is a sample, not a sweep (2^32 is not enumerable),
built so that every representation boundary the kernels reason about is present in every
run: INT_MIN and INT_MAX and their neighbours, -1/0/1, the float-special bit patterns
(+/-inf, quiet NaN, which sfpi's ``vInt ==`` lowering compares through ``SFPLE`` on
Blackhole), the +/-2^30 midpoints, a dense band of small magnitudes either side of zero
and a fixed-seed fill across the full range.

The comparisons-to-zero were wrong at INT_MIN before their arithmetic rewrite
(``gtz(INT_MIN) == 1``, ``lez(INT_MIN) == 0``, because ``0 - INT_MIN`` wraps to INT_MIN);
that lane is pinned here.

Blackhole only: the Wormhole twins of these kernels read Dst as sign-magnitude (sfpi's
Wormhole default ``vInt`` layout converts on load) and are swept by
``test_vif_equiv_sweep.py`` under this harness's sign-magnitude default encoding.
"""

import numpy as np
import pytest
import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    FastMode,
    MathOperation,
    format_dict,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    SFPU_UNARY_COMP_INT_SCALAR,
    TILE_COUNT,
    DestSync,
    generate_input_dim,
)

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1

_ZERO_COMP_OPS = {
    "eqz": (MathOperation.EqualZero, lambda x: x == 0),
    "nez": (MathOperation.NotEqualZero, lambda x: x != 0),
    "ltz": (MathOperation.LessThanZero, lambda x: x < 0),
    "gtz": (MathOperation.GreaterThanZero, lambda x: x > 0),
    "lez": (MathOperation.LessThanEqualZero, lambda x: x <= 0),
    "gez": (MathOperation.GreaterThanEqualZero, lambda x: x >= 0),
}

_SCALAR_COMP_OPS = {
    "unary_eq": (MathOperation.UnaryEq, lambda x, s: x == s),
    "unary_ne": (MathOperation.UnaryNe, lambda x, s: x != s),
    "unary_gt": (MathOperation.UnaryGt, lambda x, s: x > s),
    "unary_lt": (MathOperation.UnaryLt, lambda x, s: x < s),
    "unary_ge": (MathOperation.UnaryGe, lambda x, s: x >= s),
    "unary_le": (MathOperation.UnaryLe, lambda x, s: x <= s),
}

# The kernel's fixed default (5) plus every scalar the sign split and the le/ge
# neighbour trick treat specially: INT_MAX (le answers all ones, INT_MAX + 1 would wrap),
# INT_MIN (ge likewise), the neighbours of zero and zero itself (the sign of the scalar
# selects the vector form), and a float-special bit pattern (+inf, 0x7F800000) to probe the
# eq/ne lowering on a pattern the SFPU's float compare would treat as non-finite.
_SCALARS = [INT32_MIN, INT32_MIN + 1, -1, 0, 1, 5, 0x7F800000, INT32_MAX - 1, INT32_MAX]

# Float-special bit patterns, as int32: +inf, +inf+1ulp, quiet NaN, INT_MAX (a NaN pattern),
# -inf, -NaN, and -1 (all ones, also a -NaN pattern).
_FLOAT_SPECIAL_PATTERNS = [
    0x7F800000,
    0x7F800001,
    0x7FC00000,
    0x7FFFFFFF,
    0xFF800000 - 2**32,
    0xFFC00000 - 2**32,
    -1,
]

_FACE_ELEMENTS = 16 * 16
# StimuliSpec.custom writes its values at the start of ONE face and zero-fills the rest,
# so the domain is handed over face by face: 256 values per face, 4 faces per tile.
_TILES = 4
_FACES = _TILES * 4
_VALUES = _FACES * _FACE_ELEMENTS


def _int32_domain() -> list[int]:
    """The crafted int32 probe set: boundaries, scalar neighbourhoods, dense band, fill."""
    special = [
        0,
        1,
        -1,
        2,
        -2,
        INT32_MAX,
        INT32_MAX - 1,
        INT32_MIN,
        INT32_MIN + 1,
        2**30,
        -(2**30),
        2**24,
        -(2**24),
        2**16,
        -(2**16),
        2**15,
        -(2**15),
        0xFFFF,
        -0xFFFF,
    ]
    special += _FLOAT_SPECIAL_PATTERNS
    # Both neighbours of every scalar, so each op is driven exactly at and either side of
    # its tie for every parametrised scalar.
    for s in _SCALARS:
        special += [v for v in (s - 1, s, s + 1) if INT32_MIN <= v <= INT32_MAX]
    special = list(dict.fromkeys(special))
    dense = [v for m in range(1, 513) for v in (m, -m)]
    rng = np.random.default_rng(20260928)
    remaining = _VALUES - len(special) - len(dense)
    assert remaining > 0
    fill = rng.integers(
        INT32_MIN, INT32_MAX, size=remaining, dtype=np.int64, endpoint=True
    )
    vals = special + dense + fill.tolist()
    assert len(vals) == _VALUES
    return vals


def _uint32_domain() -> list[int]:
    """UInt32 probe set: zero, one, the 2^31 boundary and above, all ones, and a fill."""
    special = [
        0,
        1,
        2,
        2**31 - 1,
        2**31,
        2**31 + 1,
        2**32 - 1,
        2**32 - 2,
        0x7F800000,
        0xFF800000,
    ]
    rng = np.random.default_rng(20260929)
    fill = rng.integers(
        0, 2**32 - 1, size=_VALUES - len(special), dtype=np.int64, endpoint=True
    )
    # Every 8th lane zero, so eqz/nez see both answers in every face.
    fill[::8] = 0
    return special + fill.tolist()


def _uint16_domain() -> list[int]:
    special = [0, 1, 2, 0x7FFF, 0x8000, 0x8001, 0xFFFF, 0xFFFE]
    rng = np.random.default_rng(20260930)
    fill = rng.integers(
        0, 0xFFFF, size=_VALUES - len(special), dtype=np.int64, endpoint=True
    )
    fill[::8] = 0
    return special + fill.tolist()


def _face_spec(vals: list[int]) -> StimuliSpec:
    return StimuliSpec.custom_faces(
        {f: vals[f * _FACE_ELEMENTS : (f + 1) * _FACE_ELEMENTS] for f in range(_FACES)}
    )


def _run(mathop, fmt: DataFormat, vals: list[int], scalar=None, twos_complement=False):
    """Drive one kernel over the value list; return (host inputs, results) as int64."""
    formats = InputOutputFormat(fmt, fmt)
    input_dimensions = [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1] * _TILES]
    # 32-bit inputs unpack straight into a 32-bit Dst. UInt16 keeps the 16-bit Dst ttnn runs
    # it with: the kernel's DataLayout::U16 loads/stores address Dst's 16-bit view, which in
    # 32-bit mode is not where the datacopy wrote and the packer reads.
    dest_acc = DestAccumulation.Yes if fmt.is_32_bit() else DestAccumulation.No

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=fmt,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=fmt,
        input_dimensions_B=input_dimensions,
        spec_A=_face_spec(vals),
    )
    assert tile_cnt_A == _TILES

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            APPROX_MODE(ApproximationMode.No),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(True),
            MATH_OP(mathop=mathop),
            # Only emitted when swept: sfpu_operations.h keys off #ifdef, and every other
            # unary test has to keep compiling without the macro.
            *([] if scalar is None else [SFPU_UNARY_COMP_INT_SCALAR(scalar)]),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            fmt,
            src_B,
            fmt,
            fmt,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            twos_complement=twos_complement,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=fmt.is_32_bit() and dest_acc == DestAccumulation.Yes,
    )
    res = configuration.run().result
    x = src_A.to(torch.int64).numpy()
    got = torch.tensor(res, dtype=format_dict[fmt]).to(torch.int64).numpy()
    return x, got


def _assert_exact(case: str, x: np.ndarray, got: np.ndarray, want: np.ndarray):
    assert (
        got.shape == want.shape
    ), f"{case}: {got.shape} results for {want.shape} inputs"
    diff = np.flatnonzero(got != want)
    if diff.size == 0:
        return
    detail = ", ".join(
        f"x={int(x[i])} ({int(x[i]) & 0xFFFFFFFF:#010x}) got={int(got[i])} want={int(want[i])}"
        for i in diff[:8]
    )
    raise AssertionError(
        f"{case}: {diff.size}/{got.size} lanes differ from the exact integer compare "
        f"(first {min(8, diff.size)}: {detail})"
    )


@blackhole_only
@pytest.mark.parametrize("op_name", list(_ZERO_COMP_OPS))
def test_comp_int32_zero(op_name):
    mathop, golden = _ZERO_COMP_OPS[op_name]
    vals = _int32_domain()
    x, got = _run(mathop, DataFormat.Int32, vals, twos_complement=True)
    # The lanes the rewrite exists for: both int32 extremes must have reached the device.
    assert INT32_MIN in x and INT32_MAX in x
    _assert_exact(f"{op_name}__int32", x, got, golden(x).astype(np.int64))


@blackhole_only
@pytest.mark.parametrize("scalar", _SCALARS)
@pytest.mark.parametrize("op_name", list(_SCALAR_COMP_OPS))
def test_comp_int32_scalar(op_name, scalar):
    mathop, golden = _SCALAR_COMP_OPS[op_name]
    vals = _int32_domain()
    x, got = _run(mathop, DataFormat.Int32, vals, scalar=scalar, twos_complement=True)
    assert INT32_MIN in x and INT32_MAX in x and scalar in x
    _assert_exact(
        f"{op_name}__int32__s={scalar}", x, got, golden(x, scalar).astype(np.int64)
    )


@blackhole_only
@pytest.mark.parametrize("op_name", ["eqz", "nez"])
def test_comp_uint32_zero(op_name):
    mathop, golden = _ZERO_COMP_OPS[op_name]
    vals = _uint32_domain()
    x, got = _run(mathop, DataFormat.UInt32, vals)
    assert 2**31 in x and 2**32 - 1 in x
    _assert_exact(f"{op_name}__uint32", x, got, golden(x).astype(np.int64))


@blackhole_only
@pytest.mark.parametrize("op_name", ["eqz", "nez"])
def test_comp_uint16_zero(op_name):
    mathop, golden = _ZERO_COMP_OPS[op_name]
    vals = _uint16_domain()
    x, got = _run(mathop, DataFormat.UInt16, vals)
    assert 0x8000 in x and 0xFFFF in x
    _assert_exact(f"{op_name}__uint16", x, got, golden(x).astype(np.int64))
