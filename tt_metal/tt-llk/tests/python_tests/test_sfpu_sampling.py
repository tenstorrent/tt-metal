# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Sampling SFPU helpers test.

Covers every entry point of
hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sampling.h:

  recip_scalar<legacy_compat>          1/x                     rows 0-3
  clamp_max_scalar(max)                min(x, max)             rows 0-3
  mul_unary_scalar_first_column(k)     x * k                    rows 0-15
  binary_comp_first_column<le|lt|ge>   (in0 OP in1) ? 1 : 0     rows 0-15
  add/sub/mul_binary_first_column      in0 +|-|* in1            rows 0-15

DEST layout the kernels imply
----------------------------
One SFPLOAD/SFPSTORE moves 4 DEST rows x 8 of a face's 16 columns, and each
helper advances `dst_reg += 2` per iteration (= +4 DEST address units = the next
4-row band). So a "first column" helper walks rows 0..15 of face 0 but only ever
touches *half* the columns of each row -- the callers only read column 0, hence
the name. `recip_scalar` / `clamp_max_scalar` do a single slot, so rows 0-3 only.

Which half is a property of the SFPU DEST addressing: address +0 selects a face's
even columns and +2 its odd columns (cf. COL_REDUCE_ODD_COLUMNS /
COL_REDUCE_COLUMN_OFFSETS "even, odd, even, odd" in llk_sfpu/ckernel_sfpu_reduce.h,
and the expert numbering `_topk_moe_generate_indices_` produces, which is only
consecutive under the interleaved reading). These helpers load at +0, so they hit
the EVEN columns.

That assumption is asserted separately from the arithmetic, so a failure is
self-diagnosing:
  * `assert_row_multiset` is column-mapping independent -- with a
    column-uniform input row, "8 lanes transformed, 8 lanes untouched" holds
    whichever half the hardware picks. If this fails, the op's math is wrong.
  * `assert_even_columns` pins the even-column reading. If the multiset check
    passes and only this fails, the mapping is first-half/second-half instead and
    only TOUCHED_COLUMNS below needs flipping.

Inputs are column-uniform inside each row (all 16 columns of a face row carry the
same value) precisely so the first check is possible.

Store rounding
--------------
LRegs hold fp32 and SFPSTORE into a 16-bit DEST TRUNCATES rather than rounds, so
the golden has to model each helper's actual store. Only
`calculate_sampling_recip_scalar` compensates (an explicit
`convert<vFloat16b>(out, RoundMode::Nearest)` on the !(DST_ACCUM_MODE || APPROX)
path); every arithmetic helper stores the raw fp32 result and so loses up to one
bf16 ulp versus the correctly-rounded value. That is why ROUND_TO_NEAREST_OPS
exists. Whether the arithmetic helpers *should* round is a question for the kernel
author -- this test pins the current behaviour so a change is caught rather than
silently accepted. Because the modelled result is exact, the comparisons are
bit-exact for everything except the iterative reciprocal.

dest_acc=Yes is not swept: the header is written against a 16-bit DEST
(`DST_ACCUM_MODE` only selects the reciprocal's iteration count, and
`legacy_compat` is documented as having to stay bit-identical on the bf16 path).
"""

import struct

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    SAMPLING_LEGACY_COMPAT,
    SAMPLING_OP,
    SFPU_UNARY_SCALAR,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

FACE_DIM = 16
NUM_TILES = 3  # in0, in1, zero-initialised output

# Columns of face 0 the helpers write (see module docstring).
TOUCHED_COLUMNS = list(range(0, FACE_DIM, 2))
UNTOUCHED_COLUMNS = list(range(1, FACE_DIM, 2))

# Rows of face 0 each helper walks.
SINGLE_SLOT_ROWS = 4
FIRST_COLUMN_ROWS = 16

CLAMP_MAX = 0.75
MUL_SCALAR = 3.0

# op name -> (is_binary, rows walked)
SAMPLING_OPS = {
    "recip_scalar": (False, SINGLE_SLOT_ROWS),
    "clamp_max_scalar": (False, SINGLE_SLOT_ROWS),
    "mul_unary_scalar": (False, FIRST_COLUMN_ROWS),
    "le": (True, FIRST_COLUMN_ROWS),
    "lt": (True, FIRST_COLUMN_ROWS),
    "ge": (True, FIRST_COLUMN_ROWS),
    "add": (True, FIRST_COLUMN_ROWS),
    "sub": (True, FIRST_COLUMN_ROWS),
    "mul": (True, FIRST_COLUMN_ROWS),
}

# Ops whose kernel explicitly rounds before storing. LRegs hold fp32, and SFPSTORE
# into a 16-bit DEST *truncates* rather than rounds -- the same hazard
# ckernel_sfpu_exp.h documents ("SFPSTORE will truncate it. This can reduce
# accuracy ... To avoid this issue, we explicitly convert to bfloat16 using
# round-to-nearest"). Of the sampling helpers only calculate_sampling_recip_scalar
# compensates, with convert<vFloat16b>(out, RoundMode::Nearest) on the
# !(DST_ACCUM_MODE || APPROX) path. Every arithmetic helper stores the raw fp32
# result and therefore truncates, losing up to one bf16 ulp. The golden models each
# op's actual store mode; if a helper later gains a round-to-nearest convert,
# ROUND_TO_NEAREST_OPS is the one place to update.
ROUND_TO_NEAREST_OPS = {"recip_scalar"}

# Ops whose result is not exactly reproducible in fp32 and so cannot be compared
# bit-exactly: the reciprocal is iterative (3 Newton steps under legacy_compat,
# 1 otherwise). Everything else is exact -- sums and products of two 8-significand-
# bit bf16 values need at most ~16 mantissa bits, so fp32 computes them exactly and
# the only lossy step is the modelled store.
APPROXIMATE_OPS = {"recip_scalar"}
RECIP_REL_TOL = 2e-2


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _bf16_truncate(values: torch.Tensor) -> torch.Tensor:
    """Model SFPSTORE of an fp32 LReg into a 16-bit DEST: truncate, do not round.

    Clearing the low 16 bits of the IEEE-754 pattern drops mantissa bits without
    touching sign or exponent, i.e. it moves toward zero for either sign -- exactly
    what the hardware store does.
    """
    return (
        (values.to(torch.float32).contiguous().view(torch.int32) & ~0xFFFF)
        .view(torch.float32)
        .clone()
    )


def _store_to_dest(values: torch.Tensor, op: str) -> torch.Tensor:
    """Apply the store rounding this op's kernel actually performs."""
    if op in ROUND_TO_NEAREST_OPS:
        return values.to(torch.bfloat16).to(torch.float32)
    return _bf16_truncate(values)


def _matches(got: float, want: float, op: str) -> bool:
    """Exact for the arithmetic ops; relative tolerance for the iterative reciprocal."""
    if op in APPROXIMATE_OPS:
        return abs(got - want) <= RECIP_REL_TOL * max(abs(want), 1.0)
    return got == want


def _apply_op(op: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Scalar/elementwise reference for one sampling op, in fp32."""
    if op == "recip_scalar":
        return 1.0 / a
    if op == "clamp_max_scalar":
        return torch.minimum(a, torch.full_like(a, CLAMP_MAX))
    if op == "mul_unary_scalar":
        return a * MUL_SCALAR
    if op == "le":
        return (a <= b).to(torch.float32)
    if op == "lt":
        return (a < b).to(torch.float32)
    if op == "ge":
        return (a >= b).to(torch.float32)
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    raise ValueError(f"unknown sampling op {op}")


def _column_uniform_tile(row_values: torch.Tensor, torch_format) -> torch.Tensor:
    """A [32, 32] tile whose every row is one repeated value (row-major, untilized)."""
    return (
        row_values.reshape(TILE_DIM, 1)
        .expand(TILE_DIM, TILE_DIM)
        .to(torch_format)
        .clone()
    )


def assert_row_multiset(
    result_face, expected_transformed, expected_untouched, rows, op
):
    """Column-mapping independent check (see module docstring)."""
    for row in range(rows):
        got = sorted(result_face[row, :FACE_DIM].tolist())
        want = sorted(
            [expected_transformed[row].item()] * len(TOUCHED_COLUMNS)
            + [expected_untouched[row].item()] * len(UNTOUCHED_COLUMNS)
        )
        ok = len(got) == len(want) and all(
            _matches(g, w, op) for g, w in zip(got, want)
        )
        assert ok, (
            f"{op}: face-0 row {row} contents wrong regardless of column mapping.\n"
            f"  got  {got}\n  want {want}"
        )


def assert_even_columns(
    result_face, expected_transformed, expected_untouched, rows, op
):
    """Pins the even-column reading of the SFPU DEST address (see module docstring)."""
    for row in range(rows):
        for col in TOUCHED_COLUMNS:
            got = result_face[row, col].item()
            want = expected_transformed[row].item()
            assert _matches(got, want, op), (
                f"{op}: face-0 [{row}, {col}] (an even column) should be transformed: "
                f"got {got}, want {want}"
            )
        for col in UNTOUCHED_COLUMNS:
            got = result_face[row, col].item()
            want = expected_untouched[row].item()
            assert _matches(got, want, op), (
                f"{op}: face-0 [{row}, {col}] (an odd column) should be untouched: "
                f"got {got}, want {want}"
            )


@parametrize(
    dest_acc=[DestAccumulation.No],
    op=list(SAMPLING_OPS.keys()),
    legacy_compat=[True, False],
)
def test_sfpu_sampling(dest_acc, op, legacy_compat):
    is_binary, rows_walked = SAMPLING_OPS[op]

    if op != "recip_scalar" and not legacy_compat:
        # legacy_compat only parameterises calculate_sampling_recip_scalar; skip the
        # duplicate build for every other op.
        pytest.skip("legacy_compat only applies to recip_scalar")

    torch.manual_seed(0)

    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    torch_format = format_dict[formats.input_format]

    # One value per row, repeated across the row. Kept away from 0 (reciprocal) and
    # spread across CLAMP_MAX so the clamp actually triggers on some rows.
    in0_rows = torch.empty(TILE_DIM, dtype=torch.float32).uniform_(0.25, 2.0)
    in1_rows = torch.empty(TILE_DIM, dtype=torch.float32).uniform_(0.25, 2.0)
    # Force a few exact ties so le/lt/ge are distinguished from each other.
    in1_rows[0:4] = in0_rows[0:4]

    in0_tile = _column_uniform_tile(in0_rows, torch_format)
    in1_tile = _column_uniform_tile(in1_rows, torch_format)
    zero_tile = torch.zeros((TILE_DIM, TILE_DIM), dtype=torch_format)

    src_A = torch.cat(
        [
            tilize_block(
                t.flatten(), [TILE_DIM, TILE_DIM], stimuli_format=formats.input_format
            ).flatten()
            for t in (in0_tile, in1_tile, zero_tile)
        ]
    )
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch_format)

    scalar_bits = _f32_bits(CLAMP_MAX if op == "clamp_max_scalar" else MUL_SCALAR)

    configuration = TestConfig(
        "sources/sfpu_sampling_test.cpp",
        formats,
        templates=[
            SAMPLING_OP(op=op),
            SAMPLING_LEGACY_COMPAT(legacy_compat=legacy_compat),
            SFPU_UNARY_SCALAR(value_bits=scalar_bits),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=NUM_TILES,
            tile_count_B=1,
            tile_count_res=NUM_TILES,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    res_from_L1 = configuration.run().result
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    result_tiles = [
        untilize_block(
            res_tensor[t * ELEMENTS_PER_TILE : (t + 1) * ELEMENTS_PER_TILE],
            formats.output_format,
            [TILE_DIM, TILE_DIM],
        ).reshape(TILE_DIM, TILE_DIM)
        for t in range(NUM_TILES)
    ]

    # bf16 rounds the inputs, so compute the reference from the rounded values, then
    # push it through the store rounding this op's kernel performs (see
    # ROUND_TO_NEAREST_OPS).
    in0_ref = in0_rows.to(torch_format).to(torch.float32)
    in1_ref = in1_rows.to(torch_format).to(torch.float32)
    transformed = _store_to_dest(_apply_op(op, in0_ref, in1_ref), op)

    if is_binary:
        # in0 / in1 tiles must come back untouched; the op only writes tile 2, whose
        # untouched lanes keep the zero background that was copied in.
        assert passed_test(
            in0_tile.flatten(), result_tiles[0].flatten(), formats.output_format
        ), f"{op}: DEST tile 0 (in0) was modified"
        assert passed_test(
            in1_tile.flatten(), result_tiles[1].flatten(), formats.output_format
        ), f"{op}: DEST tile 1 (in1) was modified"
        out_face = result_tiles[2]
        untouched = torch.zeros(TILE_DIM, dtype=torch.float32)
    else:
        # In-place on tile 0; tiles 1 and 2 must be untouched.
        assert passed_test(
            in1_tile.flatten(), result_tiles[1].flatten(), formats.output_format
        ), f"{op}: DEST tile 1 was modified by an in-place unary op"
        assert passed_test(
            zero_tile.flatten(), result_tiles[2].flatten(), formats.output_format
        ), f"{op}: DEST tile 2 was modified by an in-place unary op"
        out_face = result_tiles[0]
        untouched = in0_ref

    assert_row_multiset(out_face, transformed, untouched, rows_walked, op)
    assert_even_columns(out_face, transformed, untouched, rows_walked, op)

    # Rows the helper never walks must be untouched. The input tiles are
    # column-uniform across all 32 columns, so the expected value is the same for
    # face 0 and face 1.
    for row in range(rows_walked, TILE_DIM):
        for col in range(TILE_DIM):
            got = out_face[row, col].item()
            want = untouched[row].item()
            assert got == want, (
                f"{op}: row {row} is outside the walked range but changed at "
                f"column {col}: got {got}, want {want}"
            )
    # Face 1 (columns 16-31) is never addressed, even in the walked rows.
    for row in range(rows_walked):
        for col in range(FACE_DIM, TILE_DIM):
            got = out_face[row, col].item()
            want = untouched[row].item()
            assert got == want, (
                f"{op}: face 1 (columns 16-31) must be untouched, but "
                f"[{row}, {col}] holds {got}, want {want}"
            )
