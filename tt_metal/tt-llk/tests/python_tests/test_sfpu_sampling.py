# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Sampling SFPU helpers test.

Covers every entry point of
hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sampling.h:

  recip_scalar<legacy_compat>          1/x                    rows 0-3
  clamp_max_scalar(max)                min(x, max)            rows 0-3
  mul_unary_scalar_first_column(k)     x * k                  rows 0-15
  binary_comp_first_column<le|lt|ge>   (in0 OP in1) ? 1 : 0   rows 0-15
  add/sub/mul_binary_first_column      in0 +|-|* in1          rows 0-15

One SFPLOAD/SFPSTORE covers 4 DEST rows x 8 of a face's 16 columns, and the
"first column" helpers step +4 address units per iteration, so they walk rows 0-15
of face 0 but only half its columns -- the callers only ever read column 0. Address
+0 selects the even columns (cf. COL_REDUCE_COLUMN_OFFSETS in
llk_sfpu/ckernel_sfpu_reduce.h), which is what assert_even_columns pins.
assert_row_multiset checks the same rows without assuming that mapping, so if it
passes and assert_even_columns does not, flip TOUCHED_COLUMNS. Inputs are
column-uniform per row to make that split possible.

SFPSTORE into a 16-bit DEST truncates rather than rounds, and only recip_scalar
compensates (convert<vFloat16b>(Nearest), and only when !(DST_ACCUM_MODE ||
APPROX)), so the golden models the store per op and per DEST width. dest_acc is
swept because it changes both that and the reciprocal's iteration count. Formats
are same-in-same-out only: these helpers never touch srcA, so mixed pairs would
only re-test unpack/pack. Float32 with dest_acc=No is skipped, as elsewhere in the
SFPU suites.
"""

import struct

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    SAMPLING_LEGACY_COMPAT,
    SAMPLING_OP,
    SFPU_UNARY_SCALAR,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True)

FACE_DIM = 16
NUM_TILES = 3  # in0, in1, zero-initialised output

# Columns of face 0 the helpers write.
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

# recip_scalar is the only helper that converts before storing, so it is the only
# one that rounds rather than truncates on a 16-bit DEST.
ROUND_TO_NEAREST_OPS = {"recip_scalar"}

# The reciprocal is iterative, so it never compares bit-exactly.
APPROXIMATE_OPS = {"recip_scalar"}
RECIP_REL_TOL = 2e-2

# Sizes the tolerance for the one conversion the golden does not model: an fp32 DEST
# packed down to bf16.
BF16_ULP_REL = 2**-8
ULP_ALLOWANCE = 2


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _dest_is_fp32(dest_acc: DestAccumulation) -> bool:
    return dest_acc == DestAccumulation.Yes


def _bf16_truncate(values: torch.Tensor) -> torch.Tensor:
    """SFPSTORE into a 16-bit DEST truncates; clearing the low 16 bits does the same."""
    return (
        (values.to(torch.float32).contiguous().view(torch.int32) & ~0xFFFF)
        .view(torch.float32)
        .clone()
    )


def _to_dest(values: torch.Tensor, dest_acc: DestAccumulation) -> torch.Tensor:
    """The value as it sits in DEST after unpack + datacopy."""
    if _dest_is_fp32(dest_acc):
        return values.to(torch.float32)
    return values.to(torch.bfloat16).to(torch.float32)


def _sfpu_store(
    values: torch.Tensor, op: str, dest_acc: DestAccumulation
) -> torch.Tensor:
    """What the kernel's store leaves in DEST."""
    if _dest_is_fp32(dest_acc):
        return values.to(torch.float32)  # the whole fp32 LReg lands in DEST
    if op in ROUND_TO_NEAREST_OPS:
        return values.to(torch.bfloat16).to(torch.float32)
    return _bf16_truncate(values)


def _rel_tol(op: str, dest_acc: DestAccumulation, out_format: DataFormat) -> float:
    """Zero (bit-exact) unless the packer has to convert an fp32 DEST down to bf16.

    Its rounding mode is not something this test should assert, so allow a couple of
    bf16 ulp there -- still much tighter than the 5% in helpers/utils.py.
    """
    tol = 0.0
    if _dest_is_fp32(dest_acc) and out_format != DataFormat.Float32:
        tol = ULP_ALLOWANCE * BF16_ULP_REL

    if op in APPROXIMATE_OPS:
        tol = max(tol, RECIP_REL_TOL)
    return tol


def _matches(got: float, want: float, tol: float) -> bool:
    if tol == 0.0:
        return got == want
    return abs(got - want) <= tol * abs(want) + 1e-6


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


def _bf16_row_values() -> torch.Tensor:
    """One value per row, away from 0 (reciprocal) and straddling CLAMP_MAX.

    Rounded to bf16 for every swept format so the arithmetic stays exact in fp32 and
    the golden never has to guess how the SFPU rounds fp32 results.
    """
    return (
        torch.empty(TILE_DIM, dtype=torch.float32)
        .uniform_(0.25, 2.0)
        .to(torch.bfloat16)
        .to(torch.float32)
    )


def _column_uniform_tile(row_values: torch.Tensor, torch_format) -> torch.Tensor:
    """A [32, 32] tile whose every row is one repeated value (row-major, untilized)."""
    return (
        row_values.reshape(TILE_DIM, 1)
        .expand(TILE_DIM, TILE_DIM)
        .to(torch_format)
        .clone()
    )


def assert_row_multiset(
    result_face, expected_transformed, expected_untouched, rows, op, tol
):
    """Checks each walked row without assuming which half of its columns was hit."""
    for row in range(rows):
        got = sorted(result_face[row, :FACE_DIM].tolist())
        want = sorted(
            [expected_transformed[row].item()] * len(TOUCHED_COLUMNS)
            + [expected_untouched[row].item()] * len(UNTOUCHED_COLUMNS)
        )
        ok = len(got) == len(want) and all(
            _matches(g, w, tol) for g, w in zip(got, want)
        )
        assert ok, (
            f"{op}: face-0 row {row} contents wrong regardless of column mapping.\n"
            f"  got  {got}\n  want {want}"
        )


def assert_even_columns(
    result_face, expected_transformed, expected_untouched, rows, op, tol
):
    """Pins the even-column reading of the SFPU DEST address."""
    for row in range(rows):
        for col in TOUCHED_COLUMNS:
            got = result_face[row, col].item()
            want = expected_transformed[row].item()
            assert _matches(got, want, tol), (
                f"{op}: face-0 [{row}, {col}] (an even column) should be transformed: "
                f"got {got}, want {want}"
            )
        for col in UNTOUCHED_COLUMNS:
            got = result_face[row, col].item()
            want = expected_untouched[row].item()
            assert _matches(got, want, tol), (
                f"{op}: face-0 [{row}, {col}] (an odd column) should be untouched: "
                f"got {got}, want {want}"
            )


@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    op=list(SAMPLING_OPS.keys()),
    legacy_compat=[True, False],
)
def test_sfpu_sampling(formats, dest_acc, op, legacy_compat):
    is_binary, rows_walked = SAMPLING_OPS[op]

    if op != "recip_scalar" and not legacy_compat:
        pytest.skip("legacy_compat only applies to recip_scalar")

    if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")

    torch.manual_seed(0)

    torch_format = format_dict[formats.input_format]
    tol = _rel_tol(op, dest_acc, formats.output_format)

    in0_rows = _bf16_row_values()
    in1_rows = _bf16_row_values()
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
        # A 32-bit input goes straight to DEST rather than through srcA.
        unpack_to_dest=formats.input_format.is_32_bit(),
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

    # Input at DEST precision, the op in fp32, then the store.
    in0_ref = _to_dest(in0_tile[:, 0], dest_acc)
    in1_ref = _to_dest(in1_tile[:, 0], dest_acc)
    transformed = _sfpu_store(_apply_op(op, in0_ref, in1_ref), op, dest_acc)

    # Tiles the op does not write still hold the datacopied input.
    in0_dest_tile = _to_dest(in0_tile, dest_acc)
    in1_dest_tile = _to_dest(in1_tile, dest_acc)

    if is_binary:
        # The op only writes tile 2, whose other lanes keep the zero background.
        assert passed_test(
            in0_dest_tile.flatten(), result_tiles[0].flatten(), formats.output_format
        ), f"{op}: DEST tile 0 (in0) was modified"
        assert passed_test(
            in1_dest_tile.flatten(), result_tiles[1].flatten(), formats.output_format
        ), f"{op}: DEST tile 1 (in1) was modified"
        out_face = result_tiles[2]
        untouched = torch.zeros(TILE_DIM, dtype=torch.float32)
    else:
        # In-place on tile 0; tiles 1 and 2 must be untouched.
        assert passed_test(
            in1_dest_tile.flatten(), result_tiles[1].flatten(), formats.output_format
        ), f"{op}: DEST tile 1 was modified by an in-place unary op"
        assert passed_test(
            zero_tile.flatten(), result_tiles[2].flatten(), formats.output_format
        ), f"{op}: DEST tile 2 was modified by an in-place unary op"
        out_face = result_tiles[0]
        untouched = in0_ref

    assert_row_multiset(out_face, transformed, untouched, rows_walked, op, tol)
    assert_even_columns(out_face, transformed, untouched, rows_walked, op, tol)

    # Rows the helper never walks. Compared exactly: these lanes are still the
    # datacopied input, which round-trips losslessly for every format combination.
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
