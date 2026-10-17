# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""``_llk_unpack_AB_<ROW>`` with an explicit ``bcast_row_idx``.

One 32x32 tile, ``Res = A + row_bcast(B, idx)``. The broadcast row of a 32x32 tile spans two
faces (cols 0-15 in face 0/2, cols 16-31 in face 1/3), so row ``idx`` of the tile is row
``idx % 16`` of faces ``(0, 1)`` for ``idx < 16`` and of faces ``(2, 3)`` otherwise; every output
face repeats its column half of that row.

The guard cases pin the two checks in the ROW branch of ``_llk_unpack_AB_``: an index past the
tile (which otherwise silently reads a row of whatever follows the tile in L1) and a BFP operand B
with a non-zero index (the base-address shift cannot select a row of a shared-exponent tile). Both
rely on ``LLK_ASSERT`` being compiled in, which the harness does unless ``TT_LLK_DISABLE_ASSERTS``
is set.
"""

import os

import pytest
import torch
from helpers.device import LLKAssertException
from helpers.format_config import DataFormat
from helpers.golden_generators import EltwiseBinaryGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import BCAST_ROW_IDX, UNPACK_ISSUED
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

FACE_DIM = 16
FACE_SIZE = FACE_DIM * FACE_DIM
INPUT_DIMENSIONS = [32, 32]

ROW_BCAST_FORMATS = input_output_formats(
    [DataFormat.Float16_b, DataFormat.Bfp8_b], same=True
)


def row_broadcast_tile(tile_b: torch.Tensor, row_idx: int) -> torch.Tensor:
    """Tilized 32x32 tile whose every row is row ``row_idx`` of ``tile_b`` (tilized)."""
    first_face = 0 if row_idx < FACE_DIM else 2
    row_in_face = row_idx % FACE_DIM

    def face_row(face: int) -> torch.Tensor:
        start = face * FACE_SIZE + row_in_face * FACE_DIM
        return tile_b[start : start + FACE_DIM]

    left = face_row(first_face).repeat(FACE_DIM)
    right = face_row(first_face + 1).repeat(FACE_DIM)
    return torch.cat([left, right, left, right])


def _make_config(formats, bcast_row_idx):
    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=INPUT_DIMENSIONS,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=INPUT_DIMENSIONS,
    )
    src_A_tilized = tilize_block(
        src_A, dimensions=INPUT_DIMENSIONS, stimuli_format=formats.input_format
    ).flatten()
    src_B_tilized = tilize_block(
        src_B, dimensions=INPUT_DIMENSIONS, stimuli_format=formats.input_format
    ).flatten()

    configuration = TestConfig(
        "sources/unpack_AB_bcast_row_idx_test.cpp",
        formats,
        templates=[],
        runtimes=[BCAST_ROW_IDX(bcast_row_idx), UNPACK_ISSUED(0)],
        variant_stimuli=StimuliConfig(
            src_A_tilized,
            formats.input_format,
            src_B_tilized,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=DestAccumulation.No,
        unpack_to_dest=False,
    )
    return configuration, src_A_tilized, src_B_tilized


def valid_row_indices(formats):
    # The base-address row select cannot address a row of a shared-exponent (BFP) tile, so a BFP
    # operand B is only meaningful at bcast_row_idx == 0.
    if formats.input_format == DataFormat.Bfp8_b:
        return [0]
    return [0, 1, 15, 16, 31]


def rejected_row_indices(formats):
    if formats.input_format == DataFormat.Bfp8_b:
        return [5, 40]
    return [40]


@parametrize(
    formats=ROW_BCAST_FORMATS,
    bcast_row_idx=rejected_row_indices,
)
def test_unpack_AB_bcast_row_idx_rejected(formats, bcast_row_idx):
    """The unpacker must refuse an index it cannot address rather than read a wrong row."""
    if os.environ.get("TT_LLK_DISABLE_ASSERTS") == "1":
        pytest.skip("needs LLK_ASSERT compiled in")

    # Math and pack wait on the UNPACK_ISSUED gate, so when the unpacker stops in the assert they
    # are spinning on L1 (reset cleanly) rather than stalled on a Tensix instruction (not reset,
    # hangs the next kernel).
    configuration, _, _ = _make_config(formats, bcast_row_idx)
    with pytest.raises(LLKAssertException):  # allow-pytest.raises: no expect_error here
        configuration.run()


@parametrize(
    formats=ROW_BCAST_FORMATS,
    bcast_row_idx=valid_row_indices,
)
def test_unpack_AB_bcast_row_idx(formats, bcast_row_idx):
    configuration, src_A_tilized, src_B_tilized = _make_config(formats, bcast_row_idx)

    # The same 16-datum groups are quantized before and after the row select (each output face row
    # is one source face row), so the golden can quantize the broadcast operand itself.
    binary_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = binary_golden(
        MathOperation.Elwadd,
        src_A_tilized,
        row_broadcast_tile(src_B_tilized, bcast_row_idx),
        formats.output_format,
        MathFidelity.LoFi,
        input_format=formats.input_format,
        input_format_B=formats.input_format,
    )

    res_from_L1 = configuration.run().result
    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(golden_tensor, res_tensor, formats.output_format)
