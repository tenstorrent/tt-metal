# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    DataCopyGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    PerfRunType,
    Transpose,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)
from helpers.utils import passed_test, result_tensor

UNPACK_TRANSPOSE_SWEEP = {
    "formats": input_output_formats(
        [DataFormat.Bfp8_b, DataFormat.Float16_b],
    ),
    "unpack_transpose_faces": [Transpose.No, Transpose.Yes],
    "unpack_transpose_within_face": [Transpose.No, Transpose.Yes],
    "input_dimensions": [[128, 128]],
}


def _skip_unsupported(
    unpack_transpose_faces,
    unpack_transpose_within_face,
):
    if (
        unpack_transpose_faces == Transpose.No
        and unpack_transpose_within_face == Transpose.No
    ):
        pytest.skip("At least one unpack transpose mode must be enabled")


def _transpose_golden(
    src_A,
    formats,
    tile_count,
    input_dimensions,
    unpack_transpose_faces,
    unpack_transpose_within_face,
):
    transpose = get_golden_generator(TransposeGolden)
    golden = src_A
    transpose_format = formats.input_format

    if unpack_transpose_faces == Transpose.Yes:
        golden = transpose.transpose_faces_multi_tile(
            golden,
            transpose_format,
            num_tiles=tile_count,
            input_dimensions=tuple(input_dimensions),
        )
        if transpose_format.is_block_float():
            transpose_format = DataFormat.Float16_b

    if unpack_transpose_within_face == Transpose.Yes:
        golden = transpose.transpose_within_faces_multi_tile(
            golden,
            transpose_format,
            num_tiles=tile_count,
            input_dimensions=tuple(input_dimensions),
        )

    datacopy = get_golden_generator(DataCopyGolden)
    return datacopy(
        golden,
        formats.output_format,
        input_dimensions=input_dimensions,
    )


def _run_unpack_transpose(
    formats,
    unpack_transpose_faces,
    unpack_transpose_within_face,
    input_dimensions,
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    _skip_unsupported(
        unpack_transpose_faces,
        unpack_transpose_within_face,
    )

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")
    if run_types is None:
        run_types = [PerfRunType.L1_TO_L1]

    src_A = src_B = None
    golden_tensor = None
    if not is_perf:
        src_A, tile_count, src_B, tile_count_B = generate_stimuli(
            stimuli_format_A=formats.input_format,
            input_dimensions_A=input_dimensions,
            stimuli_format_B=formats.input_format,
            input_dimensions_B=input_dimensions,
        )
        golden_tensor = _transpose_golden(
            src_A,
            formats,
            tile_count,
            input_dimensions,
            unpack_transpose_faces,
            unpack_transpose_within_face,
        )
    else:
        tile_count = (
            input_dimensions[0]
            * input_dimensions[1]
            // (TILE_DIMENSIONS[0] * TILE_DIMENSIONS[1])
        )
        tile_count_B = tile_count

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        DestAccumulation.No,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
    )

    test_config_kwargs = {
        "test_name": "sources/unpack_transpose_test.cpp",
        "formats": formats,
        "templates": [],
        "runtimes": [
            TILE_COUNT(tile_count),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            UNPACK_TRANS_FACES(unpack_transpose_faces),
            UNPACK_TRANS_WITHIN_FACE(unpack_transpose_within_face),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count_B,
            tile_count_res=tile_count,
        ),
        "dest_acc": DestAccumulation.No,
        "unpack_to_dest": False,
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )
    if is_perf:
        configuration.run(perf_report)
        return

    result = configuration.run().result
    result = result_tensor(result, formats.output_format, golden_tensor)
    assert passed_test(golden_tensor, result, formats.output_format)


@parametrize(**UNPACK_TRANSPOSE_SWEEP)
def test_unpack_transpose(
    formats,
    unpack_transpose_faces,
    unpack_transpose_within_face,
    input_dimensions,
):
    _run_unpack_transpose(
        formats,
        unpack_transpose_faces,
        unpack_transpose_within_face,
        input_dimensions,
    )
