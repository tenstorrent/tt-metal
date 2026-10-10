# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import UntilizeGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TEST_FACE_DIMS,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test

# Even section count: DestSync.Half packs banks 0 -> 1 -> 0 -> 1 and leaves SRC_ADDR_OFFSET on bank 0 for later tests.
NUM_SECTIONS = 4
BLOCK_CT_DIM = 1

SEMAPHORE_SYNC_FORMATS = input_output_formats(
    [DataFormat.Float16_b, DataFormat.Float16], same=True
)


def run_pack_untilize_semaphore_sync(
    formats, dest_acc, dest_sync, tile_dimensions, rows_per_section=1
):
    tile_shape = construct_tile_shape(tile_dimensions)
    input_dimensions = [
        NUM_SECTIONS * rows_per_section * tile_shape.total_row_dim(),
        BLOCK_CT_DIM * tile_shape.total_col_dim(),
    ]

    sequential_spec = StimuliSpec.sequential()
    src_A, tile_cnt, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
        spec_A=sequential_spec,
        spec_B=sequential_spec,
    )

    generate_golden = get_golden_generator(UntilizeGolden)
    golden_tensor = generate_golden(
        src_A,
        formats.output_format,
        input_dimensions,
        input_format=formats.input_format,
        tile_dimensions=tile_dimensions,
    )

    num_faces = tile_shape.total_num_faces()

    configuration = TestConfig(
        "sources/quasar/pack_untilize_semaphore_sync_quasar_test.cpp",
        formats,
        templates=[
            generate_input_dim(
                input_dimensions,
                input_dimensions,
                block_ct_dim=BLOCK_CT_DIM,
                block_rt_dim=rows_per_section,
                tile_dimensions=tile_dimensions,
            ),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[
            TEST_FACE_DIMS(tile_shape.face_r_dim),
            NUM_FACES(num_faces),
            TILE_COUNT(tile_cnt),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=tile_cnt,
            num_faces=num_faces,
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        tile_shape=tile_shape,
    ), "Assert against golden failed"


@pytest.mark.quasar
@parametrize(
    formats=SEMAPHORE_SYNC_FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=[DestSync.Half, DestSync.Full],
    tile_dimensions=[(1, 32), (2, 32)],
    rows_per_section=[1, 2],
)
def test_pack_untilize_semaphore_sync_strided_quasar(
    formats, dest_acc, dest_sync, tile_dimensions, rows_per_section
):
    run_pack_untilize_semaphore_sync(
        formats, dest_acc, dest_sync, tile_dimensions, rows_per_section
    )


@pytest.mark.quasar
@parametrize(
    formats=SEMAPHORE_SYNC_FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=[DestSync.Half, DestSync.Full],
    rows_per_section=[1, 2],
)
def test_pack_untilize_semaphore_sync_contiguous_quasar(
    formats, dest_acc, dest_sync, rows_per_section
):
    run_pack_untilize_semaphore_sync(
        formats, dest_acc, dest_sync, (32, 32), rows_per_section
    )
