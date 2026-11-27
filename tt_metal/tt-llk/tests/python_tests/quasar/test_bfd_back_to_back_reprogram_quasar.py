# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Back-to-back re-programming of the fixed buffer-descriptor id of Unp0 and Pack0.

Each block programs the engine's descriptor BFD_REPROGRAMS times in a row (decoys pointing at other
blocks, last one at this block), inits, and executes at tile 0 right after the previous block's MOP.
A stale descriptor (TTSync not holding the CFG store) gives wrong tiles. The result region is
prefilled with a sentinel, so a tile that is never written also fails.
"""

import pytest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import DataCopyGolden, get_golden_generator
from helpers.llk_params import DataCopyType, DestAccumulation, DestSync
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    TEST_FACE_DIMS,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test, result_tensor

NUM_BLOCKS_PER_RUN = 8
TILE_DIMS = [32, 32]
FORMATS = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)


class SentinelStimuli(StimuliConfig):
    """Fill the result region with a sentinel after the stimuli are written, so a result tile
    the kernel never wrote (or wrote late) cannot match the golden by accident."""

    def write(self, location="0,0"):
        super().write(location)
        self.clear_result_buffer(location)


def _wrong_tiles(golden, result):
    gold = golden.flatten().reshape(-1, 1024).float()
    res = result.flatten().reshape(-1, 1024).float()
    bad = [i for i in range(res.shape[0]) if not (res[i] == gold[i]).all()]
    return f"wrong result tiles: {bad}"


@pytest.mark.quasar
@pytest.mark.parametrize("tiles_in_block", [4, 8])
def test_bfd_back_to_back_reprogram_quasar(tiles_in_block):
    input_dimensions = [
        TILE_DIMS[0],
        TILE_DIMS[1] * tiles_in_block * NUM_BLOCKS_PER_RUN,
    ]
    tile_shape = construct_tile_shape(TILE_DIMS)
    num_faces = tile_shape.total_num_faces()

    src_A, tile_cnt, _, _ = generate_stimuli(
        stimuli_format_A=FORMATS.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=FORMATS.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=TILE_DIMS,
    )
    golden_tensor = get_golden_generator(DataCopyGolden)(
        src_A,
        FORMATS.output_format,
        num_faces=num_faces,
        face_r_dim=tile_shape.face_r_dim,
        input_dimensions=input_dimensions,
        input_format=FORMATS.input_format,
        tile_shape=tile_shape,
    )

    configuration = TestConfig(
        "sources/quasar/bfd_back_to_back_reprogram_quasar_test.cpp",
        FORMATS,
        templates=[
            DATA_COPY_TYPE(DataCopyType.A2D),
            DEST_SYNC(DestSync.Half),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(tile_shape.face_r_dim),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
            DEST_INDEX(0),
            NUM_BLOCKS(NUM_BLOCKS_PER_RUN),
            NUM_TILES_IN_BLOCK(tiles_in_block),
            generate_input_dim(
                input_dimensions, input_dimensions, tile_dimensions=TILE_DIMS
            ),
        ],
        variant_stimuli=SentinelStimuli(
            src_A,
            FORMATS.input_format,
            src_A,  # buffer B is unused by the kernel
            FORMATS.input_format,
            FORMATS.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=tile_cnt,
            num_faces=num_faces,
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=TILE_DIMS,
            use_dense_tile_dimensions=True,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
    )

    res = configuration.run().result
    res_tensor = result_tensor(res, FORMATS.output_format, golden_tensor)
    assert passed_test(golden_tensor, res_tensor, FORMATS.output_format), _wrong_tiles(
        golden_tensor, res_tensor
    )
