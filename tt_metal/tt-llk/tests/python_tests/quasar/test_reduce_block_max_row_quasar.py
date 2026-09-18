# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Standalone tt-llk test for the experimental Quasar block reduce_max_row kernel. A block of
# `block_ct_dim` operand tiles (laid out along the width dimension) is row-max reduced into a single
# result tile. Drives the runtime-block_ct_dim LLK lib path (block_ct_dim == TILE_COUNT).

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import ReduceGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    ReduceDimension,
    ReducePool,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test

# Keep 3 as an example of an odd block_ct_dim.
BLOCK_CT_DIMS = [1, 2, 3, 4, 8, 16]

TILE_DIMENSIONS = [(32, 32), (16, 32), (8, 32), (4, 32)]


@pytest.mark.quasar
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float16]),
    block_ct_dim=BLOCK_CT_DIMS,
    tile_dimensions=TILE_DIMENSIONS,
    dest_sync_mode=[DestSync.Half, DestSync.Full],
)
def test_reduce_block_max_row_quasar(
    formats,
    block_ct_dim,
    tile_dimensions,
    dest_sync_mode,
):
    tile_shape = construct_tile_shape(tile_dimensions)

    # DIAGNOSTIC stimulus: each of the block_ct_dim tiles is a distinct ASCENDING constant
    # (tile k = 0.1*(k+1)). Constant tiles are layout-invariant, so no tilize is needed.
    tile_size = tile_shape.total_tile_size()
    src_A = torch.cat(
        [torch.full((tile_size,), 0.1 * (k + 1)) for k in range(block_ct_dim)]
    )

    # Scaler tile: 1.0 (identity for MAX pool), resident in F0.
    src_B = torch.full((tile_shape.total_tile_size(),), 1)

    # Golden: row-max reduce of every tile, then MAX-accumulate across the block into one result tile
    # (reduce_to_one=True) -- exactly what the block kernel produces.
    generate_golden = get_golden_generator(ReduceGolden)
    golden_tensor = generate_golden(
        src_A,
        ReduceDimension.Row,
        ReducePool.Max,
        formats.output_format,
        tile_cnt=block_ct_dim,
        reduce_to_one=True,
        tile_shape=tile_shape,
        input_format=formats.input_format,
    )

    configuration = TestConfig(
        "sources/quasar/reduce_block_max_row_quasar_test.cpp",
        formats,
        templates=[
            UNPACKER_ENGINE_SEL(),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
            DEST_SYNC(dest_sync_mode),
        ],
        runtimes=[
            TILE_COUNT(block_ct_dim),
            TEST_FACE_DIMS(tile_shape.face_r_dim, tile_shape.face_c_dim),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
            NUM_FACES(tile_shape.total_num_faces()),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=block_ct_dim,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=tile_shape.total_num_faces(),
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
        disable_format_inference=False,
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
        print_errors=True,
    ), "Assert against golden failed"
