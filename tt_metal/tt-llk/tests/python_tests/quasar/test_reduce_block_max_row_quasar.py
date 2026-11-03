# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Standalone tt-llk test for the experimental Quasar block reduce_max_row kernel. A block of
# `block_ct_dim` operand tiles (laid out along the width dimension) is row-max reduced into a single
# result tile. Drives the runtime-block_ct_dim LLK lib path (block_ct_dim == TILE_COUNT).

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ReduceGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ReduceDimension,
    ReducePool,
)
from helpers.param_config import (
    DEST_DEPTH_BLOCK_COUNTS,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test, result_tensor

# Keep 3 as an example of an odd block_ct_dim.
BLOCK_CT_DIMS = [1, 2, 3, 4, 8, 16]

TILE_DIMENSIONS = [(32, 32), (16, 32), (8, 32), (4, 32)]


# Float32 L1 input unpacks to Float16_b on the 16-bit Dest this kernel runs on. Float32 -> Float32
# needs a 32-bit Dest, which the block max-row LLK rejects on Quasar (static_assert in
# llk_math_reduce_runtime_custom.h).
@pytest.mark.quasar
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b])
    + [InputOutputFormat(DataFormat.Float32, DataFormat.Float16_b)],
    block_ct_dim=runtime(BLOCK_CT_DIMS),
    tile_dimensions=runtime(TILE_DIMENSIONS),
    dest_sync_mode=[DestSync.Half, DestSync.Full],
)
def test_reduce_block_max_row_quasar(
    formats,
    block_ct_dim,
    tile_dimensions,
    dest_sync_mode,
    num_rows=1,
):
    """*num_rows* rows of block_ct_dim tiles, each reduced into its own result tile and Dest section."""
    tile_shape = construct_tile_shape(tile_dimensions)

    # DIAGNOSTIC stimulus: each of the block_ct_dim tiles is a distinct ASCENDING constant
    # (tile k = 0.1*(k+1), offset by the row index). Constant tiles are layout-invariant, so no
    # tilize is needed.
    tile_size = tile_shape.total_tile_size()
    row_inputs = [
        torch.cat(
            [torch.full((tile_size,), 0.1 * (k + 1) + row) for k in range(block_ct_dim)]
        )
        for row in range(num_rows)
    ]
    src_A = torch.cat(row_inputs)

    # Scaler tile: 1.0 (identity for MAX pool), resident in F0.
    src_B = torch.full((tile_shape.total_tile_size(),), 1)

    # Golden: row-max reduce of every tile, then MAX-accumulate across the block into one result tile
    # (reduce_to_one=True) -- exactly what the block kernel produces, once per row.
    generate_golden = get_golden_generator(ReduceGolden)
    golden_tensor = torch.cat(
        [
            generate_golden(
                row_input,
                ReduceDimension.Row,
                ReducePool.Max,
                formats.output_format,
                tile_cnt=block_ct_dim,
                reduce_to_one=True,
                tile_shape=tile_shape,
                input_format=formats.input_format,
            )
            for row_input in row_inputs
        ]
    )

    configuration = TestConfig(
        "sources/quasar/reduce_block_max_row_quasar_test.cpp",
        formats,
        templates=[
            UNPACKER_ENGINE_SEL(),
            DEST_SYNC(dest_sync_mode),
        ],
        runtimes=[
            TILE_COUNT(block_ct_dim),
            NUM_BLOCKS(num_rows),
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
            tile_count_A=block_ct_dim * num_rows,
            tile_count_B=1,
            tile_count_res=num_rows,
            num_faces=tile_shape.total_num_faces(),
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
    )

    res_from_L1 = configuration.run().result

    res_tensor = result_tensor(res_from_L1, formats.output_format, golden_tensor)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        tile_shape=tile_shape,
        print_errors=True,
    ), "Assert against golden failed"


@pytest.mark.quasar
@parametrize(
    formats=input_output_formats([DataFormat.Float16_b]),
    block_ct_dim=runtime([3, 8]),
    num_rows=runtime(list(DEST_DEPTH_BLOCK_COUNTS)),
    dest_sync_mode=[DestSync.Half, DestSync.Full],
)
def test_reduce_block_max_row_dest_depth_quasar(
    formats,
    block_ct_dim,
    num_rows,
    dest_sync_mode,
):
    """4 and 8 rows, one Dest section each."""
    test_reduce_block_max_row_quasar(
        formats, block_ct_dim, (32, 32), dest_sync_mode, num_rows=num_rows
    )
