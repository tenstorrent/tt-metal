# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    DestAccumulation,
    L1Accumulation,
    format_dict,
    format_tile_sizes,
)
from helpers.param_config import max_dest_tiles
from helpers.perf.blaze import execute_llk
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    DEST_INDEX,
    IN_TILE_DIMS,
    L1_ACC,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.tile_constants import calculate_tile_size_bytes, get_tile_params

FORMATS = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)

TILE_DIMS = [
    (1, 32),
    (2, 32),
    (4, 32),
    (8, 32),
    (16, 32),
    (16, 16),
    (32, 32),
]


def make_pack_tiny_tile_config(
    tile_dims,
    num_tiles,
    formats,
    dest_acc,
    *,
    test_name,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    """Execute a pack tiny-tile kernel for one tile shape and tile count."""
    tile_r, tile_c = tile_dims
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dims)
    num_faces = num_faces_r_dim * num_faces_c_dim
    input_dimensions = [tile_r * num_tiles, tile_c]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=list(tile_dims),
    )

    torch_format = format_dict[formats.output_format]
    golden_tensor = src_A.to(torch_format)

    max_tiles_in_dest = max_dest_tiles(
        dest_acc == DestAccumulation.Yes or formats.input_format.is_32_bit()
    )
    if num_faces < 4:
        max_tiles_in_dest *= 4 // num_faces

    num_tiles_in_block = min(tile_cnt_A, max_tiles_in_dest)
    num_blocks = tile_cnt_A // num_tiles_in_block
    assert num_blocks * num_tiles_in_block == tile_cnt_A, (
        f"Tile count {tile_cnt_A} not evenly divisible into "
        f"{num_blocks} blocks of {num_tiles_in_block}"
    )

    outcome = execute_llk(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
        test_config_kwargs=dict(
            test_name=test_name,
            formats=formats,
            templates=[],
            runtimes=[
                DEST_INDEX(0),
                TILE_COUNT(tile_cnt_A),
                NUM_FACES(num_faces),
                TEST_FACE_DIMS(face_r_dim),
                IN_TILE_DIMS(tile_r, tile_c),
                L1_ACC(L1Accumulation.No),
                NUM_BLOCKS(num_blocks),
                NUM_TILES_IN_BLOCK(num_tiles_in_block),
            ],
            variant_stimuli=StimuliConfig(
                src_A,
                formats.input_format,
                src_B,
                formats.input_format,
                formats.output_format,
                tile_count_A=tile_cnt_A,
                tile_count_B=tile_cnt_B,
                tile_count_res=tile_cnt_A,
                num_faces=num_faces,
                face_r_dim=face_r_dim,
                tile_dimensions=list(tile_dims),
                use_dense_tile_dimensions=True,
                operand_res_tile_size=calculate_tile_size_bytes(
                    formats.output_format, list(tile_dims), format_tile_sizes
                ),
            ),
            dest_acc=dest_acc,
        ),
    )

    return outcome, golden_tensor
