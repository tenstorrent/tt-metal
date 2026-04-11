# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test: Pack tiny tiles in contiguous L1 blocks.

Validates packing multi-tile blocks of tiny tiles (1x32, 8x32, 16x32,
16x16) as well as standard 32x32 tiles into contiguous L1 memory.

In the kernel used by this test, math writes each tile into sparse
Tile32x32 DEST slots, and _llk_pack_block_contiguous_ performs the
sparse-to-dense packing into a contiguous L1 block.
"""

import torch
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    L1Accumulation,
    format_dict,
    format_tile_sizes,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli_w_tile_dimensions
from helpers.test_config import TestConfig
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
from helpers.utils import passed_test


def _make_config(
    tile_dims,
    num_tiles,
    formats,
    dest_acc,
):
    """Build TestConfig for a given tile shape and tile count."""
    tile_r, tile_c = tile_dims
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dims)
    num_faces = num_faces_r_dim * num_faces_c_dim

    # Stack tiles vertically so input_dimensions = (num_tiles * tile_r, tile_c)
    input_dimensions = [tile_r * num_tiles, tile_c]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli_w_tile_dimensions(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=list(tile_dims),
    )

    # Golden: input round-tripped through the output format
    torch_format = format_dict[formats.output_format]
    golden_tensor = src_A.to(torch_format)

    # Determine DEST capacity (SyncHalf: 8 tiles FP16, 4 tiles FP32)
    capacity_divisor = (
        2
        if (dest_acc == DestAccumulation.Yes or formats.input_format.is_32_bit())
        else 1
    )
    max_tiles_in_dest = 8 // capacity_divisor

    # For tiny tiles with fewer faces, more tiles fit in DEST
    # Scale by the ratio of faces: a 2-face tile uses half the DEST of a 4-face tile
    if num_faces < 4:
        max_tiles_in_dest = max_tiles_in_dest * (4 // num_faces)

    num_tiles_in_block = min(tile_cnt_A, max_tiles_in_dest)
    num_blocks = tile_cnt_A // num_tiles_in_block

    assert num_blocks * num_tiles_in_block == tile_cnt_A, (
        f"Tile count {tile_cnt_A} not evenly divisible into "
        f"{num_blocks} blocks of {num_tiles_in_block}"
    )

    configuration = TestConfig(
        "sources/pack_tiny_tile_block_test.cpp",
        formats,
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
    )

    return configuration, golden_tensor, torch_format


# ── Main test ───────────────────────────────────────────────────────────────


@skip_for_wormhole
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16,
            DataFormat.Float16_b,
        ]
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    tile_dims=[
        (1, 32),  # face_r_dim=1,  num_faces=2
        (2, 32),  # face_r_dim=2,  num_faces=2
        (4, 32),  # face_r_dim=4,  num_faces=2
        (8, 32),  # face_r_dim=8,  num_faces=2
        (16, 32),  # face_r_dim=16, num_faces=2
        (16, 16),  # face_r_dim=16, num_faces=1
        (32, 32),  # face_r_dim=16, num_faces=4 (baseline)
    ],
    num_tiles=[1, 2, 4, 8],
)
def test_pack_tiny_tile_block(
    formats,
    dest_acc,
    tile_dims,
    num_tiles,
):
    configuration, golden_tensor, torch_format = _make_config(
        tile_dims, num_tiles, formats, dest_acc
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Length mismatch: got {len(res_from_L1)}, expected {len(golden_tensor)}"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)
    assert passed_test(golden_tensor, res_tensor, formats.output_format)


# ── Reconfig test: init 32x32 → reconfig to tiny → multi-tile MOP pack ─────


@skip_for_wormhole
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16,
            DataFormat.Float16_b,
        ]
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    tile_dims=[
        (1, 32),  # face_r_dim=1,  num_faces=2
        (2, 32),  # face_r_dim=2,  num_faces=2
        (4, 32),  # face_r_dim=4,  num_faces=2
        (8, 32),  # face_r_dim=8,  num_faces=2
        (16, 32),  # face_r_dim=16, num_faces=2
        (16, 16),  # face_r_dim=16, num_faces=1
    ],
    num_tiles=[2, 4, 8],
)
def test_pack_tiny_tile_reconfig(formats, dest_acc, tile_dims, num_tiles):
    """Init pack for 32x32, then full re-init for tiny tiles, then multi-tile
    block-contiguous pack. Validates the transition between tile shapes."""
    tile_r, tile_c = tile_dims
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dims)
    num_faces = num_faces_r_dim * num_faces_c_dim

    input_dimensions = [tile_r * num_tiles, tile_c]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli_w_tile_dimensions(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=list(tile_dims),
    )

    torch_format = format_dict[formats.output_format]
    golden_tensor = src_A.to(torch_format)

    capacity_divisor = (
        2
        if (dest_acc == DestAccumulation.Yes or formats.input_format.is_32_bit())
        else 1
    )
    max_tiles_in_dest = 8 // capacity_divisor
    if num_faces < 4:
        max_tiles_in_dest = max_tiles_in_dest * (4 // num_faces)

    num_tiles_in_block = min(tile_cnt_A, max_tiles_in_dest)
    num_blocks = tile_cnt_A // num_tiles_in_block

    assert num_blocks * num_tiles_in_block == tile_cnt_A

    configuration = TestConfig(
        "sources/pack_tiny_tile_reconfig_test.cpp",
        formats,
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
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Length mismatch: got {len(res_from_L1)}, expected {len(golden_tensor)}"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)
    assert passed_test(golden_tensor, res_tensor, formats.output_format)
