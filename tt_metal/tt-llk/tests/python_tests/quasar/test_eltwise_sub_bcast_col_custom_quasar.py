# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Quasar bring-up test for the SDPA blocked bcast-col SUB LLK.

Exercises _llk_unpack_AB_sub_bcast_col_custom_ / _llk_math_sub_bcast_cols_reuse_custom_:
one SrcB tile is unpacked in COL layout and held while it is subtracted (column-broadcast)
from each of ``ct_dim`` SrcA column tiles, each difference landing in its own dest slot.

The (ct_dim, num_blocks) cases are ordered so the first failure localises the bug. Math
emits its ELWSUB stream directly (no MOP) with CLR_NONE in every slot instead of
per-face CLR_SRCB_VLD, walking the COL faces via the SrcB increments in ADDR_MOD_5/6/7
(+8, -8, +8, +24); the ct_dim=1 case exercises exactly that with SrcB reuse switched off. If
ct_dim=1 fails the face walk itself is wrong; if it passes and ct_dim=2 fails, the
per-tile SETRWC(CLR_A, ..., SET_AB) is not holding SrcB across tiles.
"""

import logging

import pytest
import torch
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BroadcastGolden,
    EltwiseBinaryGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import (
    DEST_SYNC_TILE_LIMITS,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BootMode, TestConfig
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_OP,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    TEST_FACE_DIMS,
)
from helpers.tile_constants import FACE_C_DIM, MAX_NUM_FACES, get_tile_params
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

logger = logging.getLogger(__name__)

# The two shapes the LLK supports: a full 32x32 tile (4 faces of 16x16 in a 2x2 grid) and the 16x32
# tiny tile (1x2 faces). The math walk covers a face-row with four 8-row ops and reads the face-row's
# first SrcB face twice, so it needs two full 16-row faces per face-row. This rules out narrow
# (32x16) and short-face shapes by construction; see validate_tensor_shape_sub_bcast_col_custom_.
TILE_DIMENSIONS = [[32, 32], [16, 32]]

# (ct_dim, num_blocks), ascending by reuse depth so the first red variant is diagnostic.
# ct_dim=8 fills a 16-bit half-dest exactly at 32x32 (MAX_TILES_IN_HALF_DEST); the 2-block case adds
# dest-section switching and a per-block SrcB re-unpack. A 32-bit dest halves that capacity and a
# 2-face tile doubles it, so the deepest shapes are filtered per (dest_sync, dest_acc, tile) below.
_BLOCK_SHAPES = [(1, 1), (2, 1), (4, 1), (8, 1), (8, 2)]


def _block_shapes_fitting_dest(
    dest_sync: DestSync, dest_acc: DestAccumulation, tile_dimensions: list
):
    """Keep only the shapes whose ct_dim tiles all fit in one dest section.

    Math lands every tile of a block in its own dest slot starting at index 0, so ct_dim is
    bounded by the section capacity: 8 tiles for Half / 16 for Full on a 16-bit dest, halved
    to 4 / 8 when ``dest_acc=Yes`` widens dest to 32 bits. A tile with fewer than the full 2x2
    face grid owns proportionally fewer dest rows, so proportionally more of them fit.
    """
    _, num_faces_r, num_faces_c = get_tile_params(tile_dimensions)
    capacity_divisor = 2 if dest_acc == DestAccumulation.Yes else 1
    max_tiles = (
        DEST_SYNC_TILE_LIMITS[dest_sync]
        * (MAX_NUM_FACES // (num_faces_r * num_faces_c))
        // capacity_divisor
    )
    return [shape for shape in _BLOCK_SHAPES if shape[0] <= max_tiles]


@pytest.mark.quasar
@parametrize(
    formats=input_output_formats(
        [DataFormat.Float16_b, DataFormat.Float16, DataFormat.Float32], same=True
    ),
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
    implied_math_format=[ImpliedMathFormat.No, ImpliedMathFormat.Yes],
    dest_sync=[DestSync.Half, DestSync.Full],
    tile_dimensions=TILE_DIMENSIONS,
    block_shape=lambda dest_sync, dest_acc, tile_dimensions: _block_shapes_fitting_dest(
        dest_sync, dest_acc, tile_dimensions
    ),
)
def test_eltwise_sub_bcast_col_custom_quasar(
    formats,
    dest_acc,
    implied_math_format,
    dest_sync,
    tile_dimensions,
    block_shape,
    boot_mode=BootMode.DEFAULT,
):
    ct_dim, num_blocks = block_shape
    total_tiles = ct_dim * num_blocks
    face_r_dim, num_faces_r, num_faces_c = get_tile_params(tile_dimensions)
    num_faces_total = num_faces_r * num_faces_c

    # srcA is a single row of ct_dim*num_blocks column tiles; srcB is the one reused tile.
    input_dimensions_A = [tile_dimensions[0], tile_dimensions[1] * total_tiles]
    input_dimensions_B = [tile_dimensions[0], tile_dimensions[1]]

    logger.info(
        "tile=%s ct_dim=%d num_blocks=%d srcA=%s srcB=%s",
        tile_dimensions,
        ct_dim,
        num_blocks,
        input_dimensions_A,
        input_dimensions_B,
    )

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions_A,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions_B,
        tile_dimensions=tile_dimensions,
    )

    src_A_tilized = tilize_block(
        src_A,
        dimensions=input_dimensions_A,
        stimuli_format=formats.input_format,
        num_faces=num_faces_total,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    ).flatten()
    src_B_tilized = tilize_block(
        src_B,
        dimensions=input_dimensions_B,
        stimuli_format=formats.input_format,
        num_faces=num_faces_total,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    ).flatten()

    # Column broadcast replicates column 0 of each face-row across the tile. Compute it in
    # tiled space (that is what the HW sees), then untilize to build the row-major golden.
    broadcast_golden = get_golden_generator(BroadcastGolden)
    src_B_broadcasted_tilized = broadcast_golden(
        BroadcastType.Column,
        src_B_tilized,
        formats.input_format,
        num_faces=num_faces_total,
        tile_cnt=tile_cnt_B,
        face_r_dim=face_r_dim,
    )
    src_B_golden = untilize_block(
        src_B_broadcasted_tilized,
        formats.input_format,
        input_dimensions_B,
        num_faces=num_faces_total,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    ).flatten()

    # The same srcB tile is subtracted from every srcA column tile.
    src_B_golden_expanded = (
        src_B_golden.view(input_dimensions_B[0], input_dimensions_B[1])
        .repeat(1, total_tiles)
        .flatten()
    )

    generate_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = generate_golden(
        MathOperation.Elwsub,
        src_A,
        src_B_golden_expanded,
        formats.output_format,
        MathFidelity.LoFi,
    )

    configuration = TestConfig(
        "sources/quasar/eltwise_sub_bcast_col_custom_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.Elwsub),
            BROADCAST_TYPE(BroadcastType.Column),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[
            NUM_TILES_IN_BLOCK(
                ct_dim,
                input_num_tiles_in_block=ct_dim,
                output_num_tiles_in_block=ct_dim,
            ),
            NUM_BLOCKS(
                num_blocks,
                input_num_blocks=num_blocks,
                output_num_blocks=num_blocks,
            ),
            NUM_FACES(num_faces_total),
            NUM_FACES_R_DIM(num_faces_r, num_faces_r),
            NUM_FACES_C_DIM(num_faces_c, num_faces_c),
            TEST_FACE_DIMS(face_r_dim=face_r_dim, face_c_dim=FACE_C_DIM),
        ],
        variant_stimuli=StimuliConfig(
            src_A_tilized,
            formats.input_format,
            src_B_tilized,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces_total,
            face_r_dim=face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
        boot_mode=boot_mode,
    )

    res_from_L1 = configuration.run().result

    res_from_L1 = untilize_block(
        res_from_L1,
        formats.output_format,
        input_dimensions_A,
        num_faces=num_faces_total,
        tile_dimensions=tile_dimensions,
        face_r_dim=face_r_dim,
    ).flatten()

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
