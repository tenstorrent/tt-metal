# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TensorShape coverage for the math unary datacopy: one tile per shape, A2D, SrcB broadcasts and unpack-to-dest."""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    BroadcastGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    BroadcastType,
    DestAccumulation,
    DestSync,
    EltwiseBinaryReuseDestType,
    StochasticRounding,
    Transpose,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    ACC_TO_DEST,
    BROADCAST_TYPE,
    DISABLE_SRC_ZERO_FLAG,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    REUSE_DEST_TYPE,
    STOCHASTIC_ROUNDING,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)
from helpers.tile_constants import get_tile_params
from helpers.tile_shape import FACE_C_DIM, TileShape, construct_tile_shape
from helpers.utils import passed_test

DATACOPY_TILE_SHAPES = [
    [1, 32],
    [2, 32],
    [4, 32],
    [8, 32],
    [16, 32],
    [32, 16],
    [16, 16],
    [32, 32],
    [8, 16],
]


@parametrize(
    tile_dimensions=DATACOPY_TILE_SHAPES,
    formats=input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
    broadcast_type=[
        BroadcastType.None_,
        BroadcastType.Scalar,
        BroadcastType.Row,
        BroadcastType.Column,
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_eltwise_unary_datacopy_tensor_shape(
    tile_dimensions, formats, broadcast_type, dest_acc
):
    if dest_acc == DestAccumulation.No and formats.input_format.is_32_bit():
        pytest.skip("32-bit formats require dest accumulation")

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    # The shapes below hang or give wrong output, so the LLK asserts on them.
    # TODO(#47328): Column broadcast runs 4 SrcB faces; anything but 32x32 hangs on Wormhole.
    if broadcast_type == BroadcastType.Column and tile_dimensions != [32, 32]:
        pytest.skip(
            "TODO(#47328): asserted, column broadcast is 32x32 only "
            "(_llk_unpack_A_init_ and _llk_math_eltwise_unary_datacopy_init_)"
        )
    # Row broadcast of a 32x16 tile through dest takes the bottom face's row for the bottom half.
    if (
        broadcast_type == BroadcastType.Row
        and tile_dimensions == [32, 16]
        and unpack_to_dest
    ):
        pytest.skip(
            "Asserted: row broadcast with unpack to dest on 32x16 (_llk_unpack_A_init_)"
        )
    # Same for the Wormhole SrcB row broadcast, whose MOP always runs 4 faces.
    if (
        get_chip_architecture() == ChipArchitecture.WORMHOLE
        and broadcast_type == BroadcastType.Row
        and tile_dimensions == [32, 16]
    ):
        pytest.skip(
            "Asserted on Wormhole: row broadcast datacopy on 32x16 "
            "(_llk_math_eltwise_unary_datacopy_init_)"
        )

    # Same known Wormhole issue as test_bcast.py::test_unpack_bcast: https://github.com/tenstorrent/tt-llk/issues/1680
    if (
        get_chip_architecture() == ChipArchitecture.WORMHOLE
        and broadcast_type == BroadcastType.Row
        and dest_acc == DestAccumulation.Yes
        and formats.input_format == DataFormat.Float16_b
    ):
        pytest.skip(
            "Row broadcast with dest_acc=Yes broken on Wormhole for Float16_b "
            "(https://github.com/tenstorrent/tt-llk/issues/1680)"
        )

    # 8x16 (one 8-row face) is not in SUPPORTED_TILE_SIZES, so it goes through the
    # face_r_dim / num_faces path, as in test_unpack_A_targeted_tensor_shape_coverage.
    one_face_8x16 = tile_dimensions == [8, 16]
    if one_face_8x16:
        face_r_dim, num_faces_r_dim, num_faces_c_dim = 8, 1, 1
        tile_shape = TileShape(
            face_r_dim=face_r_dim,
            face_c_dim=FACE_C_DIM,
            num_faces_r_dim=num_faces_r_dim,
            num_faces_c_dim=num_faces_c_dim,
        )
        buffer_dimensions = list(TILE_DIMENSIONS)
        stimuli_kwargs = {"face_r_dim": face_r_dim, "num_faces": 1}
    else:
        face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
        tile_shape = construct_tile_shape(tile_dimensions)
        buffer_dimensions = list(tile_dimensions)
        stimuli_kwargs = {"tile_dimensions": tile_dimensions}
    num_faces = num_faces_r_dim * num_faces_c_dim
    input_dimensions = list(tile_dimensions)

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        **stimuli_kwargs,
    )

    if broadcast_type != BroadcastType.None_:
        generate_broadcast_golden = get_golden_generator(BroadcastGolden)
        golden_tensor = generate_broadcast_golden(
            broadcast_type,
            src_A,
            formats.output_format,
            num_faces=num_faces,
            tile_cnt=tile_cnt_A,
            face_r_dim=face_r_dim,
            num_faces_r_dim=num_faces_r_dim,
            num_faces_c_dim=num_faces_c_dim,
        )
    else:
        golden_tensor = src_A.to(format_dict[formats.output_format])

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        buffer_dimensions,
        buffer_dimensions,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/unpack_A_test.cpp",
        formats,
        templates=[
            STOCHASTIC_ROUNDING(StochasticRounding.No),
            BROADCAST_TYPE(broadcast_type),
            ACC_TO_DEST(False),
            REUSE_DEST_TYPE(EltwiseBinaryReuseDestType.NONE),
            DISABLE_SRC_ZERO_FLAG(False),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
            NUM_FACES(num_faces),
            NUM_FACES_R_DIM(num_faces_r_dim, num_faces_r_dim),
            NUM_FACES_C_DIM(num_faces_c_dim, num_faces_c_dim),
            TILE_COUNT(tile_cnt_A),
            TEST_FACE_DIMS(face_r_dim=face_r_dim),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            NUM_BLOCKS(num_blocks),
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
            tile_dimensions=buffer_dimensions,
            use_dense_tile_dimensions=not one_face_8x16,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=unpack_to_dest,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(golden_tensor)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        tile_shape=tile_shape,
    )

    # Datacopy/bcast should be bit-exact for float formats (no compute loss)
    if formats.input_format in (DataFormat.Float32, DataFormat.Float16_b):
        assert torch.equal(
            golden_tensor, res_tensor
        ), "Datacopy/bcast should be exact for float formats"
