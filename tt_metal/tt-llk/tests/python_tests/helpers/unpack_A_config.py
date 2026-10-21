# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TestConfig for sources/unpack_A_test.cpp on one dense tile of any shape.

Shared by the face-grid tests in test_unpack_A.py and
test_eltwise_unary_datacopy_tensor_shape.py: unpack_A -> datacopy -> pack, with
the real face grid passed to unpack and math.
"""

from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    BroadcastType,
    DestAccumulation,
    DestSync,
    EltwiseBinaryReuseDestType,
    StochasticRounding,
    Transpose,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    ACC_TO_DEST,
    BROADCAST_TYPE,
    DISABLE_SRC_ZERO_FLAG,
    INPUT_DIMENSIONS,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    PARTIAL_FACE,
    REUSE_DEST_TYPE,
    STOCHASTIC_ROUNDING,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)
from helpers.tile_constants import get_tile_params


def unpack_A_tile_config(
    formats,
    tile_dimensions,
    src_A,
    tile_cnt_A,
    src_B,
    tile_cnt_B,
    dest_acc: DestAccumulation,
    broadcast_type: BroadcastType = BroadcastType.None_,
    transpose: Transpose = Transpose.No,
    within_face_transpose: Transpose | None = None,
) -> TestConfig:
    """Build the config; transpose sets the face transpose, and also the within-face
    transpose unless within_face_transpose is given.

    Unpack to dest is used when the input is 32-bit and dest_acc is Yes.
    """
    if within_face_transpose is None:
        within_face_transpose = transpose
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        list(tile_dimensions),
        tile_dimensions,
        BlocksCalculationAlgorithm.Standard,
    )

    return TestConfig(
        "sources/unpack_A_test.cpp",
        formats,
        templates=[
            STOCHASTIC_ROUNDING(StochasticRounding.No),
            BROADCAST_TYPE(broadcast_type),
            ACC_TO_DEST(False),
            REUSE_DEST_TYPE(EltwiseBinaryReuseDestType.NONE),
            PARTIAL_FACE(
                partial_a=False,
                partial_face_pack=False,
                partial_b=False,
                partial_face_math=False,
            ),
            DISABLE_SRC_ZERO_FLAG(False),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(transpose),
            UNPACK_TRANS_WITHIN_FACE(within_face_transpose),
            NUM_FACES(num_faces),
            NUM_FACES_R_DIM(num_faces_r_dim, num_faces_r_dim),
            NUM_FACES_C_DIM(num_faces_c_dim, num_faces_c_dim),
            TILE_COUNT(tile_cnt_A),
            TEST_FACE_DIMS(face_r_dim=face_r_dim),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            NUM_BLOCKS(num_blocks),
            # One tile; unpack_A_test.cpp does not read these.
            INPUT_DIMENSIONS(1, 1),
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
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit()
        and dest_acc == DestAccumulation.Yes,
    )
