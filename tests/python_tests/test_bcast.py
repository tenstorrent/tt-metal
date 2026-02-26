# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BroadcastGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    EltwiseBinaryReuseDestType,
    StochasticRounding,
    Transpose,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli_w_tile_dimensions
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    ACC_TO_DEST,
    BROADCAST_TYPE,
    DISABLE_SRC_ZERO_FLAG,
    NUM_FACES,
    PARTIAL_FACE,
    REUSE_DEST_TYPE,
    STOCHASTIC_ROUNDING,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)
from helpers.tile_constants import get_tile_params
from helpers.utils import passed_test

supported_formats = [
    DataFormat.Int32,
    DataFormat.UInt32,
    DataFormat.UInt16,
    DataFormat.Float32,
    DataFormat.Float16_b,
    DataFormat.Bfp8_b,
]

# Sweep tile dimensions from tiny ([1,32]..[16,32]) through full ([32,32]).
# Tiny tiles have fewer faces (num_faces=2) and variable face_r_dim;
# full 32x32 tiles have 4 faces with face_r_dim=16.
# BroadcastType.None_ is a datacopy (unpack A -> DEST -> pack to L1).


@parametrize(
    # enable tiny tiles tests when they're added formally to the LLKs
    # tile_dimensions=[[1, 32], [2, 32], [4, 32], [8, 32], [16, 32], [32, 32]],
    tile_dimensions=[[32, 32]],
    formats=input_output_formats(supported_formats, same=True),
    broadcast_type=[
        BroadcastType.None_,
        BroadcastType.Column,
        BroadcastType.Row,
        BroadcastType.Scalar,
    ],
    dest_acc=[DestAccumulation.Yes, DestAccumulation.No],
)
def test_unpack_bcast(
    tile_dimensions, formats, broadcast_type, dest_acc, workers_tensix_coordinates
):
    # --- Skips -----------------------------------------------------------

    if dest_acc == DestAccumulation.No and formats.input_format in (
        DataFormat.Float32,
        DataFormat.Int32,
        DataFormat.UInt32,
    ):
        pytest.skip("32-bit formats require dest accumulation")

    # --- Skips from bugs --------------------------------------------------

    # TODO: pgardner - Column broadcast for tiny tiles needs kernel support
    if tile_dimensions != [32, 32] and broadcast_type == BroadcastType.Column:
        pytest.skip("Column broadcast not yet implemented for tiny tiles")

    # TODO: pgardner - Bfp8_b requires minimum 16 exponents per face
    if tile_dimensions[0] < 16 and formats.input_format == DataFormat.Bfp8_b:
        pytest.skip("Bfp8_b not supported for tile height < 16")

    # TODO: pgardner - known WH issue with row broadcast + dest accumulation
    if (
        get_chip_architecture() == ChipArchitecture.WORMHOLE
        and broadcast_type == BroadcastType.Row
        and dest_acc == DestAccumulation.Yes
        and formats.input_format in (DataFormat.Float16_b, DataFormat.Bfp8_b)
    ):
        pytest.skip(
            "Row broadcast with dest_acc=Yes broken on Wormhole for Float16_b/Bfp8_b"
        )

    # --- Tile geometry ---------------------------------------------------
    # get_tile_params returns (face_r_dim, num_faces_r_dim, num_faces_c_dim).
    # For tiny tiles (e.g. [4,32]): face_r_dim=4, num_faces=2.
    # For full tiles ([32,32]):     face_r_dim=16, num_faces=4.
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim
    input_dimensions = list(tile_dimensions)

    # --- Stimuli generation ----------------------------------------------
    # generate_stimuli_w_tile_dimensions produces dense data for any tile size.
    # For [32,32] this is equivalent to the legacy generate_stimuli path.
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli_w_tile_dimensions(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )

    # --- Golden model ----------------------------------------------------
    # Broadcast types use BroadcastGolden which handles all face geometries.
    # Datacopy (None_) golden is just the input cast to the output format.
    if broadcast_type != BroadcastType.None_:
        generate_broadcast_golden = get_golden_generator(BroadcastGolden)
        golden_tensor = generate_broadcast_golden(
            broadcast_type,
            src_A,
            formats.output_format,
            num_faces=num_faces,
            tile_cnt=tile_cnt_A,
            face_r_dim=face_r_dim,
        )
    else:
        golden_tensor = src_A.to(format_dict[formats.output_format])

    # --- Kernel configuration --------------------------------------------
    configuration = TestConfig(
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
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
            NUM_FACES(num_faces),
            TILE_COUNT(tile_cnt_A),
            TEST_FACE_DIMS(face_r_dim=face_r_dim),
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

    res_from_L1 = configuration.run(workers_tensix_coordinates).result

    # --- Assertions ------------------------------------------------------
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    # Pretty red/green diff output via passed_test (tolerance-based)
    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        tile_dimensions=tile_dimensions,
    )

    # Datacopy/bcast should be bit-exact for float formats (no compute loss)
    if formats.input_format in (DataFormat.Float32, DataFormat.Float16_b):
        assert torch.equal(
            golden_tensor, res_tensor
        ), "Datacopy/bcast should be exact for float formats"
