# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Full 3-thread compiler-config oracle: the same kernel as
# test_intrinsic_eltwise_binary.py, but compiled with BOTH
# TT_COMPILER_EMITS_UNPACK_CONFIG and TT_COMPILER_EMITS_PACK_CONFIG -- the
# UNPACK and PACK threads' hardware configure become compiler-managed
# config-declaration intrinsics and their per-tile data ops become inline
# unpacr/pacr words; the MATH thread computes through __builtin_xttbh_elwmul as
# always.  The compiler owns all three threads' idempotent config; the per-tile
# L1 addresses, dest-tile select, and dest-section sync stay author-owned.
#
# This is the real-hardware functional golden.  The ttsim leg is blocked by the
# documented tensix_cfg_rd32 modeling gap, so on ttsim it compiles only; on
# Blackhole hardware (pytest without --run-simulator) it runs against the torch
# golden.

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    EltwiseBinaryGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    DestSync,
    MathFidelity,
    MathOperation,
    Transpose,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    BROADCAST_TYPE,
    DEST_SYNC,
    MATH_FIDELITY,
    MATH_OP,
    NUM_BLOCKS,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    TEST_FACE_DIMS,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test


def test_intrinsic_elwmul_single_face_full_config_compiler():
    formats = input_output_formats([DataFormat.Float16_b])[0]
    input_dimensions = [16, 16]  # one 16x16 tile = one face
    tile_dimensions = [16, 16]

    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format_B,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )

    num_faces = 1
    face_r_dim = 16

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        DestAccumulation.No,
        formats,
        input_dimensions,
        tile_dimensions,
    )

    generate_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = generate_golden(
        MathOperation.Elwmul,
        tilize_block(
            src_A,
            dimensions=input_dimensions,
            stimuli_format=formats.input_format,
            num_faces=num_faces,
            tile_dimensions=tile_dimensions,
            face_r_dim=face_r_dim,
        ).flatten(),
        tilize_block(
            src_B,
            dimensions=input_dimensions,
            stimuli_format=formats.input_format_B,
            num_faces=num_faces,
            tile_dimensions=tile_dimensions,
            face_r_dim=face_r_dim,
        ).flatten(),
        formats.output_format,
        MathFidelity.LoFi,
        input_format=formats.input_format,
        input_format_B=formats.input_format_B,
    )

    configuration = TestConfig(
        "sources/intrinsic_eltwise_binary_full_config_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(MathFidelity.LoFi),
            BROADCAST_TYPE(BroadcastType.None_),
            MATH_OP(mathop=MathOperation.Elwmul),
            DEST_SYNC(),
        ],
        runtimes=[
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            NUM_BLOCKS(num_blocks),
            NUM_FACES_R_DIM(1),
            NUM_FACES_C_DIM(1),
            TEST_FACE_DIMS(face_r_dim=face_r_dim),
        ],
        variant_stimuli=StimuliConfig(
            tilize_block(
                src_A,
                dimensions=input_dimensions,
                stimuli_format=formats.input_format,
                num_faces=num_faces,
                tile_dimensions=tile_dimensions,
                face_r_dim=face_r_dim,
            ).flatten(),
            formats.input_format,
            tilize_block(
                src_B,
                dimensions=input_dimensions,
                stimuli_format=formats.input_format_B,
                num_faces=num_faces,
                tile_dimensions=tile_dimensions,
                face_r_dim=face_r_dim,
            ).flatten(),
            formats.input_format_B,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        dest_acc=DestAccumulation.No,
        unpack_to_dest=False,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
