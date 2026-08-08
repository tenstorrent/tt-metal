# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Multi-tile extension of the U10 compute-intrinsics oracle: the MATH thread
# computes through __builtin_xttbh_elwmul (the compiler owns the ALU
# config: hw_configure baseline + per-compute reconfig) but now walks the dest
# register across NUM_TILES_IN_BLOCK tiles.  Dest-walking is author-owned, per
# the agreed programming model: per-tile dest base via math::set_dst_write_addr
# (SETC16 DEST_TARGET_REG_CFG_MATH_Offset = tile<<6, LLK instrn-buffer channel),
# intra-tile row advance via TTI_INCRWC, and per-tile source-valid clear + RWC
# counter reset via TTI_SETRWC.  The compiler never re-emits the dest-base
# baseline (its state stays "known 0", satisfied), so its dest-base=0 does not
# fight the author's per-tile bases.
#
# Scope: 4x 16x16 tiles ([64,16] input, one 16-row face each = two TTELWMULs
# with an INCRWC(0,8,8,8) between).  Formats baked at compile time so the
# intrinsic receives constexpr format args.

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


def test_intrinsic_elwmul_multitile():
    formats = input_output_formats([DataFormat.Float16_b])[0]
    input_dimensions = [64, 16]  # four 16x16 tiles
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
        "sources/intrinsic_eltwise_binary_multitile_test.cpp",
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
            tile_count_A=4,
            tile_count_B=4,
            tile_count_res=4,
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
