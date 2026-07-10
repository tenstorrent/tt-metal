# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Test for eltwise binary operations with reuse_dest on Quasar.
import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    EltwiseBinaryGolden,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    EltwiseBinaryReuseDestType,
    ImpliedMathFormat,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import (
    BlocksCalculationAlgorithm,
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BootMode, TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    INPUT_TILE_CNT,
    MATH_FIDELITY,
    MATH_OP,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    OUTPUT_TILE_CNT,
    REUSE_DEST_TYPE,
    TEST_FACE_DIMS,
    generate_input_dim,
)
from helpers.tile_constants import FACE_C_DIM, get_tile_params
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

INPUT_DIMENSIONS = [
    [512, 32],
]
OUTPUT_DIMENSIONS = [
    [128, 32],
]

TILE_DIMENSIONS = [32, 32]


def _reuse_dest_tile_count(dimensions) -> int:
    tile_rows, tile_cols = TILE_DIMENSIONS
    return (dimensions[0] // tile_rows) * (dimensions[1] // tile_cols)


def valid_output_dimensions(formats, dest_sync_mode, input_dimensions) -> list:
    """Output dims compatible with reuse_dest for a given input size, format and dest_sync.

    Three constraints, all decidable at collection time so incompatible combinations are
    never generated (instead of generated then skipped):
      - input tile count must be an exact multiple of the output tile count, and
      - that multiple (`inner_dim`) must be > 1 (reuse_dest needs accumulation), and
      - the output must fit in a single block (the Quasar reuse_dest kernel uses
        block-relative indexing; multi-block accumulates wrongly).
    """
    tile_cnt_input = _reuse_dest_tile_count(input_dimensions)
    valid = []
    for out_dims in OUTPUT_DIMENSIONS:
        tile_cnt_output = _reuse_dest_tile_count(out_dims)
        if tile_cnt_output == 0 or tile_cnt_input % tile_cnt_output != 0:
            continue
        if tile_cnt_input // tile_cnt_output <= 1:
            continue
        try:
            num_blocks, _ = get_num_blocks_and_num_tiles_in_block(
                dest_sync_mode,
                DestAccumulation.No,
                formats,
                out_dims,
                (TILE_DIMENSIONS[0], TILE_DIMENSIONS[1]),
                BlocksCalculationAlgorithm.Standard,
            )
        except ValueError:
            continue  # tiles don't divide evenly into blocks for this combination
        if num_blocks > 1:
            continue
        valid.append(out_dims)
    return valid


@pytest.mark.quasar
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
        ],
    ),
    mathop=[
        MathOperation.Elwadd,
        MathOperation.Elwsub,
        MathOperation.Elwmul,
    ],
    # Math fidelity only affects multiplication; for add/sub only LoFi is meaningful.
    math_fidelity=lambda mathop: (
        [MathFidelity.LoFi]
        if mathop in [MathOperation.Elwadd, MathOperation.Elwsub]
        else [
            MathFidelity.LoFi,
            MathFidelity.HiFi2,
            MathFidelity.HiFi3,
            MathFidelity.HiFi4,
        ]
    ),
    reuse_dest_type=[
        EltwiseBinaryReuseDestType.DEST_TO_SRCA,
        EltwiseBinaryReuseDestType.DEST_TO_SRCB,
    ],
    dest_sync_mode=[DestSync.Half, DestSync.Full],
    input_dimensions=runtime(INPUT_DIMENSIONS),
    output_dimensions=runtime(valid_output_dimensions),
)
def test_eltwise_binary_reuse_dest_quasar(
    formats,
    mathop,
    reuse_dest_type,
    math_fidelity,
    dest_sync_mode,
    input_dimensions,
    output_dimensions,
    boot_mode=BootMode.DEFAULT,
):
    tile_rows, tile_cols = TILE_DIMENSIONS
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(
        [tile_rows, tile_cols]
    )
    num_faces = num_faces_r_dim * num_faces_c_dim

    tile_cnt_input = (input_dimensions[0] // tile_rows) * (
        input_dimensions[1] // tile_cols
    )
    tile_cnt_output = (output_dimensions[0] // tile_rows) * (
        output_dimensions[1] // tile_cols
    )

    inner_dim = tile_cnt_input // tile_cnt_output

    tile_dimensions_tuple = (tile_rows, tile_cols)
    output_num_blocks, output_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync_mode,
        DestAccumulation.No,
        formats,
        output_dimensions,
        tile_dimensions_tuple,
        BlocksCalculationAlgorithm.Standard,
    )
    input_tiles_in_block = inner_dim * output_tiles_in_block
    input_num_blocks = output_num_blocks

    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=[tile_rows, tile_cols],
    )
    src_A_tilized = tilize_block(
        src_A,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=[tile_rows, tile_cols],
        face_r_dim=face_r_dim,
    )
    src_B_tilized = tilize_block(
        src_B,
        dimensions=input_dimensions,
        stimuli_format=formats.input_format,
        num_faces=num_faces,
        tile_dimensions=[tile_rows, tile_cols],
        face_r_dim=face_r_dim,
    )
    src_A_t = src_A_tilized.flatten()
    src_B_t = src_B_tilized.flatten()

    tile_elements = num_faces * face_r_dim * FACE_C_DIM
    torch_format = format_dict[formats.output_format]

    golden_tensor = torch.zeros(tile_cnt_output * tile_elements, dtype=torch_format)

    eltwise_golden = (
        EltwiseBinaryGolden()
        if (mathop == MathOperation.Elwmul and math_fidelity == MathFidelity.LoFi)
        else None
    )

    math_format_for_fidelity = (
        formats.output_format if eltwise_golden is not None else None
    )

    for out_t in range(tile_cnt_output):
        block_idx = out_t // output_tiles_in_block
        tile_in_block = out_t % output_tiles_in_block
        out_start = out_t * tile_elements
        dest = src_A_t[out_start : out_start + tile_elements].to(torch_format)

        for i in range(inner_dim):
            input_tile_idx = (
                block_idx * input_tiles_in_block
                + i * output_tiles_in_block
                + tile_in_block
            )
            start = input_tile_idx * tile_elements
            end = start + tile_elements
            a_tile = src_A_t[start:end].to(torch_format)
            b_tile = src_B_t[start:end].to(torch_format)
            srcA, srcB = (
                (dest.clone(), b_tile)
                if reuse_dest_type == EltwiseBinaryReuseDestType.DEST_TO_SRCA
                else (a_tile, dest.clone())
            )

            if mathop == MathOperation.Elwadd:
                dest = srcA + srcB
            elif mathop == MathOperation.Elwsub:
                dest = srcA - srcB
            elif mathop == MathOperation.Elwmul:
                if eltwise_golden is not None:
                    mask_dtype = format_dict[math_format_for_fidelity]
                    srcA_m, srcB_m = eltwise_golden._apply_fidelity_masking(
                        math_format_for_fidelity,
                        srcA.to(mask_dtype),
                        srcB.to(mask_dtype),
                        0,
                    )
                    product = (
                        (srcA_m.to(torch.float32) * srcB_m.to(torch.float32))
                        .to(srcA_m.dtype)
                        .to(torch_format)
                    )
                    dest = product
                else:
                    dest = srcA * srcB

        golden_tensor[out_start : out_start + tile_elements] = dest.to(torch_format)

    configuration = TestConfig(
        "sources/quasar/eltwise_binary_reuse_dest_quasar_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=mathop),
            REUSE_DEST_TYPE(reuse_dest_type),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
            DEST_SYNC(dest_sync_mode),
        ],
        runtimes=[
            generate_input_dim(input_dimensions, input_dimensions),
            INPUT_TILE_CNT(tile_cnt_input),
            OUTPUT_TILE_CNT(tile_cnt_output),
            NUM_TILES_IN_BLOCK(
                output_tiles_in_block,
                input_num_tiles_in_block=input_tiles_in_block,
                output_num_tiles_in_block=output_tiles_in_block,
            ),
            NUM_BLOCKS(
                output_num_blocks,
                input_num_blocks=input_num_blocks,
                output_num_blocks=output_num_blocks,
            ),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(face_r_dim=face_r_dim, face_c_dim=FACE_C_DIM),
        ],
        variant_stimuli=StimuliConfig(
            src_A_t,
            formats.input_format,
            src_B_t,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_input,
            tile_count_B=tile_cnt_input,
            tile_count_res=tile_cnt_output,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=[tile_rows, tile_cols],
            use_dense_tile_dimensions=True,
        ),
        unpack_to_dest=False,
        dest_acc=DestAccumulation.No,
        boot_mode=boot_mode,
    )

    res_from_L1 = configuration.run().result

    # Verify results match golden
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
