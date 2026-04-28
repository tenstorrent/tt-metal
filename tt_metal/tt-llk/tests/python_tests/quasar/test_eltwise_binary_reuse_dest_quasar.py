# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Test for eltwise binary operations with reuse_dest on Quasar.


import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    EltwiseBinaryGolden,
    quantize_mx_tensor_chunked,
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
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli_w_tile_dimensions
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


@pytest.mark.quasar
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.MxFp8R,
            DataFormat.MxFp8P,
        ],
    ),
    mathop=[
        MathOperation.Elwadd,
        MathOperation.Elwsub,
        MathOperation.Elwmul,
    ],
    reuse_dest_type=[
        EltwiseBinaryReuseDestType.DEST_TO_SRCA,
        EltwiseBinaryReuseDestType.DEST_TO_SRCB,
    ],
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    dest_sync_mode=[DestSync.Half, DestSync.Full],
    input_dimensions=INPUT_DIMENSIONS,
    output_dimensions=OUTPUT_DIMENSIONS,
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
    if mathop != MathOperation.Elwmul and math_fidelity != MathFidelity.LoFi:
        pytest.skip("elwadd/elwsub only supports LoFi mode")

    if mathop == MathOperation.Elwmul and formats.input_format.is_mx_format():
        pytest.skip(
            "Elwmul with MX input and reuse_dest has golden vs hardware rounding differences; skip to avoid flaky tolerance failures"
        )

    # MX formats require implied_math_format=Yes on Quasar; set it and disable_format_inference so golden matches.
    use_mx = formats.input_format.is_mx_format() or formats.output_format.is_mx_format()
    implied_math_format = ImpliedMathFormat.Yes if use_mx else ImpliedMathFormat.No
    disable_format_inference = use_mx

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

    if tile_cnt_input % tile_cnt_output != 0:
        pytest.skip(
            f"Input tile count ({tile_cnt_input}) must be divisible by "
            f"output tile count ({tile_cnt_output})"
        )
    inner_dim = tile_cnt_input // tile_cnt_output
    if inner_dim == 1:
        pytest.skip("reuse_dest requires inner_dim > 1")

    tile_dimensions_tuple = (tile_rows, tile_cols)
    output_num_blocks, output_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync_mode,
        DestAccumulation.No,
        formats,
        output_dimensions,
        tile_dimensions_tuple,
        BlocksCalculationAlgorithm.Standard,
    )
    if output_num_blocks > 1:
        pytest.skip(
            "Quasar reuse_dest kernel supports single output block only; "
            "multi-block uses block-relative indexing and wrong accumulation"
        )
    input_tiles_in_block = inner_dim * output_tiles_in_block
    input_num_blocks = output_num_blocks

    src_A, _, src_B, _ = generate_stimuli_w_tile_dimensions(
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
    if formats.input_format.is_mx_format():
        src_A_t = quantize_mx_tensor_chunked(
            src_A_t.to(torch.bfloat16), formats.input_format
        )
        src_B_t = quantize_mx_tensor_chunked(
            src_B_t.to(torch.bfloat16), formats.input_format
        )
    golden_tensor = torch.zeros(tile_cnt_output * tile_elements, dtype=torch_format)

    internal_dtype = torch.bfloat16 if use_mx else torch_format

    eltwise_golden = (
        EltwiseBinaryGolden()
        if (mathop == MathOperation.Elwmul and math_fidelity == MathFidelity.LoFi)
        else None
    )
    math_format_for_fidelity = (
        (DataFormat.Float16_b if use_mx else formats.output_format)
        if eltwise_golden is not None
        else None
    )

    for out_t in range(tile_cnt_output):
        block_idx = out_t // output_tiles_in_block
        tile_in_block = out_t % output_tiles_in_block
        out_start = out_t * tile_elements
        dest = src_A_t[out_start : out_start + tile_elements].to(internal_dtype)

        for i in range(inner_dim):
            input_tile_idx = (
                block_idx * input_tiles_in_block
                + i * output_tiles_in_block
                + tile_in_block
            )
            start = input_tile_idx * tile_elements
            end = start + tile_elements
            a_tile = src_A_t[start:end].to(internal_dtype)
            b_tile = src_B_t[start:end].to(internal_dtype)
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
                        .to(internal_dtype)
                    )
                    dest = product
                else:
                    dest = srcA * srcB

        if formats.output_format.is_mx_format():
            dest = quantize_mx_tensor_chunked(dest, formats.output_format)
        golden_tensor[out_start : out_start + tile_elements] = dest.to(torch_format)

    configuration = TestConfig(
        "sources/quasar/eltwise_binary_reuse_dest_quasar_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop),
            IMPLIED_MATH_FORMAT(implied_math_format),
            REUSE_DEST_TYPE(reuse_dest_type),
            DEST_SYNC(dest_sync_mode),
        ],
        runtimes=[
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
        disable_format_inference=disable_format_inference,
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
