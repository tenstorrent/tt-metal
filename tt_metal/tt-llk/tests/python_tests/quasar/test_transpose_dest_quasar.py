# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from helpers.dest_params import (
    UnpackPath,
    dest_acc_modes,
    dest_sync_modes,
    unpack_to_dest_modes,
)
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    DataCopyGolden,
    TransposeGolden,
    get_golden_generator,
    quantize_mx_tensor_chunked,
)
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    PerfRunType,
    Transpose,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    BlocksCalculationAlgorithm,
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
    select_perf_input_dimensions,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    MATH_TRANSPOSE_FACES,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
    generate_input_dim,
)
from helpers.tile_constants import FACE_C_DIM, get_tile_params
from helpers.utils import passed_test

TILE_DIMENSIONS = [32, 32]

# Curated dimensions: some fit in one bank (no switching), some require
# multiple blocks (triggering dest bank switches with DstSync.Half).
TRANSPOSE_DEST_DIMENSIONS = {
    (DestAccumulation.No, DestSync.Half): [
        [32, 32],
        [32, 128],
        [32, 256],
        [32, 512],
        [64, 384],
    ],
    (DestAccumulation.No, DestSync.Full): [
        [32, 32],
        [32, 512],
        [64, 512],
    ],
    (DestAccumulation.Yes, DestSync.Half): [
        [32, 32],
        [32, 128],
        [32, 256],
        [64, 192],
    ],
    (DestAccumulation.Yes, DestSync.Full): [
        [32, 32],
        [32, 256],
        [32, 512],
    ],
}


def _transpose_dest_supported(formats):
    return not (formats.input_format.is_integer() ^ formats.output_format.is_integer())


TRANSPOSE_DEST_FORMATS = [
    fmt
    for fmt in input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.Int8,
            DataFormat.UInt8,
            DataFormat.MxInt8,
            DataFormat.MxInt4,
            DataFormat.MxInt2,
        ]
    )
    if _transpose_dest_supported(fmt)
]


def transpose_dest_dest_acc(formats):
    allowed = (
        [DestAccumulation.Yes]
        if formats.input_format.is_32_bit()
        else [DestAccumulation.No]
    )
    return dest_acc_modes(formats, allowed=allowed)


def transpose_dest_unpack_to_dest(formats, dest_acc):
    return unpack_to_dest_modes(
        formats,
        dest_acc,
        path=(
            UnpackPath.ForceTrue
            if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
            else UnpackPath.ForceFalse
        ),
    )


def transpose_dest_input_dimensions(dest_acc, dest_sync, *, is_perf=False):
    mode_dimensions = TRANSPOSE_DEST_DIMENSIONS[(dest_acc, dest_sync)]
    if not is_perf:
        return mode_dimensions
    perf_dimensions = select_perf_input_dimensions(
        mode_dimensions, use_largest_fallback=False
    )
    three_block = [64, 384]
    if three_block in mode_dimensions and three_block not in perf_dimensions:
        perf_dimensions.append(three_block)
    return perf_dimensions


def transpose_dest_implied_math_formats(*, is_perf=False):
    return (
        [ImpliedMathFormat.Yes]
        if is_perf
        else [ImpliedMathFormat.No, ImpliedMathFormat.Yes]
    )


@pytest.mark.quasar
@parametrize(
    formats=TRANSPOSE_DEST_FORMATS,
    dest_acc=transpose_dest_dest_acc,
    dest_sync=lambda: dest_sync_modes(is_perf=False),
    unpack_to_dest=transpose_dest_unpack_to_dest,
    math_transpose_faces=[Transpose.No, Transpose.Yes],
    input_dimensions=lambda dest_acc, dest_sync: transpose_dest_input_dimensions(
        dest_acc, dest_sync, is_perf=False
    ),
    implied_math_format=lambda: transpose_dest_implied_math_formats(is_perf=False),
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_transpose_dest_quasar(
    formats,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    math_transpose_faces,
    input_dimensions,
    implied_math_format,
    run_types,
    loop_factor,
    *,
    is_perf=False,
    perf_report=None,
):
    data_copy_type = DataCopyType.A2D
    tile_rows, tile_cols = TILE_DIMENSIONS
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(
        [tile_rows, tile_cols]
    )
    num_faces = num_faces_r_dim * num_faces_c_dim

    output_num_blocks, output_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Generate custom test input stimuli to check large Int32 and Float32 values
    if (
        formats.input_format == DataFormat.Int32
        and formats.output_format == DataFormat.Int32
    ):
        lo, hi = -1_000_000, 1_000_000
        n = src_A.numel()
        src_A = torch.randint(lo, hi, (n,), dtype=torch.int32).reshape_as(src_A)
        src_B = torch.randint(lo, hi, (n,), dtype=torch.int32).reshape_as(src_B)

    if (
        formats.input_format == DataFormat.Float32
        and not formats.output_format.is_mx_format()
    ):
        # The *10000 scaling stresses Int32/Float32 output paths with large
        # values, but MxInt8 cannot represent that dynamic range losslessly
        # (block-exp at ~14, per-element step ~256). Keep small-range stimuli
        # for MX outputs so quantization stays within tolerance.
        n = src_A.numel()
        src_A = (torch.randn(n, dtype=torch.float32) * 10000.0).reshape_as(src_A)
        src_B = (torch.randn(n, dtype=torch.float32) * 10000.0).reshape_as(src_B)

    # For MX output formats, defer the MX quantization until after the transpose.
    # HW transposes inside Dest at math precision (bf16), then pack re-derives
    # block exponents from the post-transpose layout. Quantizing inside
    # DataCopyGolden locks in pre-transpose block exponents that don't follow
    # elements through the 16x16 face transpose, producing wrong shared scales.
    # This matters most for MX-input cases, where the input-dequant roundtrip
    # increases per-block variance and amplifies the order-dependence.
    is_mx_output = formats.output_format.is_mx_format()
    intermediate_format = (
        DataFormat.Float16_b if is_mx_output else formats.output_format
    )

    generate_datacopy_golden = get_golden_generator(DataCopyGolden)
    datacopy_tensor = generate_datacopy_golden(
        src_A,
        intermediate_format,
        num_faces=num_faces,
        input_dimensions=input_dimensions,
        input_format=formats.input_format,
    )

    t_matrix = get_golden_generator(TransposeGolden)
    golden_tensor = t_matrix.transpose_within_faces_multi_tile(
        datacopy_tensor,
        intermediate_format,
        num_tiles=tile_cnt_A,
        untilize=False,
        input_dimensions=input_dimensions,
    )
    if math_transpose_faces == Transpose.Yes:
        golden_tensor = t_matrix.transpose_faces_multi_tile(
            golden_tensor,
            intermediate_format,
            num_tiles=tile_cnt_A,
            tilize=False,
            input_dimensions=input_dimensions,
        )

    if is_mx_output:
        golden_tensor = quantize_mx_tensor_chunked(
            golden_tensor.to(torch.bfloat16), formats.output_format
        )

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/transpose_dest_quasar_test.cpp",
        "formats": formats,
        "templates": [
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(data_copy_type),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(dest_sync),
            MATH_TRANSPOSE_FACES(math_transpose_faces),
        ],
        "runtimes": [
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            NUM_TILES_IN_BLOCK(
                output_tiles_in_block,
                input_num_tiles_in_block=output_tiles_in_block,
                output_num_tiles_in_block=output_tiles_in_block,
            ),
            NUM_BLOCKS(
                output_num_blocks,
                input_num_blocks=output_num_blocks,
                output_num_blocks=output_num_blocks,
            ),
            TEST_FACE_DIMS(face_r_dim=face_r_dim, face_c_dim=FACE_C_DIM),
            NUM_FACES_R_DIM(num_faces_r_dim),
            NUM_FACES_C_DIM(num_faces_c_dim),
            DEST_INDEX(),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=TILE_DIMENSIONS,
            use_dense_tile_dimensions=True,
        ),
        "unpack_to_dest": unpack_to_dest,
        "dest_acc": dest_acc,
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )
    if is_perf:
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
    ), "Assert against golden failed"
