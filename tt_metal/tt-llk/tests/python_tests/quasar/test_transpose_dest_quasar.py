# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
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


def generate_qsr_transpose_dest_combinations(
    formats_list: List[FormatConfig],
    *,
    is_perf=False,
):
    """
    Generate transpose dest combinations for Quasar tests.

    Args:
        formats_list: List of input/output format pairs

    Returns:
        List of (format, dest_acc, dest_sync, math_transpose_faces) tuples
    """

    def is_supported_format_conversion(in_fmt, out_fmt):
        """Check if the format conversion is supported by packer. These format conversions are NOT dependent on the dest register mode."""
        # Skip if mixing integer and non-integer formats
        if in_fmt.is_integer() ^ out_fmt.is_integer():
            return False
        return True

    def get_dest_acc_modes(in_fmt):
        """Determine valid dest register modes depending on the input format."""
        # Int32, Float32 (unpack_to_dest) requires 32bit mode dest register
        if in_fmt.is_32_bit():
            return (DestAccumulation.Yes,)
        # Int8/UInt8 in Src regs and Int32 in dest reg is unsupported for MOVB2D
        # Float16/Float16_b in Src regs and Float32 in dest reg is unsupported for MOVB2D
        return (DestAccumulation.No,)

    def is_supported_dest_mode_dependent_conversion(in_fmt, out_fmt, dest_acc):
        """Check if the format conversion is supported by packer. These format conversions are dependent on the dest register mode."""
        # Upcasting to Float32/Int32 requires dest_acc enabled
        if (
            out_fmt.is_32_bit()
            and not in_fmt.is_32_bit()
            and dest_acc == DestAccumulation.No
        ):
            return False
        # Int8<->UInt8 conversion requires dest_acc enabled
        if (
            dest_acc == DestAccumulation.No
            and in_fmt in (DataFormat.Int8, DataFormat.UInt8)
            and in_fmt != out_fmt
        ):
            return False
        return True

    # Curated dimensions: some fit in one bank (no switching), some require
    # multiple blocks (triggering dest bank switches with DstSync.Half).
    # DstSync.Half capacity: 8 tiles (16-bit dest) / 4 tiles (32-bit dest)
    # DstSync.Full capacity: 16 tiles (16-bit dest) / 8 tiles (32-bit dest)
    dimensions_by_mode = {
        (DestAccumulation.No, DestSync.Half): [
            [32, 32],  # 1 tile  → 1 block (no switch)
            [32, 128],  # 4 tiles → 1 block (no switch)
            [32, 256],  # 8 tiles → 1 block (fills half-dest exactly)
            [32, 512],  # 16 tiles → 2 blocks (1 bank switch)
            [64, 384],  # 24 tiles → 3 blocks (2 bank switches)
        ],
        (DestAccumulation.No, DestSync.Full): [
            [32, 32],  # 1 tile  → 1 block
            [32, 512],  # 16 tiles → 1 block (fills full-dest exactly)
            [64, 512],  # 32 tiles → 2 blocks
        ],
        (DestAccumulation.Yes, DestSync.Half): [
            [32, 32],  # 1 tile  → 1 block (no switch)
            [32, 128],  # 4 tiles → 1 block (fills half-dest exactly)
            [32, 256],  # 8 tiles → 2 blocks (1 bank switch)
            [64, 192],  # 12 tiles → 3 blocks (2 bank switches)
        ],
        (DestAccumulation.Yes, DestSync.Full): [
            [32, 32],  # 1 tile  → 1 block
            [32, 256],  # 8 tiles → 1 block (fills full-dest exactly)
            [32, 512],  # 16 tiles → 2 blocks
        ],
    }

    dest_sync_modes = (DestSync.Half,) if is_perf else (DestSync.Half, DestSync.Full)
    transpose_faces_modes = (Transpose.No, Transpose.Yes)
    combinations = []
    for fmt in formats_list:
        in_fmt, out_fmt = fmt.input_format, fmt.output_format

        if not is_supported_format_conversion(in_fmt, out_fmt):
            continue

        for dest_acc in get_dest_acc_modes(in_fmt):
            if is_supported_dest_mode_dependent_conversion(in_fmt, out_fmt, dest_acc):
                for dest_sync in dest_sync_modes:
                    for math_transpose_faces in transpose_faces_modes:
                        if is_perf:
                            mode_dimensions = dimensions_by_mode[(dest_acc, dest_sync)]
                            perf_dimensions = select_perf_input_dimensions(
                                mode_dimensions,
                                use_largest_fallback=False,
                            )
                            # Dest-full vs 2-block is selected via PERF_INPUT_DIMENSIONS.
                            # Keep the 3-block / 2-switch case when the mode defines it.
                            three_block = [64, 384]
                            if (
                                three_block in mode_dimensions
                                and three_block not in perf_dimensions
                            ):
                                perf_dimensions.append(three_block)
                            for dimensions in perf_dimensions:
                                combinations.append(
                                    (
                                        fmt,
                                        dest_acc,
                                        dest_sync,
                                        math_transpose_faces,
                                        dimensions,
                                    )
                                )
                            continue
                        for dimensions in dimensions_by_mode[(dest_acc, dest_sync)]:
                            combinations.append(
                                (
                                    fmt,
                                    dest_acc,
                                    dest_sync,
                                    math_transpose_faces,
                                    dimensions,
                                )
                            )

    return combinations


def transpose_dest_implied_math_formats(*, is_perf=False):
    return (
        [ImpliedMathFormat.Yes]
        if is_perf
        else [ImpliedMathFormat.No, ImpliedMathFormat.Yes]
    )


TRANSPOSE_DEST_FORMATS = input_output_formats(
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
    ],
)
PERF_TRANSPOSE_DEST_COMBINATIONS = generate_qsr_transpose_dest_combinations(
    TRANSPOSE_DEST_FORMATS,
    is_perf=True,
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_transpose_dims=generate_qsr_transpose_dest_combinations(
        TRANSPOSE_DEST_FORMATS
    ),
    implied_math_format=lambda: transpose_dest_implied_math_formats(is_perf=False),
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_transpose_dest_quasar(
    formats_dest_acc_sync_transpose_dims,
    implied_math_format,
    run_types,
    loop_factor,
    *,
    is_perf=False,
    perf_report=None,
):
    (formats, dest_acc, dest_sync, math_transpose_faces, input_dimensions) = (
        formats_dest_acc_sync_transpose_dims
    )

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

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
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
