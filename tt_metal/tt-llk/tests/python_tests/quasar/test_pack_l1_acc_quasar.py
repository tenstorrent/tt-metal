# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import (
    PackGolden,
    get_golden_generator,
    quantize_mx_tensor_chunked,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    PerfRunType,
    ReluConfig,
    format_dict,
)
from helpers.param_config import (
    BlocksCalculationAlgorithm,
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BootMode, TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PERF_RUN_TYPE,
    RELU_CONFIG,
    TEST_FACE_DIMS,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tile_constants import FACE_C_DIM, get_tile_params
from helpers.utils import passed_test

INPUT_DIMENSIONS = [[512, 64], [192, 512]]
# Nested list of [H, W] pairs: a flat [H, W] is expanded by parametrize into
# input_dimensions=H (int) and breaks generate_stimuli / rows, cols = dims.
PERF_INPUT_DIMENSIONS = [[512, 64]]
PERF_ONLY_INPUT_DIMENSIONS = [[512, 64]]
TILE_DIMENSIONS = [32, 32]
# Complete list of formats that are supported with L1 accumulation as the
# OUTPUT format. MX formats (MxInt8) are allowed only as INPUT — accumulation
# happens on the packed output in L1, and MX formats do not support L1 acc.
PACK_L1_ACC_FORMATS = input_output_formats(
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


def generate_qsr_pack_l1_acc_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate pack combinations for Quasar pack with L1 accumulation tests.

    Args:
        formats_list: List of input/output format pairs

    Returns:
        List of (format, dest_acc) tuples
    """

    def is_supported_format_conversion(in_fmt, out_fmt):
        """Check if the format conversion is supported by packer. These format conversions are NOT dependent on the dest register mode."""
        # Skip if mixing integer and non-integer formats
        if in_fmt.is_integer() ^ out_fmt.is_integer():
            return False
        # MX formats are not L1-accumulation-capable as the output; allow them
        # only on the input side.
        if out_fmt.is_mx_format():
            return False
        return True

    def get_dest_acc_modes(in_fmt):
        """Determine valid dest register modes depending on the input format."""
        # Int32, Float32 (unpack_to_dest) requires 32bit mode dest register
        if in_fmt.is_32_bit():
            return (DestAccumulation.Yes,)
        return (DestAccumulation.No, DestAccumulation.Yes)

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

    combinations = []
    for fmt in formats_list:
        in_fmt, out_fmt = fmt.input_format, fmt.output_format

        if not is_supported_format_conversion(in_fmt, out_fmt):
            continue

        for dest_acc in get_dest_acc_modes(in_fmt):
            if is_supported_dest_mode_dependent_conversion(in_fmt, out_fmt, dest_acc):
                combinations.append((fmt, dest_acc))

    return combinations


def pack_l1_acc_dest_sync_modes(*, is_perf=False):
    return [DestSync.Half] if is_perf else [DestSync.Half, DestSync.Full]


def pack_l1_acc_implied_math_formats(formats_dest_acc, *, is_perf=False):
    if is_perf:
        return [ImpliedMathFormat.Yes]
    formats = formats_dest_acc[0]
    if formats.input_format.is_mx_format():
        return [ImpliedMathFormat.Yes]
    return [ImpliedMathFormat.No, ImpliedMathFormat.Yes]


def pack_l1_acc_input_dimensions(*, is_perf=False):
    return PERF_ONLY_INPUT_DIMENSIONS if is_perf else INPUT_DIMENSIONS


ALL_PACK_L1_ACC_COMBINATIONS = generate_qsr_pack_l1_acc_combinations(
    PACK_L1_ACC_FORMATS
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc=ALL_PACK_L1_ACC_COMBINATIONS,
    implied_math_format=lambda formats_dest_acc: pack_l1_acc_implied_math_formats(
        formats_dest_acc
    ),
    dest_sync_mode=lambda: pack_l1_acc_dest_sync_modes(is_perf=False),
    input_dimensions=runtime(lambda: pack_l1_acc_input_dimensions(is_perf=False)),
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_pack_l1_acc_quasar(
    formats_dest_acc,
    implied_math_format,
    dest_sync_mode,
    input_dimensions,
    run_types,
    loop_factor,
    boot_mode=BootMode.DEFAULT,
    *,
    is_perf=False,
    perf_report=None,
):
    (formats, dest_acc) = formats_dest_acc

    tile_rows, tile_cols = TILE_DIMENSIONS
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(
        [tile_rows, tile_cols]
    )
    num_faces = num_faces_r_dim * num_faces_c_dim

    rows, cols = input_dimensions
    tile_cnt = (rows // tile_rows) * (cols // tile_cols)

    output_num_blocks, output_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        dest_sync_mode,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=TILE_DIMENSIONS,
    )

    if not is_perf:
        # Quantize MX input through the source lattice so the golden sees what HW
        # sees after unpacking from L1. Without this, raw bfloat16 stimuli flow
        # into PackGolden while HW reads MxInt4-quantized values; per-block
        # accumulation then amplifies the per-element drift.
        src_A_golden = (
            quantize_mx_tensor_chunked(src_A, formats.input_format)
            if formats.input_format.is_mx_format()
            else src_A
        )

        generate_golden = get_golden_generator(PackGolden)
        full_golden = generate_golden(
            src_A_golden,
            formats.output_format,
            num_faces=num_faces,
            input_dimensions=input_dimensions,
            face_r_dim=face_r_dim,
        )

        # This test accumulates the results of each block on top of each other
        # Slice the full golden into per-block partials and accumulate
        elements_per_block = output_tiles_in_block * num_faces * face_r_dim * FACE_C_DIM
        partials = [
            full_golden[block * elements_per_block : (block + 1) * elements_per_block]
            for block in range(output_num_blocks)
        ]
        golden_tensor = generate_golden.accumulate_l1(
            partials, data_format=formats.output_format
        )

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/pack_l1_acc_quasar_test.cpp",
        "formats": formats,
        "templates": [
            IMPLIED_MATH_FORMAT(implied_math_format),
            DEST_SYNC(dest_sync_mode),
        ],
        "runtimes": [
            generate_input_dim(input_dimensions, input_dimensions),
            TILE_COUNT(tile_cnt),
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
            RELU_CONFIG(ReluConfig.NoRelu.value),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=output_tiles_in_block,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=TILE_DIMENSIONS,
            use_dense_tile_dimensions=True,
        ),
        "unpack_to_dest": unpack_to_dest,
        "dest_acc": dest_acc,
        "boot_mode": boot_mode,
        "disable_format_inference": formats.input_format.is_mx_format(),
    }

    if is_perf:
        configuration = PerfConfig(run_types=run_types, **test_config_kwargs)
        configuration.run(perf_report)
        return

    configuration = TestConfig(
        **{
            **test_config_kwargs,
            "templates": test_config_kwargs["templates"]
            + [PERF_RUN_TYPE(PerfRunType.L1_TO_L1)],
        },
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    test_passed = passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
    )

    assert test_passed, "Assert against golden failed"
