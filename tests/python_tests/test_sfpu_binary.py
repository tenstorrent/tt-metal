# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.golden_generators import BinarySFPUGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathOperation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    MATH_OP,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.utils import passed_test


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Float16,
            DataFormat.Float16_b,
            DataFormat.Bfp8_b,
        ]
    ),
    mathop=[
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwmul,
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_binary_float(formats, dest_acc, mathop, workers_tensix_coordinates):
    if (
        TestConfig.CHIP_ARCH == ChipArchitecture.WORMHOLE
        and mathop == MathOperation.SfpuElwsub
    ):
        pytest.skip("Not currently supported in tests")

    if (
        TestConfig.CHIP_ARCH == ChipArchitecture.WORMHOLE
        and mathop in [MathOperation.SfpuElwadd, MathOperation.SfpuElwmul]
        and dest_acc == DestAccumulation.No
        and formats.input_format == DataFormat.Float32
    ):
        pytest.skip(reason="https://github.com/tenstorrent/tt-llk/issues/1092")

    if (
        TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
        and formats.input_format == DataFormat.Float16
        and dest_acc == DestAccumulation.No
    ):
        pytest.skip(
            "Float16_a isn't supported for SFPU on Blackhole without being converted to 32-bit intermediate format in dest register"
        )

    sfpu_binary(formats, dest_acc, mathop, workers_tensix_coordinates)


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Int32,
        ]
    ),
    mathop=[
        MathOperation.SfpuElwRightShift,
        MathOperation.SfpuElwLeftShift,
        MathOperation.SfpuElwLogicalRightShift,
    ],
    dest_acc=[DestAccumulation.Yes],
)
def test_sfpu_binary_int(formats, dest_acc, mathop, workers_tensix_coordinates):
    sfpu_binary(formats, dest_acc, mathop, workers_tensix_coordinates)


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.UInt32,
        ],
        same=True,
    ),
    mathop=[MathOperation.SfpuAddTopRow],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
def test_sfpu_binary_add_top_row(formats, dest_acc, mathop, workers_tensix_coordinates):
    input_dimensions = [64, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=True,
    )

    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_tensor = generate_golden(
        mathop,
        src_A,
        0,
        1,
        0,
        1,
        input_dimensions,
        formats.output_format,
    )

    golden_tensor = (
        golden_tensor.view([32, 32])
        if golden_tensor.shape == torch.Size([1024])
        else golden_tensor.view(input_dimensions)
    )

    configuration = TestConfig(
        "sources/sfpu_binary_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop),
            APPROX_MODE(),
        ],
        runtimes=[TILE_COUNT(tile_cnt_A)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
        disable_format_inference=True,
    )
    res_from_L1 = configuration.run(workers_tensix_coordinates).result

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).view(input_dimensions)

    assert len(res_tensor) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"


def sfpu_binary(formats, dest_acc, mathop, workers_tensix_coordinates):

    input_dimensions = [64, 32]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_tensor = generate_golden(
        mathop,
        src_A,  # Contains tiles 0 and 1
        0,  # src1_idx: use tile 0
        1,  # src2_idx: use tile 1
        0,  # dst_idx: write to tile 0
        32,  # num_iterations: 32 rows
        input_dimensions,  # [64, 32] = 2 tiles
        (
            DataFormat.Float16_b
            if formats.input_format == DataFormat.Bfp8_b
            else formats.input_format
        ),
    ).flatten()

    # ONLY Blackhole needs this for some reason
    if (
        formats.input_format in [DataFormat.Float16, DataFormat.Float32]
        and TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE
    ):
        dest_acc = DestAccumulation.Yes

    configuration = TestConfig(
        "sources/sfpu_binary_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop),
            APPROX_MODE(),
        ],
        runtimes=[TILE_COUNT(tile_cnt_A)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
    )
    res_from_L1 = configuration.run(workers_tensix_coordinates).result

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).flatten()

    assert len(res_tensor) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
