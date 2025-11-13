# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.device import collect_results, write_stimuli_to_l1
from helpers.format_config import DataFormat
from helpers.golden_generators import BinarySFPUGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathOperation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.tilize_untilize import untilize
from helpers.utils import passed_test


@parametrize(
    test_name="sfpu_binary_test",
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
def test_sfpu_binary_float(test_name, formats, dest_acc, mathop):
    chip_arch = get_chip_architecture()
    if chip_arch == ChipArchitecture.WORMHOLE and mathop == MathOperation.SfpuElwsub:
        pytest.skip("Not currently supported in tests")

    if (
        chip_arch == ChipArchitecture.BLACKHOLE
        and formats.input_format == DataFormat.Float16
        and dest_acc == DestAccumulation.No
    ):
        pytest.skip(
            "Float16_a isn't supported for SFPU on Blackhole without being converted to 32-bit intermediate format in dest register"
        )

    sfpu_binary(test_name, formats, dest_acc, mathop)


@parametrize(
    test_name="sfpu_binary_test",
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
def test_sfpu_binary_int(test_name, formats, dest_acc, mathop):
    sfpu_binary(test_name, formats, dest_acc, mathop)


@parametrize(
    test_name="sfpu_binary_test",
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
def test_sfpu_binary_add_top_row(test_name, formats, dest_acc, mathop):
    input_dimensions = [32, 32]

    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format, formats.input_format, input_dimensions=input_dimensions
    )

    src_A_top_row = untilize(src_A, stimuli_format=formats.input_format)[:32]
    src_B_top_row = untilize(src_B, stimuli_format=formats.input_format)[:32]
    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_tensor = generate_golden(
        mathop, src_A_top_row, src_B_top_row, formats.output_format
    )

    unpack_to_dest = formats.input_format.is_32_bit()

    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "input_A_dimensions": input_dimensions,
        "input_B_dimensions": input_dimensions,
        "mathop": mathop,
        "unpack_to_dest": unpack_to_dest,
        "tile_cnt": tile_cnt,
        "disable_format_inference": True,
    }

    res_address = write_stimuli_to_l1(
        test_config,
        src_A,
        src_B,
        formats.input_format,
        formats.input_format,
        tile_count_A=tile_cnt,
        tile_count_B=tile_cnt,
    )

    run_test(test_config)

    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # For add_top_row, we only compare the top row
    # Untilize the single tile and extract the first 32 elements (first row)
    untilized_tile = untilize(res_tensor, stimuli_format=formats.output_format)
    res_tensor = untilized_tile[:32]

    assert len(res_tensor) == len(golden_tensor)

    assert passed_test(golden_tensor, res_tensor, formats.output_format)


def sfpu_binary(test_name, formats, dest_acc, mathop):

    input_dimensions = [64, 64]

    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format, formats.input_format, input_dimensions=input_dimensions
    )

    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_tensor = generate_golden(mathop, src_A, src_B, formats.output_format)

    unpack_to_dest = formats.input_format.is_32_bit()

    # Blackhole needs this for some reason
    if formats.input_format in [DataFormat.Float16, DataFormat.Float32]:
        dest_acc = DestAccumulation.Yes

    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "input_A_dimensions": input_dimensions,
        "input_B_dimensions": input_dimensions,
        "mathop": mathop,
        "unpack_to_dest": unpack_to_dest,
        "tile_cnt": tile_cnt,
    }

    res_address = write_stimuli_to_l1(
        test_config,
        src_A,
        src_B,
        formats.input_format,
        formats.input_format,
        tile_count_A=tile_cnt,
        tile_count_B=tile_cnt,
    )

    run_test(test_config)

    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert len(res_tensor) == len(golden_tensor)

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
