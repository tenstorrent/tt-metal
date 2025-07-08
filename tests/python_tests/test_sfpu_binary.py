# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.device import (
    collect_results,
    write_stimuli_to_l1,
)
from helpers.format_arg_mapping import DestAccumulation, MathOperation, format_dict
from helpers.format_config import DataFormat
from helpers.golden_generators import BinarySFPUGolden, get_golden_generator
from helpers.param_config import (
    clean_params,
    generate_param_ids,
    generate_params,
    input_output_formats,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.utils import passed_test

# SUPPORTED FORMATS FOR TEST
supported_float_formats = [
    DataFormat.Float32,
    DataFormat.Float16,
    DataFormat.Float16_b,
    DataFormat.Bfp8_b,
]
supported_int_formats = [DataFormat.Int32]

#   INPUT-OUTPUT FORMAT SWEEP
#   input_output_formats(supported_formats)

#   FULL FORMAT SWEEP
#   format_combination_sweep(formats=supported_formats, all_same=False, same_src_reg_format=True)

#   SPECIFIC FORMAT COMBINATION
#   generate_combination(
#       [(DataFormat.Float16_b,  # index 0 is for unpack_A_src
#         DataFormat.Float16_b,  # index 1 is for unpack_A_dst
#         DataFormat.Float16_b,  # index 2 is for pack_src (if src registers have same formats)
#         DataFormat.Bfp8_b,  # index 3 is for pack_dst
#         DataFormat.Float16_b,)]) # index 4 is for math format

#   SPECIFIC INPUT-OUTPUT COMBINATION
#   [InputOutputFormat(DataFormat.Float16, DataFormat.Float32)]

float_ops = [
    MathOperation.SfpuElwadd,
    MathOperation.SfpuElwsub,
    MathOperation.SfpuElwmul,
]

int_ops = [
    MathOperation.SfpuElwRightShift,
    MathOperation.SfpuElwLeftShift,
    MathOperation.SfpuElwLogicalRightShift,
]

float_params = generate_params(
    ["sfpu_binary_test"],
    input_output_formats(supported_float_formats),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=float_ops,
)

int_params = generate_params(
    ["sfpu_binary_test"],
    input_output_formats(supported_int_formats),
    dest_acc=[DestAccumulation.Yes],
    mathop=int_ops,
)

all_params = float_params + int_params

param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize(
    "testname, formats, dest_acc, mathop", clean_params(all_params), ids=param_ids
)
def test_sfpu_binary(testname, formats, dest_acc, mathop):

    input_dimensions = [64, 64]

    chip_arch = get_chip_architecture()
    if chip_arch == ChipArchitecture.WORMHOLE and mathop == MathOperation.SfpuElwsub:
        pytest.skip("Not currently supported in tests")

    if (
        dest_acc == DestAccumulation.No
        and chip_arch == ChipArchitecture.BLACKHOLE
        and formats.input_format == DataFormat.Float16
    ):
        pytest.skip(
            "Float16_a isn't supported for SFPU on Blackhole without being converted to 32-bit intermediate format in dest register"
        )

    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format, formats.input_format, input_dimensions=input_dimensions
    )

    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_tensor = generate_golden(mathop, src_A, src_B, formats.output_format)
    res_address = write_stimuli_to_l1(
        src_A, src_B, formats.input_format, formats.input_format, tile_count=tile_cnt
    )

    unpack_to_dest = formats.input_format.is_32_bit()

    # Blackhole needs this for some reason
    if formats.input_format in [DataFormat.Float16, DataFormat.Float32]:
        dest_acc = DestAccumulation.Yes

    test_config = {
        "formats": formats,
        "testname": testname,
        "dest_acc": dest_acc,
        "mathop": mathop,
        "unpack_to_dest": unpack_to_dest,
        "tile_cnt": tile_cnt,
    }

    run_test(test_config)

    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)

    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
