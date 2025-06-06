# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.device import (
    collect_results,
    run_elf_files,
    wait_for_tensix_operations_finished,
    write_stimuli_to_l1,
)
from helpers.format_arg_mapping import DestAccumulation, MathOperation, format_dict
from helpers.format_config import DataFormat
from helpers.param_config import (
    clean_params,
    generate_param_ids,
    generate_params,
    input_output_formats,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import generate_make_command
from helpers.utils import passed_test, run_shell_command


def generate_golden(operation, operand1, operand2, data_format):
    tensor1_float = (
        operand1.clone()
        .detach()
        .to(format_dict.get(data_format, format_dict[DataFormat.Float16_b]))
    )
    tensor2_float = (
        operand2.clone()
        .detach()
        .to(format_dict.get(data_format, format_dict[DataFormat.Float16_b]))
    )

    operations = {
        MathOperation.SfpuElwadd: tensor1_float + tensor2_float,
        MathOperation.SfpuElwsub: tensor1_float - tensor2_float,
        MathOperation.SfpuElwmul: tensor1_float * tensor2_float,
        MathOperation.SfpuXlogy: torch.xlogy(tensor1_float, tensor2_float),
    }

    if operation not in operations:
        raise ValueError("Unsupported operation!")

    return operations[operation].tolist()


# SUPPORTED FORMATS FOR TEST
supported_formats = [DataFormat.Float16_b]  # , DataFormat.Float16]

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
#         DataFormat.Float16_b,  # index 4 is for math format)])

#   SPECIFIC INPUT-OUTPUT COMBINATION
#   [InputOutputFormat(DataFormat.Float16, DataFormat.Float32)]

test_formats = input_output_formats(supported_formats)
all_params = generate_params(
    ["sfpu_binary_test"],
    test_formats,
    dest_acc=[DestAccumulation.No],  # , DestAccumulation.Yes],
    mathop=[
        MathOperation.SfpuElwsub,
        MathOperation.SfpuElwadd,
        MathOperation.SfpuElwmul,
        MathOperation.SfpuXlogy,
    ],
)
param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize(
    "testname, formats, dest_acc, mathop", clean_params(all_params), ids=param_ids
)
def test_all(testname, formats, dest_acc, mathop):

    chip_arch = get_chip_architecture()
    if chip_arch == ChipArchitecture.WORMHOLE and mathop == MathOperation.SfpuElwsub:
        pytest.skip("Not currently supported in tests")

    src_A, src_B = generate_stimuli(formats.input_format, formats.input_format)

    golden = generate_golden(mathop, src_A, src_B, formats.output_format)
    write_stimuli_to_l1(src_A, src_B, formats.input_format, formats.input_format)

    unpack_to_dest = formats.input_format == DataFormat.Float32

    test_config = {
        "formats": formats,
        "testname": testname,
        "dest_acc": dest_acc,
        "mathop": mathop,
        "unpack_to_dest": unpack_to_dest,
    }

    make_cmd = generate_make_command(test_config)
    run_shell_command(f"cd .. && {make_cmd}")

    run_elf_files(testname)
    wait_for_tensix_operations_finished()

    res_from_L1 = collect_results(formats, tensor_size=len(src_A))

    assert len(res_from_L1) == len(golden)

    golden_tensor = torch.tensor(
        golden,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format
            in [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )
    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format
            in [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
