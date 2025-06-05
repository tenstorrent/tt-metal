# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from helpers.device import (
    collect_results,
    run_elf_files,
    wait_for_tensix_operations_finished,
    write_stimuli_to_l1,
)
from helpers.format_arg_mapping import DestAccumulation, format_dict
from helpers.format_config import DataFormat
from helpers.param_config import (
    clean_params,
    generate_param_ids,
    generate_params,
    input_output_formats,
)
from helpers.stimuli_generator import flatten_list, generate_stimuli
from helpers.test_config import generate_make_command
from helpers.utils import passed_test, run_shell_command


def generate_golden(operations, operand1, operand2, data_format):
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

    res = []

    # to se why this encoding look at llk_defs.h -> enum EltwiseBinaryType

    for op in operations:
        if op == 0:
            res_tmp = tensor1_float * tensor2_float
        elif op == 1:
            res_tmp = tensor1_float / tensor2_float
        elif op == 2:
            res_tmp = tensor1_float + tensor2_float
        elif op == 3:
            res_tmp = tensor1_float - tensor2_float
        else:
            raise ValueError("Unsupported operation!")

        res.append(res_tmp.tolist())

    return flatten_list(res)


# SUPPORTED FORMATS FOR TEST
supported_formats = [DataFormat.Bfp8_b, DataFormat.Float16, DataFormat.Float16_b]

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
    ["fill_dest_test"],
    test_formats,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize(
    "testname, formats, dest_acc", clean_params(all_params), ids=param_ids
)
def test_fill_dest(testname, formats, dest_acc):

    pack_start_address = 0x1C000
    pack_addresses = [pack_start_address + 0x1000 * i for i in range(16)]

    src_A, src_B = generate_stimuli(formats.input_format, formats.input_format)
    golden = generate_golden([2] * 16, src_A, src_B, formats.output_format)
    write_stimuli_to_l1(src_A, src_B, formats.input_format, formats.input_format)

    test_config = {
        "formats": formats,
        "testname": testname,
        "dest_acc": dest_acc,
    }

    make_cmd = generate_make_command(test_config)
    run_shell_command(f"cd .. && {make_cmd}")

    run_elf_files(testname)

    wait_for_tensix_operations_finished()
    res_from_L1 = []
    for address in pack_addresses:
        res_from_L1.append(
            collect_results(formats, tensor_size=len(src_A), address=address)
        )
    res_from_L1 = flatten_list(res_from_L1)

    assert len(res_from_L1) == len(golden)

    golden_tensor = torch.tensor(
        golden,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format in [DataFormat.Float16, DataFormat.Float16_b]
            else torch.bfloat16
        ),
    )
    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format in [DataFormat.Float16, DataFormat.Float16_b]
            else torch.bfloat16
        ),
    )

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
