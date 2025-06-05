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
from helpers.format_arg_mapping import DestAccumulation, MathFidelity, format_dict
from helpers.format_config import DataFormat
from helpers.param_config import (
    clean_params,
    generate_param_ids,
    generate_params,
    input_output_formats,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import generate_make_command
from helpers.tilize_untilize import tilize
from helpers.utils import passed_test, run_shell_command


def generate_golden(operand1, operand2, data_format, math_fidelity):
    torch_format = format_dict.get(data_format, format_dict[DataFormat.Float16_b])

    operand1_matrix = operand1.view(32, 32).to(torch_format)
    operand2_matrix = operand2.view(32, 32).to(torch_format)

    result_matrix = torch.matmul(operand1_matrix, operand2_matrix)

    return result_matrix.flatten()


# SUPPORTED FORMATS FOR TEST
supported_formats = [
    DataFormat.Float16_b,
    DataFormat.Float16,
    DataFormat.Bfp8_b,
    DataFormat.Float32,
]

#   INPUT-OUTPUT FORMAT SWEEP
#   input_output_formats(supported_formats)

#   FULL FORMAT SWEEP
#   format_combination_sweep(formats=supported_formats, all_same=False, same_src_reg_format=True)

#   SPECIFIC FORMAT COMBINATION
#   generate_combination(
#       [(DataFormat.Float16_b,  # index 0 is for unpack_A_src
#         DataFormat.Float16_b,  # index 1 is for unpack_A_dst
#         DataFormat.Float16_b,  # index 2 is for pack_src (if src registers have same formats)
#         DataFormat.Float16_b,  # index 3 is for pack_dst
#         DataFormat.Float16_b,  # index 4 is for math format)])

#   SPECIFIC INPUT-OUTPUT COMBINATION
#   [InputOutputFormat(DataFormat.Float16, DataFormat.Float32)]

test_formats = input_output_formats(supported_formats)
all_params = generate_params(
    ["matmul_pack_untilize_test"],
    test_formats,
    dest_acc=[DestAccumulation.Yes, DestAccumulation.No],
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
)
param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize(
    "testname, formats, dest_acc, math_fidelity",
    clean_params(all_params),
    ids=param_ids,
)
def test_matmul_pack_untilize(testname, formats, dest_acc, math_fidelity):
    arch = get_chip_architecture()
    if formats.output == DataFormat.Bfp8_b and arch == ChipArchitecture.WORMHOLE:
        pytest.skip("Pack untilize does not support Bfp8_b")

    torch_format = format_dict.get(
        formats.output_format, format_dict[DataFormat.Float16_b]
    )

    src_A, src_B = generate_stimuli(formats.input_format, formats.input_format)

    golden_tensor = generate_golden(src_A, src_B, formats.output_format, math_fidelity)

    write_stimuli_to_l1(
        tilize(src_A, torch_format),
        tilize(src_B, torch_format),
        formats.input_format,
        formats.input_format,
    )

    test_config = {
        "formats": formats,
        "testname": testname,
        "dest_acc": dest_acc,
        "math_fidelity": math_fidelity,
    }

    make_cmd = generate_make_command(test_config)
    run_shell_command(f"cd .. && {make_cmd}")

    run_elf_files(testname)

    wait_for_tensix_operations_finished()
    res_from_L1 = collect_results(formats, tensor_size=len(src_A))
    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(res_from_L1, dtype=(torch_format))

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
