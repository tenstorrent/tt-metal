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
from helpers.utils import compare_pcc, run_shell_command

torch.set_printoptions(linewidth=500, sci_mode=False, precision=2, threshold=10000)


def generate_golden(operand1, operand2, data_format, math_fidelity):

    if data_format == DataFormat.Float16_b:
        if math_fidelity in [MathFidelity.LoFi, MathFidelity.HiFi2]:  # LoFi or HiFi2
            for element in operand2:
                element = element.to(torch.int32)
                element &= 0xFFFE
        if math_fidelity == MathFidelity.LoFi:  # LoFi
            for element in operand1:
                element = element.to(torch.int32)
                element &= 0xFFF8

    operand1_matrix = operand1.view(32, 32).to(format_dict[data_format])
    operand2_matrix = operand2.view(32, 32).to(format_dict[data_format])

    result_matrix = torch.matmul(operand1_matrix, operand2_matrix)

    return result_matrix.flatten()


# SUPPORTED FORMATS FOR TEST
supported_formats = [DataFormat.Float16, DataFormat.Float16_b]

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
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
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

    src_A, src_B = generate_stimuli(formats.input_format, formats.input_format)

    golden_tensor = generate_golden(src_A, src_B, formats.output_format, math_fidelity)
    golden_tensor = golden_tensor.to(format_dict[formats.output_format])

    write_stimuli_to_l1(
        tilize(src_A, format_dict[formats.input_format]),
        tilize(src_B, format_dict[formats.input_format]),
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

    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format in [DataFormat.Float16, DataFormat.Float16_b]
            else torch.bfloat16
        ),
    )

    if formats.output_format in [DataFormat.Float16_b, DataFormat.Float16]:
        atol = 0.1
        rtol = 0.05
    elif formats.output_format == DataFormat.Bfp8_b:
        atol = 0.1
        rtol = 0.2

    for i in range(len(golden_tensor)):
        assert torch.isclose(
            golden_tensor[i], res_tensor[i], rtol=rtol, atol=atol
        ), f"Failed at index {i} with values {golden_tensor[i]} and {res_from_L1[i]}"

    _, pcc = compare_pcc(golden_tensor, res_tensor, pcc=0.99)
    assert pcc > 0.98
