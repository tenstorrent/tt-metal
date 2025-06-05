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
from helpers.format_arg_mapping import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
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


def generate_golden(op, operand1, operand2, data_format, math_fidelity):
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

    if data_format == DataFormat.Float16_b:
        if math_fidelity in [MathFidelity.LoFi, MathFidelity.HiFi2]:  # LoFi or HiFi2
            for element in operand2:
                element = element.to(torch.int32)
                element &= 0xFFFE
        if math_fidelity == MathFidelity.LoFi:  # LoFi
            for element in operand1:
                element = element.to(torch.int32)
                element &= 0xFFF8

    # First step is unpack tilize
    tensor1_float = tilize(tensor1_float, data_format)
    tensor2_float = tilize(tensor2_float, data_format)

    # Second step is to perform the operation
    if op == MathOperation.Elwadd:
        res = tensor1_float + tensor2_float
    elif op == MathOperation.Elwsub:
        res = tensor1_float - tensor2_float
    elif op == MathOperation.Elwmul:
        res = tensor1_float * tensor2_float
    else:
        raise ValueError("Unsupported operation!")

    # res = untilize(res, data_format)

    return res


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
#         DataFormat.Bfp8_b,  # index 3 is for pack_dst
#         DataFormat.Float16_b,  # index 4 is for math format)])

#   SPECIFIC INPUT-OUTPUT COMBINATION
#   [InputOutputFormat(DataFormat.Float16, DataFormat.Float32)]

test_formats = input_output_formats(supported_formats, same=True)
all_params = generate_params(
    ["tilize_calculate_untilize_L1"],
    test_formats,
    dest_acc=[DestAccumulation.No],
    mathop=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    math_fidelity=[MathFidelity.HiFi4],
    tile_cnt=1,
)
param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize(
    "testname, formats, dest_acc, mathop, math_fidelity, tile_cnt",
    clean_params(all_params),
    ids=param_ids,
)
def test_tilize_calculate_untilize_L1(
    testname, formats, dest_acc, mathop, math_fidelity, tile_cnt
):

    src_A, src_B = generate_stimuli(
        formats.input_format, formats.input_format, tile_cnt
    )

    golden_tensor = generate_golden(
        mathop, src_A, src_B, formats.output_format, math_fidelity
    )

    write_stimuli_to_l1(
        src_A, src_B, formats.input_format, formats.input_format, "0,0", tile_cnt
    )

    buffer_dest_address = 0x1E000  # Since this test calls LLK pipeline twise, unpacker will read at address in L1 that packer packed to, this address is able to be reaused for two LLK calls
    test_config = {
        "formats": formats,
        "testname": testname,
        "dest_acc": dest_acc,
        "math_fidelity": math_fidelity,
        "mathop": mathop,
    }

    make_cmd = generate_make_command(test_config)
    run_shell_command(f"cd .. && {make_cmd}")

    run_elf_files(testname)
    wait_for_tensix_operations_finished()

    res_from_L1 = collect_results(
        formats, tensor_size=len(src_A), address=buffer_dest_address
    )
    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format in [DataFormat.Float16, DataFormat.Float16_b]
            else torch.bfloat16
        ),
    )

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
