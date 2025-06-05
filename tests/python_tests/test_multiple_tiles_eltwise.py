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
from helpers.stimuli_generator import flatten_list, generate_stimuli
from helpers.test_config import generate_make_command
from helpers.utils import format_kernel_list, passed_test, run_shell_command


def generate_golden(op, operand1, operand2, data_format, math_fidelity):
    op_num = list(MathOperation).index(op) + 1
    if op.value == "Elwadd":
        assert op_num == 1
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

    if op_num == 1:
        res = tensor1_float + tensor2_float
    elif op_num == 2:
        res = tensor1_float - tensor2_float
    elif op_num == 3:
        res = tensor1_float * tensor2_float
    else:
        raise ValueError("Unsupported operation!")

    return res.tolist()


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
    ["multiple_tiles_eltwise_test"],
    test_formats,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
)
param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize("tile_cnt", range(1, 4))
@pytest.mark.parametrize(
    "testname, formats, dest_acc, mathop, math_fidelity",
    clean_params(all_params),
    ids=param_ids,
)
def test_multiple_tiles(testname, formats, dest_acc, mathop, math_fidelity, tile_cnt):

    if mathop != MathOperation.Elwmul and math_fidelity != MathFidelity.LoFi:
        pytest.skip("Fidelity does not affect Elwadd and Elwsub operations")

    pack_start_address = 0x1A000 + 2 * 4096 * tile_cnt
    pack_addresses = [pack_start_address + 0x1000 * i for i in range(tile_cnt)]
    pack_addresses_formatted = format_kernel_list(pack_addresses, as_hex=True)

    src_A, src_B = generate_stimuli(
        formats.input_format, formats.input_format, tile_cnt=tile_cnt
    )
    golden = generate_golden(mathop, src_A, src_B, formats.output_format, math_fidelity)
    write_stimuli_to_l1(
        src_A, src_B, formats.input_format, formats.input_format, "0,0", tile_cnt
    )

    test_config = {
        "formats": formats,
        "testname": testname,
        "dest_acc": dest_acc,
        "mathop": mathop,
        "kern_cnt": tile_cnt,
        "pack_addr_cnt": len(pack_addresses),
        "pack_addrs": pack_addresses_formatted,
        "math_fidelity": math_fidelity,
    }

    make_cmd = generate_make_command(test_config)
    run_shell_command(f"cd .. && {make_cmd}")

    run_elf_files(testname)
    wait_for_tensix_operations_finished()

    # check resluts from multiple tiles
    res_from_L1 = []

    for address in pack_addresses:
        res_from_L1.append(
            collect_results(
                formats, tensor_size=len(src_A) // len(pack_addresses), address=address
            )
        )

    res_from_L1 = flatten_list(res_from_L1)

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
