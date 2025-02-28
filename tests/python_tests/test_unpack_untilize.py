# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers import *

torch.set_printoptions(linewidth=500,sci_mode = False, precision=2,threshold=10000)

def generate_golden(operand1, data_format):
    
    A_untilized  = untilize(operand1,data_format)
    return A_untilized.flatten()

param_combinations = [
    (format, testname)
    for format in ["Float16_b", "Float16"]
    for testname in ["unpack_untilize_test"]
]

param_ids = [
    f" format={comb[0]} "
    for comb in param_combinations
]

@pytest.mark.parametrize(
    "format,testname",
    param_combinations,
    ids=param_ids
)

def test_unpack_untilze(format, testname):

    src_A, src_B = generate_stimuli(format)
    src_B = torch.full((1024,),0)
    
    golden_tensor = generate_golden(src_A, format)

    write_stimuli_to_l1(src_A, src_B, format)

    test_config = {
        "input_format": format,
        "output_format": format,
        "testname": testname,
    }

    make_cmd = generate_make_command(test_config)
    run_shell_command(f"cd .. && {make_cmd}")

    run_elf_files(testname)

    res_from_L1 = collect_results(format)

    run_shell_command("cd .. && make clean")

    assert len(res_from_L1) == len(golden_tensor)
    assert_tensix_operations_finished()

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[format] if format in ["Float16", "Float16_b"] else torch.bfloat16)

    if(format == "Float16_b" or format == "Float16"):
        atol = 0.1
        rtol = 0.05
    elif(format == "Bfp8_b"):
        atol = 0.1
        rtol = 0.2

    for i in range(len(golden_tensor)):
        assert torch.isclose(golden_tensor[i],res_tensor[i], rtol = rtol, atol = atol), f"Failed at index {i} with values {golden_tensor[i]} and {res_from_L1[i]}"

    _ , pcc = compare_pcc(golden_tensor, res_tensor, pcc=0.99) 
    assert pcc > 0.98
