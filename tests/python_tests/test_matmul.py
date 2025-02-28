# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers import *

def generate_golden(operand1, operand2, data_format, math_fidelity):

    if data_format == "Float16_b":
        if math_fidelity in [0, 2]:  # LoFi or HiFi2
            for element in operand2:
                element = element.to(torch.int32)
                element &= 0xFFFE
        if math_fidelity == 0:  # LoFi
            for element in operand1:
                element = element.to(torch.int32)
                element &= 0xFFF8

    return torch.matmul(tilize(operand1).view(32, 32), tilize(operand2).view(32, 32)).view(-1)

param_combinations = [
    (format, dest_acc, testname, math_fidelity)
    for format in ["Float16_b"]#,"Float16"]
    for dest_acc in ["", "DEST_ACC"]
    for testname in ["matmul_test"]
    for math_fidelity in [3,4]
]

param_ids = [
    f" format={comb[0]} | dest_acc={comb[1]} | math_fidelity={comb[3]}"
    for comb in param_combinations
]

@pytest.mark.parametrize(
    "format, dest_acc, testname, math_fidelity",
    param_combinations,
    ids=param_ids
)

def test_matmul(format, testname, dest_acc, math_fidelity):

    #src_A, src_B = generate_stimuli(format,tile_cnt=1,sfpu=False,const_face=True,const_value_A=3,const_value_B=2)  
    #src_A, src_B = generate_stimuli(format)

    src_A = torch.tensor([torch.rand(1,dtype=format_dict[format]).item()]*256 + [torch.rand(1,dtype=format_dict[format]).item()]*256 + [torch.rand(1,dtype=format_dict[format]).item()]*256 + [torch.rand(1,dtype=format_dict[format]).item()]*256, dtype=torch.bfloat16)
    src_B = torch.tensor([torch.rand(1,dtype=format_dict[format]).item()]*256 + [torch.rand(1,dtype=format_dict[format]).item()]*256 + [torch.rand(1,dtype=format_dict[format]).item()]*256 + [torch.rand(1,dtype=format_dict[format]).item()]*256, dtype=torch.bfloat16)

    print(src_A)
    print(src_B)

    golden_tensor = generate_golden(src_A, src_B, format,math_fidelity)

    write_stimuli_to_l1(tilize(src_A), tilize(src_B), format)

    test_config = {
        "input_format": format,
        "output_format": format,
        "testname": testname,
        "dest_acc": dest_acc,
        "math_fidelity" : math_fidelity
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
