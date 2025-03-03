# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers import *

def generate_golden(operations, operand1, operand2, data_format):
    tensor1_float = operand1.clone().detach().to(format_dict.get(data_format, format_dict["Float16_b"]))
    tensor2_float = operand2.clone().detach().to(format_dict.get(data_format, format_dict["Float16_b"]))

    res = []

    # to se why this encoding look at llk_defs.h -> enum EltwiseBinaryType

    for op in operations:
        if(op==0):
            res_tmp = tensor1_float * tensor2_float
        elif(op==1):
            res_tmp = tensor1_float / tensor2_float
        elif(op==2):
            res_tmp = tensor1_float + tensor2_float
        elif(op==3):
            res_tmp = tensor1_float - tensor2_float
        else:
            raise ValueError("Unsupported operation!")
        
        res.append(res_tmp.tolist())
    
    return flatten_list(res)

param_combinations = [
    (format, dest_acc, testname)
    for format in ["Float16_b", "Float16", "Bfp8_b"]
    for dest_acc in ["", "DEST_ACC"]
    for testname in ["fill_dest_test"]
]

param_ids = [
    f" format={comb[0]} | dest_acc={comb[1]}"
    for comb in param_combinations
]

@pytest.mark.parametrize(
    "format, dest_acc, testname",
    param_combinations,
    ids=param_ids
)

def test_fill_dest(format, testname, dest_acc):

    if (format == "Float16" and dest_acc == "DEST_ACC"):
        pytest.skip(reason = "This combination is not fully implemented in testing")

    pack_start_address = 0x1c000
    pack_addresses = [pack_start_address + 0x1000 * i for i in range(16)]

    src_A, src_B = generate_stimuli(format)
    golden = generate_golden([2]*16,src_A,src_B,format)
    write_stimuli_to_l1(src_A,src_B,format)

    test_config = {
        "input_format": format,
        "output_format": format,
        "testname": testname,
        "dest_acc": dest_acc,
    }

    make_cmd = generate_make_command(test_config)
    run_shell_command(f"cd .. && {make_cmd}")

    run_elf_files(testname)

    run_shell_command("cd .. && make clean")

    res_from_L1 = []

    for address in pack_addresses:
        res_from_L1.append(collect_results(format,address))
     
    res_from_L1 = flatten_list(res_from_L1)

    assert len(res_from_L1) == len(golden)
    assert_tensix_operations_finished()

    golden_tensor = torch.tensor(golden, dtype=format_dict[format] if format in ["Float16", "Float16_b"] else torch.bfloat16)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[format] if format in ["Float16", "Float16_b"] else torch.bfloat16)

    if(format == "Float16_b" or format == "Float16"):
        atol = 0.05
        rtol = 0.1
    elif(format == "Bfp8_b"):
        atol = 0.1
        rtol = 0.2

    for i in range(len(golden)):
        assert torch.isclose(golden_tensor[i],res_tensor[i], rtol = rtol, atol = atol), f"Failed at index {i} with values {golden[i]} and {res_from_L1[i]}"

  
    _ , pcc = compare_pcc(golden_tensor, res_tensor, pcc=0.99) 
    assert pcc > 0.99
