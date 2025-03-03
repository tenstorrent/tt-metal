# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math
from helpers import *

def generate_golden(operation, operand1, data_format):
    tensor1_float = operand1.clone().detach().to(format_dict[data_format] if data_format != "Bfp8_b" else torch.bfloat16)
    ops = {
        "sqrt": lambda x: math.sqrt(x),
        "square": lambda x: x * x,
        "log": lambda x: math.log(x) if x != 0 else float('nan')
    }
    if operation not in ops:
        raise ValueError("Unsupported operation!")
    return [ops[operation](num) for num in tensor1_float.tolist()][:256]


param_combinations = [
    (mathop, format, dest_acc, testname, approx_mode)
    for mathop in  ["sqrt","log","square"]
    for format in ["Float16_b", "Float16","Float32"]
    for dest_acc in ["DEST_ACC",""]
    for testname in ["eltwise_unary_sfpu_test"]
    for approx_mode in ["false","true"]
]

param_ids = [
    f"mathop={comb[0]} | format={comb[1]} | dest_acc={comb[2]} | approx_mode={comb[4]}"
    for comb in param_combinations
]

@pytest.mark.parametrize(
    "mathop, format, dest_acc, testname, approx_mode",
    param_combinations,
    ids=param_ids
)

def test_eltwise_unary_sfpu(format, mathop, testname, dest_acc, approx_mode):

    if( format in ["Float32", "Int32"] and dest_acc!="DEST_ACC"):
        pytest.skip(reason = "Skipping test for 32 bit wide data without 32 bit accumulation in Dest")

    if (format == "Float16" and dest_acc == "DEST_ACC"):
        pytest.skip(reason = "This combination is not fully implemented in testing")

    src_A,src_B = generate_stimuli(format,sfpu = True)
    golden = generate_golden(mathop, src_A, format)
    write_stimuli_to_l1(src_A, src_B, format)

    print("\n \n SRCA \n \n")
    print(src_A.tolist())

    test_config = {
        "input_format": format,
        "output_format": format,
        "testname": testname,
        "dest_acc": dest_acc,
        "mathop": mathop,
        "approx_mode": approx_mode
    }

    make_cmd = generate_make_command(test_config)
    run_shell_command(f"cd .. && {make_cmd}")
    run_elf_files(testname)
    
    res_from_L1 = collect_results(format,sfpu=True)

    run_shell_command("cd .. && make clean")

    assert len(res_from_L1) == len(golden)
    assert_tensix_operations_finished()

    golden_tensor = torch.tensor(golden, dtype=format_dict[format] if format in ["Float16", "Float16_b"] else torch.bfloat16)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[format] if format in ["Float16", "Float16_b"] else torch.bfloat16)

    if(format in ["Float16_b","Float16", "Float32"]):
        atol = 0.05
        rtol = 0.1
    elif format == "Bfp8_b":
        atol = 0.05
        rtol = 0.1

    for i in range(len(golden)):
        assert torch.isclose(golden_tensor[i],res_tensor[i], rtol = rtol, atol = atol), f"Failed at index {i} with values {golden[i]} and {res_from_L1[i]}"

    _ , pcc = compare_pcc(golden_tensor, res_tensor, pcc=0.99) 
    assert pcc > 0.99
