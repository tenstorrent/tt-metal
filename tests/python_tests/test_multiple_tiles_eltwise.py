# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers import *

mathop_map = {1: "elwadd", 2: "elwsub", 3: "elwmul"}


def generate_golden(op, operand1, operand2, data_format, math_fidelity):
    tensor1_float = (
        operand1.clone()
        .detach()
        .to(format_dict.get(data_format, format_dict["Float16_b"]))
    )
    tensor2_float = (
        operand2.clone()
        .detach()
        .to(format_dict.get(data_format, format_dict["Float16_b"]))
    )

    if data_format == "Float16_b":
        if math_fidelity in [0, 2]:  # LoFi or HiFi2
            for element in operand2:
                element = element.to(torch.int32)
                element &= 0xFFFE
        if math_fidelity == 0:  # LoFi
            for element in operand1:
                element = element.to(torch.int32)
                element &= 0xFFF8

    if op == 1:
        res = tensor1_float + tensor2_float
    elif op == 2:
        res = tensor1_float - tensor2_float
    elif op == 3:
        res = tensor1_float * tensor2_float
    else:
        raise ValueError("Unsupported operation!")

    return res.tolist()


param_combinations = [
    (mathop, tile_cnt, format, dest_acc, testname, math_fidelity)
    for mathop in range(1, 4)
    for tile_cnt in range(1, 4)
    for format in ["Float16_b", "Float16", "Bfp8_b"]
    for dest_acc in ["", "DEST_ACC"]
    for testname in ["multiple_tiles_eltwise_test"]
    for math_fidelity in [0, 2, 3, 4]
]

param_ids = [
    f"mathop={mathop_map[comb[0]]} | tile_cnt={comb[1]} | format={comb[2]} | dest_acc={comb[3]} | math_fidelity={comb[5]}"
    for comb in param_combinations
]


@pytest.mark.parametrize(
    "mathop, tile_cnt, format, dest_acc, testname, math_fidelity",
    param_combinations,
    ids=param_ids,
)
def test_multiple_tiles(format, testname, tile_cnt, mathop, dest_acc, math_fidelity):

    if mathop in range(1, 4) and format == "Float16" and dest_acc == "DEST_ACC":
        pytest.skip(reason="This combination is not fully implemented in testing")

    # prepare setup for running kernels

    pack_start_address = 0x1A000 + 2 * 4096 * tile_cnt
    pack_addresses = [pack_start_address + 0x1000 * i for i in range(tile_cnt)]
    pack_addresses_formatted = format_kernel_list(pack_addresses, as_hex=True)

    src_A, src_B = generate_stimuli(
        format, tile_cnt=tile_cnt
    )  # , const_face=True, const_value_A=3, const_value_B=2)
    golden = generate_golden(mathop, src_A, src_B, format, math_fidelity)
    write_stimuli_to_l1(src_A, src_B, format, "0,0", tile_cnt)

    if mathop != 3:
        math_fidelity = 0

    test_config = {
        "input_format": format,
        "output_format": format,
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

    run_shell_command("cd .. && make clean")

    assert_tensix_operations_finished()

    # check resluts from multiple tiles
    res_from_L1 = []

    for address in pack_addresses:
        res_from_L1.append(collect_results(format, address))

    res_from_L1 = flatten_list(res_from_L1)

    golden_tensor = torch.tensor(
        golden,
        dtype=(
            format_dict[format]
            if format in ["Float16", "Float16_b"]
            else torch.bfloat16
        ),
    )
    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[format]
            if format in ["Float16", "Float16_b"]
            else torch.bfloat16
        ),
    )

    if format == "Float16_b" or format == "Float16":
        atol = 0.05
        rtol = 0.1
    elif format == "Bfp8_b":
        atol = 0.1
        rtol = 0.2

    for i in range(len(golden_tensor)):
        assert torch.isclose(
            golden_tensor[i], res_tensor[i], rtol=rtol, atol=atol
        ), f"Failed at index {i} with values {golden_tensor[i]} and {res_from_L1[i]}"

    _, pcc = compare_pcc(golden_tensor, res_tensor, pcc=0.99)
    assert pcc > 0.99
