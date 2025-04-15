# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers import *


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


full_sweep = False
all_format_combos = generate_format_combinations(
    [DataFormat.Float16_b, DataFormat.Float16],
    all_same=True,  # , DataFormat.Bfp8_b], all_same=True
)  # Generate format combinations with all formats being the same (flag set to True), refer to `param_config.py` for more details.
all_params = generate_params(
    ["multiple_tiles_eltwise_test"],
    all_format_combos,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    math_fidelity=[
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ],
    tile_cnt=[TileCount.One, TileCount.Two, TileCount.Three],
)
param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize(
    "testname, formats, dest_acc, mathop, math_fidelity, tile_cnt",
    clean_params(all_params),
    ids=param_ids,
)
def test_multiple_tiles(testname, formats, dest_acc, mathop, math_fidelity, tile_cnt):
    if (
        mathop in [MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul]
        and formats.unpack_A_src == DataFormat.Float16
        and dest_acc == DestAccumulation.Yes
    ):
        pytest.skip(reason="This combination is not fully implemented in testing")

    pack_start_address = 0x1A000 + 2 * 4096 * tile_cnt.value
    pack_addresses = [pack_start_address + 0x1000 * i for i in range(tile_cnt.value)]
    pack_addresses_formatted = format_kernel_list(pack_addresses, as_hex=True)

    src_A, src_B = generate_stimuli(
        formats.unpack_A_src, formats.unpack_B_src, tile_cnt=tile_cnt
    )  # , const_face=True, const_value_A=3, const_value_B=2)
    golden = generate_golden(mathop, src_A, src_B, formats.pack_dst, math_fidelity)
    write_stimuli_to_l1(
        src_A, src_B, formats.unpack_A_src, formats.unpack_B_src, "0,0", tile_cnt
    )

    if mathop != MathOperation.Elwmul:
        math_fidelity = MathFidelity.LoFi

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
            format_dict[formats.pack_dst]
            if formats.pack_dst in [DataFormat.Float16, DataFormat.Float16_b]
            else torch.bfloat16
        ),
    )
    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.pack_dst]
            if formats.pack_dst in [DataFormat.Float16, DataFormat.Float16_b]
            else torch.bfloat16
        ),
    )

    if formats.pack_dst in [DataFormat.Float16_b, DataFormat.Float16]:
        atol = 0.05
        rtol = 0.1
    elif formats.pack_dst == DataFormat.Bfp8_b:
        atol = 0.1
        rtol = 0.2

    for i in range(len(golden_tensor)):
        assert torch.isclose(
            golden_tensor[i], res_tensor[i], rtol=rtol, atol=atol
        ), f"Failed at index {i} with values {golden_tensor[i]} and {res_from_L1[i]}"

    _, pcc = compare_pcc(golden_tensor, res_tensor, pcc=0.99)
    assert pcc > 0.99
