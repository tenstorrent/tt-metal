# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers import *


def generate_golden(operations, operand1, operand2, data_format):
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

    res = []

    # to se why this encoding look at llk_defs.h -> enum EltwiseBinaryType

    for op in operations:
        if op == 0:
            res_tmp = tensor1_float * tensor2_float
        elif op == 1:
            res_tmp = tensor1_float / tensor2_float
        elif op == 2:
            res_tmp = tensor1_float + tensor2_float
        elif op == 3:
            res_tmp = tensor1_float - tensor2_float
        else:
            raise ValueError("Unsupported operation!")

        res.append(res_tmp.tolist())

    return flatten_list(res)


full_sweep = False
all_format_combos = generate_format_combinations(
    [DataFormat.Float16_b, DataFormat.Float16, DataFormat.Bfp8_b], all_same=True
)  # Generate format combinations with all formats being the same (flag set to True), refer to `param_config.py` for more details.
all_params = generate_params(
    ["fill_dest_test"],
    all_format_combos,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize(
    "testname, formats, dest_acc", clean_params(all_params), ids=param_ids
)
def test_fill_dest(testname, formats, dest_acc):

    if formats.unpack_A_src == DataFormat.Float16 and dest_acc == DestAccumulation.Yes:
        pytest.skip(reason="This combination is not fully implemented in testing")

    pack_start_address = 0x1C000
    pack_addresses = [pack_start_address + 0x1000 * i for i in range(16)]

    src_A, src_B = generate_stimuli(formats.unpack_A_src, formats.unpack_B_src)
    golden = generate_golden([2] * 16, src_A, src_B, formats.pack_dst)
    write_stimuli_to_l1(src_A, src_B, formats.unpack_A_src, formats.unpack_B_src)

    test_config = {
        "formats": formats,
        "testname": testname,
        "dest_acc": dest_acc,
    }

    make_cmd = generate_make_command(test_config)
    run_shell_command(f"cd .. && {make_cmd}")

    run_elf_files(testname)

    wait_for_tensix_operations_finished()
    res_from_L1 = []
    for address in pack_addresses:
        res_from_L1.append(
            collect_results(formats, tensor_size=len(src_A), address=address)
        )  # Bug patchup in (unpack.py): passing formats struct to check unpack_src with pack_dst and distinguish when input and output formats have different exponent widths then reading from L1 changes
    res_from_L1 = flatten_list(res_from_L1)

    assert len(res_from_L1) == len(golden)

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

    if (
        formats.pack_dst == DataFormat.Float16_b
        or formats.pack_dst == DataFormat.Float16
    ):
        atol = 0.05
        rtol = 0.1
    elif formats.pack_dst == DataFormat.Bfp8_b:
        atol = 0.1
        rtol = 0.2

    for i in range(len(golden)):
        assert torch.isclose(
            golden_tensor[i], res_tensor[i], rtol=rtol, atol=atol
        ), f"Failed at index {i} with values {golden[i]} and {res_from_L1[i]}"

    _, pcc = compare_pcc(golden_tensor, res_tensor, pcc=0.99)
    assert pcc > 0.99
