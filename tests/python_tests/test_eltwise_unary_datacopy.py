# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers import *


def generate_golden(operand1, format):
    return operand1


full_sweep = False
#  This is an example of how users can define and create their own format combinations for testing specific cases they're interested in
# Note these combinations might fail because we don't have date inference model to adjust unsupported format combinations
generate_format_selection = create_formats_for_testing(
    [
        (
            DataFormat.Float16_b,  # index 0 is for unpack_A_src
            DataFormat.Float16_b,  # index 1 is for unpack_A_dst
            DataFormat.Float16_b,  # index 2 is for pack_src (if src registers have same formats)
            DataFormat.Bfp8_b,  # index 3 is for pack_dst
            DataFormat.Float16_b,  # index 4 is for math format
        ),
    ]
)

all_format_combos = generate_format_combinations(
    formats=[
        DataFormat.Float32,
        DataFormat.Bfp8_b,
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Int32,
    ],
    all_same=True,
    same_src_reg_format=True,  # setting src_A and src_B register to have same format
)  # Generate format combinations with all formats being the same (flag set to True), refer to `param_config.py` for more details.
dest_acc = [DestAccumulation.Yes, DestAccumulation.No]
testname = ["eltwise_unary_datacopy_test"]
all_params = generate_params(testname, all_format_combos, dest_acc)
param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize(
    "testname, formats, dest_acc", clean_params(all_params), ids=param_ids
)
def test_unary_datacopy(testname, formats, dest_acc):
    if formats.input_format == DataFormat.Int32:
        pytest.skip("Bfp8_b to Float16 is not supported")
    if formats.input_format == DataFormat.Float16 and dest_acc == DestAccumulation.Yes:
        pytest.skip(reason="This combination is not fully implemented in testing")

    src_A, src_B = generate_stimuli(formats.input_format, formats.input_format)
    srcB = torch.full((1024,), 0)
    golden = generate_golden(src_A, formats.output_format)
    write_stimuli_to_l1(src_A, src_B, formats.input_format, formats.input_format)

    test_config = {
        "formats": formats,
        "testname": testname,
        "dest_acc": dest_acc,
    }

    make_cmd = generate_make_command(test_config)
    run_shell_command(f"cd .. && {make_cmd}")
    run_elf_files(testname)

    wait_for_tensix_operations_finished()
    res_from_L1 = collect_results(formats, tensor_size=len(src_A))
    assert len(res_from_L1) == len(golden)

    if formats.output_format in format_dict:
        atol = 0.05
        rtol = 0.1
    else:
        atol = 0.2
        rtol = 0.1

    golden_tensor = torch.tensor(
        golden,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format
            in [
                DataFormat.Float16,
                DataFormat.Float16_b,
                DataFormat.Float32,
                DataFormat.Int32,
            ]
            else torch.bfloat16
        ),
    )
    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format
            in [
                DataFormat.Float16,
                DataFormat.Float16_b,
                DataFormat.Float32,
                DataFormat.Int32,
            ]
            else torch.bfloat16
        ),
    )

    for i in range(len(golden)):
        assert torch.isclose(
            golden_tensor[i], res_tensor[i], rtol=rtol, atol=atol
        ), f"Failed at index {i} with values {golden[i]} and {res_from_L1[i]}"

    _, pcc = compare_pcc(golden_tensor, res_tensor, pcc=0.99)
    assert pcc > 0.99
