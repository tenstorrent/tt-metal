# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers import *


def generate_golden(operand1, operand2, data_format, math_fidelity):

    if data_format == DataFormat.Float16_b:
        if math_fidelity in [MathFidelity.LoFi, MathFidelity.HiFi2]:  # LoFi or HiFi2
            for element in operand2:
                element = element.to(torch.int32)
                element &= 0xFFFE
        if math_fidelity == MathFidelity.LoFi:  # LoFi
            for element in operand1:
                element = element.to(torch.int32)
                element &= 0xFFF8

    operand1_matrix = operand1.view(32, 32).to(format_dict[data_format])
    operand2_matrix = operand2.view(32, 32).to(format_dict[data_format])

    result_matrix = torch.zeros(32, 32, dtype=operand1_matrix.dtype)
    result_matrix = torch.matmul(operand1_matrix, operand2_matrix)

    return result_matrix.view(1024).to(format_dict[data_format])


all_format_combos = generate_format_combinations(
    [DataFormat.Float16_b], all_same=True
)  # Generate format combinations with all formats being the same (flag set to True), refer to `param_config.py` for more details.
all_params = generate_params(
    ["matmul_test"],
    all_format_combos,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    math_fidelity=[MathFidelity.HiFi3, MathFidelity.HiFi4],
)
param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize(
    "testname, formats, dest_acc, math_fidelity",
    clean_params(all_params),
    ids=param_ids,
)
def test_matmul(testname, formats, dest_acc, math_fidelity):

    src_A, src_B = generate_stimuli()

    golden_tensor = generate_golden(src_A, src_B, formats.pack_dst, math_fidelity)
    golden_tensor = tilize(golden_tensor, format_dict[formats.unpack_A_src])
    golden_tensor = golden_tensor.to(format_dict[formats.pack_dst])

    write_stimuli_to_l1(
        tilize(src_A, format_dict[formats.unpack_A_src]),
        tilize(src_B, format_dict[formats.unpack_B_src]),
        formats.unpack_A_src,
        formats.unpack_B_src,
    )

    test_config = {
        "formats": formats,
        "testname": testname,
        "dest_acc": dest_acc,
        "math_fidelity": math_fidelity,
    }

    make_cmd = generate_make_command(test_config)
    run_shell_command(f"cd .. && {make_cmd}")

    run_elf_files(testname)

    wait_for_tensix_operations_finished()
    res_from_L1 = collect_results(
        formats, tensor_size=len(src_A)
    )  # Bug patchup in (unpack.py): passing formats struct to check unpack_src with pack_dst and distinguish when input and output formats have different exponent widths then reading from L1 changes
    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.pack_dst]
            if formats.pack_dst in [DataFormat.Float16, DataFormat.Float16_b]
            else torch.bfloat16
        ),
    )

    if formats.pack_dst in [DataFormat.Float16_b, DataFormat.Float16]:
        atol = 0.1
        rtol = 0.05
    elif formats.pack_dst == DataFormat.Bfp8_b:
        atol = 0.1
        rtol = 0.2

    for i in range(len(golden_tensor)):
        assert torch.isclose(
            golden_tensor[i], res_tensor[i], rtol=rtol, atol=atol
        ), f"Failed at index {i} with values {golden_tensor[i]} and {res_from_L1[i]}"

    _, pcc = compare_pcc(golden_tensor, res_tensor, pcc=0.99)
    assert pcc > 0.98
