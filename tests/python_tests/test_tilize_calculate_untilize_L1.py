# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers import *


def generate_golden(op, operand1, operand2, data_format, math_fidelity):
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

    # First step is unpack tilize
    tensor1_float = tilize(tensor1_float, data_format)
    tensor2_float = tilize(tensor2_float, data_format)

    # Second step is to perform the operation
    if op == MathOperation.Elwadd:
        res = tensor1_float + tensor2_float
    elif op == MathOperation.Elwsub:
        res = tensor1_float - tensor2_float
    elif op == MathOperation.Elwmul:
        res = tensor1_float * tensor2_float
    else:
        raise ValueError("Unsupported operation!")

    # res = untilize(res, data_format)

    return res


full_sweep = False
all_format_combos = generate_format_combinations(
    [DataFormat.Float16_b, DataFormat.Float16], all_same=True
)  # Generate format combinations with all formats being the same (flag set to True), refer to `param_config.py` for more details.
all_params = generate_params(
    ["tilize_calculate_untilize_L1"],
    all_format_combos,
    dest_acc=[DestAccumulation.No],
    mathop=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    math_fidelity=[MathFidelity.HiFi4],
    tile_cnt=[TileCount.One],
)
param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize(
    "testname, formats, dest_acc, mathop, math_fidelity, tile_cnt",
    clean_params(all_params),
    ids=param_ids,
)
def test_tilize_calculate_untilize_L1(
    testname, formats, dest_acc, mathop, math_fidelity, tile_cnt
):

    src_A, src_B = generate_stimuli(
        formats.unpack_A_src, formats.unpack_B_src, tile_cnt
    )

    golden_tensor = generate_golden(
        mathop, src_A, src_B, formats.pack_dst, math_fidelity
    )

    write_stimuli_to_l1(
        src_A, src_B, formats.unpack_A_src, formats.unpack_B_src, "0,0", tile_cnt
    )

    buffer_dest_address = 0x1E000  # Since this test calls LLK pipeline twise, unpacker will read at address in L1 that packer packed to, this address is able to be reaused for two LLK calls
    test_config = {
        "formats": formats,
        "testname": testname,
        "dest_acc": dest_acc,
        "math_fidelity": math_fidelity,
        "mathop": mathop,
    }

    make_cmd = generate_make_command(test_config)
    run_shell_command(f"cd .. && {make_cmd}")

    run_elf_files(testname)
    wait_for_tensix_operations_finished()

    res_from_L1 = collect_results(
        formats, tensor_size=len(src_A), address=buffer_dest_address
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
