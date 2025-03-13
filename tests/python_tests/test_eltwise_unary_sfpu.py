# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math
from helpers import *


def generate_golden(operation, operand1, data_format):
    tensor1_float = (
        operand1.clone()
        .detach()
        .to(format_dict[data_format] if data_format != "Bfp8_b" else torch.bfloat16)
    )
    ops = {
        "sqrt": lambda x: math.sqrt(x),
        "square": lambda x: x * x,
        "log": lambda x: math.log(x) if x != 0 else float("nan"),
    }
    if operation not in ops:
        raise ValueError("Unsupported operation!")
    return [ops[operation](num) for num in tensor1_float.tolist()][:256]


full_sweep = False
all_format_combos = generate_format_combinations(
    [DataFormat.Float16_b, DataFormat.Float16, DataFormat.Float32], all_same=True
)  # Generate format combinations with all formats being the same (flag set to True), refer to `param_config.py` for more details.
all_params = generate_params(
    ["eltwise_unary_sfpu_test"],
    all_format_combos,
    dest_acc=["", "DEST_ACC"],
    approx_mode=["false", "true"],
    mathop=["sqrt", "log", "square"],
)
param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize(
    "testname, formats, dest_acc, approx_mode, mathop",
    clean_params(all_params),
    ids=param_ids,
)
def test_eltwise_unary_sfpu(testname, formats, dest_acc, approx_mode, mathop):  #
    if (
        formats.unpack_src in [DataFormat.Float32, DataFormat.Int32]
        and dest_acc != "DEST_ACC"
    ):
        pytest.skip(
            reason="Skipping test for 32 bit wide data without 32 bit accumulation in Dest"
        )
    if formats.unpack_src == DataFormat.Float16 and (
        (dest_acc == "" and get_chip_architecture() == "blackhole")
        or (dest_acc == "DEST_ACC" and get_chip_architecture() == "wormhole")
    ):
        pytest.skip(reason="This combination is not fully implemented in testing")

    #  When running hundreds of tests, failing tests may cause incorrect behavior in subsequent passing tests.
    #  To ensure accurate results, for now we reset board after each test.
    #  Fix this: so we only reset after failing tests
    if full_sweep:
        run_shell_command(f"cd .. && make clean")
        run_shell_command(f"tt-smi -r 0")

    src_A, src_B = generate_stimuli(formats.unpack_src, sfpu=True)
    golden = generate_golden(mathop, src_A, formats.pack_dst)
    write_stimuli_to_l1(src_A, src_B, formats.unpack_src)

    test_config = {
        "formats": formats,
        "testname": testname,
        "dest_acc": dest_acc,
        "mathop": mathop,
        "approx_mode": approx_mode,
    }

    make_cmd = generate_make_command(test_config)
    run_shell_command(f"cd .. && {make_cmd}")
    run_elf_files(testname)

    res_from_L1 = collect_results(
        formats, sfpu=True
    )  # Bug patchup in (unpack.py): passing formats struct to check unpack_src with pack_dst and distinguish when input and output formats have different exponent widths then reading from L1 changes

    run_shell_command("cd .. && make clean")

    assert len(res_from_L1) == len(golden)
    assert_tensix_operations_finished()

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

    if formats.pack_dst in [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
    ]:
        atol = 0.05
        rtol = 0.1
    elif formats.pack_dst == DataFormat.Bfp8_b:
        atol = 0.05
        rtol = 0.1

    for i in range(len(golden)):
        assert torch.isclose(
            golden_tensor[i], res_tensor[i], rtol=rtol, atol=atol
        ), f"Failed at index {i} with values {golden[i]} and {res_from_L1[i]}"

    _, pcc = compare_pcc(golden_tensor, res_tensor, pcc=0.99)
    assert pcc > 0.99
