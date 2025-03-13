# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers import *


def generate_golden(operation, operand1, operand2, data_format):
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

    operations = {
        "elwadd": tensor1_float + tensor2_float,
        "elwsub": tensor1_float - tensor2_float,
        "elwmul": tensor1_float * tensor2_float,
    }

    if operation not in operations:
        raise ValueError("Unsupported operation!")

    return operations[operation].tolist()


full_sweep = False
all_format_combos = generate_format_combinations(
    [DataFormat.Float32], all_same=True
)  # Generate format combinations with all formats being the same (flag set to True), refer to `param_config.py` for more details.
all_params = generate_params(
    ["sfpu_binary_test"],
    all_format_combos,
    dest_acc=["DEST_ACC"],
    mathop=["elwadd", "elwsub", "elwmul"],
)
param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize(
    "testname, formats, dest_acc, mathop", clean_params(all_params), ids=param_ids
)
@pytest.mark.skip(reason="Not fully implemented")
def test_all(testname, formats, dest_acc, mathop):
    if (
        formats.unpack_src in [DataFormat.Float32, DataFormat.Int32]
        and dest_acc != "DEST_ACC"
    ):
        pytest.skip(
            "Skipping test for 32 bit wide data without 32 bit accumulation in Dest"
        )

    #  When running hundreds of tests, failing tests may cause incorrect behavior in subsequent passing tests.
    #  To ensure accurate results, for now we reset board after each test.
    #  Fix this: so we only reset after failing tests
    if full_sweep:
        run_shell_command(f"cd .. && make clean")
        run_shell_command(f"tt-smi -r 0")

    src_A, src_B = generate_stimuli(formats.unpack_src)
    golden = generate_golden(mathop, src_A, src_B, formats.pack_dst)
    write_stimuli_to_l1(src_A, src_B, formats.unpack_src)

    test_config = {
        "formats": formats,
        "testname": testname,
        "dest_acc": dest_acc,
        "mathop": mathop,
    }

    make_cmd = generate_make_command(test_config)
    run_shell_command(f"cd .. && {make_cmd}")

    run_elf_files(testname)

    res_from_L1 = collect_results(
        formats
    )  # Bug patchup in (unpack.py): passing formats struct to check its unpack_src with pack_dst and distinguish when input and output formats have different exponent widths then reading from L1 changes

    assert len(res_from_L1) == len(golden)

    run_shell_command("cd .. && make clean")

    assert_tensix_operations_finished()

    if formats.pack_dst in [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
    ]:
        atol = 0.05
        rtol = 0.1
    elif formats.pack_dst == DataFormat.Bfp8_b:
        atol = 0.1
        rtol = 0.2

    golden_tensor = torch.tensor(
        golden,
        dtype=(
            format_dict[formats.pack_dst]
            if formats.pack_dst
            in [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )
    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.pack_dst]
            if formats.pack_dst
            in [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )

    for i in range(len(golden)):
        assert torch.isclose(
            golden_tensor[i], res_tensor[i], rtol=rtol, atol=atol
        ), f"Failed at index {i} with values {golden[i]} and {res_from_L1[i]}"

    _, pcc = comp_pcc(golden_tensor, res_tensor, pcc=0.99)
    assert pcc > 0.99
