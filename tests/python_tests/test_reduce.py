# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers import *


def generate_golden(operand1, reduce_dim, pool_type, data_format):

    result = torch.zeros(1024, dtype=format_dict[data_format]).view(32, 32)

    f0 = operand1[:256].view(16, 16)
    f1 = operand1[256:512].view(16, 16)
    f2 = operand1[512:768].view(16, 16)
    f3 = operand1[768:].view(16, 16)

    print_faces(operand1)

    def apply_pooling(tensor, pool_type, dim):
        if pool_type == "max":
            return torch.max(tensor, dim=dim).values
        elif pool_type == "avg":
            return torch.mean(tensor, dim=dim)
        elif pool_type == "sum":
            return torch.sum(tensor, dim=dim)
        else:
            pytest.skip("Nonexisting pool type")

    if reduce_dim == "reduce_col":
        left_half = torch.cat((f0, f2), 0)
        right_half = torch.cat((f1, f3), 0)

        left_half_max = apply_pooling(left_half, pool_type, dim=0)
        right_half_max = apply_pooling(right_half, pool_type, dim=0)

        result[0][0:16] = left_half_max.view(1, 16)
        result[0][16:32] = right_half_max.view(1, 16)

    elif reduce_dim == "reduce_row":
        top_half = torch.cat((f0, f1), 1)
        bottom_half = torch.cat((f2, f3), 1)

        top_half_max = apply_pooling(top_half, pool_type, dim=1)
        bottom_half_max = apply_pooling(bottom_half, pool_type, dim=1)

        result[:16, 0] = top_half_max.view(16)
        result[16:32, 0] = bottom_half_max.view(16)
    elif reduce_dim == "reduce_scalar":

        result[0][0] = apply_pooling(operand1.view(1024), pool_type, dim=0)

    else:
        pytest.skip("To be implemented")

    print(result)

    return result.view(1024)


full_sweep = False
all_format_combos = generate_format_combinations(
    [DataFormat.Float16_b, DataFormat.Float16], all_same=True
)  # Generate format combinations with all formats being the same (flag set to True), refer to `param_config.py` for more details.
all_params = generate_params(
    ["reduce_test"],
    all_format_combos,
    dest_acc=[""],
    reduce_dim=["reduce_col"],
    pool_type=["max", "sum", "avg"],
)
param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize(
    "testname, formats, dest_acc, reduce_dim, pool_type",
    clean_params(all_params),
    ids=param_ids,
)
@pytest.mark.skip(reason="Not fully implemented")
def test_reduce(testname, formats, dest_acc, reduce_dim, pool_type):

    #  When running hundreds of tests, failing tests may cause incorrect behavior in subsequent passing tests.
    #  To ensure accurate results, for now we reset board after each test.
    #  Fix this: so we only reset after failing tests
    if full_sweep:
        run_shell_command(f"cd .. && make clean")
        run_shell_command(f"tt-smi -r 0")

    src_A, src_B = generate_stimuli(formats.unpack_src)

    if pool_type in ["max", "sum"]:  # result in srcA should be divided by 1
        src_B = torch.full((1024,), 1)
    else:
        # reduce average divides by length of elements in array we reduce
        if reduce_dim in ["reduce_col", "reduce_row"]:
            src_B = torch.full((1024,), 1 / 32)
        else:
            src_B = torch.full((1024,), torch.sqrt(torch.tensor(1 / 1024)))

    golden_tensor = generate_golden(src_A, reduce_dim, pool_type, formats.pack_dst)
    write_stimuli_to_l1(src_A, src_B, formats.unpack_src)

    test_config = {
        "formats": formats,
        "testname": testname,
        "dest_acc": dest_acc,
        "reduce_dim": reduce_dim,
        "pool_type": pool_type,
        "mathop": reduce_dim,
    }

    make_cmd = generate_make_command(test_config)
    run_shell_command(f"cd .. && {make_cmd}")

    run_elf_files(testname)

    res_from_L1 = collect_results(
        formats
    )  # Bug patchup in (unpack.py): passing formats struct to check unpack_src with pack_dst and distinguish when input and output formats have different exponent widths then reading from L1 changes

    run_shell_command("cd .. && make clean")

    assert len(res_from_L1) == len(golden_tensor)
    assert_tensix_operations_finished()

    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.pack_dst]
            if formats.pack_dst in [DataFormat.Float16, DataFormat.Float16_b]
            else torch.bfloat16
        ),
    )
    res_tensor = untilize(res_tensor, formats.pack_dst)

    print("RES IN L1")
    print(res_tensor.view(32, 32))

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
