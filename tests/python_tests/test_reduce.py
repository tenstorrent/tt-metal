# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from helpers.device import (
    collect_results,
    run_elf_files,
    wait_for_tensix_operations_finished,
    write_stimuli_to_l1,
)
from helpers.format_arg_mapping import (
    DestAccumulation,
    ReduceDimension,
    ReducePool,
    format_dict,
)
from helpers.format_config import DataFormat
from helpers.param_config import (
    clean_params,
    generate_param_ids,
    generate_params,
    input_output_formats,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import generate_make_command
from helpers.tilize_untilize import untilize
from helpers.utils import compare_pcc, print_faces, run_shell_command


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

    if reduce_dim == ReduceDimension.Column:
        left_half = torch.cat((f0, f2), 0)
        right_half = torch.cat((f1, f3), 0)

        left_half_max = apply_pooling(left_half, pool_type, dim=0)
        right_half_max = apply_pooling(right_half, pool_type, dim=0)

        result[0][0:16] = left_half_max.view(1, 16)
        result[0][16:32] = right_half_max.view(1, 16)

    elif reduce_dim == ReduceDimension.Row:
        top_half = torch.cat((f0, f1), 1)
        bottom_half = torch.cat((f2, f3), 1)

        top_half_max = apply_pooling(top_half, pool_type, dim=1)
        bottom_half_max = apply_pooling(bottom_half, pool_type, dim=1)

        result[:16, 0] = top_half_max.view(16)
        result[16:32, 0] = bottom_half_max.view(16)
    elif reduce_dim == ReduceDimension.Scalar:

        result[0][0] = apply_pooling(operand1.view(1024), pool_type, dim=0)

    else:
        pytest.skip("To be implemented")

    return result.view(1024)


# SUPPORTED FORMATS FOR TEST
supported_formats = [DataFormat.Bfp8_b, DataFormat.Float16, DataFormat.Float16_b]

#   INPUT-OUTPUT FORMAT SWEEP
#   input_output_formats(supported_formats)

#   FULL FORMAT SWEEP
#   format_combination_sweep(formats=supported_formats, all_same=False, same_src_reg_format=True)

#   SPECIFIC FORMAT COMBINATION
#   generate_combination(
#       [(DataFormat.Float16_b,  # index 0 is for unpack_A_src
#         DataFormat.Float16_b,  # index 1 is for unpack_A_dst
#         DataFormat.Float16_b,  # index 2 is for pack_src (if src registers have same formats)
#         DataFormat.Bfp8_b,  # index 3 is for pack_dst
#         DataFormat.Float16_b,  # index 4 is for math format)])

#   SPECIFIC INPUT-OUTPUT COMBINATION
#   [InputOutputFormat(DataFormat.Float16, DataFormat.Float32)]

test_formats = input_output_formats(supported_formats)
all_params = generate_params(
    ["reduce_test"],
    test_formats,
    dest_acc=[DestAccumulation.No],
    reduce_dim=[ReduceDimension.Column],
    pool_type=[ReducePool.Max, ReducePool.Sum, ReducePool.Average],
)

param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize(
    "testname, formats, dest_acc, reduce_dim, pool_type",
    clean_params(all_params),
    ids=param_ids,
)
@pytest.mark.skip(reason="Not fully implemented")
def test_reduce(testname, formats, dest_acc, reduce_dim, pool_type):

    src_A, src_B = generate_stimuli(formats.input_format, formats.input_format)

    if pool_type in [
        ReducePool.Max,
        ReducePool.Sum,
    ]:  # result in srcA should be divided by 1
        src_B = torch.full((1024,), 1)
    else:
        # reduce average divides by length of elements in array we reduce
        if reduce_dim in [ReduceDimension.Column, ReduceDimension.Row]:
            src_B = torch.full((1024,), 1 / 32)
        else:
            src_B = torch.full((1024,), torch.sqrt(torch.tensor(1 / 1024)))

    golden_tensor = generate_golden(src_A, reduce_dim, pool_type, formats.output_format)
    write_stimuli_to_l1(src_A, src_B, formats.input_format, formats.input_format)

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
    wait_for_tensix_operations_finished()

    res_from_L1 = collect_results(
        formats, tensor_size=len(src_A)
    )  # Bug patchup in (unpack.py): passing formats struct to check unpack_src with pack_dst and distinguish when input and output formats have different exponent widths then reading from L1 changes
    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format in [DataFormat.Float16, DataFormat.Float16_b]
            else torch.bfloat16
        ),
    )
    res_tensor = untilize(res_tensor, formats.output_format)

    print(res_tensor.view(32, 32))

    if formats.output_format in [DataFormat.Float16_b, DataFormat.Float16]:
        atol = 0.1
        rtol = 0.05
    elif formats.output_format == DataFormat.Bfp8_b:
        atol = 0.1
        rtol = 0.2

    for i in range(len(golden_tensor)):
        assert torch.isclose(
            golden_tensor[i], res_tensor[i], rtol=rtol, atol=atol
        ), f"Failed at index {i} with values {golden_tensor[i]} and {res_from_L1[i]}"

    _, pcc = compare_pcc(golden_tensor, res_tensor, pcc=0.99)
    assert pcc > 0.98
