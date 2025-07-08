# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from helpers.device import (
    collect_results,
    write_stimuli_to_l1,
)
from helpers.format_arg_mapping import (
    DestAccumulation,
    MathOperation,
    ReduceDimension,
    ReducePool,
    format_dict,
)
from helpers.format_config import DataFormat
from helpers.golden_generators import ReduceGolden, get_golden_generator
from helpers.param_config import (
    clean_params,
    generate_param_ids,
    generate_params,
    input_output_formats,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.tilize_untilize import untilize
from helpers.utils import passed_test

# Helper dictionary to map reduce dimensions to math operations
mathop_mapping = {
    ReduceDimension.Row: MathOperation.ReduceRow,
    ReduceDimension.Column: MathOperation.ReduceColumn,
    ReduceDimension.Scalar: MathOperation.ReduceScalar,
}

# SUPPORTED FORMATS FOR TEST
supported_formats = [
    DataFormat.Float16_b,
    DataFormat.Float16,
    DataFormat.Float32,
    DataFormat.Bfp8_b,
]

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

formats = input_output_formats(supported_formats)
all_params = generate_params(
    ["reduce_test"],
    formats,
    dest_acc=[DestAccumulation.No],
    reduce_dim=[ReduceDimension.Row, ReduceDimension.Column, ReduceDimension.Scalar],
    pool_type=[ReducePool.Max, ReducePool.Average, ReducePool.Sum],
)

param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize(
    "testname, formats, dest_acc, reduce_dim, pool_type",
    clean_params(all_params),
    ids=param_ids,
)
def test_reduce(testname, formats, dest_acc, reduce_dim, pool_type):

    input_dimensions = [32, 32]

    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format, formats.input_format, input_dimensions=input_dimensions
    )

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

    generate_golden = get_golden_generator(ReduceGolden)
    golden_tensor = generate_golden(src_A, reduce_dim, pool_type, formats.output_format)
    res_address = write_stimuli_to_l1(
        src_A, src_B, formats.input_format, formats.input_format, tile_count=tile_cnt
    )

    mathop = mathop_mapping[reduce_dim]

    test_config = {
        "formats": formats,
        "testname": testname,
        "dest_acc": dest_acc,
        "reduce_dim": reduce_dim,
        "pool_type": pool_type,
        "mathop": mathop,
    }

    run_test(test_config)

    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)
    assert len(res_from_L1) == len(golden_tensor)

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = untilize(res_tensor, formats.output_format)

    # run_shell_command(f"cd .. && make clean") -> TODO: Investigate

    # E           RuntimeError: Build failed: cd .. && make clean
    # E           rm: cannot remove 'build/elf': Directory not empty
    # E           make: *** [Makefile:129: clean] Error 1

    assert passed_test(golden_tensor, res_tensor, formats.output_format)
