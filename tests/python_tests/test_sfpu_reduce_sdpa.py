# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.device import collect_results, write_stimuli_to_l1
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DestAccumulation,
    MathOperation,
    ReducePool,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test


@parametrize(
    test_name="sfpu_reduce_sdpa_test",
    formats=input_output_formats(
        [DataFormat.Float16_b],  # Only Float16_b is supported for SDPA reduce
        same=True,
    ),
    dest_acc=[DestAccumulation.No],
    mathop=[MathOperation.ReduceColumn],
    reduce_pool=[ReducePool.Max],  # Only MAX is supported for SDPA reduce
    input_dimensions=[
        [128, 64],  # 4x2 subblock
    ],
)
def test_sfpu_reduce_sdpa(
    test_name, formats, dest_acc, mathop, reduce_pool, input_dimensions
):

    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format, formats.input_format, input_dimensions=input_dimensions
    )

    src_A = tilize_block(src_A, input_dimensions).flatten()

    # Generate dummy src_B
    src_B = torch.zeros_like(src_A)

    # GOLDEN GENERATION
    # *******************************************************

    # Undo tilization so src_A is standard [32, 32]
    src_A_untilized = untilize_block(src_A, formats.input_format, input_dimensions)

    # Take max along the height (dim=0) for each column
    col_max = torch.max(src_A_untilized, dim=0).values

    # Construct golden tensor: first row is column max, others are zero
    golden_tensor = torch.zeros_like(src_A_untilized)
    golden_tensor[0, :] = col_max

    # *******************************************************

    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "input_A_dimensions": input_dimensions,
        "input_B_dimensions": input_dimensions,
        "mathop": mathop,
        "pool_type": reduce_pool,
        "unpack_to_dest": False,  # Must be False since math kernel does A2D copy
        "tile_cnt": tile_cnt,  # Keep tile_cnt for future multi-tile support
    }

    res_address = write_stimuli_to_l1(
        test_config,
        src_A,
        src_B,
        formats.input_format,
        formats.input_format,
        tile_count_A=tile_cnt,
        tile_count_B=tile_cnt,
    )

    run_test(test_config)

    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = untilize_block(res_tensor, formats.output_format, input_dimensions)

    # Check only the first row for correctness, not full tensors
    assert passed_test(golden_tensor[0], res_tensor[0], formats.output_format)
