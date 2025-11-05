# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from helpers.device import (
    collect_results,
    write_stimuli_to_l1,
)
from helpers.format_config import DataFormat
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    ReducePool,
    format_dict,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test
from helpers.tilize_untilize import untilize
from helpers.utils import passed_test

max_tiles = 4  # max number of tiles in 32-bit dest is 4
tile_dim = 32

dimension_combinations = [
    [m, n]
    for m in range(tile_dim, max_tiles * tile_dim + 1, tile_dim)
    for n in range(tile_dim, max_tiles * tile_dim + 1, tile_dim)
    if m * n <= max_tiles * tile_dim * tile_dim
]


@parametrize(
    test_name="sfpu_reduce_test",
    formats=input_output_formats(
        [DataFormat.Float32, DataFormat.UInt32, DataFormat.Int32],
        same=True,
    ),
    mathop=[MathOperation.ReduceColumn],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    negative_number=[False, True],
    reduce_pool=[ReducePool.Sum, ReducePool.Average],
    dimension_combinations=dimension_combinations,
)
def test_sfpu_reduce(
    test_name,
    formats,
    dest_acc,
    mathop,
    reduce_pool,
    negative_number,
    dimension_combinations,
):
    if negative_number and formats.input_format == DataFormat.UInt32:
        pytest.skip(
            f"Skipping negative_numbers=True for unsigned format {formats.input_format}"
        )

    input_dimensions = dimension_combinations
    torch_format = format_dict[formats.input_format]

    src_A, src_B, tile_cnt = generate_stimuli(
        formats.input_format, formats.input_format, input_dimensions=input_dimensions
    )
    src_A = torch.ones(tile_cnt * 1024, dtype=torch_format)

    # Generate 4 faces with all 2s for easy verification (column sums = 32*2 = 64, which is a multiple of 32)
    sign = -1 if negative_number else 1
    src_A *= sign

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        mathop,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        reduce_pool,
    )

    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "input_A_dimensions": input_dimensions,
        "input_B_dimensions": input_dimensions,
        "mathop": mathop,
        "pool_type": reduce_pool,
        "approx_mode": ApproximationMode.No,
        "unpack_to_dest": True,
        "tile_cnt": tile_cnt,
    }

    res_address = write_stimuli_to_l1(
        test_config,
        src_A,
        src_B,
        formats.input_format,
        formats.input_format,
        tile_count_A=tile_cnt,
        tile_count_B=1,
    )
    run_test(test_config)

    torch_format = format_dict[formats.output_format]
    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # collect all only the first row from each tile in result tensor (f0 and f1 row, 32 datums)
    # SFPU reduce operation stores the result in the top row of each tile:
    # - 16 datums in face 0 (first 16 elements of top row)
    # - 16 datums in face 1 (last 16 elements of top row)
    # However, we pack out the full tile (1024 elements), so we need to extract
    # only the top row. Since the result is in tilized format, we must untilize
    # to get row-major ordering, then extract the first 32 elements which
    # correspond to the first row of face 0 and face 1.
    # We do so for each tile we reduced
    reduce_result = []
    golden_result = []
    for i in range(tile_cnt):
        # Calculate starting indices for this tile
        start_res = i * 1024  # Each tile has 1024 elements in result tensor
        start_golden = i * 32  # Each tile contributes 32 elements to golden

        # Extract and untilize the current tile, then get first 32 elements (top row)
        result_tile_i = untilize(
            res_tensor[start_res : start_res + 1024], formats.output_format
        ).flatten()[:32]

        # Accumulate results from all tiles
        reduce_result.extend(result_tile_i)
        golden_result.extend(golden_tensor[start_golden : start_golden + 32])

    # Convert to tensors and verify results match expected values
    reduce_tensor = torch.tensor(reduce_result, dtype=torch_format)
    golden_tensor = torch.tensor(golden_result, dtype=torch_format)
    assert passed_test(golden_tensor, reduce_tensor, formats.output_format)
