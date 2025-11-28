# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from helpers.device import (
    collect_results,
    write_stimuli_to_l1,
)
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    UnarySFPUGolden,
    get_golden_generator,
)
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
from helpers.test_config import run_test
from helpers.tilize_untilize import tilize_block, untilize_block
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
        [
            DataFormat.Float32,
            DataFormat.Int32,
            DataFormat.UInt32,
            DataFormat.UInt16,
            DataFormat.Float16_b,
        ],
        same=True,
    ),
    mathop=[MathOperation.ReduceColumn],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    negative_number=[False, True],
    reduce_pool=[ReducePool.Max, ReducePool.Average, ReducePool.Sum],
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
    if negative_number and (
        formats.input_format in [DataFormat.UInt32, DataFormat.UInt16]
        or reduce_pool == ReducePool.Max
    ):
        pytest.skip(
            f"Skipping negative_numbers=True for unsigned format {formats.input_format}"
        )

    input_dimensions = dimension_combinations
    torch_format = format_dict[formats.input_format]

    # STIMULI GENERATION
    ELEMENTS_PER_TILE = 1024  # 32 * 32
    tile_cnt = input_dimensions[0] * input_dimensions[1] // ELEMENTS_PER_TILE
    max, min = 1000, (
        0 if formats.input_format in [DataFormat.UInt32, DataFormat.UInt16] else -1000
    )
    src_A = torch.randint(
        low=min, high=max, size=(tile_cnt * 1024,), dtype=torch_format
    )
    src_B = torch.zeros_like(src_A)

    sign = -1 if negative_number else 1
    src_A *= sign

    # Max Reduction can do block and single tile reduction whereas Sum/Avg only do single tile reduction, convert Sum/Avg golden to do block reduction by retilizing input to src_A
    # Dimensions for Max reduction work column wise, for Sum/Avg processing tiles independently is same as column reduction on dst block dimension [32, num_tiles * 32] where num rows is 32 i.e RT_DIM=1 (same as a single tile)
    dst_dim = input_dimensions
    if reduce_pool != ReducePool.Max:
        dst_dim = [32, tile_cnt * 32]
    src_A = tilize_block(
        src_A, dst_dim, stimuli_format=formats.input_format
    ).flatten()  # Input tensor is tilized in dst register
    src_A_untilized = untilize_block(
        src_A, formats.input_format, dst_dim
    )  # Passed into golden since PyTorch library has no concept of tilization

    golden_tensor = get_golden_generator(UnarySFPUGolden)(
        MathOperation.ReduceColumn,
        src_A_untilized,
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
        "disable_format_inference": True,
        "unpack_to_dest": True,
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

    res_from_L1 = collect_results(formats, tile_count=tile_cnt, address=res_address)
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = untilize_block(res_tensor, formats.output_format, dst_dim)

    assert passed_test(golden_tensor[0], res_tensor[0], formats.output_format)
