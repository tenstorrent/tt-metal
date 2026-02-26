# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
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
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    MATH_OP,
    TILE_COUNT,
    generate_input_dim,
)
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


def get_format_input_bounds(formats: InputOutputFormat) -> list[tuple[int, int]]:
    """Get valid stimuli bounds based on data format.
    - range needs to be cut off at 1000 for Sum reduction kernels with UInt16 input format to avoid overflow.
    """
    if formats.input_format in [DataFormat.UInt32, DataFormat.UInt16]:
        return [(0, 1000)]
    return [(-1000, 1000), (0, 1000), (-1000, 0)]


def get_supported_reduce_axioms(reduce_pool: ReducePool) -> list[MathOperation]:
    if reduce_pool == ReducePool.Sum:
        return [MathOperation.ReduceRow, MathOperation.ReduceColumn]
    return [MathOperation.ReduceColumn]


@parametrize(
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
    mathop=lambda reduce_pool: get_supported_reduce_axioms(reduce_pool),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_bounds=lambda formats: get_format_input_bounds(formats),
    reduce_pool=[ReducePool.Min, ReducePool.Max, ReducePool.Sum, ReducePool.Average],
    dimension_combinations=dimension_combinations,
)
def test_sfpu_reduce(
    formats,
    dest_acc,
    mathop,
    reduce_pool,
    input_bounds,
    dimension_combinations,
    workers_tensix_coordinates,
):

    if reduce_pool in [ReducePool.Average, ReducePool.Min] and TestConfig.WITH_COVERAGE:
        pytest.skip(reason="https://github.com/tenstorrent/tt-llk/issues/1040")

    # if dest_acc == DestAccumulation.No and formats.input_format.is_32_bit():
    # pytest.skip(reason="Dest must be in 32bit mode when input is 32bit")

    min_value, max_value = input_bounds
    input_dimensions = dimension_combinations
    torch_format = format_dict[formats.input_format]

    # STIMULI GENERATION
    ELEMENTS_PER_TILE = 1024
    tile_cnt = input_dimensions[0] * input_dimensions[1] // ELEMENTS_PER_TILE
    src_A = torch.randint(
        low=min_value, high=max_value, size=(tile_cnt * 1024,), dtype=torch_format
    )
    src_B = torch.zeros_like(src_A)
    src_A = torch.ones_like(src_A)

    # Max Reduction can do block and single tile reduction whereas Sum/Avg only do single tile reduction, convert Sum/Avg golden to do block reduction by retilizing input to src_A
    # Dimensions for Max reduction work column wise, for Sum/Avg processing tiles independently is same as column reduction on dst block dimension [32, num_tiles * 32] where num rows is 32 i.e RT_DIM=1 (same as a single tile)
    dst_dim = (
        [32, tile_cnt * 32]
        if mathop == MathOperation.ReduceColumn
        else input_dimensions
    )
    src_A = tilize_block(
        src_A, dst_dim, stimuli_format=formats.input_format
    ).flatten()  # Input tensor is tilized in dst register
    src_A_untilized = untilize_block(
        src_A, formats.input_format, dst_dim
    )  # Passed into golden since PyTorch library has no concept of tilization

    golden_tensor = get_golden_generator(UnarySFPUGolden)(
        mathop,
        src_A_untilized,
        formats.output_format,
        dest_acc,
        formats.input_format,
        dst_dim,
        reduce_pool=reduce_pool,
    )

    configuration = TestConfig(
        "sources/sfpu_reduce_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            APPROX_MODE(ApproximationMode.No),
            MATH_OP(mathop=mathop, pool_type=reduce_pool),
        ],
        runtimes=[TILE_COUNT(tile_cnt)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=1,
            tile_count_res=tile_cnt,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=True,
        disable_format_inference=True,
    )
    res_from_L1 = configuration.run(workers_tensix_coordinates).result

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = untilize_block(res_tensor, formats.output_format, dst_dim)

    if mathop == MathOperation.ReduceColumn:
        assert passed_test(golden_tensor[0], res_tensor[0], formats.output_format)
    elif mathop == MathOperation.ReduceRow:
        assert passed_test(golden_tensor[:, 0], res_tensor[:, 0], formats.output_format)
    else:
        raise ValueError(f"Unsupported math operation: {mathop}")
