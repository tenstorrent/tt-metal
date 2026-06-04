# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    TILE_DIM,
    TILE_DIMENSIONS,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    MathOperation,
    ReducePool,
    format_dict,
)
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

max_tiles = 4

dimension_combinations = [
    [m, n]
    for m in range(TILE_DIM, max_tiles * TILE_DIM + 1, TILE_DIM)
    for n in range(TILE_DIM, max_tiles * TILE_DIM + 1, TILE_DIM)
    if m * n <= max_tiles * TILE_DIM * TILE_DIM
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


def get_reduce_formats(reduce_pool: ReducePool) -> list[InputOutputFormat]:
    """Pick the I/O data format for a given reduce operation.

    A reduce-sum accumulates many input values into a single output element and
    can exceed the 16-bit range (e.g. a 128-wide row sum of values up to ~1000
    reaches ~128k), which would silently wrap a UInt16 output. To keep the full
    sum we keep UInt16 inputs but widen only the output to UInt32. Min/Max/Average
    reduce at most 32 values per output column and therefore cannot overflow the
    16-bit output, so they stay UInt16 on both sides.
    """
    if reduce_pool == ReducePool.Sum:
        return [InputOutputFormat(DataFormat.UInt16, DataFormat.UInt32)]
    return input_output_formats([DataFormat.UInt16], same=True)


def is_valid_reduce_dimension(mathop, dest_acc, formats, dim):
    """Check if a dimension is valid for the given reduce operation."""

    try:
        num_blocks, _ = get_num_blocks_and_num_tiles_in_block(
            DestSync.Half,
            dest_acc,
            formats,
            dim,
            TILE_DIMENSIONS,
            BlocksCalculationAlgorithm.Standard,
        )
        if mathop == MathOperation.ReduceColumn:
            return True
        else:
            return num_blocks == 1  # ReduceRow needs full matrix in one block in dest
    except ValueError:
        return False


@parametrize(
    formats=lambda reduce_pool: get_reduce_formats(reduce_pool),
    mathop=lambda reduce_pool: get_supported_reduce_axioms(reduce_pool),
    dest_acc=[DestAccumulation.Yes],
    input_bounds=lambda formats: get_format_input_bounds(formats),
    reduce_pool=[ReducePool.Min, ReducePool.Max, ReducePool.Sum, ReducePool.Average],
    dimension_combinations=lambda mathop, dest_acc, formats: [
        dim
        for dim in dimension_combinations
        if is_valid_reduce_dimension(mathop, dest_acc, formats, dim)
    ],
)
def test_sfpu_reduce(
    formats,
    dest_acc,
    mathop,
    reduce_pool,
    input_bounds,
    dimension_combinations,
):

    if reduce_pool in [ReducePool.Average, ReducePool.Min] and TestConfig.WITH_COVERAGE:
        pytest.skip(reason="https://github.com/tenstorrent/tt-llk/issues/1040")

    if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No:
        pytest.skip(
            reason="32-bit formats require DestAccumulation.Yes (HW cannot unpack into SrcA/SrcB)"
        )

    min_value, max_value = input_bounds
    input_dimensions = dimension_combinations
    torch_format = format_dict[formats.input_format]

    print("\n" + "=" * 100)
    print("[DEBUG] TEST PARAMETERS")
    print("=" * 100)
    print(f"[DEBUG] formats            : {formats}")
    print(f"[DEBUG]   input_format     : {formats.input_format}")
    print(f"[DEBUG]   output_format    : {formats.output_format}")
    print(f"[DEBUG] mathop             : {mathop}")
    print(f"[DEBUG] reduce_pool        : {reduce_pool}")
    print(f"[DEBUG] dest_acc           : {dest_acc}")
    print(f"[DEBUG] input_bounds       : {input_bounds}")
    print(f"[DEBUG] input_dimensions   : {input_dimensions}")
    print(f"[DEBUG] torch_format       : {torch_format}")

    # STIMULI GENERATION
    tile_cnt = input_dimensions[0] * input_dimensions[1] // ELEMENTS_PER_TILE

    # Calculate blocking parameters
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        dimension_combinations,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    print(f"[DEBUG] tile_cnt           : {tile_cnt}")
    print(f"[DEBUG] num_blocks         : {num_blocks}")
    print(f"[DEBUG] num_tiles_in_block : {num_tiles_in_block}")
    print(f"[DEBUG] ELEMENTS_PER_TILE  : {ELEMENTS_PER_TILE}")

    # STIMULI GENERATION
    src_A = torch.randint(
        low=min_value,
        high=max_value,
        size=(tile_cnt * ELEMENTS_PER_TILE,),
        dtype=torch_format,
    )
    src_B = torch.zeros_like(src_A)

    # Max Reduction can do block and single tile reduction whereas Sum/Avg only do single tile reduction, convert Sum/Avg golden to do block reduction by retilizing input to src_A
    # Dimensions for Max reduction work column wise, for Sum/Avg processing tiles independently is same as column reduction on dst block dimension [32, num_tiles * 32] where num rows is 32 i.e RT_DIM=1 (same as a single tile)
    dst_dim = (
        [32, tile_cnt * 32]
        if mathop == MathOperation.ReduceColumn
        else input_dimensions
    )

    print("\n" + "-" * 100)
    print("[DEBUG] STIMULI")
    print("-" * 100)
    print(f"[DEBUG] dst_dim                : {dst_dim}")
    print(f"[DEBUG] src_A (pre-tilize) shape: {tuple(src_A.shape)}")
    print(
        f"[DEBUG] src_A (pre-tilize) unique values: {torch.unique(src_A).tolist()[:16]}"
    )

    src_A = tilize_block(
        src_A, dst_dim, stimuli_format=formats.input_format
    ).flatten()  # Input tensor is tilized in dst register
    src_A_untilized = untilize_block(
        src_A, formats.input_format, dst_dim
    )  # Passed into golden since PyTorch library has no concept of tilization

    print(f"[DEBUG] src_A (tilized) shape   : {tuple(src_A.shape)}")
    print(f"[DEBUG] src_A_untilized shape   : {tuple(src_A_untilized.shape)}")
    print(f"[DEBUG] src_A_untilized:\n{src_A_untilized}")

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
        runtimes=[
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            TILE_COUNT(tile_cnt),
        ],
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
        compile_time_formats=True,
    )
    res_from_L1 = configuration.run().result

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    print("\n" + "-" * 100)
    print("[DEBUG] RESULTS")
    print("-" * 100)
    print(f"[DEBUG] raw res_from_L1 len     : {len(res_from_L1)}")
    print(f"[DEBUG] res_tensor (tilized) shape: {tuple(res_tensor.shape)}")

    res_tensor = untilize_block(res_tensor, formats.output_format, dst_dim)

    print(f"[DEBUG] golden_tensor shape     : {tuple(golden_tensor.shape)}")
    print(f"[DEBUG] res_tensor   shape      : {tuple(res_tensor.shape)}")
    print(f"[DEBUG] golden_tensor:\n{golden_tensor}")
    print(f"[DEBUG] res_tensor:\n{res_tensor}")

    if mathop == MathOperation.ReduceColumn:
        golden_slice = golden_tensor[0]
        res_slice = res_tensor[0]
    elif mathop == MathOperation.ReduceRow:
        golden_slice = golden_tensor[:, 0]
        res_slice = res_tensor[:, 0]
    else:
        raise ValueError(f"Unsupported math operation: {mathop}")

    print("\n" + "-" * 100)
    print("[DEBUG] COMPARED SLICE (golden vs result)")
    print("-" * 100)
    print(f"[DEBUG] golden_slice : {golden_slice.tolist()}")
    print(f"[DEBUG] res_slice    : {res_slice.tolist()}")
    diff_idx = (golden_slice != res_slice).nonzero(as_tuple=True)[0].tolist()
    print(f"[DEBUG] mismatch indices ({len(diff_idx)}): {diff_idx}")
    for i in diff_idx:
        print(
            f"[DEBUG]   idx {i:>4}: golden={golden_slice[i].item()} != res={res_slice[i].item()}"
        )

    assert passed_test(golden_slice, res_slice, formats.output_format)
