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


# Base data formats exercised by the reduce suite.
REDUCE_BASE_FORMATS = [
    DataFormat.Float32,
    DataFormat.Int32,
    DataFormat.UInt32,
    DataFormat.UInt16,
    DataFormat.Float16_b,
]


def get_reduce_formats(reduce_pool: ReducePool) -> list[InputOutputFormat]:
    """Input/output format pairs for the reduce suite.

    A UInt16 Sum/Average reduction can exceed the UInt16 range: a single 32-wide tile row/column
    sums up to 32 values, and a multi-tile row reduction sums even more (e.g. 128 columns of up to
    1000 reaches ~128000 >> 65535). To avoid output overflow we widen the OUTPUT to UInt32 for those
    cases; the SFPU already accumulates in 32-bit and stores the full word into a 32-bit (fp32) dest.
    Every other case keeps input == output.
    """
    widening = reduce_pool in (ReducePool.Sum, ReducePool.Average)
    return [
        (
            InputOutputFormat(fmt, DataFormat.UInt32)
            if (widening and fmt == DataFormat.UInt16)
            else InputOutputFormat(fmt, fmt)
        )
        for fmt in REDUCE_BASE_FORMATS
    ]


# Relative precision (unit roundoff) of the floating-point dest/output formats.
# bf16 has 7 explicit mantissa bits, fp16 has 10, fp32 has 23.
_FLOAT_FORMAT_EPS = {
    DataFormat.Float16_b: 2.0**-8,
    DataFormat.Float16: 2.0**-11,
    DataFormat.Float32: 2.0**-24,
}


def get_reduce_sum_atol(
    output_format, reduce_pool, mathop, input_dimensions, input_bounds
):
    """Absolute tolerance for accumulating float reductions (Sum/Average).

    Summing N values of magnitude up to M in a low-precision float format accumulates
    rounding error that scales like sqrt(N) * M * eps (the partial sums grow ~sqrt(N)*M
    and each store rounds by ~eps). On rows whose terms nearly cancel, this absolute
    error dwarfs the tiny true result, so a fixed atol/rtol spuriously fails even though
    the hardware reduction is correct (PCC stays ~0.99999). We size atol to that bound
    (with a 2x safety margin) so cancellation rows pass while genuine errors still fail.

    Returns None for non-accumulating ops (Max/Min) and integer formats, leaving the
    default exact/loose tolerances in place.
    """
    if reduce_pool not in (ReducePool.Sum, ReducePool.Average):
        return None

    eps = _FLOAT_FORMAT_EPS.get(output_format)
    if eps is None:  # integer formats reduce exactly; keep exact comparison
        return None

    max_term = max(abs(input_bounds[0]), abs(input_bounds[1]))
    # Number of terms accumulated per output element.
    num_terms = input_dimensions[1] if mathop == MathOperation.ReduceRow else TILE_DIM

    safety_factor = 2.0
    atol = safety_factor * max_term * eps * (num_terms**0.5)

    # Average divides the accumulated sum by the reduced extent, shrinking the error too.
    if reduce_pool == ReducePool.Average:
        atol /= num_terms

    # Keep at least the baseline absolute tolerance used elsewhere.
    return max(0.05, atol)


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
    formats=get_reduce_formats,
    mathop=get_supported_reduce_axioms,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    input_bounds=get_format_input_bounds,
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

    if (
        formats.input_format == DataFormat.UInt16
        and formats.output_format.is_32_bit()
        and dest_acc == DestAccumulation.No
    ):
        pytest.skip(
            reason="UInt16 Sum/Average widens the output to UInt32, which needs a 32-bit (fp32) dest; "
            "DestAccumulation.No has no room to store/pack the widened result"
        )

    min_value, max_value = input_bounds
    input_dimensions = dimension_combinations
    torch_format = format_dict[formats.input_format]

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

    # STIMULI GENERATION
    stimuli_size = (tile_cnt * ELEMENTS_PER_TILE,)
    if formats.input_format.is_integer():
        src_A = torch.randint(
            low=min_value,
            high=max_value,
            size=stimuli_size,
            dtype=torch_format,
        )
    else:
        # Float formats need real fractional values, not integer-valued floats, so the
        # float accumulation/rounding paths are actually exercised (randint would only
        # ever produce whole numbers like 42.0).
        src_A = torch.empty(stimuli_size, dtype=torch_format).uniform_(
            min_value, max_value
        )
    src_B = torch.zeros_like(src_A)

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
    res_tensor = untilize_block(res_tensor, formats.output_format, dst_dim)

    if mathop == MathOperation.ReduceColumn:
        golden_slice = golden_tensor[0]
        res_slice = res_tensor[0]
    elif mathop == MathOperation.ReduceRow:
        golden_slice = golden_tensor[:, 0]
        res_slice = res_tensor[:, 0]
    else:
        raise ValueError(f"Unsupported math operation: {mathop}")

    # Accumulating float reductions lose precision proportional to the number of summed
    # terms; size the absolute tolerance accordingly (PCC still guards correctness).
    reduce_atol = get_reduce_sum_atol(
        formats.output_format, reduce_pool, mathop, input_dimensions, input_bounds
    )

    passed = passed_test(
        golden_slice, res_slice, formats.output_format, custom_atol=reduce_atol
    )

    if not passed:
        # Dump the exact input/golden/result to the (CI) console so a non-reproducible-locally
        # failure can be replayed. Print the full tensors (torch truncates by default) and the
        # tilized src_A bytes that were actually sent to L1 so the case can be hardcoded.
        torch.set_printoptions(
            threshold=1_000_000, precision=6, sci_mode=False, linewidth=200
        )
        print("\n================= SFPU REDUCE FAILURE DEBUG =================")
        print(
            f"formats={formats} mathop={mathop} dest_acc={dest_acc} "
            f"reduce_pool={reduce_pool} input_bounds={input_bounds} dims={input_dimensions}"
        )
        print(
            f"num_blocks={num_blocks} num_tiles_in_block={num_tiles_in_block} tile_cnt={tile_cnt}"
        )
        print(f"src_A (tilized, sent to L1):\n{src_A.tolist()}")
        print(f"src_A_untilized (golden input):\n{src_A_untilized}")
        print(f"golden_slice:\n{golden_slice}")
        print(f"res_slice:\n{res_slice}")
        mism = (golden_slice != res_slice).nonzero().flatten().tolist()
        print(f"mismatch indices ({len(mism)}): {mism}")
        print("============================================================\n")

    assert passed
