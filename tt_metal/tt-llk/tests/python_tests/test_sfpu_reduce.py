# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import skip_for_blackhole
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
from helpers.logger import logger
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

# Integer reduction-identity sentinels used to pad sub-tile column reduces
# (see get_reduce_pad_value). Sourced from torch.iinfo so the limits are named
# rather than hard-coded hex/decimal literals.
INT32_MAX = torch.iinfo(torch.int32).max  # 0x7FFFFFFF
INT32_MIN = torch.iinfo(torch.int32).min  # 0x80000000
# MAX-reduce identity for two's-complement Int32: INT32_MIN would also work as the
# additive-order minimum, but ttnn's get_pad_value uses INT32_MIN + 1, so we match it.
INT32_PAD_MIN = INT32_MIN + 1  # -0x7FFFFFFF
UINT16_MAX = torch.iinfo(torch.uint16).max  # 0xFFFF

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


def get_supported_reduce_axioms(
    reduce_pool: ReducePool, formats: InputOutputFormat
) -> list[MathOperation]:
    # Row reduce (REDUCE_ROW) supports SUM/MAX/MIN for every format and AVG for float formats only
    # (the row AVG divisor is the runtime column count, which only the float reciprocal-multiply
    # divides exactly; integer AVG stays column-only). See ckernel_sfpu_reduce.h::calculate_reduce.
    if reduce_pool in (ReducePool.Sum, ReducePool.Max, ReducePool.Min):
        return [MathOperation.ReduceRow, MathOperation.ReduceColumn]
    # Only Float32/Float16_b: the kernel's `is_float_format` AVG row gate treats just these two as
    # float, so a Float16 row AVG would hit the calculate_reduce static_assert at compile time.
    if reduce_pool == ReducePool.Average and formats.input_format in (
        DataFormat.Float32,
        DataFormat.Float16_b,
    ):
        return [MathOperation.ReduceRow, MathOperation.ReduceColumn]
    return [MathOperation.ReduceColumn]


def use_int32_twos_complement(
    formats: InputOutputFormat, reduce_pool: ReducePool, mathop: MathOperation
) -> bool:
    """Whether Int32 stimuli/results use two's-complement (not sign-magnitude) L1 encoding.

    This matches how ttnn feeds the device: Int32 reduce operands sit in DEST as two's-complement.

    Both the column and the row MAX/MIN paths expect two's-complement operands. On Wormhole they now
    load with plain ``INT32`` (bits preserved) and correct the ordering in software via a signed
    compare-and-swap, so INT32_MIN is handled over the full range (see ckernel_sfpu_reduce.h). On
    Blackhole they still cast two's-complement -> sign-magnitude explicitly around the sign-magnitude
    ``SFPSWAP(VEC_MIN_MAX)`` comparator (tt-isa ``SFPSWAP.md``). Either way they expect two's-complement
    operands. Row MAX must agree with the column path because a multi-axis reduce chains
    column-then-row over the same DEST (the column path leaves two's-complement there), so the row
    path consumes and produces two's-complement just like the column path.

    SUM loads with plain ``INT32`` so the word reaches ``SFPIADD`` (a two's-complement adder)
    unchanged. Sign-magnitude stimuli would hide the SUM bug where ``INT32_2S_COMP`` corrupts
    negatives, so SUM operands are two's-complement too (for both row and column SUM).

    AVG is excluded: it still loads with ``INT32_2S_COMP`` (its divide-by-32 step assumes that
    mode), so sign-magnitude remains the right encoding for it.
    """
    if formats.input_format != DataFormat.Int32:
        return False
    if reduce_pool == ReducePool.Sum:
        return True
    if reduce_pool in (ReducePool.Max, ReducePool.Min):
        # Both column and row MAX/MIN take two's-complement (row MAX now matches the column path so
        # the chained multi-axis reduce is consistent).
        return True
    return False


def get_reduce_pad_value(reduce_pool: ReducePool, input_format: DataFormat):
    """Identity fill for the padded (non-data) rows of a sub-tile column reduce.

    Mirrors ttnn's ``get_pad_value``: the pad must never win the reduction, so the device result
    over the full 32-row tile equals the golden over just the real rows. For Int32 MAX/MIN the
    operands are two's-complement (see ``use_int32_twos_complement``), so the sentinels are the
    ordinary two's-complement reduction identities.
    """
    if reduce_pool == ReducePool.Max:
        if input_format == DataFormat.Int32:
            # INT32_MIN + 1 (matches ttnn get_pad_value; avoids INT32_MIN).
            return INT32_PAD_MIN
        if input_format.is_integer():
            return 0  # unsigned formats: 0 is the smallest representable value
        return -3.0e30  # float "-inf"-ish, finite so PCC stays well-defined
    if reduce_pool == ReducePool.Min:
        if input_format == DataFormat.Int32:
            return INT32_MAX
        if input_format == DataFormat.UInt32:
            # SFPSWAP compares in sign-magnitude (tt-isa SFPSWAP.md), so it only orders UInt32 values
            # with bit 31 clear, i.e. [0, 2^31). The usual MIN identity 0xFFFFFFFF has bit 31 set and
            # reads as the most-negative sign-magnitude value, so it would wrongly win. INT32_MAX
            # (0x7FFFFFFF) is the largest value the comparator ranks as maximal and never wins for
            # stimuli in [0, 1000].
            return INT32_MAX
        if input_format == DataFormat.UInt16:
            # 0xFFFF fits in the 31-bit sign-magnitude positive range the comparator orders.
            return UINT16_MAX
        if input_format.is_integer():
            return INT32_MAX
        return 3.0e30  # float "+inf"-ish
    # Sum (Average is excluded from the sub-tile sweep): additive identity.
    return 0


def get_reduce_extents(
    mathop: MathOperation,
    reduce_pool: ReducePool,
    formats: InputOutputFormat,
    dimension_combinations: list[int],
) -> list[int]:
    """Number of real (unpadded) rows on the column-reduce axis.

    ``TILE_DIM`` (32) is the full-tile case. Smaller values keep only the first N rows as real data
    and pad the rest with the reduction identity, exercising the padded column-reduce path ttnn
    takes for ``dim=0``/``dim=1``. The sub-tile sweep is restricted to:
      * ``ReduceColumn`` (only the column reduce has a paddable 32-row axis),
      * a single tile column ``[32, 32]`` (padding is independent of the column-tile count, so
        sweeping every dimension combo only adds redundant cases), and
      * Max/Min/Sum (identity padding is exact; Average's divisor would not match a golden reduced
        over only the real rows).
    Int32 sweeps the small extents; other formats get a thin slice to keep the padding path guarded.
    """
    full = [TILE_DIM]
    if (
        mathop != MathOperation.ReduceColumn
        or dimension_combinations != [TILE_DIM, TILE_DIM]
        or reduce_pool == ReducePool.Average
    ):
        return full
    if formats.input_format == DataFormat.Int32:
        return [1, 13, 15, 16, 17, 30, 31, TILE_DIM]
    return [15, TILE_DIM]  # thin sanity slice for non-Int32 formats


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
    reduced_extent=get_reduce_extents,
)
def test_sfpu_reduce(
    formats,
    dest_acc,
    mathop,
    reduce_pool,
    input_bounds,
    dimension_combinations,
    reduced_extent,
):

    if reduce_pool in [ReducePool.Average, ReducePool.Min] and TestConfig.WITH_COVERAGE:
        pytest.skip(reason="https://github.com/tenstorrent/tt-llk/issues/1040")

    if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No:
        pytest.skip(
            reason="32-bit formats require DestAccumulation.Yes (HW cannot unpack into SrcA/SrcB)"
        )

    if (
        mathop == MathOperation.ReduceRow
        and reduce_pool in (ReducePool.Max, ReducePool.Min)
        and formats.input_format == DataFormat.UInt16
    ):
        pytest.skip(
            reason="UInt16 row MAX/MIN is unsupported by the kernel: without a 32-bit dest it loads "
            "with LO16 (rejected by the row MAX/MIN static_assert), and with a 32-bit dest it routes "
            "through the INT32 sign-magnitude row path, which does not mask UInt16's high bits and "
            "returns garbage. Column UInt16 MAX/MIN (the ttnn-exercised path) is still covered."
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

    # Sub-tile column reduce: keep only the first `reduced_extent` rows of the 32-row reduce axis as
    # real data and fill the rest with the reduction identity (mirrors the padding ttnn injects when
    # folding dim=0/dim=1 onto H). The real data must win so the result matches a golden reduced over
    # only the real rows.
    if mathop == MathOperation.ReduceColumn and reduced_extent < TILE_DIM:
        pad_value = get_reduce_pad_value(reduce_pool, formats.input_format)
        src_A = src_A.view(TILE_DIM, tile_cnt * TILE_DIM)
        src_A[reduced_extent:, :] = pad_value
        src_A = src_A.flatten()

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

    # Reduce only over the real rows; the padded rows must not contribute to the golden.
    golden_input = src_A_untilized
    if mathop == MathOperation.ReduceColumn and reduced_extent < TILE_DIM:
        golden_input = src_A_untilized[:reduced_extent]

    golden_tensor = get_golden_generator(UnarySFPUGolden)(
        mathop,
        golden_input,
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
            twos_complement=use_int32_twos_complement(formats, reduce_pool, mathop),
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

    assert passed_test(
        golden_slice, res_slice, formats.output_format, custom_atol=reduce_atol
    )


def _run_int32_reduce(mathop, reduce_pool, injected_value, base_range=(-1000, 1000)):
    """Build a single 32x32 Int32 tile, inject `injected_value` at a few scattered
    positions, run the SFPU reduce on device, and return (golden_slice, device_slice).
    """
    formats = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)
    dest_acc = DestAccumulation.Yes  # 32-bit formats require dest accumulation
    input_dimensions = [TILE_DIM, TILE_DIM]
    torch_format = format_dict[formats.input_format]

    tile_cnt = input_dimensions[0] * input_dimensions[1] // ELEMENTS_PER_TILE

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    stimuli_size = (tile_cnt * ELEMENTS_PER_TILE,)
    torch.manual_seed(0)
    src_A = torch.randint(
        low=base_range[0], high=base_range[1], size=stimuli_size, dtype=torch_format
    )

    # Inject the extreme value at a handful of scattered positions (6 of the 32x32 grid). These land
    # in 6 distinct columns and 6 distinct rows, so 6 of the 32 reduced columns (for column reduce) /
    # rows (for row reduce) actually see the extreme value; the remaining lanes just reduce the random
    # data. Positions taken on the 32x32 grid.
    grid = src_A.view(TILE_DIM, TILE_DIM)
    inject_positions = [(0, 0), (5, 7), (13, 3), (20, 20), (31, 31), (7, 15)]
    for r, c in inject_positions:
        grid[r, c] = injected_value
    src_A = grid.flatten()

    dst_dim = (
        [32, tile_cnt * 32]
        if mathop == MathOperation.ReduceColumn
        else input_dimensions
    )

    src_A = tilize_block(src_A, dst_dim, stimuli_format=formats.input_format).flatten()
    src_A_untilized = untilize_block(src_A, formats.input_format, dst_dim)

    golden_tensor = get_golden_generator(UnarySFPUGolden)(
        mathop,
        src_A_untilized,
        formats.output_format,
        dest_acc,
        formats.input_format,
        dst_dim,
        reduce_pool=reduce_pool,
    )

    src_B = torch.zeros_like(src_A)

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
            # Int32 MAX/MIN operands sit in DEST as two's-complement (centralized in one helper).
            twos_complement=use_int32_twos_complement(formats, reduce_pool, mathop),
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
        return golden_tensor[0], res_tensor[0]
    return golden_tensor[:, 0], res_tensor[:, 0]


# The #44750 fix is Wormhole-only: the Blackhole calculate_reduce_max_min_int32 path still converts to
# sign-magnitude around a plain SFPSWAP (INT32_MIN still loads as sign-magnitude "-0"), so this repro
# would still fail there. Skip until the Blackhole fix lands (tracked by tenstorrent/tt-metal#44750).
@skip_for_blackhole
@pytest.mark.parametrize(
    "mathop", [MathOperation.ReduceColumn, MathOperation.ReduceRow]
)
@pytest.mark.parametrize("reduce_pool", [ReducePool.Min, ReducePool.Max])
@pytest.mark.parametrize(
    "injected_value", [INT32_MIN, INT32_MAX], ids=["INT32_MIN", "INT32_MAX"]
)
@pytest.mark.parametrize(
    "base_range",
    [(-1000, 1000), (-1000, -1), (1, 1000)],
    ids=["mixed", "all_negative", "all_positive"],
)
def test_int32_reduce_extreme(mathop, reduce_pool, injected_value, base_range):
    """Repro/guard for tenstorrent/tt-metal#44750: INT32 SFPU reduce (min/max) must stay correct when
    the input contains INT32_MIN or INT32_MAX.

    The sign-magnitude SFPSWAP comparator cannot represent INT32_MIN (0x80000000 loads as sign-magnitude
    "negative zero", ranked as 0), so the earlier INT32_2S_COMP path dropped it. This injects the extreme
    values into an otherwise moderate Int32 tile and checks the device reduction against a torch golden.
    """
    if reduce_pool == ReducePool.Min and TestConfig.WITH_COVERAGE:
        pytest.skip(reason="https://github.com/tenstorrent/tt-llk/issues/1040")

    golden_slice, res_slice = _run_int32_reduce(
        mathop, reduce_pool, injected_value, base_range=base_range
    )

    golden = golden_slice.to(torch.int64)
    res = res_slice.to(torch.int64)

    mismatch = golden != res
    num_mismatch = int(mismatch.sum().item())

    if num_mismatch:
        idxs = torch.nonzero(mismatch).flatten().tolist()
        lines = [
            f"  idx={i}: golden={int(golden[i])} device={int(res[i])}"
            for i in idxs[:12]
        ]
        detail = "\n".join(lines)
        logger.info(
            "\n{} {} injected={}: {} mismatched lanes\n{}",
            reduce_pool,
            mathop,
            int(injected_value),
            num_mismatch,
            detail,
        )

    assert num_mismatch == 0, (
        f"{num_mismatch} mismatched reduction lanes for {reduce_pool} {mathop} "
        f"injected={int(injected_value)} (see stdout)"
    )
