# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    TILE_DIM,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    MathOperation,
    ReducePool,
    UnpackerEngine,
    format_dict,
)
from helpers.logger import logger
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# SyncHalf with a 32-bit Dest holds 4 tiles of 32x32 (get_dest_max_tiles); a 16-bit Dest holds 8.
# The whole block has to be resident at once because a row reduce spans its entire tile row, so the
# narrower 32-bit ceiling of 4 tiles bounds the suite for every format.
MAX_TILES = 4

# Quasar's PoolType enum is SUM/AVG/MAX - there is no MIN, so the suite covers the three that
# exist. Blackhole/Wormhole additionally expose MIN.
REDUCE_POOLS = [ReducePool.Sum, ReducePool.Average, ReducePool.Max]

# Formats the Quasar kernel implements. Float paths load with sfpmem::DEFAULT (the Dest word
# format resolves at runtime), Int32 names sfpmem::INT32 explicitly.
REDUCE_INPUT_FORMATS = [
    DataFormat.Float32,
    DataFormat.Float16_b,
    DataFormat.Float16,
    DataFormat.Int32,
]

# Relative precision of the float output formats: bf16 has 7 explicit mantissa bits, fp16 has 10,
# fp32 has 23.
_FLOAT_FORMAT_EPS = {
    DataFormat.Float16_b: 2.0**-8,
    DataFormat.Float16: 2.0**-11,
    DataFormat.Float32: 2.0**-24,
}

DIMENSION_COMBINATIONS = [
    [m, n]
    for m in range(TILE_DIM, MAX_TILES * TILE_DIM + 1, TILE_DIM)
    for n in range(TILE_DIM, MAX_TILES * TILE_DIM + 1, TILE_DIM)
    if (m // TILE_DIM) * (n // TILE_DIM) <= MAX_TILES
]

# Reduction-identity sentinels for the padded rows of a sub-tile column reduce, and the extremes
# the Int32 guard injects. Sourced from torch.iinfo so the limits are named, not hex literals.
INT32_MAX = torch.iinfo(torch.int32).max  # 0x7FFFFFFF
INT32_MIN = torch.iinfo(torch.int32).min  # -0x80000000
# MAX-reduce identity for Int32: the two's-complement additive minimum, avoiding INT32_MIN itself
# to match ttnn's get_pad_value. Either value is correct here - measured, the Quasar comparator
# orders INT32_MIN as the minimum (see test_int32_reduce_max_extreme) - so this tracks ttnn rather
# than working around a hazard.
INT32_PAD_MIN = INT32_MIN + 1  # -0x7FFFFFFF
# MAX-reduce identity for the float formats: finite, so PCC stays well-defined, and far below
# every stimulus. Float16's 65504 ceiling cannot hold the 8-bit-exponent sentinel, so it gets its
# own value that is still two orders of magnitude below its +/-100 stimuli range.
FLOAT_PAD_MIN = -3.0e30
FLOAT16_PAD_MIN = -6.0e4


def get_reduce_formats(reduce_pool: ReducePool) -> list[InputOutputFormat]:
    """Input/output format pairs for the reduce suite.

    Input format is kept as the output format: the reduce accumulates in the SFPU's fp32 lanes
    and stores back through the same Dest word, so there is no widening step to model.
    """
    return [InputOutputFormat(fmt, fmt) for fmt in REDUCE_INPUT_FORMATS]


def get_supported_reduce_axes(
    reduce_pool: ReducePool, formats: InputOutputFormat
) -> list[MathOperation]:
    """Reduce axes the kernel supports for this pool/format pair.

    Integer AVG is excluded on both axes: averaging an integer reduction has to round its
    quotient, and calculate_reduce deliberately leaves that choice to the caller rather than
    baking one in (see the static_assert in ckernel_sfpu_reduce.h).
    """
    if reduce_pool == ReducePool.Average and formats.input_format.is_integer():
        return []
    return [MathOperation.ReduceColumn, MathOperation.ReduceRow]


def get_format_input_bounds(formats: InputOutputFormat) -> list[tuple[int, int]]:
    """Stimuli ranges per format.

    Signed ranges matter most for MAX: Quasar's SFPSWAP compares its operands as two's-complement
    int32, which orders IEEE float bits correctly unless *both* are negative. The all-negative
    range is what exercises the kernel's correction swap for those lanes.

    Float16 gets a tighter magnitude because its 65504 ceiling is a real constraint on a Sum: the
    widest block here folds 128 terms into one output element, and at the other formats' +/-1000
    that total would leave the representable range. Every other format has headroom to spare.
    """
    limit = 100 if formats.input_format == DataFormat.Float16 else 1000
    return [(-limit, limit), (0, limit), (-limit, 0)]


def get_reduce_pad_value(reduce_pool: ReducePool, input_format: DataFormat):
    """Identity fill for the padded (non-data) rows of a sub-tile column reduce.

    Mirrors ttnn's ``get_pad_value``: the pad must never win the reduction, so the device result
    over the full 32-row tile equals a golden reduced over only the real rows. A wrong sentinel
    shows up as the padding leaking into the answer.

    Average is excluded from the sub-tile sweep, so only Max and Sum need an identity here.
    """
    if reduce_pool == ReducePool.Max:
        if input_format == DataFormat.Int32:
            return INT32_PAD_MIN
        if input_format == DataFormat.Float16:
            return FLOAT16_PAD_MIN
        return FLOAT_PAD_MIN
    return 0  # Sum: additive identity


def get_reduce_extents(
    mathop: MathOperation,
    reduce_pool: ReducePool,
    formats: InputOutputFormat,
    dimension_combinations: list[int],
) -> list[int]:
    """Number of real (unpadded) rows on the column-reduce axis.

    ``TILE_DIM`` (32) is the full-tile case every variant runs. Smaller values keep only the first
    N rows as real data and fill the rest with the reduction identity, exercising the padded
    column reduce ttnn takes for ``dim=0``/``dim=1``. The sub-tile sweep is restricted to:
      * ``ReduceColumn`` - only the column axis has a paddable 32-row extent,
      * a single tile ``[32, 32]`` - padding is independent of the tile count, so sweeping every
        dimension combination would only add redundant cases, and
      * Sum/Max - identity padding is exact for those. Average divides by a fixed 32 (the kernel
        multiplies by a hard-coded 1/32 reciprocal), which would not match a golden averaged over
        only the real rows.
    Int32 sweeps the small extents because its sentinel is the interesting one; the float formats
    get a thin slice that keeps the padding path guarded.
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
    return [15, TILE_DIM]  # thin sanity slice for the float formats


def get_reduce_atol(
    output_format, reduce_pool, mathop, input_dimensions, input_bounds, reduced_extent
):
    """Absolute tolerance for the accumulating reductions (Sum/Average).

    Summing N terms of magnitude up to M in a low-precision float accumulates rounding error that
    grows like sqrt(N) * M * eps. On rows whose terms nearly cancel that error dwarfs the tiny
    true total, so a fixed tolerance would fail a correct reduction. Sizing atol to that bound
    (with a 2x margin) keeps cancellation rows passing while real errors still fail.

    Returns None for Max and for integer formats, which reduce exactly.
    """
    if reduce_pool not in (ReducePool.Sum, ReducePool.Average):
        return None

    eps = _FLOAT_FORMAT_EPS.get(output_format)
    if eps is None:  # integer formats reduce exactly
        return None

    max_term = max(abs(input_bounds[0]), abs(input_bounds[1]))
    # Terms folded into one output element: a row reduce spans the block's full width, a column
    # reduce spans its real (unpadded) extent - the identity-filled rows add nothing and so
    # contribute no rounding error either.
    num_terms = (
        input_dimensions[1] if mathop == MathOperation.ReduceRow else reduced_extent
    )

    atol = 2.0 * max_term * eps * (num_terms**0.5)
    if reduce_pool == ReducePool.Average:
        atol /= num_terms

    return max(0.05, atol)


@pytest.mark.quasar
@parametrize(
    reduce_pool=REDUCE_POOLS,
    formats=get_reduce_formats,
    mathop=get_supported_reduce_axes,
    input_bounds=get_format_input_bounds,
    dimension_combinations=DIMENSION_COMBINATIONS,
    reduced_extent=get_reduce_extents,
)
def test_sfpu_reduce_quasar(
    formats,
    mathop,
    reduce_pool,
    input_bounds,
    dimension_combinations,
    reduced_extent,
):
    """SFPU reduce on Quasar: collapse a Dest block along one axis with SUM, AVG or MAX.

    ReduceColumn folds each tile's 32 rows onto that tile's row 0, so tiles reduce independently.
    ReduceRow folds a tile row's columns onto its column 0, which spans every tile in that row,
    so the whole block must be resident in Dest at once.

    The kernel writes only the axis it collapses onto and leaves the rest of the tile holding
    reduction leftovers, so the assertion compares just that axis - matching what the packer's
    consumers read after a reduce.
    """
    # Quasar's unpack-to-dest path needs the L1 format's width to match the Dest word's, so the
    # 32-bit formats take a 32-bit Dest and the 16-bit ones a 16-bit Dest. Nothing is lost by the
    # narrow Dest: the SFPU still folds in fp32 lanes and the reduce stores once per output
    # element, so the narrow format only rounds the result the packer was going to see anyway.
    dest_acc = (
        DestAccumulation.Yes
        if formats.input_format.is_32_bit()
        else DestAccumulation.No
    )

    min_value, max_value = input_bounds
    input_dimensions = dimension_combinations
    torch_format = format_dict[formats.input_format]

    tile_cnt = input_dimensions[0] * input_dimensions[1] // ELEMENTS_PER_TILE

    torch.manual_seed(0)
    stimuli_size = (tile_cnt * ELEMENTS_PER_TILE,)
    if formats.input_format.is_integer():
        src_A = torch.randint(
            low=min_value, high=max_value, size=stimuli_size, dtype=torch_format
        )
    else:
        # Fractional values, not integer-valued floats, so the accumulate/round path is actually
        # exercised (randint would only ever produce whole numbers).
        src_A = torch.empty(stimuli_size, dtype=torch_format).uniform_(
            min_value, max_value
        )
    src_B = torch.zeros_like(src_A)

    # Sub-tile column reduce: keep only the first `reduced_extent` rows of the 32-row reduce axis
    # as real data and fill the rest with the reduction identity, mirroring the padding ttnn
    # injects when folding dim=0/dim=1 onto H. The real data must win, so the device result over
    # the full tile matches a golden reduced over only the real rows.
    if mathop == MathOperation.ReduceColumn and reduced_extent < TILE_DIM:
        pad_value = get_reduce_pad_value(reduce_pool, formats.input_format)
        src_A = src_A.view(TILE_DIM, tile_cnt * TILE_DIM)
        src_A[reduced_extent:, :] = pad_value
        src_A = src_A.flatten()

    # A column reduce treats every tile independently, so its golden is the block laid out as a
    # single 32-row strip of tile_cnt tile columns. A row reduce spans the real 2-D block.
    dst_dim = (
        [TILE_DIM, tile_cnt * TILE_DIM]
        if mathop == MathOperation.ReduceColumn
        else input_dimensions
    )

    src_A = tilize_block(src_A, dst_dim, stimuli_format=formats.input_format).flatten()
    # Golden is computed on the untilized view; torch has no concept of tilization.
    src_A_untilized = untilize_block(src_A, formats.input_format, dst_dim)

    # Reduce over the real rows only; the padded rows must not reach the golden.
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
        "sources/quasar/sfpu_reduce_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=mathop, pool_type=reduce_pool),
            generate_input_dim(input_dimensions, input_dimensions),
            IMPLIED_MATH_FORMAT(),
            # The reduce is SFPU-only: operands reach DEST through the unpack-to-dest
            # engine, with no FPU datacopy staging them via SrcA.
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
            NUM_FACES(4),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=tile_cnt,
            num_faces=4,
            # Int32 operands reach Dest as two's-complement, which is what SFPIADD adds in and
            # what SFPSWAP's two's-complement compare orders.
            twos_complement=formats.input_format == DataFormat.Int32,
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
        golden_slice, res_slice = golden_tensor[0], res_tensor[0]
    else:
        golden_slice, res_slice = golden_tensor[:, 0], res_tensor[:, 0]

    assert passed_test(
        golden_slice,
        res_slice,
        formats.output_format,
        custom_atol=get_reduce_atol(
            formats.output_format,
            reduce_pool,
            mathop,
            input_dimensions,
            input_bounds,
            reduced_extent,
        ),
    ), "Assert against golden failed"


def _run_int32_reduce_max(mathop, injected_value, base_range):
    """One 32x32 Int32 tile with `injected_value` at a few scattered positions, MAX-reduced on
    device. Returns (golden_slice, device_slice).

    A near-copy of the sweep body above rather than a shared helper: that one is driven by the
    parametrize axes and this one hardcodes the format, the pool and a hand-built stimulus, so
    threading both through one function would need a flag per difference.
    """
    formats = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)
    dest_acc = DestAccumulation.Yes  # Int32 is 32-bit, so it needs a 32-bit Dest
    reduce_pool = ReducePool.Max
    input_dimensions = [TILE_DIM, TILE_DIM]
    torch_format = format_dict[formats.input_format]

    tile_cnt = input_dimensions[0] * input_dimensions[1] // ELEMENTS_PER_TILE

    torch.manual_seed(0)
    src_A = torch.randint(
        low=base_range[0],
        high=base_range[1],
        size=(tile_cnt * ELEMENTS_PER_TILE,),
        dtype=torch_format,
    )

    # Six scattered positions, in six distinct rows and six distinct columns, so six of the 32
    # reduced lanes actually see the extreme value under either axis and the rest reduce ordinary
    # data. A single injection could pass on a kernel that drops the lane entirely.
    grid = src_A.view(TILE_DIM, TILE_DIM)
    for r, c in [(0, 0), (5, 7), (13, 3), (20, 20), (31, 31), (7, 15)]:
        grid[r, c] = injected_value
    src_A = grid.flatten()

    dst_dim = (
        [TILE_DIM, tile_cnt * TILE_DIM]
        if mathop == MathOperation.ReduceColumn
        else input_dimensions
    )

    src_A = tilize_block(src_A, dst_dim, stimuli_format=formats.input_format).flatten()
    src_A_untilized = untilize_block(src_A, formats.input_format, dst_dim)
    src_B = torch.zeros_like(src_A)

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
        "sources/quasar/sfpu_reduce_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=mathop, pool_type=reduce_pool),
            generate_input_dim(input_dimensions, input_dimensions),
            IMPLIED_MATH_FORMAT(),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
            NUM_FACES(4),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=tile_cnt,
            twos_complement=True,
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


@pytest.mark.quasar
@pytest.mark.parametrize(
    "mathop", [MathOperation.ReduceColumn, MathOperation.ReduceRow]
)
@pytest.mark.parametrize(
    "injected_value", [INT32_MIN, INT32_MAX], ids=["INT32_MIN", "INT32_MAX"]
)
@pytest.mark.parametrize(
    "base_range",
    [(-1000, 1000), (-1000, -1), (1, 1000)],
    ids=["mixed", "all_negative", "all_positive"],
)
def test_int32_reduce_max_extreme(mathop, injected_value, base_range):
    """Guard for the Int32 range ends of a MAX reduce - the one bit pattern that pins the
    comparator's domain.

    Quasar's Int32 load mode is ``p_sfpu::sfpmem::INT32``, which aliases ``SMAG32``, the
    sign-magnitude int32 mode. Read as sign-magnitude, INT32_MIN (0x80000000) is negative zero and
    ranks as 0 - beating every negative operand instead of losing to them; read as two's
    complement it is the minimum and loses to all of them. The kernel's ``reduce_combine``
    documents the latter (SFPSWAP compares as two's complement, Dest already holds
    two's-complement Int32) and the sweep above feeds ``twos_complement=True`` on that basis, but
    no sweep variant contains the pattern that separates the two readings.

    Measured green on Quasar: the SMAG32 load preserves the word rather than converting it, so the
    documented two's-complement ordering holds and the padding sentinel above is free to follow
    ttnn. This test is what keeps that true - Blackhole shipped the other behaviour
    (tenstorrent/tt-metal#44750) until it grew a dedicated two's-complement compare-and-swap path.

    ``all_negative`` is the decisive range: with INT32_MIN injected every real operand is negative,
    so a sign-magnitude comparator would return the injected value as the maximum while the golden
    returns the largest genuine one.
    """
    golden_slice, res_slice = _run_int32_reduce_max(mathop, injected_value, base_range)

    golden = golden_slice.to(torch.int64)
    res = res_slice.to(torch.int64)

    mismatch = golden != res
    num_mismatch = int(mismatch.sum().item())

    if num_mismatch:
        idxs = torch.nonzero(mismatch).flatten().tolist()
        detail = "\n".join(
            f"  idx={i}: golden={int(golden[i])} device={int(res[i])}"
            for i in idxs[:12]
        )
        logger.info(
            "\nMax {} injected={}: {} mismatched lanes\n{}",
            mathop,
            int(injected_value),
            num_mismatch,
            detail,
        )

    assert num_mismatch == 0, (
        f"{num_mismatch} mismatched reduction lanes for Max {mathop} "
        f"injected={int(injected_value)} over base_range={base_range} (see log)"
    )
