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
    PerfRunType,
    ReducePool,
    UnpackerEngine,
    format_dict,
)
from helpers.logger import logger
from helpers.param_config import parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# A row reduce needs its whole block in Dest at once. Dest holds 4 tiles at 32-bit and 8 at
# 16-bit, so the narrower limit bounds the suite for every format.
MAX_TILES = 4

REDUCE_POOLS = [ReducePool.Sum, ReducePool.Average, ReducePool.Max, ReducePool.Min]

# Formats the kernel implements. Floats load with sfpmem::DEFAULT (Dest word format resolved at
# runtime); Int32 names sfpmem::INT32 outright.
REDUCE_INPUT_FORMATS = [
    DataFormat.Float32,
    DataFormat.Float16_b,
    DataFormat.Float16,
    DataFormat.Int32,
]

# Relative precision per float format: bf16 has 7 explicit mantissa bits, fp16 10, fp32 23.
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

# Pad sentinels for the sub-tile column reduce, and the extremes the Int32 guard injects. From
# torch.iinfo so the limits are named rather than hex literals.
INT32_MAX = torch.iinfo(torch.int32).max  # 0x7FFFFFFF
INT32_MIN = torch.iinfo(torch.int32).min  # -0x80000000
# MAX pad for Int32. INT32_MIN would work just as well: the Quasar comparator ranks it as the
# minimum, which test_int32_reduce_extreme measures. This value only tracks ttnn's get_pad_value.
INT32_PAD_MIN = INT32_MIN + 1  # -0x7FFFFFFF
# Float pads: finite, so PCC stays defined, and beyond every stimulus. 3e30 needs an 8-bit
# exponent, which fp16 does not have, so Float16 gets its own pair inside its 65504 ceiling.
FLOAT_PAD_MIN = -3.0e30
FLOAT_PAD_MAX = 3.0e30
FLOAT16_PAD_MIN = -6.0e4
FLOAT16_PAD_MAX = 6.0e4


def get_reduce_formats(reduce_pool: ReducePool) -> list[InputOutputFormat]:
    """Input/output format pairs for the reduce suite.

    Output always matches input: the reduce accumulates in the SFPU's fp32 lanes and stores back
    through the same Dest word, so there is no widening step to model.
    """
    return [InputOutputFormat(fmt, fmt) for fmt in REDUCE_INPUT_FORMATS]


def get_supported_reduce_axes(
    reduce_pool: ReducePool, formats: InputOutputFormat
) -> list[MathOperation]:
    """Reduce axes the kernel supports for this pool/format pair.

    Integer AVG is column-only. A column always divides by 32, so the kernel does it with a shift
    (truncating toward zero, matching Blackhole and the golden). A row divides by the runtime
    column count, which only the float reciprocal-multiply divides exactly.
    """
    if reduce_pool == ReducePool.Average and formats.input_format.is_integer():
        return [MathOperation.ReduceColumn]
    return [MathOperation.ReduceColumn, MathOperation.ReduceRow]


def get_format_input_bounds(formats: InputOutputFormat) -> list[tuple[int, int]]:
    """Stimuli ranges per format.

    The all-negative range is what matters for MAX/MIN. Floats are compared as fp32 (SFPSWAP
    imm12 bit 0), and both-negative pairs are the case where getting that wrong shows up, so this
    range is what confirms the hardware ordering.

    Float16 gets a tighter magnitude: the widest block folds 128 terms into one element, which at
    +/-1000 would overflow its 65504 ceiling. The other formats have headroom.
    """
    limit = 100 if formats.input_format == DataFormat.Float16 else 1000
    return [(-limit, limit), (0, limit), (-limit, 0)]


def get_reduce_pad_value(reduce_pool: ReducePool, input_format: DataFormat):
    """Fill value for the padded rows of a sub-tile column reduce, mirroring ttnn's get_pad_value.

    The pad must never win, so that reducing the full 32 rows on device matches a golden reduced
    over only the real ones. A wrong value shows up as padding leaking into the answer.

    Average is excluded from the sub-tile sweep, so only Max, Min and Sum need one.
    """
    if reduce_pool == ReducePool.Max:
        if input_format == DataFormat.Int32:
            return INT32_PAD_MIN
        if input_format == DataFormat.Float16:
            return FLOAT16_PAD_MIN
        return FLOAT_PAD_MIN
    if reduce_pool == ReducePool.Min:
        if input_format == DataFormat.Int32:
            return INT32_MAX
        if input_format == DataFormat.Float16:
            return FLOAT16_PAD_MAX
        return FLOAT_PAD_MAX
    return 0  # Sum: additive identity


def get_reduce_extents(
    mathop: MathOperation,
    reduce_pool: ReducePool,
    formats: InputOutputFormat,
    dimension_combinations: list[int],
) -> list[int]:
    """How many rows of the column-reduce axis hold real data.

    32 is the full tile, which every variant runs. Anything less keeps only the first N rows and
    pads the rest with the reduction identity - the path ttnn takes for ``dim=0``/``dim=1``.

    Only swept where it means something:
      * ``ReduceColumn`` - only the column axis has a paddable 32-row extent.
      * a single tile ``[32, 32]`` - padding does not depend on the tile count, so other
        dimensions would only add redundant cases.
      * Sum/Max/Min - padding is exact for those. Average always divides by 32 (the kernel uses
        a hard-coded 1/32), which would not match a golden averaged over only the real rows.

    Int32 sweeps the small extents since its sentinel is the interesting one; the floats get a
    thin slice just to keep the path guarded.
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

    Summing N terms of magnitude M in a low-precision float accumulates error like
    sqrt(N) * M * eps. Where terms nearly cancel, that error dwarfs the tiny true total and a fixed
    tolerance would fail a correct reduction - so size atol to that bound, with a 2x margin.

    Returns None for Max and for the integer formats, which reduce exactly.
    """
    if reduce_pool not in (ReducePool.Sum, ReducePool.Average):
        return None

    eps = _FLOAT_FORMAT_EPS.get(output_format)
    if eps is None:  # integer formats reduce exactly
        return None

    max_term = max(abs(input_bounds[0]), abs(input_bounds[1]))
    # Terms folded per output element. A row spans the block's full width; a column spans only its
    # real rows, since padded ones add nothing and so round nothing either.
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
    *,
    run_types=(PerfRunType.L1_TO_L1,),
    loop_factor=1,
    is_perf=False,
    perf_report=None,
):
    """SFPU reduce on Quasar: collapse a Dest block along one axis with SUM, AVG or MAX.

    ReduceColumn folds each tile's 32 rows onto its row 0, so tiles reduce independently.
    ReduceRow folds a tile row's columns onto column 0, which spans every tile in that row, so the
    whole block has to be in Dest at once.

    The kernel writes only the axis it collapses and leaves the rest of the tile holding leftovers,
    so the assert compares just that axis - which is what a reduce's consumers read.
    """
    # Not an axis: unpack-to-dest needs the L1 format's width to match the Dest word's, so the
    # pairing is forced. Nothing is lost by a narrow Dest - the SFPU still folds in fp32 lanes and
    # stores once per output element, so it only rounds what the packer would have rounded anyway.
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
        # Fractional, not whole-numbered floats, so the accumulate/round path is really exercised.
        src_A = torch.empty(stimuli_size, dtype=torch_format).uniform_(
            min_value, max_value
        )
    src_B = torch.zeros_like(src_A)

    # Sub-tile column reduce: keep the first `reduced_extent` rows as real data and pad the rest
    # with the reduction identity, the way ttnn does for dim=0/dim=1. The real data has to win, so
    # that reducing all 32 rows on device matches a golden over only the real ones.
    if mathop == MathOperation.ReduceColumn and reduced_extent < TILE_DIM:
        pad_value = get_reduce_pad_value(reduce_pool, formats.input_format)
        src_A = src_A.view(TILE_DIM, tile_cnt * TILE_DIM)
        src_A[reduced_extent:, :] = pad_value
        src_A = src_A.flatten()

    # Column reduces treat tiles independently, so lay the golden out as one 32-row strip of
    # tile_cnt tiles. A row reduce needs the real 2-D block.
    dst_dim = (
        [TILE_DIM, tile_cnt * TILE_DIM]
        if mathop == MathOperation.ReduceColumn
        else input_dimensions
    )

    src_A = tilize_block(src_A, dst_dim, stimuli_format=formats.input_format).flatten()
    # Golden runs on the untilized view - torch knows nothing about tilization.
    src_A_untilized = untilize_block(src_A, formats.input_format, dst_dim)

    # The padded rows must not reach the golden.
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

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/sfpu_reduce_quasar_test.cpp",
        "formats": formats,
        "templates": [
            MATH_OP(mathop=mathop, pool_type=reduce_pool),
            generate_input_dim(input_dimensions, input_dimensions),
            IMPLIED_MATH_FORMAT(),
            # SFPU-only op: operands reach Dest through unpack-to-dest, with no FPU
            # datacopy staging them via SrcA.
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
            DEST_SYNC(),
        ],
        "runtimes": [
            TILE_COUNT(tile_cnt),
            NUM_FACES(4),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=tile_cnt,
            num_faces=4,
            # Int32 reaches Dest as two's-complement - what SFPIADD adds in and what
            # SFPSWAP's compare orders.
            twos_complement=formats.input_format == DataFormat.Int32,
        ),
        "dest_acc": dest_acc,
        "unpack_to_dest": True,
        "disable_format_inference": True,
        "compile_time_formats": True,
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )

    if is_perf:
        configuration.run(perf_report)
        return

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


def _run_int32_reduce(mathop, reduce_pool, injected_value, base_range):
    """MAX/MIN-reduce one 32x32 Int32 tile with `injected_value` scattered through it.

    Returns (golden_slice, device_slice). A near-copy of the sweep body rather than a shared
    helper: that one is driven by parametrize axes and this one hardcodes the format, the pool and
    a hand-built stimulus, so merging them would need a flag per difference.
    """
    formats = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)
    dest_acc = DestAccumulation.Yes  # Int32 is 32-bit, so it needs a 32-bit Dest
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

    # Six positions in six distinct rows and six distinct columns, so six of the 32 reduced lanes
    # see the extreme value under either axis and the rest reduce ordinary data. One injection
    # could pass on a kernel that drops the lane entirely.
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

    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=(PerfRunType.L1_TO_L1,),
        test_config_kwargs={
            "test_name": "sources/quasar/sfpu_reduce_quasar_test.cpp",
            "formats": formats,
            "templates": [
                MATH_OP(mathop=mathop, pool_type=reduce_pool),
                generate_input_dim(input_dimensions, input_dimensions),
                IMPLIED_MATH_FORMAT(),
                UNPACKER_ENGINE_SEL(UnpackerEngine.UnpDest),
                DEST_SYNC(),
            ],
            "runtimes": [
                TILE_COUNT(tile_cnt),
                NUM_FACES(4),
                TEST_FACE_DIMS(),
                DEST_INDEX(0),
                LOOP_FACTOR(1),
            ],
            "variant_stimuli": StimuliConfig(
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
            "dest_acc": dest_acc,
            "unpack_to_dest": True,
            "disable_format_inference": True,
            "compile_time_formats": True,
        },
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
@pytest.mark.parametrize("reduce_pool", [ReducePool.Max, ReducePool.Min])
@pytest.mark.parametrize(
    "injected_value", [INT32_MIN, INT32_MAX], ids=["INT32_MIN", "INT32_MAX"]
)
@pytest.mark.parametrize(
    "base_range",
    [(-1000, 1000), (-1000, -1), (1, 1000)],
    ids=["mixed", "all_negative", "all_positive"],
)
def test_int32_reduce_extreme(mathop, reduce_pool, injected_value, base_range):
    """Guard for the Int32 range ends of a MAX/MIN reduce, pinning the comparator's domain.

    INT32_MIN (0x80000000) is the one pattern where the two readings of an Int32 word disagree. Read
    as two's complement it is the minimum and loses to everything; read as sign-magnitude it is
    negative zero, ranks as 0, and beats every negative operand instead.

    That matters here because Quasar's Int32 load mode, ``sfpmem::INT32``, aliases the
    sign-magnitude ``SMAG32``, while the kernel assumes two's complement - and no variant in the
    sweep above contains the pattern that would tell the two apart. Measured green, so the load
    preserves the word rather than converting it. Blackhole shipped the other behaviour
    (tenstorrent/tt-metal#44750) until it grew a dedicated compare-and-swap path; this keeps
    Quasar from regressing into it.

    ``all_negative`` with INT32_MIN injected is the decisive case: every real operand is negative,
    so a sign-magnitude comparator would hand MAX the injected value instead of the largest genuine
    one. MIN is swept alongside because it is the same SFPSWAP with its operands reversed, so it
    reads the same domain.
    """
    golden_slice, res_slice = _run_int32_reduce(
        mathop, reduce_pool, injected_value, base_range
    )

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
            "\n{} {} injected={}: {} mismatched lanes\n{}",
            reduce_pool,
            mathop,
            int(injected_value),
            num_mismatch,
            detail,
        )

    assert num_mismatch == 0, (
        f"{num_mismatch} mismatched reduction lanes for {reduce_pool} {mathop} "
        f"injected={int(injected_value)} over base_range={base_range} (see log)"
    )
