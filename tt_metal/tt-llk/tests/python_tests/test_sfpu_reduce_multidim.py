# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Multi-dimensional (column-then-row) SFPU reduce tests.

These reduce over BOTH dims of every 32x32 tile by chaining a REDUCE_COL pass and a REDUCE_ROW
pass under a SINGLE shared ``init_reduce`` (see ``sources/sfpu_reduce_multidim_test.cpp``), which
mirrors how a multi-axis reduce (e.g. ``ttir.max`` ``dim=[1,2]``) is lowered to two ``sfpu_reduce``
calls sharing one ``sfpu_reduce_init``.

This configuration is what exposed an Int32 MAX/MIN regression: the column Int32 path runs first
(``INT32_2S_COMP``, and its own ``init_reduce_max_min_int32`` that flips the SFPSWAP-direction config
bit), then the row path runs trusting state established by the shared init. Because the row stage
inherited the column stage's inverted comparator direction, it misordered values and returned wrong
results once negatives/extremes were present. The single-axis ``test_sfpu_reduce`` suite only
exercises one path per init (with its matching init) and therefore could not catch it.

Pool coverage is MAX, SUM, MIN (all formats) and AVG (float formats only): the kernel's ``REDUCE_ROW``
path supports SUM/MAX/MIN for all formats and AVG for float formats, so each is expressible as a
column-then-row chain at the LLK API level. MIN in particular exercises the same shared-init hazard as
MAX: the row stage re-establishes the SFPSWAP comparator direction (SFPCONFIG bit 8) rather than
trusting whatever a preceding column calculate left there. AVG is a meaningful multi-axis reduction on a
full 32x32 tile: the column pass writes each column's mean (over 32 rows) into row 0, then the row pass
averages those 32 column-means (over 32 columns) into element [0][0], which equals the overall tile
mean. It stays float-only because integer row AVG is unsupported (the row divisor is the runtime column
count, divided exactly only by the float reciprocal-multiply).
"""

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM, TILE_DIMENSIONS
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    ReducePool,
    format_dict,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    REDUCE_POOL_TYPE,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# Number of stacked 32x32 row-tiles to reduce independently. Each tile yields one multi-axis
# reduction result, so >= 2 keeps the PCC check non-degenerate. Stays <= 4 so the whole column of
# tiles fits in one (fp32) dest section, which REDUCE_ROW requires.
ROW_TILE_COUNTS = [2, 4]

# Float input formats: the only ones whose REDUCE_ROW AVG path exists in the kernel.
MULTIDIM_FLOAT_FORMATS = (DataFormat.Float32, DataFormat.Float16_b)


def get_multidim_pools(formats: InputOutputFormat) -> list[ReducePool]:
    """Pools whose REDUCE_ROW path exists in the kernel (so a column-then-row chain is valid).

    SUM/MAX/MIN work for every format. AVG only for float: integer row AVG is unsupported (the row
    divisor is the runtime column count, divided exactly only by the float reciprocal-multiply), so an
    integer avg-of-avgs chain is not expressible.
    """
    pools = [ReducePool.Max, ReducePool.Sum, ReducePool.Min]
    if formats.input_format in MULTIDIM_FLOAT_FORMATS:
        pools.append(ReducePool.Average)
    return pools


# Formats exercised. Int32 is the regression target (its column path flips the SFPSWAP-direction
# config and uses two's-complement); the others are baselines that should pass on both buggy and
# fixed kernels and guard the shared-init chain for non-Int32 paths.
MULTIDIM_FORMATS = [
    DataFormat.Int32,
    DataFormat.Float32,
    DataFormat.UInt32,
    DataFormat.Float16_b,
]


def get_multidim_input_bounds(formats: InputOutputFormat) -> list[tuple[int, int]]:
    """Stimuli bounds. Signed formats include negative ranges, which are required to expose the
    Int32 MAX regression (the row stage running with the column stage's inverted comparator only
    misorders values once negatives/extremes are present)."""
    if formats.input_format in (DataFormat.UInt32, DataFormat.UInt16):
        return [(0, 1000)]
    return [(-1000, 1000), (-1000, 0)]


def use_int32_twos_complement(formats: InputOutputFormat) -> bool:
    """The column MAX/MIN path expects two's-complement operands in DEST (it casts 2sC -> sign-
    magnitude around the SFPSWAP and back), which is how ttnn feeds the device. SUM also reaches a
    two's-complement adder. So Int32 multi-dim stimuli/results use two's-complement encoding.
    """
    return formats.input_format == DataFormat.Int32


# Relative precision (unit roundoff) of the float output formats (bf16: 7 mantissa bits, fp32: 23).
_FLOAT_FORMAT_EPS = {
    DataFormat.Float16_b: 2.0**-8,
    DataFormat.Float32: 2.0**-24,
}


def get_multidim_sum_avg_atol(output_format, reduce_pool, input_bounds) -> float | None:
    """Absolute tolerance for the float multi-axis SUM/AVG chain.

    Both chains accumulate N = 32*32 terms (a column pass over 32 rows then a row pass over 32
    column results). Low-precision float accumulation/rounding contributes ~ max_term * eps * sqrt(N)
    absolute error to the underlying sum. SUM keeps that error; AVG's divide-by-N shrinks it to
    ~ max_term * eps / sqrt(N). On near-cancelling tiles the true result is tiny, so the fixed 0.05
    atol can spuriously fail even though PCC stays ~1.0; size atol to that bound (2x margin). Returns
    None for MAX/MIN and integer formats, leaving the default tolerances in place.
    """
    if reduce_pool not in (ReducePool.Sum, ReducePool.Average):
        return None
    eps = _FLOAT_FORMAT_EPS.get(output_format)
    if eps is None:
        return None
    num_terms = TILE_DIM * TILE_DIM
    max_term = max(abs(input_bounds[0]), abs(input_bounds[1]))
    sum_error = 2.0 * max_term * eps * (num_terms**0.5)
    atol = sum_error if reduce_pool == ReducePool.Sum else sum_error / num_terms
    return max(0.05, atol)


def reduce_block(block: torch.Tensor, reduce_pool: ReducePool) -> torch.Tensor:
    """Reduce a 32x32 tile over both dims to a single value (the multi-axis golden)."""
    flat = block.reshape(-1)
    if reduce_pool == ReducePool.Max:
        return flat.max()
    if reduce_pool == ReducePool.Sum:
        return flat.sum()
    if reduce_pool == ReducePool.Min:
        return flat.min()
    if reduce_pool == ReducePool.Average:
        # avg-of-column-means over a full 32x32 tile equals the overall tile mean (float-only).
        return flat.mean()
    raise ValueError(f"Unsupported multi-dim reduce pool: {reduce_pool}")


@parametrize(
    formats=[InputOutputFormat(fmt, fmt) for fmt in MULTIDIM_FORMATS],
    reduce_pool=lambda formats: get_multidim_pools(formats),
    num_row_tiles=ROW_TILE_COUNTS,
    dest_acc=[DestAccumulation.Yes],
    input_bounds=lambda formats: get_multidim_input_bounds(formats),
)
def test_sfpu_reduce_multidim(
    formats,
    reduce_pool,
    num_row_tiles,
    dest_acc,
    input_bounds,
):
    # A column of num_row_tiles tiles, one column-tile wide: [num_row_tiles*32, 32].
    input_dimensions = [num_row_tiles * TILE_DIM, TILE_DIM]
    tile_cnt = num_row_tiles

    # REDUCE_ROW needs every tile resident in dest at once (single block). If the column of tiles
    # does not fit, the configuration is not expressible for this chained kernel.
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        input_dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )
    if num_blocks != 1:
        pytest.skip(
            reason="Row reduction requires all tiles in one dest section; "
            f"{tile_cnt} tiles do not fit for {formats.input_format.name}."
        )

    min_value, max_value = input_bounds
    torch_format = format_dict[formats.input_format]

    # STIMULI GENERATION
    stimuli_size = (tile_cnt * ELEMENTS_PER_TILE,)
    if formats.input_format.is_integer():
        src_A = torch.randint(
            low=min_value, high=max_value, size=stimuli_size, dtype=torch_format
        )
    else:
        src_A = torch.empty(stimuli_size, dtype=torch_format).uniform_(
            min_value, max_value
        )
    src_B = torch.zeros_like(src_A)

    # Tilize into dest layout (tiles stacked vertically, one column-tile wide).
    src_A = tilize_block(
        src_A, input_dimensions, stimuli_format=formats.input_format
    ).flatten()
    src_A_untilized = untilize_block(src_A, formats.input_format, input_dimensions)

    # Golden: reduce each 32x32 tile (32-row block) over both dims -> one value per tile.
    golden_blocks = src_A_untilized.reshape(num_row_tiles, TILE_DIM, TILE_DIM)
    golden_vec = torch.stack(
        [reduce_block(golden_blocks[r], reduce_pool) for r in range(num_row_tiles)]
    ).to(format_dict[formats.output_format])

    configuration = TestConfig(
        "sources/sfpu_reduce_multidim_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            APPROX_MODE(ApproximationMode.No),
            REDUCE_POOL_TYPE(reduce_pool),
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
            twos_complement=use_int32_twos_complement(formats),
        ),
        dest_acc=dest_acc,
        unpack_to_dest=True,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    res_from_L1 = configuration.run().result

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_matrix = untilize_block(res_tensor, formats.output_format, input_dimensions)

    # Each tile's multi-axis result lives at its element [0][0] -> row r*32, column 0.
    res_vec = res_matrix[0::TILE_DIM, 0]

    # The float SUM/AVG chains accumulate rounding error; size atol to it (PCC still guards
    # correctness). MAX/MIN and integer formats keep the default tolerances.
    reduce_atol = get_multidim_sum_avg_atol(
        formats.output_format, reduce_pool, input_bounds
    )

    assert passed_test(
        golden_vec, res_vec, formats.output_format, custom_atol=reduce_atol
    )
