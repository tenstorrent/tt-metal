# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
softmax_k SFPU test.

Covers tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_softmax_k.h
(`_init_softmax_k_` / `_softmax_k_<k>`).

`_softmax_k_<k>` is a per-row softmax over the first k of the 16 columns of face 0's
first row band, with the row maximum supplied by the caller instead of being reduced
on the fly:

  DEST rows 0-3,  columns 0-15 : x        (in, overwritten with the result)
  DEST rows 8-11, columns 0-15 : max(x)   (in, broadcast across the row)

  out[r][c] = exp(x[r][c] - m[r]) / sum_{c' < k} exp(x[r][c'] - m[r])   c < k
  out[r][c] = 0                                                          c >= k

Only 4 rows are processed (one SFPU row band); the rest of the tile is left alone.

Columns >= k are padding: their input may hold anything and their output must be
exactly 0.0. The main sweep fills the padding with random logits drawn like the valid
ones; `test_sfpu_softmax_k_garbage_padding` uses 1e30 / inf / NaN.

Every k in 1..16 is swept on both DEST widths. Float32 input is skipped with
dest_acc=No, as elsewhere in the SFPU suites.

dest_acc also selects the exp implementation -- `_sfpu_exp_21f_bf16_tti_` for No,
`_sfpu_exp_fp32_accurate_tti_` for Yes (the sfpi loop only under DISABLE_SFPLOADMACRO)
-- and the DEST width the kernel's three store/reload round-trips quantize to, which
the golden models.
"""

import torch
from conftest import blackhole_only
from helpers.constraints import skip_if_32bit_input_without_dest_acc
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    TILE_DIM,
    SoftmaxKGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, DestSync, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.blaze import execute_llk
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    LOOP_FACTOR,
    SOFTMAX_K,
    TILE_COUNT,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test, result_tensor

FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32])
DEST_SYNCS = [DestSync.Half]

FACE_DIM = 16
SOFTMAX_ROWS = 4  # one SFPU row band: DEST rows 0-3
MAX_ROW_BASE = 8  # the caller-supplied maxima live at DEST rows 8-11

# Every supported k (1..16).
ALL_K = list(range(1, FACE_DIM + 1))

# What the kernel must ignore in the padding columns: a large finite value, +inf and
# NaN, cycling across the padding columns of every processed row.
GARBAGE_PADDING = (1.0e30, float("inf"), float("nan"))


def _build_input_tile(
    k: int, torch_format, *, zero_logit: bool = False, garbage_padding: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the [32, 32] input tile and return (tile, logits[SOFTMAX_ROWS, k]).

    Layout: rows 0-3 hold k logits followed by 16-k padding columns, rows 8-11 hold
    that row's maximum (over the k valid logits only) broadcast across all 16 columns,
    everything else is zero.

    The padding holds random values drawn like the logits (and often above the row
    maximum), or GARBAGE_PADDING with `garbage_padding`. `zero_logit` (k >= 2) plants
    a valid logit of exactly 0.0 in column 0 and makes column 1 positive so the 0.0 is
    never the row maximum.
    """
    tile = torch.zeros((TILE_DIM, TILE_DIM), dtype=torch.float32)

    def draw(shape):
        values = torch.empty(shape, dtype=torch.float32).uniform_(0.5, 4.0)
        return values * torch.where(
            torch.rand(shape) < 0.5, -torch.ones(()), torch.ones(())
        )

    logits = draw((SOFTMAX_ROWS, k))
    if zero_logit:
        logits[:, 1] = logits[:, 1].abs()
        logits[:, 0] = 0.0
    # Round to the storage format up front so the golden sees the same values the
    # device does.
    logits = logits.to(torch_format).to(torch.float32)
    padding = draw((SOFTMAX_ROWS, FACE_DIM - k))

    for row in range(SOFTMAX_ROWS):
        tile[row, :k] = logits[row]
        tile[MAX_ROW_BASE + row, :FACE_DIM] = logits[row].max()
        for column in range(k, FACE_DIM):
            tile[row, column] = (
                GARBAGE_PADDING[column % len(GARBAGE_PADDING)]
                if garbage_padding
                else padding[row, column - k]
            )

    return tile.to(torch_format), logits


def _run_softmax_k(
    formats,
    dest_acc,
    k,
    *,
    dest_sync=DestSync.Half,
    zero_logit: bool = False,
    garbage_padding: bool = False,
    tile_count: int = 1,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    """Run `tile_count` softmax_k tiles in one program and check every tile region.

    With `is_perf` the same program is handed to the perf harness instead and nothing
    is checked (perf_sfpu_softmax_k.py drives this path).
    """
    torch.manual_seed(0)

    torch_format = format_dict[formats.input_format]
    golden_generator = get_golden_generator(SoftmaxKGolden)

    input_tiles, golden_tiles = [], []
    for _ in range(tile_count):
        input_tile, logits = _build_input_tile(
            k, torch_format, zero_logit=zero_logit, garbage_padding=garbage_padding
        )
        input_tiles.append(input_tile)
        golden_tiles.append(
            golden_generator(input_tile, logits, k, dest_acc, SOFTMAX_ROWS)
        )

    src_A = torch.cat(
        [
            tilize_block(
                tile.flatten(),
                [TILE_DIM, TILE_DIM],
                stimuli_format=formats.input_format,
            ).flatten()
            for tile in input_tiles
        ]
    )
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch_format)

    outcome = execute_llk(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
        test_config_kwargs=dict(
            test_name="sources/sfpu_softmax_k_test.cpp",
            formats=formats,
            templates=[
                SOFTMAX_K(softmax_k=k),
                DEST_SYNC(dest_sync),
            ],
            runtimes=[
                LOOP_FACTOR(1),
                TILE_COUNT(tile_count),
            ],
            variant_stimuli=StimuliConfig(
                src_A,
                formats.input_format,
                src_B,
                formats.input_format,
                formats.output_format,
                tile_count_A=tile_count,
                tile_count_B=1,
                tile_count_res=tile_count,
            ),
            dest_acc=dest_acc,
            # A 32-bit input goes straight to DEST rather than through srcA.
            unpack_to_dest=formats.input_format.is_32_bit(),
        ),
    )
    if is_perf:
        return

    res_tensor = result_tensor(outcome.result, formats.output_format)
    for tile, golden_tile in enumerate(golden_tiles):
        res_tile = untilize_block(
            res_tensor[tile * ELEMENTS_PER_TILE : (tile + 1) * ELEMENTS_PER_TILE],
            formats.output_format,
            [TILE_DIM, TILE_DIM],
        ).reshape(TILE_DIM, TILE_DIM)
        _check_softmax_k_tile(golden_tile, res_tile, formats, k, tile)


def _check_softmax_k_tile(golden_tile, res_tile, formats, k, tile):
    # The interesting values first, so a failure message points at the softmax rather
    # than at collateral damage elsewhere in the tile.
    assert passed_test(
        golden_tile[:SOFTMAX_ROWS, :k].flatten(),
        res_tile[:SOFTMAX_ROWS, :k].flatten(),
        formats.output_format,
        print_errors=True,
    ), f"softmax over k={k} columns does not match golden (tile {tile})"

    # The padding columns must come back exactly 0.0 whatever their input held. Checked
    # exactly: a leaked exp(x - m) / sum is often below the tolerance above.
    if k < FACE_DIM:
        padding = res_tile[:SOFTMAX_ROWS, k:FACE_DIM].to(torch.float32)
        assert bool((padding == 0.0).all()), (
            f"columns {k}-{FACE_DIM - 1} are padding and must come back exactly 0.0 "
            f"(k={k}, tile {tile}):\n{padding}"
        )

    # Each processed row must sum to 1 over its k valid columns. This is the direct
    # check that no padding lane leaked into the row sum.
    for row in range(SOFTMAX_ROWS):
        row_sum = res_tile[row, :k].to(torch.float32).sum().item()
        assert (
            abs(row_sum - 1.0) < 5e-2
        ), f"row {row} softmax sums to {row_sum}, expected 1.0 (k={k}, tile {tile})"

    # Face 1 of the processed rows -- DEST rows 16-19, which the kernel must not touch.
    # This is where an extra vector-mode pass would land: with VectorMode::RC instead of
    # RC_custom the SFPU dispatch re-bases DEST by +16 rows and runs the kernel a second
    # time over this region. Choosing RC_custom is the one non-boilerplate decision in
    # the driver, so it gets its own assertion.
    assert passed_test(
        golden_tile[:SOFTMAX_ROWS, FACE_DIM:].flatten(),
        res_tile[:SOFTMAX_ROWS, FACE_DIM:].flatten(),
        formats.output_format,
    ), f"softmax_k modified face 1 (columns 16-31) of its own row band (tile {tile})"

    # And the rest of the tile must be untouched -- in particular the maxima the
    # caller staged at rows 8-11, which the kernel only reads.
    assert passed_test(
        golden_tile[SOFTMAX_ROWS:, :].flatten(),
        res_tile[SOFTMAX_ROWS:, :].flatten(),
        formats.output_format,
    ), f"softmax_k modified DEST rows outside its 4-row band (tile {tile})"


# The softmax_k LLKs exist only in the Blackhole tree, so the driver does not compile
# elsewhere; skip rather than fail on other architectures.
@blackhole_only
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    dest_sync=DEST_SYNCS,
    k=ALL_K,
)
def test_sfpu_softmax_k(
    formats,
    dest_acc,
    dest_sync,
    k,
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    _run_softmax_k(
        formats,
        dest_acc,
        k,
        dest_sync=dest_sync,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@blackhole_only
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    k=[2, 8],
)
def test_sfpu_softmax_k_zero_logit(formats, dest_acc, k):
    """A valid logit of exactly 0.0 in an even column (column 0).

    0.0 is an ordinary input on both DEST widths and must come back as exp(-m) / sum.
    Column 1 is forced positive so the 0.0 is never the row maximum.
    """
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    _run_softmax_k(formats, dest_acc, k, zero_logit=True)


@blackhole_only
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    k=[1, 2, 8, 15],
)
def test_sfpu_softmax_k_garbage_padding(formats, dest_acc, k):
    """Padding columns holding 1e30 / +inf / NaN.

    The valid columns must be unaffected. k = 15 has padding only in the odd block
    (column 15); k = 1 has the widest padding.
    """
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    _run_softmax_k(formats, dest_acc, k, garbage_padding=True)


@blackhole_only
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    k=[7, 8],
)
def test_sfpu_softmax_k_two_tiles(formats, dest_acc, k):
    """Two tiles in one program: the second call must not depend on any SFPU state
    (lane flags, LREG contents, the LCONST_neg1 the max-subtract reads) the first call
    left behind.
    """
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    _run_softmax_k(formats, dest_acc, k, tile_count=2)
