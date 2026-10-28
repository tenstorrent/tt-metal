# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
softmax_k SFPU test.

Covers tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_softmax_k.h
(`_init_softmax_k_` / `_softmax_k_<k>`).

`_softmax_k_<k>` is a per-row softmax over the 16 columns of face 0's first row
band, with the row maximum supplied by the caller instead of being reduced on the
fly:

  DEST rows 0-3,  columns 0-15 : x        (in, overwritten with the result)
  DEST rows 8-11, columns 0-15 : max(x)   (in, broadcast across the row)

  out[r][c] = exp(x[r][c] - m[r]) / sum_{c' < k} exp(x[r][c'] - m[r])   c < k
  out[r][c] = 0                                                          c >= k

Only 4 rows are processed (one SFPU row band); the rest of the tile is left alone.

Padding contract on a 16-bit DEST: columns >= k must be exactly 0.0 and valid inputs
must be non-zero. The kernel derives a condition code from |even-column value| before
subtracting the max and only re-enables all lanes (SFPENCC) after the exponential, so
a 0.0 lane skips the subtract and the exp and stays 0, then gets multiplied by the
reciprocal (0 * r = 0). On a 32-bit DEST no condition code is set: every column goes
through subtract + exp and the columns >= k are re-zeroed by index afterwards, so
neither requirement applies there (`test_sfpu_softmax_k_zero_logit` feeds a valid
0.0, `test_sfpu_softmax_k_garbage_padding` fills the padding with 1e30 / inf / NaN).
The main sweep's stimuli avoid 0 so one input tile serves both widths.

Note that the golden here is independent of how SFPU lanes map onto DEST columns:
the sum runs over all 16 columns of the row and each lane is written back to the
column it came from, so both the interleaved (even/odd) and half/half readings
give the same expected tile.

Every k in 1..16 is swept (k = 1 is a softmax over a single column: 1.0 there, zeros
elsewhere). On a 16-bit DEST odd k is the interesting half: the tail column k is odd
and its even partner k-1 is a valid non-zero lane, so the padding lane inherits an
enabled condition code and would be exponentiated. `_zero_padding_columns_` is what
clears it, via an SFPCONFIG write targeting LREG11 masked to the single SFPU instance
holding lane k (1u << (k - 1)). On that width the path is only reached for odd k < 16,
so without odd k in the sweep it ships untested.

dest_acc=Yes is swept for every k as well. The lane<->column pairing does not depend
on the DEST width (SFPLOAD.md: lane L reads row L/8, column 2*(L%8) for address 0 and
2*(L%8)+1 for address 2, for Dst16b and Dst32b alike), but the 32-bit path sets no
lane predicate at all: its exponential is sfpi code whose top-level v_endif re-enables
every lane (SFPENCC) before the store, so a hand-set condition code could not survive
`calculate_exponential` (the shipped-code symptom, before `_zero_padding_columns_`
existed: k=2 came back as [0, 1/8] x 8 per row, the odd padding lanes exponentiated
and summed). Instead every padding column of both blocks goes through subtract + exp,
leaving exp(-max), and `_zero_padding_columns_` re-zeroes the columns >= k by
SFPU-instance index, so every k < 16 must pass here.
Float32 input is skipped with dest_acc=No, as elsewhere in the SFPU suites.

dest_acc also selects the exp implementation -- `_sfpu_exp_21f_bf16_tti_` for No,
the `_ckernel_sfpu_exp_accurate_` sfpi loop for Yes -- and the DEST width the
kernel's three store/reload round-trips quantize to, which the golden models. Both
implementations are well inside the suite's 5% tolerance, so treat that axis as a
compile-and-structure check on two different kernels rather than as a numeric one.

`_zero_padding_columns_` programs LREG11 -- the core-wide -1.0 constant
(`p_sfpu::LCONST_neg1`) the max-subtract SFPMADs read -- and puts -1.0 back with
`_restore_lconst_neg1_()` before returning. A single tile cannot see a missing or
wrong restore (the clobber happens after the subtract); `test_sfpu_softmax_k_two_tiles`
runs the kernel twice in one program so the second tile's subtract reads whatever the
first call left: with the restore gone, instances the clear marked with 1.0 compute
x + max instead of x - max (odd k: the valid column k-1 blows up on both widths; the
rest get x, which the softmax is invariant to).
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
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import SOFTMAX_K, TILE_COUNT
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32])

FACE_DIM = 16
SOFTMAX_ROWS = 4  # one SFPU row band: DEST rows 0-3
MAX_ROW_BASE = 8  # the caller-supplied maxima live at DEST rows 8-11

# Every supported k (1..16). Odd values exercise the odd-tail clearing -- see docstring.
ALL_K = list(range(1, FACE_DIM + 1))

# What the 32-bit path must ignore in the padding columns: a large finite value, +inf
# and NaN, cycling across the padding columns of every processed row.
GARBAGE_PADDING = (1.0e30, float("inf"), float("nan"))


def _build_input_tile(
    k: int, torch_format, *, zero_logit: bool = False, garbage_padding: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the [32, 32] input tile and return (tile, logits[SOFTMAX_ROWS, k]).

    Layout: rows 0-3 hold k non-zero logits followed by 16-k zero padding columns,
    rows 8-11 hold that row's maximum broadcast across all 16 columns, everything
    else is zero.

    `zero_logit` (32-bit DEST only, k >= 2) plants a valid logit of exactly 0.0 in
    column 0 and makes column 1 positive so the 0.0 is never the row maximum.
    `garbage_padding` (32-bit DEST only) fills the padding columns with GARBAGE_PADDING
    instead of 0.0; the row maxima still come from the k valid logits only.
    """
    tile = torch.zeros((TILE_DIM, TILE_DIM), dtype=torch.float32)

    # Non-zero logits (0.0 is the padding marker, so keep magnitudes away from it).
    logits = torch.empty((SOFTMAX_ROWS, k), dtype=torch.float32).uniform_(0.5, 4.0)
    logits *= torch.where(
        torch.rand((SOFTMAX_ROWS, k)) < 0.5, -torch.ones(()), torch.ones(())
    )
    if zero_logit:
        logits[:, 1] = logits[:, 1].abs()
        logits[:, 0] = 0.0
    # Round to the storage format up front so the golden sees the same values the
    # device does.
    logits = logits.to(torch_format).to(torch.float32)

    for row in range(SOFTMAX_ROWS):
        tile[row, :k] = logits[row]
        tile[MAX_ROW_BASE + row, :FACE_DIM] = logits[row].max()
        if garbage_padding:
            for column in range(k, FACE_DIM):
                tile[row, column] = GARBAGE_PADDING[column % len(GARBAGE_PADDING)]

    return tile.to(torch_format), logits


def _run_softmax_k(
    formats,
    dest_acc,
    k,
    *,
    zero_logit: bool = False,
    garbage_padding: bool = False,
    tile_count: int = 1,
):
    """Run `tile_count` softmax_k tiles in one program and check every tile region."""
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
            golden_generator(input_tile, logits, k, dest_acc, SOFTMAX_ROWS, FACE_DIM)
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

    configuration = TestConfig(
        "sources/sfpu_softmax_k_test.cpp",
        formats,
        templates=[
            SOFTMAX_K(softmax_k=k),
        ],
        runtimes=[
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
    )

    res_from_L1 = configuration.run().result
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    for tile, golden_tile in enumerate(golden_tiles):
        res_tile = untilize_block(
            res_tensor[tile * ELEMENTS_PER_TILE : (tile + 1) * ELEMENTS_PER_TILE],
            formats.output_format,
            [TILE_DIM, TILE_DIM],
        ).reshape(TILE_DIM, TILE_DIM)
        _check_softmax_k_tile(golden_tile, res_tile, formats, k, tile)


def _check_softmax_k_tile(golden_tile, res_tile, formats, k, tile):
    # The interesting rows first, so a failure message points at the softmax rather
    # than at collateral damage elsewhere in the tile.
    assert passed_test(
        golden_tile[:SOFTMAX_ROWS, :FACE_DIM].flatten(),
        res_tile[:SOFTMAX_ROWS, :FACE_DIM].flatten(),
        formats.output_format,
        print_errors=True,
    ), f"softmax over k={k} columns does not match golden (tile {tile})"

    # The padding columns, checked exactly rather than through the 5% tolerance above.
    # The whole point of odd k is `_zero_padding_columns_`, and a leaked
    # exp(-m) / sum term is at most ~0.05 whenever the row max is >= ~2.95 -- i.e. the
    # common case for these stimuli -- so the tolerance check above turns detection of
    # a wrong `1u << (k - 1)` mask into a fixed-seed coin flip. On a 32-bit DEST this
    # exact check is also what guards the per-instance clearing for every k < 16 (a leak
    # there is the same exp(-m) / sum term, in any padding column).
    if k < FACE_DIM:
        padding = res_tile[:SOFTMAX_ROWS, k:FACE_DIM]
        assert bool((padding.to(torch.float32) == 0.0).all()), (
            f"columns {k}-{FACE_DIM - 1} are padding and must come back exactly 0.0 "
            f"(k={k}, tile {tile}):\n{padding}"
        )

    # Each processed row must still sum to 1 over its k valid columns.
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
    k=ALL_K,
)
def test_sfpu_softmax_k(formats, dest_acc, k):
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    _run_softmax_k(formats, dest_acc, k)


@blackhole_only
@parametrize(
    formats=FORMATS,
    k=[2, 8],
)
def test_sfpu_softmax_k_zero_logit(formats, k):
    """A valid logit of exactly 0.0 in an even column (column 0) on a 32-bit DEST.

    The 32-bit path sets no |even column| > 0 lane predicate, so a 0.0 logit is an
    ordinary input there and must come back as exp(-m) / sum, not 0. It would fail if
    the 16-bit predicate were applied on this width: SFPU instance 0 (columns 0 and 1)
    would skip the max-subtract while the sfpi exp still exponentiated it unshifted.
    Column 1 is forced positive so the 0.0 is never the row maximum.
    """
    _run_softmax_k(formats, DestAccumulation.Yes, k, zero_logit=True)


@blackhole_only
@parametrize(
    formats=FORMATS,
    k=[2, 8, 15],
)
def test_sfpu_softmax_k_garbage_padding(formats, k):
    """Padding columns holding 1e30 / +inf / NaN on a 32-bit DEST.

    The 32-bit path clears the columns >= k by SFPU-instance index after the exp, so
    their input value is irrelevant: whatever subtract + exp made of the garbage
    (inf, NaN) is overwritten with an exact 0.0 and never reaches the row sum. k = 15
    pins the odd-block-only clear (instance 7, column 15). The exact-0.0 padding check
    in `_check_softmax_k_tile` is what pins the clear itself.
    """
    _run_softmax_k(formats, DestAccumulation.Yes, k, garbage_padding=True)


@blackhole_only
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    k=[7, 8],
)
def test_sfpu_softmax_k_two_tiles(formats, dest_acc, k):
    """Two tiles in one program: the second call's max-subtract reads the LREG11 the
    first call's `_zero_padding_columns_` left behind, so `_restore_lconst_neg1_()` is
    under test. k = 7 detects a missing restore on both widths (the instance holding the
    valid column 6 and the padding column 7 is marked 1.0, so column 6 gets x + max on
    the second tile); k = 8 on the 32-bit DEST only marks padding instances, whose x + max
    is cleared anyway, and the remaining x (no subtract) leaves the softmax unchanged --
    it is the control that the two-tile flow itself is sound.
    """
    skip_if_32bit_input_without_dest_acc(formats, dest_acc)

    _run_softmax_k(formats, dest_acc, k, tile_count=2)
