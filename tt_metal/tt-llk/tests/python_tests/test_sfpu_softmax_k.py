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

Padding contract: columns >= k must be exactly 0.0. The kernel derives a condition
code from |even-column value| before subtracting the max and only re-enables all
lanes (SFPENCC) after the exponential, so a 0.0 lane skips the subtract and the
exp and stays 0, then gets multiplied by the reciprocal (0 * r = 0). Valid inputs
must be non-zero, so the stimuli deliberately avoid 0.

Note that the golden here is independent of how SFPU lanes map onto DEST columns:
the sum runs over all 16 columns of the row and each lane is written back to the
column it came from, so both the interleaved (even/odd) and half/half readings
give the same expected tile.

Both even and odd k are swept, from k=1 (a single live column: its odd partner is the
tail lane) to k=16. Odd k is the interesting half: the tail column k is
odd and its even partner k-1 is a valid non-zero lane, so the padding lane inherits
an enabled condition code and would be exponentiated. `_zero_paired_odd_tail_lane_`
is what clears it, via an SFPCONFIG write targeting LREG11 masked to the single SFPU
instance holding lane k (1u << (k - 1)). That path is only reached for odd k < 16, so
without odd k in the sweep it ships untested.

dest_acc=Yes is swept for k=16 only. The kernel compiles for a 32-bit DEST and the
full-width case is correct there, but every k < 16 returns a softmax taken over all
16 lanes instead of k -- measured on Blackhole, e.g. k=2 comes back as
[0, 1/8] x 8 per row (row still sums to 1, but over 16 live lanes rather than 2).
The padding predication is what breaks: the condition code comes from the
even-column value and covers the even/odd column PAIR, which only holds while two
bf16 datums share one 32-bit DEST word. With a 32-bit DEST each datum owns a word,
the pair relationship is gone, and the lanes meant to be masked stay enabled.
Float32 input is skipped with dest_acc=No, as elsewhere in the SFPU suites.

The input population is a runtime axis (same ELF, different stimuli), so every
compiled variant is exercised on each of:

  uniform  |x| in [0.5, 4], random sign -- the original stimuli (seed 0).
  ties     every live logit of a row equal: exp(0) = 1 in every column and the
           result is exactly 1/k, so any rounding change in the row sum or the
           reciprocal shows up as a flipped output bit.
  wide     |x| in [0.5, 120]: most x - max land below the exp underflow knee, so
           rows mix O(1) exponentials with tiny and flushed ones.
  large    |x| in [1e30, 3e38]: x - max is either 0 or overflows to -inf.
  neg_inf  about a third of the live logits are -inf (the row max stays finite).
  nan      one live column of rows 0 and 2 is NaN, the staged max is the max of the
           other logits (k >= 2 only). The kernel has no NaN contract, so only the
           other rows are checked (they must be unaffected); what the NaN rows
           return depends on the exp flavour -- measured on Blackhole: the bf16
           exp turns NaN into +inf, so the row comes back +inf (NaN on a Float32
           output of the fp32 path) in the NaN column and 0 elsewhere; the
           accurate exp on a Float16_b input maps it to 0 and the rest of the row
           is a softmax of the others. The population is here so a result-buffer
           hash can pin that behaviour across kernel changes.

None of them contains an exact 0.0 in a live column (that is the padding marker).

dest_acc also selects the exp implementation -- `_sfpu_exp_21f_bf16_tti_` for No,
the `_ckernel_sfpu_exp_accurate_` sfpi loop for Yes -- and the DEST width the
kernel's three store/reload round-trips quantize to, which the golden models. Both
implementations are well inside the suite's 5% tolerance, so treat that axis as a
compile-and-structure check on two different kernels rather than as a numeric one.
"""

import math

import pytest
import torch
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    TILE_DIM,
    SoftmaxKGolden,
    get_golden_generator,
)
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize, runtime
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import SOFTMAX_K, TILE_COUNT
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

FORMATS = input_output_formats([DataFormat.Float16_b, DataFormat.Float32])

FACE_DIM = 16
SOFTMAX_ROWS = 4  # one SFPU row band: DEST rows 0-3
MAX_ROW_BASE = 8  # the caller-supplied maxima live at DEST rows 8-11

# Every supported k. Odd values exercise `_zero_paired_odd_tail_lane_` -- see docstring.
ALL_K = list(range(1, FACE_DIM + 1))

POPULATIONS = ["uniform", "ties", "wide", "large", "neg_inf", "nan"]
NAN_ROWS = (0, 2)  # rows of the "nan" population that carry a NaN logit


def _random_signs(shape) -> torch.Tensor:
    return torch.where(torch.rand(shape) < 0.5, -torch.ones(()), torch.ones(()))


def _build_logits(k: int, population: str) -> torch.Tensor:
    """Non-zero logits[SOFTMAX_ROWS, k] for *population* (see module docstring)."""
    shape = (SOFTMAX_ROWS, k)
    # The uniform population is drawn first and exactly as before, so its stimuli
    # (seed 0) are unchanged.
    logits = torch.empty(shape, dtype=torch.float32).uniform_(0.5, 4.0)
    logits *= _random_signs(shape)
    if population == "ties":
        logits = logits[:, :1].expand(shape).clone()
    elif population == "wide":
        logits = torch.empty(shape).uniform_(0.5, 120.0) * _random_signs(shape)
    elif population == "large":
        logits = torch.exp(
            torch.empty(shape).uniform_(math.log(1e30), math.log(3e38))
        ) * _random_signs(shape)
    elif population == "neg_inf":
        dropped = torch.rand(shape) < 1.0 / 3.0
        # Keep at least one finite logit per row so the staged max is finite.
        dropped[:, 0] = False
        logits[dropped] = -math.inf
    elif population == "nan":
        for row in NAN_ROWS:
            logits[row, k // 2] = math.nan
    return logits


def _build_input_tile(
    k: int, torch_format, population: str = "uniform"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the [32, 32] input tile and return (tile, logits[SOFTMAX_ROWS, k]).

    Layout: rows 0-3 hold k non-zero logits followed by 16-k zero padding columns,
    rows 8-11 hold that row's maximum (ignoring NaN) broadcast across all 16
    columns, everything else is zero.
    """
    tile = torch.zeros((TILE_DIM, TILE_DIM), dtype=torch.float32)

    # Non-zero logits (0.0 is the padding marker, so keep magnitudes away from it).
    # Round to the storage format up front so the golden sees the same values the
    # device does.
    logits = _build_logits(k, population).to(torch_format).to(torch.float32)

    for row in range(SOFTMAX_ROWS):
        tile[row, :k] = logits[row]
        finite = logits[row][~torch.isnan(logits[row])]
        tile[MAX_ROW_BASE + row, :FACE_DIM] = finite.max()

    return tile.to(torch_format), logits


@skip_for_wormhole
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    k=ALL_K,
    population=runtime(POPULATIONS),
)
def test_sfpu_softmax_k(formats, dest_acc, k, population):
    if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 inputs with dest_acc=No are not supported")

    if dest_acc == DestAccumulation.Yes and k < FACE_DIM:
        # The padding predication only works on a 16-bit DEST -- see module docstring.
        pytest.skip("k < 16 with a 32-bit dest exponentiates the padding lanes")

    if population == "nan" and k == 1:
        pytest.skip("a NaN logit in the only live column leaves no row max to stage")

    torch.manual_seed(0)

    torch_format = format_dict[formats.input_format]

    input_tile, logits = _build_input_tile(k, torch_format, population)

    golden_generator = get_golden_generator(SoftmaxKGolden)
    golden_tile = golden_generator(
        input_tile, logits, k, dest_acc, SOFTMAX_ROWS, FACE_DIM
    )
    # Rows whose golden is defined (the NaN rows have none -- see module docstring).
    nan_rows = list(NAN_ROWS) if population == "nan" else []
    rows = [row for row in range(SOFTMAX_ROWS) if row not in nan_rows]

    src_A = tilize_block(
        input_tile.flatten(), [TILE_DIM, TILE_DIM], stimuli_format=formats.input_format
    ).flatten()
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch_format)

    configuration = TestConfig(
        "sources/sfpu_softmax_k_test.cpp",
        formats,
        templates=[
            SOFTMAX_K(softmax_k=k),
        ],
        runtimes=[
            TILE_COUNT(1),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=dest_acc,
        # A 32-bit input goes straight to DEST rather than through srcA.
        unpack_to_dest=formats.input_format.is_32_bit(),
    )

    res_from_L1 = configuration.run().result
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tile = untilize_block(
        res_tensor, formats.output_format, [TILE_DIM, TILE_DIM]
    ).reshape(TILE_DIM, TILE_DIM)

    # The interesting rows first, so a failure message points at the softmax rather
    # than at collateral damage elsewhere in the tile.
    assert passed_test(
        golden_tile[rows, :FACE_DIM].flatten(),
        res_tile[rows, :FACE_DIM].flatten(),
        formats.output_format,
        print_errors=True,
    ), f"softmax over k={k} columns does not match golden ({population})"

    # The padding columns, checked exactly rather than through the 5% tolerance above.
    # The whole point of odd k is `_zero_paired_odd_tail_lane_`, and a leaked
    # exp(-m) / sum term is at most ~0.05 whenever the row max is >= ~2.95 -- i.e. the
    # common case for these stimuli -- so the tolerance check above turns detection of
    # a wrong `1u << (k - 1)` mask into a fixed-seed coin flip.
    if k < FACE_DIM:
        padding = res_tile[rows, k:FACE_DIM]
        assert bool((padding.to(torch.float32) == 0.0).all()), (
            f"columns {k}-{FACE_DIM - 1} are padding and must come back exactly 0.0 "
            f"(k={k}):\n{padding}"
        )

    # Each processed row must still sum to 1 over its k valid columns.
    for row in rows:
        row_sum = res_tile[row, :k].to(torch.float32).sum().item()
        assert (
            abs(row_sum - 1.0) < 5e-2
        ), f"row {row} softmax sums to {row_sum}, expected 1.0 (k={k})"

    # Face 1 of the processed rows -- DEST rows 16-19, which the kernel must not touch.
    # This is where an extra vector-mode pass would land: with VectorMode::RC instead of
    # RC_custom the SFPU dispatch re-bases DEST by +16 rows and runs the kernel a second
    # time over this region. Choosing RC_custom is the one non-boilerplate decision in
    # the driver, so it gets its own assertion.
    assert passed_test(
        golden_tile[:SOFTMAX_ROWS, FACE_DIM:].flatten(),
        res_tile[:SOFTMAX_ROWS, FACE_DIM:].flatten(),
        formats.output_format,
    ), "softmax_k modified face 1 (columns 16-31) of its own row band"

    # And the rest of the tile must be untouched -- in particular the maxima the
    # caller staged at rows 8-11, which the kernel only reads.
    assert passed_test(
        golden_tile[SOFTMAX_ROWS:, :].flatten(),
        res_tile[SOFTMAX_ROWS:, :].flatten(),
        formats.output_format,
    ), "softmax_k modified DEST rows outside its 4-row band"
