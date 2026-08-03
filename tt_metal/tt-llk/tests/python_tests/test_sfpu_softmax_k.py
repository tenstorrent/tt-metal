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

ONLY EVEN k IS SWEPT. Odd k routes through `_zero_paired_odd_tail_lane_`, which
uses two identifiers (SFPCONFIG_TARGET_LREG11, SFPCONFIG_MOD_SET_LREG11) that are
declared nowhere in tt-metal or tt-llk; because they are plain identifiers inside
a template, the header does not compile at all until they exist. See the
"UPSTREAM BLOCKER" comment in sources/sfpu_softmax_k_test.cpp -- the test source
#defines them so the file parses, and this sweep stays on the even-k path where
that branch is dead code. Extend to odd k once the LLK declares them.

dest_acc=Yes is not swept: `_init_softmax_k_` hardcodes the bf16 exp setup and the
kernel's DEST addressing assumes the 16-bit layout.
"""

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import SOFTMAX_K, TILE_COUNT
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

FACE_DIM = 16
SOFTMAX_ROWS = 4  # one SFPU row band: DEST rows 0-3
MAX_ROW_BASE = 8  # the caller-supplied maxima live at DEST rows 8-11

# Only even k -- see module docstring.
EVEN_K = [2, 4, 6, 8, 10, 12, 14, 16]


def _build_input_tile(k: int, torch_format) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the [32, 32] input tile and return (tile, logits[SOFTMAX_ROWS, k]).

    Layout: rows 0-3 hold k non-zero logits followed by 16-k zero padding columns,
    rows 8-11 hold that row's maximum broadcast across all 16 columns, everything
    else is zero.
    """
    tile = torch.zeros((TILE_DIM, TILE_DIM), dtype=torch.float32)

    # Non-zero logits (0.0 is the padding marker, so keep magnitudes away from it).
    logits = torch.empty((SOFTMAX_ROWS, k), dtype=torch.float32).uniform_(0.5, 4.0)
    logits *= torch.where(
        torch.rand((SOFTMAX_ROWS, k)) < 0.5, -torch.ones(()), torch.ones(())
    )
    # Round to the storage format up front so the golden sees the same values the
    # device does.
    logits = logits.to(torch_format).to(torch.float32)

    for row in range(SOFTMAX_ROWS):
        tile[row, :k] = logits[row]
        tile[MAX_ROW_BASE + row, :FACE_DIM] = logits[row].max()

    return tile.to(torch_format), logits


def _softmax_golden(tile: torch.Tensor, logits: torch.Tensor, k: int) -> torch.Tensor:
    """Expected output tile: rows 0-3 replaced by the softmax, everything else as-is."""
    golden = tile.to(torch.float32).clone()
    for row in range(SOFTMAX_ROWS):
        m = logits[row].max()
        e = torch.exp(logits[row] - m)
        golden[row, :k] = e / e.sum()
        golden[row, k:FACE_DIM] = 0.0
    return golden


@parametrize(
    dest_acc=[DestAccumulation.No],
    k=EVEN_K,
)
def test_sfpu_softmax_k(dest_acc, k):
    torch.manual_seed(0)

    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    torch_format = format_dict[formats.input_format]

    input_tile, logits = _build_input_tile(k, torch_format)
    golden_tile = _softmax_golden(input_tile, logits, k)

    src_A = tilize_block(
        input_tile.flatten(), [TILE_DIM, TILE_DIM], stimuli_format=formats.input_format
    ).flatten()
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch_format)

    configuration = TestConfig(
        "sources/sfpu_softmax_k_test.cpp",
        formats,
        templates=[
            SOFTMAX_K(k=k),
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
        unpack_to_dest=False,
    )

    res_from_L1 = configuration.run().result
    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tile = untilize_block(
        res_tensor, formats.output_format, [TILE_DIM, TILE_DIM]
    ).reshape(TILE_DIM, TILE_DIM)

    # The interesting rows first, so a failure message points at the softmax rather
    # than at collateral damage elsewhere in the tile.
    assert passed_test(
        golden_tile[:SOFTMAX_ROWS, :FACE_DIM].flatten(),
        res_tile[:SOFTMAX_ROWS, :FACE_DIM].flatten(),
        formats.output_format,
        print_errors=True,
    ), f"softmax over k={k} columns does not match golden"

    # Each processed row must still sum to 1 over its k valid columns.
    for row in range(SOFTMAX_ROWS):
        row_sum = res_tile[row, :k].to(torch.float32).sum().item()
        assert (
            abs(row_sum - 1.0) < 5e-2
        ), f"row {row} softmax sums to {row_sum}, expected 1.0 (k={k})"

    # And the rest of the tile must be untouched -- in particular the maxima the
    # caller staged at rows 8-11, which the kernel only reads.
    assert passed_test(
        golden_tile[SOFTMAX_ROWS:, :].flatten(),
        res_tile[SOFTMAX_ROWS:, :].flatten(),
        formats.output_format,
    ), "softmax_k modified DEST rows outside its 4-row band"
