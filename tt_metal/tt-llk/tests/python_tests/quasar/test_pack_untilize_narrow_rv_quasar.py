# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# NARROW-ROW untilize via RV_PACR (Quasar) DEMO test — MULTI-TILE.
#
# Validates that RV_PACR tile-mode-per-row produces a TIGHT untilized output with
# a variable-width last tile per tile-row — the narrow_row capability the
# pack-untilize config stride cannot express. Scope: one tile-row of NUM_TILES
# 32x32 tiles, 16-bit formats. The first NUM_TILES-1 tiles are packed full (32
# wide); the last tile is packed at a swept width in {8, 16, 24, 32}: 8/16 use only
# col-group g=0 (faces 0,2), 24/32 also use g=1 (faces 1,3), and 32 == normal
# untilize. Each op always writes a full 16-datum face-row; non-face-aligned widths
# (8, 24) keep only the low datums, the spill being overwritten by the next tile row.
#
# Golden = the first matrix_w columns of each of the 32 untilized rows, packed
# tight and row-major (32 * matrix_w datums), where
#   matrix_w = (NUM_TILES-1)*TILE_WIDTH + last_tile_width.
# The device writes exactly that tight matrix_w-wide buffer; we read it back and
# compare the first 32*matrix_w datums.

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import UntilizeGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LAST_TILE_W_DATUMS,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
    generate_input_dim,
)
from helpers.utils import passed_test


def print_full_tile(data, data_format, label="result", tile_width=32):
    """Dump a full buffer as tile_width-wide rows (same layout for result and golden),
    so the device output and the golden can be compared line-by-line."""
    t = (
        data
        if isinstance(data, torch.Tensor)
        else torch.tensor(data, dtype=format_dict[data_format])
    )
    flat = t.flatten()
    n = flat.numel()
    nz = (flat != 0).nonzero().flatten().tolist()
    print(f"\n[RV-narrow] {label} len={n}  nonzero_count={len(nz)}")
    if nz:
        print(f"[RV-narrow] {label} first 64 nonzero indices: {nz[:64]}")
        print(f"[RV-narrow] {label} nonzero index range: [{nz[0]} .. {nz[-1]}]")
    if n % tile_width == 0:
        rows = flat.reshape(-1, tile_width)
        for r in range(rows.shape[0]):
            vals = " ".join(f"{v:5.2f}" for v in rows[r].tolist())
            print(f"[RV-narrow] {label} row {r:2d}: {vals}")
    else:
        print(f"[RV-narrow] {label} flat: {flat.tolist()}")


# Must match the C++ kernel. The last tile in each row is packed narrow via skip-face-1
# (faces 0,2 => cols 0-15). The op always writes a full 16-datum face-row, but the kept
# width is LAST_TILE_WIDTH (16 or 8) -- for 8 the upper half is spill overwritten by the
# next row / tile 0, so only the low 8 columns of the last tile are meaningful.
TILE_WIDTH = 32
TILE_HEIGHT = 32
# Kept widths of the last tile to sweep (must match the LAST_TILE_W_DATUMS the kernel is
# built with; the matrix width / L1 stride follows from it). 8/16 use only col-group g=0
# (faces 0,2); 24/32 also use g=1 (faces 1,3). 32 == normal (full) untilize; 8 and 24 are
# non-face-aligned widths whose boundary-face spill is overwritten by the next tile row.
LAST_TILE_WIDTHS = [8, 16, 24, 32]
# Number of tiles per tile-row (last one narrow). Must match FULL_CT_DIM in the kernel;
# set the input to TILE_HEIGHT x (NUM_TILES*TILE_WIDTH) so MATH produces that many tiles.
NUM_TILES = 4
INPUT_DIMS = [TILE_HEIGHT, NUM_TILES * TILE_WIDTH]

# 16-bit formats only: RV_PACR tile-mode l1_addr is 16B-aligned == 8 datums for
# 16-bit. Sub-16-bit formats cannot hit 8-datum granularity.
NARROW_RV_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
    ],
)


@pytest.mark.quasar
@parametrize(formats=NARROW_RV_FORMATS, last_tile_width=LAST_TILE_WIDTHS)
def test_pack_untilize_narrow_rv_quasar(formats, last_tile_width):
    dest_acc = DestAccumulation.No
    dest_sync_mode = DestSync.Half
    input_dimensions = INPUT_DIMS
    matrix_w = (
        NUM_TILES - 1
    ) * TILE_WIDTH + last_tile_width  # output row width (datums)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Full untilize of the NUM_TILES-wide input (32 rows x NUM_TILES*32 cols, row-major).
    # The device output is the same row-major layout but the LAST tile keeps only its
    # first LAST_TILE_WIDTH columns -> output row width = matrix_w. So the golden is the
    # first matrix_w columns of each untilized row.
    generate_golden = get_golden_generator(UntilizeGolden)
    full_untilized = generate_golden(
        src_A,
        formats.output_format,
        input_dimensions,
        input_format=formats.input_format,
    )
    full_w = NUM_TILES * TILE_WIDTH
    narrow_golden = (
        full_untilized.reshape(TILE_HEIGHT, full_w)[:, :matrix_w]
        .flatten()
        .to(format_dict[formats.output_format])
    )
    narrow_len = TILE_HEIGHT * matrix_w

    num_faces = 4
    configuration = TestConfig(
        "sources/quasar/pack_untilize_narrow_rv_quasar_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            DEST_SYNC(dest_sync_mode),
            UNPACKER_ENGINE_SEL(),
            TEST_FACE_DIMS(),
            NUM_FACES(num_faces),
            TILE_COUNT(tile_cnt_A),
            LAST_TILE_W_DATUMS(last_tile_width),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    res_full = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert (
        res_full.numel() >= narrow_len
    ), f"Result too short: {res_full.numel()} < {narrow_len}"

    # Device output is a tight matrix_w-wide row-major buffer (32 rows x matrix_w).
    res_narrow = res_full[:narrow_len]

    # Raw full-buffer dump (matrix_w-wide rows) — shows per-tile column placement and
    # any spill past matrix_w, so a mis-strided tile is easy to spot when NUM_TILES>1.
    print_full_tile(
        res_full, formats.output_format, label="result-raw", tile_width=matrix_w
    )

    # Dump golden vs result as matrix_w-wide rows for line-by-line comparison.
    print_full_tile(
        narrow_golden, formats.output_format, label="golden", tile_width=matrix_w
    )
    print_full_tile(
        res_narrow, formats.output_format, label="result", tile_width=matrix_w
    )

    assert passed_test(
        narrow_golden, res_narrow, formats.output_format, print_errors=True
    ), "Narrow RV_PACR multi-tile untilize output does not match golden"
