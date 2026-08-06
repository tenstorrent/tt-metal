# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# pack_untilize via RV_PACR (Quasar) — two modes in one test (see the CASES list).
#
# whole_tile mode: "normal" whole-tile untilize via one HW-streamed RV_PACR op per tile
#   (untilize=1). Proof-of-life for the RISC-V-descriptor pack path. A single 32x32 tile
#   must be byte-identical to the MOP/PACR_UNTILIZE golden.
#
# narrow mode: NARROW-ROW untilize — RV_PACR tile-mode, one op per DEST face-row. Produces
#   a TIGHT untilized output whose LAST tile per tile-row is a swept width in {8,16,24,32}.
#   This is the narrow_row capability the HW pack-untilize config stride cannot express.
#   The first NUM_TILES-1 tiles are packed full (32 wide).
#   8/16 widths of last tile use only col-group g=0 (faces 0,2), 24/32 also use g=1 (faces 1,3).
#   Each op writes a full 16-datum face-row. Non-face-aligned widths (8, 24) keep only the low datums
#   The spill that appears as a consequence of non-face-aligned widths gets overwritten by the next tile row.
#
# Golden (both modes) = the first matrix_w columns of each of the 32 untilized rows, packed
# tight and row-major (32 * matrix_w datums), where matrix_w = (num_tiles-1)*TILE_WIDTH + last_tile_width
# (whole_tile: num_tiles=1, last_tile_width=32 -> matrix_w=32 = full untilize).
# The device writes exactly that tight matrix_w-wide buffer. We read back and compare the first 32*matrix_w datums.

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    UntilizeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    PerfRunType,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.perf import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LAST_TILE_W_DATUMS,
    LOOP_FACTOR,
    NUM_FACES,
    PERF_RUN_TYPE,
    RV_WHOLE_TILE,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
    generate_input_dim,
)
from helpers.utils import passed_test

TILE_WIDTH = TILE_DIMENSIONS[1]
TILE_HEIGHT = TILE_DIMENSIONS[0]
# Kept widths of the narrow last tile to sweep
# (must match LAST_TILE_W_DATUMS in the kernel, the matrix width / L1 stride follows from it).
# 8/16 widths use only col-group g=0 (faces 0,2), 24/32 also use g=1 (faces 1,3).
# 8 and 24 are non-face-aligned widths whose boundary-face spill is overwritten by the next tile row.
LAST_TILE_WIDTHS = [8, 16, 24, 32]

# Number of tiles per tile-row for the narrow mode (last one narrow). Must match FULL_CT_DIM
# in the kernel; the input is TILE_HEIGHT x (num_tiles*TILE_WIDTH) so MATH produces that many
# tiles. whole_tile mode is single-tile (num_tiles=1).
NUM_TILES_NARROW_MODE = 4

# Test cases: (whole_tile, num_tiles, last_tile_width).
#   whole_tile=True  -> normal HW-streamed untilize, one 32x32 tile (last_tile_width=32).
#   whole_tile=False -> narrow per-face-row untilize, NUM_TILES tiles, swept last-tile width.
CASES = [(False, NUM_TILES_NARROW_MODE, w) for w in LAST_TILE_WIDTHS] + [
    (True, 1, TILE_WIDTH)
]

# 16-bit formats only: RV_PACR tile-mode l1_addr is 16B-aligned == 8 datums for
# 16-bit. Sub-16-bit formats cannot hit 8-datum granularity.
# 32-bit formats can hit 4-datum granularity, but the RV_PACR pack loop is not yet wired up for 32-bit formats.
NARROW_RV_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
    ],
)


@pytest.mark.quasar
@parametrize(
    formats=NARROW_RV_FORMATS,
    case=CASES,
)
def test_pack_untilize_narrow_rv_quasar(
    formats,
    case,
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    whole_tile, num_tiles, last_tile_width = case

    dest_acc = DestAccumulation.No
    dest_sync_mode = DestSync.Half
    input_dimensions = [TILE_HEIGHT, num_tiles * TILE_WIDTH]
    matrix_w = (
        num_tiles - 1
    ) * TILE_WIDTH + last_tile_width  # output row width (datums)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    num_faces = 4

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    # Shared config across the correctness (TestConfig) and perf (PerfConfig) paths.
    # LOOP_FACTOR repeats the steady-state unpack/math/pack work, it is 1 for correctness
    # (single untilized matrix, identical to the original behavior) and larger for perf.
    test_config_kwargs = {
        "test_name": "sources/quasar/pack_untilize_narrow_rv_quasar_test.cpp",
        "formats": formats,
        # Only true compile-time params are templates. Everything runtime-eligible is a runtime
        # arg so it does NOT trigger a recompile per value — critical for the width sweep
        # (LAST_TILE_W_DATUMS) and for sharing binaries between the functional (loop_factor=1)
        # and perf (loop_factor=32) runs. RV_WHOLE_TILE stays a template (if constexpr + the
        # single-tile static_assert), but it doesn't add compiles beyond FULL_CT_DIM (1 vs 4).
        "templates": [
            generate_input_dim(input_dimensions, input_dimensions),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            DEST_SYNC(dest_sync_mode),
            UNPACKER_ENGINE_SEL(),
            RV_WHOLE_TILE(whole_tile),
        ],
        "runtimes": [
            TEST_FACE_DIMS(),
            NUM_FACES(num_faces),
            TILE_COUNT(tile_cnt_A),
            LAST_TILE_W_DATUMS(last_tile_width),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
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
        "unpack_to_dest": False,
        "dest_acc": dest_acc,
    }

    if is_perf:
        PerfConfig(run_types=run_types, **test_config_kwargs).run(perf_report)
        return

    # Golden: untilize keeping the first matrix_w columns of each row (narrow_row_width).
    # For whole_tile (num_tiles=1, width=32) matrix_w=32, i.e. the full untilize.
    generate_golden = get_golden_generator(UntilizeGolden)
    narrow_golden = generate_golden(
        src_A,
        formats.output_format,
        input_dimensions,
        input_format=formats.input_format,
        narrow_row_width=matrix_w,
    ).to(format_dict[formats.output_format])
    narrow_len = TILE_HEIGHT * matrix_w

    configuration = TestConfig(
        **{
            **test_config_kwargs,
            "templates": test_config_kwargs["templates"]
            + [PERF_RUN_TYPE(PerfRunType.L1_TO_L1)],
        },
    )

    res_from_L1 = configuration.run().result

    res_full = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    assert (
        res_full.numel() >= narrow_len
    ), f"Result too short: {res_full.numel()} < {narrow_len}"

    # Device output is a tight matrix_w-wide row-major buffer (32 rows x matrix_w).
    res_narrow = res_full[:narrow_len]

    assert passed_test(
        narrow_golden,
        res_narrow,
        formats.output_format,
    ), "RV_PACR untilize output does not match golden"
