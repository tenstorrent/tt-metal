# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# NARROW-ROW untilize via RV_PACR (Quasar) DEMO test.
#
# Validates that RV_PACR tile-mode-per-row produces a TIGHT ROW_NUM_DATUMS-wide
# untilized output — the narrow_row capability the pack-untilize config stride
# cannot express. Scope: single 32x32 tile, 16-bit formats, ROW_NUM_DATUMS=8.
#
# Golden = the first ROW_NUM_DATUMS columns of each of the 32 untilized rows,
# packed tight (32 * ROW_NUM_DATUMS datums). The device writes that tight region
# to the front of a full-tile result buffer; we read the full tile back and
# compare only the first 32*ROW_NUM_DATUMS datums.

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


# Must match ROW_NUM_DATUMS in the C++ kernel.
ROW_NUM_DATUMS = 8
TILE_WIDTH = 32
TILE_HEIGHT = 32
SINGLE_TILE_DIMS = [TILE_HEIGHT, TILE_WIDTH]

# 16-bit formats only: RV_PACR tile-mode l1_addr is 16B-aligned == 8 datums for
# 16-bit. Sub-16-bit formats cannot hit 8-datum granularity.
NARROW_RV_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
    ],
)


@pytest.mark.quasar
@parametrize(formats=NARROW_RV_FORMATS)
def test_pack_untilize_narrow_rv_quasar(formats):
    (formats,) = formats
    dest_acc = DestAccumulation.No
    dest_sync_mode = DestSync.Half
    input_dimensions = SINGLE_TILE_DIMS

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Full untilize, then keep the first ROW_NUM_DATUMS columns of each of the
    # 32 rows and pack them tight -> narrow golden (32 * ROW_NUM_DATUMS datums).
    generate_golden = get_golden_generator(UntilizeGolden)
    full_untilized = generate_golden(
        src_A,
        formats.output_format,
        input_dimensions,
        input_format=formats.input_format,
    )
    narrow_golden = (
        full_untilized.reshape(TILE_HEIGHT, TILE_WIDTH)[:, :ROW_NUM_DATUMS]
        .flatten()
        .to(format_dict[formats.output_format])
    )
    narrow_len = TILE_HEIGHT * ROW_NUM_DATUMS

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

    # Diagnostic: dump the full untilized golden and the whole result buffer the SAME
    # way (tile_width-wide rows), so they can be compared line-by-line.
    print_full_tile(
        full_untilized, formats.output_format, label="golden", tile_width=TILE_WIDTH
    )
    print_full_tile(
        res_from_L1, formats.output_format, label="result", tile_width=TILE_WIDTH
    )

    res_full = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    full_len = TILE_HEIGHT * TILE_WIDTH
    assert (
        res_full.numel() >= full_len
    ), f"Result too short: {res_full.numel()} < {full_len}"

    # The kernel untilizes into a full-width (TILE_WIDTH) row-major buffer with the low
    # ROW_NUM_DATUMS columns of each row holding the narrow data. Extract cols
    # [0:ROW_NUM_DATUMS] of each of the TILE_HEIGHT rows and compare to the tight golden.
    res_narrow = (
        res_full[:full_len]
        .reshape(TILE_HEIGHT, TILE_WIDTH)[:, :ROW_NUM_DATUMS]
        .flatten()
    )
    assert res_narrow.numel() == narrow_len

    print_full_tile(
        narrow_golden,
        formats.output_format,
        label="narrow-golden",
        tile_width=ROW_NUM_DATUMS,
    )
    print_full_tile(
        res_narrow,
        formats.output_format,
        label="narrow-result",
        tile_width=ROW_NUM_DATUMS,
    )

    assert passed_test(
        narrow_golden, res_narrow, formats.output_format, print_errors=True
    ), "Narrow RV_PACR untilize output does not match golden"
