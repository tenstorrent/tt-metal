# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Welford SFPU kernel: mean and population variance against float64, with and without a reciprocal
table, over whole tiles and over a row range of every tile, and the no-table reciprocal against the
host's fp32 division bit for bit."""

import numpy as np
import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM
from helpers.llk_params import ApproximationMode, DestAccumulation, format_dict
from helpers.param_config import parametrize, runtime
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    TILE_COUNT,
    WELFORD_RECIP_BASE,
    WELFORD_RECIP_SIZE,
    WELFORD_ROWS,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# 256 entries cover the 4 x 32 samples of the largest case; 0 is the no-table form.
RECIPROCAL_TABLE_SIZES = [256, 0]

# An fp32 Welford over at most 128 samples is good to about 1e-6; the default tolerance would let a
# reciprocal of the wrong count through.
FLOAT32_TOLERANCE = {"custom_atol": 1e-4, "custom_rtol": 1e-4}


def _run_welford(
    formats, dest_acc, num_tiles, recip_size, start_row=0, num_rows=TILE_DIM
):
    torch.manual_seed(0)
    torch_format = format_dict[formats.input_format]

    # [rows, cols] = [num_tiles * 32 samples, 32 parallel columns].
    input_dimensions = [num_tiles * TILE_DIM, TILE_DIM]
    src_A = torch.empty((num_tiles * ELEMENTS_PER_TILE,), dtype=torch_format).uniform_(
        -4.0, 4.0
    )
    src_B = torch.zeros_like(src_A)
    golden_input = src_A.view(input_dimensions[0], input_dimensions[1]).to(
        torch.float64
    )
    src_A_tilized = tilize_block(
        src_A, input_dimensions, stimuli_format=formats.input_format
    ).flatten()

    configuration = TestConfig(
        "sources/sfpu_welford_test.cpp",
        formats,
        templates=[
            APPROX_MODE(ApproximationMode.No),
            WELFORD_RECIP_SIZE(recip_size),
            WELFORD_ROWS(start_row, num_rows),
        ],
        runtimes=[TILE_COUNT(num_tiles)],
        variant_stimuli=StimuliConfig(
            src_A_tilized,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=num_tiles,
            tile_count_B=1,
            tile_count_res=2,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
        disable_format_inference=True,
        compile_time_formats=True,
    )
    res_from_L1 = configuration.run().result

    res = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    mean_tile = untilize_block(
        res[:ELEMENTS_PER_TILE], formats.output_format, [TILE_DIM, TILE_DIM]
    )
    var_tile = untilize_block(
        res[ELEMENTS_PER_TILE:], formats.output_format, [TILE_DIM, TILE_DIM]
    )
    # The kernel writes the statistics into row 0 of each tile.
    mean_dev = mean_tile.reshape(TILE_DIM, TILE_DIM)[0].to(torch.float32)
    var_dev = var_tile.reshape(TILE_DIM, TILE_DIM)[0].to(torch.float32)

    samples = golden_input.view(num_tiles, TILE_DIM, TILE_DIM)[
        :, start_row : start_row + num_rows
    ].reshape(-1, TILE_DIM)
    mean_ref = samples.mean(dim=0).to(torch.float32)
    var_ref = samples.var(dim=0, unbiased=False).to(torch.float32)

    tolerance = FLOAT32_TOLERANCE if formats.output_format == DataFormat.Float32 else {}
    assert passed_test(
        mean_ref, mean_dev, formats.output_format, **tolerance
    ), "Welford mean does not match golden"
    assert passed_test(
        var_ref, var_dev, formats.output_format, **tolerance
    ), "Welford variance does not match golden"


@parametrize(
    formats=[
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    num_tiles=[1, 4],
    recip_size=RECIPROCAL_TABLE_SIZES,
)
def test_sfpu_welford(formats, dest_acc, num_tiles, recip_size):
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip(
            "a Float32 input reaches the kernel through unpack to DEST, which needs a 32-bit DEST"
        )
    _run_welford(formats, dest_acc, num_tiles, recip_size)


# Row ranges from row 0, from inside a four-row block, and across the face boundary at row 16.
@parametrize(
    formats=[
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
    ],
    rows=[(0, 5), (3, 6), (13, 19)],
    recip_size=RECIPROCAL_TABLE_SIZES,
)
def test_sfpu_welford_partial_rows(formats, rows, recip_size):
    start_row, num_rows = rows
    dest_acc = (
        DestAccumulation.Yes
        if formats.input_format.is_32_bit()
        else DestAccumulation.No
    )
    _run_welford(formats, dest_acc, 2, recip_size, start_row, num_rows)


_RECIPS_PER_TILE = 32
# 128 tiles of Float32 (512 KiB of L1) per run.
_RECIP_TILES = 128
_RECIP_PER_RUN = _RECIPS_PER_TILE * _RECIP_TILES

# Slab column starts in store order: even and odd columns of the left face, then of the right face.
_SLAB_COLUMN_START = (0, 1, 16, 17)


def _window_around(count):
    return count - _RECIP_PER_RUN // 2


@parametrize(
    # 1..65536 exhaustively; windows across 2^20, 2^22 (where Blackhole's two SFPU forms meet), 2^24
    # and 2^31 and the top of the uint32 range; and windows at significands 1.25, 1.5 and 1.75 from
    # 2^23 up, where the midpoint test multiplies full significands. Every window runs the same ELF.
    base=runtime(
        list(range(0, 2**16, _RECIP_PER_RUN))
        + [_window_around(2**e) for e in (20, 22, 24, 31)]
        + [2**32 - 1 - _RECIP_PER_RUN]
        + [
            _window_around(m * 2**e)
            for m, e in ((3, 22), (5, 25), (7, 27), (5, 29), (3, 30), (7, 29))
        ]
    ),
)
def test_sfpu_welford_reciprocal(base):
    if isinstance(base, tuple):
        # A single sweep axis reaches the test as a one-element tuple.
        (base,) = base
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    src_A = torch.zeros((ELEMENTS_PER_TILE,), dtype=torch.float32)
    src_B = torch.zeros_like(src_A)

    configuration = TestConfig(
        "sources/sfpu_welford_recip_test.cpp",
        formats,
        templates=[APPROX_MODE(ApproximationMode.No)],
        runtimes=[TILE_COUNT(_RECIP_TILES), WELFORD_RECIP_BASE(base)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=_RECIP_TILES,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=False,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    res_from_L1 = configuration.run().result
    res = torch.tensor(res_from_L1, dtype=torch.float32)

    counts = np.arange(base + 1, base + _RECIP_PER_RUN + 1, dtype=np.uint32)
    expected = (np.float32(1.0) / counts.astype(np.float32)).view(np.uint32)

    mismatches = []
    slabs_not_uniform = 0
    for tile in range(_RECIP_TILES):
        tile_values = untilize_block(
            res[tile * ELEMENTS_PER_TILE : (tile + 1) * ELEMENTS_PER_TILE],
            DataFormat.Float32,
            [TILE_DIM, TILE_DIM],
        )
        bits = tile_values.reshape(TILE_DIM, TILE_DIM).numpy().view(np.uint32)
        for slab in range(_RECIPS_PER_TILE):
            row = 16 * (slab >> 4) + 4 * ((slab >> 2) & 3)
            col = _SLAB_COLUMN_START[slab & 3]
            got = bits[row : row + 4, col : col + 16 : 2]
            if not np.all(got == got[0, 0]):
                slabs_not_uniform += 1
            index = tile * _RECIPS_PER_TILE + slab
            if got[0, 0] != expected[index]:
                mismatches.append(
                    (int(counts[index]), int(expected[index]), int(got[0, 0]))
                )

    print(
        f"WELFORD_RECIP base={base}: {_RECIP_PER_RUN} counts, {len(mismatches)} differ from the host's fp32 "
        f"division, {slabs_not_uniform} slabs not lane-uniform"
    )
    assert slabs_not_uniform == 0, "a reciprocal slab is not lane-uniform"
    assert not mismatches, (
        f"{len(mismatches)} of {_RECIP_PER_RUN} reciprocals differ from the host's fp32 division; "
        f"first (count, expected bits, got bits): {[(c, hex(w), hex(g)) for c, w, g in mismatches[:8]]}"
    )
