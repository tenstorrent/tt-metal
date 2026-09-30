# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Functional coverage for the Welford SFPU kernel.

Two tests:

- `test_sfpu_welford` drives `sources/sfpu_welford_test.cpp`: the mean and the population variance of
  T tiles of 32 samples for 32 columns against a float64 reference, with a 256-entry reciprocal table
  (the form the layernorm kernels run, with a host-built table) and without one (the form ttnn.var and
  ttnn.std run), on bf16 and fp32 inputs and both DEST widths.
- `test_sfpu_welford_reciprocal` drives `sources/sfpu_welford_recip_test.cpp`: the reciprocal the
  kernel computes for the running count when it has no table, bit for bit against the host's fp32
  division, for every count from 1 to 16384 and for windows around 2^16 (where the count no longer
  fits one immediate load) and 2^20.
"""

import numpy as np
import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM
from helpers.llk_params import ApproximationMode, DestAccumulation, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    TILE_COUNT,
    WELFORD_RECIP_BASE,
    WELFORD_RECIP_SIZE,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# The layernorm kernels pass a table with one entry per sample of the reduced dimension; 256 covers
# the 4 x 32 samples of the largest case here with room to spare. 0 is the no-table form.
RECIPROCAL_TABLE_SIZES = [256, 0]


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
        pytest.skip("a Float32 input reaches the kernel through unpack to DEST, which needs a 32-bit DEST")

    torch.manual_seed(0)
    torch_format = format_dict[formats.input_format]

    # [rows, cols] = [num_tiles * 32 samples, 32 parallel columns].
    input_dimensions = [num_tiles * TILE_DIM, TILE_DIM]
    src_A = torch.empty((num_tiles * ELEMENTS_PER_TILE,), dtype=torch_format).uniform_(-4.0, 4.0)
    src_B = torch.zeros_like(src_A)
    golden_input = src_A.view(input_dimensions[0], input_dimensions[1]).to(torch.float64)
    src_A_tilized = tilize_block(src_A, input_dimensions, stimuli_format=formats.input_format).flatten()

    configuration = TestConfig(
        "sources/sfpu_welford_test.cpp",
        formats,
        templates=[APPROX_MODE(ApproximationMode.No), WELFORD_RECIP_SIZE(recip_size)],
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
    mean_tile = untilize_block(res[:ELEMENTS_PER_TILE], formats.output_format, [TILE_DIM, TILE_DIM])
    var_tile = untilize_block(res[ELEMENTS_PER_TILE:], formats.output_format, [TILE_DIM, TILE_DIM])
    # The kernel writes the statistics into row 0 of each tile.
    mean_dev = mean_tile.reshape(TILE_DIM, TILE_DIM)[0].to(torch.float32)
    var_dev = var_tile.reshape(TILE_DIM, TILE_DIM)[0].to(torch.float32)

    mean_ref = golden_input.mean(dim=0).to(torch.float32)
    var_ref = golden_input.var(dim=0, unbiased=False).to(torch.float32)

    assert passed_test(mean_ref, mean_dev, formats.output_format), "Welford mean does not match golden"
    assert passed_test(var_ref, var_dev, formats.output_format), "Welford variance does not match golden"


# 32 reciprocals per result tile; 128 tiles of Float32 (512 KiB of L1) per run.
_RECIP_TILES = 128
_RECIP_PER_RUN = 32 * _RECIP_TILES

# The four slabs of a 4-row group, in the order the kernel stores them: even and odd columns of the
# left face, even and odd columns of the right face.
_SLAB_COLUMN_START = (0, 1, 16, 17)


@parametrize(
    # 1..16384 exhaustively, then the window in which the count stops fitting one 16-bit immediate
    # load and a window around 2^20.
    base=[0, 4096, 8192, 12288, 2**16 - 2048, 2**20 - 2048],
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
        templates=[APPROX_MODE(ApproximationMode.No), WELFORD_RECIP_BASE(base)],
        runtimes=[TILE_COUNT(_RECIP_TILES)],
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
            res[tile * ELEMENTS_PER_TILE : (tile + 1) * ELEMENTS_PER_TILE], DataFormat.Float32, [TILE_DIM, TILE_DIM]
        )
        bits = tile_values.reshape(TILE_DIM, TILE_DIM).numpy().view(np.uint32)
        for slab in range(32):
            row = 16 * (slab >> 4) + 4 * ((slab >> 2) & 3)
            col = _SLAB_COLUMN_START[slab & 3]
            got = bits[row : row + 4, col : col + 16 : 2]
            if not np.all(got == got[0, 0]):
                slabs_not_uniform += 1
            want = expected[tile * 32 + slab]
            if got[0, 0] != want:
                mismatches.append((int(counts[tile * 32 + slab]), int(want), int(got[0, 0])))

    print(
        f"WELFORD_RECIP base={base}: {_RECIP_PER_RUN} counts, {len(mismatches)} mismatches, "
        f"{slabs_not_uniform} slabs not lane-uniform"
    )
    assert slabs_not_uniform == 0, "a reciprocal slab is not lane-uniform"
    assert not mismatches, (
        f"{len(mismatches)} of {_RECIP_PER_RUN} reciprocals differ from the host's fp32 division; "
        f"first (count, expected bits, got bits): {[(c, hex(w), hex(g)) for c, w, g in mismatches[:8]]}"
    )
