# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Functional coverage for the column-wise cumulative sum (llk_sfpu/ckernel_sfpu_cumsum.h) over two
tiles in one DEST block: with `chain` off every tile is its own 32-row scan, with `chain` on the second
tile continues the first (`first = false`). LLK_CUMSUM_DUMP=<path> appends the raw output bits.
"""

import os
import struct

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM
from helpers.llk_params import ApproximationMode, DestAccumulation, FastMode, MathOperation, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    CUMSUM_CHAIN,
    FAST_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# Two tiles down one column, one DEST block, so the chained variant scans 64 rows.
INPUT_DIMENSIONS = [2 * TILE_DIM, TILE_DIM]
TILE_CNT = 2


@parametrize(
    formats=[
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    chain=[False, True],
)
def test_sfpu_cumsum(formats, dest_acc, chain):
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip("a Float32 input reaches the kernel through unpack to DEST, which needs a 32-bit DEST")

    torch.manual_seed(0)
    torch_format = format_dict[formats.input_format]

    # Up to 64 adds per column; |x| <= 1 keeps every partial sum inside the formats.
    src_A = torch.empty((TILE_CNT * ELEMENTS_PER_TILE,), dtype=torch.float32).uniform_(-1.0, 1.0).to(torch_format)
    src_B = torch.zeros_like(src_A)
    x = src_A.view(INPUT_DIMENSIONS[0], INPUT_DIMENSIONS[1]).to(torch.float32)
    if chain:
        golden = torch.cumsum(x, dim=0)
    else:
        golden = torch.cat([torch.cumsum(x[:TILE_DIM], dim=0), torch.cumsum(x[TILE_DIM:], dim=0)], dim=0)
    src_A_tilized = tilize_block(src_A, INPUT_DIMENSIONS, stimuli_format=formats.input_format).flatten()

    configuration = TestConfig(
        "sources/sfpu_cumsum_test.cpp",
        formats,
        templates=[
            generate_input_dim(INPUT_DIMENSIONS, INPUT_DIMENSIONS),
            APPROX_MODE(ApproximationMode.No),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(False),
            MATH_OP(mathop=MathOperation.Cumsum),
            CUMSUM_CHAIN(chain),
        ],
        runtimes=[TILE_COUNT(TILE_CNT), NUM_BLOCKS(1), NUM_TILES_IN_BLOCK(TILE_CNT)],
        variant_stimuli=StimuliConfig(
            src_A_tilized,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=TILE_CNT,
            tile_count_B=TILE_CNT,
            tile_count_res=TILE_CNT,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
    )
    res_from_L1 = configuration.run().result

    res = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res = untilize_block(res, formats.output_format, INPUT_DIMENSIONS).reshape(INPUT_DIMENSIONS[0], INPUT_DIMENSIONS[1])

    dump = os.environ.get("LLK_CUMSUM_DUMP")
    if dump:
        with open(dump, "a") as fh:
            for i, v in enumerate(res.flatten().to(torch.float32).tolist()):
                b = struct.unpack(">I", struct.pack(">f", float(v)))[0]
                fh.write("%s\t%s\t%d\t%d\t0x%08x\n" % (formats.input_format.name, dest_acc.name, int(chain), i, b))

    assert passed_test(golden, res.to(torch.float32), formats.output_format), "cumsum result does not match golden"
