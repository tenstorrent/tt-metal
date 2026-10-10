# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Column-wise cumulative sum over two tiles in one DEST block, each tile its own scan or, with `chain`,
the second continuing the first."""

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    TILE_DIM,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    MathOperation,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
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

# A 32-bit add chain of at most 64 terms is good to about 1e-6; the default tolerance would also pass a sum
# that went through a 16-bit DEST.
FLOAT32_TOLERANCE = {"custom_atol": 1e-4, "custom_rtol": 1e-4}


@parametrize(
    formats=[
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    chain=[False, True],
)
def test_sfpu_cumsum(formats, dest_acc, chain):
    torch.manual_seed(0)
    torch_format = format_dict[formats.input_format]
    # A Float32 input unpacks straight to a 32-bit DEST; with a 16-bit DEST it goes through SrcA and the datacopy.
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )

    # Up to 64 adds per column; |x| <= 1 keeps every partial sum inside the formats.
    src_A = (
        torch.empty((TILE_CNT * ELEMENTS_PER_TILE,), dtype=torch.float32)
        .uniform_(-1.0, 1.0)
        .to(torch_format)
    )
    src_B = torch.zeros_like(src_A)
    if chain:
        x = src_A.view(INPUT_DIMENSIONS[0], INPUT_DIMENSIONS[1]).to(torch.float32)
        golden = torch.cumsum(x, dim=0)
    else:
        golden = (
            get_golden_generator(UnarySFPUGolden)(
                MathOperation.Cumsum,
                src_A,
                formats.output_format,
                dest_acc,
                formats.input_format,
                INPUT_DIMENSIONS,
            )
            .reshape(INPUT_DIMENSIONS[0], INPUT_DIMENSIONS[1])
            .to(torch.float32)
        )
    src_A_tilized = tilize_block(
        src_A, INPUT_DIMENSIONS, stimuli_format=formats.input_format
    ).flatten()

    configuration = TestConfig(
        "sources/sfpu_cumsum_test.cpp",
        formats,
        templates=[
            generate_input_dim(INPUT_DIMENSIONS, INPUT_DIMENSIONS),
            APPROX_MODE(ApproximationMode.No),
            FAST_MODE(FastMode.No),
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
        unpack_to_dest=unpack_to_dest,
    )
    res_from_L1 = configuration.run().result

    res = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res = untilize_block(res, formats.output_format, INPUT_DIMENSIONS).reshape(
        INPUT_DIMENSIONS[0], INPUT_DIMENSIONS[1]
    )

    tolerance = FLOAT32_TOLERANCE if unpack_to_dest else {}
    assert passed_test(
        golden, res.to(torch.float32), formats.output_format, **tolerance
    ), "cumsum result does not match golden"
