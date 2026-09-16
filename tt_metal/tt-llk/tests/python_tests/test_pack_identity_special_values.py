# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""An identity pack must not destroy NaN.

A2D datacopy with the same format in and out requires no format conversion, yet the packer's
default rounding path maps NaN to infinity when the exponent is eight bits wide. Assert that a
bfloat16 NaN survives unpack -> Dst -> pack unchanged.
"""

import torch
from helpers.format_config import DataFormat
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    Tilize,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block, input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    TILIZE,
    generate_input_dim,
)

DIMS = [32, 32]
NUM_FACES_V = 4
# quiet NaN, signalling NaN, +Inf, -Inf, 1.0
PATTERNS = [0x7FC0, 0x7F81, 0x7F80, 0xFF80, 0x3F80]


def test_identity_pack_preserves_nan():
    formats = input_output_formats([DataFormat.Float16_b])[0]
    n = DIMS[0] * DIMS[1]

    raw = torch.zeros(n, dtype=torch.int16)
    for i, b in enumerate(PATTERNS):
        raw[i] = b - (1 << 16) if b >= 0x8000 else b
    src_A = raw.view(torch.bfloat16)

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        DestAccumulation.No,
        formats,
        DIMS,
        [32, 32],
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/eltwise_unary_datacopy_test.cpp",
        formats,
        templates=[generate_input_dim(DIMS, DIMS), TILIZE(Tilize.No)],
        runtimes=[
            DEST_INDEX(0),
            TILE_COUNT(1),
            NUM_FACES(NUM_FACES_V),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_A,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=NUM_FACES_V,
        ),
        dest_acc=DestAccumulation.No,
    )

    res = configuration.run().result
    got = [int(x.item()) & 0xFFFF for x in res.flatten()[: len(PATTERNS)].view(torch.int16)]

    for src, out in zip(PATTERNS, got):
        is_nan_in = (src & 0x7F80) == 0x7F80 and (src & 0x007F)
        is_nan_out = (out & 0x7F80) == 0x7F80 and (out & 0x007F)
        assert bool(is_nan_in) == bool(is_nan_out), (
            f"identity pack changed 0x{src:04X} into 0x{out:04X}; a NaN must stay a NaN"
        )
