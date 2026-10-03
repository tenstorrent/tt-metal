# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""An identity pack must not destroy NaN.

A2D datacopy with the same format in and out requires no format conversion, yet the packer's
default rounding path maps NaN to infinity when the exponent is eight bits wide. Assert that a
bfloat16 NaN reaches L1 as a NaN, and that the infinities and the finite value next to it are
carried through bit for bit.

The NaN encoding is asserted as a class rather than as bits, because this harness cannot carry
one: pack_bfp16 serialises the stimulus through float32, which quiets every payload, and
unpack_bfp16 reads the result back through a Python float, which drops the sign. Both are host
steps, measured on both sides of the packer change. The bits are asserted over a torch round
trip in tests/ttnn/unit_tests/base_functionality/test_bfloat16_nan_identity_ops.py.
"""

import numpy as np
import torch
from helpers.format_config import DataFormat
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    PerfRunType,
    Tilize,
)
from helpers.pack import pack_bfp16
from helpers.param_config import (
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
)
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PERF_RUN_TYPE,
    TILE_COUNT,
    TILIZE,
    generate_input_dim,
)

DIMS = [32, 32]
NUM_FACES_V = 4
# Positive and negative quiet NaN, +Inf, -Inf, 1.0.
PATTERNS = [0x7FC0, 0xFFC0, 0x7F80, 0xFF80, 0x3F80]
NAN_PATTERNS = {0x7FC0, 0xFFC0}


def test_identity_pack_preserves_nan():
    formats = input_output_formats([DataFormat.Float16_b])[0]
    n = DIMS[0] * DIMS[1]

    raw = torch.zeros(n, dtype=torch.int16)
    for i, b in enumerate(PATTERNS):
        raw[i] = b - (1 << 16) if b >= 0x8000 else b
    src_A = raw.view(torch.bfloat16)

    # Guard the stimulus path itself, so a serialiser that quieted one of these would fail here
    # rather than be read as the packer destroying it.
    sent = np.frombuffer(pack_bfp16(src_A), dtype=np.uint16)[: len(PATTERNS)]
    assert list(sent) == PATTERNS, (
        f"the stimulus reached L1 as {[f'0x{b:04X}' for b in sent]}, not as the patterns "
        "under test; the packer is not what this run measured"
    )

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
        templates=[
            generate_input_dim(DIMS, DIMS),
            TILIZE(Tilize.No),
            PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
        ],
        runtimes=[
            DEST_INDEX(0),
            TILE_COUNT(1),
            NUM_FACES(NUM_FACES_V),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            LOOP_FACTOR(1),
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
    got = [
        int(x.item()) & 0xFFFF for x in res.flatten()[: len(PATTERNS)].view(torch.int16)
    ]

    # A NaN is checked as a class and everything else bit for bit. Checking only "is it still a
    # NaN" for the whole set would pass a path that turned the infinities or the finite value
    # into one, which is the opposite half of what the identity path exists to prevent.
    for src, out in zip(PATTERNS, got):
        if src in NAN_PATTERNS:
            assert (out & 0x7F80) == 0x7F80 and (out & 0x007F) != 0, (
                f"identity pack turned the NaN 0x{src:04X} into 0x{out:04X}; without the "
                "identity path it arrives as an infinity, which is the defect"
            )
        else:
            assert src == out, (
                f"identity pack changed 0x{src:04X} into 0x{out:04X}; a pack that needs no "
                "conversion must carry the word through unchanged"
            )
