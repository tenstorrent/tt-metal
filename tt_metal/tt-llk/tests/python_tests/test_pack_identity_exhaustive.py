# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exhaustive: every bfloat16 bit pattern through an identity pack.

bfloat16 has 65536 representable bit patterns and a tile holds 1024 datums, so a single
[256, 256] tensor covers the entire type. An A2D datacopy with the same format in and out
performs no conversion, so every output word must equal its input word exactly.

Prints the complete set of values that change, grouped by class. Not a pass/fail test.
"""

import pytest
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

DIMS = [256, 256]
N = DIMS[0] * DIMS[1]
TILE_CNT = N // (32 * 32)


def _classify(b):
    exp = b & 0x7F80
    man = b & 0x007F
    if exp == 0x7F80:
        return "NaN" if man else "Inf"
    if exp == 0:
        return "zero" if man == 0 else "denormal"
    return "normal"


@pytest.mark.parametrize("dest_acc", [DestAccumulation.No])
def test_identity_pack_all_bf16(dest_acc):
    formats = input_output_formats([DataFormat.Float16_b])[0]

    raw = torch.arange(N, dtype=torch.int32) & 0xFFFF
    raw = torch.where(raw >= 0x8000, raw - (1 << 16), raw).to(torch.int16)
    src_A = raw.view(torch.bfloat16)

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half, dest_acc, formats, DIMS, [32, 32], BlocksCalculationAlgorithm.Standard
    )

    configuration = TestConfig(
        "sources/eltwise_unary_datacopy_test.cpp",
        formats,
        templates=[generate_input_dim(DIMS, DIMS), TILIZE(Tilize.No)],
        runtimes=[
            DEST_INDEX(0),
            TILE_COUNT(TILE_CNT),
            NUM_FACES(4),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_A,
            formats.input_format,
            formats.output_format,
            tile_count_A=TILE_CNT,
            tile_count_B=TILE_CNT,
            tile_count_res=TILE_CNT,
            num_faces=4,
        ),
        dest_acc=dest_acc,
    )

    res = configuration.run().result
    out = res.flatten()[:N].view(torch.int16)
    got = [int(x.item()) & 0xFFFF for x in out]
    src = [int(x.item()) & 0xFFFF for x in raw]

    diffs = [(s, g) for s, g in zip(src, got) if s != g]
    print(f"\n=== all {N} bfloat16 bit patterns through an identity pack, dest_acc={dest_acc.name} ===")
    print(f"differ: {len(diffs)} of {N}")

    from collections import Counter

    by_class = Counter(_classify(s) for s, _ in diffs)
    total = Counter(_classify(s) for s in src)
    for cls in ("normal", "denormal", "zero", "Inf", "NaN"):
        print(f"  {cls:9s} changed {by_class.get(cls, 0):6d} of {total.get(cls, 0):6d}")

    seen = set()
    shown = 0
    for s, g in diffs:
        key = (_classify(s), _classify(g))
        if key in seen:
            continue
        seen.add(key)
        print(f"    example {key[0]:8s} -> {key[1]:8s} : 0x{s:04X} -> 0x{g:04X}")
        shown += 1
        if shown > 12:
            break
