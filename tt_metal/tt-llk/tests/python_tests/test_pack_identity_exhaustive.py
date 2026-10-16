# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Every 16 bit pattern of Float16_b and Float16 through an identity pack (A2D datacopy, same format in and out)."""

from collections import Counter

import numpy as np
import pytest
import torch
from helpers.format_config import DataFormat
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    PerfRunType,
    Tilize,
)
from helpers.pack import pack_bfp16, pack_fp16
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

DIMS = [256, 256]
N = DIMS[0] * DIMS[1]
TILE_CNT = N // (32 * 32)

# (torch dtype, pack function, exponent mask, mantissa mask) per format.
FIELDS = {
    DataFormat.Float16_b: (torch.bfloat16, pack_bfp16, 0x7F80, 0x007F),
    DataFormat.Float16: (torch.float16, pack_fp16, 0x7C00, 0x03FF),
}


def _classify(b, exp_mask, man_mask):
    exp = b & exp_mask
    man = b & man_mask
    if exp == exp_mask:
        return "NaN" if man else "Inf"
    if exp == 0:
        return "zero" if man == 0 else "denormal"
    return "normal"


@pytest.mark.parametrize("fmt", [DataFormat.Float16_b, DataFormat.Float16])
@pytest.mark.parametrize("dest_acc", [DestAccumulation.No])
def test_identity_pack_all_patterns(fmt, dest_acc):
    formats = input_output_formats([fmt])[0]
    dtype, pack_fn, exp_mask, man_mask = FIELDS[fmt]

    raw = torch.arange(N, dtype=torch.int32) & 0xFFFF
    raw = torch.where(raw >= 0x8000, raw - (1 << 16), raw).to(torch.int16)
    src_A = raw.view(dtype)

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
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
            TILE_COUNT(TILE_CNT),
            NUM_FACES(4),
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
            tile_count_A=TILE_CNT,
            tile_count_B=TILE_CNT,
            tile_count_res=TILE_CNT,
            num_faces=4,
        ),
        dest_acc=dest_acc,
    )

    res = configuration.run().result
    got = [int(x) & 0xFFFF for x in res.flatten()[:N].view(torch.int16).tolist()]
    # The words that reached L1: pack_bfp16 quiets NaN payloads, pack_fp16 keeps them.
    sent = [int(b) for b in np.frombuffer(pack_fn(src_A), dtype=np.uint16)]

    def cls(b):
        return _classify(b, exp_mask, man_mask)

    diffs = [(s, g) for s, g in zip(sent, got) if s != g]
    changed = Counter(cls(s) for s, _ in diffs)
    total = Counter(cls(s) for s in sent)
    print(f"\n{fmt.name}: {len(diffs)} of {N} patterns differ")
    for c in ("normal", "Inf", "zero", "denormal", "NaN"):
        print(f"  {c:9s} changed {changed.get(c, 0):6d} of {total.get(c, 0):6d}")
    for c in changed:
        for s, g in [d for d in diffs if cls(d[0]) == c][:3]:
            print(f"  {c:8s} 0x{s:04X} -> {cls(g):8s} 0x{g:04X}")

    # Normal and Inf words need no conversion and must be carried bit for bit.
    carried = [
        (s, g) for s, g in zip(sent, got) if cls(s) in ("normal", "Inf") and s != g
    ]
    assert (
        not carried
    ), f"{len(carried)} normal/Inf patterns changed, first 0x{carried[0][0]:04X} -> 0x{carried[0][1]:04X}"

    # Denormals and -0 are flushed to +0 by MOVA2D upstream of the packer.
    # fp16 denormals are expected to flush like bf16 ones (MOVA2D).
    flushed = [
        (s, g)
        for s, g in zip(sent, got)
        if cls(s) in ("denormal", "zero") and g != 0x0000
    ]
    assert (
        not flushed
    ), f"{len(flushed)} zero exponent patterns did not come back as +0, first 0x{flushed[0][0]:04X} -> 0x{flushed[0][1]:04X}"

    # NaN is checked as a class: the readback quiets payloads and, for bf16, drops the sign.
    not_nan = [(s, g) for s, g in zip(sent, got) if cls(s) == "NaN" and cls(g) != "NaN"]
    assert (
        not not_nan
    ), f"{len(not_nan)} NaN patterns stopped being NaN, first 0x{not_nan[0][0]:04X} -> 0x{not_nan[0][1]:04X}"
