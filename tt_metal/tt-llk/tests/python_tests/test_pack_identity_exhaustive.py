# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exhaustive: every bfloat16 bit pattern through an identity pack.

bfloat16 has 65536 representable bit patterns and a tile holds 1024 datums, so a single
[256, 256] tensor covers the entire type. An A2D datacopy with the same format in and out
performs no conversion, so the pack must not introduce one.

Three classes are asserted separately because two of them are decided upstream of the packer
and would otherwise read as pack behaviour:

  normal, Inf  the word is carried through unchanged, sign included. This is the identity
               claim, and it holds over 65026 of the 65536 patterns.
  denormal, -0 flushed to +0 by MOVA2D, which replaces any SrcA datum whose exponent field is
               zero. Measured identically with and without the packer change.
  NaN          still a NaN. Without the packer change every NaN arrives as an infinity, which
               is the defect. The encoding cannot be asserted from here: the harness quiets
               every NaN payload on the way in (float32 cast in pack_bfp16) and drops the NaN
               sign on the way back (unpack_bfp16 goes through a Python float), so a NaN is
               observable here only as a class. tests/ttnn/unit_tests/base_functionality/
               test_bfloat16_nan_identity_ops.py asserts the bits over a torch round trip.

Prints the complete set of values that change, grouped by class, before asserting.
"""

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
    out = res.flatten()[:N].view(torch.int16)
    got = [int(x.item()) & 0xFFFF for x in out]
    # What actually reached L1, not what the test asked for: pack_bfp16 serialises through
    # float32, and that cast quiets every NaN payload to the canonical encoding.
    sent = [int(b) for b in np.frombuffer(pack_bfp16(src_A), dtype=np.uint16)]

    diffs = [(s, g) for s, g in zip(sent, got) if s != g]
    print(
        f"\n=== all {N} bfloat16 bit patterns through an identity pack, dest_acc={dest_acc.name} ==="
    )
    print(f"differ: {len(diffs)} of {N}")

    from collections import Counter

    by_class = Counter(_classify(s) for s, _ in diffs)
    total = Counter(_classify(s) for s in sent)
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

    # The diagnostic above is what makes a failure readable; the three assertions below are what
    # make it a failure. They are split by class because only the first is about the packer.
    carried = [
        (s, g)
        for s, g in zip(sent, got)
        if _classify(s) in ("normal", "Inf") and s != g
    ]
    assert not carried, (
        f"the identity pack changed {len(carried)} of the 65026 normal and infinite patterns, "
        f"first 0x{carried[0][0]:04X} -> 0x{carried[0][1]:04X}; these need no conversion and "
        "must be carried through unchanged"
    )

    flushed = [
        (s, g)
        for s, g in zip(sent, got)
        if _classify(s) in ("denormal", "zero") and g != 0x0000
    ]
    assert not flushed, (
        f"{len(flushed)} of the 256 zero-exponent patterns did not come back as +0, first "
        f"0x{flushed[0][0]:04X} -> 0x{flushed[0][1]:04X}; MOVA2D replaces them upstream of the "
        "pack, so this is a change in the datacopy rather than in the packer"
    )

    not_nan = [
        (s, g)
        for s, g in zip(sent, got)
        if _classify(s) == "NaN" and _classify(g) != "NaN"
    ]
    assert not not_nan, (
        f"{len(not_nan)} of the 254 NaN patterns stopped being NaN, first 0x{not_nan[0][0]:04X} "
        f"-> 0x{not_nan[0][1]:04X}; this is the defect the identity path exists to fix, and "
        "without it every one of them arrives as an infinity"
    )
