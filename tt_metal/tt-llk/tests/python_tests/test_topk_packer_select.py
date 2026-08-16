# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Is packer-resident Top-K selection BIT-EXACT?

WHAT IS BEING CHECKED
---------------------
``perf_topk_pipeline.py`` shows that a threshold + compaction pass done entirely
inside the packer costs 4.175 cyc/vector end to end against
``_topk_xl_merge_``'s 6.997 in the same kernel. That is only interesting if the
survivors that come out are exactly the survivors a torch golden picks, with
their indices, for every awkward value.

The mechanism under test, in one line: Dest holds fused FP32 sort keys
``[bf16 value (high 16) | u16 (index+1) (low 16)]``; the packer's
MIN_THRESHOLD_RELU stage zeroes every datum ``x <= Threshold`` (in FP32, after
early format conversion -- ``Packers/ReLU.md``); zero-compression then elides
the zeroed words as it packs. No SFPU instruction is involved.

WHY THE FUSED WORD IS THE RIGHT THING TO COMPARE
------------------------------------------------
``[bf16 v | u16 (i+1)]`` read as FP32 is ``v`` plus a strictly positive mantissa
perturbation that is monotone in ``i``. So an FP32 compare against a bf16
threshold ``T`` orders by value and breaks ties by index -- and because the index
field is ``i+1`` (never zero), a datum whose value is EXACTLY ``T`` compares
strictly greater and survives. That is a deterministic tie rule, so this file
asserts it rather than avoiding it.

CASES, and what each one can catch
----------------------------------
  allabove   every datum strictly above the threshold. The decoded stream must
             contain exactly 1024 non-zero words. This is the ONLY case that
             catches silent discard: any arm that drops survivors still looks
             perfect on a sparse pattern.
  topk32     32 scattered survivors -- the case that matters.
  negatives  half the values negative, threshold positive. MIN_THRESHOLD_RELU
             cannot express a negative THRESHOLD, but negative DATA is fine: it
             falls below any positive threshold and must be zeroed.
  ties       many datums whose value is exactly the threshold. Per the argument
             above every one of them must survive, ordered by index.
  specials   -0.0 / +0.0 / +-Inf / +-NaN in the value field. FP32 comparison and
             the sign-magnitude total order SFPGT uses disagree on precisely
             these, so what the packer does here is a measurement, not an
             assumption -- the golden below encodes the FP32 (IEEE) rule the
             ReLU functional model is written in, and a mismatch is a finding.

DECODING
--------
The compressed layout is a row-start index array, then per 32-datum group the
surviving datums followed by 32 four-bit run counters. On Blackhole the counter
holds the number of zeroes PRECEDING its datum, not following -- the opposite of
the Wormhole documentation. A decoder written to the documented semantics is
bit-perfect on symmetric patterns and garbage on asymmetric ones, which is why
``front32``-shaped cases exist in the sibling files and why the shared decoder
(``test_pack_compress_int32.decode_compressed32``) is reused verbatim here
rather than rewritten.
"""

import json
import math
import os
import struct
from pathlib import Path

import pytest
import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import PackGolden
from helpers.llk_params import DestAccumulation, DestSync, PackerReluType, Tilize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    NUM_FACES,
    RELU_CONFIG,
    TILIZE,
    generate_input_dim,
)
from test_pack_compress_int32 import decode_compressed32
from test_pack_zero_compress import COMPRESS_PARAMS, DIAG_TILE, RES_TILES, SENTINEL

TILE_DATUMS = 1024
NUM_COMPRESSION_ROWS = 16
ROW_START_SECTION_SIZE = 4

REPORT_DIR = Path(
    os.environ.get(
        "PZC_REPORT_DIR",
        "/tmp/claude-1000/-home-nachiket-tt-metal/d24a23b1-86c2-4faa-aead-36085a95861d/scratchpad",
    )
)


def bf16_bits(x: float) -> int:
    """Top 16 bits of the FP32 encoding of x, i.e. its bf16 pattern.

    Done by hand rather than through torch.bfloat16 so that NaN payloads and
    signed zeros survive: torch's float->bfloat16 cast rounds to nearest even and
    is free to canonicalise a NaN, which would quietly defeat the specials case.
    """
    return struct.unpack("<I", struct.pack("<f", x))[0] >> 16


def fuse(value: float, index: int) -> int:
    """The topk_xl sort key: [bf16 value | u16 (index+1)]."""
    return (bf16_bits(value) << 16) | ((index + 1) & 0xFFFF)


def as_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def bf16_ladder(n=TILE_DATUMS):
    vals = []
    for e in range(8):
        for m in range(128):
            vals.append((2.0**e) * (1.0 + m / 128.0))
    return vals[:n]


def make_case(case):
    """(values[1024], threshold) for one case. Values are exactly representable
    in bf16, so the fused word's high half round-trips with no rounding."""
    ladder = bf16_ladder()
    g = torch.Generator().manual_seed(11)
    perm = torch.randperm(TILE_DATUMS, generator=g).tolist()

    if case == "allabove":
        # Threshold 1.0; the ladder starts at 1.0, so shift it up by one octave
        # to make EVERY datum strictly greater.
        return [2.0 * ladder[perm[i]] for i in range(TILE_DATUMS)], 1.0

    if case == "topk32":
        return [ladder[perm[i]] for i in range(TILE_DATUMS)], ladder[TILE_DATUMS - 32]

    if case == "negatives":
        vals = []
        for i in range(TILE_DATUMS):
            v = ladder[perm[i]]
            vals.append(-v if (i % 2) else v)
        return vals, ladder[TILE_DATUMS - 64]

    if case == "ties":
        # 64 datums sit exactly ON the threshold; the rest straddle it.
        thr = ladder[512]
        vals = []
        for i in range(TILE_DATUMS):
            if i % 16 == 0:
                vals.append(thr)
            else:
                vals.append(ladder[perm[i]])
        return vals, thr

    if case == "specials":
        thr = ladder[512]  # a positive, ordinary bf16
        specials = [
            0.0,
            -0.0,
            math.inf,
            -math.inf,
            float("nan"),
            -float("nan"),
            5.877471754111438e-39,  # smallest normal bf16 (2^-126)
            -5.877471754111438e-39,
        ]
        vals = []
        for i in range(TILE_DATUMS):
            vals.append(specials[i % len(specials)] if i < 256 else ladder[perm[i]])
        return vals, thr


def sign_mag_is_smaller(c: int, d: int) -> bool:
    """``c < d`` under the sign-magnitude total order
    ``-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN``.

    Transcribed from SFPGT.md's ``SignMagIsSmaller``: remap sign-mag
    ``-0 ... -(2^31-1)`` to two's-complement ``-1 ... -2^31``, then compare as
    two's complement.
    """

    def remap(u):
        u &= 0xFFFFFFFF
        s = u - (1 << 32) if u >= (1 << 31) else u  # to int32
        u ^= ((s >> 30) & 0xFFFFFFFF) >> 1
        u &= 0xFFFFFFFF
        return u - (1 << 32) if u >= (1 << 31) else u

    return remap(c) < remap(d)


def golden_words(values, threshold):
    """What the packer must emit for ``x <= Threshold ? 0 : x``, evaluated on the
    FUSED word (not on the bare value) under the SIGN-MAGNITUDE TOTAL ORDER.

    Two things are load-bearing here and both were established by measurement,
    not assumed:

    * The comparison is on the fused word, which is why a datum whose value
      equals the threshold survives -- its non-zero index field makes it
      strictly larger.
    * The order is sign-magnitude total order, NOT IEEE. ``Packers/ReLU.md``
      writes the stage as C floats, which would make every NaN comparison false
      and so let -NaN survive. On silicon it does not: the ``specials`` case
      below, written first against the IEEE reading, mismatched on exactly the
      64 datums whose fused word is a NEGATIVE NaN (a -Inf or -NaN value with an
      index in the mantissa is a -NaN) and on nothing else. +NaN survives, -NaN
      is zeroed -- i.e. the packer orders the way SFPGT does
      (-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN). That is also the
      behaviour Top-K wants, since it is a total order.
    """
    thr = bf16_bits(threshold) << 16
    out = []
    for i, v in enumerate(values):
        w = fuse(v, i)
        # zero unless w > thr in the total order
        out.append(w if sign_mag_is_smaller(thr, w) else 0)
    return out


def build_config(values):
    words = [fuse(v, i) for i, v in enumerate(values)]
    src_A = torch.tensor(
        [w - (1 << 32) if w >= (1 << 31) else w for w in words], dtype=torch.int32
    ).view(torch.float32)
    src_B = torch.zeros(TILE_DATUMS, dtype=torch.float32)
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    return src_A, src_B, formats, words


@pytest.mark.parametrize(
    "case", ["allabove", "topk32", "negatives", "ties", "specials"]
)
@blackhole_only
def test_packer_resident_topk_select(case):
    values, threshold = make_case(case)
    src_A, src_B, formats, words = build_config(values)

    relu_config = PackGolden.generate_relu_config(
        PackerReluType.MinThresholdRelu, threshold, DataFormat.Float32
    )

    configuration = TestConfig(
        "sources/pack_zero_compress_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            TILIZE(Tilize.No),
            DEST_SYNC(DestSync.Half),
            COMPRESS_PARAMS(True, ROW_START_SECTION_SIZE, DIAG_TILE, 0, False, False),
        ],
        runtimes=[
            RELU_CONFIG(relu_config),
            NUM_FACES(num_faces=4),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            DataFormat.Float32,
            src_B,
            DataFormat.Float32,
            DataFormat.Float32,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=RES_TILES,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
    )

    configuration.run()
    stim = configuration.variant_stimuli
    loc = TestConfig.TENSIX_LOCATION
    stim.clear_result_buffer(loc, fill_byte=SENTINEL)
    configuration.run_elf_files()
    configuration.wait_for_tensix_operations_finished()
    raw = bytes(stim.collect_raw_result_bytes(loc))

    tile_bytes = stim.buf_res_tile_size
    diag_off = DIAG_TILE * tile_bytes
    diag = [
        int.from_bytes(raw[diag_off + 4 * i : diag_off + 4 * i + 4], "little")
        for i in range(16)
    ]
    body = raw[:diag_off]
    assert diag[0] == 0xC0DEBA5E and diag[12] == 0xC0DEE0D1, "kernel did not complete"

    dec = decode_compressed32(body, ROW_START_SECTION_SIZE, NUM_COMPRESSION_ROWS)
    got = dec["decoded"]
    want = golden_words(values, threshold)

    n_survivors_want = sum(1 for w in want if w != 0)
    n_survivors_got = sum(1 for w in got if w != 0)

    mismatches = [
        (i, f"0x{want[i]:08X}", f"0x{got[i]:08X}" if i < len(got) else None)
        for i in range(TILE_DATUMS)
        if i >= len(got) or got[i] != want[i]
    ]

    info = {
        "case": case,
        "threshold": threshold,
        "packed_size_bytes": diag[3] * 16,
        "decoded_len": len(got),
        "survivors_expected": n_survivors_want,
        "survivors_decoded": n_survivors_got,
        "augmented_datums": dec["total_augmented_datums"],
        "num_mismatches": len(mismatches),
        "first_mismatches": mismatches[:12],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"topk_select_{case}.json").write_text(json.dumps(info, indent=2))
    print(f"\n==== topk_select_{case} ====")
    for k in (
        "threshold",
        "packed_size_bytes",
        "decoded_len",
        "survivors_expected",
        "survivors_decoded",
        "augmented_datums",
        "num_mismatches",
    ):
        print(f"  {k:22s} {info[k]}")
    if mismatches:
        print(f"  first_mismatches      {mismatches[:6]}")

    assert len(got) == TILE_DATUMS, f"decoded {len(got)} datums, expected {TILE_DATUMS}"

    if case == "allabove":
        # The load-bearing assertion: nothing was silently discarded.
        assert n_survivors_got == TILE_DATUMS, info
        assert dec["total_augmented_datums"] == TILE_DATUMS, info

    assert not mismatches, info
