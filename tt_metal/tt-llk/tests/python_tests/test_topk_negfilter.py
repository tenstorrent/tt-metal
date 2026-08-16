# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Does a NEGATIVE-threshold Top-K filter come out bit-exact on Blackhole?

THE GAP
-------
The zero-SFPU Top-K path -- packer ``MIN_THRESHOLD_RELU`` doing the compare and
zero-compression doing the compaction -- cannot express a negative threshold:
``Packers/ReLU.md:41`` makes ``signbit(Threshold)`` UndefinedBehavior. Negative
DATA is fine (it falls below any non-negative threshold); a negative THRESHOLD
is not. Signed logits (MoE routing, vocab sampling) routinely need one.

This file measures two things on silicon:

``test_packer_negative_threshold``
    What the packer ACTUALLY does when handed a negative threshold. The doc says
    UndefinedBehavior, which is a statement about what was verified, not
    necessarily about what the comparator does. If it happens to implement
    ``x <= T ? 0 : x`` for ``T < 0`` then the gap closes with zero SFPU
    instructions and nothing else in this file matters for cost.

``test_sfpu_negfilter``
    The SFPU fallback: ``sources/topk_negfilter_common.h``, two SFPLOADMACROs
    per 32-element vector computing ``Dst[i] = (Dst[i] > T) ? Dst[i] : +0.0``
    with SFPGT's -1/0 mask and an SFPAND. Bitwise only, so survivors keep their
    EXACT bits (no denormal flush, no NaN canonicalisation) and losers are
    exactly ``0x00000000``, which is what zero-compression needs.

WHY THE FUSED WORD IS THE RIGHT THING TO COMPARE
------------------------------------------------
Dst holds ``[bf16 value (high 16) | u16 (index+1) (low 16)]`` read as FP32.
Comparing the whole word orders by value and breaks ties by index. Note the tie
rule is SIGN-DEPENDENT and this file asserts it rather than avoiding it: the
index field only ever ADDS magnitude, so a datum whose value equals a POSITIVE
threshold compares strictly greater and survives, while a datum whose value
equals a NEGATIVE threshold is strictly more negative and is zeroed.

CASES
-----
  negtopk32    all-negative data, negative threshold, exactly 32 survivors --
               the case that matters
  negmixed     half the values negative, negative threshold
  negallabove  every datum strictly above a negative threshold. The decoded
               stream must contain exactly 1024 non-zero words. The ONLY case
               that catches silent discard: any arm that drops survivors still
               looks perfect on a sparse pattern
  negties      64 datums whose value is exactly the (negative) threshold. Every
               one of them must be ZEROED, per the sign-dependent tie rule
  negspecials  +-0 / +-Inf / +-NaN / denormals against a negative threshold.
               IEEE and the sign-magnitude total order disagree precisely here
  postopk32    positive threshold -- cross-check that the SFPU filter picks the
               same survivors the packer-resident path does

DECODING
--------
Shared with ``test_topk_packer_select.py``: row-start index array, then per
32-datum group the surviving datums followed by 32 four-bit run counters, where
on Blackhole the counter holds the number of zeroes PRECEDING its datum.
"""

import json
import math
import os
import struct
from dataclasses import dataclass
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
    TemplateParameter,
    generate_input_dim,
)
from test_pack_compress_int32 import decode_compressed32

TILE_DATUMS = 1024
NUM_COMPRESSION_ROWS = 16
ROW_START_SECTION_SIZE = 4
RES_TILES = 8
DIAG_TILE = 7
SENTINEL = 0xA5

REPORT_DIR = Path(
    os.environ.get(
        "PZC_REPORT_DIR",
        "/tmp/claude-1000/-home-nachiket-tt-metal/d24a23b1-86c2-4faa-aead-36085a95861d/scratchpad",
    )
)


@dataclass
class NEGFILTER_PARAMS(TemplateParameter):
    """Compile-time knobs for ``sources/topk_negfilter_test.cpp``."""

    filter_en: bool = False
    thr_bits: int = 0
    compress_en: bool = True
    row_start_section_size: int = ROW_START_SECTION_SIZE
    diag_tile_index: int = DIAG_TILE
    downsample_mask: int = 0

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                f"constexpr bool FILTER_EN = {str(self.filter_en).lower()};",
                f"constexpr std::uint32_t THR_BITS = {self.thr_bits}u;",
                f"constexpr bool COMPRESS_EN = {str(self.compress_en).lower()};",
                f"constexpr std::uint32_t ROW_START_SECTION_SIZE = {self.row_start_section_size};",
                f"constexpr std::uint32_t DIAG_TILE_INDEX = {self.diag_tile_index};",
                f"constexpr std::uint32_t DOWNSAMPLE_MASK = {self.downsample_mask};",
            ]
        )


# ---------------------------------------------------------------------------
# Bit-level helpers. Done by hand rather than through torch.bfloat16 so that NaN
# payloads and signed zeros survive: torch's float->bfloat16 cast rounds to
# nearest even and is free to canonicalise a NaN, which would quietly defeat the
# specials case.
# ---------------------------------------------------------------------------


def bf16_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0] >> 16


def fuse(value: float, index: int) -> int:
    """The topk_xl sort key: [bf16 value | u16 (index+1)]."""
    return (bf16_bits(value) << 16) | ((index + 1) & 0xFFFF)


def bf16_ladder(n=TILE_DATUMS):
    """n distinct, exactly-representable bfloat16 magnitudes in [1, 256)."""
    vals = []
    for e in range(8):
        for m in range(128):
            vals.append((2.0**e) * (1.0 + m / 128.0))
    return vals[:n]


def sign_mag_is_smaller(c: int, d: int) -> bool:
    """``c < d`` under ``-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN``.

    Transcribed from SFPGT.md's ``SignMagIsSmaller``.
    """

    def remap(u):
        u &= 0xFFFFFFFF
        s = u - (1 << 32) if u >= (1 << 31) else u
        u ^= ((s >> 30) & 0xFFFFFFFF) >> 1
        u &= 0xFFFFFFFF
        return u - (1 << 32) if u >= (1 << 31) else u

    return remap(c) < remap(d)


def golden_words(values, threshold):
    """``w > Threshold ? w : 0`` on the FUSED word under the total order."""
    thr = bf16_bits(threshold) << 16
    return [
        (w if sign_mag_is_smaller(thr, w) else 0)
        for w in (fuse(v, i) for i, v in enumerate(values))
    ]


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

_SORTED_LADDER = sorted(bf16_ladder())


def make_case(case):
    """(values[1024], threshold)."""
    ladder = bf16_ladder()
    g = torch.Generator().manual_seed(11)
    perm = torch.randperm(TILE_DATUMS, generator=g).tolist()

    if case == "negtopk32":
        # Every datum negative. Survivors are the ones whose MAGNITUDE is below
        # the threshold's, so picking the 33rd smallest magnitude as |T| leaves
        # exactly 32 survivors -- and the datum whose value IS T must be zeroed
        # (its index field pushes it further from zero).
        thr = -_SORTED_LADDER[32]
        vals = [-_SORTED_LADDER[i] for i in perm]
        return vals, thr

    if case == "negmixed":
        # Half negative. Every positive datum survives any negative threshold;
        # the negatives split on magnitude.
        thr = -_SORTED_LADDER[64]
        vals = []
        for i in range(TILE_DATUMS):
            v = ladder[perm[i]]
            vals.append(-v if (i % 2) else v)
        return vals, thr

    if case == "negallabove":
        # The ladder tops out at 255.0, so -256.0 is strictly below every datum
        # (fused index field included) and NOTHING may be discarded.
        vals = []
        for i in range(TILE_DATUMS):
            v = ladder[perm[i]]
            vals.append(-v if (i % 2) else v)
        return vals, -256.0

    if case == "negties":
        # 64 datums sit exactly ON the negative threshold and must all be zeroed.
        thr = -_SORTED_LADDER[512]
        vals = []
        for i in range(TILE_DATUMS):
            if i % 16 == 0:
                vals.append(thr)
            else:
                v = ladder[perm[i]]
                vals.append(-v if (i % 2) else v)
        return vals, thr

    if case == "negspecials":
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
            if i < 256:
                vals.append(specials[i % len(specials)])
            else:
                v = ladder[perm[i]]
                vals.append(-v if (i % 2) else v)
        return vals, -16.0

    if case == "postopk32":
        return [ladder[perm[i]] for i in range(TILE_DATUMS)], _SORTED_LADDER[-32]

    raise ValueError(case)


def build_and_run(values, threshold, filter_en, relu_en):
    """Run one variant and return (decoded, diag, packed_bytes)."""
    words = [fuse(v, i) for i, v in enumerate(values)]
    src_A = torch.tensor(
        [w - (1 << 32) if w >= (1 << 31) else w for w in words], dtype=torch.int32
    ).view(torch.float32)
    src_B = torch.zeros(TILE_DATUMS, dtype=torch.float32)
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)

    relu_config = (
        PackGolden.generate_relu_config(
            PackerReluType.MinThresholdRelu, threshold, DataFormat.Float32
        )
        if relu_en
        else 0
    )
    # A negative bf16 threshold sets bit 31 of the packed config word, and the
    # runtime-parameter marshaller packs RELU_CONFIG with struct 'i'. Hand it the
    # two's-complement view of the same 32 bits.
    if relu_config >= (1 << 31):
        relu_config -= 1 << 32

    # The SFPU compares the fused FP32 word against the same bf16 threshold the
    # packer ReLU would use, with the index field zero.
    thr_bits = bf16_bits(threshold) << 16

    configuration = TestConfig(
        "sources/topk_negfilter_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            TILIZE(Tilize.No),
            DEST_SYNC(DestSync.Half),
            NEGFILTER_PARAMS(filter_en=filter_en, thr_bits=thr_bits),
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
    return decode_compressed32(body, ROW_START_SECTION_SIZE, NUM_COMPRESSION_ROWS), diag


def report(tag, case, threshold, dec, diag, want, strict):
    got = dec["decoded"]
    n_want = sum(1 for w in want if w != 0)
    n_got = sum(1 for w in got if w != 0)
    mismatches = [
        (i, f"0x{want[i]:08X}", f"0x{got[i]:08X}" if i < len(got) else None)
        for i in range(TILE_DATUMS)
        if i >= len(got) or got[i] != want[i]
    ]
    info = {
        "tag": tag,
        "case": case,
        "threshold": threshold,
        "packed_size_bytes": diag[3] * 16,
        "decoded_len": len(got),
        "survivors_expected": n_want,
        "survivors_decoded": n_got,
        "augmented_datums": dec["total_augmented_datums"],
        "num_mismatches": len(mismatches),
        "first_mismatches": mismatches[:12],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"negfilter_{tag}_{case}.json").write_text(json.dumps(info, indent=2))
    print(f"\n==== {tag}/{case} ====")
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
    if strict:
        assert len(got) == TILE_DATUMS, info
        assert not mismatches, info
    return info


@pytest.mark.parametrize(
    "case",
    ["negtopk32", "negmixed", "negallabove", "negties", "negspecials", "postopk32"],
)
@blackhole_only
def test_sfpu_negfilter(case):
    """The SFPU fallback, packer ReLU OFF -- the filter must be bit-exact."""
    values, threshold = make_case(case)
    dec, diag = build_and_run(values, threshold, filter_en=True, relu_en=False)
    want = golden_words(values, threshold)
    info = report("sfpu", case, threshold, dec, diag, want, strict=True)

    if case == "negallabove":
        # The load-bearing assertion: nothing was silently discarded.
        assert info["survivors_decoded"] == TILE_DATUMS, info
        assert dec["total_augmented_datums"] == TILE_DATUMS, info


@pytest.mark.parametrize("case", ["negtopk32", "negmixed", "negspecials"])
@blackhole_only
def test_packer_negative_threshold(case):
    """MEASUREMENT, not a check: what does the packer's MIN_THRESHOLD_RELU do
    with signbit(Threshold) set? ``Packers/ReLU.md:41`` calls it
    UndefinedBehavior. If it turns out to implement the same compare it does for
    non-negative thresholds, the SFPU fallback is unnecessary. This test does
    not fail on a mismatch -- it records what happened.
    """
    values, threshold = make_case(case)
    dec, diag = build_and_run(values, threshold, filter_en=False, relu_en=True)
    want = golden_words(values, threshold)
    info = report("packerneg", case, threshold, dec, diag, want, strict=False)

    # Competing hypothesis: the comparator ignores the threshold's sign bit, so
    # a threshold of -T behaves as +T.
    got = dec["decoded"]
    want_abs = golden_words(values, abs(threshold))
    abs_mm = [
        (i, values[i], f"0x{want_abs[i]:08X}", f"0x{got[i]:08X}")
        for i in range(TILE_DATUMS)
        if i >= len(got) or got[i] != want_abs[i]
    ]
    abs_mismatches = len(abs_mm)
    if abs_mm:
        print(f"  |T| first mismatches  {abs_mm[:8]}")
    print(
        f"  VERDICT               "
        f"{'MATCHES signed golden' if info['num_mismatches'] == 0 else 'DIVERGES from signed golden'}"
    )
    print(
        f"  |T| HYPOTHESIS        "
        f"{'CONFIRMED (behaves as +|T|)' if abs_mismatches == 0 else f'rejected ({abs_mismatches} mismatches)'}"
    )
