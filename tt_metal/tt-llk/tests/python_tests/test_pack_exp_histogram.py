# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Experimental probe: does the Blackhole packer's EXPONENT HISTOGRAM work?

WormholeB0/TensixTile/TensixCoprocessor/Packers/ExponentHistogram.md documents a
per-packer ``std::uint8_t ExponentHistogram[32]`` incremented for every datum fetched from
Dst, gated on ``ThreadConfig[*].ENABLE_ACC_STATS_Enable``, cleared by ``CLREXPHIST``,
and read into GPRs by ``SETDMAREG`` modes 6/7 (histogram halves) and 9 (running max
exponent).  BlackholeA0 has no packer documentation at all, so every claim below that
is not tagged "docs say" is a silicon observation from this file.

Each test packs a tile whose exponent distribution is known exactly, then compares the
sum of the four packers' 32-bin histograms against that distribution.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, Tilize
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

TILE_DATUMS = 1024
RES_TILES = 8
DIAG_TILE = 7
NUM_PACKERS = 4
NUM_BINS = 32
POISON = 0x3EAD3EAD

REPORT_DIR = Path(
    os.environ.get(
        "PEH_REPORT_DIR",
        "/tmp/claude-1000/-home-nachiket-tt-metal/d24a23b1-86c2-4faa-aead-36085a95861d/scratchpad",
    )
)

_BF16 = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
_FP16 = InputOutputFormat(DataFormat.Float16, DataFormat.Float16)


@dataclass
class HIST_PARAMS(TemplateParameter):
    hist_en: bool = True
    hist_en_unpack: bool = False
    hist_en_math: bool = False
    clr_mode: int = 2
    num_packs: int = 1
    clr_between: bool = False
    diag_tile_index: int = DIAG_TILE
    downsample_mask: int = 0

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                f"constexpr bool HIST_EN = {str(self.hist_en).lower()};",
                f"constexpr bool HIST_EN_UNPACK = {str(self.hist_en_unpack).lower()};",
                f"constexpr bool HIST_EN_MATH = {str(self.hist_en_math).lower()};",
                f"constexpr std::uint32_t CLR_MODE = {self.clr_mode};",
                f"constexpr std::uint32_t NUM_PACKS = {self.num_packs};",
                f"constexpr bool CLR_BETWEEN = {str(self.clr_between).lower()};",
                f"constexpr std::uint32_t DIAG_TILE_INDEX = {self.diag_tile_index};",
                f"constexpr std::uint32_t DOWNSAMPLE_MASK = {self.downsample_mask};",
            ]
        )


# ---------------------------------------------------------------------------
# Golden: what the histogram should contain for a given tile
# ---------------------------------------------------------------------------


def raw_exponents(t, fmt):
    """The raw biased exponent field of every datum, per the WH page's `d.Exponent`."""
    bits = t.view(torch.uint16).to(torch.int32).tolist()
    if fmt is DataFormat.Float16_b:  # 8 bits of exponent
        return [(b >> 7) & 0xFF for b in bits]
    if fmt is DataFormat.Float16:  # 5 bits of exponent
        return [(b >> 10) & 0x1F for b in bits]
    raise ValueError(fmt)


def golden_bins(t, fmt):
    """Bin = Exponent & 31, summed over the whole tile (i.e. over all four packers)."""
    hist = [0] * NUM_BINS
    for e in raw_exponents(t, fmt):
        hist[e & 31] += 1
    return hist


# ---------------------------------------------------------------------------
# Stimuli
# ---------------------------------------------------------------------------


def tile_from_counts(counts, dtype):
    """counts: list of (value, n). Values are laid out contiguously, then shuffled
    deterministically so no packer can be handed a trivially-ordered slice."""
    vals = []
    for v, n in counts:
        vals.extend([v] * n)
    assert len(vals) == TILE_DATUMS, len(vals)
    g = torch.Generator().manual_seed(5)
    perm = torch.randperm(TILE_DATUMS, generator=g).tolist()
    out = [0.0] * TILE_DATUMS
    for i, p in enumerate(perm):
        out[p] = vals[i]
    return torch.tensor(out, dtype=dtype)


# ---------------------------------------------------------------------------
# Run + decode
# ---------------------------------------------------------------------------


def build_config(src_A, formats, params: HIST_PARAMS):
    src_B = torch.zeros(TILE_DATUMS, dtype=src_A.dtype)
    return TestConfig(
        "sources/pack_exp_histogram_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            TILIZE(Tilize.No),
            DEST_SYNC(DestSync.Half),
            params,
        ],
        runtimes=[
            RELU_CONFIG(0),
            NUM_FACES(num_faces=4),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=RES_TILES,
        ),
        dest_acc=DestAccumulation.No,
    )


def run_and_decode(configuration):
    configuration.run()
    stim = configuration.variant_stimuli
    raw = bytes(stim.collect_raw_result_bytes(TestConfig.TENSIX_LOCATION))
    tile_bytes = stim.buf_res_tile_size
    off = DIAG_TILE * tile_bytes
    d = [
        int.from_bytes(raw[off + 4 * i : off + 4 * i + 4], "little") for i in range(48)
    ]

    per_packer = []
    for p in range(NUM_PACKERS):
        words = d[1 + 8 * p : 1 + 8 * p + 8]
        bins = []
        for w in words:
            bins.extend([(w >> (8 * k)) & 0xFF for k in range(4)])
        per_packer.append(bins)

    total = [sum(per_packer[p][b] for p in range(NUM_PACKERS)) for b in range(NUM_BINS)]

    return {
        "sentinel_ok": d[0] == 0xC0DEBA5E and d[44] == 0xC0DEE0D1,
        "raw_words": [f"0x{w:08X}" for w in d[:45]],
        "per_packer": per_packer,
        "total": total,
        "packer_totals": [sum(h) for h in per_packer],
        "grand_total": sum(total),
        "max_exp_word": f"0x{d[33]:08X}",
        "max_exp": d[33] & 0xFF,
        "poison_control": [f"0x{w:08X}" for w in d[34:38]],
        "poison_control_ok": all(w == POISON for w in d[34:38]),
        "any_poison_in_hist": any(w == POISON for w in d[1:33]),
        "packed_size_t2_units": d[38],
        "hist_en": d[39],
        "clr_mode": d[40],
        "num_packs": d[41],
        "num_faces": d[42],
    }


def nz(hist):
    return {b: c for b, c in enumerate(hist) if c}


def _report(info, name):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"peh_{name}.json").write_text(json.dumps(info, indent=2))
    print(f"\n==== {name} ====")
    print(f"  sentinel_ok            {info['sentinel_ok']}")
    print(
        f"  poison_control_ok      {info['poison_control_ok']} {info['poison_control']}"
    )
    print(f"  any_poison_in_hist     {info['any_poison_in_hist']}")
    print(
        f"  hist_en/clr/num_packs  {info['hist_en']}/{info['clr_mode']}/{info['num_packs']}"
    )
    print(f"  packed_size_t2_units   {info['packed_size_t2_units']}")
    print(
        f"  max_exp_word           {info['max_exp_word']}  (low byte {info['max_exp']})"
    )
    print(
        f"  packer_totals          {info['packer_totals']} (grand {info['grand_total']})"
    )
    for p in range(NUM_PACKERS):
        print(f"  packer{p} nonzero bins   {nz(info['per_packer'][p])}")
    print(f"  SUM nonzero bins       {nz(info['total'])}")
    if "expected" in info:
        print(f"  EXPECTED nonzero bins  {nz(info['expected'])}")
        print(f"  exact_match            {info['exact_match']}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# bf16: 2^k has biased exponent 127+k, bin (127+k)&31.
#   1.0    -> exp 127 -> bin 31
#   16.0   -> exp 131 -> bin 3
#   256.0  -> exp 135 -> bin 7
_BF16_THREE = [(1.0, 32), (16.0, 100), (256.0, 892)]

# fp16: 2^k has biased exponent 15+k, bin 15+k (5-bit field, no aliasing possible).
#   1.0     -> exp 15 -> bin 15
#   16.0    -> exp 19 -> bin 19
#   0.015625 (2^-6) -> exp 9 -> bin 9
_FP16_THREE = [(1.0, 32), (16.0, 100), (2.0**-6, 892)]


@pytest.mark.parametrize("hist_en", [True, False])
def test_hist_known_distribution_bf16(hist_en):
    """Task 2: exactly 32 datums at one exponent, 100 at another, 892 at a third."""
    src = tile_from_counts(_BF16_THREE, torch.bfloat16)
    cfg = build_config(src, _BF16, HIST_PARAMS(hist_en=hist_en, clr_mode=2))
    info = run_and_decode(cfg)
    info["expected"] = golden_bins(src, DataFormat.Float16_b) if hist_en else [0] * 32
    info["exact_match"] = info["total"] == info["expected"]
    _report(info, f"bf16_three_{'on' if hist_en else 'off'}")
    assert info["sentinel_ok"]
    assert info["poison_control_ok"], "GPR poison writes did not land"


@pytest.mark.parametrize("hist_en", [True, False])
def test_hist_known_distribution_fp16(hist_en):
    """Same, in a 5-bit-exponent format where Exponent & 31 cannot alias."""
    src = tile_from_counts(_FP16_THREE, torch.float16)
    cfg = build_config(src, _FP16, HIST_PARAMS(hist_en=hist_en, clr_mode=2))
    info = run_and_decode(cfg)
    info["expected"] = golden_bins(src, DataFormat.Float16) if hist_en else [0] * 32
    info["exact_match"] = info["total"] == info["expected"]
    _report(info, f"fp16_three_{'on' if hist_en else 'off'}")
    assert info["sentinel_ok"]


def test_hist_alias_8to1():
    """Task 3: bin = Exponent & 31, so bf16 exponents 127 and 159 must land together."""
    # 2^0 -> exp 127 -> bin 31 ; 2^32 -> exp 159 -> bin 31 ; 0.0 -> exp 0 -> bin 0
    src = tile_from_counts(
        [(1.0, 100), (float(2.0**32), 100), (0.0, 824)], torch.bfloat16
    )
    cfg = build_config(src, _BF16, HIST_PARAMS(clr_mode=2))
    info = run_and_decode(cfg)
    info["expected"] = golden_bins(src, DataFormat.Float16_b)
    info["exact_match"] = info["total"] == info["expected"]
    _report(info, "alias")
    assert info["sentinel_ok"]


@pytest.mark.parametrize("num_packs", [1, 2])
def test_hist_saturation(num_packs):
    """Task 3: all 1024 datums share one exponent, so every packer sees >=256 of them.
    8-bit saturating -> 255 per packer; wrapping -> 0 or 4."""
    src = tile_from_counts([(1.0, TILE_DATUMS)], torch.bfloat16)
    cfg = build_config(src, _BF16, HIST_PARAMS(clr_mode=2, num_packs=num_packs))
    info = run_and_decode(cfg)
    info["expected_if_no_saturation"] = golden_bins(src, DataFormat.Float16_b)
    _report(info, f"saturate_np{num_packs}")
    assert info["sentinel_ok"]


@pytest.mark.parametrize("clr_between", [False, True])
def test_hist_accumulate_across_packs(clr_between):
    """Task 3: is CLREXPHIST needed between tiles? Two packs of the same tile."""
    src = tile_from_counts(_BF16_THREE, torch.bfloat16)
    cfg = build_config(
        src, _BF16, HIST_PARAMS(clr_mode=2, num_packs=2, clr_between=clr_between)
    )
    info = run_and_decode(cfg)
    one = golden_bins(src, DataFormat.Float16_b)
    info["expected_one_pack"] = one
    info["expected_two_packs"] = [min(255 * NUM_PACKERS, 2 * c) for c in one]
    _report(info, f"accum_clr{int(clr_between)}")
    assert info["sentinel_ok"]


def test_hist_no_clear():
    """Task 3: what the histogram holds when CLREXPHIST is never issued -- i.e. whether
    state survives from whatever ran on this core before."""
    src = tile_from_counts(_BF16_THREE, torch.bfloat16)
    cfg = build_config(src, _BF16, HIST_PARAMS(clr_mode=0))
    info = run_and_decode(cfg)
    info["expected_if_clean"] = golden_bins(src, DataFormat.Float16_b)
    _report(info, "noclear")
    assert info["sentinel_ok"]


@pytest.mark.parametrize("clr_mode", [1, 2])
def test_hist_clear_from_thread(clr_mode):
    """CLREXPHIST is a MATH-resource instruction; can the PACK thread issue it (1),
    and does issuing it from the MATH thread (2) order correctly against the PACRs?"""
    src = tile_from_counts(_BF16_THREE, torch.bfloat16)
    cfg = build_config(src, _BF16, HIST_PARAMS(clr_mode=clr_mode))
    info = run_and_decode(cfg)
    info["expected"] = golden_bins(src, DataFormat.Float16_b)
    info["exact_match"] = info["total"] == info["expected"]
    _report(info, f"clrmode{clr_mode}")
    assert info["sentinel_ok"]


@pytest.mark.parametrize("which", ["pack", "unpack", "math"])
def test_hist_enable_thread(which):
    """The WH functional model ORs ENABLE_ACC_STATS_Enable across all three threads.
    Verify that on BH by enabling it from exactly one thread at a time."""
    src = tile_from_counts(_BF16_THREE, torch.bfloat16)
    p = HIST_PARAMS(
        hist_en=(which == "pack"),
        hist_en_unpack=(which == "unpack"),
        hist_en_math=(which == "math"),
        clr_mode=2,
    )
    cfg = build_config(src, _BF16, p)
    info = run_and_decode(cfg)
    info["expected"] = golden_bins(src, DataFormat.Float16_b)
    info["exact_match"] = info["total"] == info["expected"]
    _report(info, f"enable_{which}")
    assert info["sentinel_ok"]


@pytest.mark.parametrize("top_exp_k", [0, 5, 20])
def test_hist_max_exponent(top_exp_k):
    """Mode 9 is documented as packer 0's running max exponent. One datum carries the
    maximum; everything else is 1.0 (bf16 exponent 127)."""
    src = tile_from_counts(
        [(float(2.0**top_exp_k), 1), (1.0, TILE_DATUMS - 1)], torch.bfloat16
    )
    cfg = build_config(src, _BF16, HIST_PARAMS(clr_mode=2))
    info = run_and_decode(cfg)
    info["expected_max_exp"] = max(raw_exponents(src, DataFormat.Float16_b))
    _report(info, f"maxexp_k{top_exp_k}")
    print(f"  expected_max_exp       {info['expected_max_exp']}")
    assert info["sentinel_ok"]
