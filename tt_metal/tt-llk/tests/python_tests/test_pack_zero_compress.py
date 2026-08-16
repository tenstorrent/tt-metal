# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Experimental probe: does the Blackhole packer's zero-compression path work?

Packs one tile of a known sparse pattern with THCON_SEC0_REG1_Disable_zero_compress
cleared, and compares the bytes actually written to L1 against the uncompressed
baseline. The result buffer is pre-filled with a 0xA5 sentinel so the write extent is
directly observable, and the packer's sideband metadata registers (PackerTileSize,
AllZeroFlags) are dumped by the kernel into a scratch tile.

The decoder below implements the on-disk format documented for Wormhole
(tt-isa-documentation .../Packers/Compression.md) and checks that it reproduces the
source tile bit-exactly, which is what makes "compression happened" into "compression
is usable".

Nothing in the tree enables compression, and BlackholeA0 has no packer documentation,
so everything here is a measurement, not a check against a known-good golden.
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
SENTINEL = 0xA5

# The Default pack MOP issues num_faces * (face_r_dim / 4) = 16 PACRs per 32x32 tile,
# and (with Concat clear) every PACR starts a new compression row.
NUM_COMPRESSION_ROWS = 16

REPORT_DIR = Path(
    os.environ.get(
        "PZC_REPORT_DIR",
        "/tmp/claude-1000/-home-nachiket-tt-metal/d24a23b1-86c2-4faa-aead-36085a95861d/scratchpad",
    )
)


@dataclass
class COMPRESS_PARAMS(TemplateParameter):
    compress_en: bool = False
    row_start_section_size: int = 0
    diag_tile_index: int = DIAG_TILE
    downsample_mask: int = 0
    enable_out_fifo: bool = False
    concat_rows: bool = False

    def convert_to_cpp(self) -> str:
        return "\n".join(
            [
                f"constexpr bool COMPRESS_EN = {str(self.compress_en).lower()};",
                f"constexpr std::uint32_t ROW_START_SECTION_SIZE = {self.row_start_section_size};",
                f"constexpr std::uint32_t DIAG_TILE_INDEX = {self.diag_tile_index};",
                f"constexpr std::uint32_t DOWNSAMPLE_MASK = {self.downsample_mask};",
                f"constexpr bool ENABLE_OUT_FIFO = {str(self.enable_out_fifo).lower()};",
                f"constexpr bool CONCAT_ROWS = {str(self.concat_rows).lower()};",
            ]
        )


def make_pattern(name: str):
    """Flat 1024-element tile in L1/datum order. Survivors carry distinct values
    (1, 2, 3, ...) so the packed stream can be decoded back to source positions."""
    t = torch.zeros(TILE_DATUMS, dtype=torch.bfloat16)
    if name == "allzero":
        idx = []
    elif name == "dense":
        idx = list(range(TILE_DATUMS))
    elif name == "stride16":  # zero runs of exactly 15 (4-bit counter maximum)
        idx = list(range(0, TILE_DATUMS, 16))
    elif name == "stride17":  # zero runs of 16 -- one past the 4-bit counter maximum
        idx = list(range(0, TILE_DATUMS, 17))
    elif name == "stride64":  # zero runs of 63
        idx = list(range(0, TILE_DATUMS, 64))
    elif name == "front32":  # 32 survivors up front, then a 992-long zero run
        idx = list(range(32))
    elif name == "topk32":  # 32 survivors scattered, the realistic top-k case
        g = torch.Generator().manual_seed(7)
        idx = sorted(torch.randperm(TILE_DATUMS, generator=g)[:32].tolist())
    else:
        raise ValueError(name)

    for k, i in enumerate(idx):
        # 1..250 is exact in bfloat16, so the value identifies the survivor ordinal.
        t[i] = float((k % 250) + 1)
    return t, idx


def build_config(pattern, compress_en, rss, downsample_mask=0, out_fifo=False):
    src_A, idx = make_pattern(pattern)
    src_B = torch.zeros(TILE_DATUMS, dtype=torch.bfloat16)
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)

    configuration = TestConfig(
        "sources/pack_zero_compress_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            TILIZE(Tilize.No),
            DEST_SYNC(DestSync.Half),
            COMPRESS_PARAMS(compress_en, rss, DIAG_TILE, downsample_mask, out_fifo),
        ],
        runtimes=[
            RELU_CONFIG(0),
            NUM_FACES(num_faces=4),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            DataFormat.Float16_b,
            src_B,
            DataFormat.Float16_b,
            DataFormat.Float16_b,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=RES_TILES,
        ),
        dest_acc=DestAccumulation.No,
    )
    return configuration, src_A, idx


def run_and_dump(configuration):
    """Run once through the harness, then re-run over a fresh 0xA5 sentinel so the
    exact set of bytes the packer wrote is observable."""
    configuration.run()

    stim = configuration.variant_stimuli
    loc = TestConfig.TENSIX_LOCATION
    stim.clear_result_buffer(loc, fill_byte=SENTINEL)
    configuration.run_elf_files()
    configuration.wait_for_tensix_operations_finished()
    return bytes(stim.collect_raw_result_bytes(loc))


# ---------------------------------------------------------------------------
# Decoder for the documented compressed layout (16-bit datums, non-BFP)
# ---------------------------------------------------------------------------

GROUP_DATUMS = 32
DATUM_BYTES = 2
GROUP_BYTES = GROUP_DATUMS * DATUM_BYTES + 16  # datums, then 32 four-bit counters


def decode_compressed(body, rss_units, num_rows):
    data_start = rss_units * 16
    rsi = [
        int.from_bytes(body[2 * i : 2 * i + 2], "little") for i in range(num_rows + 1)
    ]

    def group_base(g):
        return data_start + g * GROUP_BYTES

    def datum(k):
        g, j = divmod(k, GROUP_DATUMS)
        off = group_base(g) + j * DATUM_BYTES
        return int.from_bytes(body[off : off + DATUM_BYTES], "little")

    def counter(k):
        g, j = divmod(k, GROUP_DATUMS)
        b = body[group_base(g) + GROUP_DATUMS * DATUM_BYTES + j // 2]
        return (b >> (4 * (j % 2))) & 0xF

    # Observed on Blackhole: the four-bit counter attached to an augmented datum is the
    # number of zeroes that precede it, not (as the Wormhole page says) the number that
    # follow it. See ZERO_RUN_SEMANTICS in the report.
    out = []
    per_row = []
    for r in range(num_rows):
        n = 0
        for k in range(rsi[r], rsi[r + 1]):
            z = counter(k)
            out.extend([0] * z)
            out.append(datum(k))
            n += z + 1
        per_row.append(n)

    total_aug = rsi[num_rows]
    footprint = (
        data_start + ((total_aug + GROUP_DATUMS - 1) // GROUP_DATUMS) * GROUP_BYTES
    )
    return {
        "rsi": rsi,
        "total_augmented_datums": total_aug,
        "datums_per_row": per_row,
        "predicted_footprint_bytes": footprint,
        "decoded": out,
    }


def summarize(raw, tile_bytes, idx, src_A, rss, compress_en):
    diag_off = DIAG_TILE * tile_bytes
    diag = [
        int.from_bytes(raw[diag_off + 4 * i : diag_off + 4 * i + 4], "little")
        for i in range(16)
    ]
    body = raw[:diag_off]
    written = [i for i, b in enumerate(body) if b != SENTINEL]

    info = {
        "diag": [f"0x{d:08X}" for d in diag],
        "packed_size_t0_units": diag[1],
        "packed_size_t1_units": diag[2],
        "packed_size_t2_units": diag[3],
        "packed_size_t2_bytes": diag[3] * 16,
        "all_zero_flags": f"0x{diag[4]:08X}",
        "acc_packed_size_t2": diag[5],
        "fifo_status": f"0x{diag[6]:08X}",
        "fifo_tile_size": f"0x{diag[13]:08X}",
        "fifo_zeromask": f"0x{diag[14]:08X}",
        "cfg_word0": f"0x{diag[7]:08X}",
        "cfg_word1": f"0x{diag[8]:08X}",
        "cfg_word2": f"0x{diag[9]:08X}",
        "cfg_word3": f"0x{diag[10]:08X}",
        "res_base": f"0x{diag[11]:08X}",
        "kernel_num_faces": diag[15],
        "sentinel_ok": diag[0] == 0xC0DEBA5E and diag[12] == 0xC0DEE0D1,
        "num_bytes_touched": len(written),
        "first_touched": written[0] if written else None,
        "last_touched": written[-1] if written else None,
        "num_nonzero_src": len(idx),
        "src_nonzero_positions": idx[:64],
    }

    src_bits = src_A.view(torch.uint16).tolist()
    info["body_hex"] = body[: (written[-1] + 16 if written else 64)].hex()

    if compress_en:
        dec = decode_compressed(body, rss, NUM_COMPRESSION_ROWS)
        decoded = dec["decoded"]
        info["rsi"] = dec["rsi"]
        info["total_augmented_datums"] = dec["total_augmented_datums"]
        info["datums_per_row"] = dec["datums_per_row"]
        info["predicted_footprint_bytes"] = dec["predicted_footprint_bytes"]
        info["decoded_len"] = len(decoded)
        info["decode_exact"] = decoded == src_bits
        if not info["decode_exact"]:
            mism = [
                (i, src_bits[i], decoded[i] if i < len(decoded) else None)
                for i in range(TILE_DATUMS)
                if i >= len(decoded) or decoded[i] != src_bits[i]
            ]
            info["decode_mismatches"] = mism[:20]
            info["num_decode_mismatches"] = len(mism)
    else:
        got = [
            int.from_bytes(body[2 * i : 2 * i + 2], "little")
            for i in range(TILE_DATUMS)
        ]
        info["decode_exact"] = got == src_bits
        info["decoded_len"] = TILE_DATUMS

    return info


PATTERNS = ["allzero", "topk32", "front32", "stride64", "stride17", "stride16", "dense"]


def _report(info, name):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"pzc_{name}.json").write_text(json.dumps(info, indent=2))
    print(f"\n==== {name} ====")
    for k in (
        "sentinel_ok",
        "num_nonzero_src",
        "num_bytes_touched",
        "first_touched",
        "last_touched",
        "packed_size_t2_units",
        "packed_size_t2_bytes",
        "predicted_footprint_bytes",
        "total_augmented_datums",
        "datums_per_row",
        "decoded_len",
        "decode_exact",
        "num_decode_mismatches",
        "all_zero_flags",
        "fifo_status",
        "fifo_tile_size",
        "fifo_zeromask",
        "kernel_num_faces",
        "cfg_word0",
        "cfg_word2",
        "cfg_word3",
        "rsi",
    ):
        if k in info:
            print(f"  {k:28s} {info[k]}")


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("compress_en", [False, True])
def test_pack_zero_compress(pattern, compress_en):
    rss = 4 if compress_en else 0  # 64 B reserved for the row-start-index array
    configuration, src_A, idx = build_config(pattern, compress_en, rss)
    raw = run_and_dump(configuration)
    tile_bytes = configuration.variant_stimuli.buf_res_tile_size

    info = summarize(raw, tile_bytes, idx, src_A, rss, compress_en)
    info.update(
        pattern=pattern,
        compress_en=compress_en,
        row_start_section_size=rss,
        tile_bytes=tile_bytes,
    )
    _report(info, f"{pattern}_{'on' if compress_en else 'off'}")

    assert info["sentinel_ok"], f"pack kernel did not reach the metadata dump: {info}"
    assert info["decode_exact"], "packed stream does not decode back to the source tile"


@pytest.mark.parametrize("pattern", ["allzero", "topk32", "dense"])
def test_pack_zero_compress_out_fifo(pattern):
    """Same probe with Enable_out_fifo set, to see whether the metadata FIFO
    (packed size + AllZeroFlags per compression row) becomes readable."""
    rss = 4
    configuration, src_A, idx = build_config(pattern, True, rss, out_fifo=True)
    raw = run_and_dump(configuration)
    tile_bytes = configuration.variant_stimuli.buf_res_tile_size

    info = summarize(raw, tile_bytes, idx, src_A, rss, True)
    info.update(pattern=pattern, compress_en=True, out_fifo=True, tile_bytes=tile_bytes)
    _report(info, f"{pattern}_fifo")

    assert info["sentinel_ok"], f"pack kernel did not reach the metadata dump: {info}"


@pytest.mark.parametrize("rss", [1, 2, 3, 4, 8])
def test_pack_zero_compress_rss_sweep(rss):
    """How much room the row-start-index array actually needs, and what happens when
    it is under-provisioned (the data stream starts at base + rss*16 regardless)."""
    configuration, src_A, idx = build_config("topk32", True, rss)
    raw = run_and_dump(configuration)
    tile_bytes = configuration.variant_stimuli.buf_res_tile_size
    info = summarize(raw, tile_bytes, idx, src_A, rss, True)
    info.update(pattern="topk32", compress_en=True, row_start_section_size=rss)
    _report(info, f"rss{rss}")
    assert info["sentinel_ok"]


@pytest.mark.parametrize("mask", [0x5555, 0x0001, 0x00FF])
def test_pack_downsample_mask(mask):
    """Secondary probe: does Downsample_mask perform a real vector-compress on BH?
    Compression stays off so the output is a plain contiguous datum stream."""
    configuration, src_A, idx = build_config("dense", False, 0, downsample_mask=mask)
    raw = run_and_dump(configuration)
    tile_bytes = configuration.variant_stimuli.buf_res_tile_size
    info = summarize(raw, tile_bytes, idx, src_A, 0, False)
    info.update(pattern="dense", downsample_mask=f"0x{mask:04X}")
    info["decode_exact"] = None  # not meaningful here
    _report(info, f"downsample_{mask:04X}")
    print(
        f"  first 32 u16: {[int.from_bytes(raw[2*i:2*i+2],'little') for i in range(32)]}"
    )
    assert info["sentinel_ok"]


# ---------------------------------------------------------------------------
# The payoff: packer MIN_THRESHOLD_RELU (zero the sub-threshold datums) combined
# with zero-compression (elide the zeroes) -- filter + compaction in one pack.
# ---------------------------------------------------------------------------


def _bf16_ladder(n=TILE_DATUMS):
    """n distinct, exactly-representable bfloat16 values in [1, 256)."""
    vals = []
    for e in range(8):
        for m in range(128):
            vals.append((2.0**e) * (1.0 + m / 128.0))
    return vals[:n]


def build_threshold_config(num_survivors, compress_en):
    vals = _bf16_ladder()
    g = torch.Generator().manual_seed(11)
    perm = torch.randperm(TILE_DATUMS, generator=g).tolist()
    # position i holds vals[perm[i]] -- every value distinct, order scrambled
    data = [vals[perm[i]] for i in range(TILE_DATUMS)]
    src_A = torch.tensor(data, dtype=torch.bfloat16)
    threshold = vals[TILE_DATUMS - num_survivors]

    from helpers.golden_generators import PackGolden
    from helpers.llk_params import PackerReluType

    relu_config = PackGolden.generate_relu_config(
        PackerReluType.MinThresholdRelu, threshold, DataFormat.Float16_b
    )

    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    configuration = TestConfig(
        "sources/pack_zero_compress_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            TILIZE(Tilize.No),
            DEST_SYNC(DestSync.Half),
            COMPRESS_PARAMS(compress_en, 4 if compress_en else 0, DIAG_TILE, 0, False),
        ],
        runtimes=[RELU_CONFIG(relu_config), NUM_FACES(num_faces=4)],
        variant_stimuli=StimuliConfig(
            src_A,
            DataFormat.Float16_b,
            torch.zeros(TILE_DATUMS, dtype=torch.bfloat16),
            DataFormat.Float16_b,
            DataFormat.Float16_b,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=RES_TILES,
        ),
        dest_acc=DestAccumulation.No,
    )
    return configuration, src_A, threshold


@pytest.mark.parametrize("num_survivors", [32, 64, 128])
@pytest.mark.parametrize("compress_en", [False, True])
def test_pack_threshold_compact(num_survivors, compress_en):
    configuration, src_A, threshold = build_threshold_config(num_survivors, compress_en)
    raw = run_and_dump(configuration)
    tile_bytes = configuration.variant_stimuli.buf_res_tile_size
    rss = 4 if compress_en else 0

    info = summarize(raw, tile_bytes, [], src_A, rss, compress_en)
    src_bits = src_A.view(torch.uint16).tolist()
    golden_lt = [b if v >= threshold else 0 for b, v in zip(src_bits, src_A.tolist())]
    golden_le = [b if v > threshold else 0 for b, v in zip(src_bits, src_A.tolist())]

    if compress_en:
        dec = decode_compressed(
            raw[: DIAG_TILE * tile_bytes], rss, NUM_COMPRESSION_ROWS
        )
        got = dec["decoded"]
        info["total_augmented_datums"] = dec["total_augmented_datums"]
        info["rsi"] = dec["rsi"]
    else:
        got = [
            int.from_bytes(raw[2 * i : 2 * i + 2], "little") for i in range(TILE_DATUMS)
        ]
    info["threshold"] = threshold
    info["num_survivors_requested"] = num_survivors
    info["num_survivors_observed"] = sum(1 for x in got if x != 0)
    info["matches_ge_threshold"] = got == golden_lt
    info["matches_gt_threshold"] = got == golden_le
    info["compress_en"] = compress_en
    _report(info, f"thresh{num_survivors}_{'on' if compress_en else 'off'}")
    print(f"  threshold                    {threshold}")
    print(f"  num_survivors_requested      {num_survivors}")
    print(f"  num_survivors_observed       {info['num_survivors_observed']}")
    print(f"  matches_ge_threshold         {info['matches_ge_threshold']}")
    print(f"  matches_gt_threshold         {info['matches_gt_threshold']}")

    assert info["sentinel_ok"]
    assert info["matches_ge_threshold"] or info["matches_gt_threshold"]


@pytest.mark.parametrize("pattern", ["allzero", "topk32", "front32", "dense"])
def test_pack_zero_compress_concat(pattern):
    """One compression row for the whole tile (Concat set on every PACR but the last),
    to measure how much of the augmented-datum count is per-row overhead."""
    rss = 1
    src_A, idx = make_pattern(pattern)
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    configuration = TestConfig(
        "sources/pack_zero_compress_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            TILIZE(Tilize.No),
            DEST_SYNC(DestSync.Half),
            COMPRESS_PARAMS(True, rss, DIAG_TILE, 0, False, True),
        ],
        runtimes=[RELU_CONFIG(0), NUM_FACES(num_faces=4)],
        variant_stimuli=StimuliConfig(
            src_A,
            DataFormat.Float16_b,
            torch.zeros(TILE_DATUMS, dtype=torch.bfloat16),
            DataFormat.Float16_b,
            DataFormat.Float16_b,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=RES_TILES,
        ),
        dest_acc=DestAccumulation.No,
    )
    raw = run_and_dump(configuration)
    tile_bytes = configuration.variant_stimuli.buf_res_tile_size
    # compress_en=False here only tells summarize() to skip its 16-row decode; the
    # single-row decode is done explicitly below.
    info = summarize(raw, tile_bytes, idx, src_A, rss, False)
    dec = decode_compressed(raw[: DIAG_TILE * tile_bytes], rss, 1)
    info["concat_rsi"] = dec["rsi"]
    info["concat_total_augmented"] = dec["total_augmented_datums"]
    info["concat_decode_exact"] = dec["decoded"] == src_A.view(torch.uint16).tolist()
    info["concat_decoded_len"] = len(dec["decoded"])
    _report(info, f"concat_{pattern}")
    for k in (
        "concat_rsi",
        "concat_total_augmented",
        "concat_decoded_len",
        "concat_decode_exact",
    ):
        print(f"  {k:28s} {info[k]}")
    assert info["sentinel_ok"]
