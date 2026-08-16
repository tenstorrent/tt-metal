# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Does the Blackhole packer's zero-compression path work on 32-bit datums?

test_pack_zero_compress.py proved zero-run elision works for 16-bit (Float16_b)
datums and decodes bit-exactly. The arrangement that actually competes with
_topk_xl_merge_ needs it for 32-bit datums:

    SFPU (1.003 cyc/vector, measured) writes the fused
    [bf16 value (high 16) | u16 index (low 16)] INT32 sort key for survivors and
    0 for everyone else; the packer elides the zeroed words. Survivors then carry
    their own indices, so there is no serial run-length position decode on the
    RISC-V, and the compare is an SFPGT -- which, unlike the packer's
    MIN_THRESHOLD_RELU, handles negative thresholds
    (WormholeB0/.../Packers/ReLU.md:41 makes a signbit threshold undefined) and
    does not reinterpret the fused word as a float.

The docs say compression elides "datums which are zero" and never state the
datum format; nothing says anything about int32. If the zero test is
float-format-only, the whole arrangement collapses. This file tests it directly,
by reusing sources/pack_zero_compress_test.cpp unchanged with Int32 formats.

Two things are checked:
  1. Does the write footprint shrink at all (elision happened)?
  2. Does the compressed stream decode back to the source tile, with the
     32-datum group re-sized to 32*4 B of data + 16 B of four-bit counters?
"""

import json
import os
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
    generate_input_dim,
)
from test_pack_zero_compress import COMPRESS_PARAMS, DIAG_TILE, RES_TILES, SENTINEL

TILE_DATUMS = 1024
NUM_COMPRESSION_ROWS = 16

REPORT_DIR = Path(
    os.environ.get(
        "PZC_REPORT_DIR",
        "/tmp/claude-1000/-home-nachiket-tt-metal/d24a23b1-86c2-4faa-aead-36085a95861d/scratchpad",
    )
)


def make_fused_pattern(num_survivors, seed=7):
    """Flat 1024-word int32 tile in L1/datum order.

    Survivors hold [bf16 value | u16 (index+1)]; everyone else holds 0. The +1 on
    the index means a survivor sitting at position 0 with a tiny value can still
    never produce an all-zero word, which the packer would elide as a hole.
    """
    t = torch.zeros(TILE_DATUMS, dtype=torch.int32)
    if num_survivors <= 0:
        idx = []
    elif num_survivors >= TILE_DATUMS:
        idx = list(range(TILE_DATUMS))
    else:
        g = torch.Generator().manual_seed(seed)
        idx = sorted(torch.randperm(TILE_DATUMS, generator=g)[:num_survivors].tolist())

    for k, i in enumerate(idx):
        val_bits = int(
            torch.tensor([float((k % 250) + 1)], dtype=torch.bfloat16)
            .view(torch.uint16)
            .item()
        )
        t[i] = (val_bits << 16) | ((i + 1) & 0xFFFF)
    return t, idx


def build_config(num_survivors, compress_en):
    src_A, idx = make_fused_pattern(num_survivors)
    src_B = torch.zeros(TILE_DATUMS, dtype=torch.int32)
    formats = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)
    rss = 4 if compress_en else 0

    configuration = TestConfig(
        "sources/pack_zero_compress_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            TILIZE(Tilize.No),
            DEST_SYNC(DestSync.Half),
            COMPRESS_PARAMS(compress_en, rss, DIAG_TILE, 0, False, False),
        ],
        runtimes=[
            RELU_CONFIG(0),
            NUM_FACES(num_faces=4),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            DataFormat.Int32,
            src_B,
            DataFormat.Int32,
            DataFormat.Int32,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=RES_TILES,
        ),
        # 32-bit datums live in a 32-bit Dest, filled by the unpacker directly
        # (the same path test_topk_xl.py uses for its Int32 input).
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
    )
    return configuration, src_A, idx, rss


def run_and_dump(configuration):
    configuration.run()
    stim = configuration.variant_stimuli
    loc = TestConfig.TENSIX_LOCATION
    stim.clear_result_buffer(loc, fill_byte=SENTINEL)
    configuration.run_elf_files()
    configuration.wait_for_tensix_operations_finished()
    return bytes(stim.collect_raw_result_bytes(loc))


GROUP_DATUMS = 32
DATUM_BYTES = 4
GROUP_BYTES = GROUP_DATUMS * DATUM_BYTES + 16  # datums, then 32 four-bit counters


def decode_compressed32(body, rss_units, num_rows):
    """Same layout as the 16-bit decoder in test_pack_zero_compress.py with the
    datum widened to 4 B. Counter semantics follow the Blackhole observation
    (zeroes PRECEDING the datum), not the Wormhole doc."""
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

    out = []
    for r in range(num_rows):
        for k in range(rsi[r], rsi[r + 1]):
            z = counter(k)
            out.extend([0] * z)
            out.append(datum(k))

    total_aug = rsi[num_rows]
    return {
        "rsi": rsi,
        "total_augmented_datums": total_aug,
        "decoded": out,
    }


@pytest.mark.parametrize("num_survivors", [0, 32, 128, 1024])
@pytest.mark.parametrize("compress_en", [False, True])
def test_pack_compress_int32(num_survivors, compress_en):
    configuration, src_A, idx, rss = build_config(num_survivors, compress_en)
    raw = run_and_dump(configuration)
    tile_bytes = configuration.variant_stimuli.buf_res_tile_size

    diag_off = DIAG_TILE * tile_bytes
    diag = [
        int.from_bytes(raw[diag_off + 4 * i : diag_off + 4 * i + 4], "little")
        for i in range(16)
    ]
    body = raw[:diag_off]
    written = [i for i, b in enumerate(body) if b != SENTINEL]

    src_words = [x & 0xFFFFFFFF for x in src_A.tolist()]

    info = {
        "num_survivors": num_survivors,
        "compress_en": compress_en,
        "sentinel_ok": diag[0] == 0xC0DEBA5E and diag[12] == 0xC0DEE0D1,
        "packed_size_t2_units": diag[3],
        "packed_size_t2_bytes": diag[3] * 16,
        "num_bytes_touched": len(written),
        "first_touched": written[0] if written else None,
        "last_touched": written[-1] if written else None,
        "cfg_word2": f"0x{diag[9]:08X}",
        "cfg_word3": f"0x{diag[10]:08X}",
        "tile_bytes": tile_bytes,
    }

    if compress_en:
        dec = decode_compressed32(body, rss, NUM_COMPRESSION_ROWS)
        info["rsi"] = dec["rsi"]
        info["total_augmented_datums"] = dec["total_augmented_datums"]
        info["decoded_len"] = len(dec["decoded"])
        info["decode_exact"] = dec["decoded"] == src_words
        if not info["decode_exact"]:
            mism = [
                (
                    i,
                    f"0x{src_words[i]:08X}",
                    (f"0x{dec['decoded'][i]:08X}" if i < len(dec["decoded"]) else None),
                )
                for i in range(TILE_DATUMS)
                if i >= len(dec["decoded"]) or dec["decoded"][i] != src_words[i]
            ]
            info["num_decode_mismatches"] = len(mism)
            info["decode_mismatches"] = mism[:16]
    else:
        got = [
            int.from_bytes(body[4 * i : 4 * i + 4], "little")
            for i in range(TILE_DATUMS)
        ]
        info["decode_exact"] = got == src_words
        info["decoded_len"] = TILE_DATUMS

    info["body_hex"] = body[: (written[-1] + 16 if written else 64)].hex()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"pzc32_{num_survivors}_{'on' if compress_en else 'off'}"
    (REPORT_DIR / f"{name}.json").write_text(json.dumps(info, indent=2))
    print(f"\n==== {name} ====")
    for k in (
        "sentinel_ok",
        "num_bytes_touched",
        "first_touched",
        "last_touched",
        "packed_size_t2_bytes",
        "total_augmented_datums",
        "decoded_len",
        "decode_exact",
        "num_decode_mismatches",
        "cfg_word2",
        "cfg_word3",
        "rsi",
    ):
        if k in info:
            print(f"  {k:28s} {info[k]}")

    assert info["sentinel_ok"], f"pack kernel did not reach the metadata dump: {info}"
