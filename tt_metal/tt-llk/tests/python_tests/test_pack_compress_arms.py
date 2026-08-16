# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Did each perf arm of ``perf_pack_zero_compress.py`` actually compress?

WHY THIS FILE EXISTS
--------------------
The pack cost sweep reported

    plain   bf16  uncompressed 0.783  compressed 1.189  (delta 0.406)
    fused32 int32 uncompressed 1.241  compressed 1.648  (delta 0.408)
    relu    bf16  uncompressed 0.784  compressed 0.786  (delta 0.002)

A ~0.002 delta on the relu arm is not "compression is free" -- it is what a build
whose compression-enable silently failed measures. The two config writes live in
``THCON_SEC0_REG1``, the same word the packer ReLU config touches, so a
write-ordering or RMW-clobber failure is a live hypothesis and CANNOT be settled
from timings: a no-op arm and a free arm are numerically identical.

``PackerTileSize`` (``RISCV_TDMA_REG_PACKED_SIZE + 0x080``, 16 B units) is the
packer's own report of how many bytes it emitted for the last tile. It is
data, not inference. This file runs each perf arm's EXACT stimulus through
``sources/pack_zero_compress_test.cpp`` (which already dumps that register) and
reports the emitted byte count with compression off and on.

Expected, if compression engaged:
  * a sparse tile SHRINKS a lot (32 survivors out of 1024).
  * a dense tile GROWS (2048 -> 2624 B for bf16): 32 four-bit run counters per
    32 datums, plus the row-start index array. No zeroes to elide.
  * an arm where nothing changes did not compress.

FOURTH ARM (relu32) -- the interesting one, not in the perf sweep.
``Packers/ReLU.md`` says the ReLU stage runs on the value AFTER early format
conversion, in FP32, and that an INTEGER intermediate format must use
ApplyRelu 0 or 1. But the fused Top-K sort key -- ``[bf16 value (high 16) |
u16 index (low 16)]`` -- IS a well-formed FP32 number whose ordering by FP32
compare is exactly "by value, ties broken by index". So if the tile is carried
as Float32 rather than Int32, MIN_THRESHOLD_RELU can do the whole threshold
filter in the packer with ZERO SFPU instructions, and the fused word still
survives bit-exactly. The only real restriction is that the threshold cannot be
negative (``signbit(Threshold)`` is UndefinedBehavior); negative DATA is fine --
it simply falls below any positive threshold.
"""

import json
import os
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
from test_pack_zero_compress import COMPRESS_PARAMS, DIAG_TILE, RES_TILES, SENTINEL

TILE_DATUMS = 1024
ROW_START_SECTION_SIZE = 4

REPORT_DIR = Path(
    os.environ.get(
        "PZC_REPORT_DIR",
        "/tmp/claude-1000/-home-nachiket-tt-metal/d24a23b1-86c2-4faa-aead-36085a95861d/scratchpad",
    )
)


def _scatter(num_survivors, seed=7):
    """Same positions the perf sweep used, so the byte counts are comparable."""
    if num_survivors <= 0:
        return []
    if num_survivors >= TILE_DATUMS:
        return list(range(TILE_DATUMS))
    g = torch.Generator().manual_seed(seed)
    return sorted(torch.randperm(TILE_DATUMS, generator=g)[:num_survivors].tolist())


def _bf16_ladder(n=TILE_DATUMS):
    vals = []
    for e in range(8):
        for m in range(128):
            vals.append((2.0**e) * (1.0 + m / 128.0))
    return vals[:n]


def _fused_words(num_survivors):
    """[bf16 value | u16 (index+1)] for survivors, 0 elsewhere, as int32 bits."""
    words = [0] * TILE_DATUMS
    for k, i in enumerate(_scatter(num_survivors)):
        val_bits = int(
            torch.tensor([float((k % 250) + 1)], dtype=torch.bfloat16)
            .view(torch.uint16)
            .item()
        )
        words[i] = (val_bits << 16) | ((i + 1) & 0xFFFF)
    return words


def stimulus(arm, density):
    """(src tensor, relu_config, formats, dest_acc, unpack_to_dest)."""
    bf16 = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    i32 = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)
    f32 = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)

    if arm == "plain":
        t = torch.zeros(TILE_DATUMS, dtype=torch.bfloat16)
        for k, i in enumerate(_scatter(density)):
            t[i] = float((k % 250) + 1)
        return t, 0, bf16, DestAccumulation.No, False

    if arm == "relu":
        # EXACTLY the perf sweep's relu stimulus: dense distinct ladder, packer
        # zeroes everything at or below the threshold.
        vals = _bf16_ladder()
        g = torch.Generator().manual_seed(11)
        perm = torch.randperm(TILE_DATUMS, generator=g).tolist()
        t = torch.tensor(
            [vals[perm[i]] for i in range(TILE_DATUMS)], dtype=torch.bfloat16
        )
        threshold = vals[TILE_DATUMS - density]
        relu = PackGolden.generate_relu_config(
            PackerReluType.MinThresholdRelu, threshold, DataFormat.Float16_b
        )
        return t, relu, bf16, DestAccumulation.No, False

    if arm == "fused32":
        t = torch.tensor(_fused_words(density), dtype=torch.int32)
        return t, 0, i32, DestAccumulation.Yes, True

    if arm == "relu32":
        # Dense fused FP32 sort keys; the PACKER does the thresholding.
        # value = 1..1024 scaled so the k-th largest is a known positive bf16.
        vals = _bf16_ladder()
        g = torch.Generator().manual_seed(11)
        perm = torch.randperm(TILE_DATUMS, generator=g).tolist()
        words = []
        for i in range(TILE_DATUMS):
            vb = int(
                torch.tensor([vals[perm[i]]], dtype=torch.bfloat16)
                .view(torch.uint16)
                .item()
            )
            words.append((vb << 16) | ((i + 1) & 0xFFFF))
        t = torch.tensor(words, dtype=torch.int32).view(torch.float32)
        threshold = vals[TILE_DATUMS - density]
        relu = PackGolden.generate_relu_config(
            PackerReluType.MinThresholdRelu, threshold, DataFormat.Float32
        )
        return t, relu, f32, DestAccumulation.Yes, True

    raise ValueError(arm)


def build_config(arm, density, compress_en):
    src_A, relu_config, formats, dest_acc, unpack_to_dest = stimulus(arm, density)
    src_B = torch.zeros_like(src_A)
    rss = ROW_START_SECTION_SIZE if compress_en else 0

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
            RELU_CONFIG(relu_config),
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
        dest_acc=dest_acc,
        unpack_to_dest=unpack_to_dest,
    )
    return configuration, relu_config


def run_and_dump(configuration):
    configuration.run()
    stim = configuration.variant_stimuli
    loc = TestConfig.TENSIX_LOCATION
    stim.clear_result_buffer(loc, fill_byte=SENTINEL)
    configuration.run_elf_files()
    configuration.wait_for_tensix_operations_finished()
    return bytes(stim.collect_raw_result_bytes(loc))


@pytest.mark.parametrize("density", [32, 1024])
@pytest.mark.parametrize("compress_en", [False, True])
@pytest.mark.parametrize("arm", ["plain", "relu", "fused32", "relu32"])
@blackhole_only
def test_pack_compress_arm_emitted_size(arm, density, compress_en):
    if arm in ("relu", "relu32") and density >= TILE_DATUMS:
        # MIN_THRESHOLD_RELU needs a strictly positive threshold, so "everything
        # survives" is not expressible on the relu arms.
        pytest.skip("relu arms cannot express density == TILE_DATUMS")

    configuration, relu_config = build_config(arm, density, compress_en)
    raw = run_and_dump(configuration)
    tile_bytes = configuration.variant_stimuli.buf_res_tile_size

    diag_off = DIAG_TILE * tile_bytes
    diag = [
        int.from_bytes(raw[diag_off + 4 * i : diag_off + 4 * i + 4], "little")
        for i in range(16)
    ]
    body = raw[:diag_off]
    written = [i for i, b in enumerate(body) if b != SENTINEL]

    info = {
        "arm": arm,
        "density": density,
        "compress_en": compress_en,
        "relu_config": f"0x{relu_config:08X}",
        "sentinel_ok": diag[0] == 0xC0DEBA5E and diag[12] == 0xC0DEE0D1,
        "packed_size_bytes": diag[3] * 16,
        "num_bytes_touched": len(written),
        "last_touched": written[-1] if written else None,
        "cfg_word2": f"0x{diag[9]:08X}",
        "cfg_word3": f"0x{diag[10]:08X}",
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"arm_{arm}_{density}_{'on' if compress_en else 'off'}"
    (REPORT_DIR / f"{name}.json").write_text(json.dumps(info, indent=2))
    print(f"\n==== {name} ====")
    for k, v in info.items():
        print(f"  {k:22s} {v}")

    assert info["sentinel_ok"], f"pack kernel did not reach the metadata dump: {info}"
