# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host proof of the frame arithmetic: ``frames.locate`` (Python) and ``kv_frames::locate`` (the kernels' header,
compiled here with g++) against the contiguous arithmetic the readers use today:

    sparse_mla/kernels/sm_common.hpp   sm::kv_row_bank / sm::kv_row_offset  (compiled into the same driver)
    sparse_mla/layout.py               kv_row_location
    sparse_mla/kernels/sm_append.cpp   chip and row of a position (lines 57, 58, 62)
    indexer_scan/kernels/scan_reader.cpp + indexer_scan.py (458)   a bank's key stream: buffer + slot offset +
                                       (block / banks) * 4352; visible rows of a chip

No device. indexer_scan.py imports ttnn (no tensor is touched): run under HY4_MOCK=1.
"""

from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path

import pytest

from models.experimental.hy4_preview.fused.kv_store import frames as fr
from models.experimental.hy4_preview.fused.sparse_mla.layout import SparseMlaConfig, kv_row_location

HERE = Path(__file__).resolve().parent
KERNELS = HERE.parent / "kernels"
SM_KERNELS = HERE.parent.parent / "sparse_mla" / "kernels"
CASES = 200_000

DRIVER = r"""
#include <cstdint>
#include <cstdio>
#include "sm_common.hpp"
#include "kv_frames.hpp"
// stdin, one case per line: R slot row stride base slot_bytes row_stride position
// stdout: today's (bank, offset) of page slot * R + row; the table's (bank, offset); the append's chip and row
// of `position` (sm_append.cpp: BLOCK 128, CHIPS 32); kv_frames' chip and row; mapped() of row and of row + 256.
int main() {
    unsigned long long R, slot, row, stride, base, slot_bytes, row_stride, position;
    static uint32_t table[kv_frames::TABLE_FRAMES];
    while (scanf("%llu %llu %llu %llu %llu %llu %llu %llu", &R, &slot, &row, &stride, &base, &slot_bytes, &row_stride,
                 &position) == 8) {
        const uint32_t page = static_cast<uint32_t>(slot * R + row);
        for (uint32_t k = 0; k < kv_frames::TABLE_FRAMES; ++k) {
            table[k] = kv_frames::UNMAPPED;
        }
        table[row >> 8] = static_cast<uint32_t>(base);
        const volatile uint32_t* l1 = table;  // the kernels read the table through a volatile L1 pointer
        const kv_frames::Loc at = kv_frames::locate(l1, static_cast<uint32_t>(slot_bytes), static_cast<uint32_t>(row),
                                                    static_cast<uint32_t>(row_stride));
        constexpr uint32_t BLOCK = 128, CHIPS = 32;
        const uint32_t p = static_cast<uint32_t>(position);
        const uint32_t append_row = (p / (BLOCK * CHIPS)) * BLOCK + p % BLOCK;
        const uint32_t append_chip = (p / BLOCK) % CHIPS;
        printf("%u %u %u %u %u %u %u %u %d %d\n", sm::kv_row_bank<8, 32>(page),
               sm::kv_row_offset<8, 32>(page, static_cast<uint32_t>(stride)), at.bank, at.offset, append_chip,
               append_row, kv_frames::chip_of(p), kv_frames::stripe_row(p),
               kv_frames::mapped(l1, static_cast<uint32_t>(row)) ? 1 : 0,
               kv_frames::mapped(l1, static_cast<uint32_t>(row) + 256) ? 1 : 0);
    }
    return 0;
}
"""


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    gxx = shutil.which("g++")
    if gxx is None:
        pytest.skip("no g++ on this host")
    d = tmp_path_factory.mktemp("kvf")
    (d / "driver.cpp").write_text(DRIVER)
    subprocess.run(
        [
            gxx,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Werror",
            f"-I{KERNELS}",
            f"-I{SM_KERNELS}",
            "-o",
            str(d / "driver"),
            str(d / "driver.cpp"),
        ],
        check=True,
    )

    def run(cases):
        text = "\n".join(" ".join(str(int(v)) for v in c) for c in cases) + "\n"
        out = subprocess.run([str(d / "driver")], input=text, capture_output=True, text=True, check=True).stdout
        rows = [tuple(int(v) for v in line.split()) for line in out.splitlines()]
        assert len(rows) == len(cases)
        return rows

    return run


def _cases(rng, n, kind: fr.Kind, pool: bool):
    """(case for the driver, expectation inputs). Slot-major caches of 1 .. 128 frames per slot, up to 16 users
    (today's 8 x 1M and 16 x 1M included), or a pool of up to 1,400 frames."""
    out = []
    for _ in range(n):
        layer = rng.randrange(kind.layers)
        if pool:
            frames = rng.choice([1, 12, 757, 1353, 1400])
            frame = rng.randrange(frames)
            geometry = fr.Geometry.pool(kind)
            k = rng.randrange(fr.TABLE_FRAMES)  # where the frame sits in the conversation
            row = k * fr.FRAME_ROWS + rng.randrange(fr.FRAME_ROWS)
            base = geometry.frame_base(frame)
            # today's arithmetic sees the pool as a cache of capacity 8192 whose "user" is the frame
            R, slot, today_row = fr.FRAME_ROWS, frame * kind.layers + layer, row % fr.FRAME_ROWS
        else:
            capacity = rng.choice([1, 2, 3, 12, 97, 100, 128])
            users = rng.choice([1, 8, 16])
            user = rng.randrange(users)
            geometry = fr.Geometry.contiguous(kind, capacity)
            row = rng.randrange(capacity * fr.FRAME_ROWS)
            base = geometry.identity_table(user)[row >> 8]
            R, slot, today_row = capacity * fr.FRAME_ROWS, user * kind.layers + layer, row
        position = rng.randrange(1 << 20)
        out.append(
            (
                R,
                slot,
                today_row,
                kind.row_stride or 136,
                base,
                geometry.slot_bytes(layer),
                kind.row_stride,
                position,
                row,
                geometry,
                layer,
            )
        )
    return out


@pytest.mark.parametrize("pool", [False, True], ids=["identity-table-of-todays-cache", "frame-major-pool"])
def test_latent_row_equals_todays_arithmetic(driver, pool):
    """The packed latent row: sm_reader.cpp / sm_append.cpp address it with sm::kv_row_bank / kv_row_offset at page
    (u * 78 + layer) * R + row. Under the identity table of that cache (and, for a pool, with user := frame,
    R := 256) kv_frames::locate gives the same bank and the same byte, in C++ and in Python."""
    rng = random.Random(11 + pool)
    cases = _cases(rng, CASES, fr.LATENT, pool)
    got = driver([c[:8] for c in cases])
    banks, cfgs = set(), {}
    for c, g in zip(cases, got):
        R, slot, today_row, stride, base, slot_bytes, row_stride, position, row, geometry, layer = c
        today = (g[0], g[1])
        cfg = cfgs.setdefault(R, SparseMlaConfig(rows_per_slot=R))
        assert today == kv_row_location(cfg, slot, today_row), c
        assert (g[2], g[3]) == today, (c, g)
        table = {row >> 8: base}
        assert fr.locate(table, slot_bytes, row, row_stride) == today, c
        assert g[8] == 1 and g[9] == 0  # the row's frame is mapped, the next one is not
        banks.add(g[2])
    assert banks == set(range(8))


def test_stripe_chip_and_row(driver):
    """sm_append.cpp's chip and row of a position = kv_frames::chip_of / stripe_row = frames.py; and a frame keeps
    them: position p of a conversation is row (p // 8192) * 256 + row(p % 8192) of the same chip."""
    rng = random.Random(3)
    cases = _cases(rng, 50_000, fr.LATENT, False)
    for c, g in zip(cases, driver([c[:8] for c in cases])):
        p = c[7]
        assert (g[4], g[5]) == (g[6], g[7]) == (fr.chip_of(p), fr.stripe_row(p)), p
        k, q = divmod(p, fr.FRAME_TOKENS)
        assert fr.chip_of(p) == fr.chip_of(q) and fr.stripe_row(p) == k * fr.FRAME_ROWS + fr.stripe_row(q)
        assert fr.position_of(fr.chip_of(p), fr.stripe_row(p)) == p


def test_bank_is_kept_by_frames_of_8192_and_not_by_4096():
    """KV_STORE.md 1.2 / 1.3: with capacities that are multiples of 8192 the bank of a row is (row // 32) % 8
    whatever the slot and the capacity, so a row keeps its bank when its frame moves; a 4096-token unit does not."""
    rng = random.Random(5)
    for _ in range(50_000):
        capacity = rng.randrange(1, 129)
        cfg = SparseMlaConfig(rows_per_slot=capacity * fr.FRAME_ROWS)
        slot, row = rng.randrange(16 * 78), rng.randrange(capacity * fr.FRAME_ROWS)
        assert kv_row_location(cfg, slot, row)[0] == (row >> 5) & 7
    moved = 0
    for slot in range(1, 64):
        cfg = SparseMlaConfig(rows_per_slot=128)  # a 4096-token "frame": 4 shards
        moved += kv_row_location(cfg, slot, 0)[0] != kv_row_location(cfg, 0, 0)[0]
    assert moved == 32  # every odd slot starts on bank 4


def test_key_stream_equals_todays_scan(driver):
    """scan_reader.cpp reads block j of bank b of user's slot at buffer + slot_offset + j * 4352, with
    slot_offset = slot * bank_blocks * 4352 (indexer_scan.py 458). Under the identity table that is
    locate(table, rank_bytes, stream_row(b, j), 0); in a pool the frame's base replaces j * 4352."""
    scan = pytest.importorskip("models.experimental.hy4_preview.fused.indexer_scan.indexer_scan")
    rng = random.Random(9)
    kind = fr.INDEX
    assert scan.KEY_BLOCK_BYTES == kind.shard_bytes
    for _ in range(20_000):
        capacity = rng.choice([1, 4, 12, 100, 128])
        users, rank = rng.choice([8, 16]), rng.randrange(kind.layers)
        user = rng.randrange(users)
        cfg = scan.ScanConfig(stripe_rows=capacity * fr.FRAME_ROWS)
        slot = user * kind.layers + rank
        slot_offset = slot * (cfg.bank_blocks * scan.KEY_BLOCK_BYTES)
        bank, j = rng.randrange(8), rng.randrange(capacity)
        today = slot_offset + j * scan.KEY_BLOCK_BYTES
        geometry = fr.Geometry.contiguous(kind, capacity)
        table = geometry.identity_table(user)
        assert fr.stream_block(table, geometry.slot_bytes(rank), bank, j) == today
        row = ((j * 8 + bank) * 32) + rng.randrange(32)
        assert fr.locate(table, geometry.slot_bytes(rank), row, 0) == (bank, today)
        # the pool: the "user" of today's arithmetic is the frame, and a slot has ONE block per bank
        frame = rng.randrange(1400)
        pool = fr.Geometry.pool(kind)
        one = scan.ScanConfig(stripe_rows=fr.FRAME_ROWS)
        assert one.bank_blocks == 1
        today_pool = (frame * kind.layers + rank) * one.bank_blocks * scan.KEY_BLOCK_BYTES
        assert fr.locate({j: pool.frame_base(frame)}, pool.slot_bytes(rank), row, 0) == (bank, today_pool)
        # visible rows of a chip: the reader's prefix
        context, chip = rng.randrange(capacity * fr.FRAME_TOKENS + 1), rng.randrange(32)
        assert fr.visible_rows(context, chip) == scan.visible_rows(context, chip, 32, 128, cfg.stripe_rows)
    cases = [(256, 0, r, 136, 7 * 4352, 3 * 4352, 0, 0) for r in range(0, 256, 13)]
    for c, g in zip(cases, driver(cases)):
        assert (g[2], g[3]) == ((c[2] >> 5) & 7, 10 * 4352)


def test_sizes():
    assert fr.FRAME_TOKENS == 8192 and fr.FRAME_ROWS == 256
    assert fr.LATENT.frame_bytes_per_chip + fr.INDEX.frame_bytes_per_chip == 14_788_608  # 14.8 MB per chip
    assert fr.BYTES_PER_TOKEN_PER_CHIP == 1805.25
    # the largest pool of KV_STORE.md (1,353 frames) and today's 16 x 1M stay below 2^32 in a bank
    assert fr.Geometry.pool(fr.LATENT).bytes_per_bank(1353) < 2**32
    assert fr.Geometry.contiguous(fr.LATENT, 128).bytes_per_bank(16) < 2**32
    assert fr.Geometry.pool(fr.LATENT).tensor_shape(757) == (757 * 78, 1, 256, 656)
    assert fr.Geometry.contiguous(fr.INDEX, 100).tensor_shape(1) == (21, 1, 25600, 128)
