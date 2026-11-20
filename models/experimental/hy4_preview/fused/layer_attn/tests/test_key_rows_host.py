# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only (PLAN.md section 20): the index-key append's row writer, block by block.

tests/host_key_rows.cpp runs the writer's sequence with the kernels' own header (kernels/attn_key_rows.hpp: the
groups of a pass, the bytes of a tile row) on a store this test makes; here the result is compared with a model
written separately (tests/key_bytes.py: the tile's bytes as arrays; the pass as "token rows ascending, each row into
its block"):
  * every block that gets a row is read ONCE and written ONCE, no other block is read or written;
  * after the pass the store is the model's: every patched row holds its key's bytes, every other byte of every
    block is what it was (the other 31 rows of a touched block included);
  * two token rows of one pass in one block are both there; two at the SAME stripe row: the higher token row's
    key, and same_row() names the pair (the writer's status word), and names nothing in any other pass.
The bytes model itself (tests/key_bytes.py) is held against the device (ttnn gives a host no raw view of a BFP8
tile): tests/test_key_append.py, where the writer keeps the blocks it read and their values are compared with ttnn's.

    python -m pytest models/experimental/hy4_preview/fused/layer_attn/tests/test_key_rows_host.py
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from models.experimental.hy4_preview.fused.layer_attn.tests import key_bytes as KB

HERE = Path(__file__).parent
NONE = 0xFFFFFFFF
USERS = STORE = 8


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    exe = tmp_path_factory.mktemp("key_rows") / "host_key_rows"
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-Wall", "-Werror", "-o", str(exe), str(HERE / "host_key_rows.cpp")], check=True
    )
    return exe


def make_case(rng, kind):
    """(where [8], rows [8], packed [4352], addresses [8], store [8, 4352])."""
    addresses = (rng.choice(1 << 20, STORE, replace=False).astype(np.uint32) * 4352 * 16 % (1 << 28)) & ~np.uint32(15)
    addresses = np.unique(addresses | rng.integers(0, 8, STORE).astype(np.uint32))
    while len(addresses) < STORE:  # (distinct "bank | offset" numbers)
        addresses = np.unique(np.append(addresses, np.uint32(rng.integers(1, 1 << 24)) * 16))
    addresses = rng.permutation(addresses[:STORE]).astype(np.uint32)
    store = rng.integers(0, 256, (STORE, KB.BLOCK_BYTES), dtype=np.uint8)
    packed = rng.integers(0, 256, KB.BLOCK_BYTES, dtype=np.uint8)
    where = np.full(USERS, NONE, dtype=np.uint32)
    rows = rng.integers(0, 32, USERS).astype(np.uint32)
    if kind == "single":  # one row a user: every written row in a block of its own
        live = rng.random(USERS) < 0.6
        where[live] = addresses[rng.permutation(STORE)][: int(live.sum())]
    elif kind == "block":  # 1 + k rows of one conversation: consecutive stripe rows, most of them in one block
        k = int(rng.integers(2, USERS + 1))
        first = int(rng.integers(0, 32))
        for i in range(k):
            where[i] = addresses[(first + i) // 32]
            rows[i] = (first + i) % 32
    elif kind == "same_row":  # two token rows at one stripe row of one block
        where[[1, 5]] = addresses[3]
        rows[[1, 5]] = int(rng.integers(0, 32))
        where[2] = addresses[3]
        rows[2] = (int(rows[1]) + 1) % 32
    else:  # anything
        live = rng.random(USERS) < 0.8
        where[live] = addresses[rng.integers(0, 3, int(live.sum()))]
    return where, rows, packed, addresses, store


def model(where, rows, packed, addresses, store):
    after = store.copy()
    touched = set()
    for u in range(USERS):  # token rows ascending, each into its block
        if where[u] == NONE:
            continue
        b = int(np.nonzero(addresses == where[u])[0][0])
        after[b] = KB.put_row(after[b], int(rows[u]), KB.row_bytes(packed, u))
        touched.add(b)
    return after, touched


@pytest.mark.parametrize("kind", ["single", "block", "same_row", "any"])
def test_writer(binary, tmp_path, kind):
    rng = np.random.default_rng({"single": 1, "block": 2, "same_row": 3, "any": 4}[kind])
    cases = [make_case(rng, kind) for _ in range(300)]
    with open(tmp_path / "in.bin", "wb") as f:
        for where, rows, packed, addresses, store in cases:
            f.write(where.tobytes() + rows.tobytes() + packed.tobytes())
            for b in range(STORE):
                f.write(addresses[b : b + 1].tobytes() + store[b].tobytes())
    out = subprocess.run(
        [str(binary), str(tmp_path / "in.bin"), str(tmp_path / "out.bin")], capture_output=True, text=True
    )
    assert out.returncode == 0, (out.returncode, out.stdout, out.stderr)
    raw = np.fromfile(tmp_path / "out.bin", dtype=np.uint8)
    per = 2 * 4 * STORE + 16 + STORE * KB.BLOCK_BYTES
    assert raw.size == per * len(cases), (raw.size, per * len(cases))
    patched = shared = reported = 0
    for i, (where, rows, packed, addresses, store) in enumerate(cases):
        rec = raw[i * per : (i + 1) * per]
        reads, writes = rec[: 4 * STORE].view(np.uint32), rec[4 * STORE : 8 * STORE].view(np.uint32)
        twice = int(rec[8 * STORE : 8 * STORE + 4].view(np.uint32)[0])
        pairs = [
            (u, v)
            for u in range(USERS)
            for v in range(u + 1, USERS)
            if where[u] != NONE and where[v] == where[u] and rows[v] == rows[u]
        ]
        assert twice == (pairs[0][0] | pairs[0][1] << 8 if pairs else NONE), (kind, i, hex(twice), pairs[:2])
        reported += bool(pairs)
        got = rec[8 * STORE + 16 :].reshape(STORE, KB.BLOCK_BYTES)
        want, touched = model(where, rows, packed, addresses, store)
        once = np.array([1 if b in touched else 0 for b in range(STORE)], dtype=np.uint32)
        assert np.array_equal(reads, once) and np.array_equal(writes, once), (kind, i, reads, writes, sorted(touched))
        assert np.array_equal(got, want), (kind, i, np.argwhere(got != want)[:4])
        # the other rows of every touched block, byte for byte, and every untouched block
        for b in range(STORE):
            mine = sorted({int(rows[u]) for u in range(USERS) if where[u] == addresses[b]})
            keep = np.ones(KB.BLOCK_BYTES, dtype=bool)
            for r in mine:
                keep &= KB.other_rows_mask(r)
            assert np.array_equal(got[b][keep], store[b][keep]), (kind, i, b)
            patched += len(mine)
            shared += len(mine) > 1
    print(
        f"HY4_MK8A key rows host [{kind}]: {len(cases)} passes, {patched} rows patched, {shared} blocks with more than one row of a pass, "
        f"{reported} passes with one stripe row given twice (reported as the first such pair, else not at all)"
    )
    assert reported == (len(cases) if kind == "same_row" else 0) or kind == "any"


def test_bytes_model():
    """key_bytes against itself and against the header's two functions as numbers: a row's bytes are 4 x 2 x 17
    distinct bytes, the 32 rows of a block partition its 4,352 bytes, and values() reads what put_row() placed."""
    seen = np.zeros(KB.BLOCK_BYTES, dtype=np.int32)
    for row in range(32):
        seen += ~KB.other_rows_mask(row)
        for h in range(2):  # (attn_key_rows.hpp exp_byte / man_byte)
            exp_byte = 16 * (2 * (row >> 4) + h) + (row & 15)
            man_byte = 64 + 256 * (2 * (row >> 4) + h) + 16 * (row & 15)
            probe = np.zeros(KB.BLOCK_BYTES, dtype=np.uint8)
            data = np.zeros((KB.TILES, 2, 17), dtype=np.uint8)
            data[2, h, 0], data[2, h, 1:] = 200, np.arange(1, 17)
            got = KB.put_row(probe, row, data).reshape(KB.TILES, KB.TILE_BYTES)[2]
            assert got[exp_byte] == 200 and np.array_equal(got[man_byte : man_byte + 16], np.arange(1, 17))
    assert np.all(seen == 1), "the rows' bytes do not partition the block"
    block = np.zeros(KB.BLOCK_BYTES, dtype=np.uint8)
    data = np.zeros((KB.TILES, 2, 17), dtype=np.uint8)
    data[1, 1, 0] = 127 + 3  # exponent: the row's largest value in [8, 16)
    data[1, 1, 1 + 5] = 0x80 | 0x48  # -(72 / 64) * 8 = -9.0 at column 32 + 16 + 5
    v = KB.values(KB.put_row(block, 21, data))
    assert v[21, 32 + 16 + 5] == -9.0 and np.count_nonzero(v) == 1
