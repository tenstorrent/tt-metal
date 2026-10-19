# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side SWA block table bookkeeping (kv.SwaBlockTable) through chunked prefill and decode; no device."""

import pytest
import torch

from models.experimental.mimo_v2_6_flash.tt.kv import SwaBlockTable, swa_blocks_per_user
from models.experimental.mimo_v2_6_flash.tt.model import prefill_padded_len

BLOCK, WINDOW, MAX_SEQ, MAX_BATCH = 64, 128, 16384, 32


def check(table, live):
    """Every position some later window or chunk can read is in a real block, and no real block serves two
    (slot, logical block) entries. ``live``: slot -> [(first, end)] position ranges that must be readable."""
    owners = {}
    for slot, ranges in live.items():
        for first, end in ranges:
            for j in range(first // BLOCK, (end - 1) // BLOCK + 1):
                b = int(table.table[slot, j])
                assert b != 0, f"slot {slot}: block {j} (positions {j * BLOCK}..) is not backed"
    for slot in range(table.table.shape[0]):
        for j in range(table.table.shape[1]):
            b = int(table.table[slot, j])
            if b:
                assert b not in owners, f"block {b} backs both {owners[b]} and {(slot, j)}"
                owners[b] = (slot, j)


@pytest.mark.parametrize("chunk", [64, 128, 192, 512, 2048, 4096])
def test_swa_table_chunked_prefill_then_decode(chunk):
    users = 4
    table = SwaBlockTable(users, MAX_BATCH, MAX_SEQ // BLOCK, BLOCK, WINDOW, swa_blocks_per_user(WINDOW, BLOCK))
    lengths = [1, 200, 2049, 9001]
    live = {}
    for slot, n in enumerate(lengths):
        for s in range(0, n, chunk):
            real = min(chunk, n - s)
            padded_end = s + prefill_padded_len(real)
            table.prefill(slot, s, s + real, padded_end)
            # The chunk reads the window before it and leaves the window before its end (and the padding) readable.
            live[slot] = [(max(0, s + real - WINDOW), padded_end)]
            if s:
                blocks = table.window_blocks(slot, s)
                assert len(blocks) == (s - table.window_start(s)) // BLOCK and all(blocks)
                live[slot].append((table.window_start(s), s))
            check(table, live)
        live[slot] = [(max(0, n - WINDOW + 1), n)]
    for step in range(300):
        for slot, n in enumerate(lengths):
            pos = n + step
            table.advance(slot, pos)
            live[slot] = [(max(0, pos - WINDOW + 1), pos + 1)]
        check(table, live)


def test_swa_table_new_user_frees_the_old_one():
    table = SwaBlockTable(1, MAX_BATCH, MAX_SEQ // BLOCK, BLOCK, WINDOW, swa_blocks_per_user(WINDOW, BLOCK))
    for _ in range(3):
        for s in range(0, 4096, 1024):
            table.prefill(0, s, s + 1024, s + 1024)
        for pos in range(4096, 4096 + 200):
            table.advance(0, pos)
    assert int((table.table[0] != 0).sum()) <= swa_blocks_per_user(WINDOW, BLOCK)
    assert torch.equal(table.table[1:], torch.zeros_like(table.table[1:]))


def test_swa_table_reuse_after_the_context_end():
    """A user that decoded up to the last position of the context gives every block back to the next user."""
    blocks = swa_blocks_per_user(WINDOW, BLOCK)
    table = SwaBlockTable(1, MAX_BATCH, MAX_SEQ // BLOCK, BLOCK, WINDOW, blocks)
    table.prefill(0, 0, 1024, 1024)
    for pos in range(1024, MAX_SEQ):
        table.advance(0, pos)
        assert len(table.live[0]) <= blocks
    table.prefill(0, 0, 100, prefill_padded_len(100))  # a new user in the same slot
    mapped = {j for j in range(table.table.shape[1]) if table.table[0, j]}
    assert mapped == table.live[0] == {0, 1}
    assert len(table.free[0]) == blocks - len(mapped)
