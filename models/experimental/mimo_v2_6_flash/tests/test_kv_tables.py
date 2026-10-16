# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the paged-KV tables in tt/kv.py (no device needed)."""

import pytest
import torch

from models.experimental.mimo_v2_6_flash.tt import kv as kv_lib

TP = 8
BLOCK = 64
WINDOW = 128


@pytest.mark.parametrize("max_chunk", [64, 512, 4096])
def test_sliding_window_tables_keep_window_and_chunk_apart(max_chunk):
    """A chunk at any block-aligned start must not write over the window it reads, nor over another user."""
    num_users, max_batch, max_seq = 4, 32, 64 * 1024
    tables, num_blocks = kv_lib.layer_page_tables(
        num_users, max_batch, max_seq, BLOCK, TP, rep=1, sliding_window=WINDOW, max_chunk=max_chunk, seed=0
    )
    assert tables.shape == (TP, max_batch, max_seq // BLOCK)
    assert (tables == tables[0]).all(), "SWA tables are the same on every chip"
    per_user = kv_lib.swa_blocks_per_user(WINDOW, max_chunk, BLOCK)
    assert num_blocks == num_users * per_user
    owned = [set(tables[0, u].tolist()) for u in range(num_users)]
    for u in range(num_users):
        assert len(owned[u]) == per_user
        for v in range(u):
            assert not owned[u] & owned[v], f"users {u} and {v} share a block"
    for start in range(0, max_seq - max_chunk, BLOCK * 7):
        lo = max(0, start - WINDOW + 1) // BLOCK
        hi = -(-(start + max_chunk) // BLOCK)
        for u in range(num_users):
            needed = tables[0, u, lo:hi]
            assert needed.unique().numel() == needed.numel(), f"user {u}: window + chunk at {start} alias"


def test_global_tables_split_users_by_parity():
    num_users, max_batch, logical = 4, 32, 16
    rep = kv_lib.kv_replication(TP, 4)
    tables, num_blocks = kv_lib.page_tables(num_users, max_batch, logical, TP, rep, seed=3)
    scratch = num_blocks - 1
    for chip in range(TP):
        mine = kv_lib.owned_slots(num_users, rep, chip % rep)
        used = set()
        for slot in range(max_batch):
            row = set(tables[chip, slot].tolist())
            if slot in mine:
                assert scratch not in row and not row & used
                used |= row
            else:
                assert row == {scratch}
    masks = kv_lib.slot_owner_masks(TP, rep)
    for slot in range(num_users):
        pos = torch.full((max_batch,), -1, dtype=torch.int32)
        pos[slot] = 100
        owned = kv_lib.owned_positions(pos, TP, rep)[:, slot] >= 0
        assert torch.equal(masks[slot % rep].flatten().bool(), owned)
