# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The multi-block draft call's host inputs (``tt/v41/dspark_blocks.py``), on the workstation: no ttnn, no
device. ``python -m pytest --noconftest -p no:cacheprovider
models/demos/deepseek_v3_d_p/tests/v41/test_dspark_blocks.py``
(the tree's conftest opens the cluster at collection)."""

import importlib.util
import pathlib

import pytest
import torch

_PATH = pathlib.Path(__file__).resolve().parents[2] / "tt" / "v41" / "dspark_blocks.py"
_SPEC = importlib.util.spec_from_file_location("dspark_blocks", _PATH)
db = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(db)  # the module alone: tt.v41's package __init__ imports ttnn


def _single_reference(token: int, p: int, ring_slots: int, write: bool, tap_row: int) -> dict:
    """The one-block inputs as ``TtDSparkDrafter.host_inputs`` built them (the
    reference's ``get_dspark_topk_idxs``: the written ring slots, then the block)."""
    ids = torch.full((db.ROWS,), db.NOISE, dtype=torch.int32)
    ids[0] = token
    sel = torch.zeros(db.ROWS, db.ROWS)
    ring_idx = torch.full((db.ROWS,), db.SENTINEL, dtype=torch.int32)
    if write:
        sel[0, tap_row] = 1.0
        ring_idx[0] = p % ring_slots
    valid = min(db.RING, p + 1)
    rows = [db.SENTINEL] * db.INDEX_WIDTH
    if ring_slots == db.RING:
        rows[:valid] = range(valid)
    else:
        rows[:valid] = [q % ring_slots for q in range(p - valid + 1, p + 1)]
    rows[valid : valid + db.BLOCK] = range(ring_slots, ring_slots + db.BLOCK)
    index = torch.tensor(rows, dtype=torch.int32).view(1, -1).expand(db.ROWS, -1).contiguous()
    return {"ids": ids, "sel": sel, "ring_idx": ring_idx, "index": index}


@pytest.mark.parametrize("ring_slots", [128, 160])
@pytest.mark.parametrize("p", [5, 127, 128, 300, 1054655])
def test_one_block_matches_the_single_call(ring_slots, p):
    """One entry of user 0 reproduces the single call's token, selector, ring index and block rows' index rows
    (the rows past the block are noise rows whose index rows were block 0's before and are the sentinel now:
    their outputs are never read)."""
    for write in (False, True):
        got = db.rank_inputs([{"user": 0, "token": 7, "position": p, "taps": [(3, p)] if write else []}], ring_slots, 1)
        ref = _single_reference(7, p, ring_slots, write, 3)
        assert torch.equal(got["ids"], ref["ids"])
        assert torch.equal(got["sel"], ref["sel"])
        assert torch.equal(got["ring_idx"], ref["ring_idx"])
        assert torch.equal(got["index"][: db.BLOCK], ref["index"][: db.BLOCK])
        valid = min(db.RING, p + 1)
        assert torch.equal(got["index"][db.BLOCK :, :valid], ref["index"][db.BLOCK :, :valid])
        assert (got["index"][db.BLOCK :, valid:] == db.SENTINEL).all()
        assert torch.equal(got["block_pos"], torch.arange(p + 1, p + 1 + db.ROWS))
        assert int(got["main_pos"][0]) == (p if write else 0)


def test_four_blocks_four_rings():
    """Four slots of a rank at different frontiers with different accepted counts: each block's token on its
    first row, its taps on its rows, its ring's rows in its index rows, its own block rows after them, the
    sentinel elsewhere; nothing of one slot in another's rows."""
    ring_slots, users = 160, 4
    entries = [
        {"user": 0, "token": 11, "position": 300, "taps": [(0, 298), (1, 299), (2, 300)]},
        {"user": 1, "token": 12, "position": 1000, "taps": [(4, 1000)]},
        {"user": 2, "token": 13, "position": 64, "taps": []},
        {"user": 3, "token": 14, "position": 2047, "taps": [(12, 2046), (13, 2047)]},
    ]
    t = db.rank_inputs(entries, ring_slots, users)
    block_base = users * ring_slots
    for a, e in enumerate(entries):
        u, p, r0 = e["user"], e["position"], db.BLOCK * a
        assert int(t["ids"][r0]) == e["token"]
        assert all(int(t["ids"][r0 + i]) == db.NOISE for i in range(1, db.BLOCK))
        for i, (tap_row, q) in enumerate(e["taps"]):
            assert t["sel"][r0 + i].sum() == 1 and t["sel"][r0 + i, tap_row] == 1
            assert int(t["ring_idx"][r0 + i]) == u * ring_slots + q % ring_slots
            assert int(t["main_pos"][r0 + i]) == q
        for i in range(len(e["taps"]), db.BLOCK):
            assert t["sel"][r0 + i].sum() == 0 and int(t["ring_idx"][r0 + i]) == db.SENTINEL
        valid = min(db.RING, p + 1)
        want = [u * ring_slots + q % ring_slots for q in range(p - valid + 1, p + 1)]
        want += list(range(block_base + r0, block_base + r0 + db.BLOCK))
        for i in range(db.BLOCK):
            row = t["index"][r0 + i].tolist()
            assert row[: len(want)] == want
            assert all(v == db.SENTINEL for v in row[len(want) :])
            # the user's ring region alone, then the block rows
            assert all(u * ring_slots <= v < (u + 1) * ring_slots for v in row[:valid])
        assert t["block_pos"][r0 : r0 + db.BLOCK].tolist() == list(range(p + 1, p + 1 + db.BLOCK))
    # the rows past the blocks: noise, no write, block 0's window alone (never an all-sentinel row)
    window = t["index"][0].tolist()[: db.RING]
    for r in range(db.BLOCK * len(entries), db.ROWS):
        assert int(t["ids"][r]) == db.NOISE and int(t["ring_idx"][r]) == db.SENTINEL
        assert t["sel"][r].sum() == 0
        row = t["index"][r].tolist()
        assert row[: db.RING] == window and all(v == db.SENTINEL for v in row[db.RING :])
    assert t["index"].max() < block_base + db.ROWS


def test_mesh_inputs_stack_the_ranks():
    a = [{"user": 0, "token": 1, "position": 200, "taps": [(0, 200)]}]
    b = [{"user": 3, "token": 2, "position": 400, "taps": []}]
    t = db.mesh_inputs([a, b, a, b], 160, 4)
    assert t["ids"].shape == (4, db.ROWS) and t["index"].shape == (4, db.ROWS, db.INDEX_WIDTH)
    assert torch.equal(t["ids"][0], t["ids"][2]) and torch.equal(t["index"][1], t["index"][3])
    assert int(t["ids"][1, 0]) == 2 and int(t["ring_idx"][1, 0]) == db.SENTINEL
    assert int(t["ring_idx"][0, 0]) == 200 % 160


def test_read_drafts():
    ids = torch.arange(db.ROWS)
    assert db.read_drafts(ids, 2) == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]


def test_limits():
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-only test uses --noconftest
        db.rank_inputs([{"user": 0, "token": 1, "position": 10, "taps": []}] * 7, 160, 7)
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-only test uses --noconftest
        db.rank_inputs([{"user": 2, "token": 1, "position": 10, "taps": []}], 160, 2)
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-only test uses --noconftest
        db.rank_inputs([{"user": 0, "token": 1, "position": 10, "taps": [(0, 11)]}], 160, 1)
