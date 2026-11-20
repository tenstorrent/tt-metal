# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the sparse-MLA geometry and goldens (no device)."""

import torch

from models.experimental.hy4_preview.fused.sparse_mla import layout as lo


def test_tiny_tile_codec():
    x = torch.randn(8, 576)
    flat = lo.pack_tiny(x)
    assert torch.equal(lo.unpack_tiny(flat, 8), x)
    # element (row, col) of tile t sits at t * 256 + (col // 16) * 128 + row * 16 + col % 16
    assert flat[3 * 256 + 1 * 128 + 5 * 16 + 2] == x[5, 3 * 32 + 16 + 2]


def test_partials_merge_to_the_reference_core():
    """The partial contract: chips' partials, an empty chip and the sink merge to the reference's absorbed ctx."""
    cfg = lo.SparseMlaConfig(rows_per_slot=4096)
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(cfg.heads, cfg.width, generator=gen)
    rows = torch.randn(300, cfg.width, generator=gen)
    sink = torch.randn(cfg.heads, generator=gen)
    parts = [lo.partial(q, rows[i::5], cfg) for i in range(5)] + [lo.partial(q, rows[:0], cfg)]
    assert (lo.merge(parts, sink, cfg) - lo.reference_ctx(q, rows, sink, cfg)).abs().max() < 1e-12
    # two empty peers and the sink: ctx = 0, not NaN
    empty = lo.partial(q, rows[:0], cfg)
    assert torch.equal(lo.merge([empty, empty], sink, cfg), torch.zeros(cfg.heads, cfg.latent, dtype=torch.float64))


def test_reference_ctx_is_the_reference():
    """layout.reference_ctx is reference/attention_decode.py absorbed_stages' ctx (same lines, no weights)."""
    from models.experimental.hy4_preview.reference.attention_decode import additive_mask

    cfg = lo.SparseMlaConfig(rows_per_slot=4096)
    gen = torch.Generator().manual_seed(1)
    q = torch.randn(cfg.heads, cfg.width, generator=gen)
    rows = torch.randn(77, cfg.width, generator=gen)
    sink = torch.randn(cfg.heads, generator=gen)
    logits = torch.matmul(q, rows.t()) * cfg.scale + additive_mask(torch.ones(77, dtype=torch.bool))
    combined = torch.cat([logits, sink.reshape(-1, 1)], dim=-1)
    probs = torch.softmax(combined - combined.max(dim=-1, keepdim=True).values, dim=-1)[:, :-1]
    want = torch.matmul(probs, rows[:, : cfg.latent])
    assert (lo.reference_ctx(q, rows, sink, cfg) - want).abs().max() < 1e-5


def test_every_chunk_has_one_block_and_users_spread():
    cfg = lo.SparseMlaConfig(rows_per_slot=4096)
    # one chunk per user: 8 users on 8 different S-blocks
    assert sorted(cfg.block_of(u, 0) for u in range(cfg.users)) == list(range(cfg.groups))
    # 2048 rows per user: every S-block gets topk / chunk / groups chunks of every user
    per_block = cfg.topk // cfg.chunk_rows // cfg.groups
    for u in range(cfg.users):
        blocks = [cfg.block_of(u, c) for c in range(cfg.chunks(cfg.topk))]
        assert all(blocks.count(b) == per_block for b in range(cfg.groups))


def test_merge_tree_reaches_the_root_once():
    for n_act in range(1, 9):
        have = {r: {r} for r in range(n_act)}  # which blocks' partials relative block r holds
        for s in range(lo.MERGE_STEPS):
            for r in range(1, n_act):
                if lo.send_step(r) == s:
                    assert lo.receives(r - (1 << s), n_act, s), (n_act, r, s)
                    have[r - (1 << s)] |= have.pop(r)
            for r in range(n_act):
                if lo.receives(r, n_act, s):
                    assert r + (1 << s) < n_act and lo.send_step(r + (1 << s)) == s
        assert have == {0: set(range(n_act))}, (n_act, have)


def test_row_location_matches_the_accessor_rule():
    """page -> (bank, offset): shard = page // 32, round-robin over 8 banks (tensor_accessor.h)."""
    cfg = lo.SparseMlaConfig(rows_per_slot=32768)
    seen = set()
    for slot, row in [(0, 0), (0, 31), (0, 32), (0, 255), (0, 256), (1, 0), (623, 32767), (77, 12345)]:
        bank, off = lo.kv_row_location(cfg, slot, row)
        page = slot * cfg.rows_per_slot + row
        assert bank == (page // 32) % 8 and off % cfg.row_stride == 0
        assert off // cfg.row_stride == (page // 256) * 32 + page % 32
        seen.add((bank, off))
    assert len(seen) == 8 and cfg.row_stride == 704 and cfg.row_bytes == 656


def test_frame_identity_addresses():
    """The row reader's two address forms give the same (bank, offset) for every row of today's slot-major store:
    the slot arithmetic (kernels/sm_common.hpp kv_row_bank / kv_row_offset; lo.kv_row_location) and fused/kv_store's
    frame table with identity tables (kernels/sm_reader.cpp SM_FRAMES; kv_frames.locate)."""
    from models.experimental.hy4_preview.fused.kv_store import frames as kv_frames

    for rows_per_slot, layers in ((256, 1), (1024, 1), (4096, 2), (17408, 1), (32768, 3)):
        cfg = lo.SparseMlaConfig(rows_per_slot=rows_per_slot, num_layers=layers)
        kind = kv_frames.Kind("latent", cfg.num_layers, cfg.row_bytes, cfg.shard_rows * cfg.row_stride, cfg.row_stride)
        g = kv_frames.Geometry.contiguous(kind, rows_per_slot // kv_frames.FRAME_ROWS)
        checked = 0
        for user in (0, 3, cfg.users - 1):
            table = g.identity_table(user)
            for layer in range(layers):
                for row in list(range(0, rows_per_slot, 37)) + [rows_per_slot - 1]:
                    want = lo.kv_row_location(cfg, cfg.slot(user, layer), row, 8)
                    got = kv_frames.locate(table, g.slot_bytes(layer), row, cfg.row_stride)
                    assert tuple(got) == tuple(want), (rows_per_slot, user, layer, row, got, want)
                    checked += 1
        assert checked


def test_selection_table_frames():
    """The frame form's table: ids are stripe rows of a conversation of several frames (beyond rows_per_slot), each
    checked against its row's frame table and, with the positions, against what the row may see."""
    import pytest

    from models.experimental.hy4_preview.fused.kv_store import frames as fr

    cfg = lo.SparseMlaConfig(rows_per_slot=256)
    two = [0, 4352] + [fr.UNMAPPED] * (fr.TABLE_FRAMES - 2)  # a conversation of two frames
    one = [8704] + [fr.UNMAPPED] * (fr.TABLE_FRAMES - 1)
    tables = [two, one] + [one] * (cfg.users - 2)
    none = torch.zeros(0, dtype=torch.long)
    lists = [torch.tensor([5, 255, 256, 300]), torch.tensor([7])] + [none] * (cfg.users - 2)
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
        lo.selection_table(cfg, lists)  # (the slot form refuses stripe rows of a second frame)
    t = lo.selection_table_frames(cfg, lists, tables)
    assert t[0].tolist()[:3] == [4, 1, 0] and t[1, :4].tolist() == [5, 255, 256, 300]
    # with the rows' positions on chip 3: position_of(3, 300) is the last the first row may hold
    own = [fr.position_of(3, 300), fr.position_of(3, 7)] + [None] * (cfg.users - 2)
    assert torch.equal(lo.selection_table_frames(cfg, lists, tables, chip=3, positions=own), t)
    for bad_lists, bad_tables, bad_own, why in (
        ([torch.tensor([5, 512])] + lists[1:], tables, own, "an id in a frame the table does not map"),
        ([lists[0], torch.tensor([7, 256])] + lists[2:], tables, own, "a second frame the one-frame row does not have"),
        ([torch.tensor([5, 5])] + lists[1:], tables, own, "a repeated id"),
        (lists, tables, [fr.position_of(3, 299)] + own[1:], "an id beyond the row's own position"),
        (lists[:2] + [torch.tensor([1])] + lists[3:], tables, own, "an idle row with ids"),
        (lists, tables, [fr.position_of(3, 600)] + own[1:], "the own position's frame is not mapped"),
    ):
        with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
            lo.selection_table_frames(cfg, bad_lists, bad_tables, chip=3, positions=bad_own)
