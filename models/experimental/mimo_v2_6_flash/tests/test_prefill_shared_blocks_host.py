# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks (no device) of prefill_forward's pass order when a prefix hit reads KV blocks another row of the same
step writes.

vLLM caches a block when it schedules the block's computation, so a request admitted in the step that computes a
shared prefix (three Video-MME questions on one video arriving together: the first one's chunk computes the video,
the others hit its blocks) gets a prefix hit on blocks no pass has written yet. Each row is a pass of its own on TT,
and the rows ran in row order (replayed chunks first), so the hit rows attended stale K / V and answered garbage
(Video-MME short: 142 / 762 long-video answers, docs 34 / 35 among them). Such a row must run after the rows that
write its blocks, eagerly."""

import torch

from models.experimental.mimo_v2_6_flash.tt import generator_vllm as gv
from models.experimental.mimo_v2_6_flash.tests.test_multimodal_adapter_host import adapter

# Two KV cache groups as in serving with the pair split: the global layers' 128-token blocks and a SWA group of 64.
BLOCK_SIZES = {0: 128, 1: 64}
WIDTH = 64  # table entries per row (8192 positions of the 128-token group)


def tables_for(rows):
    """rows: per prefill row {group: [block ids]} -> per-layer tables (layer 0 in group 0, layer 1 in group 1)."""
    out = []
    for g in (0, 1):
        t = torch.zeros(len(rows), WIDTH * (2 if g else 1), dtype=torch.int32)
        for r, row in enumerate(rows):
            ids = torch.tensor(row[g], dtype=torch.int32)
            t[r, : ids.numel()] = ids
        out.append(t)
    return out


def shared_prefix_tables(writer_end, reader_hit, reader_end):
    """Row 1 (the writer) owns blocks for [0, writer_end); row 0 (the reader) shares the writer's blocks below
    ``reader_hit`` and has its own after it."""
    rows = [{}, {}]
    for g, bs in BLOCK_SIZES.items():
        writer = list(range(100 * (g + 1), 100 * (g + 1) + -(-writer_end // bs)))
        shared = writer[: reader_hit // bs]
        own = list(range(1000 * (g + 1), 1000 * (g + 1) + -(-reader_end // bs) - len(shared)))
        rows[0][g], rows[1][g] = shared + own, writer
    return tables_for(rows)


def with_tables(monkeypatch, a, tables, replica, traced_rows=()):
    a.layer_groups, a.group_block_sizes, a.swa_groups = [0, 1], dict(BLOCK_SIZES), {1}
    monkeypatch.setattr(a, "_layer_tables", lambda *args: tables)
    monkeypatch.setattr(a, "_group_rows", gv.MiMoV2ForCausalLM._group_rows.__get__(a))
    calls = []
    enqueue = replica.prefill_enqueue

    def record(chunk, slot, start_pos=0, allow_trace=True, **kw):
        calls.append((slot, start_pos, chunk.numel(), allow_trace))
        return enqueue(chunk, slot, start_pos=start_pos, allow_trace=False, **kw)

    replica.prefill_enqueue = record
    # Chunks of the given slots would replay a trace (a later chunk of the full budget, say).
    replica._chunk_trace = lambda slot, start, tokens: {"key": (128,)} if slot in traced_rows else None
    return calls


def video(offset, length, identifier="vid"):
    """A video-like item: rows [offset + 1, offset + length - 1) are embeddings, the ends are its marker tokens."""
    is_embed = torch.ones(length, dtype=torch.bool)
    is_embed[0] = is_embed[-1] = False
    grid = torch.tensor([1, 2, 2 * (length - 2)])
    return {
        "modality": "video",
        "identifier": identifier,
        "offset": offset,
        "length": length,
        "is_embed": is_embed,
        "data": {"pixel_values_videos": torch.full((int(grid.prod()), 6), 3.0), "video_grid_thw": grid},
    }


def test_prefix_hit_on_blocks_of_this_step_runs_after_the_writer(monkeypatch):
    """Row 0 (slot 0): a request admitted with a prefix hit at 1920 on blocks that row 1 (slot 1) computes in this step
    ([1024, 2048), its last chunk). Row 0 must run after row 1 even though its row comes first and its chunk would
    replay a trace."""
    for cls in (gv.MiMoV2ForCausalLM, gv.MiMoV2OmniForCausalLM):
        a, replica = adapter(monkeypatch, cls)
        tables = shared_prefix_tables(writer_end=2048, reader_hit=1920, reader_end=2000)
        calls = with_tables(monkeypatch, a, tables, replica, traced_rows={0})
        tokens = torch.zeros(2, 2048, dtype=torch.int32)
        a.prefill_forward(tokens, [1920, 1024], [2000, 2048], None, None, empty_slots=[0, 1], mm_items=[None, None])
        assert [c[:3] for c in calls] == [(1, 1024, 1024), (0, 1920, 80)], calls
        assert calls[1][3] is False, "a row reading blocks written in its own step must not replay a trace"


def test_prefix_hit_on_blocks_of_earlier_steps_keeps_the_order(monkeypatch):
    """The same hit on blocks an earlier step wrote (row 1 now computes [2048, 2560)): nothing to wait for, so the
    order (and the replay) stay as they were."""
    a, replica = adapter(monkeypatch, gv.MiMoV2ForCausalLM)
    tables = shared_prefix_tables(writer_end=2560, reader_hit=1920, reader_end=2000)
    calls = with_tables(monkeypatch, a, tables, replica, traced_rows={0})
    tokens = torch.zeros(2, 2560, dtype=torch.int32)
    a.prefill_forward(tokens, [1920, 2048], [2000, 2560], None, None, empty_slots=[0, 1])
    assert [c[:4] for c in calls] == [(0, 1920, 80, True), (1, 2048, 512, False)], calls


def test_video_questions_sharing_a_video_in_one_step(monkeypatch):
    """The Video-MME pattern: the first question's chunk computes the end of the video ([1024, 2048)); two more
    questions on the same video, admitted in that step, hit its blocks (at 1920 and 1792; the second hit lands inside
    the video, so its chunk carries item rows). Both run after the writer, the one hitting at 1920 (whose blocks the
    other one does not write) in row order after it."""
    a, replica = adapter(monkeypatch)
    item = video(4, 2000)  # positions [4, 2004)
    rows = [{}, {}, {}]
    for g, bs in BLOCK_SIZES.items():
        writer = list(range(100 * (g + 1), 100 * (g + 1) + 2048 // bs))
        rows[2][g] = writer
        rows[0][g] = writer[: 1920 // bs] + [5000 + g]
        rows[1][g] = writer[: 1792 // bs] + [6000 + 10 * g + k for k in range(-(-(2040 - 1792) // bs))]
    calls = with_tables(monkeypatch, a, tables_for(rows), replica)
    tokens = torch.zeros(3, 2048, dtype=torch.int32)
    a.prefill_forward(
        tokens,
        [1920, 1792, 1024],
        [2030, 2040, 2048],
        None,
        None,
        empty_slots=[0, 1, 2],
        mm_items=[[item], [item], [item]],
    )
    assert [c[:3] for c in calls] == [(2, 1024, 1024), (0, 1920, 110), (1, 1792, 248)], calls
    # The hit rows' chunks carry the item rows of their range: [1920, 2003) and [1792, 2003).
    embeds = {p[1]: p[4] for p in replica.passes}
    assert embeds[0].tolist() == list(range(0, 2003 - 1920))
    assert embeds[1].tolist() == list(range(0, 2003 - 1792))


def test_reader_of_a_reader_runs_after_both(monkeypatch):
    """Row 0 hits blocks row 1 writes; row 1 hits blocks row 2 writes (all admitted together): 2, 1, 0."""
    a, replica = adapter(monkeypatch, gv.MiMoV2ForCausalLM)
    rows = [{}, {}, {}]
    for g, bs in BLOCK_SIZES.items():
        base = 100 * (g + 1)
        rows[2][g] = list(range(base, base + 1024 // bs))  # writes [0, 1024)
        rows[1][g] = rows[2][g][: 768 // bs] + list(range(base + 50, base + 50 + (1536 - 768) // bs))  # [768, 1536)
        rows[0][g] = rows[1][g][: 1280 // bs] + [9000 + g]  # hits 1280: blocks of rows 2 and 1
    calls = with_tables(monkeypatch, a, tables_for(rows), replica)
    tokens = torch.zeros(3, 1536, dtype=torch.int32)
    a.prefill_forward(tokens, [1280, 768, 0], [1300, 1536, 1024], None, None, empty_slots=[0, 1, 2])
    assert [c[0] for c in calls] == [2, 1, 0], calls


def test_null_block_is_not_a_dependency(monkeypatch):
    """vLLM's null block 0 (SWA blocks out of every window, table padding) is in every table: not a shared block."""
    a, replica = adapter(monkeypatch, gv.MiMoV2ForCausalLM)
    rows = [{}, {}]
    for g, bs in BLOCK_SIZES.items():
        rows[1][g] = [0] * (1024 // bs) + list(range(300 + 100 * g, 300 + 100 * g + 1024 // bs))  # writes [1024, 2048)
        rows[0][g] = [0] * (1920 // bs) + [7000 + g]  # its own prefix, out of the window
    calls = with_tables(monkeypatch, a, tables_for(rows), replica)
    tokens = torch.zeros(2, 2048, dtype=torch.int32)
    a.prefill_forward(tokens, [1920, 1024], [2000, 2048], None, None, empty_slots=[0, 1])
    assert [c[0] for c in calls] == [0, 1], calls


def test_freed_sliding_window_blocks_are_not_a_dependency(monkeypatch):
    """vLLM frees a running request's sliding-window blocks below its window and may hand them to another request,
    but the request's table row keeps their ids. Row 0 (continuing at 1920) still lists SWA blocks 300.. for [0, 512),
    which row 1 now writes ([0, 512) of a new request); row 0 reads only its window [1792, 1920) of the SWA group, so
    the two are independent and keep their order (no cycle when row 1 also lists a stale block row 0 writes)."""
    a, replica = adapter(monkeypatch, gv.MiMoV2ForCausalLM)
    rows = [{}, {}]
    rows[0][0] = list(range(100, 100 + 2048 // 128))  # its own global blocks
    rows[0][1] = list(range(300, 308)) + list(range(400, 400 + (2048 - 512) // 64))  # stale ids 300.. below the window
    rows[1][0] = list(range(500, 504))
    rows[1][1] = list(range(300, 308))  # reallocated: row 1 writes [0, 512) into them
    calls = with_tables(monkeypatch, a, tables_for(rows), replica)
    tokens = torch.zeros(2, 2048, dtype=torch.int32)
    a.prefill_forward(tokens, [1920, 0], [2048, 600], None, None, empty_slots=[0, 1])
    assert [c[0] for c in calls] == [0, 1], calls
    # The same block inside row 0's window is a real dependency: row 0 reads [1792, 1920) of the SWA group.
    rows[0][1][1792 // 64] = 301
    calls.clear()
    monkeypatch.setattr(a, "_layer_tables", lambda *args: tables_for(rows))
    a.prefill_forward(tokens, [1920, 0], [2048, 600], None, None, empty_slots=[0, 1])
    assert [c[0] for c in calls] == [1, 0], calls
