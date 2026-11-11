# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: in a packed forward each attention step pre-sends the sparse-KV push's prefix for the step that runs
after it (packed_push_targets, TtPrefillBlock._packed_next_step). Simulates a pipeline stage's step order over GLM-5.3
layer kinds and checks every pre-send: it targets the next step, carries that step's indices, slot and start, and
every prefix row it reads was written before it runs."""

import itertools

import pytest

from models.demos.common.prefill.prefill_chunk import PrefillChunk
from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import PushStep, packed_push_targets
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock

C = 64  # chunk size, tokens
E = PrefillChunk(0, 0, 0)


@pytest.mark.parametrize(
    "nonempty, expected",
    [
        ([True] * 4, {0: (0, 1), 1: (0, 2), 2: (0, 3), 3: (1, 0)}),
        ([True, True, True, False], {0: (0, 1), 1: (0, 2), 2: (1, 0)}),
        ([True, False, False, False], {0: (1, 0)}),
        ([False, True, False, True], {1: (0, 3), 3: (1, 1)}),
        ([False] * 4, {}),
        ([True], {0: (1, 0)}),
    ],
    ids=["all", "empty_tail", "one", "gaps", "none", "unpacked"],
)
def test_packed_push_targets(nonempty, expected):
    assert packed_push_targets(nonempty) == expected


# Forwards of the GLM multi-user test's packed schedule, plus gaps and one slot filling a whole forward.
FORWARDS = [
    [PrefillChunk(0, 0, C), PrefillChunk(2, 0, 14), PrefillChunk(1, 0, C), E],
    [PrefillChunk(0, C, 2 * C), PrefillChunk(1, C, C + 11), PrefillChunk(3, 0, C), PrefillChunk(3, C, 2 * C)],
    [PrefillChunk(0, 2 * C, 3 * C), E, E, E],
    [E, PrefillChunk(4, 0, C), E, PrefillChunk(5, 0, 7)],
    [PrefillChunk(6, k * C, (k + 1) * C) for k in range(4)],
]
# Pipeline stages over GLM-5.3 layer kinds: starting on a full or a shared layer, consecutive shared layers.
STAGES = [
    ["full", "full", "full", "shared"],
    ["shared", "full", "shared", "shared"],
    ["full", "shared", "shared", "shared", "full", "shared"],
]


def _simulate(kinds, chunks, traced):
    """Runs one packed forward's attention steps in the block's order and checks each pre-send. Returns
    (pre-sends, pushed steps, step order); every pushed (shared) step but the stage's first must receive one."""
    n = len(chunks)
    running = [traced or not c.is_empty for c in chunks]
    targets = packed_push_targets(running)
    order = [(layer, k) for layer in range(len(kinds)) for k in range(n) if running[k]]
    position = {step: i for i, step in enumerate(order)}
    # Traced runs every position; its empty ones in a scratch slot. Metadata stands in for slot and start on device.
    metadata = [("meta", k) for k in range(n)] if traced else None
    slot = lambda k: ("scratch" if chunks[k].is_empty else chunks[k].slot_id) if traced else chunks[k].slot_id
    written = {}  # (layer, slot) -> rows written so far in this forward, as (start, end) ranges with the step index
    injected = [("seed", k) for k in range(n)] if kinds[0] == "shared" else None  # an upstream stage's indices
    presends, received = [], set()
    for layer, kind in enumerate(kinds):
        indexer_indices = injected if kind == "shared" else None
        chunk_indices = []
        for k in range(n):
            if not running[k]:
                chunk_indices.append(None)
                continue
            step = (layer, k)
            # The step writes its chunk's KV before its attention (and before its pre-send is enqueued).
            if not chunks[k].is_empty:
                written.setdefault((layer, slot(k)), []).append(
                    (chunks[k].actual_start, chunks[k].actual_end, position[step])
                )
            own_indices = indexer_indices[k] if indexer_indices is not None else ("full", layer, k)
            nxt = TtPrefillBlock._packed_next_step(
                targets[k],
                chunks=chunks,
                metadata=metadata,
                indexer_indices=indexer_indices,
                own=k,
                chunk_indices=chunk_indices,
            )
            chunk_indices.append(own_indices)
            if nxt is None:
                # Only a full layer's next chunk is skipped: it computes its own top-k.
                assert kind == "full" and targets[k][0] == 0, step
                continue
            assert isinstance(nxt, PushStep)
            target = (layer + nxt.layer_offset, targets[k][1])
            # ttMLA pre-sends only for a pushed (shared) step in this stage.
            if target[0] >= len(kinds) or kinds[target[0]] != "shared":
                continue
            assert position[target] == position[step] + 1, f"{step} pre-sends for {target}, not the next step"
            j = target[1]
            # The target step's own indices: injected ones, or (next layer) the full layer's for chunk j.
            expected = injected[j] if kind == "shared" else ("full", layer, j)
            got = own_indices if nxt.indices is None else nxt.indices
            assert got == expected, f"{step} -> {target}: indices {got}, expected {expected}"
            if traced:
                assert nxt.metadata == metadata[j]
            else:
                assert (nxt.cache_user_id, nxt.kv_actual_isl) == (chunks[j].slot_id, chunks[j].actual_start)
            # PREFIX reads rows below the target chunk's start; any of them this forward writes is written already.
            for start, end, at in written.get((target[0], slot(j)), []):
                if start < chunks[j].actual_start:
                    assert at <= position[step], f"{step} pre-sends rows [{start}, {end}) written at step {at}"
            for later in order[position[step] + 1 :]:
                lk = later[1]
                if later[0] == target[0] and slot(lk) == slot(j) and not chunks[lk].is_empty:
                    assert chunks[lk].actual_start >= chunks[j].actual_start, (
                        f"{step} pre-sends {target}'s prefix before {later} writes part of it"
                    )
            presends.append((step, target))
            received.add(target)
        if kind == "full":
            injected = chunk_indices
    pushed = [step for step in order if kinds[step[0]] == "shared"]
    missing = [step for step in pushed if step not in received and position[step] > 0]
    assert not missing, f"pushed steps without a pre-send: {missing}"
    return presends, pushed, order


@pytest.mark.parametrize("traced", [False, True], ids=["eager", "traced"])
@pytest.mark.parametrize("kinds", STAGES, ids=lambda kinds: "-".join(k[0] for k in kinds))
@pytest.mark.parametrize("f", range(len(FORWARDS)))
def test_packed_presend_targets_next_step(kinds, f, traced):
    chunks = FORWARDS[f]
    presends, pushed, order = _simulate(kinds, chunks, traced)
    # Every pushed step but the stage's first got its prefix ahead of time, from exactly one pre-send.
    assert len(presends) == len([s for s in pushed if s != order[0]])


def test_packed_presend_exhaustive_small():
    """Every forward of up to three chunks over two users at up to three chunks each, on a stage of every kind."""
    slots = [(u, k) for u in range(2) for k in range(3)] + [None]
    for forward in itertools.product(slots, repeat=3):
        # A slot's chunks run in order within a forward (validate_chunks), and only once each.
        real = [c for c in forward if c is not None]
        if len(set(real)) != len(real):
            continue
        if any(a[0] == b[0] and a[1] > b[1] for i, a in enumerate(real) for b in real[i + 1 :]):
            continue
        chunks = [E if c is None else PrefillChunk(c[0], c[1] * C, (c[1] + 1) * C) for c in forward]
        for kinds in STAGES:
            for traced in (False, True):
                _simulate(kinds, chunks, traced)
