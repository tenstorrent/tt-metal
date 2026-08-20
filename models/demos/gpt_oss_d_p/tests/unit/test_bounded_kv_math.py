# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU-only unit test for the bounded sliding-window KV cache host math (PR1). No device, no ttnn ops —
pure torch/python against the production helpers in ``tt/attention/kv_cache.py``:

  * the CIRCULAR WRITE: simulating the block-cyclic writer host-side (documented offset math:
    ``chip = (p mod C) / cl``, ``row = ((p div C) mod m) * cl + (p mod cl)``) and checking that
    writing chunks 0..N-1 with the host-side ``kv_actual mod capacity`` leaves exactly the LAST
    ``capacity`` positions resident, at the right (chip, row), for every chunk count — including
    across the wrap;
  * the bounded UN-ROTATION: ``bounded_blockcyclic_positions`` (the helper gather_layer /
    kv_cache_pcc_check use) recovers the simulated cache's occupancy exactly, including the
    partially-filled (G < m slabs) and never-written cases, and degenerates to the legacy
    ``blockcyclic_positions`` layout before the first wrap;
  * the LAYER REMAP: ``build_layer_map`` / ``GptOssKVCache.layer_view`` slot math for gpt-oss's
    alternating pattern (36 layers, even=sliding) and an arbitrary mask, plus the flag-off legacy
    view and the one-shot capacity rejection.

Run:
    pytest models/demos/gpt_oss_d_p/tests/unit/test_bounded_kv_math.py
"""

import pytest
import torch

from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.gpt_oss_d_p.tt.attention.kv_cache import (
    GptOssKVCache,
    bounded_blockcyclic_positions,
    build_layer_map,
    sliding_capacity_tokens,
)


def _simulate_circular_writes(sp, chunk_global, capacity, n_chunks):
    """Host-side simulation of update_padded_kv_cache driven with kv_actual' = kv_actual mod capacity.

    Returns the per-chip cache as a [sp, cap_local] tensor of the global position each row holds
    (-1 = never written), after writing chunks 0..n_chunks-1. Placement uses the documented writer
    math on the TRUE global position p: chip = (p mod C) // cl, row = ((p // C) mod m)*cl + (p mod cl)
    — equivalent to the kernel's update_idxt walk at chunk-aligned offsets, because the kernel derives
    its offset from kv_actual and the chunk's own size (never the capacity), so the modulo just picks
    slab (p // C) mod m.
    """
    cl = chunk_global // sp
    m = capacity // chunk_global
    cap_local = capacity // sp
    cache = torch.full((sp, cap_local), -1, dtype=torch.long)
    for t in range(n_chunks):
        kv_actual = t * chunk_global
        assert kv_actual % chunk_global == 0 and capacity % chunk_global == 0  # write-path contract
        for q in range(chunk_global):  # chunk-local position; chip q//cl gets the chunk's rows [q*cl,...)
            p = t * chunk_global + q
            chip = (p % chunk_global) // cl
            row = ((p // chunk_global) % m) * cl + (p % cl)
            cache[chip, row] = p
    return cache


@pytest.mark.parametrize(
    "sp, chunk_global, m, n_chunks",
    [
        (4, 256, 2, 5),  # the design's wrap case: C=256, sp=4, m=2, chunks 0..4 (wraps at chunk 2)
        (4, 256, 2, 1),  # before the first wrap (G < m): degenerates to the legacy layout
        (2, 128, 3, 7),  # odd slab count, deeper wrap
        (1, 64, 2, 4),  # single-chip SP degenerate
    ],
)
def test_circular_write_keeps_last_capacity_positions(sp, chunk_global, m, n_chunks):
    """(1) After every chunk 0..t the resident set is EXACTLY the last min((t+1)*C, capacity) positions,
    each at its documented (chip, row) — i.e. the modulo write is a perfect circular buffer."""
    capacity = m * chunk_global
    cl = chunk_global // sp
    cap_local = capacity // sp
    for t in range(n_chunks):
        cache = _simulate_circular_writes(sp, chunk_global, capacity, t + 1)
        written = (t + 1) * chunk_global
        expected = set(range(max(0, written - capacity), written))
        resident = set(cache[cache >= 0].tolist())
        assert resident == expected, f"after chunk {t}: resident {sorted(resident)[:8]}... != last-{capacity}"
        # Every resident position sits at its documented placement (and nowhere else — the sets match
        # and the cache has exactly `capacity` cells, so placement is a bijection onto the window).
        for chip in range(sp):
            for row in range(cap_local):
                p = int(cache[chip, row])
                if p < 0:
                    continue
                assert (p % chunk_global) // cl == chip
                assert ((p // chunk_global) % m) * cl + (p % cl) == row


@pytest.mark.parametrize(
    "sp, chunk_global, m, n_chunks",
    [(4, 256, 2, 5), (4, 256, 2, 1), (2, 128, 3, 7), (1, 64, 2, 4)],
)
def test_bounded_unrotation_matches_writer(sp, chunk_global, m, n_chunks):
    """(2) The production un-rotation recovers the simulated occupancy exactly, for every write count."""
    capacity = m * chunk_global
    cap_local = capacity // sp
    for t in range(n_chunks + 1):  # t = chunks written so far, including 0 (nothing written)
        cache = _simulate_circular_writes(sp, chunk_global, capacity, t)
        pos = bounded_blockcyclic_positions(sp, chunk_global, capacity, t * chunk_global)
        # Device-major flatten of the simulation == the helper's [sp * cap_local] row order.
        assert torch.equal(pos, cache.reshape(-1)), f"mismatch after {t} chunks"
        # written_tokens mid-chunk counts the whole (padded) chunk — same occupancy as the full chunk.
        if t > 0:
            partial = (t - 1) * chunk_global + 1
            assert torch.equal(pos, bounded_blockcyclic_positions(sp, chunk_global, capacity, partial))
    # Before the first wrap (G <= m) the bounded layout IS the legacy block-cyclic layout on the
    # written rows: check against the DeepSeek helper the unbounded gather uses.
    legacy = blockcyclic_positions(sp, chunk_global, capacity)
    pos = bounded_blockcyclic_positions(sp, chunk_global, capacity, m * chunk_global)
    assert torch.equal(pos, legacy)


def _mk_cache(layer_types, *, bounded, num_users=2, max_seq_len=1024, sp=4, sliding_capacity=512):
    """Slot-math-only GptOssKVCache: tensor fields carry string tags so layer_view's routing (which
    tensor pair a layer lands in) is directly observable without a device."""
    return GptOssKVCache(
        k="K_FULL",
        v="V_FULL",
        num_users=num_users,
        num_layers=len(layer_types),
        max_seq_len=max_seq_len,
        sp=sp,
        k_sliding="K_SLIDE" if bounded else None,
        v_sliding="V_SLIDE" if bounded else None,
        sliding_capacity=sliding_capacity if bounded else None,
        layer_map=build_layer_map(layer_types) if bounded else None,
        bounded_sliding=bounded,
    )


GPT_OSS_TYPES = ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(36)]


def test_layer_map_gpt_oss_alternating():
    """(3a) gpt-oss's 36-layer alternating pattern: even layers sliding, 18 of each, ordinal = L//2."""
    layer_map = build_layer_map(GPT_OSS_TYPES)
    assert len(layer_map) == 36
    for L, (is_sliding, ordinal, n_type) in enumerate(layer_map):
        assert is_sliding == (L % 2 == 0)
        assert n_type == 18
        assert ordinal == L // 2


@pytest.mark.parametrize(
    "layer_types",
    [
        GPT_OSS_TYPES,
        # arbitrary mask: uneven runs, starts and ends full
        [
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
            "full_attention",
        ],
    ],
    ids=["gpt-oss-36", "arbitrary-7"],
)
def test_layer_view_slot_math(layer_types):
    """(3b) layer_view routes each (user, layer) to its type's cache, and per type the batch slots are
    a bijection onto [0, num_users * n_type) — no collisions, no holes, user-major."""
    num_users = 2
    kv = _mk_cache(layer_types, bounded=True, num_users=num_users)
    n_slide = sum(t == "sliding_attention" for t in layer_types)
    n_full = len(layer_types) - n_slide
    seen = {True: [], False: []}
    for user in range(num_users):
        for L, lt in enumerate(layer_types):
            k, v, batch_idx, capacity_tokens, bounded = kv.layer_view(user, L)
            is_sliding = lt == "sliding_attention"
            assert bounded == is_sliding
            if is_sliding:
                assert (k, v) == ("K_SLIDE", "V_SLIDE")
                assert capacity_tokens == kv.sliding_capacity
                assert batch_idx == user * n_slide + kv.layer_map[L][1]
            else:
                assert (k, v) == ("K_FULL", "V_FULL")
                assert capacity_tokens == kv.max_seq_len
                assert batch_idx == user * n_full + kv.layer_map[L][1]
            seen[is_sliding].append(batch_idx)
    assert sorted(seen[True]) == list(range(num_users * n_slide))
    assert sorted(seen[False]) == list(range(num_users * n_full))


def test_layer_view_legacy_when_flag_off():
    """Flag off => the legacy packed view: slot = user * num_layers + layer, full capacity, unbounded."""
    kv = _mk_cache(GPT_OSS_TYPES, bounded=False)
    for user in range(kv.num_users):
        for L in range(kv.num_layers):
            assert kv.layer_view(user, L) == ("K_FULL", "V_FULL", user * kv.num_layers + L, kv.max_seq_len, False)


def test_sliding_capacity_chunked_and_oneshot():
    """Chunked => 2 chunk slabs; one-shot => rejected until the C++ wrap_seq change (PR follow-up)."""
    assert sliding_capacity_tokens(5120, 1024, sp=4, sliding_window=128) == 2048
    with pytest.raises(NotImplementedError, match="wrap_seq"):
        sliding_capacity_tokens(1024, 1024, sp=4, sliding_window=128)
