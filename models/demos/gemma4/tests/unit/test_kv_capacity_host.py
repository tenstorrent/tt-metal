# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""KV capacity arithmetic (``tt/kv_capacity.py``), host-only: the plugin's pool, the lane engine's per-lane vs
box users, the per-chip bytes (the replicated global head), the DRAM-limited pools of the dense Galaxy engines.

    pytest models/demos/gemma4/tests/unit/test_kv_capacity_host.py -p no:cacheprovider
"""

import os
from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4.tt.attention import SPEC_RING_HEADROOM_ENV, bounded_ring_modulo
from models.demos.gemma4.tt.attention.prefill import (
    _bounded_ring_tail_blocks,
    _owner_lane_device_index,
    _paged_tail_block_range,
)
from models.demos.gemma4.tt.capacity import (
    BFP8_BYTES,
    GEMMA4_GEOMETRY,
    engine_capacity_plan,
    kv_bytes_per_token_per_chip,
    lane_engine_capacity,
    local_kv_heads,
    plugin_pool_tokens,
    pool_tokens_for_dram,
    sharded_minimum_bytes_per_token_per_chip,
    sliding_cache_bytes_per_chip,
    users_at_context,
)
from models.demos.gemma4.tt.generator import ChunkedPrefillPageTableGuardMixin
from models.demos.gemma4.tt.generator import ChunkedPrefillPageTableGuardMixin as Generator

try:
    import vllm  # noqa: F401

    _HAS_VLLM = True
except ImportError:  # the CI unit runners have no vLLM: those sections skip
    _HAS_VLLM = False

K256 = 262144
GB = 1e9


def test_the_plugin_pool_adds_the_output_reservation_and_the_hybrid_headroom():
    tokens, blocks = plugin_pool_tokens(3145728, 32, 128, sliding_window=1024, hybrid_groups=True)
    assert tokens == 3145728 + 128 * 32 + 1024 * 32 * 8 == 3411968 and blocks == 26656
    tokens, blocks = plugin_pool_tokens(262144, 32, 64, sliding_window=1024, hybrid_groups=False)
    assert tokens == 262144 + 64 * 32 and blocks == tokens // 64


def test_the_lane_engines_capacity_line_is_one_lanes_view_and_the_box_holds_lanes_times_it():
    pool, _ = plugin_pool_tokens(3145728, 32, 128, sliding_window=1024, hybrid_groups=True)
    per_lane, box = lane_engine_capacity(pool, 4, K256, 128, sliding_window=1024, sliding_groups=5)
    assert per_lane == 12 and box == 48, "the catalog's '12 users at 256K' is per lane; vLLM's line counts one lane"
    # 36 box users at 256K need 9 per lane: a 2.36M-token pool per column, 25 % below today's
    needed = 9 * K256 + 5 * (1024 + 128) * 9
    assert users_at_context(needed + 128 * 32, K256, 128, sliding_window=1024, sliding_groups=5) == 9
    # 128 users at <= 64K: 32 per lane, well inside today's pool (47 per lane at 64K)
    assert lane_engine_capacity(pool, 4, 65536, 128, sliding_window=1024, sliding_groups=5) == (47, 188)
    assert lane_engine_capacity(32 * (65536 + 5 * 1152), 4, 65536, 128, sliding_window=1024, sliding_groups=5)[0] == 32


def test_global_heads_fewer_than_the_tp_group_are_replicated_one_per_chip():
    assert (
        local_kv_heads(2, 8) == 1
        and local_kv_heads(8, 8) == 1
        and local_kv_heads(16, 8) == 2
        and local_kv_heads(1, 4) == 1
    )
    g = GEMMA4_GEOMETRY["26B"]
    per_chip = kv_bytes_per_token_per_chip(
        global_layers=g["global_layers"],
        global_kv_heads=g["global_kv_heads"],
        global_head_dim=g["global_head_dim"],
        tp=8,
    )
    assert per_chip == pytest.approx(5 * 2 * 512 * BFP8_BYTES) == pytest.approx(5440)
    floor = sharded_minimum_bytes_per_token_per_chip(global_layers=5, global_kv_heads=2, global_head_dim=512, tp=8)
    assert floor == pytest.approx(
        5440 / 4
    ), "two global heads over eight chips: each head on four chips, 4x the sharded floor"
    # the 10-23 reading: 3.78M tokens per column = 29 GB per chip -> 7.7 KB per token, of which the global pages are 5.4
    assert 29 * GB / 3.78e6 == pytest.approx(7672, rel=0.01)


def test_dense_galaxy_engines_per_chip_bytes_match_the_ledger():
    g31 = GEMMA4_GEOMETRY["31B"]
    assert kv_bytes_per_token_per_chip(
        global_layers=g31["global_layers"],
        global_kv_heads=g31["global_kv_heads"],
        global_head_dim=g31["global_head_dim"],
        tp=8,
    ) == pytest.approx(10880)
    assert kv_bytes_per_token_per_chip(
        global_layers=g31["global_layers"],
        global_kv_heads=g31["global_kv_heads"],
        global_head_dim=g31["global_head_dim"],
        tp=4,
    ) == pytest.approx(10880), "4 heads over 4 chips: one each, the same as TP8's replicated one"
    g12 = GEMMA4_GEOMETRY["12B"]
    assert kv_bytes_per_token_per_chip(
        global_layers=g12["global_layers"],
        global_kv_heads=g12["global_kv_heads"],
        global_head_dim=g12["global_head_dim"],
        tp=8,
    ) == pytest.approx(8704), "one global head: replicated on every chip at any TP"


def test_dense_galaxy_engine_pools_from_the_chips_dram():
    # 31B: 7.75 GB of weights per chip (ledger), ~2 GB for traces (0.96) and prefill activations -> 7 users; a 3 GB
    # reserve gives 6: the window's capacity line settles it
    plan31 = engine_capacity_plan(
        model=GEMMA4_GEOMETRY["31B"],
        dram_bytes=32 * GB,
        weights_bytes=7.75 * GB,
        reserve_bytes=2.0 * GB,
        context=K256,
        block_size=64,
        users_for_sliding=32,
        tp=8,
    )
    assert plan31["users"] == 7 and plan31["env_pool"] == 7 * K256 == 1835008
    assert (
        engine_capacity_plan(
            model=GEMMA4_GEOMETRY["31B"],
            dram_bytes=32 * GB,
            weights_bytes=7.75 * GB,
            reserve_bytes=3.0 * GB,
            context=K256,
            block_size=64,
            users_for_sliding=32,
            tp=8,
        )["users"]
        == 6
    )
    assert plan31["sliding_bytes"] == pytest.approx(50 * 2 * 2 * 256 * BFP8_BYTES * 32 * (1024 + 64))
    plan12 = engine_capacity_plan(
        model=GEMMA4_GEOMETRY["12B"],
        dram_bytes=32 * GB,
        weights_bytes=3.0 * GB,
        reserve_bytes=2.0 * GB,
        context=K256,
        block_size=64,
        users_for_sliding=32,
        tp=8,
    )
    assert (
        plan12["users"] == 11 and plan12["env_pool"] == 11 * K256 == 2883584
    ), "the catalog's 12B pool already sits at the DRAM floor (one replicated global head)"
    assert (
        pool_tokens_for_dram(
            1 * GB, weights_bytes=2 * GB, reserve_bytes=0, sliding_bytes=0, bytes_per_token=1000, block_size=64
        )
        == 0
    )


def test_sliding_caches_are_window_sized_per_user():
    b = sliding_cache_bytes_per_chip(
        sliding_layers=25, sliding_kv_heads=8, sliding_head_dim=256, tp=8, users=32, sliding_window=1024, block_size=128
    )
    assert b == pytest.approx(25 * 2 * 1 * 256 * BFP8_BYTES * 32 * 1152)


# ===== merged from tests/unit/test_effective_paged_block_size.py: ``_effective_paged_block_size`` must return the smallest per-layer effective block size, so
def _cache(shape):
    return [SimpleNamespace(shape=shape, padded_shape=shape), SimpleNamespace(shape=shape, padded_shape=shape)]


def _layer(head_dim, kv_heads, tp, kv_replicated=False):
    return SimpleNamespace(
        self_attn=SimpleNamespace(
            config=SimpleNamespace(head_dim=head_dim, num_key_value_heads=kv_heads),
            mesh_config=SimpleNamespace(tp=tp),
            weights=SimpleNamespace(kv_replicated=kv_replicated),
        )
    )


def test_tp4_31b_returns_sliding_block_not_full_view():
    # 31B on P150x4: sliding 16 heads -> 4/device at head_dim 256; full 4 heads -> 1/device at 512.
    shared = (10272, 4, 64, 256)
    layers = [_layer(256, 16, 4), _layer(256, 16, 4), _layer(512, 4, 4)]
    fake = SimpleNamespace(model=[SimpleNamespace(layers=layers)])
    kv = [_cache(shared) for _ in layers]
    assert Generator._effective_paged_block_size(fake, kv) == 64


def test_tp8_equal_views_unchanged():
    # Galaxy column: sliding 2/device at 256, full replicated 1/device at 512 -> both 64.
    shared = (23808, 2, 64, 256)
    layers = [_layer(256, 16, 8), _layer(512, 4, 8, kv_replicated=True)]
    fake = SimpleNamespace(model=[SimpleNamespace(layers=layers)])
    kv = [_cache(shared) for _ in layers]
    assert Generator._effective_paged_block_size(fake, kv) == 64


def test_non_hybrid_matching_caches_keep_declared_block():
    layers = [_layer(256, 16, 4), _layer(512, 4, 4)]
    fake = SimpleNamespace(model=[SimpleNamespace(layers=layers)])
    kv = [_cache((100, 4, 32, 256)), _cache((100, 1, 32, 512))]
    assert Generator._effective_paged_block_size(fake, kv) == 32


# ===== merged from tests/unit/test_shared_kv_canonical_shape.py: Shared-KV buffers take the first layer's view (the sliding layout, which the paged kernels reinterpret
try:
    from models.tt_transformers.tt.generator_vllm import _canonical_shared_kv_shapes
except ImportError:  # vLLM absent: this section's tests skip
    pass


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_first_layer_view_wins_shared_buffer():
    specs = [
        ((1024, 2, 64, 256), torch.bfloat16, 0),  # sliding (layer 0)
        ((1024, 2, 64, 256), torch.bfloat16, 1),  # sliding (layer 1)
        ((1024, 1, 64, 512), torch.bfloat16, 0),  # full (layer 5) shares t0
    ]
    out = _canonical_shared_kv_shapes(specs)
    assert out[0] == (1024, 2, 64, 256)
    assert out[1] == (1024, 2, 64, 256)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_shrunk_sliding_blocks_do_not_undersize_shared_buffer():
    specs = [
        ((16, 2, 64, 256), torch.bfloat16, 0),  # bounded-shrunk sliding
        ((1024, 1, 64, 512), torch.bfloat16, 0),  # full needs the whole pool
    ]
    out = _canonical_shared_kv_shapes(specs)
    assert out[0] == (1024, 2, 64, 256)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_unshared_layers_keep_own_shapes():
    specs = [((512, 8, 64, 128), torch.bfloat16, i) for i in range(3)]
    out = _canonical_shared_kv_shapes(specs)
    assert all(out[i] == (512, 8, 64, 128) for i in range(3))


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_inconsistent_per_block_bytes_raise(expect_error):
    specs = [
        ((1024, 2, 64, 256), torch.bfloat16, 0),
        ((1024, 1, 128, 512), torch.bfloat16, 0),  # 2x the per-block elements
    ]
    with expect_error(ValueError, "per-block element counts"):
        _canonical_shared_kv_shapes(specs)


# ===== merged from tests/unit/test_bounded_ring_page_table_cols.py: Bounded sliding page tables follow the ring, not the bare window.
try:
    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM
except ImportError:  # vLLM absent: this section's tests skip
    pass


WINDOW = 1024


BLOCK = 64


def _generator(max_batch=4):
    gen = Gemma4ForCausalLM.__new__(Gemma4ForCausalLM)
    gen._bounded_sliding_kv_cache = True
    gen.model_args = [SimpleNamespace(max_batch_size=max_batch)]
    # no lane fold: the pool is sized per max batch and the ring slots take no lane keywords
    gen.model = [SimpleNamespace(layers=[], mesh_config=None, lane_slots=0)]
    gen._text_config = lambda: SimpleNamespace(
        sliding_window=WINDOW, layer_types=["sliding_attention", "full_attention"]
    )
    gen._sliding_layer_indices = lambda: [0]
    gen._bounded_ring_slots = lambda pt, max_slots, authoritative, row_slots=None, lane_slots=None: list(
        range(int(pt.shape[0]))
    )
    return gen


class _Cache:
    """Stands in for a layer's K/V pair and for either tensor of it.

    The two sizing helpers index the cache differently (``kv_cache[0][0].shape``
    against ``kv_cache[0][layer][0].shape``), so one object answers both.
    """

    shape = (1, 1, BLOCK, 8)

    def __getitem__(self, index):
        return self


def _kv_cache():
    return [[_Cache(), _Cache()]]


def _tables(rows=2, cols=8):
    pt = torch.arange(1, rows * cols + 1, dtype=torch.int32).reshape(rows, cols)
    return [pt, pt]


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_exact_window_ring_keeps_window_sized_tables(monkeypatch):
    monkeypatch.delenv(SPEC_RING_HEADROOM_ENV, raising=False)
    gen = _generator()
    out = gen._pad_sliding_page_tables_for_bounded(_tables(), _kv_cache())
    assert tuple(out[0].shape) == (2, WINDOW // BLOCK)
    assert out[0][1].tolist() == list(range(16, 32))
    assert out[1] is _tables()[1] or torch.equal(out[1], _tables()[1])  # full-attention layer untouched
    assert gen._bounded_sliding_min_page_table_cols(_kv_cache()) == WINDOW // BLOCK


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_headroom_widens_the_tables_to_the_ring(monkeypatch):
    """With ``GEMMA4_SPEC_RING_HEADROOM_BLOCKS`` the modulo is 2048; the page
    table must give every user ring/block_size blocks, contiguous per slot, so
    the pool (ring/block_size * max_batch) and the modulo agree."""
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, str(WINDOW // BLOCK))
    gen = _generator()
    ring = 2 * WINDOW
    cols = ring // BLOCK
    out = gen._pad_sliding_page_tables_for_bounded(_tables(), _kv_cache())
    assert tuple(out[0].shape) == (2, cols)
    assert out[0][0].tolist() == list(range(0, cols))
    assert out[0][1].tolist() == list(range(cols, 2 * cols))
    assert gen._bounded_sliding_min_page_table_cols(_kv_cache()) == cols
    assert gen._bounded_sliding_physical_blocks(BLOCK) == cols * 4


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_pool_and_columns_agree_with_and_without_headroom(monkeypatch):
    for headroom in (None, WINDOW // BLOCK):
        if headroom is None:
            monkeypatch.delenv(SPEC_RING_HEADROOM_ENV, raising=False)
        else:
            monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, str(headroom))
        gen = _generator(max_batch=32)
        cols = gen._bounded_sliding_min_page_table_cols(_kv_cache())
        assert gen._bounded_sliding_physical_blocks(BLOCK) == cols * 32
        assert gen._pad_sliding_page_tables_for_bounded(_tables(), _kv_cache())[0].shape[1] == cols


# ===== merged from tests/unit/test_bounded_ring_slots.py: Bounded sliding rings: slot recycling must never evict a request that a prefill
try:
    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM
except ImportError:  # vLLM absent: this section's tests skip
    pass


def _rows(first_blocks):
    # one page-table row per request, keyed by its first global block id
    return torch.tensor([[b, b + 1, b + 2] for b in first_blocks], dtype=torch.int32)


def _generator_bounded_ring_slots():
    # Only the slot bookkeeping is exercised: a bare namespace with the row-key helper.
    return SimpleNamespace(_bounded_row_key=Gemma4ForCausalLM._bounded_row_key)


def _slots(self, first_blocks, authoritative):
    return Gemma4ForCausalLM._bounded_ring_slots(self, _rows(first_blocks), 32, authoritative)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_full_map_evicts_the_finished_request_not_a_fresh_arrival():
    self = _generator_bounded_ring_slots()
    # A long request decodes alone: the last authoritative running set is {long}.
    (long_slot,) = _slots(self, [5000], authoritative=True)
    # It finishes; 32 new requests arrive through prefill (non-authoritative), one per call.
    new_slots = [_slots(self, [100 + i], authoritative=False)[0] for i in range(32)]
    assert len(set(new_slots)) == 32, "two live requests were handed the same ring"
    assert long_slot in new_slots, "the departed request's ring was not the one recycled"
    assert 5001 not in self._bounded_ring_slot_map


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_stale_keys_from_the_last_running_set_are_recycled_oldest_first():
    self = _generator_bounded_ring_slots()
    old = [_slots(self, [b], authoritative=False)[0] for b in range(10, 42)]
    _slots(self, list(range(10, 42)), authoritative=True)  # all 32 ran together
    # They all finish; a new burst must recycle them without colliding.
    new = [_slots(self, [900 + i], authoritative=False)[0] for i in range(32)]
    assert sorted(new) == sorted(old)
    assert len(set(new)) == 32


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_authoritative_step_resets_the_fresh_set():
    self = _generator_bounded_ring_slots()
    a = _slots(self, [1], authoritative=False)[0]
    _slots(self, [1], authoritative=True)
    assert self._g4_ring_keys_since_authoritative == set()
    b = _slots(self, [2], authoritative=False)[0]
    assert a != b


# ===== merged from tests/unit/test_bounded_ring_spec_headroom.py: A bounded sliding ring must not evict history a candidate query still needs.
try:
    from models.demos.gemma4.tt import generator_vllm as gv
except ImportError:  # vLLM absent: this section's tests skip
    pass


WINDOW_bounded_ring_spec_headroom = 1024


def _evicted_in_window(position, window, ring, candidates):
    """Candidates whose write lands on a slot holding a still-live position."""
    live_lo = position - window + 1
    hit = []
    for j in range(1, candidates + 1):
        displaced = (position + j) - ring  # what that slot held before the write
        if live_lo <= displaced <= position:
            hit.append((j, displaced))
    return hit


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_an_exact_window_ring_evicts_from_the_first_candidate():
    """The defect, stated as arithmetic: this is what an unset headroom gives."""
    assert _evicted_in_window(131072, WINDOW_bounded_ring_spec_headroom, WINDOW_bounded_ring_spec_headroom, 6)[0] == (
        1,
        130049,
    )
    assert len(_evicted_in_window(131072, WINDOW_bounded_ring_spec_headroom, WINDOW_bounded_ring_spec_headroom, 6)) == 6


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_the_reserved_ring_evicts_nothing(monkeypatch):
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, str(WINDOW_bounded_ring_spec_headroom // 64))
    ring = bounded_ring_modulo(WINDOW_bounded_ring_spec_headroom)
    assert ring == 2 * WINDOW_bounded_ring_spec_headroom
    assert _evicted_in_window(131072, WINDOW_bounded_ring_spec_headroom, ring, 6) == []


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_every_candidate_query_keeps_its_full_window(monkeypatch):
    """Victor's check: each candidate query's visible positions, not just the first."""
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, str(WINDOW_bounded_ring_spec_headroom // 64))
    ring = bounded_ring_modulo(WINDOW_bounded_ring_spec_headroom)
    base = 131072
    for q in range(0, 6 + 1):  # query at each candidate position
        pos = base + q
        # positions this query must see, and the slots they occupy
        needed = range(pos - WINDOW_bounded_ring_spec_headroom + 1, pos + 1)
        slots = {p % ring: p for p in needed}
        assert len(slots) == WINDOW_bounded_ring_spec_headroom, f"query {pos} lost positions to slot aliasing"
        # no later candidate write in this same forward lands on those slots
        for j in range(q + 1, 6 + 1):
            assert (base + j) % ring not in slots or slots[(base + j) % ring] == base + j


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_the_model_sizes_the_ring_rather_than_the_operator(monkeypatch):
    """With no headroom set, a speculating class gets a ring that fits its drafts.

    The premise of this test has been wrong twice, so it records what settled it.
    First it asserted the guard warns and continues, on the grounds that "a
    server that cannot boot is worse" than wrong tokens -- tt-metal#57701 is the
    report that this produces plausible, wrong text with one startup warning as
    the only trace. Then it asserted a refusal, on the grounds that doubling the
    ring OOMs the shipped P150x8 config. That was inherited from a comment
    written before tt-metal#57655 sized the bounded page tables from the ring;
    measured afterwards on P150x8 at max_num_seqs=32 with
    GEMMA4_MAX_TOKENS_ALL_USERS=262144, the doubled ring allocates and serves.

    So the headroom is neither warned about nor refused: the model sets it,
    because it alone knows the class, K, the window, and whether bounded sliding
    resolved on. An env var every launch must get right is the wrong interface.
    """
    monkeypatch.delenv(SPEC_RING_HEADROOM_ENV, raising=False)
    # _auto_size_spec_ring writes os.environ directly; monkeypatch cannot undo a
    # value it did not set, so restore it here or it leaks into later tests.
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, "0")
    monkeypatch.delenv(SPEC_RING_HEADROOM_ENV)
    gv._auto_size_spec_ring(
        SimpleNamespace(_SPEC_N=6, __name__="Gemma4DFlash"),
        SimpleNamespace(text_config=SimpleNamespace(sliding_window=WINDOW_bounded_ring_spec_headroom)),
        bounded_sliding=True,
    )
    assert bounded_ring_modulo(WINDOW_bounded_ring_spec_headroom) == 2 * WINDOW_bounded_ring_spec_headroom
    # and the guard that backstops it is satisfied by what the model chose
    gv._reserve_spec_ring_headroom(WINDOW_bounded_ring_spec_headroom, 6, "test")


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_an_operator_value_is_left_alone(monkeypatch):
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, "48")
    gv._reserve_spec_ring_headroom(WINDOW_bounded_ring_spec_headroom, 6, "test")
    assert os.environ[SPEC_RING_HEADROOM_ENV] == "48"


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_the_ring_stays_a_power_of_two(monkeypatch):
    """Chunk starts must be multiples of both the ring and SDPA's q_chunk_size."""
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, str(WINDOW_bounded_ring_spec_headroom // 64))
    ring = bounded_ring_modulo(WINDOW_bounded_ring_spec_headroom)
    assert ring & (ring - 1) == 0


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_the_pool_is_allocated_from_the_ring_not_the_window(monkeypatch):
    """Allocation, page tables and modulo have to move together.

    Sizing the pool from the bare window while wrapping modulo the ring writes
    past the pool, which is why setting the headroom env alone was never
    sufficient.
    """
    from types import SimpleNamespace

    monkeypatch.delenv(SPEC_RING_HEADROOM_ENV, raising=False)
    m = gv.Gemma4DFlashForCausalLM.__new__(gv.Gemma4DFlashForCausalLM)
    m._bounded_sliding_kv_cache = True
    m.model_args = [SimpleNamespace(max_batch_size=32)]
    m.model = [SimpleNamespace(mesh_config=None, lane_slots=0)]  # no lane fold: the pool is per max batch
    monkeypatch.setattr(
        gv.Gemma4DFlashForCausalLM,
        "_text_config",
        lambda self: SimpleNamespace(sliding_window=WINDOW_bounded_ring_spec_headroom),
    )
    window_blocks = (WINDOW_bounded_ring_spec_headroom // 64) * 32
    assert m._bounded_sliding_physical_blocks(64) == window_blocks, "exact ring by default"
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, str(WINDOW_bounded_ring_spec_headroom // 64))
    assert m._bounded_sliding_physical_blocks(64) == window_blocks * 2, "follows the ring"


# ===== merged from tests/unit/test_bounded_ring_tail_host.py: Host-only checks of the bounded-ring sliding-tail rebuild plan (tt/attention/prefill.py
ROW = torch.arange(100, 116, dtype=torch.int32)  # a 16-block ring table row (1024 / 64), ring-local ids


def test_first_continuation_reads_the_leading_blocks():
    # first chunk of 100 tokens, continuation at 100: positions [0, 100) sit in blocks 0 and 1, rows 0..99
    blocks, first_row, take = _bounded_ring_tail_blocks(ROW, 100, 1024, 1024, 64)
    assert (blocks, first_row, take) == ([100, 101], 0, 100)
    # block-aligned offset
    assert _bounded_ring_tail_blocks(ROW, 128, 1024, 1024, 64) == ([100, 101], 0, 128)


def test_wrapped_window_walks_the_ring_in_position_order():
    # continuation at 1100 with a 1024 window: positions [76, 1100) -> slots 76..1023, then 0..75
    blocks, first_row, take = _bounded_ring_tail_blocks(ROW, 1100, 1024, 1024, 64)
    assert take == 1024 and first_row == 12  # 76 - 64
    assert blocks == [100 + b for b in range(1, 16)] + [100, 101]


def test_short_window_and_block_zero():
    row = torch.tensor([0, 1, 2, 3], dtype=torch.int32)  # the remapped sliding table may start at block 0
    assert _bounded_ring_tail_blocks(row, 64, 512, 256, 64) == ([0], 0, 64)
    # positions [172, 300): slots 172..255 (blocks 2, 3) then the wrap 256..299 (block 0); row 44 of the run
    assert _bounded_ring_tail_blocks(row, 300, 128, 256, 64) == ([2, 3, 0], 44, 128)


def test_rejects_what_the_ring_cannot_hold():
    assert _bounded_ring_tail_blocks(ROW, 0, 1024, 1024, 64) is None
    assert _bounded_ring_tail_blocks(ROW, 100, 1024, 1000, 64) is None  # modulo not block-aligned
    assert _bounded_ring_tail_blocks(torch.tensor([7], dtype=torch.int32), 100, 1024, 1024, 64) is None  # too short


# ===== merged from tests/unit/test_paged_sliding_tail.py: Host-only checks for the paged sliding-window tail of a cached-prefix continuation: which blocks are read,
W, BS, POOL = 1024, 64, 4096


def _row(ids):
    return torch.tensor(ids, dtype=torch.int32)


def test_full_window_inside_the_hit():
    row = _row(list(range(1, 21)))  # blocks 1..20 for tokens [0, 1280)
    assert _paged_tail_block_range(row, 1280, W, BS, POOL) == (256, list(range(5, 21)), 0)


def test_short_prefix_takes_everything_cached():
    row = _row([7, 8, 9, 10, 0, 0])
    assert _paged_tail_block_range(row, 256, W, BS, POOL) == (0, [7, 8, 9, 10], 0)


def test_alignment_past_the_kept_window_drops_the_null_block():
    # vLLM kept 16 real blocks for a hit at 1088 tokens; the SDPA alignment moved the offset to 1024, so the
    # window [0, 1024) starts one block before the kept range, which vLLM already nulled (block 0).
    row = _row([0] + list(range(2, 18)) + [0, 0])
    start, blocks, first_row = _paged_tail_block_range(row, 1024, W, BS, POOL)
    assert start == 64 and blocks == list(range(2, 17)) and first_row == 0


def test_unusable_rows_give_none():
    assert _paged_tail_block_range(_row([0, 0, 0, 0]), 256, W, BS, POOL) is None  # all null
    assert _paged_tail_block_range(_row([1, 2, 3, POOL + 5]), 256, W, BS, POOL) is None  # out of pool
    assert _paged_tail_block_range(_row([1, 2, 3, 0]), 200, W, BS, POOL) is None  # the covering block is null
    assert _paged_tail_block_range(_row([1, 2, 3, 4]), 0, W, BS, POOL) is None  # first chunk


def test_chunk_boundary_inside_a_block_is_served_from_the_covering_blocks():
    # vLLM's hybrid groups put the sliding layers on 256-token blocks while the admission chunks prompts at
    # 128-token granularity (shield run 40196840603 / local replay: every warned offset was 128 mod 256).
    bs = 256
    # a 142-token chat prompt split 128 + 14: the window [0, 128) is the first half of its only block
    assert _paged_tail_block_range(_row([201, 0, 0]), 128, W, bs, POOL) == (0, [201], 0)
    # continuation at 640 tokens: blocks for [0, 768), rows [0, 640) of them
    assert _paged_tail_block_range(_row([841, 842, 843, 0]), 640, W, bs, POOL) == (0, [841, 842, 843], 0)
    # continuation at 2176 (= 2048 + 128): the window [1152, 2176) spans blocks 4..8, starting 128 rows in
    row = _row(list(range(10, 30)))
    assert _paged_tail_block_range(row, 2176, W, bs, POOL) == (1152, [14, 15, 16, 17, 18], 128)
    # a row too short for the covering blocks is unusable
    assert _paged_tail_block_range(_row([201]), 384, W, bs, POOL) is None


def test_leading_null_blocks_inside_a_shifted_window():
    # the window [1152, 2176) with the first covering block already nulled by the hybrid allocator
    row = _row([0, 0, 0, 0, 0, 15, 16, 17, 18, 0])
    assert _paged_tail_block_range(row, 2176, W, 256, POOL) == (1280, [15, 16, 17, 18], 0)


def test_owner_lane_shard_index():
    assert _owner_lane_device_index(None) == 0
    assert _owner_lane_device_index(SimpleNamespace(lane_sharded=False)) == 0
    lanes_on_axis_1 = SimpleNamespace(lane_sharded=True, mesh_shape=(8, 4), sp_axis=1, _g4_single_lane_prefill_owner=2)
    assert _owner_lane_device_index(lanes_on_axis_1) == 2
    lanes_on_axis_0 = SimpleNamespace(lane_sharded=True, mesh_shape=(2, 4), sp_axis=0, _g4_single_lane_prefill_owner=1)
    assert _owner_lane_device_index(lanes_on_axis_0) == 4
    lane_parallel = SimpleNamespace(lane_sharded=True, mesh_shape=(8, 4), sp_axis=1, _g4_single_lane_prefill_owner=None)
    assert _owner_lane_device_index(lane_parallel) == 0


# ===== merged from tests/unit/test_generator_bounded_prefill.py
def _generator_with_bounded_window(window=1024):
    generator = object.__new__(ChunkedPrefillPageTableGuardMixin)
    config = SimpleNamespace(cache_position_modulo=window)
    layer = SimpleNamespace(self_attn=SimpleNamespace(config=config))
    generator.model = [SimpleNamespace(bounded_sliding_kv_cache=True, layers=[layer])]
    return generator


def test_activate_sequential_per_layer_row_refreshes_persistent_device_tables():
    """Sequential users must H2D-refresh B=1 persistent page tables.

    Host `_active` is sliced per user, but device buffers are keyed by batch=1
    and reused without content update unless ``update_persistent…`` runs.
    Rows are selected by position; the legacy slice only confirms them.
    """
    generator = object.__new__(ChunkedPrefillPageTableGuardMixin)
    full = torch.tensor([[10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=torch.int32)
    sliding = torch.tensor([[110, 111], [120, 121], [130, 131]], dtype=torch.int32)
    updates = []

    def _update(sliced, force=False):
        updates.append([t.detach().clone() for t in sliced])

    model = SimpleNamespace(
        _active_page_tables_per_layer=[full, sliding],
        update_persistent_per_layer_page_tables=_update,
    )
    generator.model = [model]

    generator._activate_sequential_per_layer_row(full[0:1])
    generator._activate_sequential_per_layer_row(full[1:2])

    assert model._active_page_tables_per_layer[0].shape == (1, 3)
    assert torch.equal(model._active_page_tables_per_layer[0], full[1:2])
    assert torch.equal(model._active_page_tables_per_layer[1], sliding[1:2])
    assert len(updates) == 2
    assert torch.equal(updates[0][0], full[0:1])
    assert torch.equal(updates[1][0], full[1:2])
    assert torch.equal(updates[1][1], sliding[1:2])


def test_bounded_last_chunk_expansion_preserves_ring_origin():
    """100,793-token regression: expanded local rows must match absolute ring slots."""
    generator = _generator_with_bounded_window()

    start, last_idx = generator._adjust_last_prefill_chunk(
        last_chunk_start=100352,
        last_token_idx_in_chunk=440,
        last_token_idx_in_seq=100792,
        chunk_size=2048,
        block_size=64,
        model_id=0,
    )

    assert start == 99328
    assert start % 1024 == 0
    assert last_idx + 1 == 1465
    assert 1024 <= last_idx + 1 <= 2048


def test_bounded_last_chunk_no_expand_when_remnant_covers_window():
    generator = _generator_with_bounded_window()
    start, last_idx = generator._adjust_last_prefill_chunk(
        last_chunk_start=98304,
        last_token_idx_in_chunk=2047,
        last_token_idx_in_seq=100351,
        chunk_size=2048,
        block_size=64,
        model_id=0,
    )
    assert start == 98304
    assert last_idx == 2047


def test_unbounded_last_chunk_is_noop():
    generator = object.__new__(ChunkedPrefillPageTableGuardMixin)
    generator.model = [SimpleNamespace(bounded_sliding_kv_cache=False, layers=[])]
    start, last_idx = generator._adjust_last_prefill_chunk(
        last_chunk_start=100352,
        last_token_idx_in_chunk=440,
        last_token_idx_in_seq=100792,
        chunk_size=2048,
        block_size=64,
        model_id=0,
    )
    assert start == 100352
    assert last_idx == 440
