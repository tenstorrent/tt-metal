# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only executable contract for a GLM-5.2 page allocator."""

import random
from types import SimpleNamespace

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import GLM52Config
from models.demos.deepseek_v3_d_p.tests.paged_prefill.support import (
    PREFILL_PAGE_TOKENS,
    DramMemorySample,
    PerfComparison,
    assert_bank_balance,
    assert_no_device_allocation_for_page_table_update,
    bank_page_counts,
    compact_full_layer_rank,
    folded_cache_slot,
    reconstruct_logical_pages,
)
from models.demos.deepseek_v3_d_p.tt.runners.glm52_paged_kv_cache import (
    GLM52_UNMAPPED_PAGE,
    Glm52PagedCacheExhausted,
    Glm52PagedKvCachePool,
)


def _cache_pool(capacity=17, slots=8, max_bundles=6):
    primary_layers = 4
    index_layers = 2
    kvpe = SimpleNamespace(storage=SimpleNamespace(shape=(capacity * primary_layers, 1, 640, 576)))
    index = SimpleNamespace(shape=(capacity * index_layers, 1, 640, 128))
    return Glm52PagedKvCachePool(
        kvpe=kvpe,
        index=index,
        device_page_table=object(),
        num_logical_slots=slots,
        max_bundles_per_slot=max_bundles,
        capacity_bundles=capacity,
        num_primary_layers=primary_layers,
        num_index_layers=index_layers,
        sync_page_table=lambda *_: None,
    )


def _assert_pool_matches_model(pool, model):
    table = pool.host_page_table
    assert pool.num_free_bundles == pool.capacity_bundles - len(model)
    assert len(set(model.values())) == len(model)
    for slot in range(pool.num_logical_slots):
        expected = {bundle: physical for (owner_slot, bundle), physical in model.items() if owner_slot == slot}
        actual = {allocation.logical_bundle: allocation.physical_bundle for allocation in pool.allocated_bundles(slot)}
        assert actual == expected
        for bundle in range(pool.max_bundles_per_slot):
            assert table[slot, bundle].item() == expected.get(bundle, GLM52_UNMAPPED_PAGE)


def test_randomized_production_allocator_state_machine(expect_error):
    rng = random.Random(20260725)
    pool = _cache_pool()
    model = {}

    for _ in range(2_000):
        slot = rng.randrange(pool.num_logical_slots)
        bundle = rng.randrange(pool.max_bundles_per_slot)
        action = rng.random()
        key = (slot, bundle)
        if action < 0.2:
            released = pool.release_slot(slot)
            for allocation in released:
                assert model.pop((slot, allocation.logical_bundle)) == allocation.physical_bundle
        elif action < 0.4 and key in model:
            released = pool.release(slot, bundle)
            assert released.physical_bundle == model.pop(key)
        else:
            if key in model:
                assert pool.allocate(slot, bundle).physical_bundle == model[key]
            elif len(model) < pool.capacity_bundles:
                allocation = pool.allocate(slot, bundle)
                model[key] = allocation.physical_bundle
            else:
                before = pool.host_page_table
                with expect_error(Glm52PagedCacheExhausted, ""):
                    pool.allocate(slot, bundle)
                assert torch.equal(pool.host_page_table, before)
        _assert_pool_matches_model(pool, model)


def test_glm52_index_cache_uses_compact_full_layer_slots(expect_error):
    indexer_types = GLM52Config.indexer_types()
    full_layers = [i for i, mode in enumerate(indexer_types) if mode == "full"]
    shared_layers = [i for i, mode in enumerate(indexer_types) if mode == "shared"]

    assert full_layers[:4] == [0, 1, 2, 6]
    assert compact_full_layer_rank(indexer_types, 6) == 3
    assert compact_full_layer_rank(indexer_types, full_layers[-1]) == len(full_layers) - 1
    with expect_error(ValueError, "does not own"):
        compact_full_layer_rank(indexer_types, shared_layers[0])

    physical_page = 3
    assert folded_cache_slot(physical_page, 77, GLM52Config.NUM_LAYERS) == 3 * 78 + 77
    assert (
        folded_cache_slot(
            physical_page,
            compact_full_layer_rank(indexer_types, 6),
            len(full_layers),
        )
        == 3 * len(full_layers) + 3
    )


def test_reference_reconstruction_obeys_logical_page_order_and_valid_tail():
    num_pages, num_layers, page_tokens, head_dim = 4, 3, 8, 2
    pool = torch.empty(num_pages * num_layers, 1, page_tokens, head_dim)
    for page in range(num_pages):
        for layer in range(num_layers):
            slot = folded_cache_slot(page, layer, num_layers)
            pool[slot].fill_(100 * page + layer)

    actual = reconstruct_logical_pages(
        pool,
        physical_pages=(3, 0, 2),
        layer_slot=1,
        num_layer_slots=num_layers,
        valid_tokens=2 * page_tokens + 3,
    )
    expected = torch.cat(
        [
            torch.full((page_tokens, head_dim), 301.0),
            torch.full((page_tokens, head_dim), 1.0),
            torch.full((3, head_dim), 201.0),
        ]
    )
    assert torch.equal(actual, expected)


def test_runtime_bank_balance_contract(expect_error):
    page_bank_ids = [page % 7 for page in range(23)]
    assert bank_page_counts(page_bank_ids, 7) == (4, 4, 3, 3, 3, 3, 3)
    assert_bank_balance(page_bank_ids, 7)
    with expect_error(AssertionError, "imbalanced"):
        assert_bank_balance([0, 0, 0, 1], 4)


def test_preallocated_pool_memory_measurement_contract(expect_error):
    pool = DramMemorySample("pool", 7, 1_000_000, 9_000_000, 1_000_000)
    after_reserve = DramMemorySample("after-reserve", 7, 1_000_000, 9_000_000, 1_000_000)
    assert_no_device_allocation_for_page_table_update(pool, after_reserve)

    grew = DramMemorySample("grew", 7, 1_004_096, 8_995_904, 995_904)
    with expect_error(AssertionError, "allocated device memory"):
        assert_no_device_allocation_for_page_table_update(pool, grew)


def test_fixed_vs_paged_perf_summary_uses_medians():
    comparison = PerfComparison(
        fixed_seconds=(1.0, 1.1, 50.0),
        paged_seconds=(1.05, 1.2, 2.0),
    )
    assert comparison.fixed_median == 1.1
    assert comparison.paged_median == 1.2
    assert comparison.ratio == pytest.approx(1.2 / 1.1)


def test_max_context_page_count_is_ceil_divided():
    max_context = GLM52Config.MAX_POSITION_EMBEDDINGS
    pages = (max_context + PREFILL_PAGE_TOKENS - 1) // PREFILL_PAGE_TOKENS
    assert pages == 205
