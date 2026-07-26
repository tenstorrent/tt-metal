# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import glm_5_2_hf_config
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_2 import GLM52Adapter
from models.demos.deepseek_v3_d_p.tt.runners.glm52_paged_kv_cache import (
    GLM52_KV_BUNDLE_TOKENS,
    GLM52_KV_PAGE_TOKENS,
    GLM52_KV_PAGES_PER_BUNDLE,
    GLM52_UNMAPPED_PAGE,
    Glm52PagedCacheExhausted,
    Glm52PagedKvCachePool,
    allocate_glm52_paged_kv_cache_pool,
)


class _SyncRecorder:
    def __init__(self):
        self.updates = []
        self.fail = False

    def __call__(self, table, persistent):
        if self.fail:
            raise RuntimeError("injected page-table upload failure")
        self.updates.append((table.clone(), persistent))


def _cache_pool(*, capacity=3, slots=2, max_bundles=3):
    primary_layers = 4
    index_layers = 2
    kvpe = SimpleNamespace(storage=SimpleNamespace(shape=(capacity * primary_layers, 1, 640, 576)))
    index = SimpleNamespace(shape=(capacity * index_layers, 1, 640, 128))
    persistent = object()
    sync = _SyncRecorder()
    pool = Glm52PagedKvCachePool(
        kvpe=kvpe,
        index=index,
        device_page_table=persistent,
        num_logical_slots=slots,
        max_bundles_per_slot=max_bundles,
        capacity_bundles=capacity,
        num_primary_layers=primary_layers,
        num_index_layers=index_layers,
        sync_page_table=sync,
    )
    return pool, sync, persistent


def test_glm52_paged_pool_uses_5120_token_bundles_of_32_token_pages():
    assert GLM52_KV_BUNDLE_TOKENS == 5120
    assert GLM52_KV_PAGE_TOKENS == 32
    assert GLM52_KV_PAGES_PER_BUNDLE == 160


def test_allocate_coordinates_primary_and_index_ownership_in_one_physical_bundle():
    pool, sync, persistent = _cache_pool()

    allocation = pool.allocate(logical_slot=1, logical_bundle=2)

    assert allocation.physical_bundle == 0
    assert allocation.logical_page_start == 320
    assert allocation.physical_page_start == 0
    assert pool.num_free_bundles == 2
    assert pool.allocation(1, 2) == allocation
    assert pool.allocated_bundles(1) == (allocation,)
    assert len(sync.updates) == 1
    assert sync.updates[0][1] is persistent

    table = pool.host_page_table
    assert table.shape == (2, 3)
    assert table[1, 2] == 0
    assert torch.count_nonzero(table != GLM52_UNMAPPED_PAGE) == 1


def test_allocate_is_idempotent_and_exhaustion_is_explicit(expect_error):
    pool, sync, _ = _cache_pool(capacity=2)

    first = pool.allocate(0, 0)
    assert pool.allocate(0, 0) == first
    assert len(sync.updates) == 1
    pool.allocate(1, 0)

    with expect_error(Glm52PagedCacheExhausted, "exhausted"):
        pool.allocate(0, 1)

    assert pool.num_free_bundles == 0


def test_allocate_chunk_reserves_one_position_owned_bundle():
    pool, sync, _ = _cache_pool(capacity=2, max_bundles=3)

    allocations = pool.allocate_chunk(0, GLM52_KV_BUNDLE_TOKENS)

    assert tuple(item.logical_bundle for item in allocations) == (1,)
    assert tuple(item.physical_bundle for item in allocations) == (0,)
    assert len(sync.updates) == 1
    assert pool.num_free_bundles == 1


def test_allocate_chunk_tracks_exact_valid_tail_independently_from_bundle():
    pool, _, _ = _cache_pool(capacity=2, max_bundles=3)

    allocation = pool.allocate_chunk(0, 0, 4173)

    assert allocation[0].logical_bundle == 0
    assert pool.valid_end(0) == 4173
    assert pool.valid_end(1) == 0


def test_allocate_chunk_exact_tail_retry_is_idempotent():
    pool, sync, _ = _cache_pool(capacity=2, max_bundles=3)

    first = pool.allocate_chunk(0, 0, 4173)
    retried = pool.allocate_chunk(0, 0, 4173)

    assert retried == first
    assert pool.valid_end(0) == 4173
    assert len(sync.updates) == 1


def test_allocate_chunk_rejects_noncontiguous_exact_prefix(expect_error):
    pool, _, _ = _cache_pool(capacity=3, max_bundles=3)
    pool.allocate_chunk(0, 0, GLM52_KV_BUNDLE_TOKENS)

    with expect_error(ValueError, "current valid_end"):
        pool.allocate_chunk(0, 2 * GLM52_KV_BUNDLE_TOKENS, 3 * GLM52_KV_BUNDLE_TOKENS)

    assert pool.allocation(0, 2) is None
    assert pool.valid_end(0) == GLM52_KV_BUNDLE_TOKENS


def test_release_slot_clears_exact_valid_tail():
    pool, _, _ = _cache_pool(capacity=2, max_bundles=3)
    pool.allocate_chunk(0, 0, 4173)

    pool.release_slot(0)

    assert pool.valid_end(0) == 0


def test_allocate_chunk_rejects_unaligned_position_ownership(expect_error):
    pool, sync, _ = _cache_pool(capacity=2, max_bundles=3)
    with expect_error(ValueError, "bundle-aligned"):
        pool.allocate_chunk(0, GLM52_KV_PAGE_TOKENS)
    assert pool.num_free_bundles == 2
    assert torch.all(pool.host_page_table == GLM52_UNMAPPED_PAGE)
    assert sync.updates == []


def test_failed_aligned_chunk_upload_does_not_commit(expect_error):
    pool, sync, _ = _cache_pool(capacity=2, max_bundles=3)
    before = pool.host_page_table
    sync.fail = True

    with expect_error(RuntimeError, "injected"):
        pool.allocate_chunk(0, GLM52_KV_BUNDLE_TOKENS)

    assert pool.allocated_bundles(0) == ()
    assert pool.num_free_bundles == 2
    assert torch.equal(pool.host_page_table, before)


def test_release_invalidates_compact_mapping_and_reuses_the_coordinated_bundle():
    pool, sync, _ = _cache_pool(capacity=2)
    old = pool.allocate(0, 0)
    pool.allocate(1, 0)

    released = pool.release(0, 0)

    assert released == old
    assert pool.allocation(0, 0) is None
    assert pool.host_page_table[0, 0] == GLM52_UNMAPPED_PAGE
    assert len(sync.updates) == 3

    replacement = pool.allocate(0, 1)
    assert replacement.physical_bundle == old.physical_bundle
    assert pool.host_page_table[0, 1] == old.physical_bundle


def test_release_slot_updates_persistent_table_once_and_releases_in_logical_order():
    pool, sync, _ = _cache_pool(capacity=3)
    second = pool.allocate(0, 2)
    first = pool.allocate(0, 0)
    other = pool.allocate(1, 0)
    updates_before_release = len(sync.updates)

    released = pool.release_slot(0)

    assert released == (first, second)
    assert len(sync.updates) == updates_before_release + 1
    assert pool.allocated_bundles(0) == ()
    assert pool.allocated_bundles(1) == (other,)
    assert torch.all(pool.host_page_table[0] == GLM52_UNMAPPED_PAGE)
    assert pool.num_free_bundles == 2


def test_failed_allocate_upload_does_not_commit_host_ownership(expect_error):
    pool, sync, _ = _cache_pool(capacity=1)
    before = pool.host_page_table
    sync.fail = True

    with expect_error(RuntimeError, "injected"):
        pool.allocate(0, 0)

    assert pool.allocation(0, 0) is None
    assert pool.num_free_bundles == 1
    assert torch.equal(pool.host_page_table, before)


def test_failed_release_upload_keeps_bundle_owned(expect_error):
    pool, sync, _ = _cache_pool(capacity=1)
    allocation = pool.allocate(0, 0)
    before = pool.host_page_table
    sync.fail = True

    with expect_error(RuntimeError, "injected"):
        pool.release(0, 0)

    assert pool.allocation(0, 0) == allocation
    assert pool.num_free_bundles == 0
    assert torch.equal(pool.host_page_table, before)


def test_failed_release_slot_upload_keeps_every_bundle_owned(expect_error):
    pool, sync, _ = _cache_pool(capacity=2)
    owned = (pool.allocate(0, 0), pool.allocate(0, 1))
    before = pool.host_page_table
    sync.fail = True

    with expect_error(RuntimeError, "injected"):
        pool.release_slot(0)

    assert pool.allocated_bundles(0) == owned
    assert pool.num_free_bundles == 0
    assert torch.equal(pool.host_page_table, before)


@pytest.mark.parametrize(
    "slot,bundle,error",
    [
        (-1, 0, IndexError),
        (2, 0, IndexError),
        (0, -1, IndexError),
        (0, 3, IndexError),
        (True, 0, TypeError),
        (0, False, TypeError),
    ],
)
def test_logical_address_validation(slot, bundle, error, expect_error):
    pool, _, _ = _cache_pool()
    with expect_error(error, ""):
        pool.allocate(slot, bundle)


def test_constructor_rejects_incoherent_primary_or_index_pool_shapes(expect_error):
    kvpe = SimpleNamespace(storage=SimpleNamespace(shape=(7, 1, 640, 576)))
    index = SimpleNamespace(shape=(4, 1, 640, 128))
    with expect_error(ValueError, "KVPE pool batch dim"):
        Glm52PagedKvCachePool(
            kvpe=kvpe,
            index=index,
            device_page_table=object(),
            num_logical_slots=1,
            max_bundles_per_slot=1,
            capacity_bundles=2,
            num_primary_layers=4,
            num_index_layers=2,
            sync_page_table=lambda *_: None,
        )


def test_explicit_factory_preserves_glm52_formats_and_owns_one_persistent_device_table(monkeypatch):
    calls = {}
    bf16_rm = object()

    def init_mla_kv_cache(**kwargs):
        calls["kvpe"] = kwargs
        batch = kwargs["num_users"] * kwargs["num_kvpe_cache_layers"]
        return SimpleNamespace(storage=SimpleNamespace(shape=(batch, 1, 640, 576)))

    def init_kvpe_cache(**kwargs):
        calls["index"] = kwargs
        batch = kwargs["num_users"] * kwargs["num_kvpe_cache_layers"]
        return SimpleNamespace(shape=(batch, 1, 640, 128))

    persistent = object()
    host_copies = []

    def from_torch(tensor, *, device, **kwargs):
        calls.setdefault("page_tables", []).append((tensor.clone(), device, kwargs))
        return persistent if device is not None else ("host", tensor.clone())

    def copy_host_to_device(host, device):
        host_copies.append((host, device))

    fake_cache_utils = types.ModuleType("models.demos.deepseek_v3_d_p.utils.kv_cache_utils")
    fake_cache_utils.MlaKvCacheFormat = SimpleNamespace(BF16_RM=bf16_rm)
    fake_cache_utils.init_mla_kv_cache = init_mla_kv_cache
    fake_cache_utils.init_kvpe_cache = init_kvpe_cache
    fake_indexer = types.ModuleType("models.demos.deepseek_v3_d_p.tt.mla.indexer")
    fake_indexer.num_full_indexer_layers = lambda config: sum(mode == "full" for mode in config.indexer_types)
    monkeypatch.setitem(sys.modules, fake_cache_utils.__name__, fake_cache_utils)
    monkeypatch.setitem(sys.modules, fake_indexer.__name__, fake_indexer)
    monkeypatch.setattr(ttnn, "from_torch", from_torch)
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", copy_host_to_device)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: ("replicate", mesh))

    mesh = object()
    config = glm_5_2_hf_config(max_seq=1 << 20)
    pool = allocate_glm52_paged_kv_cache_pool(
        mesh_device=mesh,
        hf_config=config,
        mesh_shape=(8, 4),
        sp_axis=0,
        num_primary_layers=78,
        num_logical_slots=4,
        max_sequence_length=10_001,
        capacity_bundles=3,
    )

    assert calls["kvpe"]["cache_format"] is bf16_rm
    assert calls["kvpe"]["seq_len"] == 5120
    assert calls["kvpe"]["num_kvpe_cache_layers"] == 78
    assert calls["kvpe"]["num_users"] == 3
    assert calls["index"]["dtype"] == ttnn.bfloat8_b
    assert calls["index"]["layout"] == ttnn.TILE_LAYOUT
    assert calls["index"]["num_kvpe_cache_layers"] == 21
    assert calls["index"]["num_users"] == 3
    assert pool.max_bundles_per_slot == 2
    assert pool.device_page_table is persistent
    initial, device, page_table_kwargs = calls["page_tables"][0]
    assert initial.shape == (4, 2)
    assert torch.all(initial == GLM52_UNMAPPED_PAGE)
    assert device is mesh
    assert page_table_kwargs["dtype"] == ttnn.int32
    assert page_table_kwargs["layout"] == ttnn.ROW_MAJOR_LAYOUT

    pool.allocate(0, 0)
    assert len(calls["page_tables"]) == 2
    assert calls["page_tables"][1][1] is None
    assert host_copies[0][1] is persistent


def test_glm52_adapter_enables_paged_pool_explicitly_with_independent_capacity(monkeypatch):
    import models.demos.deepseek_v3_d_p.tt.runners.glm52_paged_kv_cache as paged_module

    sentinel = object()
    calls = []
    monkeypatch.setenv("TT_GLM52_PAGED_PREFILL", "1")
    monkeypatch.setenv("TT_GLM52_KV_POOL_BUNDLES", "3")
    monkeypatch.setattr(
        paged_module,
        "allocate_glm52_paged_kv_cache_pool",
        lambda **kwargs: calls.append(kwargs) or sentinel,
    )
    config = glm_5_2_hf_config(max_seq=1 << 20)
    params = SimpleNamespace(
        chunk_size=GLM52_KV_BUNDLE_TOKENS,
        first_layer_idx=0,
        num_layers=78,
        num_users=64,
        max_seq_len=1 << 20,
        mesh_shape=(8, 4),
        sp_axis=0,
    )
    mesh = object()

    result = GLM52Adapter().allocate_kv_cache(mesh_device=mesh, hf_config=config, params=params)

    assert result is sentinel
    assert calls[0]["capacity_bundles"] == 3
    assert calls[0]["num_logical_slots"] == 64
    assert calls[0]["max_sequence_length"] == 1 << 20


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"chunk_size": GLM52_KV_BUNDLE_TOKENS // 2}, "5120-token compute chunk"),
        ({"first_layer_idx": 1, "num_layers": 77}, "one prefill pipeline rank"),
    ],
)
def test_glm52_adapter_rejects_unsupported_paged_model_shapes(monkeypatch, overrides, message, expect_error):
    monkeypatch.setenv("TT_GLM52_PAGED_PREFILL", "1")
    config = glm_5_2_hf_config(max_seq=1 << 20)
    values = {
        "chunk_size": GLM52_KV_BUNDLE_TOKENS,
        "first_layer_idx": 0,
        "num_layers": 78,
        "num_users": 1,
        "max_seq_len": 1 << 20,
        "mesh_shape": (8, 4),
        "sp_axis": 0,
    }
    values.update(overrides)

    with expect_error(ValueError, message):
        GLM52Adapter().allocate_kv_cache(mesh_device=object(), hf_config=config, params=SimpleNamespace(**values))


@pytest.mark.parametrize("capacity", ["0", "-2", "not-an-integer"])
def test_glm52_adapter_rejects_invalid_paged_capacity(monkeypatch, capacity, expect_error):
    monkeypatch.setenv("TT_GLM52_PAGED_PREFILL", "1")
    monkeypatch.setenv("TT_GLM52_KV_POOL_BUNDLES", capacity)
    config = glm_5_2_hf_config(max_seq=1 << 20)
    params = SimpleNamespace(
        chunk_size=GLM52_KV_BUNDLE_TOKENS,
        first_layer_idx=0,
        num_layers=78,
        num_users=1,
        max_seq_len=1 << 20,
        mesh_shape=(8, 4),
        sp_axis=0,
    )

    with expect_error(ValueError, "TT_GLM52_KV_POOL_BUNDLES"):
        GLM52Adapter().allocate_kv_cache(mesh_device=object(), hf_config=config, params=params)
