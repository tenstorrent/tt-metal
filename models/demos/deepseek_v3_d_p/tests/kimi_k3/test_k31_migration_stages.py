# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: Kimi-K3.1 migrates kvpe, the DSA indexer's keys, KDA recurrent and KDA convolution as stages 0-3.

K3.1 is K3 plus an indexer on its MLA layers, so it keeps K3's stages and adds the indexer-key cache right
after kvpe, slotted like kvpe (one slot per MLA layer of the rank). Every rank returns all four (the
per-stage allgather is collective), and the adapter hooks map config 1 to the MLA layers.
"""

from types import SimpleNamespace

import pytest

from models.demos.deepseek_v3_d_p.reference.kimi_k3_1_config import KimiK31Config, kimi_k3_1_hf_config
from models.demos.deepseek_v3_d_p.tt.kimi_k3.layer_schedule import KimiK3LayerSchedule
from models.demos.deepseek_v3_d_p.tt.kimi_k3.runtime import TtKimiK3Runtime
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3_1 import KimiK31Adapter

SPLITS = [(0, 24), (24, 24), (48, 24), (72, 21)]


def _tensor(address, shape):
    return SimpleNamespace(buffer_address=lambda: address, shape=shape)


def _runtime(first, count, num_users=1):
    runtime = TtKimiK3Runtime.__new__(TtKimiK3Runtime)
    runtime.config = SimpleNamespace(first_layer_idx=first, num_layers=count, num_users=num_users, dflash_enabled=False)
    runtime.hf_config = kimi_k3_1_hf_config()
    return runtime


def _caches(first, count, num_users=1, *, index=True):
    schedule = KimiK3LayerSchedule.build(KimiK31Config, first, count)
    kvpe = index_cache = None
    if schedule.num_mla_layers:
        kvpe = SimpleNamespace(storage=_tensor(0x1000, (num_users * schedule.num_mla_layers, 1, 7040, 656)))
        if index:
            index_cache = _tensor(0x1800, (num_users * schedule.num_mla_layers, 1, 7040, 128))
    ids = schedule.rank_kda_layer_ids()
    kda_states = SimpleNamespace(
        layer_ids=ids,
        num_slots=num_users,
        recurrent=_tensor(0x2000, (num_users * len(ids), 24, 128, 128)),
        convolution=_tensor(0x3000, (num_users * len(ids), 144, 192)),
    )
    return SimpleNamespace(kvpe=kvpe, index=index_cache, kda_states=kda_states)


@pytest.mark.parametrize("first,count", SPLITS, ids=[f"L{f}-{f + c}" for f, c in SPLITS])
@pytest.mark.parametrize("num_users", [1, 4])
def test_four_galaxy_split_emits_four_stages_with_index_after_kvpe(first, count, num_users):
    stages = _runtime(first, count, num_users).kv_migration_stages(_caches(first, count, num_users))
    assert len(stages) == 4
    kvpe, index, recurrent, convolution = stages
    mla_before = sum(1 for layer in KimiK31Config.mla_layer_ids() if layer < first)
    kda_before = sum(1 for layer in KimiK31Config.kda_layer_ids() if layer < first)
    assert (kvpe.base_addr, kvpe.first_layer, kvpe.count) == (0x1000, mla_before, 6)
    assert (index.base_addr, index.first_layer, index.count) == (0x1800, mla_before, 6)
    assert (recurrent.first_layer, recurrent.count) == (kda_before, count - 6)
    assert (convolution.first_layer, convolution.count) == (kda_before, count - 6)


def test_index_slots_tile_the_mla_layers():
    stages = [_runtime(f, c).kv_migration_stages(_caches(f, c))[1] for f, c in SPLITS]
    assert [(s.first_layer, s.count) for s in stages] == [(0, 6), (6, 6), (12, 6), (18, 6)]
    assert sum(s.count for s in stages) == len(KimiK31Config.mla_layer_ids()) == 24


def test_layer_zero_only_rank_keeps_four_stages():
    """Layer 0 is KDA: no kvpe and no index cache, but both stages stay as empties so the allgather lines up."""
    stages = _runtime(0, 1).kv_migration_stages(_caches(0, 1))
    assert [(s.base_addr, s.first_layer, s.count) for s in stages[:2]] == [(0, 0, 0), (0, 0, 0)]
    assert [(s.first_layer, s.count) for s in stages[2:]] == [(0, 1), (0, 1)]


def test_missing_index_cache_is_refused(expect_error):
    """A K3.1 rank with MLA layers and a kvpe cache but no index cache: the allocation and the runtime disagree."""
    with expect_error(RuntimeError, "allocate_kv_cache disagrees"):
        _runtime(0, 24).kv_migration_stages(_caches(0, 24, index=False))


def test_k3_runtime_still_refuses_an_index_cache(expect_error):
    runtime = _runtime(0, 24)
    runtime.hf_config = None  # Kimi-K3: no indexer
    caches = _caches(0, 24)
    with expect_error(RuntimeError, "allocate_kv_cache disagrees"):
        runtime.kv_migration_stages(caches)
    caches.index = None
    assert len(runtime.kv_migration_stages(caches)) == 3


def test_adapter_hooks_name_the_four_configs():
    adapter = KimiK31Adapter()
    assert [adapter.cache_kind(i) for i in range(5)] == ["kvpe", "index", "kda_recurrent", "kda_convolution", "other"]
    mla = {layer: layer for layer in KimiK31Config.mla_layer_ids()}
    assert adapter.cache_layer_rows(0, 93) == mla
    assert adapter.cache_layer_rows(1, 93) == mla
    assert adapter.cache_layer_rows(3, 24) == {layer: layer for layer in KimiK31Config.kda_layer_ids() if layer < 24}
    assert adapter.cache_head_dim(0) == 576
    assert adapter.cache_head_dim(1) == KimiK31Config.INDEX_HEAD_DIM == 128
    assert adapter.cache_head_dim(2) is None


def test_registry_serves_k31_with_raw_basis_index_keys():
    from models.demos.common.prefill.adapter import get_adapter
    from models.demos.deepseek_v3_d_p.tt.mla.indexer import index_keys_hadamard_rotated, resolve_has_indexer
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

    adapter = get_adapter("kimi_k3_1")
    assert isinstance(adapter, KimiK31Adapter)
    assert adapter.default_sparse_kv_cache_format == MlaKvCacheFormat.SCALED_FP8
    config = adapter.load_hf_config()
    assert resolve_has_indexer(config)
    # K3.1's decode keeps indexer keys in the model's basis; the migrated cache must match.
    assert not index_keys_hadamard_rotated(config)


def test_kv_chunk_matches_k31_decode_page():
    """Migration copies raw bytes, so prefill's KV chunk must be decode's page: K3.1's decode keeps GLM's scaled-fp8
    cache, 32 positions at a 704-byte stride (tt-blaze ``k31_glm_kv_cache``). Prefill's row is 656 packed bytes,
    one native page each, padded to the 64-byte DRAM alignment."""
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, MlaKvCacheGeometry

    geometry = MlaKvCacheGeometry.from_config(kimi_k3_1_hf_config())
    assert MlaKvCacheFormat.SCALED_FP8.storage_width(geometry) == 656
    dram_alignment = 64
    aligned_row = -(-656 // dram_alignment) * dram_alignment
    assert 32 * aligned_row == 22528  # GLM_SCALED_KV_PAGE_BYTES on the decode side
