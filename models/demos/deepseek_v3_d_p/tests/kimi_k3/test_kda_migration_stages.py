# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: Kimi-K3 exposes kvpe, KDA recurrent and KDA convolution as migration stages 0, 1, 2.

Every rank must return exactly three stages (the per-stage allgather is collective), numbered in
compacted slot space, and the adapter hooks must map the three configs back to model layers.
"""

from types import SimpleNamespace

import pytest

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tt.kimi_k3.layer_schedule import KimiK3LayerSchedule
from models.demos.deepseek_v3_d_p.tt.kimi_k3.runtime import TtKimiK3Runtime
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3 import KimiK3Adapter

SPLITS = [(0, 24), (24, 24), (48, 24), (72, 21)]


def _tensor(address, shape):
    return SimpleNamespace(buffer_address=lambda: address, shape=shape)


def _runtime(first, count, num_users=1, dflash=False):
    runtime = TtKimiK3Runtime.__new__(TtKimiK3Runtime)
    runtime.config = SimpleNamespace(
        first_layer_idx=first, num_layers=count, num_users=num_users, dflash_enabled=dflash
    )
    return runtime


def _caches(first, count, num_users=1, *, kvpe=True, kda=True):
    schedule = KimiK3LayerSchedule.build(KimiK3Config, first, count)
    kvpe_cache = None
    if kvpe and schedule.num_mla_layers:
        kvpe_cache = SimpleNamespace(storage=_tensor(0x1000, (num_users * schedule.num_mla_layers, 1, 7040, 576)))
    kda_states = None
    ids = schedule.kda_layer_ids_local()
    if kda and ids:
        kda_states = SimpleNamespace(
            layer_ids=ids,
            num_slots=num_users,
            recurrent=_tensor(0x2000, (num_users * len(ids), 24, 128, 128)),
            convolution=_tensor(0x3000, (num_users * len(ids), 144, 192)),
        )
    return SimpleNamespace(kvpe=kvpe_cache, index=None, kda_states=kda_states)


@pytest.mark.parametrize("first,count", SPLITS, ids=[f"L{f}-{f + c}" for f, c in SPLITS])
def test_four_galaxy_split_emits_three_stages_in_compacted_slot_space(first, count):
    stages = _runtime(first, count).kv_migration_stages(_caches(first, count))
    assert len(stages) == 3
    kvpe, recurrent, convolution = stages
    mla_before = sum(1 for layer in KimiK3Config.mla_layer_ids() if layer < first)
    kda_before = sum(1 for layer in KimiK3Config.kda_layer_ids() if layer < first)
    assert (kvpe.base_addr, kvpe.first_layer, kvpe.count) == (0x1000, mla_before, 6)
    assert (recurrent.first_layer, recurrent.count) == (kda_before, count - 6)
    assert (convolution.first_layer, convolution.count) == (kda_before, count - 6)
    assert recurrent.base_addr == 0x2000 and convolution.base_addr == 0x3000


def test_compacted_kda_slots_tile_the_model():
    firsts = [_runtime(f, c).kv_migration_stages(_caches(f, c))[1] for f, c in SPLITS]
    assert [(s.first_layer, s.count) for s in firsts] == [(0, 18), (18, 18), (36, 18), (54, 15)]
    assert sum(s.count for s in firsts) == len(KimiK3Config.kda_layer_ids()) == 69


def test_single_galaxy_24_layer_rank():
    stages = _runtime(0, 24).kv_migration_stages(_caches(0, 24))
    assert [(s.first_layer, s.count) for s in stages] == [(0, 6), (0, 18), (0, 18)]


def test_layer_zero_only_rank_has_no_kvpe_stage_but_keeps_three():
    stages = _runtime(0, 1).kv_migration_stages(_caches(0, 1))
    assert (stages[0].base_addr, stages[0].first_layer, stages[0].count) == (0, 0, 0)
    assert [(s.first_layer, s.count) for s in stages[1:]] == [(0, 1), (0, 1)]


def test_missing_slabs_or_mismatched_slabs_are_refused(expect_error):
    with expect_error(RuntimeError, "no KDA state slabs"):
        _runtime(0, 24).kv_migration_stages(_caches(0, 24, kda=False))
    caches = _caches(24, 24)
    caches.kda_states.layer_ids = caches.kda_states.layer_ids[:-1]
    with expect_error(RuntimeError, "cover layers"):
        _runtime(24, 24).kv_migration_stages(caches)
    with expect_error(RuntimeError, "DFlash"):
        _runtime(0, 24, dflash=True).kv_migration_stages(_caches(0, 24))


def test_table_layer_rows_follow_the_gathered_stages():
    runtime = _runtime(0, 24)
    layout = [{"first_layer": f, "count": c} for f, c in [(0, 18), (18, 18), (36, 18), (54, 15)]]
    assert runtime.kda_table_layer_rows(layout) == KimiK3Config.kda_layer_ids()
    assert runtime.kda_table_layer_rows(layout[:1]) == KimiK3Config.kda_layer_ids()[:18]
    assert runtime.kv_table_layer_rows([[{"first_layer": 0, "count": 6}]]) == KimiK3Config.mla_layer_ids()[:6]


def test_adapter_hooks_name_the_three_configs():
    adapter = KimiK3Adapter()
    assert [adapter.cache_kind(i) for i in range(4)] == ["kvpe", "kda_recurrent", "kda_convolution", "other"]
    assert adapter.cache_layer_rows(0, 93) == {layer: layer for layer in KimiK3Config.mla_layer_ids()}
    assert adapter.cache_layer_rows(1, 93) == {layer: layer for layer in KimiK3Config.kda_layer_ids()}
    assert adapter.cache_layer_rows(2, 24) == {layer: layer for layer in KimiK3Config.kda_layer_ids() if layer < 24}
    assert len(adapter.cache_layer_rows(2, 24)) == 18
    assert adapter.cache_head_dim(0) == 576 and adapter.cache_head_dim(1) is None
    assert set(KimiK3Config.kda_layer_ids()) | set(KimiK3Config.mla_layer_ids()) == set(range(93))
