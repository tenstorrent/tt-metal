# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: Kimi-K3 exposes kvpe, KDA recurrent and KDA convolution as migration stages 0, 1, 2.

Every rank must return exactly three stages (the per-stage allgather is collective), numbered in
compacted slot space, and the adapter hooks must map the three configs back to model layers.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tt.kimi_k3.layer_schedule import KimiK3LayerSchedule
from models.demos.deepseek_v3_d_p.tt.kimi_k3.runtime import TtKimiK3Runtime
from models.demos.deepseek_v3_d_p.tt.runners.adapters.deepseek_v3 import DeepSeekV3Adapter
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3 import KimiK3Adapter
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import merged_num_layers

SPLITS = [(0, 24), (24, 24), (48, 24), (72, 21)]


@pytest.mark.parametrize("num_layers", [1, 3, 5, 25])
def test_migration_rejects_unfenced_kda_exports(num_layers, expect_error):
    with expect_error(RuntimeError, "trailing KDA exports"):
        KimiK3Adapter().validate_migration_completion(num_layers)


@pytest.mark.parametrize("adapter,ack_layers,mtp_levels", [(KimiK3Adapter(), 1, 0), (DeepSeekV3Adapter(), 4, 1)])
@pytest.mark.parametrize("completion_delta", [None, -6, -1, 0, 1])
def test_migration_requires_exact_model_aware_completion(
    monkeypatch, adapter, ack_layers, mtp_levels, completion_delta, expect_error
):
    from models.demos.common.prefill.runners import prefill_producer as producer
    from models.demos.common.prefill.runners.migration_driver import _require_prefill_completion

    monkeypatch.setattr(producer, "ADAPTER", adapter)
    monkeypatch.setattr(producer, "NUM_LAYERS", 4)
    monkeypatch.setattr(producer, "NUM_ACK_LAYERS", ack_layers)
    monkeypatch.setattr(producer, "MTP_LEVELS", mtp_levels)
    expected = (ack_layers + mtp_levels) * 6
    completion = None if completion_delta is None else expected + completion_delta
    # Exercise the real drain loop with an immediate deadline for missing/partial acks.
    channel = None if completion is None else SimpleNamespace(try_consume_all=Mock(side_effect=[completion, 0]))
    migrate = Mock()
    if completion == expected:
        _require_prefill_completion(producer, None, channel, 6, timeout_s=-1)
        migrate()
        migrate.assert_called_once()
    else:
        with expect_error(RuntimeError, "completion channel|complete layer acknowledgements"):
            _require_prefill_completion(producer, None, channel, 6, timeout_s=-1)
            migrate()
        migrate.assert_not_called()


def test_migration_does_not_drain_when_no_ack_can_cover_exports(expect_error):
    from models.demos.common.prefill.runners.migration_driver import _require_prefill_completion

    producer = SimpleNamespace(ADAPTER=KimiK3Adapter(), NUM_LAYERS=5, _drain_layer_acks=Mock())
    with expect_error(RuntimeError, "trailing KDA exports"):
        _require_prefill_completion(producer, object(), object(), 1)
    producer._drain_layer_acks.assert_not_called()


@pytest.mark.parametrize("failure", ["prefix", "channel", "timeout", "excess_acks"])
def test_prefill_failure_releases_validators_before_resident_broadcast(monkeypatch, failure, expect_error):
    from models.demos.common.prefill.runners import migration_driver as migration
    from models.demos.common.prefill.runners import prefill_producer as producer

    monkeypatch.setattr(migration.sys, "argv", ["migration_driver"])
    monkeypatch.delenv("PREFILL_PRODUCER_MANIFEST", raising=False)
    monkeypatch.setattr(producer, "_load_env_config", Mock())
    monkeypatch.setattr(producer, "_mr_config", Mock(return_value=(0, 2)))
    monkeypatch.setattr(producer, "_require_shared_table_path", Mock())
    cfg = SimpleNamespace(num_users=1, chunks_min=1, chunks_max=1, max_requests=1, verify=False)
    monkeypatch.setattr(producer, "_config_from_env", Mock(return_value=cfg))
    monkeypatch.setattr(producer, "ADAPTER", KimiK3Adapter())
    monkeypatch.setattr(producer, "NUM_LAYERS", 5 if failure == "prefix" else 4)
    monkeypatch.setattr(producer, "NUM_ACK_LAYERS", 1)
    monkeypatch.setattr(producer, "MTP_LEVELS", 0)
    monkeypatch.setattr(producer, "_read_kv_chunk_table", Mock(return_value=None))
    monkeypatch.setattr(
        producer, "_connect_layer_ack_channel", Mock(return_value=None if failure == "channel" else object())
    )
    monkeypatch.setattr(producer, "_drain_layer_acks", Mock(return_value=7 if failure == "excess_acks" else 0))
    monkeypatch.setattr(producer, "_resolve_slot_prompts", Mock(return_value=({}, {}, {})))
    schedule = Mock(return_value=SimpleNamespace(total_pushes=6, wall_s=0.0, completed=1))
    monkeypatch.setattr(producer, "run_schedule", schedule)
    broadcast = Mock()
    monkeypatch.setattr(producer, "_mr_bcast_resident", broadcast)
    driver = Mock()
    monkeypatch.setattr(migration, "MigrationDriver", Mock(return_value=driver))
    service = Mock()
    monkeypatch.setattr(migration.ttnn, "H2DStreamService", SimpleNamespace(connect=Mock(return_value=service)))
    collective = Mock(return_value=[0, 0])
    monkeypatch.setattr(migration.ttnn, "distributed_context_allgather_int", collective)

    error = {
        "prefix": "trailing KDA exports",
        "channel": "completion channel",
        "timeout": "received 0, expected 6",
        "excess_acks": "received 7, expected 6",
    }[failure]
    with expect_error(RuntimeError, error):
        migration.main()
    collective.assert_called_once_with(0)
    if failure in ("prefix", "channel"):
        schedule.assert_not_called()
        driver.attach.assert_not_called()
    else:
        schedule.assert_called_once()
    driver.run.assert_not_called()
    broadcast.assert_not_called()

    # A validator receives that failure status and never joins the resident broadcast.
    collective.reset_mock()
    with expect_error(RuntimeError, "rank zero prefill failed"):
        migration._run_validator(1, 2, SimpleNamespace())
    collective.assert_called_once_with(0)
    broadcast.assert_not_called()


@pytest.mark.parametrize("world_size", [1, 2])
def test_successful_prefill_releases_validators(monkeypatch, world_size):
    from models.demos.common.prefill.runners import migration_driver as migration

    collective = Mock(return_value=[1, 0])
    monkeypatch.setattr(migration.ttnn, "distributed_context_allgather_int", collective)
    with migration._coordinate_prefill_completion(world_size):
        pass
    if world_size > 1:
        collective.assert_called_once_with(1)
    else:
        collective.assert_not_called()


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
    ids = schedule.rank_kda_layer_ids()
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


def test_kda_only_later_rank_keeps_the_kvpe_layout_contiguous():
    first, second = (_runtime(f, c).kv_migration_stages(_caches(f, c))[0] for f, c in [(0, 12), (12, 3)])
    assert (second.base_addr, second.first_layer, second.count) == (0, 3, 0)
    layout = [{"first_layer": s.first_layer, "count": s.count} for s in (first, second)]
    assert merged_num_layers(layout) == 3


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


def test_layer_position_range_follows_the_contract():
    """Contract section 5: MLA layers over the request, KDA layers over the version window decode reads next."""
    adapter = KimiK3Adapter()
    assert adapter.layer_position_range(3, 100) == (0, 100)
    assert adapter.layer_position_range(0, 100) == (110_592, 147_456)  # the contract's own example, v = 3
    assert adapter.layer_position_range(0, 56_320) == (7 * 36_864, 8 * 36_864)
    assert adapter.layer_position_range(0, 1) == (0, 36_864)
    assert adapter.layer_position_range(92, 56_320) == (0, 56_320)


def test_driver_groups_consecutive_same_axis_layers_into_runs():
    from models.demos.common.prefill.runners.migration_driver import _layer_runs

    real_len = 56_320
    runs = _layer_runs(range(KimiK3Config.NUM_LAYERS), real_len, KimiK3Adapter().layer_position_range)
    window = (7 * 36_864, 8 * 36_864)
    assert runs[:4] == [(0, 3, *window), (3, 4, 0, real_len), (4, 7, *window), (7, 8, 0, real_len)]
    assert runs[-1] == (91, 93, 0, real_len)  # layers 91 and 92 are both MLA: one call
    assert len(runs) == 46
    assert all(runs[idx][1] == runs[idx + 1][0] for idx in range(len(runs) - 1))
    mla = set(KimiK3Config.mla_layer_ids())
    for start, end, pos_start, pos_end in runs:
        kinds = {layer in mla for layer in range(start, end)}
        assert len(kinds) == 1
        assert (pos_start, pos_end) == ((0, real_len) if kinds.pop() else window)
    # A subset keeps its own layers only and never merges across a gap.
    assert _layer_runs([22, 23, 27], real_len, KimiK3Adapter().layer_position_range) == [
        (22, 23, *window),
        (23, 24, 0, real_len),
        (27, 28, 0, real_len),
    ]
