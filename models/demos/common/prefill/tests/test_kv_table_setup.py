# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Device-free checks of setup_kv_chunk_table's dispatch: which transport each
PREFILL_ENABLE_MIGRATION x PREFILL_MIGRATION_EXPORT_TO_FILE combination drives, who builds and
publishes the table, and how a runtime that reports no KV stage is handled."""

import pytest

from models.demos.common.prefill.runners import migration
from models.demos.common.prefill.runners.migration import KvCacheStage, setup_kv_chunk_table

STAGE_LAYOUTS = [[{"rank": 0}]]


class _Runtime:
    def __init__(self, stages):
        self.stages = stages
        self.built = []

    def kv_migration_stages(self, kv_caches, first_layer_idx, num_my_layers):
        return self.stages

    def build_kv_chunk_table(self, kv_caches, path, *, first_layer_idx, num_my_layers, stage_layouts):
        self.built.append((path, stage_layouts))
        return path


@pytest.fixture
def calls(monkeypatch, tmp_path):
    seen = []

    def record(name, result=None):
        def fn(*args, **kwargs):
            seen.append(name)
            return result

        return fn

    monkeypatch.setenv("PREFILL_MIGRATION_TABLE_PATH", str(tmp_path / "table.pb"))
    monkeypatch.delenv("PREFILL_ENABLE_MIGRATION", raising=False)
    monkeypatch.delenv("PREFILL_MIGRATION_EXPORT_TO_FILE", raising=False)
    monkeypatch.setattr(migration, "remove_stale_device_map_sidecars", record("remove_sidecars"))
    monkeypatch.setattr(migration, "export_device_map_file_and_gather_stage_layouts", record("export", STAGE_LAYOUTS))
    monkeypatch.setattr(migration, "deliver_device_map_and_gather_stage_layouts", record("deliver", STAGE_LAYOUTS))
    monkeypatch.setattr(migration, "allgather_kv_stage_layouts", record("allgather", STAGE_LAYOUTS))
    monkeypatch.setattr(migration, "serialize_device_map", record("serialize_json"))
    monkeypatch.setattr(migration, "publish_serialized_table_and_wait_ready", record("publish", "client"))
    return seen


def _setup(runtime, *, rank=0, num_ranks=1):
    return setup_kv_chunk_table(
        runtime, None, None, (4, 8), rank=rank, num_ranks=num_ranks, first_layer_idx=0, num_my_layers=4
    )


@pytest.mark.parametrize(
    "enable, export, transport, publishes",
    [
        ("0", "0", ["remove_sidecars", "allgather", "serialize_json"], False),
        ("1", "0", ["remove_sidecars", "deliver", "serialize_json"], True),
        ("0", "1", ["export"], False),
        ("1", "1", ["export"], False),
    ],
    ids=["mock", "publish", "file-export", "file-export-wins-over-publish"],
)
def test_rank0_transport_matrix(monkeypatch, calls, enable, export, transport, publishes):
    monkeypatch.setenv("PREFILL_ENABLE_MIGRATION", enable)
    monkeypatch.setenv("PREFILL_MIGRATION_EXPORT_TO_FILE", export)
    runtime = _Runtime([KvCacheStage(0x1000, 0, 4)])

    endpoint = _setup(runtime)

    assert calls == transport + (["publish"] if publishes else [])
    assert endpoint == ("client" if publishes else None)
    assert [layouts for _, layouts in runtime.built] == [STAGE_LAYOUTS]


@pytest.mark.parametrize("enable", ["0", "1"], ids=["mock", "publish"])
def test_non_first_rank_gathers_but_never_builds_or_publishes(monkeypatch, calls, enable):
    monkeypatch.setenv("PREFILL_MIGRATION_TABLE_PATH", "/data/shared/table.pb")
    monkeypatch.setenv("PREFILL_ENABLE_MIGRATION", enable)
    runtime = _Runtime([KvCacheStage(0x1000, 4, 4)])

    assert _setup(runtime, rank=1, num_ranks=2) is None
    assert "publish" not in calls
    assert not runtime.built


def test_no_stage_single_rank_mock_serves_without_table(calls):
    runtime = _Runtime([])

    assert _setup(runtime) is None
    assert calls == ["remove_sidecars"]
    assert not runtime.built


@pytest.mark.parametrize(
    "enable, export, num_ranks",
    [("1", "0", 1), ("0", "1", 1), ("0", "0", 2)],
    ids=["publish", "file-export", "multi-rank"],
)
def test_no_stage_raises_when_a_table_is_required(monkeypatch, calls, enable, export, num_ranks):
    monkeypatch.setenv("PREFILL_MIGRATION_TABLE_PATH", "/data/shared/table.pb")
    monkeypatch.setenv("PREFILL_ENABLE_MIGRATION", enable)
    monkeypatch.setenv("PREFILL_MIGRATION_EXPORT_TO_FILE", export)
    runtime = _Runtime([])

    with pytest.raises(RuntimeError, match="reported no KV cache stage"):
        _setup(runtime, num_ranks=num_ranks)
    assert not {"allgather", "deliver", "export", "publish"} & set(calls)


def test_runtime_without_stage_hook_is_rejected(calls):
    with pytest.raises(RuntimeError, match="kv_migration_stages"):
        _setup(object())
