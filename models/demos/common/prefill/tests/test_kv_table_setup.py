# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Device-free checks of setup_kv_chunk_table: every rank exports its device map, rank 0 alone builds
the table, storage placement is validated, and a runtime that reports no KV stage is handled."""

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
def exports(monkeypatch, tmp_path):
    seen = []

    def export(mesh_device, kv_stages, mesh_shape, device_map_path):
        seen.append(device_map_path)
        return STAGE_LAYOUTS

    monkeypatch.setenv("PREFILL_MIGRATION_TABLE_PATH", str(tmp_path / "table.pb"))
    monkeypatch.setenv("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_device_map.txt")
    monkeypatch.setattr(migration, "export_device_map_file_and_gather_stage_layouts", export)
    return seen


def _setup(runtime, *, rank=0, num_ranks=1):
    return setup_kv_chunk_table(
        runtime, None, None, (4, 8), rank=rank, num_ranks=num_ranks, first_layer_idx=0, num_my_layers=4
    )


def test_rank0_exports_device_map_and_builds_table(exports, tmp_path):
    runtime = _Runtime([KvCacheStage(0x1000, 0, 4)])

    assert _setup(runtime) is None
    assert exports == ["/tmp/prefill_device_map.txt"]
    assert runtime.built == [(str(tmp_path / "table.pb"), STAGE_LAYOUTS)]


def test_non_first_rank_exports_but_never_builds(exports):
    runtime = _Runtime([KvCacheStage(0x1000, 4, 4)])

    assert _setup(runtime, rank=1, num_ranks=2) is None
    assert exports == ["/tmp/prefill_device_map.txt"]
    assert not runtime.built


def test_shared_device_map_path_rejected_multi_rank(monkeypatch, exports, expect_error, tmp_path):
    monkeypatch.setenv("PREFILL_MIGRATION_DEVICE_MAP_PATH", str(tmp_path / "device_map.txt"))

    with expect_error(ValueError, "not on per-host storage"):
        _setup(_Runtime([KvCacheStage(0x1000, 0, 4)]), num_ranks=2)
    assert not exports


def test_shared_device_map_path_allowed_single_rank(monkeypatch, exports, tmp_path):
    monkeypatch.setenv("PREFILL_MIGRATION_DEVICE_MAP_PATH", str(tmp_path / "device_map.txt"))

    _setup(_Runtime([KvCacheStage(0x1000, 0, 4)]))
    assert exports == [str(tmp_path / "device_map.txt")]


def test_per_host_table_path_rejected_multi_rank(monkeypatch, exports, expect_error):
    monkeypatch.setenv("PREFILL_MIGRATION_TABLE_PATH", "/tmp/table.pb")

    with expect_error(ValueError, "is on per-host storage"):
        _setup(_Runtime([KvCacheStage(0x1000, 0, 4)]), num_ranks=2)
    assert not exports


def test_no_stage_single_rank_serves_without_table(exports):
    runtime = _Runtime([])

    assert _setup(runtime) is None
    assert not exports
    assert not runtime.built


def test_no_stage_raises_multi_rank(exports, expect_error):
    with expect_error(RuntimeError, "reported no KV cache stage"):
        _setup(_Runtime([]), num_ranks=2)
    assert not exports


def test_runtime_without_stage_hook_is_rejected(exports, expect_error):
    with expect_error(RuntimeError, "kv_migration_stages"):
        _setup(object())
