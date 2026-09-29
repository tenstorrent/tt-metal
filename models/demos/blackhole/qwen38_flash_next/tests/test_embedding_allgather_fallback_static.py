# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract for the S=1 owner-select embedding fallback."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.ttnn import embedding as embedding_module

REPO_ROOT = Path(__file__).resolve().parents[5]
EMBEDDING = REPO_ROOT / "models/demos/blackhole/qwen38_flash_next/ttnn/embedding.py"


def _helper_source() -> str:
    source = EMBEDDING.read_text(encoding="utf-8")
    return source.split("def _all_gather_owner_select_hidden_fallback(", 1)[1].split(
        "\ndef _localize_token_ids_on_host(", 1
    )[0]


def test_owner_select_uses_retained_axis1_one_link_all_gather_and_no_reduction() -> None:
    helper = _helper_source()
    call = helper.split("gathered = ttnn.experimental.all_gather_async(", 1)[1].split("\n        )", 1)[0]
    for contract in (
        "multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=TP_AXIS)",
        "num_links=1",
        "dim=3",
        "cluster_axis=TP_AXIS",
        "memory_config=ttnn.DRAM_MEMORY_CONFIG",
        "topology=ttnn.Topology.Linear",
        "chunks_per_sync=1",
        "num_workers_per_link=1",
        "num_buffers_per_channel=2",
    ):
        assert contract in call
    for forbidden in (
        "ttnn.add(",
        "ttnn.sum(",
        "fast_reduce_nc(",
        "ttnn.reduce_scatter(",
        "tt_all_reduce(",
        "ttnn.to_torch(",
        "ttnn.from_torch(",
        "copy_host_to_device_tensor(",
    ):
        assert forbidden not in helper


def test_owner_select_is_one_aligned_logical_coordinate_slice_then_partition() -> None:
    helper = _helper_source()
    assert helper.count("ttnn.slice(") == 1
    assert "owner_start = active_vocab_shard * HIDDEN_SIZE" in helper
    assert "owner_end = owner_start + HIDDEN_SIZE" in helper
    assert "(0, 0, 0, owner_start)" in helper
    assert "(1, 1, 1, owner_end)" in helper
    assert "active_vocab_coordinate != expected_coordinate" in helper
    assert "hidden = ttnn.mesh_partition(" in helper
    assert "dim=3" in helper and "cluster_axis=TP_AXIS" in helper
    assert "placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3" in helper
    assert helper.count("ttnn.synchronize_device(mesh_device)") == 3
    assert helper.count("if not resident_async:") == 4


def test_owner_select_rejects_aliases_and_retains_consumer_safe_ownership() -> None:
    helper = _helper_source()
    for contract in (
        "type(active_vocab_shard) is not int",
        "type(active_vocab_coordinate) is not tuple",
        "any(type(value) is not int for value in active_vocab_coordinate)",
        "if partition_drained:\n                _deallocate(hidden, selected)",
        "if owner_slice_drained:\n                _deallocate(gathered)",
        "if all_gather_drained:\n                _deallocate(local_partial)",
        "len(retained_locals) != TP_SIZE * 4",
    ):
        assert contract in helper


@pytest.mark.parametrize(
    ("shard", "coordinate"),
    ((-1, (0, -1)), (4, (0, 4)), (True, (0, 1)), (0, (False, 0)), (2, (0, 3))),
)
def test_owner_select_fails_closed_before_device_ops(monkeypatch, expect_error, shard, coordinate) -> None:
    calls = []
    fake_ttnn = SimpleNamespace(Topology=SimpleNamespace(Linear="linear"))
    contract = SimpleNamespace(validate_mesh=lambda mesh: calls.append(mesh))
    monkeypatch.setattr(embedding_module, "ttnn", fake_ttnn)
    with expect_error(RuntimeError, "S=1 embedding owner"):
        embedding_module._all_gather_owner_select_hidden_fallback(
            object(),
            active_vocab_shard=shard,
            active_vocab_coordinate=coordinate,
            mesh_device=SimpleNamespace(shape=(1, 4)),
            mesh_contract=contract,
            replicated_reference=object(),
            tt_ccl=object(),
            collective_topology="linear",
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("policy", "expected_syncs"),
    (
        (embedding_module.Qwen38TTNNEmbeddingSyncPolicy.CORRECTNESS_FENCED, 3),
        (embedding_module.Qwen38TTNNEmbeddingSyncPolicy.RESIDENT_ASYNC, 0),
    ),
)
def test_owner_select_executes_ordered_ag_slice_partition_contract(monkeypatch, policy, expected_syncs) -> None:
    calls = []
    events = []
    deallocated = []

    class Topology:
        def distribution_shape(self):
            return (1, 4)

        def mesh_coords(self):
            return ((0, 0), (0, 1), (0, 2), (0, 3))

    class Tensor:
        def __init__(self, name, shape, padded):
            self.name, self.shape, self.padded_shape = name, shape, padded
            self.dtype, self.layout, self._topology = "bf16", "tile", Topology()

        def memory_config(self):
            return "dram"

        def tensor_topology(self):
            return self._topology

        def update_tensor_topology(self, topology):
            self._topology = topology

    source = Tensor("source", (1, 1, 1, 2560), (1, 1, 32, 2560))
    reference = Tensor("reference", (1, 1, 1, 1), (1, 1, 32, 32))
    gathered = Tensor("gathered", (1, 1, 1, 10240), (1, 1, 32, 10240))
    selected = Tensor("selected", (1, 1, 1, 2560), (1, 1, 32, 2560))
    hidden = Tensor("hidden", (1, 1, 1, 640), (1, 1, 32, 640))

    def all_gather(value, **kwargs):
        calls.append(("all-gather", value, kwargs))
        return gathered

    def slice_op(value, start, end, **kwargs):
        calls.append(("slice", value, start, end, kwargs))
        assert value is gathered and start == (0, 0, 0, 5120) and end == (1, 1, 1, 7680)
        return selected

    def mesh_partition(value, **kwargs):
        calls.append(("mesh-partition", value, kwargs))
        assert value is selected
        return hidden

    fake_ttnn = SimpleNamespace(
        bfloat16="bf16",
        TILE_LAYOUT="tile",
        DRAM_MEMORY_CONFIG="dram",
        Topology=SimpleNamespace(Linear="linear"),
        PlacementReplicate=lambda: "replicate",
        TensorTopology=lambda *_args: Topology(),
        experimental=SimpleNamespace(all_gather_async=all_gather),
        slice=slice_op,
        mesh_partition=mesh_partition,
        get_device_tensors=lambda _tensor: tuple(object() for _ in range(4)),
        synchronize_device=lambda mesh: calls.append(("sync", mesh)),
        deallocate=lambda tensor: deallocated.append(tensor.name),
    )
    contract = SimpleNamespace(
        validate_mesh=lambda mesh: calls.append(("validate-mesh", mesh)),
        validate_tensor=lambda tensor, **kwargs: calls.append(("validate", tensor, kwargs)),
        mark_collective_shard=lambda tensor, **kwargs: calls.append(("mark", tensor, kwargs)),
    )
    manager = SimpleNamespace(
        get_num_links=lambda cluster_axis: 1,
        get_and_cycle_ag_semaphore_handles=lambda cluster_axis: "axis1-semaphore",
    )
    mesh = SimpleNamespace(shape=(1, 4))
    monkeypatch.setattr(embedding_module, "ttnn", fake_ttnn)

    result = embedding_module._all_gather_owner_select_hidden_fallback(
        source,
        active_vocab_shard=2,
        active_vocab_coordinate=(0, 2),
        mesh_device=mesh,
        mesh_contract=contract,
        replicated_reference=reference,
        tt_ccl=manager,
        collective_topology="linear",
        synchronization_policy=policy,
        _retain_async_failure_owners=lambda *_args: None,
        _diagnostic_stage_callback=events.append,
    )

    assert result is hidden
    assert [call[0] for call in calls if call[0] in {"all-gather", "slice", "mesh-partition"}] == [
        "all-gather",
        "slice",
        "mesh-partition",
    ]
    assert [call[0] for call in calls].count("sync") == expected_syncs
    assert events == [
        "before-all-gather-async",
        "after-all-gather-async-enqueue",
        "before-all-gather-async-synchronize",
        "after-all-gather-async-synchronize",
        "before-owner-contribution-slice",
        "after-owner-contribution-slice-enqueue",
        "before-owner-contribution-slice-synchronize",
        "after-owner-contribution-slice-synchronize",
        "before-mesh-partition",
        "after-mesh-partition-enqueue",
        "before-mesh-partition-synchronize",
        "after-mesh-partition-synchronize",
    ]
    assert deallocated == ["source", "gathered", "selected"]


def test_resident_async_failure_retains_live_owners_and_submits_no_cleanup(monkeypatch, expect_error) -> None:
    deallocated = []
    retained = []

    class Topology:
        def distribution_shape(self):
            return (1, 4)

        def mesh_coords(self):
            return ((0, 0), (0, 1), (0, 2), (0, 3))

    class Tensor:
        def __init__(self, name, shape, padded):
            self.name, self.shape, self.padded_shape = name, shape, padded
            self.dtype, self.layout, self._topology = "bf16", "tile", Topology()

        def memory_config(self):
            return "dram"

        def tensor_topology(self):
            return self._topology

        def update_tensor_topology(self, topology):
            self._topology = topology

    source = Tensor("source", (1, 1, 1, 2560), (1, 1, 32, 2560))
    reference = Tensor("reference", (1, 1, 1, 1), (1, 1, 32, 32))
    gathered = Tensor("gathered", (1, 1, 1, 10240), (1, 1, 32, 10240))
    locals_by_tensor = {
        source: tuple(object() for _ in range(4)),
        gathered: tuple(object() for _ in range(4)),
    }

    fake_ttnn = SimpleNamespace(
        bfloat16="bf16",
        TILE_LAYOUT="tile",
        DRAM_MEMORY_CONFIG="dram",
        Topology=SimpleNamespace(Linear="linear"),
        PlacementReplicate=lambda: "replicate",
        TensorTopology=lambda *_args: Topology(),
        experimental=SimpleNamespace(all_gather_async=lambda *_args, **_kwargs: gathered),
        slice=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic slice enqueue failure")),
        get_device_tensors=lambda tensor: locals_by_tensor[tensor],
        synchronize_device=lambda *_args: pytest.fail("resident async embedding drained the host"),
        deallocate=lambda tensor: deallocated.append(tensor),
    )
    contract = SimpleNamespace(
        validate_mesh=lambda _mesh: None,
        validate_tensor=lambda _tensor, **_kwargs: None,
    )
    manager = SimpleNamespace(
        get_num_links=lambda cluster_axis: 1,
        get_and_cycle_ag_semaphore_handles=lambda cluster_axis: "axis1-semaphore",
    )
    monkeypatch.setattr(embedding_module, "ttnn", fake_ttnn)

    def retain(error, *owners):
        retained.append(error)
        retained.extend(owners)

    with expect_error(RuntimeError, "synthetic slice enqueue failure"):
        embedding_module._all_gather_owner_select_hidden_fallback(
            source,
            active_vocab_shard=0,
            active_vocab_coordinate=(0, 0),
            mesh_device=SimpleNamespace(shape=(1, 4)),
            mesh_contract=contract,
            replicated_reference=reference,
            tt_ccl=manager,
            collective_topology="linear",
            synchronization_policy=embedding_module.Qwen38TTNNEmbeddingSyncPolicy.RESIDENT_ASYNC,
            _retain_async_failure_owners=retain,
        )

    assert isinstance(retained[0], RuntimeError)
    assert retained[1:5] == [source, gathered, None, None]
    assert retained[5:] == [*locals_by_tensor[source], *locals_by_tensor[gathered]]
    assert deallocated == []
