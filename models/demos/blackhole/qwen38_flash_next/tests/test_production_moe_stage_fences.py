# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device ordering and lifetime gates for production TP4 MoE fences."""

from __future__ import annotations

import itertools
from unittest import mock

import pytest

import models.demos.blackhole.qwen38_flash_next.ttnn.moe as moe_module
from models.demos.blackhole.qwen38_flash_next.ttnn.moe import (
    HIDDEN_SIZE,
    MOE_STAGE_FENCES,
    ROUTED_EXPERTS,
    TOP_K,
    Qwen38TTNNMoE,
    Qwen38TTNNMoERowContract,
    Qwen38TTNNMoESyncPolicy,
    Qwen38TTNNRouting,
    Qwen38TTNNSharedPartial,
)


class _FakeTensor:
    _ids = itertools.count(1)

    def __init__(self, name: str, shape):
        self.name = name
        self.shape = tuple(shape)
        self.tensor_id = next(self._ids)


def _bare_moe() -> Qwen38TTNNMoE:
    instance = object.__new__(Qwen38TTNNMoE)
    instance.row_contract = Qwen38TTNNMoERowContract(1)
    instance.rows = 1
    instance.synchronization_policy = Qwen38TTNNMoESyncPolicy.CORRECTNESS_FENCED
    instance.mesh_device = object()
    instance.mesh_contract = mock.Mock()
    instance.tt_ccl = object()
    instance.weights = object()
    instance.compute_config = object()
    instance.routing_l1_memory_config = object()
    instance.expert_mapping = _FakeTensor("expert_mapping", (4, ROUTED_EXPERTS))
    instance.local_combine_output = _FakeTensor("local_combine_output", (TOP_K, 1, HIDDEN_SIZE))
    instance._owned_buffers_released = False
    instance._poisoned_error = None
    instance._poisoned_device_owners = []
    instance.collective_topology = object()
    return instance


def _forward_fixture(module: Qwen38TTNNMoE):
    contract = module.row_contract
    tensors = {
        "hidden_sharded": _FakeTensor("hidden_sharded", contract.hidden_sharded),
        "packed_w0_w1": _FakeTensor("packed_w0_w1", (1,)),
        "packed_w2": _FakeTensor("packed_w2", (1,)),
        "full_hidden": _FakeTensor("full_hidden", contract.full_hidden),
        "scores": _FakeTensor("scores", contract.routing),
        "indices": _FakeTensor("indices", contract.routing),
        "shared_partial": _FakeTensor("shared_partial", contract.full_hidden),
        "routed_partial": _FakeTensor("routed_partial", contract.full_hidden),
        "local_sum": _FakeTensor("local_sum", contract.full_hidden),
        "output": _FakeTensor("output", contract.output_sharded),
    }
    return tensors, Qwen38TTNNRouting(tensors["scores"], tensors["indices"])


def _install_forward_mocks(module, tensors, routing, events, monkeypatch) -> None:
    def route(full, *, hidden_tiles=None, phase_observer=None):
        assert hidden_tiles is None  # the one-row form hands the gathered shard itself to the router
        if phase_observer is not None:
            for phase in (
                "before-router-logits",
                "after-router-logits",
                "before-router-topk",
                "after-router-topk",
            ):
                phase_observer(phase)
        events.append(("route", full.name))
        return routing

    def no_reshard(*_args, **_kwargs):
        raise AssertionError("forward must hand the gathered shard to the linears without a reshard")

    def routed(full, route_result, w01, w2, *, phase_observer=None):
        if phase_observer is not None:
            for phase in (
                "before-routed-dispatch",
                "after-routed-dispatch",
                "before-moe-compute-launch",
                "after-moe-compute-launch",
                "before-selective-reduce",
                "after-selective-reduce",
            ):
                phase_observer(phase)
        events.append(("routed", full.name, route_result is routing, w01.name, w2.name))
        return tensors["routed_partial"]

    module._all_gather_hidden = mock.Mock(
        side_effect=lambda hidden: events.append(("all-gather", hidden.name)) or tensors["full_hidden"]
    )
    module._route = mock.Mock(side_effect=route)
    module._shared_partial = mock.Mock(
        side_effect=lambda hidden, full, tiles=None: events.append(("shared", hidden.name, full.name))
        or tensors["shared_partial"]
    )
    module._routed_partial = mock.Mock(side_effect=routed)
    monkeypatch.setattr(moe_module.ttnn, "to_memory_config", no_reshard)
    module._synchronize_stage = mock.Mock(side_effect=lambda stage: events.append(("fence", stage)))
    module.mesh_contract.mark_local_partial.side_effect = lambda tensor, **_kwargs: events.append(
        ("mark-local", tensor.name)
    )
    module.mesh_contract.mark_collective_shard.side_effect = lambda tensor, **_kwargs: events.append(
        ("mark-collective", tensor.name)
    )
    monkeypatch.setattr(
        moe_module.ttnn,
        "add",
        lambda left, right, **_kwargs: events.append(("add", left.name, right.name)) or tensors["local_sum"],
    )
    monkeypatch.setattr(
        moe_module,
        "tt_all_reduce",
        lambda tensor, *_args, **_kwargs: events.append(("all-reduce", tensor.name)) or tensors["output"],
    )
    monkeypatch.setattr(
        moe_module.ttnn,
        "deallocate",
        lambda tensor: events.append(("deallocate", tensor.name)),
    )


def test_named_stage_fence_is_deterministic_and_fail_closed(expect_error, monkeypatch) -> None:
    module = _bare_moe()
    synchronize = mock.Mock()
    monkeypatch.setattr(moe_module.ttnn, "synchronize_device", synchronize)

    for stage in MOE_STAGE_FENCES:
        module._synchronize_stage(stage)

    assert synchronize.call_args_list == [mock.call(module.mesh_device) for _stage in MOE_STAGE_FENCES]
    with expect_error(ValueError, match="unknown production MoE stage fence"):  # allow-pytest.raises: pure contract
        module._synchronize_stage("not-a-stage")
    assert synchronize.call_count == len(MOE_STAGE_FENCES)


def test_resident_async_stage_boundaries_are_trace_safe_and_never_drain(expect_error, monkeypatch) -> None:
    module = _bare_moe()
    module.synchronization_policy = Qwen38TTNNMoESyncPolicy.RESIDENT_ASYNC
    synchronize = mock.Mock(side_effect=AssertionError("resident trace path drained the host"))
    monkeypatch.setattr(moe_module.ttnn, "synchronize_device", synchronize)

    for stage in MOE_STAGE_FENCES:
        module._synchronize_stage(stage)

    synchronize.assert_not_called()
    with expect_error(ValueError, match="unknown production MoE stage fence"):  # allow-pytest.raises: pure contract
        module._synchronize_stage("not-a-stage")


def test_resident_async_forward_executes_all_six_boundaries_without_a_host_drain(monkeypatch) -> None:
    module = _bare_moe()
    module.synchronization_policy = Qwen38TTNNMoESyncPolicy.RESIDENT_ASYNC
    tensors, routing = _forward_fixture(module)
    events = []
    _install_forward_mocks(module, tensors, routing, events, monkeypatch)
    # Use the real policy method rather than the event-only fence stub.
    del module._synchronize_stage
    synchronize = mock.Mock(side_effect=AssertionError("resident trace path drained the host"))
    monkeypatch.setattr(moe_module.ttnn, "synchronize_device", synchronize)

    result = module.forward(
        tensors["hidden_sharded"],
        tensors["packed_w0_w1"],
        tensors["packed_w2"],
        return_routing=True,
    )

    assert result.hidden_sharded is tensors["output"]
    assert result.routing is routing
    synchronize.assert_not_called()


def test_resident_async_failure_poison_retains_live_owners_and_skips_cleanup(expect_error, monkeypatch) -> None:
    module = _bare_moe()
    module.synchronization_policy = Qwen38TTNNMoESyncPolicy.RESIDENT_ASYNC
    tensors, routing = _forward_fixture(module)
    events = []
    _install_forward_mocks(module, tensors, routing, events, monkeypatch)
    del module._synchronize_stage
    module._shared_partial = mock.Mock(side_effect=RuntimeError("injected async enqueue failure"))
    synchronize = mock.Mock(side_effect=AssertionError("resident failure path drained the host"))
    monkeypatch.setattr(moe_module.ttnn, "synchronize_device", synchronize)

    with expect_error(RuntimeError, match="live device owners retained and cleanup skipped"):
        module.forward(tensors["hidden_sharded"], tensors["packed_w0_w1"], tensors["packed_w2"])

    synchronize.assert_not_called()
    assert module._poisoned_error is not None
    assert module._poisoned_device_owners == [
        tensors["hidden_sharded"],
        tensors["packed_w0_w1"],
        tensors["packed_w2"],
        tensors["full_hidden"],
        tensors["scores"],
        tensors["indices"],
    ]
    assert not any(event[0] == "deallocate" for event in events)
    with expect_error(RuntimeError, match="process/mesh teardown"):
        module.release_owned_buffers()


def test_forward_fences_six_stages_and_holds_collective_inputs(monkeypatch) -> None:
    module = _bare_moe()
    tensors, routing = _forward_fixture(module)
    events = []
    _install_forward_mocks(module, tensors, routing, events, monkeypatch)

    result = module.forward(
        tensors["hidden_sharded"],
        tensors["packed_w0_w1"],
        tensors["packed_w2"],
        return_routing=True,
    )

    assert result.hidden_sharded is tensors["output"]
    assert result.routing is routing
    assert events == [
        ("all-gather", "hidden_sharded"),
        ("fence", "all-gather-hidden"),
        ("route", "full_hidden"),
        ("fence", "route"),
        ("shared", "hidden_sharded", "full_hidden"),
        ("fence", "shared-partial"),
        ("routed", "full_hidden", True, "packed_w0_w1", "packed_w2"),
        ("fence", "routed-partial"),
        ("add", "routed_partial", "shared_partial"),
        ("deallocate", "routed_partial"),
        ("deallocate", "shared_partial"),
        ("mark-local", "local_sum"),
        ("fence", "local-add-mark"),
        ("all-reduce", "local_sum"),
        ("mark-collective", "output"),
        ("fence", "final-all-reduce"),
        ("deallocate", "local_sum"),
        ("deallocate", "full_hidden"),
    ]
    released = {event[1] for event in events if event[0] == "deallocate"}
    assert not released.intersection(
        {
            "hidden_sharded",
            "packed_w0_w1",
            "packed_w2",
            "expert_mapping",
            "local_combine_output",
            "output",
            "scores",
            "indices",
        }
    )


def test_forward_phase_observer_brackets_exact_moe_mechanics_without_tensors(monkeypatch) -> None:
    module = _bare_moe()
    tensors, routing = _forward_fixture(module)
    events = []
    phases = []
    _install_forward_mocks(module, tensors, routing, events, monkeypatch)

    def observe(phase: str) -> None:
        assert type(phase) is str
        phases.append(phase)

    result = module.forward(
        tensors["hidden_sharded"],
        tensors["packed_w0_w1"],
        tensors["packed_w2"],
        return_routing=True,
        phase_observer=observe,
    )

    assert result.hidden_sharded is tensors["output"]
    assert result.routing is routing
    assert phases == [
        "before-hidden-all-gather",
        "after-hidden-all-gather",
        "before-router-logits",
        "after-router-logits",
        "before-router-topk",
        "after-router-topk",
        "before-shared-partial",
        "after-shared-partial",
        "before-routed-dispatch",
        "after-routed-dispatch",
        "before-moe-compute-launch",
        "after-moe-compute-launch",
        "before-selective-reduce",
        "after-selective-reduce",
        "before-partial-combine",
        "after-partial-combine",
        "before-output-reduce-scatter",
        "after-output-reduce-scatter",
        "before-output-release",
        "after-output-release",
    ]


def test_forward_with_the_dense_composite_brackets_one_program_and_releases_its_rows(monkeypatch) -> None:
    """With the MoE dense composite on (moe_dense_fused), forward takes _dense_composite instead of _route and
    _shared_partial (the router linear, the shared linear and ONE program for the top-k, the shared eltwise + down
    linear and the routed dispatch untilize), hands the composite's rows to the fused local sum and releases them
    right after it, before the combine collective; the phase observer sees the shared-partial pair bracketing the
    top-k pair (the shared work runs under the top-k), the six fences stay six."""

    module = _bare_moe()
    tensors, routing = _forward_fixture(module)
    events = []
    _install_forward_mocks(module, tensors, routing, events, monkeypatch)
    module.moe_dense_fused = True
    module.moe_post_fused = True  # the fused local sum owns the expert-owner rows too
    module.expert_owner = _FakeTensor("expert_owner", (1, ROUTED_EXPERTS))
    shared = Qwen38TTNNSharedPartial(tensors["shared_partial"], _FakeTensor("sigmoid", (1, 1, 32, 32)))
    sparse_rows = _FakeTensor("sparse_rows", module.row_contract.full_hidden)
    composite_phases = (
        "before-router-logits",
        "after-router-logits",
        "before-shared-partial",
        "before-router-topk",
        "after-router-topk",
        "after-shared-partial",
    )

    def composite(full, observe):
        for phase in composite_phases:
            observe(phase)
        events.append(("composite", full.name))
        return routing, shared, sparse_rows

    def local_sum(full, route_result, w01, w2, shared_arg, *, phase_observer, sparse_rows=None):
        for phase in (
            "before-routed-dispatch",
            "after-routed-dispatch",
            "before-moe-compute-launch",
            "after-moe-compute-launch",
            "before-selective-reduce",
            "after-selective-reduce",
        ):
            phase_observer(phase)
        events.append(("local-sum", full.name, route_result is routing, shared_arg is shared, sparse_rows.name))
        return tensors["local_sum"]

    module._dense_composite = composite
    module._routed_local_sum = local_sum
    phases = []
    result = module.forward(
        tensors["hidden_sharded"], tensors["packed_w0_w1"], tensors["packed_w2"], phase_observer=phases.append
    )
    assert result.hidden_sharded is tensors["output"]
    module._route.assert_not_called()
    module._shared_partial.assert_not_called()
    module._routed_partial.assert_not_called()
    assert phases == [
        "before-hidden-all-gather",
        "after-hidden-all-gather",
        *composite_phases,
        "before-routed-dispatch",
        "after-routed-dispatch",
        "before-moe-compute-launch",
        "after-moe-compute-launch",
        "before-selective-reduce",
        "after-selective-reduce",
        "before-partial-combine",
        "after-partial-combine",
        "before-output-reduce-scatter",
        "after-output-reduce-scatter",
        "before-output-release",
        "after-output-release",
    ]
    fences = [event[1] for event in events if event[0] == "fence"]
    assert fences == list(MOE_STAGE_FENCES)
    order = [event for event in events if event[0] in ("composite", "local-sum", "deallocate", "all-reduce")]
    local = next(i for i, event in enumerate(order) if event[0] == "local-sum")
    assert order[local] == ("local-sum", "full_hidden", True, True, "sparse_rows")
    released = order.index(("deallocate", "sparse_rows"))
    assert local < released < order.index(("all-reduce", "local_sum"))
    assert order.index(("deallocate", "shared_partial")) < order.index(("all-reduce", "local_sum"))
    assert order.index(("deallocate", "sigmoid")) < order.index(("all-reduce", "local_sum"))
    assert not any(event[0] == "add" for event in events)


def test_forward_phase_observer_failure_defers_until_routed_partial_has_cleanup_ownership(
    expect_error, monkeypatch
) -> None:
    module = _bare_moe()
    tensors, routing = _forward_fixture(module)
    events = []
    _install_forward_mocks(module, tensors, routing, events, monkeypatch)

    def fail_after_moe_compute_launch(phase: str) -> None:
        if phase == "after-moe-compute-launch":
            raise RuntimeError("injected MoE phase observer failure")

    with expect_error(RuntimeError, match="injected MoE phase observer failure"):
        module.forward(
            tensors["hidden_sharded"],
            tensors["packed_w0_w1"],
            tensors["packed_w2"],
            phase_observer=fail_after_moe_compute_launch,
        )

    released = [event[1] for event in events if event[0] == "deallocate"]
    assert released == ["routed_partial", "shared_partial", "full_hidden", "scores", "indices"]
    assert not set(released).intersection(
        {"hidden_sharded", "packed_w0_w1", "packed_w2", "expert_mapping", "local_combine_output"}
    )


@pytest.mark.parametrize("failed_owner", ["routed_partial", "shared_partial"])
def test_partial_release_failure_retries_only_the_still_owned_tensor(
    expect_error, monkeypatch, failed_owner: str
) -> None:
    module = _bare_moe()
    tensors, routing = _forward_fixture(module)
    events = []
    _install_forward_mocks(module, tensors, routing, events, monkeypatch)
    attempts = []
    injected = False

    def fail_once(tensor):
        nonlocal injected
        attempts.append(tensor.name)
        if tensor.name == failed_owner and not injected:
            injected = True
            raise RuntimeError(f"{failed_owner} release failed")
        events.append(("deallocate", tensor.name))

    monkeypatch.setattr(moe_module.ttnn, "deallocate", fail_once)
    with expect_error(RuntimeError, match="release failed"):  # allow-pytest.raises: injected failure
        module.forward(
            tensors["hidden_sharded"],
            tensors["packed_w0_w1"],
            tensors["packed_w2"],
            return_routing=True,
        )

    assert attempts.count(failed_owner) == 2
    if failed_owner == "shared_partial":
        assert attempts.count("routed_partial") == 1
    assert all(attempts.count(name) == 1 for name in {"local_sum", "full_hidden", "scores", "indices"})
    assert not set(attempts).intersection(
        {"hidden_sharded", "packed_w0_w1", "packed_w2", "expert_mapping", "local_combine_output", "output"}
    )


def test_full_hidden_release_failure_does_not_retry_released_local_sum(expect_error, monkeypatch) -> None:
    module = _bare_moe()
    tensors, routing = _forward_fixture(module)
    events = []
    _install_forward_mocks(module, tensors, routing, events, monkeypatch)
    attempts = []
    injected = False

    def fail_full_hidden_once(tensor):
        nonlocal injected
        attempts.append(tensor.name)
        if tensor.name == "full_hidden" and not injected:
            injected = True
            raise RuntimeError("full_hidden release failed")
        events.append(("deallocate", tensor.name))

    monkeypatch.setattr(moe_module.ttnn, "deallocate", fail_full_hidden_once)
    with expect_error(RuntimeError, match="release failed"):  # allow-pytest.raises: injected failure
        module.forward(
            tensors["hidden_sharded"],
            tensors["packed_w0_w1"],
            tensors["packed_w2"],
            return_routing=True,
        )

    assert attempts.count("local_sum") == 1
    assert attempts.count("full_hidden") == 2
    assert attempts.count("output") == 1
    assert attempts.count("scores") == attempts.count("indices") == 1
    assert not set(attempts).intersection(
        {"hidden_sharded", "packed_w0_w1", "packed_w2", "expert_mapping", "local_combine_output"}
    )


def test_routing_indices_release_failure_does_not_retry_released_scores(expect_error, monkeypatch) -> None:
    module = _bare_moe()
    tensors, routing = _forward_fixture(module)
    events = []
    _install_forward_mocks(module, tensors, routing, events, monkeypatch)
    attempts = []
    injected = False

    def fail_indices_once(tensor):
        nonlocal injected
        attempts.append(tensor.name)
        if tensor.name == "indices" and not injected:
            injected = True
            raise RuntimeError("indices release failed")
        events.append(("deallocate", tensor.name))

    monkeypatch.setattr(moe_module.ttnn, "deallocate", fail_indices_once)
    with expect_error(RuntimeError, match="release failed"):  # allow-pytest.raises: injected failure
        module.forward(tensors["hidden_sharded"], tensors["packed_w0_w1"], tensors["packed_w2"])

    assert attempts.count("scores") == 1
    assert attempts.count("indices") == 2
    assert attempts.count("local_sum") == attempts.count("full_hidden") == 1
    assert attempts.count("output") == 1
    assert not set(attempts).intersection(
        {"hidden_sharded", "packed_w0_w1", "packed_w2", "expert_mapping", "local_combine_output"}
    )


def test_final_fence_failure_releases_unreturned_output_and_live_inputs(expect_error, monkeypatch) -> None:
    module = _bare_moe()
    tensors, routing = _forward_fixture(module)
    events = []
    _install_forward_mocks(module, tensors, routing, events, monkeypatch)

    def fence(stage):
        events.append(("fence", stage))
        if stage == "final-all-reduce":
            raise RuntimeError("final fence failed")

    module._synchronize_stage = mock.Mock(side_effect=fence)
    with expect_error(RuntimeError, match="final fence failed"):  # allow-pytest.raises: injected failure
        module.forward(
            tensors["hidden_sharded"],
            tensors["packed_w0_w1"],
            tensors["packed_w2"],
            return_routing=True,
        )

    released = [event[1] for event in events if event[0] == "deallocate"]
    assert released == [
        "routed_partial",
        "shared_partial",
        "output",
        "local_sum",
        "full_hidden",
        "scores",
        "indices",
    ]
    assert not set(released).intersection(
        {"hidden_sharded", "packed_w0_w1", "packed_w2", "expert_mapping", "local_combine_output"}
    )


def test_diagnostic_failure_hook_drains_before_first_exception_cleanup_release(expect_error, monkeypatch) -> None:
    module = _bare_moe()
    tensors, routing = _forward_fixture(module)
    events = []
    _install_forward_mocks(module, tensors, routing, events, monkeypatch)

    def fence(stage):
        events.append(("fence", stage))
        if stage == "final-all-reduce":
            raise RuntimeError("final fence failed")

    def before_cleanup(error, owners):
        assert str(error) == "final fence failed"
        assert tuple(owners) == (
            "full_hidden",
            "hidden_tiles",
            "routing_scores",
            "routing_indices",
            "routing_tiles",
            "shared_partial",
            "sparse_rows",
            "routed_partial",
            "local_sum",
            "output",
        )
        events.append(("diagnostic-drain", tuple(name for name, value in owners.items() if value is not None)))

    module._synchronize_stage = mock.Mock(side_effect=fence)
    module._diagnostic_before_exception_cleanup = before_cleanup
    with expect_error(RuntimeError, match="final fence failed"):  # allow-pytest.raises: injected failure
        module.forward(
            tensors["hidden_sharded"],
            tensors["packed_w0_w1"],
            tensors["packed_w2"],
            return_routing=True,
        )

    drain_index = next(index for index, event in enumerate(events) if event[0] == "diagnostic-drain")
    cleanup_indices = [
        index
        for index, event in enumerate(events)
        if event[0] == "deallocate" and event[1] in {"output", "local_sum", "full_hidden", "scores", "indices"}
    ]
    assert cleanup_indices
    assert drain_index < min(cleanup_indices)


def test_failed_diagnostic_pre_cleanup_drain_retains_live_owners_and_skips_cleanup(expect_error, monkeypatch) -> None:
    module = _bare_moe()
    tensors, routing = _forward_fixture(module)
    events = []
    retained = []
    _install_forward_mocks(module, tensors, routing, events, monkeypatch)

    def fence(stage):
        events.append(("fence", stage))
        if stage == "final-all-reduce":
            raise RuntimeError("final fence failed")

    def fail_before_cleanup(_error, owners):
        retained.extend(value for value in owners.values() if value is not None)
        events.append(("diagnostic-drain-failed",))
        raise RuntimeError("injected diagnostic drain failure")

    module._synchronize_stage = mock.Mock(side_effect=fence)
    module._diagnostic_before_exception_cleanup = fail_before_cleanup
    with expect_error(RuntimeError, match="diagnostic pre-cleanup drain also failed; cleanup skipped"):
        module.forward(
            tensors["hidden_sharded"],
            tensors["packed_w0_w1"],
            tensors["packed_w2"],
            return_routing=True,
        )

    assert retained == [
        tensors["full_hidden"],
        tensors["scores"],
        tensors["indices"],
        tensors["local_sum"],
        tensors["output"],
    ]
    drain_index = events.index(("diagnostic-drain-failed",))
    assert not any(index > drain_index and event[0] == "deallocate" for index, event in enumerate(events))
