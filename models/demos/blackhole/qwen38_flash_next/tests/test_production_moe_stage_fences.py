# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device ordering and lifetime gates for production TP4 MoE fences."""

from __future__ import annotations

import itertools
from types import SimpleNamespace
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

ROUTER_PHASES = ("before-router-logits", "after-router-logits", "before-router-topk", "after-router-topk")
ROUTED_PHASES = (
    "before-routed-dispatch",
    "after-routed-dispatch",
    "before-moe-compute-launch",
    "after-moe-compute-launch",
    "before-selective-reduce",
    "after-selective-reduce",
)
COMBINE_PHASES = (
    "before-partial-combine",
    "after-partial-combine",
    "before-output-reduce-scatter",
    "after-output-reduce-scatter",
    "before-output-release",
    "after-output-release",
)
GATHER_PHASES = ("before-hidden-all-gather", "after-hidden-all-gather")
NEVER_RELEASED = {"hidden_sharded", "packed_w0_w1", "packed_w2", "expert_mapping", "local_combine_output"}


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
    instance.tt_ccl = SimpleNamespace(get_num_links=lambda axis: 2)
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


@pytest.fixture
def moe(monkeypatch):
    """A bare one-row MoE whose forward stages, fences, collectives and releases are fakes logging to ``events``; the
    fence named ``failing_fence`` raises."""
    module = _bare_moe()
    contract = module.row_contract
    tensors = {
        name: _FakeTensor(name, shape)
        for name, shape in (
            ("hidden_sharded", contract.hidden_sharded),
            ("packed_w0_w1", (1,)),
            ("packed_w2", (1,)),
            ("full_hidden", contract.full_hidden),
            ("scores", contract.routing),
            ("indices", contract.routing),
            ("shared_partial", contract.full_hidden),
            ("routed_partial", contract.full_hidden),
            ("local_sum", contract.full_hidden),
            ("output", contract.output_sharded),
        )
    }
    routing = Qwen38TTNNRouting(tensors["scores"], tensors["indices"])
    events = []
    fakes = SimpleNamespace(module=module, tensors=tensors, routing=routing, events=events, failing_fence=None)

    def route(full, *, hidden_tiles=None, phase_observer=None):
        assert hidden_tiles is None  # the one-row form hands the gathered shard itself to the router
        for phase in ROUTER_PHASES if phase_observer is not None else ():
            phase_observer(phase)
        events.append(("route", full.name))
        return routing

    def no_reshard(*_args, **_kwargs):
        raise AssertionError("forward must hand the gathered shard to the linears without a reshard")

    def routed(full, route_result, w01, w2, *, phase_observer=None):
        for phase in ROUTED_PHASES if phase_observer is not None else ():
            phase_observer(phase)
        events.append(("routed", full.name, route_result is routing, w01.name, w2.name))
        return tensors["routed_partial"]

    def fence(stage):
        events.append(("fence", stage))
        if stage == fakes.failing_fence:
            raise RuntimeError(f"{stage} fence failed")

    module._all_gather_hidden = mock.Mock(
        side_effect=lambda hidden: events.append(("all-gather", hidden.name)) or tensors["full_hidden"]
    )
    module._route = mock.Mock(side_effect=route)
    module._shared_partial = mock.Mock(
        side_effect=lambda hidden, full, tiles=None: events.append(("shared", hidden.name, full.name))
        or tensors["shared_partial"]
    )
    module._routed_partial = mock.Mock(side_effect=routed)
    module._synchronize_stage = mock.Mock(side_effect=fence)
    module.mesh_contract.mark_local_partial.side_effect = lambda tensor, **_: events.append(("mark-local", tensor.name))
    module.mesh_contract.mark_collective_shard.side_effect = lambda tensor, **_: events.append(
        ("mark-collective", tensor.name)
    )
    monkeypatch.setattr(moe_module.ttnn, "to_memory_config", no_reshard)
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
    monkeypatch.setattr(moe_module.ttnn, "deallocate", lambda tensor: events.append(("deallocate", tensor.name)))
    fakes.forward = lambda **kwargs: module.forward(
        tensors["hidden_sharded"], tensors["packed_w0_w1"], tensors["packed_w2"], **kwargs
    )
    return fakes


def _released(events) -> list[str]:
    return [event[1] for event in events if event[0] == "deallocate"]


@pytest.mark.parametrize(
    "policy", [Qwen38TTNNMoESyncPolicy.CORRECTNESS_FENCED, Qwen38TTNNMoESyncPolicy.RESIDENT_ASYNC], ids=str
)
def test_named_stage_fence_is_deterministic_and_fail_closed(expect_error, monkeypatch, policy) -> None:
    """A correctness-fenced stage drains the mesh once per named stage; a resident (trace-safe) one never drains."""
    module = _bare_moe()
    module.synchronization_policy = policy
    drains = policy is Qwen38TTNNMoESyncPolicy.CORRECTNESS_FENCED
    synchronize = mock.Mock(side_effect=None if drains else AssertionError("resident trace path drained the host"))
    monkeypatch.setattr(moe_module.ttnn, "synchronize_device", synchronize)

    for stage in MOE_STAGE_FENCES:
        module._synchronize_stage(stage)
    with expect_error(ValueError, match="unknown production MoE stage fence"):  # allow-pytest.raises: pure contract
        module._synchronize_stage("not-a-stage")

    assert synchronize.call_args_list == [mock.call(module.mesh_device)] * (len(MOE_STAGE_FENCES) if drains else 0)


def _resident_async(moe, monkeypatch) -> mock.Mock:
    """The resident policy with the real fence method; returns the ``synchronize_device`` that must stay uncalled."""
    moe.module.synchronization_policy = Qwen38TTNNMoESyncPolicy.RESIDENT_ASYNC
    del moe.module._synchronize_stage
    synchronize = mock.Mock(side_effect=AssertionError("resident trace path drained the host"))
    monkeypatch.setattr(moe_module.ttnn, "synchronize_device", synchronize)
    return synchronize


def test_resident_async_forward_executes_all_six_boundaries_without_a_host_drain(moe, monkeypatch) -> None:
    synchronize = _resident_async(moe, monkeypatch)
    result = moe.forward(return_routing=True)
    assert result.hidden_sharded is moe.tensors["output"] and result.routing is moe.routing
    synchronize.assert_not_called()


def test_resident_async_failure_poison_retains_live_owners_and_skips_cleanup(expect_error, moe, monkeypatch) -> None:
    synchronize = _resident_async(moe, monkeypatch)
    moe.module._shared_partial = mock.Mock(side_effect=RuntimeError("injected async enqueue failure"))
    with expect_error(RuntimeError, match="live device owners retained and cleanup skipped"):
        moe.forward()

    synchronize.assert_not_called()
    assert moe.module._poisoned_error is not None
    owners = ("hidden_sharded", "packed_w0_w1", "packed_w2", "full_hidden", "scores", "indices")
    assert moe.module._poisoned_device_owners == [moe.tensors[name] for name in owners]
    assert _released(moe.events) == []
    with expect_error(RuntimeError, match="process/mesh teardown"):
        moe.module.release_owned_buffers()


def test_forward_fences_six_stages_and_holds_collective_inputs(moe) -> None:
    result = moe.forward(return_routing=True)

    assert result.hidden_sharded is moe.tensors["output"] and result.routing is moe.routing
    assert moe.events == [
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
    assert not set(_released(moe.events)) & (NEVER_RELEASED | {"output", "scores", "indices"})


def test_forward_phase_observer_brackets_exact_moe_mechanics_without_tensors(moe) -> None:
    phases = []

    def observe(phase: str) -> None:
        assert type(phase) is str
        phases.append(phase)

    result = moe.forward(return_routing=True, phase_observer=observe)

    assert result.hidden_sharded is moe.tensors["output"] and result.routing is moe.routing
    assert phases == [
        *GATHER_PHASES,
        *ROUTER_PHASES,
        "before-shared-partial",
        "after-shared-partial",
        *ROUTED_PHASES,
        *COMBINE_PHASES,
    ]


def test_forward_with_the_dense_composite_brackets_one_program_and_releases_its_rows(moe) -> None:
    """With moe_dense_fused on, forward takes _dense_composite instead of _route and _shared_partial, hands its rows
    to the fused local sum and releases them before the combine collective; the shared-partial pair brackets the top-k
    pair and the six fences stay six."""

    module, tensors, events = moe.module, moe.tensors, moe.events
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
        return moe.routing, shared, sparse_rows

    def local_sum(full, route_result, w01, w2, shared_arg, *, phase_observer, sparse_rows=None):
        for phase in ROUTED_PHASES:
            phase_observer(phase)
        events.append(("local-sum", full.name, route_result is moe.routing, shared_arg is shared, sparse_rows.name))
        return tensors["local_sum"]

    module._dense_composite = composite
    module._routed_local_sum = local_sum
    phases = []
    result = moe.forward(phase_observer=phases.append)
    assert result.hidden_sharded is tensors["output"]
    module._route.assert_not_called()
    module._shared_partial.assert_not_called()
    module._routed_partial.assert_not_called()
    assert phases == [*GATHER_PHASES, *composite_phases, *ROUTED_PHASES, *COMBINE_PHASES]
    assert [event[1] for event in events if event[0] == "fence"] == list(MOE_STAGE_FENCES)
    order = [event for event in events if event[0] in ("composite", "local-sum", "deallocate", "all-reduce")]
    local = next(i for i, event in enumerate(order) if event[0] == "local-sum")
    assert order[local] == ("local-sum", "full_hidden", True, True, "sparse_rows")
    combine = order.index(("all-reduce", "local_sum"))
    assert local < order.index(("deallocate", "sparse_rows")) < combine
    assert order.index(("deallocate", "shared_partial")) < combine
    assert order.index(("deallocate", "sigmoid")) < combine
    assert not any(event[0] == "add" for event in events)


def test_composed_post_gates_the_fused_shared_partial_before_the_add(moe, monkeypatch) -> None:
    """QWEN38_FUSED_OFF=moe_post with the fused shared expert on: the shared partial arrives ungated with its sigmoid
    tile, so the composed sum slices the tile's first column, multiplies, adds the product and releases all three."""

    module, events = moe.module, moe.events
    shared = Qwen38TTNNSharedPartial(moe.tensors["shared_partial"], _FakeTensor("sigmoid", (1, 1, 32, 32)))
    module._shared_partial = mock.Mock(side_effect=lambda hidden, full, tiles=None: shared)
    gate = _FakeTensor("gate", (1, 1, module.rows, 1))
    gated = _FakeTensor("gated_shared_partial", module.row_contract.full_hidden)

    def slice_(tensor, start, end, *, memory_config):
        events.append(("slice", tensor.name, start, end))
        return gate

    def mul(left, right, *, memory_config):
        events.append(("mul", left.name, right.name))
        return gated

    monkeypatch.setattr(moe_module.ttnn, "slice", slice_)
    monkeypatch.setattr(moe_module.ttnn, "mul", mul)
    result = moe.forward()

    assert result.hidden_sharded is moe.tensors["output"], f"forward returned {result.hidden_sharded}"
    combine = [event for event in events if event[0] in ("slice", "mul", "add", "deallocate")][:7]
    assert combine == [
        ("slice", "sigmoid", (0, 0, 0, 0), (1, 1, module.rows, 1)),
        ("mul", "shared_partial", "gate"),
        ("deallocate", "gate"),
        ("deallocate", "shared_partial"),
        ("deallocate", "sigmoid"),
        ("add", "routed_partial", "gated_shared_partial"),
        ("deallocate", "routed_partial"),
    ], f"combine events {combine}"
    released = _released(events)
    assert released.count("gated_shared_partial") == 1, f"released {released}"


def test_forward_phase_observer_failure_defers_until_routed_partial_has_cleanup_ownership(expect_error, moe) -> None:
    def fail_after_moe_compute_launch(phase: str) -> None:
        if phase == "after-moe-compute-launch":
            raise RuntimeError("injected MoE phase observer failure")

    with expect_error(RuntimeError, match="injected MoE phase observer failure"):
        moe.forward(phase_observer=fail_after_moe_compute_launch)

    released = _released(moe.events)
    assert released == ["routed_partial", "shared_partial", "full_hidden", "scores", "indices"]
    assert not set(released) & NEVER_RELEASED


@pytest.mark.parametrize(
    "failed_owner, return_routing, attempts_per_owner",
    [
        ("routed_partial", True, {"routed_partial": 2, "output": 0}),
        ("shared_partial", True, {"shared_partial": 2, "routed_partial": 1, "output": 0}),
        ("full_hidden", True, {"full_hidden": 2, "output": 1}),
        ("indices", False, {"indices": 2, "output": 1}),
    ],
    ids=["routed_partial", "shared_partial", "full_hidden", "indices"],
)
def test_release_failure_retries_only_the_still_owned_tensor(
    expect_error, moe, monkeypatch, failed_owner: str, return_routing: bool, attempts_per_owner: dict
) -> None:
    """The failed release is retried once; every other owner (the released local sum and scores included) is
    released exactly once and forward's inputs and the instance's buffers never."""
    attempts = []

    def fail_once(tensor):
        attempts.append(tensor.name)
        if tensor.name == failed_owner and attempts.count(failed_owner) == 1:
            raise RuntimeError(f"{failed_owner} release failed")
        moe.events.append(("deallocate", tensor.name))

    monkeypatch.setattr(moe_module.ttnn, "deallocate", fail_once)
    with expect_error(RuntimeError, match="release failed"):  # allow-pytest.raises: injected failure
        moe.forward(return_routing=return_routing)

    expected = dict.fromkeys(("local_sum", "full_hidden", "scores", "indices"), 1) | attempts_per_owner
    assert {name: attempts.count(name) for name in expected} == expected, attempts
    assert not set(attempts) & NEVER_RELEASED


def test_final_fence_failure_releases_unreturned_output_and_live_inputs(expect_error, moe) -> None:
    moe.failing_fence = "final-all-reduce"
    with expect_error(RuntimeError, match="final-all-reduce fence failed"):  # allow-pytest.raises: injected failure
        moe.forward(return_routing=True)

    released = _released(moe.events)
    assert released == ["routed_partial", "shared_partial", "output", "local_sum", "full_hidden", "scores", "indices"]
    assert not set(released) & NEVER_RELEASED
