# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""A trace must not run after a weight it read at capture was freed or replaced. Needs no device."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Callable
from types import SimpleNamespace

import pytest

import ttnn

from ...layers.module import Parameter
from ...pipelines.wan.quant_config import _quantize_weight
from ...utils.tracing import Tracer

_STALE = "^TDM-130: the trace of .*forward is stale: a weight of shape \\(2, 3\\) that the trace read at capture was "


class _Device:
    def id(self) -> int:
        return 0


@pytest.fixture
def executed(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Stands in for the device: counts the trace executions."""
    runs: list[object] = []
    monkeypatch.setattr(ttnn, "begin_trace_capture", lambda device, cq_id: object())
    monkeypatch.setattr(ttnn, "end_trace_capture", lambda device, trace_id, cq_id: None)
    monkeypatch.setattr(ttnn, "execute_trace", lambda device, trace_id, cq_id, blocking: runs.append(trace_id))
    monkeypatch.setattr(ttnn, "release_trace", lambda device, trace_id: None)
    monkeypatch.setattr(ttnn, "deallocate", lambda tensor: None)
    monkeypatch.setattr(Parameter, "_check_data", lambda self, value: None)
    return runs


def _parameter() -> Parameter:
    parameter = object.__new__(Parameter)
    parameter.total_shape = (2, 3)
    parameter._data = object()
    parameter._tracers = weakref.WeakSet()
    parameter.mesh_axes = parameter.durable_mesh_axes = (None, None)
    return parameter


def _shardable(monkeypatch: pytest.MonkeyPatch) -> Parameter:
    """A parameter that `shard(1)` splits along dim 0, on stand-ins for the device ops."""
    monkeypatch.setattr(ttnn, "mesh_partition", lambda tensor, dim, **kwargs: object())
    monkeypatch.setattr(ttnn, "reshape", lambda tensor, shape: object())
    parameter = _parameter()
    parameter.device = SimpleNamespace(shape=(1, 2))
    parameter.layout = ttnn.Layout.ROW_MAJOR
    parameter.on_host = False
    parameter.memory_config = object()
    return parameter


class _CCLManager:
    def __init__(self, parameter: Parameter) -> None:
        self._memory_config = parameter.memory_config

    def all_gather(self, tensor: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(memory_config=lambda: self._memory_config)


def _tracer(*parameters: Parameter, prep_run: bool = False) -> Tracer:
    def forward() -> None:
        for parameter in parameters:
            _ = parameter.data

    return Tracer(forward, device=_Device(), prep_run=prep_run)


def test_replay_after_deallocation_raises(executed: list[object], expect_error: Callable) -> None:
    weight = _parameter()
    tracer = _tracer(weight)
    tracer()
    tracer()
    assert len(executed) == 2
    weight.deallocate()
    with expect_error(RuntimeError, _STALE + "deallocated"):
        tracer()
    assert len(executed) == 2


def test_replay_after_a_reload_over_loaded_data_raises(executed: list[object], expect_error: Callable) -> None:
    weight = _parameter()
    tracer = _tracer(weight)
    tracer()
    weight.data = object()
    with expect_error(RuntimeError, _STALE + "replaced"):
        tracer()
    assert len(executed) == 1


def test_reload_after_eviction_stays_stale_until_recaptured(executed: list[object], expect_error: Callable) -> None:
    weight = _parameter()
    tracer = _tracer(weight)
    tracer()
    weight.deallocate()
    weight.data = object()  # reloaded, at whatever address
    with expect_error(RuntimeError, _STALE + "deallocated"):
        tracer()
    tracer.release_trace()
    tracer()
    tracer()
    assert len(executed) == 3


def test_weights_the_trace_did_not_read_do_not_matter(executed: list[object]) -> None:
    read, other = _parameter(), _parameter()
    tracer = _tracer(read)
    tracer()
    other.deallocate()
    other.data = object()
    tracer()
    assert len(executed) == 2


def test_reads_in_the_preparation_run_count(executed: list[object], expect_error: Callable) -> None:
    weight = _parameter()
    calls = []

    def forward() -> None:
        if not calls:
            _ = weight.data
        calls.append(None)

    tracer = Tracer(forward, device=_Device(), prep_run=True)
    tracer()
    weight.deallocate()
    with expect_error(RuntimeError, _STALE):
        tracer()


def test_deallocation_during_capture_raises_before_the_first_execution(
    executed: list[object], expect_error: Callable
) -> None:
    weight = _parameter()

    def forward() -> None:
        _ = weight.data
        weight.deallocate()

    with expect_error(RuntimeError, _STALE + "deallocated"):
        Tracer(forward, device=_Device(), prep_run=False)()
    assert executed == []


def test_reads_outside_a_capture_are_not_recorded(executed: list[object]) -> None:
    weight = _parameter()
    tracer = _tracer(weight)
    _ = weight.data
    assert len(weight._tracers) == 0
    tracer()
    assert len(weight._tracers) == 1
    tracer()
    _ = weight.data
    assert len(weight._tracers) == 1


def test_a_released_or_dropped_tracer_is_forgotten(executed: list[object]) -> None:
    weight = _parameter()
    tracer = _tracer(weight)
    tracer()
    tracer.release_trace()
    assert len(weight._tracers) == 0
    weight.deallocate()

    weight = _parameter()
    tracer = _tracer(weight)
    tracer()
    del tracer
    gc.collect()
    assert len(weight._tracers) == 0
    weight.deallocate()


def test_two_tracers_over_one_weight_both_go_stale(executed: list[object], expect_error: Callable) -> None:
    weight = _parameter()
    first, second = _tracer(weight), _tracer(weight)
    first()
    second()
    weight.deallocate()
    for tracer in (first, second):
        with expect_error(RuntimeError, _STALE):
            tracer()


def test_a_capture_during_another_capture_raises(executed: list[object], expect_error: Callable) -> None:
    weight = _parameter()
    inner = _tracer(weight)

    def outer_forward() -> None:
        inner()

    outer = Tracer(outer_forward, device=_Device(), prep_run=True)
    with expect_error(RuntimeError, "^cannot capture the trace of .*forward while the trace of .*outer_forward is"):
        outer()
    assert executed == []
    # The refused capture leaves nothing behind: each tracer captures alone afterwards.
    inner()
    weight.deallocate()
    with expect_error(RuntimeError, _STALE + "deallocated"):
        inner()


def test_a_quant_typecast_after_capture_makes_the_trace_stale(
    executed: list[object],
    monkeypatch: pytest.MonkeyPatch,
    expect_error: Callable,
) -> None:
    monkeypatch.setattr(ttnn, "typecast", lambda tensor, dtype: object())
    weight = _parameter()
    weight.dtype = "bfloat16"
    tracer = _tracer(weight)
    tracer()
    _quantize_weight(weight, "bfloat16")  # same dtype: nothing moves
    tracer()
    _quantize_weight(weight, "bfloat8_b")
    with expect_error(RuntimeError, _STALE + "replaced"):
        tracer()
    assert len(executed) == 2


def test_a_shard_after_capture_makes_the_trace_stale(
    executed: list[object], monkeypatch: pytest.MonkeyPatch, expect_error: Callable
) -> None:
    weight = _shardable(monkeypatch)
    tracer = _tracer(weight)
    tracer()
    weight.shard(1, defer_free=[])
    assert weight.is_transiently_sharded()
    with expect_error(RuntimeError, _STALE + "replaced"):
        tracer()
    assert len(executed) == 1


def test_a_gather_after_capture_makes_the_trace_stale(
    executed: list[object], monkeypatch: pytest.MonkeyPatch, expect_error: Callable
) -> None:
    weight = _shardable(monkeypatch)
    weight.shard(1, defer_free=[])
    # A capture cannot read a sharded parameter, so the trace watches it by hand. This shows that a gather
    # goes through the data setter; no pipeline can reach this sequence.
    tracer = _tracer()
    tracer()
    tracer.watch(weight)
    tracer()
    weight.gather(_CCLManager(weight), defer_free=[])
    assert not weight.is_transiently_sharded()
    with expect_error(RuntimeError, _STALE + "replaced"):
        tracer()
    assert len(executed) == 2
