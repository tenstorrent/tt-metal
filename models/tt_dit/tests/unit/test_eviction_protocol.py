# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""A traced pipeline releases its traces before it evicts a module, and captures them again. Needs no device."""

from __future__ import annotations

import gc
import threading
import weakref
from collections import namedtuple
from collections.abc import Callable

import pytest
from loguru import logger

import ttnn

from ...layers.module import Module, Parameter
from ...pipelines.ideogram4.pipeline_ideogram4 import Ideogram4Pipeline
from ...pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ...pipelines.pipeline_api import Pipeline, owns_captures
from ...utils import cache, tracing
from ...utils.tracing import Tracer

_NoParallelism = namedtuple("_NoParallelism", [])


class _Device:
    def id(self) -> int:
        return 0


@pytest.fixture(autouse=True)
def events(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stands in for the device, and records trace releases and weight frees in the order they happen."""
    log: list[str] = []
    monkeypatch.setattr(ttnn, "begin_trace_capture", lambda device, cq_id: object())
    monkeypatch.setattr(ttnn, "end_trace_capture", lambda device, trace_id, cq_id: None)
    monkeypatch.setattr(ttnn, "execute_trace", lambda device, trace_id, cq_id, blocking: log.append("execute"))
    monkeypatch.setattr(ttnn, "release_trace", lambda device, trace_id: log.append("release"))
    monkeypatch.setattr(ttnn, "deallocate", lambda tensor: None)
    monkeypatch.setattr(ttnn, "distributed_context_barrier", lambda: None)
    monkeypatch.setattr(Parameter, "_check_data", lambda self, value: None)
    # The count of captured tracers is one per process: start each test at zero.
    monkeypatch.setattr(Tracer, "_captured", weakref.WeakSet())
    return log


class _Weight:
    """Stands in for a `Parameter`: records its frees in the shared event log."""

    def __init__(self, log: list[str]) -> None:
        self._log = log

    def load_torch_tensor(self, value: object) -> None:
        pass

    def deallocate(self) -> None:
        self._log.append("deallocate")


class _Model(Module):
    def __init__(self, log: list[str]) -> None:
        super().__init__()
        # A stand-in is not registered by attribute assignment.
        self._parameters["weight"] = _Weight(log)

    def forward(self) -> None:
        pass


def _loaded(log: list[str]) -> _Model:
    model = _Model(log)
    model.load_torch_state_dict({"weight": object()})
    return model


class _Wrapper:
    """An exclusion that is not a `Module`, as the VAE adapters and the encoder pairs are."""

    def __init__(self, log: list[str], *, loaded: bool) -> None:
        self._log, self._loaded = log, loaded

    def is_loaded(self) -> bool:
        return self._loaded

    def deallocate_weights(self) -> None:
        self._loaded = False
        self._log.append("deallocate")


class _Pipeline(Pipeline):
    def __init__(self, *, trace: bool = False) -> None:
        super().__init__(offload_weights=False, trace=trace)

    def _run(self, *, capture: tuple[Callable, ...] = (), traced: bool | None = None) -> None:
        del traced
        for call in capture:
            call()


class _OwnRelease(_Pipeline):
    """A pipeline that keeps state beside its traces and releases the tracers it knows itself."""

    def __init__(self, *, trace: bool = False) -> None:
        super().__init__(trace=trace)
        self.own_releases = 0
        self.known: list[Tracer] = []

    def release_traces(self) -> None:
        self.own_releases += 1
        for tracer in self.known:
            tracer.release_trace()


def _tracer() -> Tracer:
    return Tracer(lambda: None, device=_Device(), prep_run=False)


def _captured() -> Tracer:
    """A tracer that captured outside any pipeline call: no pipeline owns it."""
    tracer = _tracer()
    tracer()
    return tracer


def _capture_as(pipeline: Pipeline, *tracers: Tracer) -> None:
    """Runs the tracers as a call of ``pipeline`` would. A traced pipeline that evicts cannot be called yet."""
    with pipeline._owning_captures():
        for tracer in tracers:
            tracer()


def _owned(pipeline: Pipeline, count: int = 1) -> list[Tracer]:
    """Tracers that captured while ``pipeline`` owned the captures."""
    tracers = [_tracer() for _ in range(count)]
    _capture_as(pipeline, *tracers)
    return tracers


def _excluding(pipeline: Pipeline, events: list[str], excluded: object | None = None) -> tuple[_Model, object]:
    """A module whose load evicts another, registered through ``pipeline``, which the caller keeps alive.

    A traced pipeline does not take an exclusion yet, so the trace flag is set after the registration.
    """
    loading = _Model(events)
    excluded = _loaded(events) if excluded is None else excluded
    trace, pipeline.trace = pipeline.trace, False
    pipeline.register_coresident_exclusions(loading, excluded)
    pipeline.trace = trace
    events.clear()
    return loading, excluded


def _warnings(action) -> list[str]:  # noqa: ANN001
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        action()
    finally:
        logger.remove(sink)
    return messages


def test_a_captured_tracer_is_counted_until_it_is_released() -> None:
    assert tracing.captured_trace_count() == 0
    idle = _tracer()
    assert tracing.captured_trace_count() == 0
    first, second = _captured(), _captured()
    assert tracing.captured_trace_count() == 2
    first.release_trace()
    assert tracing.captured_trace_count() == 1
    del second
    gc.collect()
    assert tracing.captured_trace_count() == 0
    assert not idle.trace_captured


def test_a_pipeline_owns_the_traces_captured_during_its_calls() -> None:
    pipeline, other = _Pipeline(trace=True), _Pipeline(trace=True)
    mine = [_tracer(), _tracer()]
    pipeline(capture=tuple(mine))
    theirs = _tracer()
    other(capture=(theirs,))
    stray = _captured()
    assert set(pipeline._owned_traces()) == set(mine)
    assert set(other._owned_traces()) == {theirs}
    assert tracing.current_capture_owner() is None
    pipeline.release_traces()
    assert not any(tracer.trace_captured for tracer in mine)
    assert theirs.trace_captured
    assert stray.trace_captured


def test_an_eager_call_owns_its_captures_too() -> None:
    pipeline = _Pipeline(trace=True)
    tracer = _tracer()
    pipeline.run_untraced(capture=(tracer,))
    assert set(pipeline._owned_traces()) == {tracer}
    assert tracing.current_capture_owner() is None


def test_a_second_pipeline_cannot_start_inside_a_running_call(expect_error: Callable) -> None:
    outer, inner = _Pipeline(trace=True), _Pipeline(trace=True)
    nested = _tracer()
    with expect_error(RuntimeError, "^_Pipeline cannot run inside another pipeline's call"):
        outer(capture=(lambda: inner(capture=(nested,)),))
    assert not nested.trace_captured
    assert tracing.current_capture_owner() is None


def test_pipelines_called_on_two_threads_do_not_meet() -> None:
    # Neither traces. Each call waits inside itself until the other has started too.
    pipelines = [_Pipeline(), _Pipeline()]
    both_running = threading.Barrier(2, timeout=30)
    owners: dict[int, object] = {}
    errors: list[BaseException] = []

    def call(index: int) -> None:
        def see_owner() -> None:
            both_running.wait()
            owners[index] = tracing.current_capture_owner()

        try:
            pipelines[index](capture=(see_owner,))
        except BaseException as error:  # noqa: BLE001 - reported by the assertion below
            errors.append(error)

    threads = [threading.Thread(target=call, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert errors == []
    # Each thread saw its own pipeline as the owner while both calls were running.
    assert owners[0] is pipelines[0]._owned_traces()
    assert owners[1] is pipelines[1]._owned_traces()
    assert tracing.current_capture_owner() is None


def test_a_pipeline_may_call_itself_from_its_own_call() -> None:
    pipeline = _Pipeline(trace=True)
    before, nested, after = _tracer(), _tracer(), _tracer()
    pipeline(capture=(before, lambda: pipeline(capture=(nested,)), after))
    assert set(pipeline._owned_traces()) == {before, nested, after}
    assert tracing.current_capture_owner() is None


def test_the_owner_is_cleared_when_a_call_raises(expect_error: Callable) -> None:
    def fail() -> None:
        msg = "the call failed"
        raise KeyError(msg)

    pipeline = _Pipeline(trace=True)
    with expect_error(KeyError, "the call failed"):
        pipeline(capture=(fail,))
    assert tracing.current_capture_owner() is None


def test_with_no_captured_trace_the_exclusion_is_evicted_quietly(events: list[str]) -> None:
    pipeline = _OwnRelease(trace=True)
    loading, excluded = _excluding(pipeline, events)
    assert _warnings(loading.evict_coresident_exclusions) == []
    assert events == ["deallocate"]
    assert not excluded.is_loaded()
    # Its own release resets state beside the traces, so it must not run when there is no trace.
    assert pipeline.own_releases == 0


def test_a_traced_pipeline_releases_its_traces_before_the_weights_are_freed(events: list[str]) -> None:
    pipeline = _Pipeline(trace=True)
    loading, excluded = _excluding(pipeline, events)
    tracers = _owned(pipeline, 2)
    gc.collect()
    events.clear()
    messages = _warnings(loading.evict_coresident_exclusions)
    assert messages == ["_Pipeline: module _Model is evicting. Releasing traces (2 released)."]
    assert events == ["release", "release", "deallocate"]
    assert not excluded.is_loaded()
    assert not any(tracer.trace_captured for tracer in tracers)


def test_an_untraced_pipeline_evicts_as_before(events: list[str]) -> None:
    pipeline = _OwnRelease(trace=False)
    loading, excluded = _excluding(pipeline, events)
    owned, stray = _owned(pipeline)[0], _captured()
    events.clear()
    assert _warnings(loading.evict_coresident_exclusions) == []
    assert events == ["deallocate"]
    assert pipeline.own_releases == 0
    assert owned.trace_captured
    assert stray.trace_captured
    assert not excluded.is_loaded()


def test_the_traces_of_another_pipeline_are_left_alone(events: list[str]) -> None:
    pipeline, other = _Pipeline(trace=True), _Pipeline(trace=True)
    loading, _ = _excluding(pipeline, events)
    mine, theirs, stray = _owned(pipeline)[0], _owned(other)[0], _captured()
    events.clear()
    messages = _warnings(loading.evict_coresident_exclusions)
    assert messages == ["_Pipeline: module _Model is evicting. Releasing traces (1 released)."]
    assert events == ["release", "deallocate"]
    assert not mine.trace_captured
    assert theirs.trace_captured
    assert stray.trace_captured


def test_a_trace_the_pipeline_does_not_own_does_not_open_the_gate(events: list[str]) -> None:
    # Traces exist in the process, but none that this pipeline captured: its own release must not run.
    pipeline, other = _OwnRelease(trace=True), _Pipeline(trace=True)
    loading, _ = _excluding(pipeline, events)
    stray, theirs = _captured(), _owned(other)[0]
    events.clear()
    assert _warnings(loading.evict_coresident_exclusions) == []
    assert events == ["deallocate"]
    assert pipeline.own_releases == 0
    assert stray.trace_captured
    assert theirs.trace_captured


def test_a_released_trace_does_not_open_the_gate(events: list[str]) -> None:
    pipeline = _OwnRelease(trace=True)
    loading, _ = _excluding(pipeline, events)
    (tracer,) = _owned(pipeline)
    tracer.release_trace()
    events.clear()
    assert _warnings(loading.evict_coresident_exclusions) == []
    assert events == ["deallocate"]
    assert pipeline.own_releases == 0


class _WarmsUp(_Pipeline):
    @owns_captures
    def warmup(self, tracer: Tracer) -> None:
        tracer()


@pytest.mark.parametrize("pipeline_class", [MiniMaxH3Pipeline, Ideogram4Pipeline])
def test_the_warm_ups_that_capture_outside_a_call_own_their_captures(pipeline_class: type[Pipeline]) -> None:
    # Every method wrapped by `owns_captures` runs the same wrapper code. Another decorator may sit around it.
    layers = [pipeline_class.warmup]
    while hasattr(layers[-1], "__wrapped__"):
        layers.append(layers[-1].__wrapped__)
    assert any(layer.__code__ is _WarmsUp.warmup.__code__ for layer in layers)
    assert layers[-1].__name__ == "warmup"


def test_a_warm_up_outside_a_call_can_own_its_captures(events: list[str]) -> None:
    pipeline = _WarmsUp(trace=True)
    loading, _ = _excluding(pipeline, events)
    tracer = _tracer()
    pipeline.warmup(tracer)
    assert set(pipeline._owned_traces()) == {tracer}
    assert tracing.current_capture_owner() is None
    events.clear()
    assert len(_warnings(loading.evict_coresident_exclusions)) == 1
    assert events == ["release", "deallocate"]


def test_the_next_traced_call_after_an_eviction_captures_again(events: list[str]) -> None:
    pipeline = _Pipeline(trace=True)
    loading, _ = _excluding(pipeline, events)
    (tracer,) = _owned(pipeline)
    assert len(_warnings(loading.evict_coresident_exclusions)) == 1
    events.clear()
    _capture_as(pipeline, tracer, tracer)
    # The first call captures and runs, the second replays: one execution each, no release.
    assert events == ["execute", "execute"]
    assert tracer.trace_captured
    assert set(pipeline._owned_traces()) == {tracer}


def test_a_recaptured_trace_watches_its_weights_again(events: list[str], expect_error: Callable) -> None:
    weight = object.__new__(Parameter)
    weight.total_shape, weight._data, weight._tracers = (2, 3), object(), weakref.WeakSet()
    pipeline = _Pipeline(trace=True)
    loading, _ = _excluding(pipeline, events)
    tracer = Tracer(lambda: weight.data and None, device=_Device(), prep_run=False)
    _capture_as(pipeline, tracer)
    assert len(weight._tracers) == 1
    loading.evict_coresident_exclusions()
    assert len(weight._tracers) == 0
    _capture_as(pipeline, tracer)
    assert len(weight._tracers) == 1
    # A free that does not go through the pipeline still makes the recaptured trace refuse to replay.
    weight.deallocate()
    with expect_error(RuntimeError, "^TDM-130: the trace of .* is stale"):
        tracer()


def test_an_exclusion_that_holds_no_weights_releases_no_trace(events: list[str]) -> None:
    pipeline = _OwnRelease(trace=True)
    loading, excluded = _excluding(pipeline, events, _Model(events))
    (tracer,) = _owned(pipeline)
    events.clear()
    assert _warnings(loading.evict_coresident_exclusions) == []
    assert "release" not in events
    assert pipeline.own_releases == 0
    assert tracer.trace_captured
    # The same for a module that an earlier eviction emptied.
    loading, excluded = _excluding(pipeline, events)
    assert len(_warnings(loading.evict_coresident_exclusions)) == 1
    assert not excluded.is_loaded()
    _capture_as(pipeline, tracer)
    events.clear()
    assert _warnings(loading.evict_coresident_exclusions) == []
    assert "release" not in events
    assert tracer.trace_captured


@pytest.mark.parametrize("loaded", [True, False])
def test_an_exclusion_that_is_not_a_module_follows_its_own_loaded_flag(events: list[str], loaded: bool) -> None:
    pipeline = _Pipeline(trace=True)
    wrapper = _Wrapper(events, loaded=loaded)
    loading, _ = _excluding(pipeline, events, wrapper)
    (tracer,) = _owned(pipeline)
    events.clear()
    messages = _warnings(loading.evict_coresident_exclusions)
    # A wrapper that says it is not loaded frees whatever it holds with no release: the trace that
    # read those weights is then stopped at its next replay, by the stale-trace check.
    assert events == (["release", "deallocate"] if loaded else ["deallocate"])
    assert len(messages) == (1 if loaded else 0)
    assert tracer.trace_captured is not loaded


def test_a_pipeline_that_keeps_state_beside_its_traces_releases_both(events: list[str]) -> None:
    pipeline = _OwnRelease(trace=True)
    loading, _ = _excluding(pipeline, events)
    (owned,) = _owned(pipeline)
    known = _captured()
    pipeline.known.append(known)
    events.clear()
    messages = _warnings(loading.evict_coresident_exclusions)
    assert messages == ["_OwnRelease: module _Model is evicting. Releasing traces (2 released)."]
    assert pipeline.own_releases == 1
    # Its own release takes the tracer it knows; the tracer captured during its call goes too.
    assert events == ["release", "release", "deallocate"]
    assert not owned.trace_captured
    assert not known.trace_captured


def test_loading_a_resident_model_evicts_through_the_pipeline(
    events: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TT_DIT_CACHE_DIR", raising=False)
    pipeline = _Pipeline(trace=True)
    loading, excluded = _loaded(events), _loaded(events)
    pipeline.trace = False
    pipeline.register_coresident_exclusions(loading, excluded)
    pipeline.trace = True
    (tracer,) = _owned(pipeline)
    events.clear()

    def load() -> None:
        cache.load_model(
            loading,
            model_name="model",
            subfolder="transformer",
            parallel_config=_NoParallelism(),
            mesh_shape=(1, 1),
            mesh_device=None,
            get_torch_state_dict=lambda: {"weight": object()},
        )

    assert _warnings(load) == ["_Pipeline: module _Model is evicting. Releasing traces (1 released)."]
    assert events == ["release", "deallocate"]
    assert loading.is_loaded()
    assert not excluded.is_loaded()
    assert not tracer.trace_captured
    # The next load of the resident model finds the exclusion empty and releases nothing.
    _capture_as(pipeline, tracer)
    events.clear()
    assert _warnings(load) == []
    assert "release" not in events
    assert tracer.trace_captured


def test_exclusions_registered_without_a_pipeline_are_freed_directly(events: list[str]) -> None:
    loading, excluded = _Model(events), _loaded(events)
    loading._register_coresident_exclusions(excluded)
    tracer = _captured()
    events.clear()
    assert _warnings(loading.evict_coresident_exclusions) == []
    assert events == ["deallocate"]
    assert tracer.trace_captured


def test_a_module_does_not_keep_its_pipeline_alive(events: list[str]) -> None:
    pipeline = _Pipeline(trace=True)
    loading, excluded = _excluding(pipeline, events)
    (tracer,) = _owned(pipeline)
    events.clear()
    gone = weakref.ref(pipeline)
    del pipeline
    gc.collect()
    assert gone() is None
    # With its pipeline gone the module frees the exclusion itself, and nobody releases the trace.
    assert _warnings(loading.evict_coresident_exclusions) == []
    assert events == ["deallocate"]
    assert not excluded.is_loaded()
    assert tracer.trace_captured
