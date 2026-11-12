# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The trace / offload_weights rule of the pipeline base class. Needs no device."""

from __future__ import annotations

import gc

import pytest
from loguru import logger

from ...pipelines import pipeline_api
from ...pipelines.pipeline_api import Pipeline

_RULE = "^TDM-130: trace=True is unsupported with offload_weights=True"


class _FakeModule:
    def __init__(self) -> None:
        self.exclusions: list[_FakeModule] = []

    def _register_coresident_exclusions(self, *modules: _FakeModule) -> None:
        self.exclusions.extend(modules)


class _FakePipeline(Pipeline):
    def __init__(self, *, offload_weights: bool, trace: bool) -> None:
        super().__init__(offload_weights=offload_weights, trace=trace)
        self.calls: list[dict] = []

    def _run(self, *, prompts: list[str], traced: bool | None = None, **kwargs: object) -> str:
        self.calls.append({"prompts": prompts, "traced": self.trace if traced is None else traced, **kwargs})
        return "output"


def test_conflict_raises_at_construction() -> None:
    with pytest.raises(ValueError, match=_RULE):
        _FakePipeline(offload_weights=True, trace=True)


@pytest.mark.parametrize(("offload_weights", "trace"), [(False, False), (False, True), (True, False)])
def test_other_combinations_construct_and_run(offload_weights: bool, trace: bool) -> None:
    pipeline = _FakePipeline(offload_weights=offload_weights, trace=trace)
    assert pipeline.offload_weights is offload_weights
    assert pipeline(prompts=["a"]) == "output"
    assert pipeline.calls == [{"prompts": ["a"], "traced": trace}]


def test_conflict_raises_at_call_without_init() -> None:
    pipeline = object.__new__(_FakePipeline)
    assert pipeline.offload_weights is False
    pipeline.offload_weights = True
    pipeline.trace = True
    with pytest.raises(ValueError, match=_RULE):
        pipeline(prompts=["a"])


def test_conflict_raises_at_call_after_attributes_change() -> None:
    pipeline = _FakePipeline(offload_weights=False, trace=True)
    pipeline.offload_weights = True
    with pytest.raises(ValueError, match=_RULE):
        pipeline(prompts=["a"])
    assert pipeline.calls == []


def test_registering_exclusions_makes_the_pipeline_offload() -> None:
    pipeline = _FakePipeline(offload_weights=False, trace=False)
    loaded, evicted = _FakeModule(), _FakeModule()
    pipeline.register_coresident_exclusions(loaded, evicted, None)
    assert loaded.exclusions == [evicted]
    assert pipeline.offload_weights is True


def test_registering_on_a_traced_pipeline_raises() -> None:
    pipeline = _FakePipeline(offload_weights=False, trace=True)
    with pytest.raises(ValueError, match=_RULE):
        pipeline.register_evictable(_FakeModule())


def test_refused_exclusion_is_not_registered_on_the_module() -> None:
    pipeline = _FakePipeline(offload_weights=False, trace=True)
    loaded, evicted = _FakeModule(), _FakeModule()
    with pytest.raises(ValueError, match=_RULE):
        pipeline.register_coresident_exclusions(loaded, evicted)
    assert loaded.exclusions == []
    assert pipeline.offload_weights is False


def test_offload_stays_set_while_a_module_is_registered() -> None:
    pipeline = _FakePipeline(offload_weights=True, trace=False)
    module = _FakeModule()
    pipeline.register_evictable(module)
    pipeline.offload_weights = False
    assert pipeline.offload_weights is True


def test_registering_nothing_keeps_the_pipeline_resident() -> None:
    pipeline = _FakePipeline(offload_weights=False, trace=True)
    pipeline.register_evictable(None)
    pipeline.register_coresident_exclusions(_FakeModule(), None)
    assert pipeline.offload_weights is False
    assert pipeline(prompts=["a"]) == "output"


def test_evictable_modules_are_held_weakly() -> None:
    pipeline = _FakePipeline(offload_weights=False, trace=False)
    module = _FakeModule()
    pipeline.register_evictable(module)
    del module
    gc.collect()
    assert pipeline.offload_weights is False


def _warnings_of(action) -> list[str]:  # noqa: ANN001
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        action()
    finally:
        logger.remove(sink)
    return messages


@pytest.mark.parametrize(
    ("offload_weights", "trace", "traced", "expected"),
    [
        (False, True, True, "per-call `traced` argument is deprecated and ignored"),
        (False, False, False, "per-call `traced` argument is deprecated and ignored"),
        (True, False, None, "per-call `traced` argument is deprecated and ignored"),
        (False, True, False, "traced=False IGNORED: the pipeline was constructed with trace=True"),
        (False, False, True, "traced=True IGNORED: the pipeline was constructed with trace=False and runs eager."),
        (True, False, True, "runs eager because it offloads weights (TDM-130)"),
    ],
)
def test_a_per_call_traced_never_changes_what_runs(
    offload_weights: bool, trace: bool, traced: bool | None, expected: str
) -> None:
    pipeline = _FakePipeline(offload_weights=offload_weights, trace=trace)
    messages = _warnings_of(lambda: pipeline(prompts=["a"], traced=traced))
    assert pipeline.calls == [{"prompts": ["a"], "traced": trace}]
    assert len(messages) == 1
    assert expected in messages[0]


def test_a_disagreeing_traced_names_construction_before_run_untraced() -> None:
    pipeline = _FakePipeline(offload_weights=False, trace=True)
    (message,) = _warnings_of(lambda: pipeline(prompts=["a"], traced=False))
    assert "trace buffers stay in use (in DRAM when the mesh has no trace region)" in message
    assert message.index("construct the pipeline with trace=False") < message.index("run_untraced()")
    assert "only before the first traced call" in message


def test_a_per_call_traced_is_reported_once_per_kind() -> None:
    pipeline = _FakePipeline(offload_weights=False, trace=True)

    def calls() -> None:
        for traced in (True, True, False, False, True):
            pipeline(prompts=["a"], traced=traced)

    messages = _warnings_of(calls)
    assert ["IGNORED" in m for m in messages] == [False, True]
    assert [call["traced"] for call in pipeline.calls] == [True] * 5
    other = _FakePipeline(offload_weights=False, trace=True)
    assert len(_warnings_of(lambda: other(prompts=["a"], traced=True))) == 1


def test_offloading_in_the_warning_follows_registration() -> None:
    pipeline = _FakePipeline(offload_weights=False, trace=False)
    module = _FakeModule()
    pipeline.register_evictable(module)
    (message,) = _warnings_of(lambda: pipeline(prompts=["a"], traced=True))
    assert "because it offloads weights (TDM-130)" in message


def test_a_call_without_traced_logs_no_warning() -> None:
    pipeline = _FakePipeline(offload_weights=False, trace=True)
    assert _warnings_of(lambda: pipeline(prompts=["a"])) == []


class _View:
    def __init__(self, total_bytes_per_bank: int) -> None:
        self.total_bytes_per_bank = total_bytes_per_bank
        self.num_banks = 8


@pytest.mark.parametrize(
    ("trace", "region", "by_default", "expected"),
    [
        (True, 0, True, "tracing is on by default on this mesh and no trace region is reserved"),
        (True, 0, False, "trace=True with no trace region reserved; trace buffers are taken from DRAM."),
        (True, 1 << 20, True, None),
        (False, 0, True, None),
    ],
)
def test_trace_region_warning(
    monkeypatch: pytest.MonkeyPatch, trace: bool, region: int, by_default: bool, expected: str | None
) -> None:
    class _Mesh:
        pass

    monkeypatch.setattr(pipeline_api.ttnn, "MeshDevice", _Mesh)
    monkeypatch.setattr(pipeline_api.ttnn, "get_memory_view", lambda device, buffer_type: _View(region))
    pipeline = _FakePipeline(offload_weights=False, trace=trace)
    messages = _warnings_of(lambda: pipeline.warn_if_no_trace_region(_Mesh(), by_default=by_default))
    assert len(messages) == (0 if expected is None else 1)
    if expected is not None:
        assert expected in messages[0]
    if by_default and expected is not None:
        assert "Pass trace=False to run eager" in messages[0]


@pytest.mark.parametrize("device", [None, object()])
def test_trace_region_warning_skips_what_is_not_a_mesh_device(device: object) -> None:
    pipeline = _FakePipeline(offload_weights=False, trace=True)
    assert _warnings_of(lambda: pipeline.warn_if_no_trace_region(device, by_default=True)) == []


def test_run_untraced_is_eager_on_a_traced_pipeline() -> None:
    pipeline = _FakePipeline(offload_weights=False, trace=True)
    pipeline.run_untraced(prompts=["warmup"])
    pipeline.run_single_prompt(prompt="a", num_inference_steps=2)
    assert [call["traced"] for call in pipeline.calls] == [False, True]
    assert pipeline.calls[1]["prompts"] == ["a"]
    assert pipeline.calls[1]["negative_prompts"] is None


def test_run_untraced_on_an_offloading_pipeline() -> None:
    pipeline = _FakePipeline(offload_weights=True, trace=False)
    assert pipeline.run_untraced(prompts=["a"]) == "output"
    assert pipeline.calls == [{"prompts": ["a"], "traced": False}]


def test_run_untraced_cannot_be_asked_to_trace() -> None:
    pipeline = _FakePipeline(offload_weights=False, trace=True)
    with pytest.raises(TypeError, match="traced"):
        pipeline.run_untraced(prompts=["a"], traced=True)
    assert pipeline.calls == []


@pytest.mark.parametrize("entry", ["__call__", "run_untraced"])
def test_component_flags_cannot_widen_trace(entry: str) -> None:
    pipeline = _FakePipeline(offload_weights=True, trace=False)
    with pytest.raises(ValueError, match="encoder_traced, vae_traced cannot be set"):
        getattr(pipeline, entry)(prompts=["a"], vae_traced=True, encoder_traced=True)
    assert pipeline.calls == []
    getattr(pipeline, entry)(prompts=["a"], vae_traced=False, encoder_traced=None)
    assert len(pipeline.calls) == 1


def test_component_flags_narrow_a_traced_pipeline() -> None:
    pipeline = _FakePipeline(offload_weights=False, trace=True)
    pipeline(prompts=["a"], vae_traced=True, encoder_traced=False)
    assert pipeline.calls == [{"prompts": ["a"], "traced": True, "vae_traced": True, "encoder_traced": False}]


def test_subclass_cannot_define_call() -> None:
    with pytest.raises(TypeError, match="must implement _run, not __call__"):

        class _Bypass(Pipeline):
            def __call__(self) -> None:
                pass


@pytest.mark.parametrize(
    ("offload_weights", "trace", "logged"), [(False, False, True), (False, True, False), (True, False, False)]
)
def test_resident_untraced_pipeline_says_so_once(offload_weights: bool, trace: bool, logged: bool) -> None:
    messages: list[str] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        pipeline = _FakePipeline(offload_weights=offload_weights, trace=trace)
        pipeline(prompts=["a"])
    finally:
        logger.remove(sink)
    expected = ["_FakePipeline: static mesh, running untraced (offload_weights=False, trace=False)\n"]
    assert [m for m in messages if "running untraced" in m] == (expected if logged else [])


@pytest.mark.parametrize("flag", ["traced", "vae_traced"])
def test_subclass_cannot_default_a_trace_flag_to_on(flag: str) -> None:
    namespace: dict = {}
    exec(f"def _run(self, *, {flag}=True): pass", namespace)  # noqa: S102
    with pytest.raises(TypeError, match=f"must not default {flag} to True"):
        type("_Widening", (Pipeline,), {"_run": namespace["_run"]})
    exec(f"def _run(self, *, {flag}=None): pass", namespace)  # noqa: S102
    type("_Narrowing", (Pipeline,), {"_run": namespace["_run"]})
