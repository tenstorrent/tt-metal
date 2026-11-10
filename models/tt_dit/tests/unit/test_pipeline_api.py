# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The trace / offload_weights rule of the pipeline base class. Needs no device."""

from __future__ import annotations

import gc

import pytest
from loguru import logger

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


def test_call_takes_no_traced_argument() -> None:
    pipeline = _FakePipeline(offload_weights=False, trace=True)
    with pytest.raises(TypeError, match="takes no `traced` argument. For one eager run, use run_untraced"):
        pipeline(prompts=["a"], traced=False)


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
