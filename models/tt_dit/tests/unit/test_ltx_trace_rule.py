# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The LTX pipelines refuse to trace while offloading weights, before they build anything. Needs no device."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest
from loguru import logger

import ttnn

from ...encoders.gemma.encoder_pair import GemmaTokenizerEncoderPair
from ...pipelines.ltx import pipeline_ltx
from ...pipelines.ltx.pipeline_ltx import LTXPipeline
from ...pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline
from ...pipelines.ltx.pipeline_ltx_one_stage import LTXOneStagePipeline
from ...pipelines.ltx.pipeline_ltx_two_stages import LTXTwoStagesPipeline
from ...pipelines.pipeline_api import Pipeline

_RULE = "^TDM-130: trace=True is unsupported with offload_weights=True"
_PIPELINES = [LTXPipeline, LTXDistilledPipeline, LTXOneStagePipeline, LTXTwoStagesPipeline]


@pytest.mark.parametrize("pipeline_class", _PIPELINES)
def test_construction_raises_before_any_device_use(pipeline_class: type[LTXPipeline], expect_error: Callable) -> None:
    with expect_error(ValueError, _RULE):
        pipeline_class(None, None, None, offload_weights=True, trace=True)


@pytest.mark.parametrize("pipeline_class", _PIPELINES)
def test_generate_goes_through_the_base(pipeline_class: type[LTXPipeline], expect_error: Callable) -> None:
    assert pipeline_class.__call__ is Pipeline.__call__
    pipeline = object.__new__(pipeline_class)
    pipeline.offload_weights = True
    pipeline.trace = True
    with expect_error(ValueError, _RULE):
        pipeline.generate("a prompt")


@pytest.mark.parametrize("pipeline_class", _PIPELINES)
def test_has_no_per_call_eager_run(pipeline_class: type[LTXPipeline], expect_error: Callable) -> None:
    with expect_error(NotImplementedError, "no per-call eager run"):
        object.__new__(pipeline_class).run_untraced("a prompt")


class _Module:
    def __init__(self) -> None:
        self.exclusions: list[_Module] = []

    def _register_coresident_exclusions(self, *modules: _Module) -> None:
        self.exclusions.extend(modules)


def test_encoder_pair_registers_through_its_pipeline() -> None:
    calls: list[tuple[_Module, tuple[_Module, ...]]] = []
    pair = object.__new__(GemmaTokenizerEncoderPair)
    pair.offload_weights = True
    pair._register_exclusions_with = lambda module, *others: calls.append((module, others))
    encoder, dit, vae = _Module(), _Module(), _Module()
    pair.register_coresident_peers([dit, vae])
    pair._register_exclusions(encoder)
    assert calls == [(encoder, (dit, vae)), (dit, (encoder,)), (vae, (encoder,))]
    assert encoder.exclusions == []


@pytest.mark.parametrize(
    ("trace", "offload_weights", "encoder_trace"),
    [(True, False, True), (False, False, False), (False, True, False), (True, True, False)],
)
def test_encoder_pair_traces_only_with_a_traced_resident_pipeline(
    trace: bool, offload_weights: bool, encoder_trace: bool
) -> None:
    pair = GemmaTokenizerEncoderPair(
        None,
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
        trace=trace,
        offload_weights=offload_weights,
        register_exclusions=lambda module, *others: None,
    )
    assert pair._encoder_trace is encoder_trace


def _create(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mesh_shape: tuple[int, int] = (4, 8),
    blackhole: bool = True,
    level: str = "WARNING",
    **kwargs: object,
) -> tuple[dict, list[str]]:
    monkeypatch.setattr(ttnn.device, "is_blackhole", lambda: blackhole)
    monkeypatch.setattr(pipeline_ltx, "CCLManager", lambda *args, **kwargs: None)
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level=level)
    try:
        built = LTXPipeline.create_pipeline(
            SimpleNamespace(shape=mesh_shape), pipeline_class=lambda **constructor_kwargs: constructor_kwargs, **kwargs
        )
    finally:
        logger.remove(sink)
    return built, messages


_SIZE = {"num_frames": 121, "height": 512, "width": 768}


def test_create_pipeline_takes_the_deprecated_keyword_names(monkeypatch: pytest.MonkeyPatch) -> None:
    built, messages = _create(monkeypatch, dynamic_load=False, traced=True)
    assert (built["offload_weights"], built["trace"]) == (False, True)
    assert "dynamic_load" not in built
    assert "traced" not in built
    assert [("`dynamic_load` is deprecated" in m, "`traced` is deprecated" in m) for m in messages] == [
        (True, False),
        (False, True),
    ]


def test_a_deprecated_dynamic_load_overrides_the_mesh_default(monkeypatch: pytest.MonkeyPatch) -> None:
    built, _ = _create(monkeypatch, dynamic_load=True)
    assert (built["offload_weights"], built["trace"]) == (True, False)


def test_create_pipeline_defaults_and_new_names_log_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    built, messages = _create(monkeypatch)
    assert (built["offload_weights"], built["trace"]) == (False, False)
    built, more = _create(monkeypatch, offload_weights=False, trace=True)
    assert (built["offload_weights"], built["trace"]) == (False, True)
    assert messages == more == []


@pytest.mark.parametrize("kwargs", [{"traced": True, "trace": False}, {"dynamic_load": True, "offload_weights": False}])
def test_a_deprecated_name_that_contradicts_the_new_one_raises(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict, expect_error: Callable
) -> None:
    with expect_error(TypeError, "pass only"):
        _create(monkeypatch, **kwargs)


def test_deprecated_names_still_meet_the_trace_rule(expect_error: Callable) -> None:
    kwargs = {"dynamic_load": True, "traced": True}
    offload_weights = pipeline_ltx._renamed_keyword(kwargs, "dynamic_load", "offload_weights", None)  # noqa: SLF001
    trace = pipeline_ltx._renamed_keyword(kwargs, "traced", "trace", None)  # noqa: SLF001
    with expect_error(ValueError, _RULE):
        LTXPipeline(None, None, None, offload_weights=offload_weights, trace=trace)


@pytest.mark.parametrize(
    ("blackhole", "mesh_shape", "offload_weights"),
    [
        (True, (4, 8), False),
        (True, (2, 4), True),
        (False, (1, 8), False),
        (False, (2, 4), False),
        (False, (4, 8), False),
    ],
)
def test_trace_defaults_to_the_meshes_that_keep_weights_resident(
    monkeypatch: pytest.MonkeyPatch, blackhole: bool, mesh_shape: tuple[int, int], offload_weights: bool
) -> None:
    built, messages = _create(monkeypatch, mesh_shape=mesh_shape, blackhole=blackhole, **_SIZE)
    assert built["offload_weights"] is offload_weights
    assert built["trace"] is (not offload_weights)
    assert built["trace_by_default"] is True
    assert messages == []


def test_without_a_size_to_warm_up_at_the_default_is_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    for size in ({}, {**_SIZE, "num_frames": 0}, {**_SIZE, "height": 0}, {**_SIZE, "width": 0}):
        built, messages = _create(monkeypatch, level="INFO", **size)
        assert (built["offload_weights"], built["trace"], built["trace_by_default"]) == (False, False, True)
        assert ["running untraced by default" in m for m in messages] == [True]


def test_an_offloading_default_is_eager_without_a_message(monkeypatch: pytest.MonkeyPatch) -> None:
    built, messages = _create(monkeypatch, mesh_shape=(2, 4), level="INFO")
    assert (built["offload_weights"], built["trace"]) == (True, False)
    assert messages == []


@pytest.mark.parametrize(
    ("kwargs", "trace"),
    [
        ({"trace": False, **_SIZE}, False),
        ({"traced": False, **_SIZE}, False),
        ({"trace": True}, True),
        ({"traced": True}, True),
    ],
)
def test_an_explicit_trace_is_kept(monkeypatch: pytest.MonkeyPatch, kwargs: dict, trace: bool) -> None:
    built, _ = _create(monkeypatch, **kwargs)
    assert (built["trace"], built["trace_by_default"]) == (trace, False)


@pytest.mark.parametrize("name", ["offload_weights", "dynamic_load"])
def test_an_offload_override_moves_the_default(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    built, _ = _create(monkeypatch, **{name: True}, **_SIZE)
    assert (built["offload_weights"], built["trace"], built["trace_by_default"]) == (True, False, True)


@pytest.mark.parametrize("by_default", [True, False])
def test_construction_checks_the_trace_region(monkeypatch: pytest.MonkeyPatch, by_default: bool) -> None:
    class _Checked(Exception):
        pass

    def check(self: Pipeline, mesh_device: object, *, by_default: bool) -> None:
        raise _Checked(self.trace, mesh_device, by_default)

    monkeypatch.setattr(Pipeline, "warn_if_no_trace_region", check)
    mesh_device = object()
    try:
        LTXPipeline(mesh_device, None, None, trace=True, trace_by_default=by_default)
    except _Checked as checked:
        assert checked.args == (True, mesh_device, by_default)
    else:
        pytest.fail("the trace region was not checked")
