# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 refuses to trace while it evicts stages, before it builds anything. Needs no device."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from loguru import logger

import ttnn

from ...pipelines.minimax_h3 import pipeline_minimax_h3 as module
from ...pipelines.minimax_h3.pipeline_minimax_h3 import _DYNAMIC_LENGTH_ENV, _PRESETS_BH, _PRESETS_WH, MiniMaxH3Pipeline
from ...pipelines.minimax_h3.pipeline_minimax_h3_turbo import MiniMaxH3TurboPipeline
from ...pipelines.pipeline_api import Pipeline

_RULE = "^TDM-130: trace=True is unsupported with offload_weights=True"


class _Resolved(Exception):
    """Stops construction once the base has the resolved values."""


def _construct(monkeypatch: pytest.MonkeyPatch, preset: dict, **kwargs: object) -> None:
    monkeypatch.delenv(_DYNAMIC_LENGTH_ENV, raising=False)
    monkeypatch.setattr(module, "resolve_mesh_preset", lambda shape, required=True: preset)
    MiniMaxH3Pipeline(
        mesh_device=SimpleNamespace(shape=(4, 8)), weights_dir="unused", topology=ttnn.Topology.Ring, **kwargs
    )


def _resolved(monkeypatch: pytest.MonkeyPatch, preset: dict, **kwargs: object) -> tuple[bool, bool]:
    def init(self: Pipeline, *, offload_weights: bool, trace: bool) -> None:
        raise _Resolved(offload_weights, trace)

    monkeypatch.setattr(Pipeline, "__init__", init)
    with pytest.raises(_Resolved) as resolved:
        _construct(monkeypatch, preset, **kwargs)
    return resolved.value.args


def test_explicit_audio_trace_on_the_evicting_preset_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match=_RULE):
        _construct(monkeypatch, _PRESETS_WH[(4, 8)], trace_audio=True, dynamic_length=False)


def test_explicit_audio_trace_with_explicit_eviction_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match=_RULE):
        _construct(monkeypatch, _PRESETS_BH[(4, 32)], coresident=False, trace_audio=True)


@pytest.mark.parametrize(
    ("preset", "kwargs"), [(_PRESETS_WH[(4, 8)], {}), (_PRESETS_BH[(4, 32)], {"coresident": False})]
)
def test_explicit_denoise_trace_is_dropped_loudly_by_an_evicting_pipeline(
    monkeypatch: pytest.MonkeyPatch, preset: dict, kwargs: dict
) -> None:
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        resolved = _resolved(monkeypatch, preset, trace_denoise=True, **kwargs)
    finally:
        logger.remove(sink)
    assert resolved == (True, False)
    assert [m for m in messages if "trace_denoise=True IGNORED" in m and "(TDM-130)" in m]


def test_a_resident_pipeline_keeps_an_explicit_denoise_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        resolved = _resolved(monkeypatch, _PRESETS_BH[(4, 32)], trace_denoise=True)
    finally:
        logger.remove(sink)
    assert resolved == (False, True)
    assert not [m for m in messages if "IGNORED" in m]


def test_presets_resolve_without_a_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    # (offload_weights, trace)
    assert _resolved(monkeypatch, _PRESETS_WH[(4, 8)]) == (True, False)
    assert _resolved(monkeypatch, _PRESETS_BH[(4, 32)]) == (False, True)
    assert _resolved(monkeypatch, _PRESETS_BH[(4, 8)]) == (False, False)


def test_preset_traces_yield_to_explicit_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[str] = []
    monkeypatch.setattr(MiniMaxH3Pipeline, "_host_log", lambda self, message: logged.append(message))
    assert _resolved(monkeypatch, _PRESETS_BH[(4, 32)], coresident=False) == (True, False)
    assert "coresident=False: the preset's trace_denoise and trace_audio yield to evicting stages" in logged


def test_no_yield_is_logged_when_nothing_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[str] = []
    monkeypatch.setattr(MiniMaxH3Pipeline, "_host_log", lambda self, message: logged.append(message))
    _resolved(monkeypatch, _PRESETS_WH[(4, 8)])
    _resolved(monkeypatch, _PRESETS_BH[(4, 32)])
    assert not [message for message in logged if "yield" in message]


@pytest.mark.parametrize("pipeline_class", [MiniMaxH3Pipeline, MiniMaxH3TurboPipeline])
def test_calls_go_through_the_base(pipeline_class: type[MiniMaxH3Pipeline]) -> None:
    assert pipeline_class.__call__ is Pipeline.__call__


@pytest.mark.parametrize("pipeline_class", [MiniMaxH3Pipeline, MiniMaxH3TurboPipeline])
def test_has_no_per_call_eager_run(pipeline_class: type[MiniMaxH3Pipeline]) -> None:
    with pytest.raises(NotImplementedError, match="no per-call eager run"):
        object.__new__(pipeline_class).run_untraced("a prompt")
