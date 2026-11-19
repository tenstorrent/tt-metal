# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The Wan pipelines trace by default exactly where they keep their weights resident. Needs no device."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import pytest

import ttnn

from ...experimental.pipelines.pipeline_wan_svi import WanPipelineSVI
from ...pipelines.pipeline_api import Pipeline
from ...pipelines.wan import pipeline_wan
from ...pipelines.wan.pipeline_wan import WanPipeline, WanPipelineConfig
from ...pipelines.wan.pipeline_wan_i2v import WanPipelineI2V


class _Resolved(Exception):
    """Stops construction once the base has the resolved values."""


def _handed_to_base(monkeypatch: pytest.MonkeyPatch, construct: Callable[[], object]) -> tuple[bool, bool]:
    def init(self: Pipeline, *, offload_weights: bool, trace: bool) -> None:
        raise _Resolved(offload_weights, trace)

    monkeypatch.setattr(Pipeline, "__init__", init)
    try:
        construct()
    except _Resolved as resolved:
        return resolved.args
    msg = "construction did not reach the base"
    raise AssertionError(msg)


@pytest.mark.parametrize("pipeline_class", [WanPipeline, WanPipelineI2V])
def test_construction_hands_the_config_flags_to_the_base(
    pipeline_class: type[WanPipeline], monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(offload_weights=True, trace=True)
    assert _handed_to_base(monkeypatch, lambda: pipeline_class(device=None, config=config)) == (True, True)


@pytest.mark.parametrize("pipeline_class", [WanPipeline, WanPipelineI2V, WanPipelineSVI])
def test_a_call_reaches_run_on_an_offloading_traced_pipeline(
    pipeline_class: type[WanPipeline], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline_class, "_run", lambda self, **kwargs: kwargs)
    pipeline = object.__new__(pipeline_class)
    pipeline.offload_weights = True
    pipeline.trace = True
    assert pipeline(prompts=["a"], num_inference_steps=1) == {"prompts": ["a"], "num_inference_steps": 1}


@pytest.mark.parametrize("pipeline_class", [WanPipeline, WanPipelineI2V, WanPipelineSVI])
def test_calls_go_through_the_base(pipeline_class: type[WanPipeline]) -> None:
    assert pipeline_class.__call__ is Pipeline.__call__


@pytest.mark.parametrize("blackhole", [False, True])
def test_trace_defaults_to_the_meshes_that_keep_weights_resident(
    monkeypatch: pytest.MonkeyPatch, blackhole: bool
) -> None:
    monkeypatch.setattr(ttnn.device, "is_blackhole", lambda: blackhole)
    presets = pipeline_wan._PRESETS_BH if blackhole else pipeline_wan._PRESETS_WH  # noqa: SLF001
    assert {preset["offload_weights"] for preset in presets.values()} == {True, False}
    for mesh_shape, preset in presets.items():
        config = WanPipelineConfig.default(mesh_shape=mesh_shape, checkpoint_name="checkpoint")
        assert config.trace is (not preset["offload_weights"]), mesh_shape
        assert config.trace_by_default is True


def test_an_explicit_trace_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ttnn.device, "is_blackhole", lambda: True)
    resident = WanPipelineConfig.default(mesh_shape=(4, 8), checkpoint_name="checkpoint", trace=False)
    assert (resident.trace, resident.trace_by_default) == (False, False)
    offloading = WanPipelineConfig.default(mesh_shape=(2, 4), checkpoint_name="checkpoint", trace=True)
    assert (offloading.offload_weights, offloading.trace, offloading.trace_by_default) == (True, True, False)


def test_offload_override_moves_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ttnn.device, "is_blackhole", lambda: True)
    config = WanPipelineConfig.default(mesh_shape=(4, 8), checkpoint_name="checkpoint", offload_weights=True)
    assert (config.offload_weights, config.trace) == (True, False)
