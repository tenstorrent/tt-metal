# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The Wan pipelines refuse to trace while offloading weights, before touching a device."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import ttnn

from ...experimental.pipelines.pipeline_wan_svi import WanPipelineSVI
from ...pipelines.pipeline_api import Pipeline
from ...pipelines.wan import pipeline_wan
from ...pipelines.wan.pipeline_wan import WanPipeline, WanPipelineConfig
from ...pipelines.wan.pipeline_wan_i2v import WanPipelineI2V

_RULE = "^TDM-130: trace=True is unsupported with offload_weights=True"


@pytest.mark.parametrize("pipeline_class", [WanPipeline, WanPipelineI2V])
def test_construction_raises_before_any_device_use(pipeline_class: type[WanPipeline]) -> None:
    config = SimpleNamespace(offload_weights=True, trace=True)
    with pytest.raises(ValueError, match=_RULE):
        pipeline_class(device=None, config=config)


@pytest.mark.parametrize("pipeline_class", [WanPipeline, WanPipelineI2V, WanPipelineSVI])
def test_call_raises_on_an_offloading_traced_pipeline(pipeline_class: type[WanPipeline]) -> None:
    pipeline = object.__new__(pipeline_class)
    pipeline.offload_weights = True
    pipeline.trace = True
    with pytest.raises(ValueError, match=_RULE):
        pipeline(prompts=["a"], num_inference_steps=1)


@pytest.mark.parametrize("pipeline_class", [WanPipeline, WanPipelineI2V, WanPipelineSVI])
def test_calls_go_through_the_base(pipeline_class: type[WanPipeline]) -> None:
    assert pipeline_class.__call__ is Pipeline.__call__
    with pytest.raises(TypeError, match="takes no `traced` argument"):
        object.__new__(pipeline_class)(prompts=["a"], num_inference_steps=1, traced=True)


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


def test_an_explicit_trace_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ttnn.device, "is_blackhole", lambda: True)
    resident = WanPipelineConfig.default(mesh_shape=(4, 8), checkpoint_name="checkpoint", trace=False)
    assert resident.trace is False
    offloading = WanPipelineConfig.default(mesh_shape=(2, 4), checkpoint_name="checkpoint", trace=True)
    assert (offloading.offload_weights, offloading.trace) == (True, True)


def test_offload_override_moves_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ttnn.device, "is_blackhole", lambda: True)
    config = WanPipelineConfig.default(mesh_shape=(4, 8), checkpoint_name="checkpoint", offload_weights=True)
    assert (config.offload_weights, config.trace) == (True, False)
