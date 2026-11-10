# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The pipelines that keep every weight resident take `trace` at construction. Needs no device."""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from ...pipelines.fibo.pipeline_fibo import FiboPipeline, FiboPipelineConfig
from ...pipelines.flux1 import pipeline_flux1
from ...pipelines.flux1.pipeline_flux1 import Flux1Pipeline, Flux1PipelineConfig
from ...pipelines.ideogram4.pipeline_ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig
from ...pipelines.mochi.pipeline_mochi import MochiPipeline, MochiPipelineConfig
from ...pipelines.motif import pipeline_motif
from ...pipelines.motif.pipeline_motif import MotifPipeline, MotifPipelineConfig
from ...pipelines.pipeline_api import Pipeline
from ...pipelines.stable_diffusion_35_large import pipeline_stable_diffusion_35_large
from ...pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
    StableDiffusion3Pipeline,
    StableDiffusion3PipelineConfig,
)

# The default of `trace` is each pipeline's former per-call default.
_PIPELINES = [
    # `None`: resolved by `default()`; see test_trace_defaults_on_where_the_pipeline_is_served_traced.
    (Flux1Pipeline, Flux1PipelineConfig, None),
    (MotifPipeline, MotifPipelineConfig, None),
    (StableDiffusion3Pipeline, StableDiffusion3PipelineConfig, None),
    (MochiPipeline, MochiPipelineConfig, False),
    (FiboPipeline, FiboPipelineConfig, True),
    (Ideogram4Pipeline, Ideogram4PipelineConfig, True),
]


@pytest.mark.parametrize(
    ("module", "config_class"),
    [
        (pipeline_flux1, Flux1PipelineConfig),
        (pipeline_motif, MotifPipelineConfig),
        (pipeline_stable_diffusion_35_large, StableDiffusion3PipelineConfig),
    ],
)
def test_trace_defaults_on_where_the_pipeline_is_served_traced(
    monkeypatch: pytest.MonkeyPatch, module: object, config_class: type
) -> None:
    monkeypatch.setattr(module, "is_blackhole", lambda: True, raising=False)
    mesh_shape = next(iter(getattr(module, "_PRESETS_BH", None) or module._PRESETS))  # noqa: SLF001
    config = config_class.default(mesh_shape=mesh_shape, checkpoint_name="checkpoint")
    assert config.trace is True
    for trace in (True, False):
        config = config_class.default(mesh_shape=mesh_shape, checkpoint_name="checkpoint", trace=trace)
        assert config.trace is trace
    assert (
        inspect.signature(module.__dict__[config_class.__name__.removesuffix("Config")].create_pipeline)
        .parameters["trace"]
        .default
        is None
    )


@pytest.mark.parametrize(
    ("pipeline_class", "config_class", "default"), [row for row in _PIPELINES if row[2] is not None]
)
def test_trace_is_a_config_field_with_the_former_default(
    pipeline_class: type[Pipeline], config_class: type, default: bool
) -> None:
    assert "trace" in {field.name for field in dataclasses.fields(config_class)}
    parameter = inspect.signature(config_class.default).parameters.get("trace")
    # A `default(trace=None)` leaves the value to the config field.
    if parameter is not None and parameter.default is not None:
        resolved = parameter.default
    else:
        resolved = config_class.__dataclass_fields__["trace"].default
    assert resolved is default


@pytest.mark.parametrize(("pipeline_class", "config_class", "default"), _PIPELINES)
def test_calls_go_through_the_base(pipeline_class: type[Pipeline], config_class: type, default: bool) -> None:
    assert issubclass(pipeline_class, Pipeline)
    assert pipeline_class.__call__ is Pipeline.__call__
    with pytest.raises(TypeError, match="takes no `traced` argument"):
        object.__new__(pipeline_class)(prompts=["a"], num_inference_steps=1, traced=True)


@pytest.mark.parametrize(("pipeline_class", "config_class", "default"), _PIPELINES)
def test_component_flags_default_to_narrowing(
    pipeline_class: type[Pipeline], config_class: type, default: bool
) -> None:
    for name, parameter in inspect.signature(pipeline_class._run).parameters.items():
        if name == "traced" or name.endswith("_traced"):
            assert parameter.default in (None, False), f"{pipeline_class.__name__}._run({name}=...)"
