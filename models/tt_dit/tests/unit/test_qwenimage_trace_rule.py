# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image traces exactly when it offloads no weights. Needs no device."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch

import ttnn

from ...pipelines.pipeline_api import Pipeline
from ...pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline, QwenImagePipelineConfig
from ...pipelines.qwenimage.text_encoder import TextEncoder

_RULE = "^TDM-130: trace=True is unsupported with offload_weights=True"
_T2I, _EDIT = "Qwen/Qwen-Image", "Qwen/Qwen-Image-Edit"


def _config(monkeypatch: pytest.MonkeyPatch, *, blackhole: bool, checkpoint: str, **kwargs: object):
    monkeypatch.setattr(ttnn.device, "is_blackhole", lambda: blackhole)
    return QwenImagePipelineConfig.default(mesh_shape=(2, 4), checkpoint_name=checkpoint, **kwargs)


def test_text_to_image_on_the_wormhole_2x4_preset_stays_traced(monkeypatch: pytest.MonkeyPatch) -> None:
    # The preset offloads the vision tower, which a text-to-image checkpoint does not have.
    config = _config(monkeypatch, blackhole=False, checkpoint=_T2I)
    assert config.offload_vision_tower
    assert (config.offloads_weights, config.trace) == (False, True)


def test_edit_on_the_wormhole_2x4_preset_runs_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(monkeypatch, blackhole=False, checkpoint=_EDIT)
    assert (config.offloads_weights, config.trace) == (True, False)


def test_edit_on_blackhole_stays_traced(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(monkeypatch, blackhole=True, checkpoint=_EDIT)
    assert (config.offloads_weights, config.trace) == (False, True)


def test_offloading_the_text_encoder_runs_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(monkeypatch, blackhole=True, checkpoint=_T2I, offload_text_encoder=True)
    assert (config.offloads_weights, config.trace) == (True, False)
    config = _config(
        monkeypatch, blackhole=True, checkpoint=_T2I, offload_text_encoder=True, use_torch_text_encoder=True
    )
    assert (config.offloads_weights, config.trace) == (False, True)


def test_an_explicit_trace_is_kept_for_the_pipeline_to_refuse(
    monkeypatch: pytest.MonkeyPatch, expect_error: Callable
) -> None:
    config = _config(monkeypatch, blackhole=False, checkpoint=_EDIT, trace=True)
    assert (config.offloads_weights, config.trace) == (True, True)
    with expect_error(ValueError, _RULE):
        QwenImagePipeline(device=None, config=config)


def test_construction_raises_before_any_device_use(expect_error: Callable) -> None:
    with expect_error(ValueError, _RULE):
        QwenImagePipeline(device=None, config=SimpleNamespace(offloads_weights=True, trace=True))


def test_calls_go_through_the_base() -> None:
    assert QwenImagePipeline.__call__ is Pipeline.__call__


def _encoder(*, offload: bool, offload_vision: bool, vision: object | None) -> TextEncoder:
    encoder = object.__new__(TextEncoder)
    encoder._offload, encoder._offload_vision = offload, offload_vision
    encoder._encoder, encoder._vision = "encoder", vision
    return encoder


def test_text_encoder_names_the_modules_it_offloads() -> None:
    assert _encoder(offload=False, offload_vision=False, vision="vision").offloaded_modules == ()
    assert _encoder(offload=False, offload_vision=True, vision="vision").offloaded_modules == ("vision",)
    assert _encoder(offload=True, offload_vision=False, vision=None).offloaded_modules == ("encoder", None)


class _Module:
    def __init__(self) -> None:
        self.exclusions: list[object] = []

    def _register_coresident_exclusions(self, *modules: object) -> None:
        self.exclusions.extend(modules)


def _pipeline(*, trace: bool, text_encoder: TextEncoder, transformers: int = 2) -> QwenImagePipeline:
    pipeline = object.__new__(QwenImagePipeline)
    pipeline.trace = trace
    pipeline._text_encoder = text_encoder
    pipeline._transformers = [_Module() for _ in range(transformers)]
    return pipeline


def test_registering_what_the_text_encoder_offloads(expect_error: Callable) -> None:
    # A tower the checkpoint does not have is not offloaded, so the pipeline may trace.
    pipeline = _pipeline(trace=True, text_encoder=_encoder(offload=False, offload_vision=True, vision=None))
    pipeline._register_text_encoder_exclusions()
    assert pipeline.offload_weights is False

    pipeline = _pipeline(trace=True, text_encoder=_encoder(offload=False, offload_vision=True, vision=_Module()))
    with expect_error(ValueError, _RULE):
        pipeline._register_text_encoder_exclusions()
    assert [t.exclusions for t in pipeline._transformers] == [[], []]
    assert pipeline.offload_weights is False


def test_each_transformer_excludes_what_the_text_encoder_offloads() -> None:
    encoder_module, vision = _Module(), _Module()
    text_encoder = _encoder(offload=True, offload_vision=False, vision=vision)
    text_encoder._encoder = encoder_module
    pipeline = _pipeline(trace=False, text_encoder=text_encoder)
    pipeline._register_text_encoder_exclusions()
    assert [t.exclusions for t in pipeline._transformers] == [[encoder_module, vision]] * 2
    # One way: the encoder reloads itself beside the resident transformers.
    assert encoder_module.exclusions == vision.exclusions == []
    assert pipeline.offload_weights is True


def test_a_resident_text_encoder_registers_nothing() -> None:
    pipeline = _pipeline(trace=True, text_encoder=_encoder(offload=False, offload_vision=False, vision=_Module()))
    pipeline._register_text_encoder_exclusions()
    assert [t.exclusions for t in pipeline._transformers] == [[], []]
    assert pipeline.offload_weights is False


def test_each_transformer_is_loaded_through_the_checkpoint_with_its_own_manager() -> None:
    calls: list[tuple] = []

    class _Checkpoint:
        def load_weights(self, model: object, **kwargs: object) -> None:
            calls.append((model, kwargs))

    pipeline = _pipeline(trace=False, text_encoder=_encoder(offload=True, offload_vision=False, vision=None))
    pipeline._checkpoint, pipeline._ccl_managers, pipeline._parallel_config = _Checkpoint(), ["m0", "m1"], "parallel"
    pipeline._ensure_transformers_loaded()
    assert calls == [
        (transformer, {"ccl_manager": manager, "parallel_config": "parallel", "is_fsdp": False})
        for transformer, manager in zip(pipeline._transformers, ["m0", "m1"])
    ]


def _encoding(monkeypatch: pytest.MonkeyPatch, *, offload: bool) -> tuple[TextEncoder, list[str]]:
    """A text encoder whose device work is replaced by a record of the weight calls."""
    events: list[str] = []
    encoder = _encoder(offload=offload, offload_vision=False, vision=None)
    encoder._processor = None
    monkeypatch.setattr(TextEncoder, "_reshape", lambda self: contextlib.nullcontext())
    monkeypatch.setattr(TextEncoder, "_load_weights", lambda self: events.append("load"))
    monkeypatch.setattr(TextEncoder, "_deallocate_weights", lambda self: events.append("deallocate"))
    monkeypatch.setattr(TextEncoder, "_encode_prompt", lambda self, prompt, *, traced: torch.zeros(1, 4, 8))
    return encoder, events


@pytest.mark.parametrize(("offload", "expected"), [(True, ["load"]), (False, [])])
def test_encode_cfg_loads_an_offloaded_encoder_and_leaves_it_loaded(
    monkeypatch: pytest.MonkeyPatch, offload: bool, expected: list[str]
) -> None:
    encoder, events = _encoding(monkeypatch, offload=offload)
    outputs = encoder.encode_cfg(
        ["a"], [""], num_images_per_prompt=1, cfg_enabled=True, batch_passes=False, traced=False
    )
    assert [tuple(o.shape) for o in outputs] == [(1, 4, 8)] * 2
    assert events == expected


def test_encode_cfg_refuses_to_trace_while_offloaded(monkeypatch: pytest.MonkeyPatch, expect_error: Callable) -> None:
    encoder, events = _encoding(monkeypatch, offload=True)
    with expect_error(ValueError, "^the text encoder cannot trace while its weights are offloaded"):
        encoder.encode_cfg(["a"], [""], num_images_per_prompt=1, cfg_enabled=True, batch_passes=False, traced=True)
    assert events == []


@pytest.mark.parametrize(("trace", "expected"), [(None, False), (False, False), (True, True)])
def test_create_pipeline_passes_trace_and_none_keeps_the_resolution(
    monkeypatch: pytest.MonkeyPatch, trace: bool | None, expected: bool
) -> None:
    built: list[QwenImagePipelineConfig] = []
    monkeypatch.setattr(ttnn.device, "is_blackhole", lambda: False)
    monkeypatch.setattr(QwenImagePipeline, "__init__", lambda self, *, device, config: built.append(config))
    # Edit on the Wormhole 2x4 offloads the vision tower, so an unset trace resolves to off.
    QwenImagePipeline.create_pipeline(mesh_device=SimpleNamespace(shape=(2, 4)), checkpoint_name=_EDIT, trace=trace)
    assert built[0].trace is expected
