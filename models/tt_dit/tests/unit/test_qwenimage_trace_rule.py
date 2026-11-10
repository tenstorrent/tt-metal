# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image traces exactly when it offloads no weights. Needs no device."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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


def test_an_explicit_trace_is_kept_for_the_pipeline_to_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(monkeypatch, blackhole=False, checkpoint=_EDIT, trace=True)
    assert (config.offloads_weights, config.trace) == (True, True)
    with pytest.raises(ValueError, match=_RULE):
        QwenImagePipeline(device=None, config=config)


def test_construction_raises_before_any_device_use() -> None:
    with pytest.raises(ValueError, match=_RULE):
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


def test_registering_what_the_text_encoder_offloads() -> None:
    pipeline = object.__new__(QwenImagePipeline)
    pipeline.trace = True
    pipeline.register_evictable(*_encoder(offload=False, offload_vision=True, vision=None).offloaded_modules)
    assert pipeline.offload_weights is False

    class _Vision:
        pass

    vision = _Vision()
    with pytest.raises(ValueError, match=_RULE):
        pipeline.register_evictable(*_encoder(offload=False, offload_vision=True, vision=vision).offloaded_modules)


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
