# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device-free checks that a denoise re-capture leaves the audio decoder's traces and tpad masks alone."""

from types import SimpleNamespace

import pytest
import torch

from ....models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder
from ....models.audio_vae.vocoder_ltx import Vocoder
from ....pipelines.minimax_h3 import pipeline_minimax_h3 as pm


class _ReachedMetadata(Exception):
    """Raised by the fake `_device_metadata`: `_denoise` got past its trace-release block."""

    def __init__(self):
        super().__init__("reached _device_metadata")


def _audio_decoder():
    vocoder = object.__new__(Vocoder)
    object.__setattr__(vocoder, "_tpad_mask_cache", {(256, 49, "bf16"): ("M", "inv")})
    decoder = object.__new__(MiniMaxH3AudioDecoder)
    object.__setattr__(decoder, "decoder", vocoder)
    return decoder


def _fake_pipeline(*, trace_denoise, precomputed):
    transformer = SimpleNamespace(released=0)
    transformer.release_traces = lambda: setattr(transformer, "released", transformer.released + 1)
    fake = SimpleNamespace(
        arena_caps=SimpleNamespace(
            prompt=1 << 20, video_rows=1 << 20, audio_rows=1 << 20, condition_video_rows=1 << 20
        ),
        task="fl2va",
        bucket_denoise=False,
        sp_factor=4,
        adaln_slot_roles=None,
        trace_denoise=trace_denoise,
        _buckets={},
        _trace_adaln_cache=None,
        _audio_decoder=_audio_decoder(),
        _transformer=transformer,
        _log=lambda message: None,
        _precomputed_adaln=lambda: precomputed,
        _step_endpoints=lambda scheduler, audio_scheduler: None,
        _prepare_adaln=lambda slot_roles: (object(), None),
    )

    def device_metadata(layout, rung):
        raise _ReachedMetadata

    fake._device_metadata = device_metadata
    fake.release_traces = lambda: pm.MiniMaxH3Pipeline.release_traces(fake)
    fake._release_denoise_traces = lambda: pm.MiniMaxH3Pipeline._release_denoise_traces(fake)
    return fake


def _denoise_until_metadata(fake, monkeypatch, expect_error):
    monkeypatch.setattr(pm, "build_slot_routing", lambda layout, roles: (None, ()))
    layout = SimpleNamespace(
        num_condition_video_rows=0,
        num_condition_audio_rows=0,
        text_indices=torch.zeros(8),
        sequence_length=1000,
    )
    scheduler = SimpleNamespace(timesteps=torch.zeros(8))
    with expect_error(_ReachedMetadata, "reached _device_metadata"):
        pm.MiniMaxH3Pipeline._denoise(
            fake, fake._transformer, layout, None, torch.zeros(4, 1), torch.zeros(4, 1), scheduler, scheduler
        )


@pytest.mark.parametrize(
    ("trace_denoise", "precomputed"),
    [(False, True), (True, False), (True, True)],
    ids=["new-adaln-table", "cold-bucket", "both"],
)
def test_denoise_release_keeps_the_audio_tpad_masks(trace_denoise, precomputed, monkeypatch, expect_error):
    fake = _fake_pipeline(trace_denoise=trace_denoise, precomputed=precomputed)
    masks = fake._audio_decoder.decoder._tpad_mask_cache
    _denoise_until_metadata(fake, monkeypatch, expect_error)
    assert fake._transformer.released >= 1
    # `_capture_audio` traces with no prep run, so a cleared mask would be uploaded inside the capture.
    assert masks == {(256, 49, "bf16"): ("M", "inv")}


def test_release_traces_still_drops_the_audio_tpad_masks():
    fake = _fake_pipeline(trace_denoise=False, precomputed=False)
    pm.MiniMaxH3Pipeline.release_traces(fake)
    assert fake._transformer.released == 1
    assert fake._audio_decoder.decoder._tpad_mask_cache == {}
