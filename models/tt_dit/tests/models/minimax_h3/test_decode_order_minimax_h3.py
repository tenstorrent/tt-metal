# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device-free check on the decode order after the denoise: a coresident pipeline decodes the audio
first and hands it to `on_audio` before the video VAE starts; elsewhere the video goes first."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ....pipelines.events import null_callback
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline


def _pipeline(coresident: bool, calls: list):
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    pipeline.coresident = coresident
    pipeline.dynamic_length = False
    pipeline.vae_residency_mode = "keep"
    pipeline.vae_output_type = "yuv420"
    pipeline.audio_config = {"decoder_rates": [8, 100]}
    pipeline.last_program_cache_misses = {}
    pipeline.mesh_device = SimpleNamespace(num_program_cache_entries=lambda: 0)
    pipeline._vae = SimpleNamespace(modules=[])
    pipeline._log = lambda message: None
    pipeline._prepare_transformer = lambda: "transformer"
    pipeline._denoise = lambda transformer, layout, embeds, video, audio, *args, **kwargs: (video, audio)
    pipeline._prepare_audio_decoder = lambda: "audio decoder"

    def decode_video(vae, rows, *args, on_frames=None):
        calls.append("video")
        return np.zeros((9, 6, 4), dtype=np.uint8)

    def decode_audio(decoder, rows, *args):
        calls.append("audio")
        return torch.zeros(1, 2, 16)

    pipeline._decode_video = decode_video
    pipeline._decode_audio = decode_audio
    return pipeline


def _run(pipeline, on_audio):
    return pipeline._denoise_and_decode(
        layout=SimpleNamespace(num_condition_video_rows=0, num_condition_audio_rows=0),
        prompt_embeds=None,
        video_rows=torch.zeros(1),
        audio_rows=torch.zeros(1),
        scheduler=None,
        audio_scheduler=None,
        num_inference_steps=1,
        num_latent_frames=1,
        latent_height=1,
        latent_width=1,
        num_audio_latents=1,
        on_event=null_callback,
        on_audio=on_audio,
    )


@pytest.mark.parametrize(
    "coresident, order", [(True, ["audio", "on_audio", "video"]), (False, ["video", "audio", "on_audio"])]
)
def test_the_audio_decodes_before_the_video_only_when_coresident(coresident, order):
    calls = []
    handed = []

    def on_audio(audio, rate):
        calls.append("on_audio")
        handed.append((audio, rate))

    out = _run(_pipeline(coresident, calls), on_audio)
    assert calls == order
    ((audio, rate),) = handed
    assert audio is out.audio and rate == out.sampling_rate and out.num_frames == 9


def test_an_on_audio_error_fails_the_request_before_the_video_decode():
    calls = []

    def on_audio(audio, rate):
        raise RuntimeError("encoder gone")

    with pytest.raises(RuntimeError, match="encoder gone"):
        _run(_pipeline(True, calls), on_audio)
    assert calls == ["audio"]
