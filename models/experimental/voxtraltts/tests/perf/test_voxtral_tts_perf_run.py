# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT-only Voxtral TTS workload for device-perf profiling.

Runs ONLY ``pipe.forward_device_resident()`` (production path) — no CPU reference
``generate()`` and no PCC comparison — so the device-perf report reflects the TT
pipeline alone (prefill + AR generation + waveform decode). The forward is bracketed
by tracy ``start``/``stop`` signposts so the measured region excludes model load.

No warm-up forward: device *kernel* durations are steady-state regardless of first-run
host-side JIT compile, and a second forward only doubles the profiler op-marker volume
(report post-processing scales with op count, so a warm-up pushes report generation
past the timeout). ``generate_steps`` is small for the same reason — per-step device
time is identical across AR steps, so a few steps fully characterise the pipeline while
keeping the profiler log tractable.
"""
from __future__ import annotations

import gc

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.voxtral_tts import VoxtralTTSPipeline

try:
    from tracy import signpost

    _USE_SIGNPOST = True
except ModuleNotFoundError:
    _USE_SIGNPOST = False

_DEMO_TEXT = (
    "Voxtral is a four billion parameter open weight text to speech model "
    "released by Mistral AI in two thousand twenty six, designed for low "
    "latency multilingual voice generation across English, Spanish, French, "
    "Portuguese, Hindi, German, Dutch, and Italian. It builds on the "
    "Ministral three billion language backbone with a flow matching acoustic "
    "decoder and produces audio at twelve point five hertz with high quality, "
    "suitable for streaming voice applications and real time agent deployments."
)
_DEMO_VOICE = "casual_male"


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_voxtral_tts_device_perf_run(device, reset_seeds, request):
    """TT-only forward (no CPU ref, no PCC); ``start``/``stop`` signposts bracket the measured run."""
    # Few AR steps: per-step device time is identical, so this characterises the pipeline
    # while keeping the profiler op log small enough for report generation to finish.
    generate_steps = 2
    name = resolve_voxtral_model_name_or_skip()

    try:
        pipe = VoxtralTTSPipeline.from_model_name(
            device,
            model_name_or_path=name,
            text_max_seq_len=DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN,
        )
    except Exception as exc:
        pytest.skip(f"TT pipeline load failed: {exc}")
    pipe_holder = [pipe]

    def _cleanup_pipe() -> None:
        if pipe_holder[0] is not None:
            pipe_holder[0].cleanup_all()
            pipe_holder[0] = None

    request.addfinalizer(_cleanup_pipe)

    if _USE_SIGNPOST:
        signpost(header="start")
    tt_out = pipe.forward_device_resident(
        text=_DEMO_TEXT,
        voice=_DEMO_VOICE,
        max_tokens=generate_steps,
        seed=0,
    )
    ttnn.synchronize_device(device)
    if _USE_SIGNPOST:
        signpost(header="stop")

    assert torch.isfinite(tt_out.waveform).all(), "TT forward produced non-finite waveform samples"
    logger.info(
        f"TT-only device-perf run: codes shape={tuple(tt_out.codes_b37t.shape)} "
        f"waveform samples={int(tt_out.waveform.numel())} hit_end_audio={tt_out.hit_end_audio}"
    )

    pipe.cleanup_all()
    pipe_holder[0] = None
    del pipe
    del tt_out
    gc.collect()
