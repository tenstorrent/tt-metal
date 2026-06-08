# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage TT-only Voxtral TTS workloads for device-perf profiling.

Splits the pipeline into the four stages so each can be profiled (and optimized) in
isolation, instead of the full e2e run (which produces an enormous profiler log):

    text_prefill     -> pipe.text.prefill_from_embeds(...)
    text_decode      -> pipe.text.decode_step_from_embeds(...)   (a few steps)
    acoustic_forward -> pipe.acoustic_codes_forward(...)         (semantic head + FM noise + FM)
    audio_decode     -> pipe.decode_waveform_from_codes_tt(...)  (tokenizer decode)

The full pipeline is built once per run (same config as the e2e perf test, so numbers
are comparable) but only the selected stage runs inside the ``start``/``stop`` tracy
signposts — so the device-perf report measures that stage alone. Stage inputs use the
real prompt/voice or correctly-shaped tensors; device-kernel durations depend on shapes,
not values. Prompt is short on purpose: per-token prefill/decode device time is identical
regardless of length, so a short prompt keeps each stage's profiler log small.
"""
from __future__ import annotations

import gc

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN
from models.experimental.voxtraltts.reference.voxtral_request import compose_speech_request
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.voxtral_tts import ACOUSTIC_CFG_ALPHA_DEFAULT, VoxtralTTSPipeline

try:
    from tracy import signpost

    _USE_SIGNPOST = True
except ModuleNotFoundError:
    _USE_SIGNPOST = False

# Short prompt: per-token prefill/decode device cost is length-independent, so this keeps
# the prefill stage's op log small while still exercising the full per-token compute.
_PERF_TEXT = "Voxtral runs on Tenstorrent."
_PERF_VOICE = "casual_male"

_STAGES = ("text_prefill", "text_decode", "acoustic_forward", "audio_decode")

_TEXT_DECODE_STEPS = 2
_ACOUSTIC_REPS = 2
_AUDIO_FRAMES = 2

# The device profiler captures timing for only a bounded number of ops (~100k here). The full
# voice-padded prompt is hundreds of tokens of per-token prefill, which alone exhausts that
# budget — so anything measured AFTER it (e.g. decode steps) gets empty durations (0.00 us).
# Cap the prefill length so the measured region stays within the profiler's capacity. Per-token
# prefill/decode device cost is essentially position-independent (matmuls fixed; only SDPA grows
# with KV length), so a modest context is representative; total prefill latency = per-token x prompt_len.
_PREFILL_TOKENS = 64  # profiled prefill length for the text_prefill stage
_DECODE_SETUP_TOKENS = 64  # KV context built (outside markers) before measuring decode steps


def _start() -> None:
    if _USE_SIGNPOST:
        signpost(header="start")


def _stop() -> None:
    if _USE_SIGNPOST:
        signpost(header="stop")


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("stage", _STAGES)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_voxtral_tts_stage_perf_run(device, reset_seeds, request, stage):
    """Profile a single pipeline stage (selected by ``stage``) between start/stop signposts."""
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

    # Model load dispatches ~68k device ops; drain the profiler buffer here so those load
    # markers don't overflow it (and drop the measured stage's markers) -> avoids the
    # "Profiler DRAM buffers were full, markers were dropped!" warnings. Mirrors the e2e
    # perf inner test (test_voxtral_tts_perf_inference.py).
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)

    dim = int(pipe.config.dim)
    n_codebooks = int(pipe.config.audio_model_args.n_acoustic_codebook) + 1  # + semantic row

    if stage == "text_prefill":
        prompt_token_ids = compose_speech_request(_PERF_TEXT, name, voice=_PERF_VOICE)["prompt_token_ids"]
        embeds = pipe._build_voice_injected_embeds(prompt_token_ids, _PERF_VOICE)
        # Cap profiled prefill length so the measured region fits the device profiler's capacity
        # (the full voice-padded prompt overflows it, leaving ops untimed). Per-token cost is
        # representative; scale by the real prompt length for total prefill latency.
        embeds = embeds[:_PREFILL_TOKENS]
        ttnn.synchronize_device(device)
        _start()
        _ = pipe.text.prefill_from_embeds(embeds, start_pos=0)
        ttnn.synchronize_device(device)
        _stop()

    elif stage == "text_decode":
        prompt_token_ids = compose_speech_request(_PERF_TEXT, name, voice=_PERF_VOICE)["prompt_token_ids"]
        embeds = pipe._build_voice_injected_embeds(prompt_token_ids, _PERF_VOICE)
        # Short setup prefill (outside markers) just to populate a representative KV cache. Using the
        # full prompt here exhausts the device profiler's op-timing budget, so the measured decode
        # steps end up with empty (0.00 us) durations.
        setup = embeds[:_DECODE_SETUP_TOKENS]
        _ = pipe.text.prefill_from_embeds(setup, start_pos=0)
        pos = int(setup.shape[0])
        mm_embed = torch.zeros(dim, dtype=torch.bfloat16)  # shape is what matters for device perf
        ttnn.synchronize_device(device)
        _start()
        for i in range(_TEXT_DECODE_STEPS):
            _ = pipe.text.decode_step_from_embeds(mm_embed, pos + i)
            ttnn.synchronize_device(device)
        _stop()

    elif stage == "acoustic_forward":
        # Current acoustic API: ``acoustic_codes_forward`` (host wrapper) takes a torch [1, dim]
        # hidden, builds the FM noise via ``fm_noise_tt`` and runs ``acoustic.forward`` on device —
        # i.e. the full per-step acoustic stage (semantic head + Euler flow matching). This replaces
        # the older ``acoustic.forward(llm_hidden, cfg_alpha)`` signature.
        llm_hidden = torch.randn(1, dim, dtype=torch.bfloat16)
        cfg_alpha = torch.tensor(ACOUSTIC_CFG_ALPHA_DEFAULT, dtype=torch.bfloat16)
        ttnn.synchronize_device(device)
        _start()
        for rep in range(_ACOUSTIC_REPS):
            _ = pipe.acoustic_codes_forward(llm_hidden, cfg_alpha, noise_seed=rep)
        ttnn.synchronize_device(device)
        _stop()

    elif stage == "audio_decode":
        codes_b37t = torch.zeros(1, n_codebooks, _AUDIO_FRAMES, dtype=torch.long)
        ttnn.synchronize_device(device)
        _start()
        wav = pipe.decode_waveform_from_codes_tt(codes_b37t)  # returns a host torch tensor
        ttnn.synchronize_device(device)
        _stop()
        assert torch.isfinite(wav).all(), "audio decode produced non-finite samples"

    else:
        pytest.fail(f"unknown stage '{stage}'")

    logger.info(f"voxtral stage perf: stage={stage} completed")

    pipe.cleanup_all()
    pipe_holder[0] = None
    del pipe
    gc.collect()
