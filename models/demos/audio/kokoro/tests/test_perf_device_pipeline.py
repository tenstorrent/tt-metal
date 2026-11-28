# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end on-device TTS latency + real-time factor for Kokoro-82M (P150).

Measures ``KokoroDevicePipeline.synthesize_device()`` — the fully-on-device path
(TT plbert + prosody predictor + text encoder + ISTFTNet vocoder). Reports latency
and real-time factor (audio seconds produced per wall-clock second) via
``prep_perf_report``.

    pytest -m models_performance_bare_metal \
        models/demos/audio/kokoro/tests/test_perf_device_pipeline.py
"""
import sys
import time
import types

sys.modules.setdefault("spacy", types.ModuleType("spacy"))  # avoid misaki.en -> spaCy

import pytest
import torch
from loguru import logger

from models.demos.audio.kokoro.tt.device_pipeline import KokoroDevicePipeline
from models.perf.perf_utils import prep_perf_report

MODEL_ID = "hexgrad/Kokoro-82M"
VOICE = "af_heart"
SAMPLE_RATE = 24000
# A representative single-shot clip (~2.3 s of audio). Long text is chunked to the
# 510-token plbert context in serving; single-chip convs don't size to very long seqs.
TEXT = "Kokoro runs on Tenstorrent."
EXPECTED_LATENCY_S = 1.2  # measured ~0.88 s / 2.38 s clip (~2.7x real-time) on P150, + margin


class _Utt:
    def __init__(self, text):
        self.text = text


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.models_performance_bare_metal
def test_perf_synthesize_device(device):
    from huggingface_hub import hf_hub_download
    from kokoro.model import KModel
    from misaki.espeak import EspeakFallback

    km = KModel(repo_id=MODEL_ID).eval()
    g2p = EspeakFallback(british=False)
    phonemes, _ = g2p(_Utt(TEXT))
    ids = [i for i in (km.vocab.get(p) for p in phonemes) if i is not None]
    input_ids = torch.LongTensor([[0, *ids, 0]])
    pack = torch.load(hf_hub_download(MODEL_ID, f"voices/{VOICE}.pt"), weights_only=True)
    ref_s = pack[len(ids) - 1]

    pipe = KokoroDevicePipeline(km, device)

    t0 = time.time()
    audio = pipe.synthesize_device(input_ids, ref_s, 1.0)
    inference_and_compile_time = time.time() - t0
    audio_s = audio.numel() / SAMPLE_RATE

    iters = 3
    t0 = time.time()
    for _ in range(iters):
        pipe.synthesize_device(input_ids, ref_s, 1.0)
    inference_time = (time.time() - t0) / iters
    rtf = audio_s / inference_time

    logger.info(
        f"on-device TTS: audio {audio_s:.2f}s, latency {inference_time:.2f}s, "
        f"RTF {rtf:.2f}x (compile+first {inference_and_compile_time:.1f}s)"
    )
    prep_perf_report(
        model_name="kokoro82m_tts_ondevice",
        batch_size=1,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=120.0,
        expected_inference_time=EXPECTED_LATENCY_S,
        comments="synthesize_device",
    )
