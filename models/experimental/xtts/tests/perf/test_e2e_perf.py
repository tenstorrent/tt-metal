# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end performance test for XTTS-v2 on Blackhole P150.

Times ``TtXtts.inference_fully_traced()`` -- the same three-chained-trace (setup + decode +
vocoder) path ``demo/xtts_demo.py`` uses -- on the demo's default text AND its default reference
voice (30 s of audio = 8 conditioning windows; see REF_CLIPS), and asserts each replay leg plus
the overall RTF stay within a generous margin of a baseline measured on this device
(see README.md#performance):

    205 codes -> 9.515 s audio | setup 0.042 s | decode 1.652 s (~8.06 ms/code) |
    vocoder 0.021 s (~0.102 ms/code) | total replay 1.714 s | RTF 0.180

Decode/vocoder are gated on a per-code RATE (ms/code), not absolute time, because the number of
codes generated is itself sampled (temperature 0.65) and therefore not perfectly fixed run to
run; ``reset_seeds`` makes it *close* to fixed, but a rate-based gate is robust either way. Setup
and RTF are close to length-invariant, so they're gated directly.

The margin is intentionally generous (40%) because this is XTTS's first perf baseline -- there is
no run-to-run history yet to know normal hardware variance, and this repo has no CI wiring for
this model (see README.md#ci) to progressively tighten it later.

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    pytest models/experimental/xtts/tests/perf/test_e2e_perf.py -v -s
"""

import math

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import profiler
from models.experimental.xtts.reference.xtts_conditioning import GPT_COND_LEN_SEC, MEL_SR, load_coqui_test_audio
from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_gpt_generate import STOP_TEXT_TOKEN, wrap_text_ids
from models.experimental.xtts.reference.xtts_hifi_decoder import OUTPUT_SAMPLE_RATE, XttsHifiDecoderFull
from models.experimental.xtts.reference.xtts_mel import SAMPLE_RATE as SPK_SR
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.tt.xtts_inference import TtXtts
from models.perf.perf_utils import prep_perf_report

TILE = 32

# Same reference voice as demo/xtts_demo.py's --ref-audio default: four single-speaker coqui-ai/TTS
# LJSpeech clips joined to 32.6 s, clipped to gpt_cond_len (30 s) = 8 conditioning windows.
# This USED to be a 3 s single-window clip, which left most of the conditioning encoder out of the
# setup gate -- setup replay scales with the window count (measured: 1 window 12.8 ms, 3 windows
# 21 ms, 8 windows 42 ms), so a 3 s reference gated under a third of what the demo actually runs.
REF_CLIPS = ("LJ001-0001.wav", "LJ001-0003.wav", "LJ001-0004.wav", "LJ001-0005.wav")
COND_SECONDS = GPT_COND_LEN_SEC  # conditioning window (coqui gpt_cond_len), as the demo
SPK_SECONDS = 8  # speaker-embedding window, as the demo -- 30 s clashes L1 in the speaker ResNet

# Same default text (after the demo's own trailing-punctuation strip) and sampling recipe as
# demo/xtts_demo.py -- the scenario this baseline was measured on.
DEMO_TEXT = (
    "Voice synthesis has come a long way, and modern systems can already generate natural sounding "
    "speech with remarkable accuracy. Hey how are you doing"
)
MAX_NEW_TOKENS = 240  # demo default
TEMPERATURE, TOP_K, TOP_P, REP_PENALTY = 0.65, 50, 0.85, 5.0

# Baseline measured on Blackhole P150 (see README.md#performance), with a generous 40% margin --
# see the module docstring for why. Retune these as real run-to-run history accumulates.
MARGIN = 0.40
EXPECTED_SETUP_S = 0.042  # 8 conditioning windows (see REF_CLIPS); was 0.013 on a 3 s 1-window clip
EXPECTED_DECODE_MS_PER_CODE = 8.12
EXPECTED_VOCODER_MS_PER_CODE = 0.116
EXPECTED_RTF = 0.180  # includes the 8-window setup leg; was 0.177 on a 3 s 1-window clip
EXPECTED_COMPILE_S = 44.0  # one-time trace capture + JIT; not gated, reported for the record only


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536, "trace_region_size": 52428800}], indirect=True)
def test_xtts_e2e_perf(device, reset_seeds):
    from scipy.signal import resample_poly

    sd = load_xtts_state_dict()
    tt = TtXtts(device, sd, XttsHifiDecoderFull(sd))

    wav = load_coqui_test_audio(samples=REF_CLIPS, max_seconds=COND_SECONDS)  # [1, s] @ 22050
    g = math.gcd(SPK_SR, MEL_SR)
    # The speaker path is capped independently at SPK_SECONDS, exactly as the demo does: the speaker
    # encoder does not fit a 30 s mel (L1 clash in its ResNet), and a time-pooled speaker embedding
    # saturates well before that. The GPT conditioning above uses the FULL 30 s.
    spk_src = wav[0].numpy()[: MEL_SR * SPK_SECONDS]
    spk_wav = torch.from_numpy(resample_poly(spk_src, SPK_SR // g, MEL_SR // g).astype("float32")).unsqueeze(0)
    spk_wav_tt = ttnn.from_torch(
        spk_wav.reshape(1, -1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32
    )

    wrapped = wrap_text_ids(preprocess_text(DEMO_TEXT, lang="en"))
    pad = (-wrapped.shape[1]) % TILE
    if pad:
        wrapped = F.pad(wrapped, (0, pad), value=STOP_TEXT_TOKEN)

    prompt_len = 32 + wrapped.shape[1]  # 32 conditioning latents + wrapped text tokens
    max_seq = -(-(prompt_len + MAX_NEW_TOKENS + 2) // TILE) * TILE

    profiler.start("inference_and_compile_time")
    wav_dev, codes, perf = tt.inference_fully_traced(
        wrapped,
        wav,  # raw reference wav; 80-mel computed on device inside the setup trace
        spk_wav_tt,
        max_seq,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REP_PENALTY,
    )
    profiler.end("inference_and_compile_time")

    n_codes = codes.shape[1]
    assert n_codes > 0, "generated no codes -- nothing to measure"
    audio_s = ttnn.to_torch(wav_dev).float().numel() / OUTPUT_SAMPLE_RATE
    rtf = perf["replay_s"] / audio_s
    decode_ms_per_code = perf["decode_replay_s"] / n_codes * 1000
    vocoder_ms_per_code = perf["vocoder_replay_s"] / n_codes * 1000

    logger.info(
        f"codes={n_codes} audio={audio_s:.3f}s | setup={perf['setup_replay_s'] * 1000:.2f}ms "
        f"decode={decode_ms_per_code:.3f}ms/code vocoder={vocoder_ms_per_code:.4f}ms/code "
        f"replay={perf['replay_s']:.3f}s RTF={rtf:.3f} compile={perf['compile_s']:.1f}s"
    )

    prep_perf_report(
        model_name="xtts_v2",
        batch_size=1,
        inference_and_compile_time=perf["compile_s"] + perf["replay_s"],
        inference_time=perf["replay_s"],
        expected_compile_time=EXPECTED_COMPILE_S,
        expected_inference_time=EXPECTED_RTF * audio_s,
        comments="fully_traced",
    )

    def _bound(expected):
        return expected * (1 + MARGIN)

    assert perf["setup_replay_s"] <= _bound(EXPECTED_SETUP_S), (
        f"setup replay {perf['setup_replay_s'] * 1000:.2f} ms exceeds "
        f"{_bound(EXPECTED_SETUP_S) * 1000:.2f} ms ({int(MARGIN * 100)}% over the "
        f"{EXPECTED_SETUP_S * 1000:.2f} ms baseline)"
    )
    assert decode_ms_per_code <= _bound(EXPECTED_DECODE_MS_PER_CODE), (
        f"decode {decode_ms_per_code:.3f} ms/code exceeds {_bound(EXPECTED_DECODE_MS_PER_CODE):.3f} ms/code "
        f"({int(MARGIN * 100)}% over the {EXPECTED_DECODE_MS_PER_CODE:.3f} ms/code baseline)"
    )
    assert vocoder_ms_per_code <= _bound(EXPECTED_VOCODER_MS_PER_CODE), (
        f"vocoder {vocoder_ms_per_code:.4f} ms/code exceeds {_bound(EXPECTED_VOCODER_MS_PER_CODE):.4f} ms/code "
        f"({int(MARGIN * 100)}% over the {EXPECTED_VOCODER_MS_PER_CODE:.4f} ms/code baseline)"
    )
    assert rtf <= _bound(EXPECTED_RTF), (
        f"RTF {rtf:.3f} exceeds {_bound(EXPECTED_RTF):.3f} ({int(MARGIN * 100)}% over the "
        f"{EXPECTED_RTF:.3f} baseline) -- replay {perf['replay_s']:.3f}s for {audio_s:.3f}s of audio"
    )
