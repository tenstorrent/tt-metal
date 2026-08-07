# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end performance test for XTTS-v2 on Blackhole P150.

Times ``TtXtts.inference_fully_traced()`` -- the same three-chained-trace (setup + decode +
vocoder) path ``demo/xtts_demo.py`` uses -- on the demo's default text + reference voice, and
asserts each replay leg plus the overall RTF stay within a generous margin of a baseline
measured on this device (see README.md#performance):

    189 codes -> 8.888 s audio | setup 0.013 s | decode 1.535 s (~8.12 ms/code) |
    vocoder 0.022 s (~0.116 ms/code) | total replay 1.569 s | RTF 0.177

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
from models.experimental.xtts.reference.xtts_conditioning import MEL_SR, load_reference_audio
from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_gpt_generate import STOP_TEXT_TOKEN, wrap_text_ids
from models.experimental.xtts.reference.xtts_hifi_decoder import OUTPUT_SAMPLE_RATE, XttsHifiDecoderFull
from models.experimental.xtts.reference.xtts_mel import SAMPLE_RATE as SPK_SR
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.tt.xtts_inference import TtXtts
from models.perf.perf_utils import prep_perf_report

TILE = 32
COND_SECONDS = 3

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
EXPECTED_SETUP_S = 0.013
EXPECTED_DECODE_MS_PER_CODE = 8.12
EXPECTED_VOCODER_MS_PER_CODE = 0.116
EXPECTED_RTF = 0.177
EXPECTED_COMPILE_S = 44.0  # one-time trace capture + JIT; not gated, reported for the record only


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536, "trace_region_size": 52428800}], indirect=True)
def test_xtts_e2e_perf(device, reset_seeds):
    from scipy.signal import resample_poly

    sd = load_xtts_state_dict()
    tt = TtXtts(device, sd, XttsHifiDecoderFull(sd))

    wav = load_reference_audio(sample="en_sample.wav", max_seconds=COND_SECONDS)  # [1, s] @ 22050
    g = math.gcd(SPK_SR, MEL_SR)
    spk_wav = torch.from_numpy(resample_poly(wav[0].numpy(), SPK_SR // g, MEL_SR // g).astype("float32")).unsqueeze(0)
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
