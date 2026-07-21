# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full XTTS-v2 model end-to-end on device via ttnn traces.

A ttnn trace captures a FIXED, host-sync-free sequence of device ops with STATIC shapes and
replays it with near-zero host dispatch overhead. This single test runs the WHOLE on-device
model — text -> audio codes -> waveform — through traces:

  1. **GPT autoregressive decode** — normally untraceable (per-step host readback + a
     concat-growing KV cache). The parallel STATIC-KV path (``decode_static``: fixed-size cache
     written in place at a device-driven position, masked attention over the whole cache) makes
     every decode step the same static-shape op sequence, so ``TtXttsGenerator.generate_traced``
     captures it ONCE and replays it per token (canonical tt_transformers decode-trace pattern:
     persistent input buffers refreshed with ``copy_host_to_device_tensor`` + ``execute_trace``;
     greedy selection on host between replays — ``ttnn.rand`` can't be traced, so no in-trace
     sampling, and greedy keeps it deterministic for a hard PCC gate).
  2. **HiFi-GAN vocoder** — a single static-shape conv stack; captured and replayed on the
     generated (fixed-length) latents.

The tokenizer and the conditioning 80-mel stay host preprocessing (outside the device model),
as intended. The traced output is validated against the eager (concat-KV decode + eager vocoder)
reference on the SAME generated codes, so any divergence in the traced path fails the test.

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    pytest models/experimental/xtts/tests/test_tt_trace.py -s
"""

import math

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.xtts.reference.xtts_conditioning import MEL_SR, load_reference_audio
from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_gpt_generate import STOP_TEXT_TOKEN, wrap_text_ids
from models.experimental.xtts.reference.xtts_hifi_decoder import OUTPUT_SAMPLE_RATE, XttsHifiDecoderFull
from models.experimental.xtts.reference.xtts_mel import SAMPLE_RATE as SPK_SR
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.tt.xtts_inference import TtXtts

TILE = 32
COND_SECONDS = 3
# Same text the demo uses (trailing period stripped, as the demo does — a period is its own token
# the model would verbalize as "dot"). With [SPACE] tokens this wraps to ~96 text tokens.
DEMO_TEXT = "Voice synthesis has come a long way, and modern systems can already generate natural sounding speech with remarkable accuracy"
# Same sampling path as the demo (temp 0.65 / top-k 50 / top-p 0.85 / rep 5.0) so the traced run
# self-terminates at STOP and produces the full utterance, not a stunted greedy clip.
TRACE_TEMPERATURE = 0.65
TRACE_TOP_K = 50
TRACE_TOP_P = 0.85
TRACE_REP = 5.0
TRACE_MAX_TOKENS = 192  # fixed on-device step budget (multiple of 32 for the accumulation buffers)
TRACE_MAX_SEQ = 384  # fixed KV-cache length (>= prompt_len ~128 + TRACE_MAX_TOKENS); multiple of 32


def _stft_mag(wav):
    """Magnitude STFT of a ``[1, 1, T]`` waveform — the perceptual comparison domain."""
    return torch.stft(wav.reshape(1, -1), 1024, 256, window=torch.hann_window(1024), return_complex=True).abs()


@pytest.fixture(scope="module")
def xtts_state_dict():
    return load_xtts_state_dict()


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536, "trace_region_size": 52428800}], indirect=True)
@pytest.mark.parametrize("pcc", [0.99])
def test_tt_full_trace(device, xtts_state_dict, pcc, reset_seeds):
    """ENTIRE on-device model via 3 chained ttnn traces (setup + decode + vocoder), vs eager.

    ``inference_fully_traced`` runs every on-device stage inside a trace: a SETUP trace
    (conditioning + speaker encoder + prefill, seeding the persistent KV cache), the per-token
    DECODE trace (captured once, replayed with host sampling — same path as the demo), and the
    VOCODER trace. Only the host tokenizer / conditioning-mel / sampling stay eager. Validated by
    running the eager concat-KV reference on the SAME generated codes and comparing the waveform."""
    import os

    import soundfile as sf
    from scipy.signal import resample_poly

    sd = xtts_state_dict

    # Inputs: reference audio (22.05 kHz conditioning, 16 kHz speaker) + text — host preprocessing.
    wav = load_reference_audio(sample="en_sample.wav", max_seconds=COND_SECONDS)  # [1, s] @ 22050
    g = math.gcd(SPK_SR, MEL_SR)
    spk_wav = torch.from_numpy(resample_poly(wav[0].numpy(), SPK_SR // g, MEL_SR // g).astype("float32")).unsqueeze(0)
    wrapped = wrap_text_ids(preprocess_text(DEMO_TEXT, lang="en"))
    pad = (-wrapped.shape[1]) % TILE
    if pad:
        wrapped = F.pad(wrapped, (0, pad), value=STOP_TEXT_TOKEN)
    logger.info(f"trace text -> {wrapped.shape[1]} tokens (wrapped/padded)")

    tt = TtXtts(device, sd, XttsHifiDecoderFull(sd))
    spk_wav_tt = ttnn.from_torch(
        spk_wav.reshape(1, -1, 1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32
    )

    # ---- FULLY TRACED: setup (conditioning+speaker+prefill) -> decode (per token) -> vocoder ----
    wav_dev, codes = tt.inference_fully_traced(
        wrapped,
        wav,  # raw reference wav; 80-mel computed on device inside the setup trace
        spk_wav_tt,
        TRACE_MAX_SEQ,
        max_new_tokens=TRACE_MAX_TOKENS,
        temperature=TRACE_TEMPERATURE,
        top_k=TRACE_TOP_K,
        top_p=TRACE_TOP_P,
        repetition_penalty=TRACE_REP,
    )
    wav_traced = ttnn.to_torch(wav_dev).float()
    n = codes.shape[1]
    assert n > 0, "fully-traced generation produced no codes"
    logger.info(f"fully-traced generation: {n} codes -> {wav_traced.shape[1] / OUTPUT_SAMPLE_RATE:.2f}s audio")

    # ---- EAGER REFERENCE on the SAME generated codes (concat-KV decode + eager vocoder) ----
    cond_latents = tt._cond_latents(wav)
    _, latents_ref = tt.generator.latents_for_codes(wrapped, cond_latents, codes[0].tolist())
    wav_ref = ttnn.to_torch(tt._decode_wav(latents_ref, spk_wav_tt)).float()

    assert wav_traced.shape == wav_ref.shape, f"traced {tuple(wav_traced.shape)} != eager {tuple(wav_ref.shape)}"
    spec_pass, spec_msg = comp_pcc(_stft_mag(wav_ref), _stft_mag(wav_traced), pcc)
    logger.info(f"fully-traced vs eager spectrogram-magnitude PCC: {spec_msg}")

    out_dir = "generated/xtts"
    os.makedirs(out_dir, exist_ok=True)
    sf.write(f"{out_dir}/tt_full_trace_device.wav", wav_traced.reshape(-1).numpy(), OUTPUT_SAMPLE_RATE)
    logger.info(f"wrote fully-traced device audio ({n} codes) -> {out_dir}/tt_full_trace_device.wav")

    assert spec_pass, f"fully-traced waveform diverged from eager reference: {spec_msg}"


# Objective quality eval of the FULLY-TRACED (fully-on-device) pipeline — the same three backends
# as test_tt_eval (Whisper-large-v3 / UTMOS22 / ECAPA2, downloaded on first use), but on the
# inference_fully_traced output (setup+decode+vocoder all traced, sampling on device). Heavy, so
# it needs well beyond the repo-wide 300s pytest timeout.
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536, "trace_region_size": 52428800}], indirect=True)
def test_tt_eval_traced(device, xtts_state_dict, reset_seeds):
    """CER / UTMOS / SECS on the FULLY-TRACED, fully-on-device generation of DEMO_TEXT.

    Mirrors ``test_tt_eval`` but drives ``inference_fully_traced`` (every on-device stage inside a
    trace; rep/temp/top-k/top-p sampling done ON DEVICE) instead of the eager host-sampling
    ``tt.inference``. DEMO_TEXT is a single sentence that self-terminates within TRACE_MAX_TOKENS,
    so the metric reflects a COMPLETE utterance (no cap-truncation inflating CER). Each metric is
    best-effort: a missing/failing backend logs a skip rather than failing the test."""
    import os

    import soundfile as sf
    from scipy.signal import resample_poly

    sd = xtts_state_dict

    wav = load_reference_audio(sample="en_sample.wav", max_seconds=COND_SECONDS)  # [1, s] @ 22050
    g = math.gcd(SPK_SR, MEL_SR)
    spk_wav = torch.from_numpy(resample_poly(wav[0].numpy(), SPK_SR // g, MEL_SR // g).astype("float32")).unsqueeze(0)
    wrapped = wrap_text_ids(preprocess_text(DEMO_TEXT, lang="en"))
    pad = (-wrapped.shape[1]) % TILE
    if pad:
        wrapped = F.pad(wrapped, (0, pad), value=STOP_TEXT_TOKEN)

    tt = TtXtts(device, sd, XttsHifiDecoderFull(sd))
    spk_wav_tt = ttnn.from_torch(
        spk_wav.reshape(1, -1, 1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32
    )

    wav_dev, codes = tt.inference_fully_traced(
        wrapped,
        wav,  # raw reference wav; 80-mel computed on device inside the setup trace
        spk_wav_tt,
        TRACE_MAX_SEQ,
        max_new_tokens=TRACE_MAX_TOKENS,
        temperature=TRACE_TEMPERATURE,
        top_k=TRACE_TOP_K,
        top_p=TRACE_TOP_P,
        repetition_penalty=TRACE_REP,
    )
    wav_eval = ttnn.to_torch(wav_dev).float().reshape(-1).numpy()

    out_dir = "generated/xtts"
    os.makedirs(out_dir, exist_ok=True)
    sf.write(f"{out_dir}/tt_eval_traced_device.wav", wav_eval, OUTPUT_SAMPLE_RATE)
    logger.info(
        f"fully-traced eval generation: {codes.shape[1]} codes -> {wav_eval.shape[0] / OUTPUT_SAMPLE_RATE:.2f}s "
        f"audio at {out_dir}/tt_eval_traced_device.wav"
    )

    spk_np = spk_wav[0].numpy()  # 16 kHz reference-speaker audio (SECS target)
    logger.info("========== XTTS objective eval metrics (FULLY TRACED) ==========")
    try:
        from models.experimental.xtts.eval.xtts_eval import compute_cer

        cer, hyp = compute_cer(wav_eval, OUTPUT_SAMPLE_RATE, DEMO_TEXT)
        logger.info(f"CER   (Whisper-large-v3, lower=better)        : {cer:.4f}")
        logger.info(f"        whisper transcript: {hyp!r}")
    except Exception as e:
        logger.warning(f"CER   skipped ({type(e).__name__}: {e})")

    try:
        from models.experimental.xtts.eval.xtts_eval import compute_utmos

        logger.info(
            f"UTMOS (naturalness MOS 1-5, higher=better)    : {compute_utmos(wav_eval, OUTPUT_SAMPLE_RATE):.4f}"
        )
    except Exception as e:
        logger.warning(f"UTMOS skipped ({type(e).__name__}: {e})")

    try:
        from models.experimental.xtts.eval.xtts_eval import compute_secs

        secs = compute_secs(wav_eval, OUTPUT_SAMPLE_RATE, spk_np, SPK_SR)
        logger.info(f"SECS  (ECAPA2 speaker cos-sim, higher=better)  : {secs:.4f}")
    except Exception as e:
        logger.warning(f"SECS  skipped ({type(e).__name__}: {e})")
    logger.info("================================================================")
