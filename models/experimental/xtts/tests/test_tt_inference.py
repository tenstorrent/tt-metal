# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Phase-3 end-to-end test: text + reference audio -> waveform, TTNN vs torch reference.

Wires the whole XTTS-v2 model on device (conditioning -> GPT KV-cache greedy decode ->
HiFi-GAN decoder) and validates the waveform against the full torch reference with real
coqui/XTTS-v2 weights and a real reference clip.

Teacher-forced on the reference codes: greedy code sequences drift under bf16 (see
test_tt_gpt_generate), so both paths decode the *same* codes, making the waveforms
comparable. This validates the end-to-end wiring + the GPT-latents -> decoder handoff
with real latents (individual modules are validated in their own tests).

The conditioning 80-mel is computed on host (the one remaining host tensor op); the
speaker-encoder mel frontend runs on device.

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    pytest models/experimental/xtts/tests/test_tt_inference.py -s
"""

import math

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_gpt_block import load_xtts_state_dict
from models.experimental.xtts.reference.xtts_gpt_generate import STOP_TEXT_TOKEN, wrap_text_ids
from models.experimental.xtts.reference.xtts_conditioning import MEL_SR, load_reference_audio, wav_to_mel
from models.experimental.xtts.reference.xtts_mel import SAMPLE_RATE as SPK_SR
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.reference.xtts_hifi_decoder import OUTPUT_SAMPLE_RATE, XttsHifiDecoderFull
from models.experimental.xtts.reference.xtts_inference import XttsReference
from models.experimental.xtts.tt.xtts_inference import TtXtts

TILE = 32
MAX_NEW_TOKENS = 16
COND_SECONDS = 3

# A bigger, real sentence for the objective eval metrics (CER/UTMOS/SECS), which need
# natural, intelligible speech of some length to be meaningful.
EVAL_TEXT = (
    "The quick brown fox jumps over the lazy dog while the sun sets slowly over the hills. "
    "Text to speech synthesis on Tenstorrent hardware is fast, natural, and efficient."
)
# Cap on the real (sampled) generation used for eval. The fp32 HiFi-GAN vocoder decodes the
# whole code sequence in one shot; ~150 codes stays under its single-shot memory ceiling on a
# fresh device while giving Whisper / UTMOS / ECAPA2 enough speech to score. Generation usually
# self-terminates (STOP) before this cap.
EVAL_MAX_TOKENS = 150
# XTTS's natural sampling settings (self-terminates via STOP with healthy output).
EVAL_TEMPERATURE = 0.75
EVAL_TOP_K = 50
EVAL_TOP_P = 0.85  # nucleus sampling (coqui XTTS default)
EVAL_REP_PENALTY = 5.0


def _stft_mag(wav):
    """Magnitude STFT of a ``[1, 1, T]`` waveform — the perceptual comparison domain."""
    return torch.stft(wav.reshape(1, -1), 1024, 256, window=torch.hann_window(1024), return_complex=True).abs()


@pytest.fixture(scope="module")
def xtts_state_dict():
    return load_xtts_state_dict()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("pcc", [0.99])
def test_tt_inference(device, xtts_state_dict, pcc, reset_seeds):
    from scipy.signal import resample_poly

    sd = xtts_state_dict

    # Inputs: real reference audio (22.05 kHz for conditioning, resampled to 16 kHz for
    # the speaker encoder) + text.
    wav = load_reference_audio(sample="en_sample.wav", max_seconds=COND_SECONDS)  # [1, samples] @ 22050
    cond_mel = wav_to_mel(wav, sd["mel_stats"].cpu())  # host 80-mel [1, 80, s]
    g = math.gcd(SPK_SR, MEL_SR)
    spk_wav = torch.from_numpy(resample_poly(wav[0].numpy(), SPK_SR // g, MEL_SR // g).astype("float32")).unsqueeze(0)

    wrapped = wrap_text_ids(preprocess_text("hello world", lang="en"))
    pad = (-wrapped.shape[1]) % TILE
    if pad:
        wrapped = F.pad(wrapped, (0, pad), value=STOP_TEXT_TOKEN)

    # Reference end to end (also the source of the codes teacher-forced into TT).
    reference = XttsReference(sd)
    wav_ref, codes_ref, _ = reference.inference(wrapped, cond_mel, spk_wav, max_new_tokens=MAX_NEW_TOKENS)

    # TTNN end to end, same codes.
    tt = TtXtts(device, sd, reference.decoder_full)
    spk_wav_tt = ttnn.from_torch(
        spk_wav.reshape(1, -1, 1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32
    )
    wav_tt_dev, _ = tt.inference(wrapped, cond_mel, spk_wav_tt, force_codes=codes_ref[0].tolist())
    wav_tt = ttnn.to_torch(wav_tt_dev).float().permute(0, 2, 1)  # [1, T_out, 1] -> [1, 1, T_out]

    logger.info(f"codes={codes_ref.shape[1]}, ref wav {tuple(wav_ref.shape)}, tt wav {tuple(wav_tt.shape)}")
    assert wav_tt.shape == wav_ref.shape, f"waveform shape {tuple(wav_tt.shape)} != {tuple(wav_ref.shape)}"

    # A GAN vocoder maps tiny bf16-GPT latent differences to small phase/sample shifts that
    # tank sample-wise waveform correlation without changing what is heard. The perceptually
    # meaningful gate is the magnitude-spectrogram PCC; raw-waveform PCC is informational.
    wave_pcc = comp_pcc(wav_ref, wav_tt, 0.0)[1]
    spec_pass, spec_msg = comp_pcc(_stft_mag(wav_ref), _stft_mag(wav_tt), pcc)
    logger.info(comp_allclose(wav_ref, wav_tt))
    logger.info(f"end-to-end raw-waveform PCC (informational): {wave_pcc}")
    logger.info(f"end-to-end spectrogram-magnitude PCC: {spec_msg}")
    assert spec_pass, f"end-to-end spectrogram PCC below {pcc}: {spec_msg}"


# A real sampled generation + three heavy eval backends (Whisper-large-v3, UTMOS22, ECAPA2,
# downloaded on first use), so it needs well beyond the repo-wide 300s pytest timeout.
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_tt_eval(device, xtts_state_dict, reset_seeds):
    """Objective quality eval of a *real* on-device sampled generation of EVAL_TEXT.

    Runs the full text + reference-audio -> waveform pipeline with XTTS's natural sampling
    (not teacher-forced), writes the audio, and reports three standard TTS metrics computed
    on the device output:

      * CER   — Whisper-large-v3 transcript vs the input text (pronunciation/intelligibility).
      * UTMOS — UTMOS22 naturalness MOS (objective proxy for perceived quality).
      * SECS  — ECAPA2 speaker-embedding cosine similarity to the reference speaker.

    Each metric is best-effort: a missing/failing backend logs a skip rather than failing the
    test. Generation length is capped (EVAL_MAX_TOKENS) to stay under the fp32 HiFi-GAN
    vocoder's single-shot memory ceiling on this device.
    """
    import os

    import soundfile as sf
    from scipy.signal import resample_poly

    sd = xtts_state_dict

    # Inputs: real reference audio (22.05 kHz conditioning; resampled to 16 kHz for the speaker
    # encoder + SECS target) + a bigger, real text sentence.
    wav = load_reference_audio(sample="en_sample.wav", max_seconds=COND_SECONDS)  # [1, s] @ 22050
    cond_mel = wav_to_mel(wav, sd["mel_stats"].cpu())  # host 80-mel [1, 80, s]
    g = math.gcd(SPK_SR, MEL_SR)
    spk_wav = torch.from_numpy(resample_poly(wav[0].numpy(), SPK_SR // g, MEL_SR // g).astype("float32")).unsqueeze(0)

    wrapped = wrap_text_ids(preprocess_text(EVAL_TEXT, lang="en"))
    pad = (-wrapped.shape[1]) % TILE
    if pad:
        wrapped = F.pad(wrapped, (0, pad), value=STOP_TEXT_TOKEN)

    # TtXtts needs a reference HiFi decoder only to source decoder / speaker-encoder / mel
    # weights (no torch GPT here — generation is entirely on device).
    tt = TtXtts(device, sd, XttsHifiDecoderFull(sd))
    spk_wav_tt = ttnn.from_torch(
        spk_wav.reshape(1, -1, 1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32
    )

    # Real (sampled) generation of the whole sentence — self-terminates via STOP; greedy is
    # intentionally NOT used (it collapses/repeats and never stops, so it can't make a sentence).
    ttnn.manual_seed(1234, device=device)  # reproducible-ish sampled decode
    wav_eval_dev, codes_eval = tt.inference(
        wrapped,
        cond_mel,
        spk_wav_tt,
        max_new_tokens=EVAL_MAX_TOKENS,
        temperature=EVAL_TEMPERATURE,
        top_k=EVAL_TOP_K,
        repetition_penalty=EVAL_REP_PENALTY,
        top_p=EVAL_TOP_P,
    )
    wav_eval = ttnn.to_torch(wav_eval_dev).float().reshape(-1).numpy()

    out_dir = "generated/xtts"
    os.makedirs(out_dir, exist_ok=True)
    sf.write(f"{out_dir}/tt_eval_device.wav", wav_eval, OUTPUT_SAMPLE_RATE)
    logger.info(
        f"eval generation: {codes_eval.shape[1]} codes -> {wav_eval.shape[0] / OUTPUT_SAMPLE_RATE:.2f}s "
        f"audio at {out_dir}/tt_eval_device.wav"
    )

    spk_np = spk_wav[0].numpy()  # 16 kHz reference-speaker audio (SECS target)
    logger.info("================ XTTS objective eval metrics ================")
    try:
        from models.experimental.xtts.eval.xtts_eval import compute_cer

        cer, hyp = compute_cer(wav_eval, OUTPUT_SAMPLE_RATE, EVAL_TEXT)
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
    logger.info("============================================================")
