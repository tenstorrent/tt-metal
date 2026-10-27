# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.reference.xtts_gpt_generate import STOP_TEXT_TOKEN, wrap_text_ids
from models.experimental.xtts.reference.xtts_conditioning import (
    MEL_SR,
    load_reference_audio,
    wav_to_mel,
)
from models.experimental.xtts.reference.xtts_mel import SAMPLE_RATE as SPK_SR
from models.experimental.xtts.reference.xtts_text_embedding import preprocess_text
from models.experimental.xtts.reference.xtts_hifi_decoder import OUTPUT_SAMPLE_RATE, XttsHifiDecoderFull
from models.experimental.xtts.reference.xtts_inference import XttsReference
from models.experimental.xtts.tt.xtts_inference import TtXtts

TILE = 32
MAX_NEW_TOKENS = 16
COND_SECONDS = 3

EVAL_TEXT = (
    "The quick brown fox jumps over the lazy dog while the sun sets slowly over the hills. "
    "Text to speech synthesis on Tenstorrent hardware is fast, natural, and efficient."
)
EVAL_MAX_TOKENS = 150
EVAL_TEMPERATURE = 0.75
EVAL_TOP_K = 50
EVAL_TOP_P = 0.85
EVAL_REP_PENALTY = 5.0


def _stft_mag(wav):
    """Return STFT magnitude spectrogram of a waveform for PCC comparison."""
    return torch.stft(wav.reshape(1, -1), 1024, 256, window=torch.hann_window(1024), return_complex=True).abs()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
# Spectrogram gate: GAN vocoder turns tiny bf16 latent diffs into phase shifts that wreck waveform PCC.
# Sits at ~0.9958; the two stages that set it are GPT matmul fidelity (MM_FIDELITY in xtts_gpt_block)
# and vocoder conv fidelity (_CONV_FIDELITY in xtts_hifigan) — dropping either to LoFi/HiFi2 fails this.
@pytest.mark.parametrize("pcc", [0.99])
def test_tt_inference(device, xtts_state_dict, pcc, reset_seeds):
    """Compare end-to-end TTNN inference spectrogram to the PyTorch reference via PCC."""
    from scipy.signal import resample_poly

    sd = xtts_state_dict

    wav = load_reference_audio(sample="en_sample.wav", max_seconds=COND_SECONDS)
    cond_mel = wav_to_mel(wav, sd["mel_stats"].cpu())
    g = math.gcd(SPK_SR, MEL_SR)
    spk_wav = torch.from_numpy(resample_poly(wav[0].numpy(), SPK_SR // g, MEL_SR // g).astype("float32")).unsqueeze(0)

    wrapped = wrap_text_ids(preprocess_text("hello world", lang="en"))
    pad = (-wrapped.shape[1]) % TILE
    if pad:
        wrapped = F.pad(wrapped, (0, pad), value=STOP_TEXT_TOKEN)

    reference = XttsReference(sd)
    wav_ref, codes_ref = reference.inference(wrapped, cond_mel, spk_wav, max_new_tokens=MAX_NEW_TOKENS)

    tt = TtXtts(device, sd, reference.decoder_full)
    spk_wav_tt = ttnn.from_torch(
        spk_wav.reshape(1, -1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32
    )
    wav_tt_dev, _ = tt.inference(wrapped, wav, spk_wav_tt, force_codes=codes_ref[0].tolist())
    wav_tt = ttnn.to_torch(wav_tt_dev).float().permute(0, 2, 1)

    logger.info(f"codes={codes_ref.shape[1]}, ref wav {tuple(wav_ref.shape)}, tt wav {tuple(wav_tt.shape)}")
    assert wav_tt.shape == wav_ref.shape, f"waveform shape {tuple(wav_tt.shape)} != {tuple(wav_ref.shape)}"

    wave_pcc = comp_pcc(wav_ref, wav_tt, 0.0)[1]
    spec_pass, spec_msg = comp_pcc(_stft_mag(wav_ref), _stft_mag(wav_tt), pcc)
    logger.info(comp_allclose(wav_ref, wav_tt))
    logger.info(f"end-to-end raw-waveform PCC (informational): {wave_pcc}")
    logger.info(f"end-to-end spectrogram-magnitude PCC: {spec_msg}")
    assert spec_pass, f"end-to-end spectrogram PCC below {pcc}: {spec_msg}"


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_tt_eval(device, xtts_state_dict, reset_seeds):
    """Run sampled TTNN inference and log CER/UTMOS/SECS eval metrics."""
    import os

    import soundfile as sf
    from scipy.signal import resample_poly

    sd = xtts_state_dict

    wav = load_reference_audio(sample="en_sample.wav", max_seconds=COND_SECONDS)
    g = math.gcd(SPK_SR, MEL_SR)
    spk_wav = torch.from_numpy(resample_poly(wav[0].numpy(), SPK_SR // g, MEL_SR // g).astype("float32")).unsqueeze(0)

    wrapped = wrap_text_ids(preprocess_text(EVAL_TEXT, lang="en"))
    pad = (-wrapped.shape[1]) % TILE
    if pad:
        wrapped = F.pad(wrapped, (0, pad), value=STOP_TEXT_TOKEN)

    tt = TtXtts(device, sd, XttsHifiDecoderFull(sd))
    spk_wav_tt = ttnn.from_torch(
        spk_wav.reshape(1, -1).float(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.float32
    )

    ttnn.manual_seed(1234, device=device)
    wav_eval_dev, codes_eval = tt.inference(
        wrapped,
        wav,
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

    spk_np = spk_wav[0].numpy()
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
