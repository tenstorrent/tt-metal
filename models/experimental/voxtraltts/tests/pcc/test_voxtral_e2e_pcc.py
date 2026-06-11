# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E2E PCC using the standard teacher-forced methodology.
"""
from __future__ import annotations

import gc

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.audio_tokenizer_ops import audio_tokenizer_decode_reference
from models.experimental.voxtraltts.reference.cpu_reference import VoxtralCPUReference
from models.experimental.voxtraltts.reference.voxtral_config import DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN
from models.experimental.voxtraltts.tests.common import (
    VOXTRAL_STANDARD_CHAR_TEXT,
    log_per_step_code_match,
    resolve_voxtral_model_name_or_skip,
)
from models.experimental.voxtraltts.tt.voxtral_tt_args import voxtral_text_hf_aligned_optimizations
from models.experimental.voxtraltts.tt.voxtral_tts import VoxtralTTSPipeline

WAVEFORM_PCC_TARGET = 0.99

ASR_WER_TARGET = 0.30
ASR_SAMPLE_RATE = 16000
WHISPER_MODEL = "openai/whisper-small"

_DEMO_TEXT = VOXTRAL_STANDARD_CHAR_TEXT
_ASR_TEXT = VOXTRAL_STANDARD_CHAR_TEXT
_DEMO_VOICE = "casual_male"
_OUTPUT_SAMPLE_RATE = 24000


def _log_pcc(label: str, pcc_value: float, target: float) -> None:
    status = "PASS" if pcc_value >= target else "LOW"
    logger.info(f"  {label}: PCC={pcc_value:.4f}  target>={target:.4f}  [{status}]")


def _align_1d(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    n = min(int(a.numel()), int(b.numel()))
    return a[:n], b[:n]


def _transcribe_waveform(waveform: torch.Tensor, src_sr: int) -> str:
    """Transcribe a 1D float waveform with Whisper-small (mirrors Qwen3-TTS ``_transcribe``)."""
    import librosa
    import numpy as np
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    audio = waveform.detach().reshape(-1).float().cpu().numpy().astype(np.float32)
    if src_sr != ASR_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=src_sr, target_sr=ASR_SAMPLE_RATE)

    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    whisper = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).eval()
    # Whisper only "hears" a fixed 30s window (the feature extractor truncates to 30s).
    # Long audio (e.g. the full _DEMO_TEXT paragraph ~46s) must be transcribed in 30s
    # chunks and concatenated, else everything past 30s is silently dropped.
    chunk = 30 * ASR_SAMPLE_RATE
    segments = []
    with torch.no_grad():
        for start in range(0, max(len(audio), 1), chunk):
            seg = audio[start : start + chunk]
            if seg.size == 0:
                continue
            inputs = processor(seg, sampling_rate=ASR_SAMPLE_RATE, return_tensors="pt", return_attention_mask=True)
            generate_kwargs = {"input_features": inputs.input_features, "language": "en", "task": "transcribe"}
            if getattr(inputs, "attention_mask", None) is not None:
                generate_kwargs["attention_mask"] = inputs.attention_mask
            ids = whisper.generate(**generate_kwargs)
            segments.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return " ".join(s for s in segments if s).strip()


def _normalize_words(s: str) -> list[str]:
    import re

    return re.sub(r"[^a-z0-9\s]", " ", s.lower()).split()


def _word_error_rate(reference: str, hypothesis: str) -> float:
    """WER = Levenshtein word edit distance / #reference words (the ssinghal/voxtral_tts metric)."""
    ref = _normalize_words(reference)
    hyp = _normalize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n


def _word_overlap(transcription: str, reference: str) -> float:
    """Fraction of reference words present in the transcription (Qwen3-TTS metric; informational)."""
    ref_words = set(_normalize_words(reference))
    if not ref_words:
        return 1.0
    return len(ref_words & set(_normalize_words(transcription))) / len(ref_words)


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_ttnn_voxtral_tts_staged_pcc(device, reset_seeds, request):
    """Teacher-forced E2E waveform PCC (shared codes) + logged free-run divergence."""
    generate_steps = 8
    name = resolve_voxtral_model_name_or_skip()

    try:
        cpu = VoxtralCPUReference(model_name_or_path=name, dtype="bfloat16", device="cpu")
    except Exception as exc:
        pytest.skip(f"CPU reference load failed: {exc}")

    try:
        pipe = VoxtralTTSPipeline.from_model_name(
            device,
            model_name_or_path=name,
            text_max_seq_len=DEFAULT_VOXTRAL_TT_TEXT_MAX_SEQ_LEN,
            text_optimizations=voxtral_text_hf_aligned_optimizations,
        )
    except Exception as exc:
        pytest.skip(f"TT pipeline load failed: {exc}")
    pipe_holder = [pipe]

    def _cleanup_pipe() -> None:
        if pipe_holder[0] is not None:
            pipe_holder[0].cleanup_all()
            pipe_holder[0] = None

    request.addfinalizer(_cleanup_pipe)

    # ---------------------------------------------------------------------
    # Step 1-2: one CPU generate produces the shared code history + ref wav.
    # ---------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("CPU GENERATE (produces shared code history = teacher-forcing input)")
    logger.info("=" * 70)
    ref_wav_gen, ref_codes = cpu.generate(
        text=_DEMO_TEXT,
        voice=_DEMO_VOICE,
        max_tokens=generate_steps,
        seed=0,
        return_tokenizer_codes=True,
    )
    assert torch.isfinite(ref_wav_gen).all(), "CPU reference produced non-finite waveform samples"
    assert int(ref_codes.shape[2]) > 0, "CPU reference produced no acoustic frames"

    # ---------------------------------------------------------------------
    # Step 2-4 (PCC gate): decode the SAME codes through the reference torch
    # tokenizer and the TT tokenizer, then compare. This is the canonical
    # "same input -> ref output / tt output -> compare" methodology, on a
    # continuous tensor (waveform). Mirrors SpeechT5's teacher-forced PCC.
    # ---------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("TEACHER-FORCED E2E (decode shared codes: torch ref vs TT)")
    logger.info("=" * 70)
    ref_wav = audio_tokenizer_decode_reference(ref_codes, pipe.audio_tokenizer_sd, pipe.config.audio_tokenizer_args)
    tt_wav = pipe.decode_waveform_from_codes_tt(ref_codes)
    ttnn.synchronize_device(device)
    assert torch.isfinite(tt_wav).all(), "TT tokenizer produced non-finite waveform samples"

    ref_flat, tt_flat = _align_1d(ref_wav, tt_wav)
    _, wav_pcc = comp_pcc(ref_flat, tt_flat, pcc=WAVEFORM_PCC_TARGET)
    _log_pcc("waveform (teacher-forced shared codes)", float(wav_pcc), WAVEFORM_PCC_TARGET)

    # ---------------------------------------------------------------------
    # Informational only: free-running TT generation. Reads ~0.77 (fp32-softmax
    # aligned) because the discrete-code AR feedback diverges from the CPU rollout
    # once the first few FSQ-boundary flips occur. NOT asserted (see module docstring).
    # ---------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("FREE-RUN DIAGNOSTIC (TT generates its own codes; informational, not gated)")
    logger.info("=" * 70)
    tt_out = pipe.forward_device_resident(
        text=_DEMO_TEXT,
        voice=_DEMO_VOICE,
        max_tokens=generate_steps,
        seed=0,
    )
    ttnn.synchronize_device(device)

    n_frames = min(int(tt_out.codes_b37t.shape[2]), int(ref_codes.shape[2]))
    tt_codes = tt_out.codes_b37t[:, :, :n_frames]
    ref_codes_aligned = ref_codes[:, :, :n_frames]
    log_per_step_code_match(ref_codes_aligned, tt_codes)

    sem_matches = int((tt_codes[:, 0] == ref_codes_aligned[:, 0]).sum().item())
    sem_total = int(tt_codes[:, 0].numel())
    ac_matches = int((tt_codes[:, 1:] == ref_codes_aligned[:, 1:]).sum().item())
    ac_total = int(tt_codes[:, 1:].numel())
    logger.info(f"  semantic-code match: {sem_matches / max(sem_total, 1):.4f}  ({sem_matches}/{sem_total})")
    logger.info(f"  acoustic-code match: {ac_matches / max(ac_total, 1):.4f}  ({ac_matches}/{ac_total})")

    free_ref, free_tt = _align_1d(ref_wav_gen, tt_out.waveform)
    _, free_pcc = comp_pcc(free_ref, free_tt, pcc=WAVEFORM_PCC_TARGET)
    _log_pcc("waveform (free-run, north-star, NOT gated)", float(free_pcc), WAVEFORM_PCC_TARGET)

    # The actual correctness gate is the teacher-forced continuous comparison.
    assert bool(wav_pcc >= WAVEFORM_PCC_TARGET), f"teacher-forced waveform PCC below target: {wav_pcc}"

    ttnn.synchronize_device(device)
    pipe.cleanup_all()
    pipe_holder[0] = None
    del pipe
    del tt_out
    gc.collect()


@torch.no_grad()
@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_ttnn_voxtral_tts_free_run_asr(device, reset_seeds, request):
    """Free-run E2E correctness via ASR WER.

    Sample-wise free-run waveform PCC is structurally unreachable for a discrete-FSQ AR
    model. This local no-API test uses Whisper as the available in-repo ASR backend.
    """
    name = resolve_voxtral_model_name_or_skip()
    try:
        import librosa  # noqa: F401
        from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: F401
    except Exception as exc:
        pytest.skip(f"Whisper ASR deps unavailable: {exc}")

    try:
        pipe = VoxtralTTSPipeline.from_model_name(
            device,
            model_name_or_path=name,
            text_max_seq_len=2048,
            text_optimizations=voxtral_text_hf_aligned_optimizations,
        )
    except Exception as exc:
        pytest.skip(f"TT pipeline load failed: {exc}")
    pipe_holder = [pipe]

    def _cleanup_pipe() -> None:
        if pipe_holder[0] is not None:
            pipe_holder[0].cleanup_all()
            pipe_holder[0] = None

    request.addfinalizer(_cleanup_pipe)

    logger.info("=" * 70)
    logger.info("FREE-RUN ASR GATE (TT generates audio; Whisper transcribes; WER)")
    logger.info("=" * 70)
    out = pipe.generate_with_codes(text=_ASR_TEXT, voice=_DEMO_VOICE, max_tokens=1500, seed=0)
    ttnn.synchronize_device(device)

    assert out.codes_b37t.shape[2] > 0, "free-run generation produced no acoustic frames"
    assert torch.isfinite(out.waveform).all(), "free-run waveform has non-finite samples"

    transcription = _transcribe_waveform(out.waveform, _OUTPUT_SAMPLE_RATE)
    wer = _word_error_rate(_ASR_TEXT, transcription)
    overlap = _word_overlap(transcription, _ASR_TEXT)
    duration_s = float(out.waveform.reshape(-1).numel()) / _OUTPUT_SAMPLE_RATE
    logger.info(f"  target       : {_ASR_TEXT!r}")
    logger.info(f"  transcription: {transcription!r}")
    logger.info(
        f"  WER          : {wer:.2%}  target<{ASR_WER_TARGET:.0%}  "
        f"[{'PASS' if wer < ASR_WER_TARGET else 'HIGH'}]  (word overlap={overlap:.2%}, "
        f"frames={int(out.codes_b37t.shape[2])}, audio={duration_s:.2f}s, hit_end={out.hit_end_audio})"
    )

    assert wer < ASR_WER_TARGET, (
        f"free-run ASR WER {wer:.2%} >= {ASR_WER_TARGET:.0%}; " f"transcription={transcription!r} target={_ASR_TEXT!r}"
    )

    ttnn.synchronize_device(device)
    pipe.cleanup_all()
    pipe_holder[0] = None
    del pipe
    del out
    gc.collect()


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_cpu_reference_free_run_asr_diagnostic(reset_seeds):
    """DIAGNOSTIC (not gated): torch CPU-reference free-run on the SAME _ASR_TEXT/seed.

    Establishes the reference's frame count / duration / WER so we can tell whether the
    TT free-run (``test_ttnn_voxtral_tts_free_run_asr``) stops *earlier* than the reference
    (=> a TT accuracy bug) or the reference also stops there (=> the prompt/target itself).
    Runs entirely on CPU (no TT device required).
    """
    name = resolve_voxtral_model_name_or_skip()
    try:
        import librosa  # noqa: F401
        from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: F401
    except Exception as exc:
        pytest.skip(f"Whisper ASR deps unavailable: {exc}")

    try:
        cpu = VoxtralCPUReference(model_name_or_path=name, dtype="bfloat16", device="cpu")
    except Exception as exc:
        pytest.skip(f"CPU reference load failed: {exc}")

    logger.info("=" * 70)
    logger.info("CPU-REFERENCE FREE-RUN ASR DIAGNOSTIC (torch generates audio; Whisper transcribes)")
    logger.info("=" * 70)
    ref_wav, ref_codes = cpu.generate(
        text=_ASR_TEXT,
        voice=_DEMO_VOICE,
        max_tokens=1500,
        seed=0,
        return_tokenizer_codes=True,
    )
    assert torch.isfinite(ref_wav).all(), "CPU reference produced non-finite waveform samples"

    n_frames = int(ref_codes.shape[2])
    hit_end = n_frames < 1500
    transcription = _transcribe_waveform(ref_wav, _OUTPUT_SAMPLE_RATE)
    wer = _word_error_rate(_ASR_TEXT, transcription)
    overlap = _word_overlap(transcription, _ASR_TEXT)
    duration_s = float(ref_wav.reshape(-1).numel()) / _OUTPUT_SAMPLE_RATE
    logger.info(f"  target       : {_ASR_TEXT!r}")
    logger.info(f"  transcription: {transcription!r}")
    logger.info(
        f"  WER          : {wer:.2%}  (word overlap={overlap:.2%}, "
        f"frames={n_frames}, audio={duration_s:.2f}s, hit_end={hit_end})"
    )

    gc.collect()
