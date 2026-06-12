# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E2E PCC using the standard teacher-forced methodology.
"""
from __future__ import annotations

import gc
import os

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
from models.experimental.voxtraltts.demo.decode_trace_2cq import num_command_queues_for_decode
from models.experimental.voxtraltts.tt.voxtral_tt_args import voxtral_text_hf_aligned_optimizations
from models.experimental.voxtraltts.tt.voxtral_tts import VoxtralTTSPipeline

os.environ.setdefault("VOXTRAL_DECODE_TRACE", "1")

WAVEFORM_PCC_TARGET = 0.99

# Free-run correctness is gated by Whisper ASR intelligibility (WER), NOT sample-wise waveform
# PCC (the discrete-FSQ AR feedback makes the latter structurally unreachable). Mirrors the
# ``ssinghal/voxtral_tts`` Phase-4 metric: Whisper WER on a standard prompt, target WER < 0.30.
ASR_WER_TARGET = 0.30
ASR_SAMPLE_RATE = 16000
WHISPER_MODEL = "openai/whisper-small"
_OUTPUT_SAMPLE_RATE = 24000

_DEMO_TEXT = VOXTRAL_STANDARD_CHAR_TEXT
_DEMO_VOICE = "casual_male"


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
    # Whisper only "hears" a fixed 30s window (the feature extractor truncates to 30s); transcribe
    # long audio in 30s chunks and concatenate, else everything past 30s is silently dropped.
    chunk = 30 * ASR_SAMPLE_RATE
    segments = []
    with torch.no_grad():
        for start in range(0, max(len(audio), 1), chunk):
            seg = audio[start : start + chunk]
            if seg.size == 0:
                continue
            feats = processor(seg, sampling_rate=ASR_SAMPLE_RATE, return_tensors="pt").input_features
            ids = whisper.generate(feats, language="en", task="transcribe")
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


def _trace_device_params() -> dict[str, int]:
    return {
        "trace_region_size": int(os.environ.get("VOXTRAL_TRACE_REGION_SIZE", str(200_000_000))),
        "num_command_queues": num_command_queues_for_decode(),
    }


@pytest.fixture()
def voxtral_qb_device():
    """Open the SAME mesh the demo uses so PCC runs on the BH QuietBox exactly like the demo:
    a 1x4 mesh (P150x4) when >=4 chips are present, else single-chip (P150). Mirrors
    ``demo._open_device`` (FABRIC_1D + physical_device_ids=[0,1,2,3]); does not touch the demo.
    Set ``VOXTRAL_TTS_FORCE_SINGLE_DEVICE=1`` to force single-chip for an A/B comparison.
    """
    import ttnn
    from models.experimental.voxtraltts.demo.demo import _open_device

    mesh, original = _open_device()
    # The captured decode-trace path (decode_step_from_embeds_tt) is not yet TP-aware: it feeds the
    # decoder a full-width replicated hidden, which the DistributedNorm all_gather can't shard on a
    # 1xN mesh. The untraced decode path IS TP-aware (matches the demo) and produces identical codes,
    # so force trace OFF on multi-device. Single-chip keeps its default (trace as configured).
    if mesh.get_num_devices() > 1:
        os.environ["VOXTRAL_DECODE_TRACE"] = "0"
    try:
        yield mesh
    finally:
        try:
            if original is not None:
                ttnn.SetDefaultDevice(original)
        except Exception:
            pass
        ttnn.close_mesh_device(mesh)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_ttnn_voxtral_tts_staged_pcc(voxtral_qb_device, reset_seeds, request):
    """Teacher-forced E2E waveform PCC (shared codes) + logged free-run divergence."""
    device = voxtral_qb_device
    generate_steps = 1
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
@pytest.mark.timeout(3600)
def test_ttnn_voxtral_tts_free_run_asr(voxtral_qb_device, reset_seeds, request):
    """Free-run E2E correctness via Whisper ASR WER (the meaningful free-run gate), on the QB.

    Sample-wise free-run waveform PCC is structurally unreachable for a discrete-FSQ AR model,
    so — like the ``ssinghal/voxtral_tts`` Phase-4 Whisper verification — we generate audio
    free-run on the device (4-chip QB via ``voxtral_qb_device``), transcribe it with Whisper, and
    assert the word error rate is below ``ASR_WER_TARGET`` (WER < 0.30).
    """
    device = voxtral_qb_device
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
            text_max_seq_len=2048,  # full _DEMO_TEXT paragraph needs ~575 frames + ~280 prompt positions
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
    logger.info("FREE-RUN ASR GATE (TT generates audio on QB; Whisper transcribes; WER)")
    logger.info("=" * 70)
    out = pipe.generate_with_codes(text=_DEMO_TEXT, voice=_DEMO_VOICE, max_tokens=1500, seed=0)
    ttnn.synchronize_device(device)

    assert out.codes_b37t.shape[2] > 0, "free-run generation produced no acoustic frames"
    assert torch.isfinite(out.waveform).all(), "free-run waveform has non-finite samples"

    transcription = _transcribe_waveform(out.waveform, _OUTPUT_SAMPLE_RATE)
    wer = _word_error_rate(_DEMO_TEXT, transcription)
    overlap = _word_overlap(transcription, _DEMO_TEXT)
    duration_s = float(out.waveform.reshape(-1).numel()) / _OUTPUT_SAMPLE_RATE
    logger.info(f"  target       : {_DEMO_TEXT!r}")
    logger.info(f"  transcription: {transcription!r}")
    logger.info(
        f"  WER          : {wer:.2%}  target<{ASR_WER_TARGET:.0%}  "
        f"[{'PASS' if wer < ASR_WER_TARGET else 'HIGH'}]  (word overlap={overlap:.2%}, "
        f"frames={int(out.codes_b37t.shape[2])}, audio={duration_s:.2f}s, hit_end={out.hit_end_audio})"
    )

    assert (
        wer < ASR_WER_TARGET
    ), f"free-run ASR WER {wer:.2%} >= {ASR_WER_TARGET:.0%}; transcription={transcription!r} target={_DEMO_TEXT!r}"

    ttnn.synchronize_device(device)
    pipe.cleanup_all()
    pipe_holder[0] = None
    del pipe
    del out
    gc.collect()
