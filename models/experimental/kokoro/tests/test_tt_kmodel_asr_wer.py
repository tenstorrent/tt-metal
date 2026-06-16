# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Kokoro E2E intelligibility via Whisper ASR word error rate.

Mirrors the Voxtral free-run ASR gate from commit ``c6c1415c`` on
``origin/ign/voxtral_debug2``: generate audio, transcribe with Whisper-small from
``transformers``, and assert word error rate (WER) is below ``ASR_WER_TARGET``.

Sample-wise waveform PCC is a poor free-run gate for TTS; ASR WER measures whether
the synthesized speech is intelligible relative to the input text prompt.
"""

from __future__ import annotations

import gc
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.experimental.kokoro.reference.model import KModel
from models.experimental.kokoro.tests.kokoro_checkpoint import find_checkpoint
from models.experimental.kokoro.tt.tt_kmodel import KokoroConfig, TTKModel, preprocess_tt_kmodel

ASR_WER_TARGET = 0.30
ASR_SAMPLE_RATE = 16000
WHISPER_MODEL = "openai/whisper-small"
_OUTPUT_SAMPLE_RATE = KokoroConfig.sample_rate_hz

_VOICE = "af_heart"
_LANG_CODE = "a"
# Common-word prompt for the WER gate (avoids coined brand words that Whisper
# systematically misspells and that would inflate WER without reflecting intelligibility).
_ASR_TEXT = "Hello world this is a speech synthesis test."


@dataclass(frozen=True)
class _Cfg:
    """Vocoder fallback / STFT-formulation config (shared matrix with the cFW2VD / mel-PCC /
    SECS metric tests; defined locally here to avoid a circular import)."""

    label: str
    disable_complex: bool
    use_torch_stft_fallback: bool
    use_torch_phase_fallback: bool


_CONFIGS = (
    _Cfg("phase_fallback", disable_complex=False, use_torch_stft_fallback=False, use_torch_phase_fallback=True),
    _Cfg("stft_and_phase_fallback", disable_complex=False, use_torch_stft_fallback=True, use_torch_phase_fallback=True),
    _Cfg("no_fallback", disable_complex=False, use_torch_stft_fallback=False, use_torch_phase_fallback=False),
    _Cfg("dc_phase_fallback", disable_complex=True, use_torch_stft_fallback=False, use_torch_phase_fallback=True),
    _Cfg("dc_no_fallback", disable_complex=True, use_torch_stft_fallback=False, use_torch_phase_fallback=False),
)


def _load_pipeline():
    try:
        from kokoro import KPipeline

        return KPipeline(lang_code=_LANG_CODE, model=False)
    except ImportError:
        return None


def _phonemize(text: str) -> tuple[str, torch.Tensor]:
    pipe = _load_pipeline()
    if pipe is None:
        pytest.skip("kokoro package not installed: pip install 'kokoro>=0.9.2'")

    results = list(pipe(text, voice=_VOICE))
    if not results:
        pytest.skip(f"KPipeline produced no chunks for: {text!r}")

    phonemes = results[0].phonemes
    if not phonemes:
        pytest.skip("KPipeline produced empty phonemes for first chunk.")

    pack = pipe.load_voice(_VOICE)
    ref_s = pack[len(phonemes) - 1]
    if not isinstance(ref_s, torch.Tensor):
        ref_s = torch.tensor(ref_s)
    ref_s = ref_s.float().cpu()
    if ref_s.dim() == 1:
        ref_s = ref_s.unsqueeze(0)
    return phonemes, ref_s


def _transcribe_waveform(waveform: torch.Tensor, src_sr: int) -> str:
    """Transcribe a 1D float waveform with Whisper-small."""
    import librosa
    import numpy as np
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    audio = waveform.detach().reshape(-1).float().cpu().numpy().astype(np.float32)
    if src_sr != ASR_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=src_sr, target_sr=ASR_SAMPLE_RATE)

    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    whisper = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).eval()
    feats = processor(audio, sampling_rate=ASR_SAMPLE_RATE, return_tensors="pt").input_features
    with torch.no_grad():
        ids = whisper.generate(feats, language="en", task="transcribe")
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def _normalize_words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", s.lower()).split()


def _word_error_rate(reference: str, hypothesis: str) -> float:
    """WER = Levenshtein word edit distance / #reference words."""
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
    """Fraction of reference words present in the transcription (informational)."""
    ref_words = set(_normalize_words(reference))
    if not ref_words:
        return 1.0
    return len(ref_words & set(_normalize_words(transcription))) / len(ref_words)


def _load_kmodel(disable_complex: bool = True) -> KModel:
    ckpt_path = find_checkpoint()
    if ckpt_path is None:
        pytest.skip("Kokoro-82M checkpoint not found locally.")
    return KModel(
        repo_id=KokoroConfig.repo_id,
        model=str(ckpt_path),
        disable_complex=disable_complex,
    ).eval()


def _tt_audio(device, ref: KModel, params, phonemes: str, ref_s: torch.Tensor, cfg: _Cfg) -> torch.Tensor:
    tt_model = TTKModel(
        device,
        ref,
        params,
        use_torch_stft_fallback=cfg.use_torch_stft_fallback,
        use_torch_phase_fallback=cfg.use_torch_phase_fallback,
        disable_complex=cfg.disable_complex,
    )
    out = tt_model(phonemes=phonemes, ref_s=ref_s, speed=1.0, deterministic=True)
    return out.audio.detach().float().squeeze()


def _ref_audio(ref: KModel, phonemes: str, ref_s: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = ref.forward(phonemes=phonemes, ref_s=ref_s, speed=1.0, return_output=False)
    return out.detach().float().squeeze()


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("cfg", _CONFIGS, ids=[c.label for c in _CONFIGS])
def test_ttnn_kokoro_asr_wer(device, reset_seeds, cfg: _Cfg):
    """TT Kokoro E2E correctness via Whisper ASR WER, per vocoder-fallback config.

    Generate audio on the Tenstorrent device with the ``cfg`` vocoder fallbacks / STFT
    formulation, transcribe with Whisper, and assert WER is below ``ASR_WER_TARGET``
    (0.30, matching Voxtral Phase-4).
    """
    try:
        import librosa  # noqa: F401
        from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: F401
    except Exception as exc:
        pytest.skip(f"Whisper ASR deps unavailable: {exc}")

    phonemes, ref_s = _phonemize(_ASR_TEXT)
    ref = _load_kmodel(cfg.disable_complex)
    params = preprocess_tt_kmodel(ref, device)

    cfg_str = (
        f"disable_complex={cfg.disable_complex}, stft_fallback={cfg.use_torch_stft_fallback}, "
        f"phase_fallback={cfg.use_torch_phase_fallback}"
    )
    logger.info("=" * 70)
    logger.info(f"KOKORO ASR GATE [{cfg.label}] ({cfg_str}) (TT generates audio; Whisper transcribes; WER)")
    logger.info("=" * 70)

    torch.manual_seed(0)
    waveform = _tt_audio(device, ref, params, phonemes, ref_s, cfg)
    ttnn.synchronize_device(device)

    assert waveform.numel() > 0, f"TT Kokoro [{cfg.label}] produced no audio samples"
    assert torch.isfinite(waveform).all(), f"TT Kokoro [{cfg.label}] waveform has non-finite samples"

    transcription = _transcribe_waveform(waveform, _OUTPUT_SAMPLE_RATE)
    wer = _word_error_rate(_ASR_TEXT, transcription)
    overlap = _word_overlap(transcription, _ASR_TEXT)
    duration_s = float(waveform.reshape(-1).numel()) / _OUTPUT_SAMPLE_RATE
    logger.info(f"  config       : {cfg_str}")
    logger.info(f"  target       : {_ASR_TEXT!r}")
    logger.info(f"  transcription: {transcription!r}")
    logger.info(
        f"  WER          : {wer:.2%}  target<{ASR_WER_TARGET:.0%}  "
        f"[{'PASS' if wer < ASR_WER_TARGET else 'HIGH'}]  (word overlap={overlap:.2%}, "
        f"phonemes={len(phonemes)}, audio={duration_s:.2f}s)"
    )

    assert wer < ASR_WER_TARGET, (
        f"Kokoro ASR WER {wer:.2%} >= {ASR_WER_TARGET:.0%} [{cfg.label}]; "
        f"transcription={transcription!r} target={_ASR_TEXT!r}"
    )

    ttnn.synchronize_device(device)
    gc.collect()


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_cpu_reference_kokoro_asr_diagnostic(reset_seeds):
    """DIAGNOSTIC (not gated): torch CPU-reference on the SAME ``_ASR_TEXT``/seed.

    Establishes the reference duration / WER so we can tell whether the TT free-run
    (``test_ttnn_kokoro_asr_wer``) is worse than the reference (=> a TT accuracy bug)
    or the reference also reads high WER (=> the prompt/target itself). Runs on CPU only.
    """
    try:
        import librosa  # noqa: F401
        from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: F401
    except Exception as exc:
        pytest.skip(f"Whisper ASR deps unavailable: {exc}")

    phonemes, ref_s = _phonemize(_ASR_TEXT)
    ref = _load_kmodel()

    logger.info("=" * 70)
    logger.info("CPU-REFERENCE KOKORO ASR DIAGNOSTIC (torch generates audio; Whisper transcribes)")
    logger.info("=" * 70)

    torch.manual_seed(0)
    ref_wav = _ref_audio(ref, phonemes, ref_s)
    assert torch.isfinite(ref_wav).all(), "CPU reference produced non-finite waveform samples"

    transcription = _transcribe_waveform(ref_wav, _OUTPUT_SAMPLE_RATE)
    wer = _word_error_rate(_ASR_TEXT, transcription)
    overlap = _word_overlap(transcription, _ASR_TEXT)
    duration_s = float(ref_wav.reshape(-1).numel()) / _OUTPUT_SAMPLE_RATE
    logger.info(f"  target       : {_ASR_TEXT!r}")
    logger.info(f"  transcription: {transcription!r}")
    logger.info(
        f"  WER          : {wer:.2%}  (word overlap={overlap:.2%}, "
        f"phonemes={len(phonemes)}, audio={duration_s:.2f}s)"
    )

    gc.collect()
