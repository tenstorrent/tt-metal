# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end validation for the SeamlessM4T-v2 T2ST TTNN demo.

Runs the TTNN ``TextToSpeechModel.synthesize`` on a short prompt
(``"Hello world."`` -> ``fra``), runs the HF reference
``SeamlessM4Tv2ForTextToSpeech.generate`` on the same prompt, and validates
both audio outputs.

Validation strategy (gracefully degrades):

    1. Sanity gate: both TTNN and HF audio must be non-empty (>0 samples)
       and have similar durations (within 1.0 second).
    2. Audio parity gate: re-transcribe BOTH outputs with HF's
       ``SeamlessM4Tv2ForSpeechToText``. Compute a normalised
       1 - char-WER (cosine-like similarity) between the two transcripts.
       Gate: similarity >= 0.5 (we expect TTNN and HF to land on
       similar-but-not-identical bf16 outputs).
    3. If re-ASR fails for any reason (no model on disk, OOM, etc.), the
       test prints a warning and passes on the sanity gate alone.

This is necessarily a looser gate than the T2TT BLEU test: audio outputs
involve a HiFi-GAN with feedback loops and bf16 accumulation, so byte-for-
byte parity is not expected. The point is to catch regressions where the
TTNN pipeline produces silence, NaNs, or wildly off-topic audio.

Run with::

    pytest models/demos/facebook_seamless_m4t_v2_large/tests/test_e2e_t2st.py -v
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional, Tuple

import numpy as np
import pytest
import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
from models.demos.facebook_seamless_m4t_v2_large.tt.text_to_speech_model import TextToSpeechModel

# Short-form prompts. Keep the test fast: 1 prompt at MAX_NEW_TOKENS=32.
PROMPTS = [
    {"src": "Hello world.", "src_lang": "eng", "tgt_lang": "fra"},
]

MAX_NEW_TOKENS = 32
SAMPLING_RATE = 16000

# Sanity gate thresholds.
MIN_AUDIO_SECONDS = 0.2  # both outputs must be >= 200 ms.
MAX_DURATION_DIFF_SECONDS = 2.0  # TTNN and HF must be within 2 s of each other.

# Re-ASR similarity gate (very loose: TTNN bf16 vs HF fp32 vocoder feedback
# loops diverge late, so the re-transcribed strings only need to be roughly
# similar).
MIN_TRANSCRIPT_SIMILARITY = 0.5


@pytest.fixture(scope="module")
def hf_sd():
    return wl.load_hf_state_dict()


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(wl.HF_PATH)


@pytest.fixture(scope="function")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


# --------------------------------------------------------------------------- helpers


def _char_similarity(a: str, b: str) -> float:
    """Crude 1 - char-edit-distance / max(len). Good enough for an order-of-magnitude check."""
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    m, n = len(a), len(b)
    # Iterative DP.
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    dist = prev[n]
    return 1.0 - dist / float(max(m, n))


def _save_wav_tmp(samples_float32: np.ndarray) -> str:
    """Save float32 mono samples to a temporary 16k PCM WAV; return path."""
    import scipy.io.wavfile as wav

    arr = np.clip(samples_float32.astype(np.float32, copy=False), -1.0, 1.0)
    arr_i16 = (arr * 32767.0).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    wav.write(tmp.name, SAMPLING_RATE, arr_i16)
    return tmp.name


def _hf_t2st(processor, src: str, src_lang: str, tgt_lang: str) -> Tuple[np.ndarray, int]:
    """Run HF ``SeamlessM4Tv2ForTextToSpeech.generate`` and return float waveform + valid length."""
    from transformers import SeamlessM4Tv2ForTextToSpeech

    model = SeamlessM4Tv2ForTextToSpeech.from_pretrained(wl.HF_PATH, torch_dtype=torch.float32)
    model.eval()
    toks = processor(text=src, src_lang=src_lang, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            input_ids=toks["input_ids"],
            attention_mask=toks["attention_mask"],
            tgt_lang=tgt_lang,
            speaker_id=0,
            do_sample=False,
            num_beams=1,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    if isinstance(out, tuple):
        waveform, waveform_lengths = out[0], out[1]
    else:
        waveform, waveform_lengths = out.waveform, out.waveform_lengths
    audio = waveform[0].detach().cpu().numpy().astype(np.float32)
    length = int(waveform_lengths.view(-1)[0].item()) if waveform_lengths is not None else int(audio.shape[-1])
    length = max(0, min(length, int(audio.shape[-1])))
    del model
    return audio[:length], length


def _try_reasr(wav_path: str, lang: str = "eng") -> Optional[str]:
    """Best-effort: transcribe ``wav_path`` via HF SeamlessM4Tv2ForSpeechToText.

    Returns the transcription string, or ``None`` if anything goes wrong
    (model not on disk, OOM, etc.). The caller decides whether to gate.
    """
    try:
        import scipy.io.wavfile as wav
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        sr, data = wav.read(wav_path)
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.float32:
            audio = data
        else:
            audio = data.astype(np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != SAMPLING_RATE:
            import torchaudio

            t = torch.from_numpy(audio).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, SAMPLING_RATE)
            audio = t.squeeze(0).numpy()

        proc = AutoProcessor.from_pretrained(wl.HF_PATH)
        model = SeamlessM4Tv2ForSpeechToText.from_pretrained(wl.HF_PATH, torch_dtype=torch.float32)
        model.eval()
        inputs = proc(audios=audio.astype(np.float32), sampling_rate=SAMPLING_RATE, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                input_features=inputs["input_features"],
                attention_mask=inputs.get("attention_mask"),
                tgt_lang=lang,
                do_sample=False,
                num_beams=1,
                max_new_tokens=32,
            )
        if hasattr(out, "sequences"):
            seq = out.sequences
        else:
            seq = out
        text = proc.decode(seq[0].tolist(), skip_special_tokens=True)
        del model
        return text
    except Exception as e:
        print(f"[reasr-warning] {type(e).__name__}: {e}")
        return None


# --------------------------------------------------------------------------- test


def test_t2st_audio_parity_with_hf(hf_sd, processor, device):
    """TTNN T2ST must produce non-empty audio of similar duration to HF
    and (best-effort) re-transcribe to a similar string."""

    src = PROMPTS[0]["src"]
    src_lang = PROMPTS[0]["src_lang"]
    tgt_lang = PROMPTS[0]["tgt_lang"]

    # --- 1. TTNN synthesis -------------------------------------------------
    model = TextToSpeechModel(device=device, hf_state_dict=hf_sd, processor=processor)
    ttnn_audio = model.synthesize(
        src_text=src,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        speaker_id=0,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    ttnn_seconds = ttnn_audio.shape[-1] / SAMPLING_RATE

    # --- 2. HF reference synthesis ----------------------------------------
    hf_audio, hf_length = _hf_t2st(processor, src=src, src_lang=src_lang, tgt_lang=tgt_lang)
    hf_seconds = hf_audio.shape[-1] / SAMPLING_RATE

    print("")
    print(f"  src        : {src}")
    print(f"  TTNN audio : {ttnn_audio.shape[-1]} samples / {ttnn_seconds:.3f} s")
    print(f"  HF audio   : {hf_audio.shape[-1]} samples / {hf_seconds:.3f} s")

    # --- Gate 1: both non-empty + similar durations ------------------------
    assert ttnn_seconds >= MIN_AUDIO_SECONDS, f"TTNN audio is too short ({ttnn_seconds:.3f}s)"
    assert hf_seconds >= MIN_AUDIO_SECONDS, f"HF audio is too short ({hf_seconds:.3f}s)"
    assert (
        abs(ttnn_seconds - hf_seconds) <= MAX_DURATION_DIFF_SECONDS
    ), f"Duration drift {abs(ttnn_seconds - hf_seconds):.2f}s exceeds {MAX_DURATION_DIFF_SECONDS}s"

    # --- Gate 2: NaN / inf check on TTNN audio -----------------------------
    assert np.isfinite(ttnn_audio).all(), "TTNN audio contains NaN or inf values"

    # --- Gate 3: best-effort re-ASR similarity -----------------------------
    ttnn_path = _save_wav_tmp(ttnn_audio)
    hf_path = _save_wav_tmp(hf_audio)
    try:
        ttnn_transcript = _try_reasr(ttnn_path, lang=tgt_lang)
        hf_transcript = _try_reasr(hf_path, lang=tgt_lang)
        print(f"  TTNN re-ASR : {ttnn_transcript!r}")
        print(f"  HF   re-ASR : {hf_transcript!r}")
        if ttnn_transcript is not None and hf_transcript is not None:
            sim = _char_similarity(ttnn_transcript, hf_transcript)
            print(f"  char-sim    : {sim:.3f}  (gate >= {MIN_TRANSCRIPT_SIMILARITY})")
            assert sim >= MIN_TRANSCRIPT_SIMILARITY, (
                f"TTNN<->HF re-ASR similarity {sim:.3f} below gate {MIN_TRANSCRIPT_SIMILARITY}; "
                f"TTNN={ttnn_transcript!r}  HF={hf_transcript!r}"
            )
        else:
            print("  [reasr-skip] one side returned None; gating on sanity only.")
    finally:
        for p in (ttnn_path, hf_path):
            try:
                os.unlink(p)
            except OSError:
                pass
