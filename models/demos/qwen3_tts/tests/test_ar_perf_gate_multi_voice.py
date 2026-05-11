# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Multi-voice / multi-language AR perf + accuracy gate for qwen3_tts.

For each (voice, language) reference clip in
/local/ttuser/ssinghal/tts2/tts-models/tts-2/prompts:
  1. Transcribe the reference clip with Whisper-large-v3 (language pinned)
     to obtain ref_text.
  2. Run the TTS demo with a fixed-per-language target text, that ref clip,
     and that ref_text. seed=42, --use-2cq, TT_QWEN3_CP_FP32=1.
  3. Assert steady ms/frame < 55, ECAPA cos(generated, reference) >= 0.97,
     and Whisper transcript of the output matches the target text (after
     locale-aware normalization).

Skips voices whose language is not in server.TTSConfig.codec_language_ids
(currently nl, pl).
"""
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

PROMPTS_DIR = Path("/local/ttuser/ssinghal/tts2/tts-models/tts-2/prompts")

# Suffix in filename ("Alex_en.wav") → name used by server.codec_language_ids.
LANG_MAP = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "es": "spanish",
    "ja": "japanese",
    "ko": "korean",
    "fr": "french",
    "ru": "russian",
    # "nl" and "pl" intentionally excluded — model has no codec_language_id.
}

# Whisper language code per suffix (ISO-639-1 mostly matches our suffix).
WHISPER_LANG = {
    "en": "en",
    "zh": "zh",
    "de": "de",
    "it": "it",
    "pt": "pt",
    "es": "es",
    "ja": "ja",
    "ko": "ko",
    "fr": "fr",
    "ru": "ru",
}

# Target utterance per language — ~150 chars of common vocabulary, no brand names.
LANG_TARGETS = {
    "en": "Good morning. Today is a beautiful day for a walk in the park, with bright sun and a gentle breeze through the trees.",
    "zh": "早上好。今天是个适合在公园散步的好日子,阳光明媚,微风轻轻吹过树叶。",
    "de": "Guten Morgen. Heute ist ein wunderschöner Tag für einen Spaziergang im Park, mit hellem Sonnenschein und einer sanften Brise.",
    "it": "Buongiorno. Oggi è una bellissima giornata per una passeggiata al parco, con un sole splendente e una leggera brezza.",
    "pt": "Bom dia. Hoje é um dia lindo para passear no parque, com um sol brilhante e uma brisa suave entre as árvores.",
    "es": "Buenos días. Hoy es un día hermoso para pasear por el parque, con un sol brillante y una brisa suave entre los árboles.",
    "ja": "おはようございます。今日は公園を散歩するのに素敵な日です。明るい日差しと木々を抜けるそよ風が心地よいです。",
    "ko": "좋은 아침입니다. 오늘은 공원을 산책하기에 좋은 날입니다. 밝은 햇살과 나무 사이로 부는 산들바람이 기분 좋습니다.",
    "fr": "Bonjour. Aujourd'hui est une belle journée pour une promenade dans le parc, avec un soleil éclatant et une brise légère.",
    "ru": "Доброе утро. Сегодня прекрасный день для прогулки по парку, с ярким солнцем и лёгким ветерком среди деревьев.",
}

MS_PER_FRAME_MAX = 55.0
ECAPA_COS_MIN = 0.97


def _norm(s: str) -> str:
    """Locale-aware normalization for Whisper-output vs target-text comparison.

    Lowercases, NFKC-normalizes, strips all Unicode punctuation/separator/symbol
    categories, and collapses whitespace. Works for Latin, Cyrillic, and CJK
    (CJK characters survive; CJK punctuation is removed)."""
    s = unicodedata.normalize("NFKC", s).lower().strip()
    # Drop punctuation (P*), symbols (S*), and separators (Z*).
    out_chars = []
    for ch in s:
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "S", "Z"):
            out_chars.append(" ")
        else:
            out_chars.append(ch)
    s = "".join(out_chars)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _voices():
    """Yield (wav_path, name, suffix) for each *.wav whose language we support."""
    out = []
    if not PROMPTS_DIR.exists():
        return out
    for wav in sorted(PROMPTS_DIR.glob("*.wav")):
        if "_" not in wav.stem:
            continue
        name, suf = wav.stem.rsplit("_", 1)
        if suf not in LANG_MAP:
            continue
        out.append((wav, name, suf))
    return out


@pytest.fixture(scope="session")
def whisper_lv3():
    """Whisper-large-v3 multilingual transcriber (loaded once for the session)."""
    import librosa
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    print("\nLoading whisper-large-v3 ...")
    proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
    model.eval()

    def transcribe(wav_path: str, lang: str) -> str:
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(feats, language=lang, task="transcribe", temperature=0)
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    return transcribe


@pytest.mark.parametrize(
    "wav_path,name,suf", _voices(), ids=lambda v: getattr(v, "stem", None) or str(v) if hasattr(v, "stem") else str(v)
)
def test_multi_voice_gate(wav_path, name, suf, whisper_lv3):
    lang_name = LANG_MAP[suf]
    whisper_lang = WHISPER_LANG[suf]
    target_text = LANG_TARGETS[suf]

    # 1. Transcribe reference clip to get ref_text in its language.
    ref_text = whisper_lv3(str(wav_path), whisper_lang)
    assert ref_text, f"Whisper produced empty transcription for {wav_path}"

    # 2. Run TTS demo.
    os.environ["TT_QWEN3_CP_FP32"] = "1"
    out_wav = f"/tmp/multi_voice_{name}_{suf}.wav"
    from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import run_full_ttnn_tts

    result = run_full_ttnn_tts(
        text=target_text,
        ref_audio=str(wav_path),
        ref_text=ref_text,
        output_path=out_wav,
        seed=42,
        use_2cq=True,
        language=lang_name,
    )
    assert (
        isinstance(result, dict) and "steady_ms_per_frame" in result
    ), f"run_full_ttnn_tts didn't return timing dict: {result}"

    steady = result["steady_ms_per_frame"]
    print(f"\n[{name}_{suf}] steady_ms_per_frame={steady:.2f} target={target_text[:40]!r}")

    # Run all 3 gates independently and accumulate failures so one bad gate
    # doesn't hide the others.
    failures = []

    if not (steady < MS_PER_FRAME_MAX):
        failures.append(f"ms/frame {steady:.2f} >= {MS_PER_FRAME_MAX}")

    from models.demos.qwen3_tts.tests.audio_diff import speaker_similarity_via_reference

    sims = speaker_similarity_via_reference(str(wav_path), out_wav)
    if sims is None:
        failures.append("ECAPA helper unavailable")
        cos = float("nan")
    else:
        cos = sims[0]
        print(f"[{name}_{suf}] ECAPA cos = {cos:.4f}")
        if not (cos >= ECAPA_COS_MIN):
            failures.append(f"ECAPA {cos:.4f} < {ECAPA_COS_MIN}")

    transcript = whisper_lv3(out_wav, whisper_lang)
    print(f"[{name}_{suf}] whisper = {transcript!r}")
    if _norm(transcript) != _norm(target_text):
        failures.append(f"whisper mismatch: got={_norm(transcript)!r} exp={_norm(target_text)!r}")

    # Tagged summary line for easy grepping across the run.
    status = "OK" if not failures else "FAIL"
    print(f"[GATE {status}] {name}_{suf}: ms/frame={steady:.2f} ecapa={cos:.4f} fails={failures}")

    if failures:
        pytest.fail(f"{name}_{suf}: " + " | ".join(failures))
