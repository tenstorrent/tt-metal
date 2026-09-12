# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Voice-clone quality gate for the TTNN demo: SIM and WER.

Two numbers, both computed with models *independent of the pipeline under test*:

``SIM``
    Cosine similarity between speaker embeddings of the reference clip and the
    generated audio — "does it sound like the person we asked it to imitate".
    Default checkpoint ``microsoft/wavlm-base-plus-sv``.

    **Not** the model's own ECAPA SpeakerEncoder, for two reasons. It is what the
    TTS is *conditioned on*, so scoring with it grades the model on its own
    objective; and its raw cosine is not calibrated — measured on this repo's
    reference clip, silence scores 0.9152 and a +4-semitone pitch shift scores
    0.9932, higher than a genuine clone. It cannot separate speakers.

``WER``
    Word error rate of an ASR transcript against the text we asked for —
    "are the words actually there". Default ``openai/whisper-large-v3``, which is
    what Seed-TTS-Eval prescribes for English, so the number is comparable to the
    published literature on that benchmark.

Both models are downloaded from HuggingFace on first use (whisper-large-v3 is
~3 GB) and cached. Everything here runs on CPU; the device is touched only if the
test has to generate the audio itself.

Usage — score a wav you already have::

    QWEN3_TTS_QA_WAV=/tmp/out.wav QWEN3_TTS_QA_TEXT="Good morning..." \\
      pytest -s models/demos/qwen3_tts/tests/test_qwen3_tts_voice_quality.py

Or let it generate one through the demo pipeline first (needs a device)::

    pytest -s models/demos/qwen3_tts/tests/test_qwen3_tts_voice_quality.py

Knobs: ``QWEN3_TTS_QA_SV``, ``QWEN3_TTS_QA_ASR`` (checkpoints),
``QWEN3_TTS_QA_SIM_MIN``, ``QWEN3_TTS_QA_WER_MAX`` (gates),
``QWEN3_TTS_QA_REF`` (reference clip), ``QWEN3_TTS_QA_SEED``.

On the gates
------------
``SIM_MIN`` defaults to **0.80**, which is a "did we break voice cloning" floor,
not a quality bar. Justification, measured on this reference clip with the default
checkpoint: eight generated clips across four seeds and both conv paths scored
0.8724 - 0.9488, while a +4-semitone pitch shift scored 0.2550 and white noise
0.5763. 0.80 sits clear of both. For context, an EER threshold derived from
LibriSpeech same-speaker pairs against this reference came out at 0.8895 — do not
assert that close to the observed range, a real clip landed at 0.8900.

An absolute quality claim needs a benchmark, not one clip: Seed-TTS-Eval test-en
is 1,088 utterances over ~1,000 reference speakers, and the Qwen3-TTS report
gives 1.24 WER / 0.775 SIM for this checkpoint. Those numbers are *not*
comparable to what this test prints — different reference speakers, and the
report never names its SV checkpoint. Treat this as a regression gate.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REF = REPO_ROOT / "models/demos/qwen3_tts/demo/jim_reference.wav"
DEFAULT_TEXT = (
    "Good morning. Today is a beautiful day for a walk in the park, with bright sun "
    "and a gentle breeze through the trees."
)
_SV_SR = 16000


def _env(name: str, default):
    v = os.environ.get(name)
    return default if v is None else type(default)(v) if not isinstance(default, str) else v


def _load(path, sr: int = _SV_SR) -> np.ndarray:
    import librosa

    return librosa.load(str(path), sr=sr, mono=True)[0]


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — standard WER scoring prep."""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


class _SpeakerVerifier:
    """WavLM x-vector speaker embeddings, loaded once per session."""

    def __init__(self, checkpoint: str):
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.fe = AutoFeatureExtractor.from_pretrained(checkpoint)
        self.model = WavLMForXVector.from_pretrained(checkpoint).eval()

    @torch.no_grad()
    def embed(self, wav: np.ndarray) -> torch.Tensor:
        inputs = self.fe(wav, sampling_rate=_SV_SR, return_tensors="pt", padding=True)
        return self.model(**inputs).embeddings[0].float()

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(torch.nn.functional.cosine_similarity(self.embed(a), self.embed(b), dim=0))


class _Transcriber:
    def __init__(self, checkpoint: str):
        from transformers import pipeline

        self.asr = pipeline("automatic-speech-recognition", model=checkpoint, device=-1)

    def __call__(self, wav: np.ndarray) -> str:
        # 16 kHz mono is fed directly; the pipeline resamples only if it has to, and
        # that path wants torchaudio, which this repo's venv does not carry.
        return self.asr({"raw": wav, "sampling_rate": _SV_SR})["text"]


def score_clip(generated_wav, target_text: str, reference_wav=DEFAULT_REF) -> Dict[str, object]:
    """SIM + WER for one generated clip, plus the anchors that make SIM readable.

    The anchors are the point: a bare cosine says nothing without knowing what this
    checkpoint gives for a true same-speaker pair and for a known non-match, so both
    are measured here on the caller's own reference clip.
    """
    import jiwer

    sv = _SpeakerVerifier(os.environ.get("QWEN3_TTS_QA_SV", "microsoft/wavlm-base-plus-sv"))
    asr = _Transcriber(os.environ.get("QWEN3_TTS_QA_ASR", "openai/whisper-large-v3"))

    ref, gen = _load(reference_wav), _load(generated_wav)
    sim = sv.similarity(gen, ref)

    hyp = _normalize(asr(gen))
    ref_text = _normalize(target_text)
    words = jiwer.process_words(ref_text, hyp)

    half = len(ref) // 2
    import librosa

    anchors = {
        "same speaker (reference, 1st vs 2nd half)": sv.similarity(ref[:half], ref[half:]),
        "non-match (reference pitched +4 semitones)": sv.similarity(
            librosa.effects.pitch_shift(ref, sr=_SV_SR, n_steps=4), ref
        ),
    }
    return {
        "sim": sim,
        "wer": words.wer,
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
        "hypothesis": hyp,
        "reference_text": ref_text,
        "duration_s": len(gen) / _SV_SR,
        "words_per_s": len(hyp.split()) / (len(gen) / _SV_SR),
        "anchors": anchors,
    }


def _generate(tmp_path) -> tuple:
    """Render the default prompt through the demo pipeline. Needs a device."""
    from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import _load_ref_text_for, run_full_ttnn_tts

    out = str(tmp_path / "voice_quality.wav")
    run_full_ttnn_tts(
        text=DEFAULT_TEXT,
        ref_audio=str(DEFAULT_REF),
        ref_text=_load_ref_text_for(str(DEFAULT_REF)),
        output_path=out,
        seed=int(os.environ.get("QWEN3_TTS_QA_SEED", "42")),
    )
    return out, DEFAULT_TEXT


def test_voice_quality(tmp_path):
    wav = os.environ.get("QWEN3_TTS_QA_WAV")
    if wav:
        text = os.environ.get("QWEN3_TTS_QA_TEXT", DEFAULT_TEXT)
        if not Path(wav).is_file():
            pytest.fail(f"QWEN3_TTS_QA_WAV does not exist: {wav}")
    else:
        wav, text = _generate(tmp_path)

    reference = os.environ.get("QWEN3_TTS_QA_REF", str(DEFAULT_REF))
    r = score_clip(wav, text, reference)

    sim_min = float(os.environ.get("QWEN3_TTS_QA_SIM_MIN", "0.80"))
    wer_max = float(os.environ.get("QWEN3_TTS_QA_WER_MAX", "0.10"))

    print(f"\n  clip        {wav}")
    print(f"  reference   {reference}")
    print(f"  duration    {r['duration_s']:.2f} s   {r['words_per_s']:.2f} words/s")
    print(f"  SIM         {r['sim']:.4f}   (gate > {sim_min:.2f})")
    for label, v in r["anchors"].items():
        print(f"                {v:.4f}   {label}")
    print(
        f"  WER         {r['wer']*100:.1f} %   (gate < {wer_max*100:.0f} %)   "
        f"sub {r['substitutions']} del {r['deletions']} ins {r['insertions']}"
    )
    print(f"  asked for   {r['reference_text']}")
    print(f"  heard       {r['hypothesis']}")

    assert r["sim"] > sim_min, (
        f"speaker similarity {r['sim']:.4f} <= {sim_min:.2f}: the generated voice no longer "
        f"matches the reference speaker (non-match anchor "
        f"{r['anchors']['non-match (reference pitched +4 semitones)']:.4f})"
    )
    assert r["wer"] < wer_max, (
        f"WER {r['wer']*100:.1f}% >= {wer_max*100:.0f}%: words are being dropped or garbled.\n"
        f"  asked for: {r['reference_text']}\n  heard    : {r['hypothesis']}"
    )
