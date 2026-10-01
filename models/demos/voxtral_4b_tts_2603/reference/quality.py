# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Signal-quality scores for a RENDERED waveform: intelligibility (WER) and naturalness (MOS).

PCC against the golden says two waveforms MATCH; it cannot say either is any good. These two
scores judge the audio itself, and the e2e test scores the HF golden the same way and requires the
TT output to be no worse by a stated margin -- the thresholds come from the reference, not from an
invented absolute number.

  * intelligibility: Whisper large-v3-turbo transcribes each clip; WER against the text the
    pipeline was asked to speak, both sides through Whisper's own English normalizer.
  * naturalness: UTMOS22 (strong), a no-reference MOS predictor, via SpeechMOS.

Host-side scoring of the OUTPUT. Nothing here is in, or imported by, the forward path.
"""
from __future__ import annotations

import hashlib
import importlib.machinery
import os
import sys
import types
from functools import lru_cache

import numpy as np
import torch

ASR_MODEL_ID = "openai/whisper-large-v3-turbo"
# Pinned: the scores these tests compare are only reproducible against one Whisper snapshot.
ASR_REVISION = os.environ.get("VOXTRAL_ASR_REVISION", "41f01f3fe87f28c78e2fbf8b568835947dd65ed9")
MOS_HUB_REPO = "tarepan/SpeechMOS:v1.2.0"
# UTMOS22 runs from a LOCAL copy of the SpeechMOS hub code with a checkpoint verified against this
# hash. Nothing is downloaded or executed from GitHub at test time unless VOXTRAL_ALLOW_REMOTE_MOS=1
# is set once to populate the torch hub cache (see the README).
MOS_HUB_DIR = "tarepan_SpeechMOS_v1.2.0"
MOS_CHECKPOINT = "utmos22_strong_step7459_v1.pt"
MOS_CHECKPOINT_URL = "https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/" + MOS_CHECKPOINT
MOS_CHECKPOINT_SHA256 = "38aa51ab79e2a4e09a1449758a4b37e9cbb2e8235a49662a732d33a9ba1e9bff"
SCORE_RATE = 16000


def resample(wave: torch.Tensor, orig_rate: int, new_rate: int = SCORE_RATE) -> np.ndarray:
    """Band-limited polyphase resample (scipy), mono float32."""
    from math import gcd

    from scipy.signal import resample_poly

    x = wave.detach().reshape(-1).to(torch.float64).numpy()
    if int(orig_rate) == int(new_rate):
        return x.astype(np.float32)
    g = gcd(int(orig_rate), int(new_rate))
    return resample_poly(x, int(new_rate) // g, int(orig_rate) // g).astype(np.float32)


@lru_cache(maxsize=1)
def _asr():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(ASR_MODEL_ID, revision=ASR_REVISION)
    model = WhisperForConditionalGeneration.from_pretrained(
        ASR_MODEL_ID, revision=ASR_REVISION, torch_dtype=torch.float32
    ).eval()
    return processor, model


def transcribe(waves, rate: int, batch_size: int = 8) -> list[str]:
    processor, model = _asr()
    audio = [resample(w, rate) for w in waves]
    out = []
    for i in range(0, len(audio), batch_size):
        chunk = audio[i : i + batch_size]
        feats = processor(chunk, sampling_rate=SCORE_RATE, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(feats.input_features, language="en", task="transcribe")
        out += processor.batch_decode(ids, skip_special_tokens=True)
    return [t.strip() for t in out]


def normalize(text: str) -> str:
    processor, _ = _asr()
    return processor.tokenizer.normalize(text)


def word_error_rate(hypothesis: str, reference: str) -> float:
    import jiwer

    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref.strip():
        raise ValueError("empty reference text")
    return float(jiwer.wer(ref, hyp if hyp.strip() else "<empty>"))


def _install_torchaudio_shim() -> None:
    """UTMOS imports torchaudio only to resample; this venv has none, and the clips arrive at 16 kHz."""
    if "torchaudio" in sys.modules:
        return
    try:
        import torchaudio  # noqa: F401

        return
    except ImportError:
        pass
    ta = types.ModuleType("torchaudio")
    fn = types.ModuleType("torchaudio.functional")
    ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    fn.__spec__ = importlib.machinery.ModuleSpec("torchaudio.functional", None)

    def _resample(wave, orig_freq, new_freq):
        if int(orig_freq) != int(new_freq):
            raise ValueError("the shim only passes 16 kHz through; resample before calling UTMOS")
        return wave

    fn.resample = _resample
    ta.functional = fn
    sys.modules["torchaudio"] = ta
    sys.modules["torchaudio.functional"] = fn


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=1)
def _mos():
    """UTMOS22 (strong) from the local torch hub cache, its checkpoint checked against a pinned hash."""
    _install_torchaudio_shim()
    hub = torch.hub.get_dir()
    code = os.path.join(hub, MOS_HUB_DIR)
    ckpt = os.path.join(hub, "checkpoints", MOS_CHECKPOINT)
    remote_ok = os.environ.get("VOXTRAL_ALLOW_REMOTE_MOS") == "1"
    if not (os.path.isdir(code) and os.path.isfile(ckpt)):
        if not remote_ok:
            raise RuntimeError(
                f"UTMOS22 is not cached under {hub} ({MOS_HUB_DIR}/, checkpoints/{MOS_CHECKPOINT}). Run once with "
                f"VOXTRAL_ALLOW_REMOTE_MOS=1 to fetch {MOS_HUB_REPO} and its checkpoint (this executes that "
                "repository's hub code), or copy both into the cache."
            )
        if not os.path.isdir(code):
            torch.hub.load(MOS_HUB_REPO, "utmos22_strong", trust_repo=True, pretrained=False)
        if not os.path.isfile(ckpt):
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            torch.hub.download_url_to_file(MOS_CHECKPOINT_URL, ckpt, hash_prefix=MOS_CHECKPOINT_SHA256[:16])
    if _sha256(ckpt) != MOS_CHECKPOINT_SHA256:
        raise RuntimeError(f"{ckpt} does not match the pinned UTMOS22 checkpoint hash {MOS_CHECKPOINT_SHA256}")
    model = torch.hub.load(code, "utmos22_strong", source="local", pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return model.eval()


def mos(waves, rate: int) -> list[float]:
    predictor = _mos()
    scores = []
    for w in waves:
        x = torch.from_numpy(resample(w, rate)).unsqueeze(0)
        with torch.no_grad():
            scores.append(float(predictor(x, SCORE_RATE).reshape(-1)[0]))
    return scores


def score(waves, rate: int, texts) -> dict:
    """Per-clip transcript, WER and MOS, plus corpus WER (errors summed over all clips)."""
    import jiwer

    hyps = transcribe(waves, rate)
    wers = [word_error_rate(h, t) for h, t in zip(hyps, texts)]
    refs_n = [normalize(t) for t in texts]
    hyps_n = [normalize(h) if normalize(h).strip() else "<empty>" for h in hyps]
    corpus = float(jiwer.wer(refs_n, hyps_n))
    return {"transcripts": hyps, "wer": wers, "corpus_wer": corpus, "mos": mos(waves, rate)}
