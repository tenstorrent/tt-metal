# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Audio test inputs (real speech from the repo, plus one labelled synthetic tone) and helpers.

Every speech case is cut from, or concatenated out of, the 16 kHz mono speech clips the Whisper
demo ships in-tree (`models/demos/audio/whisper/demo/dataset/conditional_generation/*.wav`,
4.0 - 7.6 s each). No in-tree clip is >= 10 s, so the 10 s and 30 s cases concatenate whole clips
in sorted order and cut at the target duration. Features come from the checkpoint's HF feature
extractor (Gemma4AudioFeatureExtractor, 128 log-mel bins, 100 frames / s).

    case        duration                         frames  tokens  notes
    short       0.64 s (clip 1, 0.50-1.14 s)      63      16      < 1 s, 16 % 12 != 0
    s2          2.00 s (clip 2, 1.00-3.00 s)      199     50      the recon smoke length
    s10         10.00 s (clips 1-2, cut)          999     250     ~10 s
    s30         30.00 s (clips 1-5, cut)          2999    750     the maximum (750 tokens)
    b2_s10_s2   batch of s10 and s2              999     250, 50 padded batch, 2 lengths
    tone3       3.00 s synthetic (440 Hz + chirp) 299     75      SYNTHETIC, extra case only
(frames = processor output incl. its semicausal padding; tokens = ceil(ceil(frames / 2) / 2))

`crop_case(name, tokens)` takes the first 4 * tokens feature frames of a case: a valid processor
input for lengths around the 12-row attention chunk (11, 12, 13, 24, 25, ...).
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import torch

import ttnn

_ROOT = Path(__file__).resolve().parents[6]  # tt-metal checkout
SPEECH_DIR = "models/demos/audio/whisper/demo/dataset/conditional_generation"
SR = 16000


def _speech_files() -> list[Path]:
    files = sorted((_ROOT / SPEECH_DIR).glob("*.wav"))
    if len(files) < 5:
        raise FileNotFoundError(f"expected >= 5 speech clips under {SPEECH_DIR}, found {len(files)}")
    return files


@functools.lru_cache(maxsize=None)
def _read(path: str) -> np.ndarray:
    import soundfile as sf

    w, sr = sf.read(path, dtype="float32")
    if sr != SR or w.ndim != 1:
        raise ValueError(f"{path}: expected 16 kHz mono, got sr={sr} shape={w.shape}")
    return w


def _concat(seconds: float) -> np.ndarray:
    out, n = [], int(round(seconds * SR))
    for f in _speech_files():
        out.append(_read(str(f)))
        if sum(len(x) for x in out) >= n:
            return np.concatenate(out)[:n]
    raise ValueError(f"in-tree speech totals less than {seconds} s")


def waveform(name: str) -> np.ndarray:
    files = _speech_files()
    if name == "short":
        return _read(str(files[0]))[int(0.50 * SR) : int(1.14 * SR)]
    if name == "s2":
        return _read(str(files[1]))[int(1.00 * SR) : int(3.00 * SR)]
    if name == "s10":
        return _concat(10.0)
    if name == "s30":
        return _concat(30.0)
    if name == "tone3":  # SYNTHETIC: 440 Hz tone + 200 -> 4000 Hz linear chirp, 3 s
        t = np.arange(3 * SR) / SR
        chirp = np.sin(2 * np.pi * (200 * t + (4000 - 200) / (2 * 3.0) * t**2))
        return (0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * chirp).astype(np.float32)
    raise KeyError(name)


CASES = {
    "short": ["short"],
    "s2": ["s2"],
    "s10": ["s10"],
    "s30": ["s30"],
    "b2_s10_s2": ["s10", "s2"],
    "tone3": ["tone3"],
}
ALL_CASES = tuple(CASES)
SPEECH_CASES = tuple(c for c in CASES if c != "tone3")
# per-layer teacher-forced tests: shortest, longest (max bucket) and the padded batch
LAYER_CASES = ("short", "s30", "b2_s10_s2")
SUBLAYERS = (0, 5, 11)  # layers whose FFN / attention / lconv in-outs are recorded on LAYER_CASES
# token lengths around the 12-query chunk (first / last block, not divisible by 12), cut from s30
CROP_TOKENS = (1, 11, 12, 13, 24, 25, 37, 100)


@functools.lru_cache(maxsize=1)
def feature_extractor(snapshot: str):
    from transformers import AutoFeatureExtractor

    return AutoFeatureExtractor.from_pretrained(snapshot)


@functools.lru_cache(maxsize=16)
def load_case(snapshot: str, case: str) -> tuple[torch.Tensor, torch.Tensor]:
    """(input_features [B, T, 128] fp32, input_features_mask [B, T] bool) from the HF feature extractor."""
    fe = feature_extractor(snapshot)
    out = fe([waveform(n) for n in CASES[case]], sampling_rate=SR, return_tensors="pt")
    return out["input_features"].float(), out["input_features_mask"].bool()


def crop_case(snapshot: str, tokens: int, case: str = "s30") -> tuple[torch.Tensor, torch.Tensor]:
    feats, mask = load_case(snapshot, case)
    T = 4 * tokens
    return feats[:, :T].clone(), mask[:, :T].clone()


def device_inputs(encoder, features: torch.Tensor, mask: torch.Tensor):
    """Upload one case's inputs and prepare the conv weights for its shape: (host inputs, device inputs)."""
    host = encoder.prepare_inputs(features, mask)
    x = encoder.to_device(host)
    encoder.prepare(x)
    return host, x


def valid(ref_rows: torch.Tensor, tt_rows: torch.Tensor, token_mask: torch.Tensor):
    """Valid-token rows of [B, N, D] reference / TT tensors, concatenated to [n, D]."""
    return ref_rows[token_mask], tt_rows[token_mask]


def upload_rows(device, x: torch.Tensor, S: int, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """[B, N, D] torch -> [B, 1, S, D] TILE device tensor, zero padded to S rows."""
    B, N, D = x.shape
    xp = torch.zeros((B, 1, S, D), dtype=torch.float32)
    xp[:, 0, :N] = x.float()
    return ttnn.from_torch(xp, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def download_rows(t: ttnn.Tensor, N: int) -> torch.Tensor:
    """[B, 1, S, D] device -> [B, N, D] float32."""
    x = ttnn.to_torch(t).float()
    return x.reshape(x.shape[0], x.shape[-2], x.shape[-1])[:, :N]
