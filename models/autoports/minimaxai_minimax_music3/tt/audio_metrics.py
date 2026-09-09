# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side evaluation metrics for generated MiniMax-Music3 audio (numpy / librosa, no device).

* :func:`log_mel_distance` - the spectral distance the stage-06 tests use to compare a device wav with the
  golden wav: root-mean-square difference, in dB, of the two log-mel spectrograms (128 mel bands, 2048-point
  FFT, 512-sample hop, both channels), evaluated over the common length. A floor of ``-80 dB`` relative to
  the louder spectrogram's peak keeps digital silence from dominating the distance.
* :func:`audio_stats` - the descriptive statistics the qualitative check reports: RMS, peak, silence fraction
  (1-second windows below -60 dBFS), stereo correlation, band energy over time.
* :func:`code_stats` - repetition statistics of the AR stage's semantic codes (degenerate-loop detector).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

N_FFT = 2048
HOP = 512
N_MELS = 128
FLOOR_DB = -80.0
BANDS_HZ = ((0, 250), (250, 2000), (2000, 8000), (8000, 22050))
SILENCE_DBFS = -60.0


def _as_stereo(x) -> np.ndarray:
    """``[2, S]`` float64 from ``[2, S]`` / ``[1, 2, S]`` / ``[S, 2]`` arrays or tensors."""
    a = np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x, dtype=np.float64)
    a = np.squeeze(a)
    if a.ndim == 1:
        a = np.stack([a, a])
    if a.shape[0] != 2 and a.shape[-1] == 2:
        a = a.T
    assert a.ndim == 2 and a.shape[0] == 2, a.shape
    return a


def log_mel(x, sr: int = 44100) -> np.ndarray:
    """``[2, n_mels, frames]`` log-power mel spectrogram in dB (``10 log10``), un-floored."""
    import librosa

    a = _as_stereo(x)
    mels = [
        librosa.feature.melspectrogram(
            y=ch.astype(np.float32), sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0
        )
        for ch in a
    ]
    return 10.0 * np.log10(np.stack(mels) + 1e-12)


def log_mel_distance(a, b, sr: int = 44100, floor_db: float = FLOOR_DB) -> Dict[str, float]:
    """RMS and mean absolute difference (dB) between the floored log-mel spectrograms of ``a`` and ``b``."""
    a2, b2 = _as_stereo(a), _as_stereo(b)
    n = min(a2.shape[-1], b2.shape[-1])
    la, lb = log_mel(a2[:, :n], sr), log_mel(b2[:, :n], sr)
    floor = max(la.max(), lb.max()) + floor_db
    la, lb = np.maximum(la, floor), np.maximum(lb, floor)
    diff = la - lb
    return {
        "rms_db": float(np.sqrt(np.mean(diff**2))),
        "mean_abs_db": float(np.mean(np.abs(diff))),
        "max_abs_db": float(np.max(np.abs(diff))),
        "frames": int(la.shape[-1]),
        "length_mismatch_samples": int(abs(a2.shape[-1] - b2.shape[-1])),
    }


def audio_stats(x, sr: int = 44100, window_s: float = 1.0) -> Dict[str, object]:
    """Descriptive statistics of a stereo clip (see the module docstring)."""
    a = _as_stereo(x)
    samples = a.shape[-1]
    mono = a.mean(axis=0)
    rms = float(np.sqrt(np.mean(a**2)))
    peak = float(np.abs(a).max())
    win = int(round(window_s * sr))
    n_win = max(1, samples // win)
    win_rms = np.array([np.sqrt(np.mean(a[:, i * win : (i + 1) * win] ** 2)) for i in range(n_win)])
    win_db = 20.0 * np.log10(win_rms + 1e-12)
    silence_fraction = float(np.mean(win_db < SILENCE_DBFS))
    if a[0].std() > 0 and a[1].std() > 0:
        stereo_corr = float(np.corrcoef(a[0], a[1])[0, 1])
    else:
        stereo_corr = float("nan")
    # Band energy per window (fraction of the window's spectral power in each band), on the mono mix.
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    band_energy: List[List[float]] = []
    for i in range(n_win):
        spec = np.abs(np.fft.rfft(mono[i * win : (i + 1) * win])) ** 2
        total = spec.sum() + 1e-20
        band_energy.append([float(spec[(freqs >= lo) & (freqs < hi)].sum() / total) for lo, hi in BANDS_HZ])
    be = np.array(band_energy)
    # Spectral flatness of the mono mix (1.0 = white noise, ~0 = pure tones), per window
    flatness = []
    for i in range(n_win):
        spec = np.abs(np.fft.rfft(mono[i * win : (i + 1) * win])) ** 2 + 1e-20
        flatness.append(float(np.exp(np.mean(np.log(spec))) / np.mean(spec)))
    # Window-to-window similarity of the log-mel spectrum (repetition / degenerate-loop indicator)
    lm = log_mel(a, sr).mean(axis=0)  # [n_mels, frames]
    frames_per_win = max(1, int(round(win / HOP)))
    n_lm = lm.shape[-1] // frames_per_win
    win_mel = np.array([lm[:, i * frames_per_win : (i + 1) * frames_per_win].mean(axis=1) for i in range(n_lm)])
    adjacent_corr = []
    for i in range(1, n_lm):
        if win_mel[i].std() > 0 and win_mel[i - 1].std() > 0:
            adjacent_corr.append(float(np.corrcoef(win_mel[i], win_mel[i - 1])[0, 1]))
    return {
        "samples": int(samples),
        "seconds": float(samples / sr),
        "rms": rms,
        "rms_dbfs": float(20.0 * np.log10(rms + 1e-12)),
        "peak": peak,
        "clipped_fraction": float(np.mean(np.abs(a) >= 0.999)),
        "nan_count": int(np.isnan(a).sum()),
        "silence_fraction_1s": silence_fraction,
        "window_rms_dbfs": [float(v) for v in win_db],
        "stereo_correlation": stereo_corr,
        "band_hz": [list(b) for b in BANDS_HZ],
        "band_energy_mean": [float(v) for v in be.mean(axis=0)],
        "band_energy_per_window": [[round(float(v), 4) for v in row] for row in be],
        "spectral_flatness_mean": float(np.mean(flatness)),
        "adjacent_window_logmel_corr_mean": float(np.mean(adjacent_corr)) if adjacent_corr else float("nan"),
        "adjacent_window_logmel_corr_max": float(np.max(adjacent_corr)) if adjacent_corr else float("nan"),
    }


def code_stats(codes, top_n: int = 5) -> Dict[str, object]:
    """Repetition statistics of ``[F, 8]`` frame codes (column 0 = semantic code)."""
    c = np.asarray(codes.detach().cpu().numpy() if hasattr(codes, "detach") else codes)
    if c.ndim == 1:
        c = c[:, None]
    sem = c[:, 0]
    frames = len(sem)
    values, counts = np.unique(sem, return_counts=True)
    order = np.argsort(-counts)
    repeats = int(np.sum(sem[1:] == sem[:-1])) if frames > 1 else 0
    # longest run of an identical semantic code
    longest = run = 1
    for i in range(1, frames):
        run = run + 1 if sem[i] == sem[i - 1] else 1
        longest = max(longest, run)
    # fraction of frames whose full 8-code tuple repeats the previous frame
    full_repeat = int(np.sum(np.all(c[1:] == c[:-1], axis=1))) if frames > 1 else 0
    # n-gram repetition: fraction of 4-grams of semantic codes seen before
    ngram = 4
    seen, dup = set(), 0
    for i in range(frames - ngram + 1):
        key = tuple(sem[i : i + ngram].tolist())
        dup += key in seen
        seen.add(key)
    n_grams = max(1, frames - ngram + 1)
    return {
        "frames": int(frames),
        "distinct_semantic_codes": int(len(values)),
        "most_common_semantic_share": float(counts[order[0]] / frames) if frames else 0.0,
        "top_semantic_codes": [[int(values[i]), int(counts[i])] for i in order[:top_n]],
        "adjacent_semantic_repeat_rate": float(repeats / max(1, frames - 1)),
        "longest_semantic_run": int(longest),
        "adjacent_full_frame_repeat_rate": float(full_repeat / max(1, frames - 1)),
        "repeated_4gram_rate": float(dup / n_grams),
        "residual_distinct_per_codebook": [int(len(np.unique(c[:, k]))) for k in range(1, c.shape[1])],
    }


def summarize_for_log(stats: Dict[str, object]) -> str:
    """One line per key that is not a long list (for loguru / README tables)."""
    keep = {k: v for k, v in stats.items() if not (isinstance(v, list) and len(v) > 8)}
    return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in keep.items())
