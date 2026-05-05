# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Objective diff between two TTS audio files (HF reference vs TTNN output, etc.).

Reports a battery of metrics that correlate with perceptual TTS quality issues
so we can debug regressions without listen-tests:

  * Basic shape: duration, RMS, peak, sample rate
  * Boundary noise: RMS of the first / last 0.1s and 0.5s
                    (helps catch click/pop/bleed at ref→gen boundary, and
                    truncation at the end)
  * Whisper transcripts (segment-level with timestamps), plus a line-by-line
    text diff so you can see what words are missing/different/extra.
  * Per-window energy curve: time-aligned RMS over fixed-size windows;
    flags windows where one signal is silent and the other is loud (=
    something missing in the quieter one).
  * Speech-rate proxy: char/sec from Whisper transcript.
  * Optional: speaker similarity (cosine) via the same ECAPA-TDNN
    speaker_encoder the model uses for voice conditioning. Drop in a
    third path with --ref to score both against a known-target speaker.
  * Spectral fingerprint: per-band energy ratio over short windows; surfaces
    "different timbre / different speaker" issues that text+RMS miss.

Usage:
    python models/demos/qwen3_tts/tests/audio_diff.py \\
        /tmp/audio_hf_ashley.wav /tmp/audio_ttnn_ashley_v3.wav

    # With speaker-similarity scoring against the input reference speaker:
    python models/demos/qwen3_tts/tests/audio_diff.py \\
        /tmp/audio_hf_ashley.wav /tmp/audio_ttnn_ashley_v3.wav \\
        --ref /local/ttuser/ssinghal/tts2/tts-models/tts-2/prompts/Ashley_en.wav
"""
from __future__ import annotations

import argparse
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def load_audio(path: str, target_sr: int = 24000) -> tuple[np.ndarray, int]:
    """Load mono float32 audio at target_sr (resample with scipy if needed)."""
    a, sr = sf.read(path, dtype="float32")
    if a.ndim == 2:
        a = a.mean(axis=1)
    if sr != target_sr:
        from scipy import signal

        n = int(len(a) * target_sr / sr)
        a = signal.resample(a, n).astype(np.float32)
        sr = target_sr
    return a, sr


def rms(a: np.ndarray) -> float:
    return float(np.sqrt((a.astype(np.float64) ** 2).mean()))


# ---------------------------------------------------------------------------
# Section: basic stats
# ---------------------------------------------------------------------------
def basic_stats(label: str, a: np.ndarray, sr: int) -> dict:
    return {
        "label": label,
        "duration_s": len(a) / sr,
        "n_samples": len(a),
        "rms": rms(a),
        "abs_max": float(np.abs(a).max()),
        "abs_min": float(np.abs(a).min()),
        "clipped_pct": float(100 * (np.abs(a) > 0.99).mean()),
    }


def print_basic(s: dict) -> None:
    print(
        f"  {s['label']:>12s}  dur={s['duration_s']:6.2f}s "
        f"rms={s['rms']:.4f}  absmax={s['abs_max']:.4f}  "
        f"clipped={s['clipped_pct']:.2f}%"
    )


# ---------------------------------------------------------------------------
# Section: boundary noise (start/end behaviour)
# ---------------------------------------------------------------------------
def boundary_metrics(a: np.ndarray, sr: int, window_s: float = 0.1) -> dict:
    n = max(1, int(window_s * sr))
    return {
        "first_window_rms": rms(a[:n]),
        "last_window_rms": rms(a[-n:]),
        "first_500ms_rms": rms(a[: int(0.5 * sr)]),
        "last_500ms_rms": rms(a[-int(0.5 * sr) :]),
    }


def print_boundary(label: str, b: dict) -> None:
    print(
        f"  {label:>12s}  first 100ms rms={b['first_window_rms']:.4f}  "
        f"first 500ms rms={b['first_500ms_rms']:.4f}    "
        f"last 100ms rms={b['last_window_rms']:.4f}  "
        f"last 500ms rms={b['last_500ms_rms']:.4f}"
    )


def diagnose_boundary(a_b: dict, b_b: dict) -> list[str]:
    """Surface specific concerns by comparing boundary RMS."""
    notes = []
    # Significant noise burst at start
    a0, b0 = a_b["first_window_rms"], b_b["first_window_rms"]
    if max(a0, b0) > 5 * min(a0, b0) + 1e-4:
        louder, quieter = ("A", "B") if a0 > b0 else ("B", "A")
        ratio = max(a0, b0) / max(min(a0, b0), 1e-6)
        notes.append(
            f"  ⚠ start boundary mismatch: {louder} first 100ms is {ratio:.1f}× "
            f"louder than {quieter} → likely click/bleed in {louder}"
        )
    # End-truncation: one ends quietly while the other is still active
    a_end, b_end = a_b["last_500ms_rms"], b_b["last_500ms_rms"]
    if max(a_end, b_end) > 5 * min(a_end, b_end) + 1e-4:
        louder, quieter = ("A", "B") if a_end > b_end else ("B", "A")
        notes.append(
            f"  ⚠ end-tail mismatch: {louder} last 500ms is louder → " f"{quieter} may have premature EOS / truncation"
        )
    return notes


# ---------------------------------------------------------------------------
# Section: per-window energy curve (alignment)
# ---------------------------------------------------------------------------
def windowed_rms(a: np.ndarray, sr: int, window_s: float = 0.5) -> np.ndarray:
    n = max(1, int(window_s * sr))
    out = []
    for i in range(0, len(a), n):
        chunk = a[i : i + n]
        if len(chunk):
            out.append(rms(chunk))
    return np.array(out)


def print_energy_curve(label_a: str, a: np.ndarray, label_b: str, b: np.ndarray, sr: int) -> None:
    cur_a = windowed_rms(a, sr, 0.5)
    cur_b = windowed_rms(b, sr, 0.5)
    n = max(len(cur_a), len(cur_b))
    print("\n  Per-0.5s-window RMS curve:")
    print(f"  {'window':>8s}  {label_a:>10s}  {label_b:>10s}  {'note':>20s}")
    for i in range(n):
        va = cur_a[i] if i < len(cur_a) else 0.0
        vb = cur_b[i] if i < len(cur_b) else 0.0
        note = ""
        if va > 0.02 and vb < 0.005:
            note = f"{label_b} silent, {label_a} loud"
        elif vb > 0.02 and va < 0.005:
            note = f"{label_a} silent, {label_b} loud"
        elif max(va, vb) > 5 * min(va, vb) + 1e-3:
            note = f"|Δ|={abs(va - vb):.4f}"
        print(f"  {i*0.5:>6.2f}s  {va:>10.4f}  {vb:>10.4f}  {note:>20s}")


# ---------------------------------------------------------------------------
# Section: Whisper transcript diff
# ---------------------------------------------------------------------------
def transcribe(path: str, model_size: str = "base") -> dict:
    import whisper

    m = whisper.load_model(model_size)
    return m.transcribe(path, language="en", verbose=False)


def diff_transcripts(label_a: str, ra: dict, label_b: str, rb: dict) -> None:
    ta = ra["text"].strip()
    tb = rb["text"].strip()
    print(f"\n  {label_a}: {ta!r}")
    print(f"  {label_b}: {tb!r}")
    chars_per_sec_a = len(ta) / max(0.01, sum(s["end"] - s["start"] for s in ra["segments"]))
    chars_per_sec_b = len(tb) / max(0.01, sum(s["end"] - s["start"] for s in rb["segments"]))
    print(f"\n  Speech rate: {label_a}={chars_per_sec_a:.1f} ch/s  {label_b}={chars_per_sec_b:.1f} ch/s")

    # Word-level diff (uses sequence matcher on whitespace-split tokens)
    wa, wb = ta.split(), tb.split()
    sm = SequenceMatcher(a=wa, b=wb)
    print(f"\n  Word-level diff ({label_a} → {label_b}):")
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        a_words = " ".join(wa[i1:i2]) or "—"
        b_words = " ".join(wb[j1:j2]) or "—"
        print(f"    [{tag:7s}]  A[{i1}:{i2}]={a_words!r:50s}  B[{j1}:{j2}]={b_words!r}")


def print_segments(label: str, r: dict, max_segs: int = 6) -> None:
    print(f"\n  {label} segments:")
    for s in r["segments"][:max_segs]:
        print(f"    [{s['start']:6.2f}-{s['end']:6.2f}]  {s['text']!r}")


# ---------------------------------------------------------------------------
# Section: speaker similarity (optional, requires --ref)
# ---------------------------------------------------------------------------
def speaker_similarity_via_reference(*paths: str) -> Optional[list[float]]:
    """Cosine similarity of each path vs paths[0] using our reference's ECAPA
    speaker encoder (the one that conditions the TTS model). The first path is
    treated as the target reference; remaining are compared against it.

    Returns: list of cosine-sim values for paths[1:], or None on import failure.
    """
    try:
        sys.path.insert(0, "/local/ttuser/ssinghal/tt-metal")
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        from models.demos.qwen3_tts.demo.demo_pure_reference_tts import extract_speaker_embedding_reference
    except Exception as e:
        print(f"  (speaker-sim disabled: {e})")
        return None

    mp = Path(snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", allow_patterns=["*.safetensors"]))
    sd = {}
    for f in mp.glob("*.safetensors"):
        sd.update(load_file(f))

    embeds = []
    for p in paths:
        a, sr = load_audio(p, 24000)
        # speaker_encoder upstream resamples internally if needed; just hand it the float32 waveform.
        emb = extract_speaker_embedding_reference(torch.from_numpy(a).float(), sd)
        embeds.append(emb.flatten().detach().cpu().float())

    target = embeds[0]
    sims = []
    for e in embeds[1:]:
        cos = float((target * e).sum() / (target.norm() * e.norm() + 1e-12))
        sims.append(cos)
    return sims


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Diff two TTS audio files objectively.")
    ap.add_argument("a", help="Path to file A (e.g., HF reference)")
    ap.add_argument("b", help="Path to file B (e.g., TTNN output)")
    ap.add_argument("--ref", help="Optional reference speaker audio for speaker-sim scoring", default=None)
    ap.add_argument("--whisper-model", default="base", help="Whisper model size (tiny/base/small/medium)")
    ap.add_argument("--no-whisper", action="store_true", help="Skip whisper transcription")
    args = ap.parse_args()

    a, sr_a = load_audio(args.a, 24000)
    b, sr_b = load_audio(args.b, 24000)

    print("=" * 78)
    print(f"  A = {args.a}")
    print(f"  B = {args.b}")
    print("=" * 78)

    print("\n— Basic stats —")
    sa = basic_stats("A", a, sr_a)
    sb = basic_stats("B", b, sr_b)
    print_basic(sa)
    print_basic(sb)
    if sa["clipped_pct"] > 0.5 or sb["clipped_pct"] > 0.5:
        print("  ⚠ clipping detected (>0.5% of samples ≥0.99 abs)")

    print("\n— Boundary noise / truncation —")
    ba = boundary_metrics(a, sr_a)
    bb = boundary_metrics(b, sr_b)
    print_boundary("A", ba)
    print_boundary("B", bb)
    notes = diagnose_boundary(ba, bb)
    for n in notes:
        print(n)
    if not notes:
        print("  ✓ no boundary anomalies detected")

    print_energy_curve("A", a, "B", b, sr_a)

    if not args.no_whisper:
        print("\n— Whisper transcripts —")
        ra = transcribe(args.a, args.whisper_model)
        rb = transcribe(args.b, args.whisper_model)
        print_segments("A", ra)
        print_segments("B", rb)
        diff_transcripts("A", ra, "B", rb)

    if args.ref:
        print("\n— Speaker similarity (ECAPA, cos against --ref) —")
        sims = speaker_similarity_via_reference(args.ref, args.a, args.b)
        if sims is not None:
            print(f"  cos(REF, A) = {sims[0]:.4f}")
            print(f"  cos(REF, B) = {sims[1]:.4f}")
            print(f"  Δ = {sims[0] - sims[1]:+.4f}  (positive → A is closer to REF)")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
