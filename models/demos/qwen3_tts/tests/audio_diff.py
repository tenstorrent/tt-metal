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
  * Per-AR-frame analysis: at 12.5 fps each codec frame produces ~80 ms /
    1920 samples. Aligning to that grid gives a frame-by-frame view of:
      - per-frame RMS, spectral centroid, zero-crossing rate
      - per-frame transient/click detection (peak sample-derivative)
      - top-K "anomaly frames" where A and B differ most in audio energy
      - if codec tokens are provided (--codes-a / --codes-b), per-frame
        token diff showing exactly which AR step diverged and what the
        audio looks like at that step
  * Spectral fingerprint: per-band energy ratio over short windows; surfaces
    "different timbre / different speaker" issues that text+RMS miss.

Usage:
    python models/demos/qwen3_tts/tests/audio_diff.py \\
        /tmp/audio_hf_ashley.wav /tmp/audio_ttnn_ashley_v3.wav

    # With per-AR-frame view + codec-token diff:
    python models/demos/qwen3_tts/tests/audio_diff.py \\
        /tmp/audio_hf_ashley.wav /tmp/audio_ttnn_ashley_v3.wav \\
        --codes-a /tmp/hf_ashley_codes.pt --codes-b /tmp/last_generated_codes.pt

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
# Section: per-AR-frame analysis (frame_idx ↔ audio window ↔ codec token)
# ---------------------------------------------------------------------------
# At 12.5 fps the codec emits one frame every 1920 samples (24kHz). The
# generated audio (after the HF-style ref-cut) starts exactly at gen frame 0
# of the model's output, so frame i covers samples [i*1920, (i+1)*1920).
SAMPLES_PER_FRAME = 1920


def per_frame_metrics(a: np.ndarray, sr: int) -> dict:
    """For each codec frame's worth of samples, compute RMS, peak |Δ|
    (transient detector), zero-crossing rate, and a coarse spectral centroid.
    Returns arrays of shape [n_frames]."""
    spf = SAMPLES_PER_FRAME
    n_frames = len(a) // spf
    rms_arr = np.zeros(n_frames, dtype=np.float64)
    peak_diff = np.zeros(n_frames, dtype=np.float64)
    zcr = np.zeros(n_frames, dtype=np.float64)
    centroid = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        chunk = a[i * spf : (i + 1) * spf].astype(np.float64)
        rms_arr[i] = np.sqrt((chunk**2).mean())
        if len(chunk) > 1:
            peak_diff[i] = float(np.abs(np.diff(chunk)).max())  # transient detector
            zcr[i] = float(((chunk[:-1] * chunk[1:]) < 0).mean())
        # Coarse spectral centroid via FFT
        spec = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sr)
        if spec.sum() > 1e-9:
            centroid[i] = float((freqs * spec).sum() / spec.sum())
    return {
        "n_frames": n_frames,
        "rms": rms_arr,
        "peak_diff": peak_diff,
        "zcr": zcr,
        "centroid": centroid,
    }


def detect_click_frames(metrics: dict, peak_diff_z: float = 4.0) -> list[int]:
    """Return frame indices flagged as containing a transient/click.

    A frame is flagged when its peak |Δ| (max sample-to-sample jump) is more
    than `peak_diff_z` standard deviations above the rolling median of its
    neighbours — a classic click-detector heuristic.
    """
    pd = metrics["peak_diff"]
    if len(pd) < 5:
        return []
    flagged = []
    for i in range(len(pd)):
        lo = max(0, i - 5)
        hi = min(len(pd), i + 6)
        nbrs = np.concatenate([pd[lo:i], pd[i + 1 : hi]])
        if len(nbrs) < 3:
            continue
        med = float(np.median(nbrs))
        mad = float(np.median(np.abs(nbrs - med))) + 1e-9
        # robust z-score; flag if current frame is way above neighbours
        if (pd[i] - med) / (1.4826 * mad) > peak_diff_z:
            flagged.append(i)
    return flagged


def print_per_frame_section(label_a: str, ma: dict, label_b: str, mb: dict, n_show_top: int = 12) -> None:
    """Show per-frame anomalies: clicks, biggest A-vs-B disagreements."""
    print("\n— Per-AR-frame analysis (12.5 fps, 1 frame = 80 ms / 1920 samples) —")
    print(f"  {label_a}: {ma['n_frames']} frames    {label_b}: {mb['n_frames']} frames")

    clicks_a = detect_click_frames(ma)
    clicks_b = detect_click_frames(mb)
    print(f"\n  Click-frame candidates ({label_a}): {clicks_a if clicks_a else 'none'}")
    print(f"  Click-frame candidates ({label_b}): {clicks_b if clicks_b else 'none'}")

    if clicks_a:
        print(f"\n  {label_a} flagged frames detail:")
        _print_frame_rows(label_a, ma, clicks_a)
    if clicks_b:
        print(f"\n  {label_b} flagged frames detail:")
        _print_frame_rows(label_b, mb, clicks_b)

    # Top-K frames where A and B differ most (only over their overlap)
    n_overlap = min(ma["n_frames"], mb["n_frames"])
    rms_diff = np.abs(ma["rms"][:n_overlap] - mb["rms"][:n_overlap])
    centroid_diff = np.abs(ma["centroid"][:n_overlap] - mb["centroid"][:n_overlap])
    top_rms = np.argsort(-rms_diff)[:n_show_top]
    print(f"\n  Top-{n_show_top} frames by |RMS_A - RMS_B|:")
    print(
        f"  {'frame':>6s} {'t (s)':>8s} {f'rms_{label_a}':>10s} {f'rms_{label_b}':>10s} {'|Δ|rms':>10s} "
        f"{'|Δ|centroid Hz':>16s}"
    )
    for fi in top_rms:
        t = fi * SAMPLES_PER_FRAME / 24000
        print(
            f"  {fi:>6d} {t:>8.2f} {ma['rms'][fi]:>10.4f} {mb['rms'][fi]:>10.4f} "
            f"{rms_diff[fi]:>10.4f} {centroid_diff[fi]:>16.0f}"
        )


def _print_frame_rows(label: str, m: dict, frames: list[int]) -> None:
    print(f"  {'frame':>6s} {'t (s)':>8s} {'rms':>9s} {'peak|Δ|':>9s} {'zcr':>8s} {'centroidHz':>12s}")
    for fi in frames:
        t = fi * SAMPLES_PER_FRAME / 24000
        print(
            f"  {fi:>6d} {t:>8.2f} {m['rms'][fi]:>9.4f} {m['peak_diff'][fi]:>9.4f} "
            f"{m['zcr'][fi]:>8.4f} {m['centroid'][fi]:>12.0f}"
        )


# ---------------------------------------------------------------------------
# Section: per-frame codec-token diff (when codes are provided)
# ---------------------------------------------------------------------------
def per_frame_codes_diff(
    label_a: str, codes_a: torch.Tensor, label_b: str, codes_b: torch.Tensor, ma: dict, mb: dict, max_show: int = 30
) -> None:
    """Walk the codec-token sequence and show where A and B diverge,
    correlated with per-frame audio metrics so we can see the audio
    signature of the divergent tokens.

    `codes_a`/`codes_b`: shape [seq, num_codebooks]; we report code 0 (the
    talker's primary token) since it drives sampling.
    """
    n = min(codes_a.shape[0], codes_b.shape[0])
    a0 = codes_a[:n, 0].long().tolist()
    b0 = codes_b[:n, 0].long().tolist()

    # First divergence
    first_diff = None
    for i in range(n):
        if a0[i] != b0[i]:
            first_diff = i
            break

    if first_diff is None:
        print(f"\n— Per-frame codec-token diff —\n  {label_a} and {label_b} match for first {n} frames.")
        return

    print(f"\n— Per-frame codec-token diff (code-0) —")
    print(f"  First divergence at frame {first_diff} " f"(t={first_diff * SAMPLES_PER_FRAME / 24000:.2f}s)")
    print(f"  {label_a}[{first_diff}] = {a0[first_diff]}    {label_b}[{first_diff}] = {b0[first_diff]}")

    # Show window around first divergence + per-frame audio metrics
    lo = max(0, first_diff - 3)
    hi = min(n, first_diff + max_show + 3)
    print(f"\n  Frames {lo}..{hi-1} (✗ = divergent):")
    print(
        f"  {'frame':>6s}  {f'{label_a}_tok':>10s}  {f'{label_b}_tok':>10s}  "
        f"{f'rms_{label_a}':>10s}  {f'rms_{label_b}':>10s}  {'|Δrms|':>8s}  flag"
    )
    for i in range(lo, hi):
        match = a0[i] == b0[i]
        ra_v = ma["rms"][i] if i < ma["n_frames"] else 0.0
        rb_v = mb["rms"][i] if i < mb["n_frames"] else 0.0
        flag = "✓" if match else "✗"
        print(
            f"  {i:>6d}  {a0[i]:>10d}  {b0[i]:>10d}  " f"{ra_v:>10.4f}  {rb_v:>10.4f}  {abs(ra_v - rb_v):>8.4f}  {flag}"
        )

    # Summary divergence stats
    matches = sum(1 for i in range(n) if a0[i] == b0[i])
    print(f"\n  Total token-level match (code-0): {matches}/{n} ({100*matches/n:.1f}%)")


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
    ap.add_argument("--codes-a", default=None, help="Optional .pt file with codec tokens [seq, 16] for A")
    ap.add_argument("--codes-b", default=None, help="Optional .pt file with codec tokens [seq, 16] for B")
    ap.add_argument(
        "--no-per-frame", action="store_true", help="Skip per-AR-frame analysis (faster for quick basic-stats runs)"
    )
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

    if not args.no_per_frame:
        ma = per_frame_metrics(a, sr_a)
        mb = per_frame_metrics(b, sr_b)
        print_per_frame_section("A", ma, "B", mb)
        if args.codes_a and args.codes_b:
            ca = torch.load(args.codes_a)
            cb = torch.load(args.codes_b)
            per_frame_codes_diff("A", ca, "B", cb, ma, mb)

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
