#!/usr/bin/env python3
"""Non-degeneracy gate for a generated diffusion artifact (video frames and/or audio).

Diffusion analog of ``models/common/readiness_check/check_degenerate_output.py`` (which checks
autoregressive text). It ingests a frames directory (PNG/JPG) and/or a wav file and asserts the output
is not mechanically degenerate: no NaN/Inf, frames not all-constant / all-black / all-white, real
temporal motion, audio not silent and in range. It deliberately does NOT judge fidelity — at tiny
resolutions / few steps abstract content is expected; only DEGENERACY fails.

Exit codes: 0 pass, 1 advisory, 2 critical (degenerate), 3 error (could not evaluate).

Usage:
  check_diffusion_degenerate.py --frames <dir> --wav <file> \
      [--motion-floor 0.5] [--audio-rms-floor 1e-4] [--missing critical|advisory]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys


def _fail(msgs: list[str], code: int) -> int:
    label = {1: "ADVISORY", 2: "CRITICAL", 3: "ERROR"}[code]
    for m in msgs:
        print(f"[check_diffusion_degenerate] {label}: {m}", file=sys.stderr)
    return code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="", help="directory of PNG/JPG frames (frame_*.png)")
    ap.add_argument("--wav", default="", help="wav file")
    ap.add_argument("--motion-floor", type=float, default=0.5, help="min mean abs consecutive-frame diff (0-255)")
    ap.add_argument("--audio-rms-floor", type=float, default=1e-4)
    ap.add_argument("--missing", choices=["critical", "advisory"], default="critical")
    args = ap.parse_args()

    if not args.frames and not args.wav:
        return _fail(["neither --frames nor --wav provided."], 3)

    try:
        import numpy as np
    except Exception as err:  # pragma: no cover
        return _fail([f"numpy unavailable: {err}"], 3)

    miss_code = 2 if args.missing == "critical" else 1
    worst = 0
    stats: list[str] = []

    # ---- video ----
    if args.frames:
        files = sorted(
            glob.glob(os.path.join(args.frames, "frame_*.png"))
            or glob.glob(os.path.join(args.frames, "*.png"))
            or glob.glob(os.path.join(args.frames, "*.jpg"))
        )
        if not files:
            worst = max(worst, miss_code)
            stats.append(f"no frames found under {args.frames}")
        else:
            try:
                import imageio.v2 as iio

                frames = np.stack([iio.imread(f) for f in files]).astype(np.float64)
            except Exception as err:
                return _fail([f"could not load frames: {err}"], 3)
            if not np.isfinite(frames).all():
                worst = max(worst, 2)
                stats.append("frames contain NaN/Inf")
            fmin, fmax = float(frames.min()), float(frames.max())
            if fmin == fmax:
                worst = max(worst, 2)
                stats.append(f"all frames constant (value={fmin})")
            # all-black / all-white per-frame
            per_frame_mean = frames.reshape(frames.shape[0], -1).mean(axis=1)
            if np.all(per_frame_mean <= 1.0) or np.all(per_frame_mean >= 254.0):
                worst = max(worst, 2)
                stats.append("every frame is all-black or all-white")
            motion = float(np.abs(np.diff(frames, axis=0)).mean()) if frames.shape[0] > 1 else 0.0
            if frames.shape[0] > 1 and motion < args.motion_floor:
                worst = max(worst, 2)
                stats.append(f"frozen video: motion {motion:.3f} < floor {args.motion_floor}")
            print(
                f"[video] n={frames.shape[0]} shape={frames.shape[1:]}, range=[{fmin:.0f},{fmax:.0f}], motion={motion:.3f}"
            )

    # ---- audio ----
    if args.wav:
        if not os.path.isfile(args.wav):
            worst = max(worst, miss_code)
            stats.append(f"wav not found: {args.wav}")
        else:
            try:
                import soundfile as sf

                wav, sr = sf.read(args.wav)
                wav = np.asarray(wav, dtype=np.float64)
            except Exception as err:
                return _fail([f"could not load wav: {err}"], 3)
            if not np.isfinite(wav).all():
                worst = max(worst, 2)
                stats.append("audio contains NaN/Inf")
            rms = float(np.sqrt((wav**2).mean())) if wav.size else 0.0
            peak = float(np.abs(wav).max()) if wav.size else 0.0
            if rms < args.audio_rms_floor:
                worst = max(worst, 2)
                stats.append(f"silent audio: rms {rms:.2e} < floor {args.audio_rms_floor:.2e}")
            if peak > 1.5:
                worst = max(worst, 1)
                stats.append(f"audio peak {peak:.2f} > 1.5 (clipping/uncNormalized?)")
            print(f"[audio] sr={sr} samples={wav.shape}, rms={rms:.4f}, peak={peak:.3f}")

    if worst == 0:
        print("[check_diffusion_degenerate] PASS")
        return 0
    return _fail(stats, worst)


if __name__ == "__main__":
    sys.exit(main())
