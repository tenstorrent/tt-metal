#!/usr/bin/env python3
"""Offline mirror of the on-device long-form energy stabilizer (generator VV_AUDIO_LIMIT / _emit_limit).

Applies the same per-frame emit-only soft-limiter to a saved .f32 render stream:
    gain = min(1, R/rms, P/peak)   per 3200-sample (0.1333 s) frame
so RMS <= R (healthy loudness) and peak <= P (no clipping).  Because the limiter is emit-only (it never
touches the token/latent trajectory), running it offline on an UN-limited render produces a stream
BYTE-IDENTICAL to what `VV_AUDIO_LIMIT=R VV_AUDIO_PEAK=P` produces at render time -- so it doubles as
(a) a way to make a clean deliverable from an existing render without re-rendering, and (b) a fast way to
sweep R/P offline before committing to a device run.  Verify the result with longform_energy_oracle.py
(energy) and longform_whisper.py (content is unchanged vs the source, by construction).

Usage: python longform_emit_limit.py <src.f32> [R=0.18] [P=0.95] [dst.f32]
"""
import os
import sys

import numpy as np

src = os.path.expanduser(sys.argv[1])
R = float(sys.argv[2]) if len(sys.argv) > 2 else 0.18
P = float(sys.argv[3]) if len(sys.argv) > 3 else 0.95
dst = os.path.expanduser(sys.argv[4]) if len(sys.argv) > 4 else src.replace(".f32", f"_lim{R}.f32")
F = 3200
a = np.fromfile(src, dtype=np.float32)
n = len(a) // F
b = a[: n * F].reshape(n, F).astype(np.float64)
rms = np.sqrt((b**2).mean(1))
pk = np.abs(b).max(1)
g_rms = np.where(rms > 1e-9, R / rms, 1.0)
g_pk = np.where(pk > 1e-9, P / pk, 1.0)
gain = np.minimum(1.0, np.minimum(g_rms, g_pk))
out = (b * gain[:, None]).astype(np.float32).reshape(-1)
out.tofile(dst)
n_eng = int((gain < 0.999).sum())
print(f"# {src} -> {dst}: {n} frames, limiter engaged on {n_eng} ({100 * n_eng / n:.1f}%)")
