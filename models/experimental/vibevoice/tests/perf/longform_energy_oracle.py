#!/usr/bin/env python3
"""Fast long-form stability oracle: per-minute peak/RMS of a render stream -> PASS/FAIL.

The perf-stack loud/clipping->collapse manifests early (by min 8-15), so a short window is a cheap
REJECT for the worst offenders. It is NOT a sufficient ACCEPT: a window that stops before a late
collapse can false-pass, and the clean loop-break baseline itself trips this oracle in its known
~min 78-83 residual band -- so "matches the clean baseline" (stream_bytecompare.py) is the real bar,
not an absolute oracle pass. Use this to fail fast, then confirm with the full render + baseline compare.

PASS  = every voiced minute has peak<=1.25 and RMS in [0.02,0.20] (matches clean 122f).
FAIL  = any voiced minute with peak>1.25 (clipping) OR RMS>0.20 (loud) OR RMS<0.02 (collapse).

Usage: python longform_energy_oracle.py <stream.f32|wav> [max_min]
"""
import os
import sys

import numpy as np

path = os.path.expanduser(sys.argv[1])
max_min = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
SR = 24000
if path.endswith(".f32"):
    a = np.fromfile(path, dtype=np.float32)
else:
    import soundfile as sf

    a, SR = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)

navail = len(a) / SR / 60
verdict = "PASS"
reasons = []
print(f"# {path}\n# {navail:.2f} min available; checking min 0..{min(max_min, navail):.0f}")
print(f"{'min':>4} {'rms':>7} {'peak':>7}")
for m in range(1, int(min(max_min, navail)) + 1):
    seg = a[int((m - 1) * 60 * SR) : int(m * 60 * SR)]
    if len(seg) == 0:
        continue
    rms = float(np.sqrt(np.mean(seg**2)))
    peak = float(np.max(np.abs(seg)))
    flag = ""
    if peak > 1.25:
        flag = "<-- CLIP"
        reasons.append(f"min{m} peak {peak:.2f}>1.25")
    elif rms > 0.20:
        flag = "<-- LOUD"
        reasons.append(f"min{m} rms {rms:.2f}>0.20")
    elif rms < 0.02:
        flag = "<-- COLLAPSE"
        reasons.append(f"min{m} rms {rms:.3f}<0.02")
    if flag:
        verdict = "FAIL"
    print(f"{m:>4} {rms:7.4f} {peak:7.3f} {flag}")
print(f"\nVERDICT: {verdict}" + (f"  ({'; '.join(reasons[:4])})" if reasons else ""))
