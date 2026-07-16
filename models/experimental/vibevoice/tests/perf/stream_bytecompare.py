#!/usr/bin/env python3
"""Byte/level compare two raw fp32 24kHz render streams (VV_STREAM_AUDIO .f32).

The acceptance gate for long-form-safe optimizations. maxabsdiff==0.0 over the common prefix means the
candidate produces bit-identical audio to the clean baseline => math-preserving => provably safe for the
90-100 min render (identical token stream -> identical KV evolution -> identical audio forever). Any
divergence prints the first differing frame; for an accepted math-changing opt, that divergence must
fall ONLY inside the baseline's known ~min 77+ residual band, not earlier.

Usage: python stream_bytecompare.py <candidate.f32> [baseline.f32]
  baseline defaults to ~/vibe-voice/vv_lb_FULL/stream_loopbreak.f32 (clean loop-break reference).
"""
import os
import sys

import numpy as np

SR = 24000
F = 3200  # samples per diffusion frame (24kHz * 3200/24000 = 0.1333 s)
cand = os.path.expanduser(sys.argv[1])
base = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/vibe-voice/vv_lb_FULL/stream_loopbreak.f32")

a = np.fromfile(cand, dtype=np.float32)
b = np.fromfile(base, dtype=np.float32)
n = min(len(a), len(b))
if n == 0:
    print(f"EMPTY: cand={len(a)} samples, base={len(b)} samples")
    sys.exit(1)
d = np.abs(a[:n] - b[:n])
maxd, meand = float(d.max()), float(d.mean())
eq = a[:n] == b[:n]
first = None if bool(eq.all()) else int(np.argmax(~eq))
print(f"cand={cand}  ({len(a) // F}f)")
print(f"base={base}  ({len(b) // F}f)")
print(f"common={n // F}f  maxabsdiff={maxd:.6f}  meanabsdiff={meand:.3e}")
if first is None:
    print("BYTE-IDENTICAL over common prefix -> math-preserving / long-form-safe")
else:
    print(f"first diff at sample {first} = frame {first // F} = {first / SR / 60:.2f} min")
