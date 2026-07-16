#!/usr/bin/env python3
"""Windowed Whisper transcription for a raw fp32 24kHz render stream (VV_STREAM_AUDIO .f32).

Intelligibility spot-check for the long-form gate: transcribe a 30s window at each requested minute
with whisper-medium. Reads only the needed slice via memmap, so it works on a still-growing file.
Coherent on-script text at a minute => that minute is fine; single-word / repeated / word-salad output
=> degeneration there. (The clean baseline has a known self-recovering degenerate patch ~min 78-83.)

Usage: python longform_whisper.py <stream.f32> <min1> [min2 ...]   (window = 30s, WIN_SEC to override)
Run ONE whisper process at a time (medium model starves the CPU if two run in parallel).
"""
import os
import sys

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SR = 24000
WIN = float(os.environ.get("WIN_SEC", "30"))
path = os.path.expanduser(sys.argv[1])
mins = [float(x) for x in sys.argv[2:]]

proc = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").eval()

nbytes = os.path.getsize(path)
total = nbytes // 4
print(f"# STREAM {path}: {total} samples = {total / SR / 60:.2f} min available")
mm = np.memmap(path, dtype="float32", mode="r", shape=(total,))
for m in mins:
    start = int(m * 60 * SR)
    n = int(WIN * SR)
    if start >= total:
        print(f"\n[min {m:5.1f}] NOT YET RENDERED (need {start / SR / 60:.1f} min, have {total / SR / 60:.1f})")
        continue
    audio = np.array(mm[start : min(start + n, total)], dtype="float32")
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    idx = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / SR))
    a16 = np.interp(idx, np.arange(len(audio)), audio).astype("float32")
    feat = proc(a16, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        ids = model.generate(feat, language="en", task="transcribe", max_new_tokens=224)
    txt = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    print(f"\n[min {m:5.1f}] rms={rms:.4f} peak={peak:.3f}")
    print(f"  {txt}")
