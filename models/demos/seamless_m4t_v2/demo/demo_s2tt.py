# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
SeamlessM4Tv2 S2TT demo: English speech (wav) -> Japanese text on Blackhole.

Usage (inside the metalcon container, device on /dev/tenstorrent/2):
    python3 models/demos/seamless_m4t_v2/demo/demo_s2tt.py --wav input.wav --tgt_lang jpn
"""

from __future__ import annotations

import argparse

import numpy as np

import ttnn
from models.demos.seamless_m4t_v2.tt.generator import SeamlessS2TTGenerator


def _load_audio(path, target_sr=16000):
    # librosa decodes wav/m4a/mp3/... (m4a/aac needs ffmpeg available)
    import librosa

    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32)


def _segment_by_silence(audio, sr, max_sec, top_db=30, min_sec=0.3):
    """VAD-style segmentation: cut only at silences (librosa.effects.split), grouping
    non-silent regions into segments up to max_sec so cuts fall on natural pauses."""
    import librosa

    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return [(0, len(audio))]
    max_len = int(max_sec * sr)
    segs = []
    seg_s, seg_e = int(intervals[0][0]), int(intervals[0][1])
    for s, e in intervals[1:]:
        if e - seg_s <= max_len:
            seg_e = int(e)
        else:
            segs.append((seg_s, seg_e))
            seg_s, seg_e = int(s), int(e)
    segs.append((seg_s, seg_e))
    # hard-split any single segment that is still longer than max_sec
    out = []
    for s, e in segs:
        if e - s <= max_len:
            out.append((s, e))
        else:
            for c in range(s, e, max_len):
                out.append((c, min(c + max_len, e)))
    return [(s, e) for s, e in out if e - s >= int(min_sec * sr)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="path to input English audio (wav/m4a/mp3/...)")
    ap.add_argument("--tgt_lang", default="jpn")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--bucket_len", type=int, default=128, help="constant-shape decode length (faster)")
    ap.add_argument("--max_sec", type=float, default=20.0,
                    help="max segment length; long audio is split at silences up to this (0 = no split). "
                         "SeamlessM4T is utterance-level, so long-form must be segmented.")
    ap.add_argument("--top_db", type=float, default=30.0, help="silence threshold for VAD segmentation")
    ap.add_argument("--repetition_penalty", type=float, default=1.5, help=">1 suppresses greedy loops")
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3, help="block repeated n-grams (0 = off)")
    ap.add_argument("--out", default=None, help="path to save the translation text")
    args = ap.parse_args()

    sr = 16000
    audio = _load_audio(args.wav)
    dur = len(audio) / sr
    print(f"loaded {args.wav}: {dur:.1f}s")

    # segment at silences (VAD) up to max_sec, so cuts fall on natural pauses
    if args.max_sec and dur > args.max_sec:
        spans = _segment_by_silence(audio, sr, args.max_sec, top_db=args.top_db)
    else:
        spans = [(0, len(audio))]
    print(f"{len(spans)} silence-bounded segment(s) (<= {args.max_sec:.0f}s each)")

    device = ttnn.open_device(device_id=0)
    try:
        gen = SeamlessS2TTGenerator.build(device)
        pieces = []
        max_len = int(args.max_sec * sr) if args.max_sec else 0
        for i, (s, e) in enumerate(spans):
            ch = audio[s:e]
            # pad to a uniform length so the encoder compiles once (segments are
            # cut at silences; the trailing pad is just silence)
            if max_len and len(ch) < max_len:
                ch = np.pad(ch, (0, max_len - len(ch)))
            text, _ = gen.generate(
                ch, device, tgt_lang=args.tgt_lang,
                max_new_tokens=args.max_new_tokens, bucket_len=args.bucket_len,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            pieces.append(text)
            print(f"[seg {i + 1}/{len(spans)} {s / sr:.1f}-{e / sr:.1f}s] {text}")
        full = "".join(pieces)
        print(f"\n=== {args.tgt_lang} translation ===\n{full}\n")
        if args.out:
            with open(args.out, "w") as f:
                f.write(full + "\n")
            print(f"saved -> {args.out}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
