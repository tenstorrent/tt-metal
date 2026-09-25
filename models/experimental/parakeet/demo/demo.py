# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
"""Standalone CLI: 16 kHz wav/flac in, transcript out, on a TT device.

Usage:
  python demo/demo.py --checkpoint /weights --device-id 0 --precision bf16 clip.flac [more.wav ...]

Log-mel extraction and detokenization run on the host with the checkpoint's transformers processor
(declared host policy). Encoder and greedy TDT decode run on TT. Clips are processed one at a time
(batch 1). The supported envelope is ≤ 32 s per clip.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def load_audio(path, target_sr):
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        raise ValueError(f"{path}: sample rate {sr} Hz, expected {target_sr} Hz (resample first)")
    return audio


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="+", help="16 kHz wav/flac files")
    ap.add_argument("--checkpoint", required=True, help="pinned nvidia/parakeet-tdt-0.6b-v3 directory")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()

    import ttnn
    from transformers import AutoProcessor

    from tt import DEVICE_OPTIONS, create_backend

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    sr = processor.feature_extractor.sampling_rate
    with open(os.path.join(args.checkpoint, "config.json")) as f:
        cfg = json.load(f)
    device = ttnn.open_device(device_id=args.device_id, **DEVICE_OPTIONS)
    try:
        model = create_backend(args.checkpoint, cfg, device, precision=args.precision)
        for path in args.audio:
            feats = processor(load_audio(path, sr), sampling_rate=sr, return_tensors="np")
            mel = np.asarray(feats["input_features"], dtype=np.float32)
            mask = feats.get("attention_mask")
            lens = (np.asarray(mask).sum(-1) if mask is not None else np.array([mel.shape[1]])).astype(np.int64)
            t0 = time.perf_counter()
            tokens = model.transcribe(mel, lens)["tokens"]
            dt = time.perf_counter() - t0
            text = processor.batch_decode(tokens, skip_special_tokens=True)[0]
            print(f"{path}\t{dt:.3f}s\t{text}", flush=True)
    finally:
        ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
