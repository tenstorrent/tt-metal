#!/usr/bin/env python3
"""Japanese STT on tt-metal: Whisper large-v3, language=ja, one N150 chip.

    TT_VISIBLE_DEVICES=2 python run_jp_stt.py --wav /tmp/jp_tts.wav
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def _load_wav(path: str) -> tuple[int, np.ndarray]:
    sr, audio = wavfile.read(path)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample

        n = int(round(audio.shape[0] * 16000 / float(sr)))
        audio = resample(audio, n).astype(np.float32)
        sr = 16000
    return sr, audio


def main() -> int:
    parser = argparse.ArgumentParser(description="Japanese Whisper STT on one N150")
    parser.add_argument("--wav", required=True, help="Input wav (any rate; stereo ok)")
    parser.add_argument("--model", default="openai/whisper-large-v3")
    parser.add_argument("--language", default="ja", help="Whisper language code (ja, not japanese)")
    parser.add_argument("--warmup", action="store_true", help="Run a 1s silence pass first")
    args = parser.parse_args()

    wav_path = Path(args.wav).expanduser().resolve()
    if not wav_path.is_file():
        print(f"ERROR: wav not found: {wav_path}", file=sys.stderr)
        return 1

    metal_home = os.environ.get("TT_METAL_HOME")
    if not metal_home:
        metal_home = str(Path(__file__).resolve().parents[3])
    if metal_home not in sys.path:
        sys.path.insert(0, metal_home)

    print("=" * 72)
    print("Japanese STT — openai/whisper-large-v3  language=ja  N150")
    print("=" * 72)
    print(f"wav:                 {wav_path}")
    print(f"TT_METAL_HOME:       {metal_home}")
    print(f"MESH_DEVICE:         {os.environ.get('MESH_DEVICE', '(unset)')}")
    print(f"TT_VISIBLE_DEVICES:  {os.environ.get('TT_VISIBLE_DEVICES', '(unset)')}")
    print(f"mesh descriptor:     {os.environ.get('TT_MESH_GRAPH_DESC_PATH', '(unset)')}")
    print(f"HF_HUB_CACHE:        {os.environ.get('HF_HUB_CACHE', '(default)')}")
    print()

    import ttnn
    from models.demos.audio.whisper.demo.demo import (
        create_functional_whisper_for_conditional_generation_inference_pipeline,
    )

    t0 = time.time()
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        l1_small_size=32768,
        trace_region_size=100000000,
    )
    mesh_device.enable_program_cache()
    print(f"[1/3] device open  {time.time() - t0:.1f}s")

    t0 = time.time()
    pipeline = create_functional_whisper_for_conditional_generation_inference_pipeline(
        mesh_device=mesh_device,
        model_repo=args.model,
        language=args.language,
        task="transcribe",
        batch_size_per_device=1,
    )
    print(f"[2/3] pipeline     {time.time() - t0:.1f}s")

    if args.warmup:
        t0 = time.time()
        dummy = np.zeros(16000, dtype=np.float32)
        try:
            _ = pipeline([(16000, dummy)], stream=False)
        except Exception as e:
            print(f"      warmup error (non-fatal): {e}")
        print(f"[3/3] warmup       {time.time() - t0:.1f}s")
    else:
        print("[3/3] warmup skipped (pass --warmup to compile traces first)")

    sr, audio = _load_wav(str(wav_path))
    duration_s = audio.shape[0] / float(sr)
    print(f"\nTranscribing {duration_s:.2f}s audio @ {sr} Hz ...")
    t0 = time.time()
    result = pipeline([(sr, audio)], stream=False)
    elapsed = time.time() - t0
    text = result[0] if result else ""
    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    print()
    print(f"TEXT: {text}")
    print(f"wall: {elapsed:.2f}s")

    try:
        ttnn.close_mesh_device(mesh_device)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
