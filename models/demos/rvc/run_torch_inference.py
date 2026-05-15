# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
RVC Torch Reference — End-to-End Inference Script

Runs the full torch reference pipeline on a sample audio file and saves:
  - output WAV file
  - intermediate tensor shapes at every pipeline stage
  - reference tensors for future PCC comparison

Usage:
    python models/demos/rvc/run_torch_inference.py
"""

import json
import os
import sys
import time
import wave

import numpy as np
import torch
from scipy import signal

# Ensure repo root is on PYTHONPATH
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.demos.rvc.torch_impl.vc.pipeline import Pipeline
from models.demos.rvc.utils.audio import load_audio
from models.demos.rvc.utils.config import Config
from models.demos.rvc.utils.f0 import F0Method


def save_wav(audio_int16: np.ndarray, path: str, sample_rate: int):
    """Save int16 numpy array as mono WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def main():
    print("=" * 60)
    print("RVC Torch Reference — End-to-End Inference")
    print("=" * 60)

    # Configuration
    version = "v2"
    num = "48k"
    if_f0 = True
    f0_method = F0Method.RMVPE  # Built-in, no external deps needed
    speaker_id = 0

    output_dir = os.path.join(os.path.dirname(__file__), "data", "output")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load config
    print("\n[1/6] Loading config...")
    config = Config()
    config.use_cpu()
    print(f"  Device: {config.device}")
    print(f"  x_pad={config.x_pad}, x_center={config.x_center}, x_max={config.x_max}")

    # Step 2: Initialize pipeline (loads Hubert + Synthesizer checkpoints)
    print(f"\n[2/6] Initializing pipeline (version={version}, f0_method={f0_method.name})...")
    t0 = time.time()
    pipeline = Pipeline(
        if_f0=if_f0,
        version=version,
        num=num,
        config=config,
        speaker_id=speaker_id,
        f0_method=f0_method,
        index_rate=0.0,  # No FAISS index for this test
        rms_mix_rate=0.25,
        protect=0.33,
        file_index=None,
        validation=True,  # Deterministic (no random noise in reparameterization)
    )
    t_init = time.time() - t0
    print(f"  Pipeline initialized in {t_init:.1f}s")
    print(f"  Target sample rate: {pipeline.tgt_sr}")
    print(f"  Hubert model: {sum(p.numel() for p in pipeline.hubert_model.parameters())/1e6:.1f}M params")
    print(f"  Synthesizer: {sum(p.numel() for p in pipeline.synthesizer.parameters())/1e6:.1f}M params")

    # Step 3: Load input audio
    print("\n[3/6] Loading input audio...")
    sr = 16000
    audio = load_audio(sr)
    audio_max = torch.abs(audio).max().item()
    if audio_max > 1:
        audio /= audio_max

    bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=sr)
    audio = signal.filtfilt(bh, ah, audio)
    audio = torch.from_numpy(audio.copy()).unsqueeze(0).to(torch.float32)

    print(f"  Input shape: {audio.shape}")
    print(f"  Duration: {audio.shape[1]/sr:.2f}s")
    print(f"  Range: [{audio.min().item():.4f}, {audio.max().item():.4f}]")

    # Step 4: Run inference
    print(f"\n[4/6] Running inference (f0_method={f0_method.name})...")
    t0 = time.time()
    with torch.no_grad():
        output = pipeline.run(audio)
    t_infer = time.time() - t0
    rtf = t_infer / (audio.shape[1] / sr)
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Inference time: {t_infer:.2f}s")
    print(f"  Real-time factor: {rtf:.2f}x (< 1.0 = faster than real-time)")

    # Step 5: Save output WAV
    print("\n[5/6] Saving output...")
    output_np = output[0].cpu().numpy().astype(np.int16)
    output_path = os.path.join(output_dir, "torch_reference_output.wav")
    save_wav(output_np, output_path, pipeline.tgt_sr)
    print(f"  Output WAV: {output_path}")
    print(f"  Output duration: {len(output_np)/pipeline.tgt_sr:.2f}s")

    # Step 6: Save reference tensors for future PCC comparison
    print("\n[6/6] Saving reference tensors...")
    ref_path = os.path.join(output_dir, "torch_reference_tensors.pt")
    torch.save({
        "input_audio": audio,
        "output_audio_int16": output,
        "config": {
            "version": version,
            "num": num,
            "if_f0": if_f0,
            "f0_method": f0_method.name,
            "speaker_id": speaker_id,
            "tgt_sr": pipeline.tgt_sr,
            "sr": sr,
            "validation": True,
        },
    }, ref_path)
    print(f"  Reference tensors: {ref_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUCCESS — Torch reference inference complete")
    print("=" * 60)
    print(f"  Input:  {audio.shape[1]/sr:.2f}s @ {sr}Hz")
    print(f"  Output: {len(output_np)/pipeline.tgt_sr:.2f}s @ {pipeline.tgt_sr}Hz ({output_path})")
    print(f"  Init:   {t_init:.1f}s")
    print(f"  Infer:  {t_infer:.2f}s (RTF={rtf:.2f})")
    print(f"  Refs:   {ref_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
