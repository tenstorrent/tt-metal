# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
RVC TTNN Runtime Profiler.

Instruments the hybrid inference pipeline to produce a detailed breakdown
of where time is spent. This is a measurement/attribution tool, NOT an
optimization pass.

Usage:
    cd <repo_root>
    python -m models.demos.rvc.profile --max_secs 3.0
"""

import sys
import os

_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_DEMO_DIR, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import ttnn  # noqa: E402

import argparse
import collections
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from safetensors.torch import load_file
from scipy import signal

from models.demos.rvc.torch_impl.vc.hubert import HubertModel
from models.demos.rvc.torch_impl.vc.synthesizer import TextEncoder, SourceModuleHnNSF
from models.demos.rvc.ttnn.runtime import TTNNFlowDecoder, TTNNGeneratorNSF
from models.demos.rvc.utils.audio import load_audio
from models.demos.rvc.utils.config import (
    Config, HubertPretrainingConfig, HubertPretrainingTask,
    get_hubert_paths, get_model_and_config_paths,
)

SR_HUBERT = 16000
SR_TARGET = 48000
WINDOW = 160
UPP = 480
bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=SR_HUBERT)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class Timer:
    """Hierarchical timer for profiling."""

    def __init__(self):
        self.records = []
        self._stack = []

    def start(self, name):
        self._stack.append((name, time.perf_counter()))

    def stop(self):
        name, t0 = self._stack.pop()
        elapsed = time.perf_counter() - t0
        self.records.append((name, elapsed))
        return elapsed

    def report(self):
        total = sum(e for _, e in self.records if not _.startswith("  "))
        print(f"\n{'=' * 70}")
        print(f"PROFILING BREAKDOWN (total measured = {total:.3f}s)")
        print(f"{'=' * 70}")
        print(f"{'Stage':<45} {'Time (s)':>10} {'%':>7}")
        print(f"{'-' * 45} {'-' * 10} {'-' * 7}")
        for name, elapsed in self.records:
            if name.startswith("  "):
                pct = elapsed / total * 100 if total > 0 else 0
                print(f"  {name.strip():<43} {elapsed:>10.4f} {pct:>6.1f}%")
            else:
                pct = elapsed / total * 100 if total > 0 else 0
                print(f"{name:<45} {elapsed:>10.4f} {pct:>6.1f}%")
        print(f"{'=' * 70}")
        return total


def run_profile(max_secs=3.0, device_id=0):
    """Run profiled hybrid inference pipeline."""
    timer = Timer()

    # ---- Load audio ----
    timer.start("Audio load + preprocess")
    audio_raw = load_audio(SR_HUBERT)
    max_samples = int(max_secs * SR_HUBERT)
    if audio_raw.shape[0] > max_samples:
        audio_raw = audio_raw[:max_samples]
    audio_np = audio_raw.numpy()
    audio_max = np.abs(audio_np).max()
    if audio_max > 1:
        audio_np = audio_np / audio_max
    audio_np = signal.filtfilt(bh, ah, audio_np)
    audio = torch.from_numpy(audio_np.copy()).unsqueeze(0).float()
    audio_secs = audio.shape[1] / SR_HUBERT
    timer.stop()

    # ---- Load checkpoint ----
    timer.start("Checkpoint load")
    synth_path, _ = get_model_and_config_paths("v2", "48k", True)
    sd = load_file(synth_path)
    timer.stop()

    # ---- Load torch modules ----
    timer.start("Torch modules load")
    cfg_path, model_path = get_hubert_paths()
    task = HubertPretrainingTask(HubertPretrainingConfig())
    with open(cfg_path) as f:
        cfg = json.load(f)
    hubert = HubertModel(cfg=cfg["model"], task_cfg=task.cfg)
    hubert.load_state_dict(load_file(model_path), strict=True)
    hubert = hubert.eval().float()

    enc_p = TextEncoder(
        embedding_dims=768, out_channels=192, hidden_channels=192,
        filter_channels=768, num_heads=2, num_layers=6, kernel_size=3, f0=True,
    )
    enc_state = {k.replace("enc_p.", ""): v.float()
                  for k, v in sd.items() if k.startswith("enc_p.")}
    enc_p.load_state_dict(enc_state, strict=True)
    enc_p = enc_p.eval()

    m_source = SourceModuleHnNSF(sampling_rate=SR_TARGET, harmonic_num=0, validation=False)
    m_state = {k.replace("dec.m_source.", ""): v.float()
                for k, v in sd.items() if k.startswith("dec.m_source.")}
    m_source.load_state_dict(m_state, strict=True)
    m_source = m_source.eval()

    emb_g = torch.nn.Embedding(109, 256)
    emb_g.weight.data = sd["emb_g.weight"].float()
    timer.stop()

    # ---- Open TTNN device ----
    timer.start("TTNN device open")
    device = ttnn.open_device(device_id=device_id, l1_small_size=32768)
    timer.stop()

    # ---- Load TTNN modules ----
    timer.start("TTNN modules load")
    flow = TTNNFlowDecoder.from_checkpoint(sd, device)
    timer.stop()
    timer.start("  flow.from_checkpoint")
    # already counted above, just for labeling
    timer.records[-1] = ("  flow.from_checkpoint", timer.records[-1][1])

    timer.start("TTNN gen load")
    gen = TTNNGeneratorNSF.from_checkpoint(sd, device)
    timer.stop()
    timer.records[-1] = ("  gen.from_checkpoint", timer.records[-1][1])

    # ==== PREPROCESSING ====
    with torch.no_grad():
        timer.start("Hubert feature extraction")
        logits = hubert(source=audio, output_layer=12)
        feats = logits
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        num_frames = feats.shape[1]
        timer.stop()

        timer.start("F0 extraction (DIO)")
        import pyworld as pw
        audio_f64 = audio_np.astype(np.float64)
        frame_period = WINDOW / SR_HUBERT * 1000.0
        f0, t = pw.dio(audio_f64, SR_HUBERT, f0_floor=50, f0_ceil=1100,
                        frame_period=frame_period, allowed_range=0.1)
        f0 = pw.stonemask(audio_f64, f0, t, SR_HUBERT)
        f0 = torch.from_numpy(f0.astype(np.float32)).unsqueeze(0)
        f0_continuous = f0.clone()
        f0_mel_min = 1127 * math.log(1 + 50 / 700)
        f0_mel_max = 1127 * math.log(1 + 1100 / 700)
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel = torch.clamp(f0_mel, min=1, max=255)
        pitch = torch.round(f0_mel).to(torch.int64)[:, :num_frames]
        pitchf = f0_continuous[:, :num_frames]
        timer.stop()

        timer.start("TextEncoder")
        m_p, logs_p = enc_p(feats, pitch)
        z_p = m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666
        timer.stop()

        timer.start("Speaker embedding")
        sid = torch.tensor([0])
        g = emb_g(sid).unsqueeze(-1)
        timer.stop()

        timer.start("SineGen")
        har_source = m_source(pitchf, UPP).transpose(1, 2)
        timer.stop()

    # ==== CHUNKED TTNN INFERENCE ====
    MAX_CHUNK = 50
    OVERLAP = 5
    n_chunks = (num_frames + MAX_CHUNK - 1) // MAX_CHUNK
    target_len = MAX_CHUNK + 2 * OVERLAP

    # Detailed per-chunk profiling
    chunk_flow_times = []
    chunk_gen_times = []
    chunk_pad_times = []
    chunk_trim_times = []

    audio_segments = []
    ttnn_z_chunks = []

    print(f"\nProfiling {n_chunks} chunks (T={num_frames}, max_chunk={MAX_CHUNK}, overlap={OVERLAP})")

    with torch.no_grad():
        for c in range(n_chunks):
            nom_start = c * MAX_CHUNK
            nom_end = min(nom_start + MAX_CHUNK, num_frames)
            ext_start = max(0, nom_start - OVERLAP)
            ext_end = min(num_frames, nom_end + OVERLAP)
            ext_len = ext_end - ext_start

            z_p_chunk = z_p[:, :, ext_start:ext_end]
            har_chunk = har_source[:, :, ext_start * UPP:ext_end * UPP]

            # Pad
            t0 = time.perf_counter()
            if ext_len < target_len:
                pad_len = target_len - ext_len
                z_p_chunk = F.pad(z_p_chunk, (0, pad_len))
                har_chunk = F.pad(har_chunk, (0, pad_len * UPP))
            chunk_pad_times.append(time.perf_counter() - t0)

            # Flow
            t0 = time.perf_counter()
            z_chunk = flow(z_p_chunk, g)
            chunk_flow_times.append(time.perf_counter() - t0)

            # Generator
            t0 = time.perf_counter()
            audio_chunk = gen(z_chunk, har_chunk, g)
            chunk_gen_times.append(time.perf_counter() - t0)

            # Trim
            t0 = time.perf_counter()
            audio_chunk = audio_chunk[:, :, :ext_len * UPP]
            z_chunk = z_chunk[:, :, :ext_len]
            left_trim = (nom_start - ext_start) * UPP
            right_trim = (ext_end - nom_end) * UPP
            nominal_audio = audio_chunk[:, :, left_trim:audio_chunk.shape[2] - right_trim if right_trim > 0 else audio_chunk.shape[2]]
            left_z = nom_start - ext_start
            right_z = ext_end - nom_end
            nominal_z = z_chunk[:, :, left_z:z_chunk.shape[2] - right_z if right_z > 0 else z_chunk.shape[2]]
            chunk_trim_times.append(time.perf_counter() - t0)

            audio_segments.append(nominal_audio)
            ttnn_z_chunks.append(nominal_z)

    audio_out = torch.cat(audio_segments, dim=2)
    output_secs = audio_out.shape[2] / SR_TARGET

    # Record aggregate timings
    timer.start("TTNN Flow (all chunks)")
    timer.records.append(("TTNN Flow (all chunks)", sum(chunk_flow_times)))
    timer.start("TTNN Generator (all chunks)")
    timer.records.append(("TTNN Generator (all chunks)", sum(chunk_gen_times)))

    # ==== REPORT ====
    total = timer.report()

    # Detailed chunk breakdown
    print(f"\n{'=' * 70}")
    print(f"PER-CHUNK BREAKDOWN")
    print(f"{'=' * 70}")
    print(f"{'Chunk':<8} {'Flow (s)':>10} {'Gen (s)':>10} {'Pad (ms)':>10} {'Trim (ms)':>10}")
    print(f"{'-' * 8} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for i in range(n_chunks):
        print(f"  {i+1:<6} {chunk_flow_times[i]:>10.4f} {chunk_gen_times[i]:>10.4f} "
              f"{chunk_pad_times[i]*1000:>10.3f} {chunk_trim_times[i]*1000:>10.3f}")

    total_flow = sum(chunk_flow_times)
    total_gen = sum(chunk_gen_times)
    total_ttnn = total_flow + total_gen
    total_pad = sum(chunk_pad_times)
    total_trim = sum(chunk_trim_times)

    print(f"\n{'=' * 70}")
    print(f"RUNTIME ATTRIBUTION")
    print(f"{'=' * 70}")
    print(f"  Input:              {audio_secs:.2f}s ({num_frames} frames)")
    print(f"  Output:             {output_secs:.2f}s @ {SR_TARGET}Hz")
    print(f"  Chunks:             {n_chunks} × {MAX_CHUNK} frames (overlap={OVERLAP})")
    print(f"")
    print(f"  TTNN Flow total:    {total_flow:.3f}s ({total_flow/total_ttnn*100:.1f}% of TTNN)")
    print(f"  TTNN Gen total:     {total_gen:.3f}s ({total_gen/total_ttnn*100:.1f}% of TTNN)")
    print(f"  TTNN total:         {total_ttnn:.3f}s")
    print(f"  Pad/trim overhead:  {(total_pad+total_trim)*1000:.1f}ms (negligible)")
    print(f"")
    print(f"  RTF (TTNN only):    {total_ttnn / output_secs:.4f}")
    print(f"  RTF (all):          {total / output_secs:.4f}")
    print(f"")

    # Count operations in generator
    n_conv1d_per_chunk = 1 + 72 + 1  # conv_pre + 12 resblocks × 6 convs + conv_post
    n_conv_transpose_per_chunk = 4  # 4 upsample stages
    n_linear_per_chunk = 1  # cond_linear
    n_h2d_per_chunk = n_conv1d_per_chunk + n_conv_transpose_per_chunk + n_linear_per_chunk
    total_h2d = n_h2d_per_chunk * n_chunks

    # Flow operations
    n_flow_conv1d_per_chunk = 4 * 3  # 4 flows × 3 WN layers
    n_flow_linear_per_chunk = 4 * (1 + 1 + 1 + 3)  # per-flow: pre + post + cond + 3 rsl
    n_flow_h2d_per_chunk = n_flow_conv1d_per_chunk + n_flow_linear_per_chunk
    total_flow_h2d = n_flow_h2d_per_chunk * n_chunks

    print(f"  OPERATION COUNTS (per chunk):")
    print(f"    Generator: {n_conv1d_per_chunk} conv1d + {n_conv_transpose_per_chunk} conv_transpose + {n_linear_per_chunk} linear = {n_h2d_per_chunk} host↔device ops")
    print(f"    Flow:      {n_flow_conv1d_per_chunk} conv1d + {n_flow_linear_per_chunk} linear = {n_flow_h2d_per_chunk} host↔device ops")
    print(f"    Total:     {n_h2d_per_chunk + n_flow_h2d_per_chunk} ops/chunk × {n_chunks} chunks = {(n_h2d_per_chunk + n_flow_h2d_per_chunk) * n_chunks} total ops")
    print(f"")

    avg_gen_per_chunk = total_gen / n_chunks
    avg_flow_per_chunk = total_flow / n_chunks
    avg_gen_per_conv = avg_gen_per_chunk / n_h2d_per_chunk
    avg_flow_per_op = avg_flow_per_chunk / n_flow_h2d_per_chunk

    print(f"  PER-OP TIMING:")
    print(f"    Generator: {avg_gen_per_chunk*1000:.1f}ms/chunk, {avg_gen_per_conv*1000:.2f}ms/op")
    print(f"    Flow:      {avg_flow_per_chunk*1000:.1f}ms/chunk, {avg_flow_per_op*1000:.2f}ms/op")
    print(f"{'=' * 70}")

    # Cleanup
    flow.deallocate()
    gen.deallocate()
    ttnn.close_device(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RVC TTNN Runtime Profiler")
    parser.add_argument("--max_secs", type=float, default=3.0, help="Max input audio seconds")
    parser.add_argument("--device_id", type=int, default=0, help="TTNN device ID")
    args = parser.parse_args()
    run_profile(max_secs=args.max_secs, device_id=args.device_id)
