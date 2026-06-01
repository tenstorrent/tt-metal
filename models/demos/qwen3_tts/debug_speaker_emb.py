# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug Step 1: Verify speaker embedding correctness."""
import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

# --- Load aikyo.wav ---
audio, sr = sf.read("/home/yito/dat/aikyo.wav", dtype="float32")
if audio.ndim > 1:
    audio = audio.mean(axis=1)
print(f"Audio: {len(audio)} samples, sr={sr}, duration={len(audio)/sr:.2f}s")

# --- Load saved embedding ---
saved = load_file("/home/yito/dat/aikyo_speaker.safetensors")
saved_emb = saved["speaker_embedding"]  # [1, 2048]
print(f"\nSaved embedding: shape={saved_emb.shape}, norm={saved_emb.norm():.6f}")
print(f"  min={saved_emb.min():.6f}, max={saved_emb.max():.6f}, mean={saved_emb.mean():.6f}")

# --- Run reference speaker encoder ---
from models.demos.qwen3_tts.reference.speaker_encoder_ref import SpeakerEncoderReference
from models.demos.qwen3_tts.tt.speaker_encoder import mel_spectrogram

model_path = os.environ.get("HF_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
snap_dirs = glob.glob(f"/home/yito/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*/")
if not snap_dirs:
    snap_dirs = [model_path]
snap_dir = snap_dirs[0]

ref_model = SpeakerEncoderReference.from_pretrained(snap_dir)
ref_model.eval()

# Run in float32
mel = mel_spectrogram(audio, sr=sr)  # [1, T, 128]
print(f"\nMel spectrogram: shape={mel.shape}")

with torch.no_grad():
    ref_emb_f32 = ref_model.float()(mel.float())  # [1, 2048]
print(f"\nRef embedding (float32): shape={ref_emb_f32.shape}, norm={ref_emb_f32.norm():.6f}")
print(f"  min={ref_emb_f32.min():.6f}, max={ref_emb_f32.max():.6f}, mean={ref_emb_f32.mean():.6f}")

# Run in bfloat16 (same as TT pipeline)
with torch.no_grad():
    ref_emb_bf16 = ref_model.bfloat16()(mel.bfloat16())
    ref_emb_bf16 = ref_emb_bf16.float()
print(f"Ref embedding (bf16→f32): shape={ref_emb_bf16.shape}, norm={ref_emb_bf16.norm():.6f}")

# --- Compare ---
def pcc(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1].item()

def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

print(f"\n=== Comparisons ===")
print(f"Saved vs Ref(f32):  PCC={pcc(saved_emb, ref_emb_f32):.6f}, cos={cosine_sim(saved_emb, ref_emb_f32):.6f}")
print(f"Saved vs Ref(bf16): PCC={pcc(saved_emb, ref_emb_bf16):.6f}, cos={cosine_sim(saved_emb, ref_emb_bf16):.6f}")
print(f"Ref(f32) vs Ref(bf16): PCC={pcc(ref_emb_f32, ref_emb_bf16):.6f}")

# --- Check magnitude relative to typical text embeddings ---
# Load text embedding weights to compare norms
full_sd = {}
for f in sorted(glob.glob(snap_dir + "*.safetensors")):
    full_sd.update(load_file(f))

text_emb_weight = full_sd.get("talker.model.text_embedding.weight")
if text_emb_weight is not None:
    print(f"\n=== Text Embedding Weight ===")
    print(f"Shape: {text_emb_weight.shape}")
    row_norms = text_emb_weight.float().norm(dim=-1)
    print(f"Row norms: min={row_norms.min():.4f}, max={row_norms.max():.4f}, mean={row_norms.mean():.4f}, median={row_norms.median():.4f}")

codec_emb_weight = full_sd.get("talker.model.codec_embedding.weight")
if codec_emb_weight is not None:
    print(f"\n=== Codec Embedding Weight ===")
    print(f"Shape: {codec_emb_weight.shape}")
    row_norms = codec_emb_weight.float().norm(dim=-1)
    print(f"Row norms: min={row_norms.min():.4f}, max={row_norms.max():.4f}, mean={row_norms.mean():.4f}, median={row_norms.median():.4f}")

# Check what x looks like after text_projection on a sample input
text_proj_fc1_w = full_sd.get("talker.text_projection.linear_fc1.weight")
text_proj_fc1_b = full_sd.get("talker.text_projection.linear_fc1.bias")
text_proj_fc2_w = full_sd.get("talker.text_projection.linear_fc2.weight")
text_proj_fc2_b = full_sd.get("talker.text_projection.linear_fc2.bias")

if all(v is not None for v in [text_proj_fc1_w, text_proj_fc1_b, text_proj_fc2_w, text_proj_fc2_b]):
    print(f"\n=== Text Projection Output Norm (sample) ===")
    sample_ids = torch.tensor([[151672, 100, 200, 300, 400]])  # fake text tokens
    text_emb = torch.nn.functional.embedding(sample_ids, text_emb_weight.float())
    x = torch.nn.functional.linear(text_emb, text_proj_fc1_w.float(), text_proj_fc1_b.float())
    x = torch.nn.functional.silu(x)
    x = torch.nn.functional.linear(x, text_proj_fc2_w.float(), text_proj_fc2_b.float())
    print(f"After text_projection: shape={x.shape}, per-token norms={x.squeeze(0).norm(dim=-1).tolist()}")
    print(f"Speaker embedding norm: {saved_emb.norm():.4f}")
    print(f"Ratio (spk_norm / avg_token_norm): {saved_emb.norm() / x.squeeze(0).norm(dim=-1).mean():.4f}")
