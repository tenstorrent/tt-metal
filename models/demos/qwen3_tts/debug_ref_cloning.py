# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug Step 2: Test reference Talker with voice cloning on CPU.

Runs the reference TalkerReference model with speaker embedding to check
if CB0 generation hits EOS at reasonable length. CPU inference is slow
(~1-2 min for 100 steps) but sufficient to verify behavior.
"""
import glob
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

# --- Build tokenizer and create text tokens ---
snap_dirs = glob.glob("/home/yito/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*/")
snap_dir = snap_dirs[0]

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(snap_dir, trust_remote_code=True)

text = "こんにちは"
language = "japanese"
tts_bos = 151672
tts_eos = 151673
tts_pad = 151671
codec_bos = 2149

prompt = f"<|tts_bos|>{language}<|text_sep|>{text}<|tts_eos|>"
token_ids = tokenizer.encode(prompt, add_special_tokens=False)
print(f"Text tokens: {token_ids} (len={len(token_ids)})")

text_token_ids = torch.tensor([token_ids], dtype=torch.long)

# --- Load reference Talker ---
print("Loading reference Talker (this takes ~30s on CPU)...")
t0 = time.time()
from models.demos.qwen3_tts.reference.talker_ref import TalkerReference, generate_cb0
talker = TalkerReference.from_safetensors(snap_dir)
talker.eval()
talker = talker.float()  # float32 for accuracy
print(f"Loaded in {time.time()-t0:.1f}s")

# --- Load speaker embedding ---
saved = load_file("/home/yito/dat/aikyo_speaker.safetensors")
speaker_emb = saved["speaker_embedding"].float()  # [1, 2048]
print(f"Speaker embedding: shape={speaker_emb.shape}, norm={speaker_emb.norm():.4f}")

# --- Test 1: Without cloning ---
print("\n=== Test 1: CB0 generation WITHOUT speaker embedding ===")
t0 = time.time()
cb0_no_clone = generate_cb0(
    talker, text_token_ids,
    speaker_emb=None,
    max_new_tokens=100,
    temperature=0.0,  # greedy for determinism
    device="cpu",
)
elapsed = time.time() - t0
print(f"Generated {cb0_no_clone.shape[1]} tokens in {elapsed:.1f}s")
print(f"Tokens: {cb0_no_clone[0].tolist()[:30]}...")
eos_id = talker.codec_vocab_size - 1
has_eos = (cb0_no_clone == eos_id).any().item()
print(f"EOS token ({eos_id}) present: {has_eos}")

# --- Test 2: With cloning ---
print("\n=== Test 2: CB0 generation WITH speaker embedding ===")
t0 = time.time()
cb0_clone = generate_cb0(
    talker, text_token_ids,
    speaker_emb=speaker_emb,
    max_new_tokens=100,
    temperature=0.0,  # greedy for determinism
    device="cpu",
)
elapsed = time.time() - t0
print(f"Generated {cb0_clone.shape[1]} tokens in {elapsed:.1f}s")
print(f"Tokens: {cb0_clone[0].tolist()[:30]}...")
has_eos = (cb0_clone == eos_id).any().item()
print(f"EOS token ({eos_id}) present: {has_eos}")

# --- Compare prefill outputs ---
print("\n=== Prefill comparison ===")
with torch.no_grad():
    logits_no, hidden_no = talker.forward_prefill(text_token_ids, speaker_emb=None)
    logits_yes, hidden_yes = talker.forward_prefill(text_token_ids, speaker_emb=speaker_emb)

print(f"Logits (no clone) last token: norm={logits_no[:,-1,:].norm():.4f}, argmax={logits_no[:,-1,:].argmax().item()}")
print(f"Logits (w/ clone) last token: norm={logits_yes[:,-1,:].norm():.4f}, argmax={logits_yes[:,-1,:].argmax().item()}")
print(f"Hidden (no clone) last token: norm={hidden_no[:,-1,:].norm():.4f}")
print(f"Hidden (w/ clone) last token: norm={hidden_yes[:,-1,:].norm():.4f}")

# Top-5 probabilities
import torch.nn.functional as F
probs_no = F.softmax(logits_no[:,-1,:], dim=-1)
probs_yes = F.softmax(logits_yes[:,-1,:], dim=-1)
top5_no = torch.topk(probs_no, 5)
top5_yes = torch.topk(probs_yes, 5)
print(f"\nTop-5 (no clone): tokens={top5_no.indices[0].tolist()}, probs={[f'{p:.4f}' for p in top5_no.values[0].tolist()]}")
print(f"Top-5 (w/ clone): tokens={top5_yes.indices[0].tolist()}, probs={[f'{p:.4f}' for p in top5_yes.values[0].tolist()]}")
