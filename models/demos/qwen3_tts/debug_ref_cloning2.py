# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug Step 2b: Test reference Talker with correct EOS token and sampling."""
import glob
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

snap_dirs = glob.glob("/home/yito/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*/")
snap_dir = snap_dirs[0]

tokenizer = AutoTokenizer.from_pretrained(snap_dir, trust_remote_code=True)

text = "こんにちは"
language = "japanese"
prompt = f"<|tts_bos|>{language}<|text_sep|>{text}<|tts_eos|>"
token_ids = tokenizer.encode(prompt, add_special_tokens=False)
text_token_ids = torch.tensor([token_ids], dtype=torch.long)
print(f"Text tokens: {len(token_ids)} tokens")

# Correct EOS token ID from HF config
CODEC_EOS = 2150

print("Loading reference Talker...")
from models.demos.qwen3_tts.reference.talker_ref import TalkerReference
talker = TalkerReference.from_safetensors(snap_dir)
talker.eval().float()

saved = load_file("/home/yito/dat/aikyo_speaker.safetensors")
speaker_emb = saved["speaker_embedding"].float()

def generate_with_eos(model, text_tokens, speaker_emb, max_tokens, temp, eos_id):
    """Autoregressive CB0 gen with correct EOS check."""
    B, S = text_tokens.shape
    x = model.text_embedding(text_tokens)
    x = model.text_projection_fc2(F.silu(model.text_projection_fc1(x)))
    if speaker_emb is not None:
        x = x + speaker_emb.unsqueeze(1)

    cos = model.rope_cos[:S].unsqueeze(0).expand(B, -1, -1)
    sin = model.rope_sin[:S].unsqueeze(0).expand(B, -1, -1)
    mask = model._causal_mask(S, x.dtype, x.device)

    max_seq = S + max_tokens
    kv_caches = model.init_kv_caches(B, max_seq, "cpu", dtype=torch.float32)

    for i, layer in enumerate(model.layers):
        x = layer(x, cos, sin, mask=mask, kv_cache=kv_caches[i], start_pos=0)

    hidden = model.norm(x)
    logits = model.codec_head(hidden)

    # Sample first token
    first_logits = logits[:, -1:, :]
    if temp > 0:
        probs = F.softmax(first_logits[:, -1, :] / temp, dim=-1)
        next_tok = torch.multinomial(probs, 1).squeeze(-1)
    else:
        next_tok = first_logits[:, -1, :].argmax(dim=-1)

    generated = [next_tok.item()]

    for step in range(max_tokens - 1):
        if next_tok.item() == eos_id:
            break

        pos = S + step
        decode_logits = model.forward_decode(next_tok.unsqueeze(1), kv_caches, start_pos=pos)

        if temp > 0:
            probs = F.softmax(decode_logits[:, -1, :] / temp, dim=-1)
            next_tok = torch.multinomial(probs, 1).squeeze(-1)
        else:
            next_tok = decode_logits[:, -1, :].argmax(dim=-1)

        generated.append(next_tok.item())

    return generated

torch.manual_seed(42)

# Test 1: No cloning
print("\n=== Reference: NO cloning, temp=0.9 ===")
t0 = time.time()
toks_no = generate_with_eos(talker, text_token_ids, None, 100, 0.9, CODEC_EOS)
elapsed = time.time() - t0
has_eos = CODEC_EOS in toks_no
print(f"Generated {len(toks_no)} tokens in {elapsed:.1f}s, EOS present: {has_eos}")
print(f"First 20: {toks_no[:20]}")
if has_eos:
    eos_pos = toks_no.index(CODEC_EOS)
    print(f"EOS at position {eos_pos}")

torch.manual_seed(42)

# Test 2: With cloning
print("\n=== Reference: WITH cloning, temp=0.9 ===")
t0 = time.time()
toks_yes = generate_with_eos(talker, text_token_ids, speaker_emb, 100, 0.9, CODEC_EOS)
elapsed = time.time() - t0
has_eos = CODEC_EOS in toks_yes
print(f"Generated {len(toks_yes)} tokens in {elapsed:.1f}s, EOS present: {has_eos}")
print(f"First 20: {toks_yes[:20]}")
if has_eos:
    eos_pos = toks_yes.index(CODEC_EOS)
    print(f"EOS at position {eos_pos}")

torch.manual_seed(42)

# Test 3: No cloning, greedy
print("\n=== Reference: NO cloning, greedy ===")
t0 = time.time()
toks_greedy = generate_with_eos(talker, text_token_ids, None, 100, 0.0, CODEC_EOS)
elapsed = time.time() - t0
has_eos = CODEC_EOS in toks_greedy
print(f"Generated {len(toks_greedy)} tokens in {elapsed:.1f}s, EOS present: {has_eos}")
print(f"First 20: {toks_greedy[:20]}")

torch.manual_seed(42)

# Test 4: With cloning, greedy
print("\n=== Reference: WITH cloning, greedy ===")
t0 = time.time()
toks_greedy_clone = generate_with_eos(talker, text_token_ids, speaker_emb, 100, 0.0, CODEC_EOS)
elapsed = time.time() - t0
has_eos = CODEC_EOS in toks_greedy_clone
print(f"Generated {len(toks_greedy_clone)} tokens in {elapsed:.1f}s, EOS present: {has_eos}")
print(f"First 20: {toks_greedy_clone[:20]}")
