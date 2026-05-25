# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run a short HF-only generation (CPU, no Code Predictor) to check CB0 sequence quality.

This uses the simplified decode loop: CB0 embedding + tts_pad_embed only.
No Code Predictor, no CB1-15 in the decode input.
This tells us whether the Talker alone produces a coherent CB0 sequence.

Usage:
    python .../hf_short_generate.py [--steps 50]
"""

import json
import math
import os
import sys

import torch

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TEXT = "こんにちは"
LANGUAGE = "japanese"
MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50


def main():
    from huggingface_hub import hf_hub_download, HfApi
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3ForCausalLM, Qwen3Config

    config_path = hf_hub_download(MODEL_PATH, "config.json")
    with open(config_path) as f:
        raw_config = json.load(f)
    tc = raw_config["talker_config"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Build model
    qwen3_config = Qwen3Config(
        vocab_size=tc.get("codec_vocab_size", 3072),
        hidden_size=tc["hidden_size"],
        intermediate_size=tc["intermediate_size"],
        num_hidden_layers=tc["num_hidden_layers"],
        num_attention_heads=tc["num_attention_heads"],
        num_key_value_heads=tc["num_key_value_heads"],
        hidden_act="silu",
        max_position_embeddings=tc.get("max_position_embeddings", 32768),
        rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
        rope_theta=tc.get("rope_theta", 1000000.0),
        tie_word_embeddings=False,
        head_dim=tc.get("head_dim", 128),
    )

    api = HfApi()
    files = api.list_repo_files(MODEL_PATH)
    shard_files = sorted([f for f in files if f.endswith(".safetensors")])
    talker_weights = {}
    for shard in shard_files:
        fpath = hf_hub_download(MODEL_PATH, shard)
        sd = load_file(fpath)
        for k, v in sd.items():
            if k.startswith("talker."):
                talker_weights[k] = v
        del sd

    hf_model = Qwen3ForCausalLM(qwen3_config)
    hf_sd = hf_model.state_dict()
    mapped = {}
    for hf_key in hf_sd:
        if hf_key == "lm_head.weight":
            src = "talker.codec_head.weight"
        elif hf_key == "model.embed_tokens.weight":
            src = "talker.model.codec_embedding.weight"
        elif hf_key.startswith("model."):
            src = f"talker.{hf_key}"
        else:
            src = None
        if src and src in talker_weights:
            mapped[hf_key] = talker_weights[src]
        else:
            alt = f"talker.model.{hf_key.removeprefix('model.')}"
            if alt in talker_weights:
                mapped[hf_key] = talker_weights[alt]
    hf_model.load_state_dict(mapped, strict=False)
    hf_model = hf_model.to(torch.bfloat16).eval()
    print("Model loaded")

    # Build embeddings
    text_embed_w = talker_weights["talker.model.text_embedding.weight"].to(torch.bfloat16)
    codec_embed_w = talker_weights["talker.model.codec_embedding.weight"].to(torch.bfloat16)
    tp1_w = talker_weights["talker.text_projection.linear_fc1.weight"].to(torch.bfloat16)
    tp1_b = talker_weights["talker.text_projection.linear_fc1.bias"].to(torch.bfloat16)
    tp2_w = talker_weights["talker.text_projection.linear_fc2.weight"].to(torch.bfloat16)
    tp2_b = talker_weights["talker.text_projection.linear_fc2.bias"].to(torch.bfloat16)

    def text_embed(ids): return torch.nn.functional.embedding(ids, text_embed_w)
    def codec_embed(ids): return torch.nn.functional.embedding(ids, codec_embed_w)
    def text_proj(x):
        h = torch.nn.functional.linear(x, tp1_w, tp1_b)
        h = torch.nn.functional.silu(h)
        return torch.nn.functional.linear(h, tp2_w, tp2_b)

    formatted = f"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = torch.tensor([tokenizer.encode(formatted)], dtype=torch.long)

    sp = text_proj(text_embed(torch.tensor([[151672, 151673, 151671]])))
    tts_bos_emb, tts_eos_emb, tts_pad_emb = sp[:, 0:1], sp[:, 1:2], sp[:, 2:3]

    language_id = tc["codec_language_id"][LANGUAGE.lower()]
    codec_tag = codec_embed(torch.tensor([[tc["codec_think_id"], tc["codec_think_bos_id"], language_id, tc["codec_think_eos_id"]]]))
    codec_suffix = codec_embed(torch.tensor([[tc["codec_pad_id"], tc["codec_bos_id"]]]))
    codec_prefill = torch.cat([codec_tag, codec_suffix], dim=1)

    role_proj = text_proj(text_embed(input_ids[:, :3]))
    n_codec_m1 = codec_prefill.shape[1] - 1
    text_side_tag = torch.cat([tts_pad_emb.expand(-1, n_codec_m1 - 1, -1), tts_bos_emb], dim=1)
    part_tag = text_side_tag + codec_prefill[:, :-1, :]

    text_content_ids = input_ids[:, 3:-5]
    text_content_proj = text_proj(text_embed(text_content_ids))
    text_with_eos = torch.cat([text_content_proj, tts_eos_emb], dim=1)
    codec_pad_emb = codec_embed(torch.tensor([[tc["codec_pad_id"]]]))
    part_text = text_with_eos + codec_pad_emb.expand(-1, text_with_eos.shape[1], -1)
    part_final = tts_pad_emb + codec_prefill[:, -1:, :]
    full_embed = torch.cat([role_proj, part_tag, part_text, part_final], dim=1)

    print(f"Prefill shape: {full_embed.shape}")

    # --- Generation loop ---
    codec_eos_id = tc.get("codec_eos_token_id", 2150)
    cb0_tokens = []

    with torch.no_grad():
        out = hf_model(inputs_embeds=full_embed, use_cache=True)
        past_kv = out.past_key_values
        logits = out.logits[0, -1, :3072].float()
        cb0 = logits.argmax().item()
        cb0_tokens.append(cb0)
        print(f"Step 0: CB0={cb0}, EOS_logit={logits[codec_eos_id].item():.2f}")

        for step in range(1, MAX_STEPS):
            if cb0 == codec_eos_id:
                print(f"EOS at step {step}")
                break

            # Simple: CB0 only + tts_pad (no Code Predictor)
            cb0_emb = torch.nn.functional.embedding(
                torch.tensor([[cb0]], dtype=torch.long), codec_embed_w
            )
            decode_input = cb0_emb + tts_pad_emb

            out = hf_model(inputs_embeds=decode_input, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            logits = out.logits[0, -1, :3072].float()
            cb0 = logits.argmax().item()
            cb0_tokens.append(cb0)

            if step < 20 or step % 10 == 0:
                eos_logit = logits[codec_eos_id].item()
                print(f"Step {step}: CB0={cb0}, EOS_logit={eos_logit:.2f}")

    print(f"\nGenerated {len(cb0_tokens)} CB0 tokens (greedy)")
    print(f"Token range: [{min(cb0_tokens)}, {max(cb0_tokens)}]")
    print(f"Unique tokens: {len(set(cb0_tokens))}")
    print(f"First 30: {cb0_tokens[:30]}")

    # Check for repetition patterns
    from collections import Counter
    token_counts = Counter(cb0_tokens)
    most_common = token_counts.most_common(5)
    print(f"Most common: {most_common}")


if __name__ == "__main__":
    main()
