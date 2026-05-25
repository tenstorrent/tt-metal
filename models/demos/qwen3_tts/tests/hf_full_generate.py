# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run HF Talker + Code Predictor on CPU to check if EOS is reached.

Uses both the Talker (Qwen3 1.7B, generates CB0) and Code Predictor
(Qwen3-style 5L, generates CB1-15) to build the full decode input at
each step, matching the HF pipeline.
"""

import json
import os
import sys

import torch

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TEXT = sys.argv[1] if len(sys.argv) > 1 else "こんにちは"
LANGUAGE = "japanese"
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 100


def main():
    from huggingface_hub import hf_hub_download, HfApi
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3ForCausalLM, Qwen3Config

    config_path = hf_hub_download(MODEL_PATH, "config.json")
    with open(config_path) as f:
        raw_config = json.load(f)
    tc = raw_config["talker_config"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Load ALL weights
    print("Loading all weights...")
    api = HfApi()
    files = api.list_repo_files(MODEL_PATH)
    shard_files = sorted([f for f in files if f.endswith(".safetensors")])
    all_weights = {}
    for shard in shard_files:
        fpath = hf_hub_download(MODEL_PATH, shard)
        sd = load_file(fpath)
        all_weights.update(sd)
        del sd
    print(f"Loaded {len(all_weights)} total keys")

    talker_weights = {k: v for k, v in all_weights.items() if k.startswith("talker.")}
    print(f"Talker keys: {len(talker_weights)}")

    # --- Build Talker ---
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
    talker_model = Qwen3ForCausalLM(qwen3_config)
    hf_sd = talker_model.state_dict()
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
    talker_model.load_state_dict(mapped, strict=False)
    talker_model = talker_model.to(torch.bfloat16).eval()
    print("Talker loaded")

    # --- Build Code Predictor ---
    cp_cfg = tc.get("code_predictor_config", {})
    cp_hidden = cp_cfg.get("hidden_size", 1024)
    cp_layers = cp_cfg.get("num_hidden_layers", 5)
    cp_heads = cp_cfg.get("num_attention_heads", 16)
    cp_kv_heads = cp_cfg.get("num_key_value_heads", 8)
    cp_inter = cp_cfg.get("intermediate_size", 3072)

    cp_qwen3_config = Qwen3Config(
        vocab_size=2048,
        hidden_size=cp_hidden,
        intermediate_size=cp_inter,
        num_hidden_layers=cp_layers,
        num_attention_heads=cp_heads,
        num_key_value_heads=cp_kv_heads,
        hidden_act="silu",
        max_position_embeddings=65536,
        rms_norm_eps=cp_cfg.get("rms_norm_eps", 1e-6),
        rope_theta=cp_cfg.get("rope_theta", 1000000.0),
        tie_word_embeddings=False,
        head_dim=cp_cfg.get("head_dim", 128),
    )
    print(f"CP config: layers={cp_layers}, dim={cp_hidden}, heads={cp_heads}/{cp_kv_heads}")

    cp_model = Qwen3ForCausalLM(cp_qwen3_config)
    cp_sd = cp_model.state_dict()
    cp_mapped = {}
    for hf_key in cp_sd:
        if hf_key == "lm_head.weight":
            continue  # CP uses 15 separate lm_heads
        elif hf_key == "model.embed_tokens.weight":
            continue  # CP uses input_projection instead
        elif hf_key.startswith("model."):
            src = f"talker.code_predictor.{hf_key}"
        else:
            src = None
        if src and src in talker_weights:
            cp_mapped[hf_key] = talker_weights[src]
        else:
            alt = f"talker.code_predictor.model.{hf_key.removeprefix('model.')}"
            if alt in talker_weights:
                cp_mapped[hf_key] = talker_weights[alt]
    cp_model.load_state_dict(cp_mapped, strict=False)
    cp_model = cp_model.to(torch.bfloat16).eval()
    print(f"Code Predictor loaded ({len(cp_mapped)}/{len(cp_sd)} keys)")

    # CP projection and heads
    cp_proj_w = talker_weights["talker.code_predictor.small_to_mtp_projection.weight"].to(torch.bfloat16)
    cp_proj_b = talker_weights["talker.code_predictor.small_to_mtp_projection.bias"].to(torch.bfloat16)
    cp_lm_heads = []
    for i in range(15):
        w = talker_weights[f"talker.code_predictor.lm_head.{i}.weight"].to(torch.bfloat16)
        cp_lm_heads.append(w)
    cp_codec_embeds = []
    for i in range(15):
        w = talker_weights[f"talker.code_predictor.model.codec_embedding.{i}.weight"].to(torch.bfloat16)
        cp_codec_embeds.append(w)

    def cp_project(x):
        return torch.nn.functional.linear(x, cp_proj_w, cp_proj_b)

    # --- Build embeddings ---
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

    print(f"\nPrefill: {full_embed.shape}")
    codec_eos_id = tc.get("codec_eos_token_id", 2150)

    # --- Generation loop ---
    cb0_tokens = []
    all_frames = []

    with torch.no_grad():
        # Prefill Talker
        talker_out = talker_model(inputs_embeds=full_embed, use_cache=True, output_hidden_states=True)
        talker_kv = talker_out.past_key_values
        # past_hidden = hidden state at last position (pre-norm? The HF code uses hidden_states[-1][:, -1:])
        past_hidden = talker_out.hidden_states[-1][:, -1:, :]  # [1, 1, 2048]

        logits = talker_out.logits[0, -1, :3072].float()
        cb0 = logits.argmax().item()
        cb0_tokens.append(cb0)
        print(f"Step 0: CB0={cb0}, EOS_logit={logits[codec_eos_id].item():.2f}")

        for step in range(1, MAX_STEPS):
            if cb0 == codec_eos_id:
                print(f"EOS at step {step}!")
                break

            # CB0 embedding
            cb0_emb = torch.nn.functional.embedding(
                torch.tensor([[cb0]], dtype=torch.long), codec_embed_w
            )  # [1, 1, 2048]

            # --- Run Code Predictor: generate CB1-15 ---
            cp_input = torch.cat([past_hidden, cb0_emb], dim=1)  # [1, 2, 2048]
            cp_input_proj = cp_project(cp_input)  # [1, 2, 1024]

            cp_out = cp_model(inputs_embeds=cp_input_proj, use_cache=True)
            cp_kv = cp_out.past_key_values
            # CP output hidden at last pos → lm_head[0] → CB1
            cp_hidden_last = cp_out.hidden_states[-1] if hasattr(cp_out, 'hidden_states') and cp_out.hidden_states else None

            # Actually, cp_model is Qwen3ForCausalLM which outputs logits. But we need custom lm_heads.
            # Get the hidden state from the model's last layer output
            # Re-run with output_hidden_states=True
            cp_out = cp_model(inputs_embeds=cp_input_proj, use_cache=True, output_hidden_states=True)
            cp_kv = cp_out.past_key_values
            cp_hidden_last = cp_out.hidden_states[-1][:, -1:, :]  # [1, 1, 1024]

            # CB1 from lm_head[0]
            cb1_logits = torch.nn.functional.linear(cp_hidden_last.squeeze(1), cp_lm_heads[0])
            cb1 = cb1_logits.argmax(-1).item()
            frame_tokens = [cb0, cb1]

            # CB2-15
            prev_tok = cb1
            for cb_step in range(1, 15):
                tok_emb = torch.nn.functional.embedding(
                    torch.tensor([[prev_tok]], dtype=torch.long), cp_codec_embeds[cb_step - 1]
                )  # [1, 1, 2048]
                tok_proj = cp_project(tok_emb)  # [1, 1, 1024]
                cp_out = cp_model(inputs_embeds=tok_proj, past_key_values=cp_kv, use_cache=True, output_hidden_states=True)
                cp_kv = cp_out.past_key_values
                cp_h = cp_out.hidden_states[-1][:, -1:, :]
                cb_logits = torch.nn.functional.linear(cp_h.squeeze(1), cp_lm_heads[cb_step])
                cb_tok = cb_logits.argmax(-1).item()
                frame_tokens.append(cb_tok)
                prev_tok = cb_tok

            all_frames.append(frame_tokens)

            # Build decode input: sum ALL 16 codec embeddings + trailing_text_hidden
            # CB0: from Talker's codec_embedding
            cb_sum = cb0_emb.clone()
            for ci in range(15):
                ci_emb = torch.nn.functional.embedding(
                    torch.tensor([[frame_tokens[ci + 1]]], dtype=torch.long), cp_codec_embeds[ci]
                )
                cb_sum = cb_sum + ci_emb
            decode_input = cb_sum + tts_pad_emb  # [1, 1, 2048]

            # Talker decode step
            talker_out = talker_model(
                inputs_embeds=decode_input, past_key_values=talker_kv,
                use_cache=True, output_hidden_states=True,
            )
            talker_kv = talker_out.past_key_values
            past_hidden = talker_out.hidden_states[-1][:, -1:, :]

            logits = talker_out.logits[0, -1, :3072].float()
            cb0 = logits.argmax().item()
            cb0_tokens.append(cb0)

            eos_logit = logits[codec_eos_id].item()
            if step < 20 or step % 10 == 0 or eos_logit > logits.max().item() - 3:
                print(f"Step {step}: CB0={cb0}, EOS_logit={eos_logit:.2f}, max_logit={logits.max().item():.2f}")

    print(f"\nGenerated {len(cb0_tokens)} CB0 tokens (greedy)")
    print(f"Unique: {len(set(cb0_tokens))}, range: [{min(cb0_tokens)}, {max(cb0_tokens)}]")
    print(f"First 30: {cb0_tokens[:30]}")
    if all_frames:
        print(f"Frame 0 (all 16 CBs): {all_frames[0]}")
        print(f"Frame 1 (all 16 CBs): {all_frames[1] if len(all_frames) > 1 else 'N/A'}")


if __name__ == "__main__":
    main()
