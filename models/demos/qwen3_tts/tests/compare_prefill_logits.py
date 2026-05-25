# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compare HF Talker prefill logits vs TT Talker prefill logits.

Builds a Qwen2-based HF Talker on CPU, runs prefill with the same input embeddings,
and compares top-k tokens and logit distributions vs the TT Talker.

Usage (inside Docker):
    # Step 1: run HF on CPU (no GPU needed)
    python models/demos/qwen3_tts/tests/compare_prefill_logits.py hf

    # Step 2: run TT on device, compare
    python models/demos/qwen3_tts/tests/compare_prefill_logits.py tt

    # Or both (slow but complete)
    python models/demos/qwen3_tts/tests/compare_prefill_logits.py both
"""

import json
import math
import os
import sys

import torch
import numpy as np

MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TEXT = "こんにちは"
LANGUAGE = "japanese"


def build_hf_talker_and_run():
    """Build HF Talker from raw weights on CPU, run prefill, return logits."""
    from huggingface_hub import hf_hub_download, HfApi
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3ForCausalLM, Qwen3Config

    config_path = hf_hub_download(MODEL_PATH, "config.json")
    with open(config_path) as f:
        raw_config = json.load(f)

    tc = raw_config["talker_config"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Build Qwen3 config matching the Talker architecture
    # Qwen3 has QK-norm (head_dim=128), no QKV bias, same as Talker
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
    print(f"Qwen3Config: layers={qwen3_config.num_hidden_layers}, "
          f"heads={qwen3_config.num_attention_heads}/{qwen3_config.num_key_value_heads}, "
          f"dim={qwen3_config.hidden_size}, vocab={qwen3_config.vocab_size}, "
          f"head_dim={qwen3_config.head_dim}")

    # Load ALL talker weights
    print("Loading talker weights from safetensors...")
    api = HfApi()
    files = api.list_repo_files(MODEL_PATH)
    shard_files = sorted([f for f in files if f.endswith(".safetensors")])

    talker_weights = {}
    for shard in shard_files:
        fpath = hf_hub_download(MODEL_PATH, shard)
        shard_dict = load_file(fpath)
        for k, v in shard_dict.items():
            if k.startswith("talker."):
                talker_weights[k] = v
        del shard_dict

    print(f"Loaded {len(talker_weights)} talker weight keys")

    # Map HF Talker weights to Qwen3ForCausalLM
    # HF Talker key: talker.model.layers.N.self_attn.{q,k,v,o}_proj.weight
    # HF Talker key: talker.model.layers.N.self_attn.{q,k}_norm.weight
    # Qwen3ForCausalLM: model.layers.N.self_attn.{same}
    # Additional: talker.model.codec_embedding.weight -> model.embed_tokens.weight
    #             talker.codec_head.weight -> lm_head.weight

    hf_model = Qwen3ForCausalLM(qwen3_config)
    hf_sd = hf_model.state_dict()
    mapped = {}

    for hf_key in hf_sd:
        if hf_key == "lm_head.weight":
            src = "talker.codec_head.weight"
        elif hf_key == "model.embed_tokens.weight":
            src = "talker.model.codec_embedding.weight"
        elif hf_key.startswith("model."):
            # Qwen3ForCausalLM uses model.layers.N... ; HF Talker uses talker.model.layers.N...
            src = f"talker.{hf_key}"
        else:
            src = None

        if src and src in talker_weights:
            mapped[hf_key] = talker_weights[src]
        else:
            alt = f"talker.model.{hf_key.removeprefix('model.')}"
            if alt in talker_weights:
                mapped[hf_key] = talker_weights[alt]

    missing = set(hf_sd.keys()) - set(mapped.keys())
    print(f"Mapped {len(mapped)}/{len(hf_sd)} keys")
    if missing:
        print(f"Missing ({len(missing)}): {sorted(missing)[:10]}")

    hf_model.load_state_dict(mapped, strict=False)
    hf_model = hf_model.to(torch.bfloat16).eval()
    print("HF Qwen3 Talker loaded on CPU (bfloat16)")

    # --- Build input embeddings ---
    text_embed_weight = talker_weights["talker.model.text_embedding.weight"].to(torch.bfloat16)
    codec_embed_weight = talker_weights["talker.model.codec_embedding.weight"].to(torch.bfloat16)
    tp_fc1_w = talker_weights["talker.text_projection.linear_fc1.weight"].to(torch.bfloat16)
    tp_fc1_b = talker_weights["talker.text_projection.linear_fc1.bias"].to(torch.bfloat16)
    tp_fc2_w = talker_weights["talker.text_projection.linear_fc2.weight"].to(torch.bfloat16)
    tp_fc2_b = talker_weights["talker.text_projection.linear_fc2.bias"].to(torch.bfloat16)

    def text_embed_fn(ids):
        return torch.nn.functional.embedding(ids, text_embed_weight)

    def codec_embed_fn(ids):
        return torch.nn.functional.embedding(ids, codec_embed_weight)

    def text_proj_fn(x):
        h = torch.nn.functional.linear(x, tp_fc1_w, tp_fc1_b)
        h = torch.nn.functional.silu(h)
        h = torch.nn.functional.linear(h, tp_fc2_w, tp_fc2_b)
        return h

    # Tokenize
    formatted = f"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n"
    token_ids = tokenizer.encode(formatted)
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    print(f"\nToken IDs ({len(token_ids)}): {token_ids}")

    # Special TTS tokens
    tts_bos_id = raw_config.get("tts_bos_token_id", 151672)
    tts_eos_id = raw_config.get("tts_eos_token_id", 151673)
    tts_pad_id = raw_config.get("tts_pad_token_id", 151671)

    special_ids = torch.tensor([[tts_bos_id, tts_eos_id, tts_pad_id]])
    special_proj = text_proj_fn(text_embed_fn(special_ids))
    tts_bos_embed = special_proj[:, 0:1, :]
    tts_eos_embed = special_proj[:, 1:2, :]
    tts_pad_embed = special_proj[:, 2:3, :]

    # Codec embeddings
    language_id = tc["codec_language_id"][LANGUAGE.lower()]
    codec_think_id = tc.get("codec_think_id", tc.get("codec_nothink_id", 2155))
    codec_think_bos_id = tc.get("codec_think_bos_id", 2156)
    codec_think_eos_id = tc.get("codec_think_eos_id", 2157)
    codec_pad_id = tc.get("codec_pad_id", 2148)
    codec_bos_id = tc.get("codec_bos_id", 2149)

    print(f"Codec IDs: think={codec_think_id}, think_bos={codec_think_bos_id}, "
          f"think_eos={codec_think_eos_id}, pad={codec_pad_id}, bos={codec_bos_id}, lang={language_id}")

    codec_tag = codec_embed_fn(torch.tensor([[codec_think_id, codec_think_bos_id, language_id, codec_think_eos_id]]))
    codec_suffix = codec_embed_fn(torch.tensor([[codec_pad_id, codec_bos_id]]))
    codec_prefill = torch.cat([codec_tag, codec_suffix], dim=1)

    # Role prefix
    role_proj = text_proj_fn(text_embed_fn(input_ids[:, :3]))

    # Tag overlay
    n_codec_m1 = codec_prefill.shape[1] - 1
    text_side_tag = torch.cat([
        tts_pad_embed.expand(-1, n_codec_m1 - 1, -1),
        tts_bos_embed,
    ], dim=1)
    part_tag = text_side_tag + codec_prefill[:, :-1, :]

    # Text content
    text_content_ids = input_ids[:, 3:-5]
    N_text = text_content_ids.shape[1]
    text_content_proj = text_proj_fn(text_embed_fn(text_content_ids))
    text_with_eos = torch.cat([text_content_proj, tts_eos_embed], dim=1)
    codec_pad_emb = codec_embed_fn(torch.tensor([[codec_pad_id]]))
    codec_pads = codec_pad_emb.expand(-1, text_with_eos.shape[1], -1)
    part_text = text_with_eos + codec_pads

    # Final
    part_final = tts_pad_embed + codec_prefill[:, -1:, :]

    full_embed = torch.cat([role_proj, part_tag, part_text, part_final], dim=1)
    print(f"Full embedding: shape={full_embed.shape}")
    for i in range(full_embed.shape[1]):
        print(f"  pos {i}: norm={full_embed[0,i].float().norm():.4f}")

    # Run HF Talker forward (CPU)
    print("\nRunning HF Talker forward on CPU...")
    with torch.no_grad():
        hf_out = hf_model(inputs_embeds=full_embed)
    hf_logits = hf_out.logits[0, -1, :3072].float()

    print(f"\n=== HF Talker Prefill Logits (last position, seq_len={full_embed.shape[1]}) ===")
    top10_vals, top10_idx = hf_logits.topk(10)
    for i, (v, idx) in enumerate(zip(top10_vals, top10_idx)):
        print(f"  Top-{i+1}: token={idx.item()}, logit={v.item():.4f}")

    codec_eos_id = tc.get("codec_eos_token_id", 2150)
    print(f"  EOS (token {codec_eos_id}) logit: {hf_logits[codec_eos_id].item():.4f}")
    print(f"  Argmax: {hf_logits.argmax().item()}")

    # Also check all position logits
    all_logits = hf_out.logits[0, :, :3072].float()
    print(f"\n  All-position argmax (last 5):")
    for p in range(max(0, full_embed.shape[1]-5), full_embed.shape[1]):
        am = all_logits[p].argmax().item()
        eos_val = all_logits[p, codec_eos_id].item()
        print(f"    pos {p}: argmax={am}, eos_logit={eos_val:.4f}")

    return full_embed.cpu(), hf_logits.cpu()


def build_tt_and_compare(hf_embed, hf_logits):
    """Build TT Talker, run prefill with the SAME embeddings, compare logits."""
    import ttnn
    from models.demos.qwen3_tts.tt.model_config import TalkerModelArgs
    from models.demos.qwen3_tts.tt.talker import TalkerTransformer

    os.environ["HF_MODEL"] = MODEL_PATH

    device_ids = ttnn.get_device_ids()
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, len(device_ids)),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )
    try:
        mesh_device.enable_program_cache()
    except AttributeError:
        ttnn.enable_program_cache(mesh_device)

    talker_args = TalkerModelArgs(
        mesh_device=mesh_device, max_batch_size=1, max_seq_len=768, use_hf_rope=True,
    )
    state_dict = talker_args.load_state_dict()
    weight_cache_path = talker_args.weight_cache_path(ttnn.bfloat16)

    talker = TalkerTransformer(
        args=talker_args, dtype=ttnn.bfloat16, mesh_device=mesh_device,
        state_dict=state_dict, weight_cache_path=weight_cache_path,
    )

    seq_len = hf_embed.shape[1]
    last_token_idx = seq_len - 1

    # Pad to multiple of 128
    padded_len = math.ceil(seq_len / 128) * 128
    padded_embed = torch.nn.functional.pad(hf_embed, (0, 0, 0, padded_len - seq_len))

    tokens_embd, rot_mats, rot_mats_local, tt_page_table, tt_chunk_page_table = (
        talker.prepare_inputs_prefill(
            padded_embed,
            start_pos=0,
            last_token_idx=last_token_idx,
        )
    )

    get_last_token = (last_token_idx // 32) * 32
    logits_tt = talker.ttnn_prefill_forward(
        tokens_embd,
        rot_mats_global=rot_mats,
        rot_mats_local=rot_mats_local,
        page_table=tt_page_table,
        chunk_page_table=tt_chunk_page_table,
        get_last_token=get_last_token,
        pre_projected=True,
    )

    logits = talker.process_output_prefill(logits_tt.cpu(), last_token_idx=last_token_idx % 32)
    logits = logits.view(1, talker_args.vocab_size).float()

    print(f"\n=== TT Talker Prefill Logits (last position) ===")
    top10_vals, top10_idx = logits.topk(5, dim=-1)
    for i, (v, idx) in enumerate(zip(top10_vals[0], top10_idx[0])):
        print(f"  Top-{i+1}: token={idx.item()}, logit={v.item():.4f}")

    codec_eos_id = 2150
    print(f"  EOS (token {codec_eos_id}) logit: {logits[0, codec_eos_id].item():.4f}")
    print(f"  Argmax: {logits.argmax().item()}")

    # Compare
    print(f"\n=== Comparison ===")
    hf_flat = hf_logits[:3072].float()
    tt_flat = logits[0, :3072].float()

    cos_sim = torch.nn.functional.cosine_similarity(hf_flat.unsqueeze(0), tt_flat.unsqueeze(0)).item()
    max_diff = (hf_flat - tt_flat).abs().max().item()
    mean_diff = (hf_flat - tt_flat).abs().mean().item()
    print(f"  Cosine similarity: {cos_sim:.6f}")
    print(f"  Max abs diff: {max_diff:.4f}")
    print(f"  Mean abs diff: {mean_diff:.4f}")

    hf_argmax = hf_flat.argmax().item()
    tt_argmax = tt_flat.argmax().item()
    print(f"  HF argmax: {hf_argmax}, TT argmax: {tt_argmax}, match: {hf_argmax == tt_argmax}")

    # Top-10 overlap
    hf_top10 = set(hf_flat.topk(10).indices.tolist())
    tt_top10 = set(tt_flat.topk(10).indices.tolist())
    print(f"  HF top-10: {sorted(hf_flat.topk(10).indices.tolist())}")
    print(f"  TT top-10: {sorted(tt_flat.topk(10).indices.tolist())}")
    print(f"  Top-10 overlap: {len(hf_top10 & tt_top10)}/10")

    ttnn.close_mesh_device(mesh_device)
    return logits


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode == "hf":
        hf_embed, hf_logits = build_hf_talker_and_run()
        torch.save({"embed": hf_embed, "logits": hf_logits},
                   "/tmp/hf_prefill_results.pt")
        print("\nSaved HF results to /tmp/hf_prefill_results.pt")

    elif mode == "tt":
        data = torch.load("/tmp/hf_prefill_results.pt")
        build_tt_and_compare(data["embed"], data["logits"])

    elif mode == "both":
        hf_embed, hf_logits = build_hf_talker_and_run()
        torch.save({"embed": hf_embed, "logits": hf_logits},
                   "/tmp/hf_prefill_results.pt")
        build_tt_and_compare(hf_embed, hf_logits)
