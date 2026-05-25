# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compare HF vs TT for the first decode step after prefill.

After prefill produces CB0 token, this script:
1. Runs Code Predictor to get CB1-15
2. Builds the decode input (sum of 16 codec embeddings + trailing_text_hidden)
3. Runs HF Qwen3 decode step (CPU)
4. Runs TT decode step (device)
5. Compares the decode logits

Usage (inside Docker, run in sequence):
    python .../compare_decode_step.py hf    # builds HF model, runs decode, saves state
    python .../compare_decode_step.py tt    # runs TT decode, compares with HF

Or run both (slow):
    python .../compare_decode_step.py both
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


def build_embeddings():
    """Build input embeddings (shared between HF and TT)."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    config_path = hf_hub_download(MODEL_PATH, "config.json")
    with open(config_path) as f:
        raw_config = json.load(f)
    tc = raw_config["talker_config"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Load embedding weights
    from huggingface_hub import HfApi
    from safetensors.torch import load_file

    api = HfApi()
    files = api.list_repo_files(MODEL_PATH)
    shard_files = sorted([f for f in files if f.endswith(".safetensors")])

    needed = {
        "talker.model.text_embedding.weight",
        "talker.model.codec_embedding.weight",
        "talker.text_projection.linear_fc1.weight",
        "talker.text_projection.linear_fc1.bias",
        "talker.text_projection.linear_fc2.weight",
        "talker.text_projection.linear_fc2.bias",
    }
    weights = {}
    for shard in shard_files:
        fpath = hf_hub_download(MODEL_PATH, shard)
        sd = load_file(fpath)
        for k in list(sd.keys()):
            if k in needed:
                weights[k] = sd[k]
        del sd
        if len(weights) == len(needed):
            break

    text_embed_w = weights["talker.model.text_embedding.weight"].to(torch.bfloat16)
    codec_embed_w = weights["talker.model.codec_embedding.weight"].to(torch.bfloat16)
    tp1_w = weights["talker.text_projection.linear_fc1.weight"].to(torch.bfloat16)
    tp1_b = weights["talker.text_projection.linear_fc1.bias"].to(torch.bfloat16)
    tp2_w = weights["talker.text_projection.linear_fc2.weight"].to(torch.bfloat16)
    tp2_b = weights["talker.text_projection.linear_fc2.bias"].to(torch.bfloat16)

    def text_embed(ids): return torch.nn.functional.embedding(ids, text_embed_w)
    def codec_embed(ids): return torch.nn.functional.embedding(ids, codec_embed_w)
    def text_proj(x):
        h = torch.nn.functional.linear(x, tp1_w, tp1_b)
        h = torch.nn.functional.silu(h)
        return torch.nn.functional.linear(h, tp2_w, tp2_b)

    # Tokenize
    formatted = f"<|im_start|>assistant\n{TEXT}<|im_end|>\n<|im_start|>assistant\n"
    token_ids = tokenizer.encode(formatted)
    input_ids = torch.tensor([token_ids], dtype=torch.long)

    # Special tokens
    tts_bos_id = raw_config.get("tts_bos_token_id", 151672)
    tts_eos_id = raw_config.get("tts_eos_token_id", 151673)
    tts_pad_id = raw_config.get("tts_pad_token_id", 151671)
    sp = text_proj(text_embed(torch.tensor([[tts_bos_id, tts_eos_id, tts_pad_id]])))
    tts_bos_emb, tts_eos_emb, tts_pad_emb = sp[:, 0:1], sp[:, 1:2], sp[:, 2:3]

    language_id = tc["codec_language_id"][LANGUAGE.lower()]
    codec_pad_id = tc.get("codec_pad_id", 2148)
    codec_bos_id = tc.get("codec_bos_id", 2149)
    codec_think_id = tc.get("codec_think_id", 2154)
    codec_think_bos_id = tc.get("codec_think_bos_id", 2156)
    codec_think_eos_id = tc.get("codec_think_eos_id", 2157)

    codec_tag = codec_embed(torch.tensor([[codec_think_id, codec_think_bos_id, language_id, codec_think_eos_id]]))
    codec_suffix = codec_embed(torch.tensor([[codec_pad_id, codec_bos_id]]))
    codec_prefill = torch.cat([codec_tag, codec_suffix], dim=1)

    role_proj = text_proj(text_embed(input_ids[:, :3]))
    n_codec_m1 = codec_prefill.shape[1] - 1
    text_side_tag = torch.cat([tts_pad_emb.expand(-1, n_codec_m1 - 1, -1), tts_bos_emb], dim=1)
    part_tag = text_side_tag + codec_prefill[:, :-1, :]

    text_content_ids = input_ids[:, 3:-5]
    text_content_proj = text_proj(text_embed(text_content_ids))
    text_with_eos = torch.cat([text_content_proj, tts_eos_emb], dim=1)
    codec_pad_emb_codec = codec_embed(torch.tensor([[codec_pad_id]]))
    part_text = text_with_eos + codec_pad_emb_codec.expand(-1, text_with_eos.shape[1], -1)

    part_final = tts_pad_emb + codec_prefill[:, -1:, :]
    full_embed = torch.cat([role_proj, part_tag, part_text, part_final], dim=1)

    return full_embed, tts_pad_emb, codec_embed_w, raw_config


def run_hf_prefill_and_decode(full_embed, tts_pad_emb, codec_embed_w, raw_config):
    """Run HF Qwen3 Talker: prefill + first decode step."""
    from huggingface_hub import hf_hub_download, HfApi
    from safetensors.torch import load_file
    from transformers import Qwen3ForCausalLM, Qwen3Config

    tc = raw_config["talker_config"]

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

    # Load weights
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

    missing = set(hf_sd.keys()) - set(mapped.keys())
    print(f"Mapped {len(mapped)}/{len(hf_sd)} keys, missing: {len(missing)}")
    hf_model.load_state_dict(mapped, strict=False)
    hf_model = hf_model.to(torch.bfloat16).eval()

    # --- Prefill ---
    print(f"\nPrefill: input shape={full_embed.shape}")
    with torch.no_grad():
        hf_out = hf_model(inputs_embeds=full_embed, use_cache=True)
    past_kv = hf_out.past_key_values
    prefill_logits = hf_out.logits[0, -1, :3072].float()
    cb0_token = prefill_logits.argmax().item()
    print(f"Prefill argmax CB0: {cb0_token}")

    # --- Build decode input ---
    # For the first decode step, we only have CB0 (no Code Predictor in this test).
    # In the real pipeline, Code Predictor generates CB1-15 and we sum all 16.
    # For comparison purposes, use just CB0 + trailing_text_hidden (matching what a
    # simple first-step comparison would test).
    # Actually let's do the FULL decode input: CB0 embedding + 15 zero-embeddings + tts_pad
    cb0_emb = torch.nn.functional.embedding(
        torch.tensor([[cb0_token]], dtype=torch.long), codec_embed_w
    )  # [1, 1, 2048]

    # Simple: CB0 only + trailing_text_hidden (= tts_pad_embed)
    decode_input_simple = cb0_emb + tts_pad_emb  # [1, 1, 2048]

    print(f"Decode input: CB0={cb0_token}, input norm={decode_input_simple.float().norm():.4f}")

    # --- First decode step ---
    with torch.no_grad():
        hf_decode_out = hf_model(
            inputs_embeds=decode_input_simple,
            past_key_values=past_kv,
            use_cache=True,
        )
    decode_logits = hf_decode_out.logits[0, -1, :3072].float()
    decode_cb0 = decode_logits.argmax().item()

    print(f"\n=== HF Decode Step 1 Logits ===")
    top5 = decode_logits.topk(5)
    for i, (v, idx) in enumerate(zip(top5.values, top5.indices)):
        print(f"  Top-{i+1}: token={idx.item()}, logit={v.item():.4f}")
    eos_id = tc.get("codec_eos_token_id", 2150)
    print(f"  EOS ({eos_id}) logit: {decode_logits[eos_id].item():.4f}")
    print(f"  Argmax: {decode_cb0}")

    # Also do step 2 with only CB0 for comparison
    cb1_emb = torch.nn.functional.embedding(
        torch.tensor([[decode_cb0]], dtype=torch.long), codec_embed_w
    )
    decode_input_2 = cb1_emb + tts_pad_emb
    with torch.no_grad():
        hf_decode_out2 = hf_model(
            inputs_embeds=decode_input_2,
            past_key_values=hf_decode_out.past_key_values,
            use_cache=True,
        )
    decode_logits_2 = hf_decode_out2.logits[0, -1, :3072].float()
    print(f"\n=== HF Decode Step 2 Logits ===")
    top5_2 = decode_logits_2.topk(5)
    for i, (v, idx) in enumerate(zip(top5_2.values, top5_2.indices)):
        print(f"  Top-{i+1}: token={idx.item()}, logit={v.item():.4f}")
    print(f"  EOS ({eos_id}) logit: {decode_logits_2[eos_id].item():.4f}")
    print(f"  Argmax: {decode_logits_2.argmax().item()}")

    return {
        "prefill_logits": prefill_logits,
        "cb0_token": cb0_token,
        "decode_logits_1": decode_logits,
        "decode_cb0_1": decode_cb0,
        "decode_logits_2": decode_logits_2,
    }


def run_tt_prefill_and_decode(full_embed, tts_pad_emb, codec_embed_w, hf_results):
    """Run TT Talker: prefill + first decode step, compare with HF."""
    import ttnn
    from models.demos.qwen3_tts.tt.model_config import TalkerModelArgs
    from models.demos.qwen3_tts.tt.talker import TalkerTransformer
    from models.tt_transformers.tt.common import Mode

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

    # --- Prefill ---
    seq_len = full_embed.shape[1]
    last_token_idx = seq_len - 1
    padded_len = math.ceil(seq_len / 128) * 128
    padded_embed = torch.nn.functional.pad(full_embed, (0, 0, 0, padded_len - seq_len))

    tokens_embd, rot_mats, rot_mats_local, tt_page_table, tt_chunk_page_table = (
        talker.prepare_inputs_prefill(padded_embed, start_pos=0, last_token_idx=last_token_idx)
    )
    get_last_token = (last_token_idx // 32) * 32

    logits_tt = talker.ttnn_prefill_forward(
        tokens_embd, rot_mats_global=rot_mats, rot_mats_local=rot_mats_local,
        page_table=tt_page_table, chunk_page_table=tt_chunk_page_table,
        get_last_token=get_last_token, pre_projected=True,
    )
    logits = talker.process_output_prefill(logits_tt.cpu(), last_token_idx=last_token_idx % 32)
    logits = logits.view(1, talker_args.vocab_size).float()
    tt_cb0 = logits.argmax().item()
    print(f"\nTT Prefill argmax CB0: {tt_cb0}")
    print(f"HF Prefill argmax CB0: {hf_results['cb0_token']}")

    # --- First decode step ---
    cb0_token = hf_results["cb0_token"]  # Use same CB0 for fair comparison
    cb0_emb = torch.nn.functional.embedding(
        torch.tensor([[cb0_token]], dtype=torch.long), codec_embed_w
    )
    decode_input = cb0_emb + tts_pad_emb  # [1, 1, 2048]

    prefill_len = last_token_idx + 1
    current_pos = torch.tensor([prefill_len], dtype=torch.int64)
    padded_pos = torch.nn.functional.pad(current_pos, (0, talker_args.max_batch_size - 1), value=0)

    dummy_tokens = torch.zeros(1, talker_args.max_batch_size, dtype=torch.long)
    _, tt_pos, tt_rot_idxs, tt_page_table_decode = talker.prepare_inputs_decode(
        dummy_tokens, padded_pos
    )

    decode_padded = torch.zeros(1, 1, 32, talker_args.dim)
    decode_padded[0, 0, 0, :] = decode_input[0, 0, :]
    tt_decode = ttnn.from_torch(
        decode_padded, device=mesh_device, dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    decode_mem = talker_args.get_residual_mem_config(Mode.DECODE, talker.prefetcher)
    tt_decode = ttnn.to_memory_config(tt_decode, decode_mem)

    tt_logits, tt_hidden = talker.ttnn_decode_forward_preembedded(
        tt_decode, tt_pos, rot_mat_idxs=tt_rot_idxs, page_table=tt_page_table_decode
    )

    decode_logits = talker.process_output_decode(tt_logits.cpu(), B=1)
    decode_logits = decode_logits[:, :, :talker_args.vocab_size].view(1, 3072).float()

    print(f"\n=== TT Decode Step 1 Logits ===")
    top5 = decode_logits[0].topk(5)
    for i, (v, idx) in enumerate(zip(top5.values, top5.indices)):
        print(f"  Top-{i+1}: token={idx.item()}, logit={v.item():.4f}")
    print(f"  EOS (2150) logit: {decode_logits[0, 2150].item():.4f}")
    print(f"  Argmax: {decode_logits[0].argmax().item()}")

    # Compare with HF
    hf_d1 = hf_results["decode_logits_1"]
    tt_d1 = decode_logits[0]
    cos = torch.nn.functional.cosine_similarity(hf_d1.unsqueeze(0), tt_d1.unsqueeze(0)).item()
    maxd = (hf_d1 - tt_d1).abs().max().item()
    print(f"\n=== Decode Step 1 Comparison ===")
    print(f"  Cosine similarity: {cos:.6f}")
    print(f"  Max abs diff: {maxd:.4f}")
    print(f"  HF argmax: {hf_d1.argmax().item()}, TT argmax: {tt_d1.argmax().item()}, match: {hf_d1.argmax().item() == tt_d1.argmax().item()}")
    hf_top10 = set(hf_d1.topk(10).indices.tolist())
    tt_top10 = set(tt_d1.topk(10).indices.tolist())
    print(f"  Top-10 overlap: {len(hf_top10 & tt_top10)}/10")

    # --- Second decode step ---
    tt_cb0_step1 = decode_logits[0].argmax().item()
    cb1_emb = torch.nn.functional.embedding(
        torch.tensor([[tt_cb0_step1]], dtype=torch.long), codec_embed_w
    )
    decode_input_2 = cb1_emb + tts_pad_emb

    current_pos_2 = torch.tensor([prefill_len + 1], dtype=torch.int64)
    padded_pos_2 = torch.nn.functional.pad(current_pos_2, (0, talker_args.max_batch_size - 1), value=0)
    _, tt_pos_2, tt_rot_idxs_2, tt_page_table_2 = talker.prepare_inputs_decode(
        dummy_tokens, padded_pos_2
    )

    decode_padded_2 = torch.zeros(1, 1, 32, talker_args.dim)
    decode_padded_2[0, 0, 0, :] = decode_input_2[0, 0, :]
    tt_decode_2 = ttnn.from_torch(
        decode_padded_2, device=mesh_device, dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_decode_2 = ttnn.to_memory_config(tt_decode_2, decode_mem)

    tt_logits_2, _ = talker.ttnn_decode_forward_preembedded(
        tt_decode_2, tt_pos_2, rot_mat_idxs=tt_rot_idxs_2, page_table=tt_page_table_2
    )
    decode_logits_2 = talker.process_output_decode(tt_logits_2.cpu(), B=1)
    decode_logits_2 = decode_logits_2[:, :, :talker_args.vocab_size].view(1, 3072).float()

    print(f"\n=== TT Decode Step 2 Logits ===")
    top5_2 = decode_logits_2[0].topk(5)
    for i, (v, idx) in enumerate(zip(top5_2.values, top5_2.indices)):
        print(f"  Top-{i+1}: token={idx.item()}, logit={v.item():.4f}")
    print(f"  EOS (2150) logit: {decode_logits_2[0, 2150].item():.4f}")
    print(f"  Argmax: {decode_logits_2[0].argmax().item()}")

    if hf_results.get("decode_logits_2") is not None:
        hf_d2 = hf_results["decode_logits_2"]
        tt_d2 = decode_logits_2[0]
        cos2 = torch.nn.functional.cosine_similarity(hf_d2.unsqueeze(0), tt_d2.unsqueeze(0)).item()
        print(f"\n=== Decode Step 2 Comparison ===")
        print(f"  Cosine similarity: {cos2:.6f}")
        print(f"  HF argmax: {hf_d2.argmax().item()}, TT argmax: {tt_d2.argmax().item()}")

    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    full_embed, tts_pad_emb, codec_embed_w, raw_config = build_embeddings()

    if mode == "hf":
        results = run_hf_prefill_and_decode(full_embed, tts_pad_emb, codec_embed_w, raw_config)
        torch.save(results, "/tmp/hf_decode_results.pt")
        print("\nSaved HF results to /tmp/hf_decode_results.pt")

    elif mode == "tt":
        hf_results = torch.load("/tmp/hf_decode_results.pt")
        run_tt_prefill_and_decode(full_embed, tts_pad_emb, codec_embed_w, hf_results)

    elif mode == "both":
        results = run_hf_prefill_and_decode(full_embed, tts_pad_emb, codec_embed_w, raw_config)
        torch.save(results, "/tmp/hf_decode_results.pt")
        run_tt_prefill_and_decode(full_embed, tts_pad_emb, codec_embed_w, results)
