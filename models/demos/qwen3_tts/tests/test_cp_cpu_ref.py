# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compare Code Predictor: TT device vs CPU (Qwen3ForCausalLM) reference.

Loads the Code Predictor weights into a Qwen3ForCausalLM model and runs it
on CPU alongside the TT Code Predictor. Compares CB1-15 outputs per frame
for the first N decode steps of "こんにちは" generation.
"""

import os
import traceback

import torch
import torch.nn.functional as F

os.environ["HF_MODEL"] = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

import ttnn
from models.demos.qwen3_tts.tt.generator import TTSGenerator
from models.tt_transformers.tt.common import Mode

# --- Build CPU Code Predictor ---
from transformers import Qwen3ForCausalLM, Qwen3Config
from safetensors.torch import load_file
import glob

def build_cpu_code_predictor():
    """Build a Qwen3ForCausalLM that matches the Code Predictor architecture."""
    cfg = Qwen3Config(
        hidden_size=1024,
        intermediate_size=3072,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        num_hidden_layers=5,
        rms_norm_eps=1e-6,
        rope_theta=1000000,
        vocab_size=2048,
        max_position_embeddings=65536,
    )
    model = Qwen3ForCausalLM(cfg)

    # Load weights from HF checkpoint
    snap_dir = glob.glob("/root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*/")[0]
    st_files = sorted(glob.glob(snap_dir + "*.safetensors"))
    full_sd = {}
    for f in st_files:
        full_sd.update(load_file(f))

    # Map HF checkpoint keys to Qwen3ForCausalLM keys
    cp_prefix = "talker.code_predictor.model."
    mapping = {}
    for key in full_sd:
        if key.startswith(cp_prefix):
            suffix = key[len(cp_prefix):]
            # Map layer keys
            new_key = "model." + suffix
            mapping[key] = new_key

    # Also need norm and build state_dict
    new_sd = {}
    for orig_key, new_key in mapping.items():
        new_sd[new_key] = full_sd[orig_key]

    # Handle norm weight (may be at talker.code_predictor.model.norm.weight)
    norm_key = "talker.code_predictor.model.norm.weight"
    if norm_key in full_sd:
        new_sd["model.norm.weight"] = full_sd[norm_key]

    # LM head — we'll handle this separately per CB
    # The main lm_head is not used (we have 15 separate heads)
    # Set a dummy lm_head
    new_sd["lm_head.weight"] = torch.randn(2048, 1024) * 0.01

    # Load
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"CPU CP model loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  Missing keys (first 10): {missing[:10]}")

    model.eval()

    # Load projection and LM heads separately (convert to float32 for CPU)
    proj_w = full_sd["talker.code_predictor.small_to_mtp_projection.weight"].float()  # [1024, 2048]
    proj_b = full_sd["talker.code_predictor.small_to_mtp_projection.bias"].float()  # [1024]

    lm_heads = []
    for i in range(15):
        w = full_sd[f"talker.code_predictor.lm_head.{i}.weight"].float()  # [2048, 1024]
        lm_heads.append(w)

    # Codec embeddings for CB1-15
    codec_embs = []
    for i in range(15):
        w = full_sd[f"talker.code_predictor.model.codec_embedding.{i}.weight"].float()  # [2048, 2048]
        codec_embs.append(w)

    # Talker codec embedding (for CB0)
    talker_codec_emb = full_sd["talker.model.codec_embedding.weight"].float()  # [3072, 2048]

    return model, proj_w, proj_b, lm_heads, codec_embs, talker_codec_emb


def cpu_predict_codebooks(model, proj_w, proj_b, lm_heads, codec_embs,
                          talker_hidden, cb0_token, talker_codec_emb):
    """Run Code Predictor on CPU using KV-cache (HF-style)."""
    B = talker_hidden.shape[0]

    # Embed CB0
    cb0_emb = F.embedding(cb0_token.unsqueeze(-1), talker_codec_emb)  # [B, 1, 2048]

    # Initial context: [talker_hidden, CB0_embed] in 2048-dim
    context = torch.cat([talker_hidden, cb0_emb], dim=1)  # [B, 2, 2048]

    # Project to 1024-dim
    projected = F.linear(context, proj_w, proj_b)  # [B, 2, 1024]

    generated = [cb0_token.unsqueeze(-1)]

    # Prefill through the model
    with torch.no_grad():
        out = model.model(inputs_embeds=projected, use_cache=True)
        hidden = out.last_hidden_state  # [B, 2, 1024]
        kv_cache = out.past_key_values

    # CB1 from last position
    last_hidden = hidden[:, -1:, :]  # [B, 1, 1024]
    logits_cb1 = F.linear(last_hidden, lm_heads[0])  # [B, 1, 2048]
    cb1_token = torch.argmax(logits_cb1[:, -1, :], dim=-1)  # [B]
    generated.append(cb1_token.unsqueeze(-1))

    # CB2-15: decode with KV cache
    for step in range(1, 15):
        # Embed previous token via codec_embeddings[step-1]
        prev_emb = F.embedding(generated[-1], codec_embs[step - 1])  # [B, 1, 2048]
        prev_projected = F.linear(prev_emb, proj_w, proj_b)  # [B, 1, 1024]

        with torch.no_grad():
            out = model.model(inputs_embeds=prev_projected, past_key_values=kv_cache, use_cache=True)
            hidden = out.last_hidden_state
            kv_cache = out.past_key_values

        logits = F.linear(hidden[:, -1:, :], lm_heads[step])  # [B, 1, 2048]
        next_token = torch.argmax(logits[:, -1, :], dim=-1)  # [B]
        generated.append(next_token.unsqueeze(-1))

    return torch.cat(generated, dim=-1)  # [B, 16]


# --- Main ---
device_ids = ttnn.get_device_ids()
mesh = ttnn.open_mesh_device(
    ttnn.MeshShape(1, len(device_ids)),
    dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
)
try:
    mesh.enable_program_cache()
except AttributeError:
    ttnn.enable_program_cache(mesh)

try:
    # Build TT pipeline
    gen = TTSGenerator.build("Qwen/Qwen3-TTS-12Hz-1.7B-Base", mesh, max_seq_len=2560)
    args = gen.talker_args
    codec_embed_weight = gen.talker.codec_embed_weight

    # Build CPU reference
    print("\n=== Building CPU Code Predictor reference ===")
    cpu_model, proj_w, proj_b, lm_heads, codec_embs, talker_codec_emb = build_cpu_code_predictor()

    # Verify talker codec embedding matches
    cos_sim = F.cosine_similarity(
        talker_codec_emb.float().flatten(), codec_embed_weight.float().flatten(), dim=0
    )
    print(f"Talker codec_embed match: cosine_sim={cos_sim:.6f}")

    # Build input embeds for "こんにちは"
    input_embeds, trailing_text_hidden, tts_pad_embed = gen._build_input_embeds("こんにちは", "japanese")

    # --- Prefill ---
    B = 1
    norms = input_embeds.squeeze(0).norm(dim=-1)
    nonzero_mask = norms > 0
    last_token_idx = nonzero_mask.nonzero()[-1].item()

    tokens_embd, rot_mats, rot_mats_local, tt_page_table, tt_chunk_page_table = (
        gen.talker.prepare_inputs_prefill(input_embeds, start_pos=0, last_token_idx=last_token_idx)
    )

    get_last_token = (last_token_idx // 32) * 32
    logits_tt, prefill_hidden_tt = gen.talker.ttnn_prefill_forward_with_hidden(
        tokens_embd, rot_mats_global=rot_mats, rot_mats_local=rot_mats_local,
        page_table=tt_page_table, chunk_page_table=tt_chunk_page_table,
        get_last_token=get_last_token, pre_projected=True,
    )

    logits = gen.talker.process_output_prefill(logits_tt.cpu(), last_token_idx=last_token_idx % 32)
    logits = logits.view(1, 1, args.vocab_size)

    prefill_hidden_torch = ttnn.to_torch(prefill_hidden_tt)
    last_in_block = last_token_idx % 32
    talker_hidden = prefill_hidden_torch[:, :, last_in_block:last_in_block + 1, :args.dim]
    talker_hidden = talker_hidden.permute(0, 2, 1, 3).reshape(B, 1, args.dim)

    cb0_token = torch.argmax(logits[:, -1, :], dim=-1)
    print(f"\nPrefill CB0: {cb0_token.item()}")
    print(f"Talker hidden norm: {talker_hidden.float().norm():.4f}")

    # --- Compare Code Predictor for first N frames ---
    prefill_len = last_token_idx + 1
    N_FRAMES = 8

    for step in range(N_FRAMES):
        if cb0_token.item() == args.codec_eos_token_id:
            print(f"\nEOS at step {step}")
            break

        # TT Code Predictor
        tt_frame = gen.code_predictor.predict_codebooks(talker_hidden, cb0_token, codec_embed_weight)

        # CPU Code Predictor
        cpu_frame = cpu_predict_codebooks(
            cpu_model, proj_w, proj_b, lm_heads, codec_embs,
            talker_hidden.float(), cb0_token, talker_codec_emb,
        )

        # Compare
        tt_cbs = tt_frame[0].tolist()
        cpu_cbs = cpu_frame[0].tolist()
        match_count = sum(1 for a, b in zip(tt_cbs, cpu_cbs) if a == b)

        print(f"\n--- Frame {step} (CB0={cb0_token.item()}) ---")
        print(f"  TT  CB0-15: {tt_cbs}")
        print(f"  CPU CB0-15: {cpu_cbs}")
        print(f"  Match: {match_count}/16")

        if match_count < 16:
            for i in range(16):
                if tt_cbs[i] != cpu_cbs[i]:
                    print(f"  DIFF at CB{i}: TT={tt_cbs[i]}, CPU={cpu_cbs[i]}")

        # Build next decode input (using TT frame results)
        cb0_emb = F.embedding(cb0_token.unsqueeze(-1), codec_embed_weight)
        cb_emb_sum = cb0_emb
        for cb_idx in range(args.num_code_groups - 1):
            cb_tok = tt_frame[:, cb_idx + 1]
            cb_emb = F.embedding(
                cb_tok.unsqueeze(-1), gen.code_predictor.codec_embeddings[cb_idx]
            )
            cb_emb_sum = cb_emb_sum + cb_emb

        decode_input = cb_emb_sum + trailing_text_hidden

        # Talker decode
        current_pos = torch.tensor([prefill_len + step], dtype=torch.int64)
        padded_pos = F.pad(current_pos, (0, args.max_batch_size - 1), value=0)
        dummy_tokens = torch.zeros(1, args.max_batch_size, dtype=torch.long)
        _, tt_pos, tt_rot_idxs, tt_pt = gen.talker.prepare_inputs_decode(dummy_tokens, padded_pos)

        decode_padded = torch.zeros(1, 1, 32, args.dim)
        decode_padded[0, 0, 0, :] = decode_input[0, 0, :]
        tt_decode = ttnn.from_torch(
            decode_padded, device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        decode_mem = args.get_residual_mem_config(Mode.DECODE, gen.talker.prefetcher)
        tt_decode = ttnn.to_memory_config(tt_decode, decode_mem)

        tt_logits, tt_hidden = gen.talker.ttnn_decode_forward_preembedded(
            tt_decode, tt_pos, rot_mat_idxs=tt_rot_idxs, page_table=tt_pt
        )

        hidden_torch = ttnn.to_torch(tt_hidden)
        talker_hidden = hidden_torch[:, :, :B, :args.dim].permute(0, 2, 1, 3).reshape(B, 1, args.dim)

        decode_logits = gen.talker.process_output_decode(tt_logits.cpu(), B=1)
        decode_logits = decode_logits[:, :, :args.vocab_size]
        cb0_token = torch.argmax(decode_logits[:, -1, :], dim=-1)

except Exception as e:
    traceback.print_exc()
finally:
    ttnn.close_mesh_device(mesh)
