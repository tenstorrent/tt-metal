# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision analysis: TT Code Predictor vs CPU reference (with POST-NORM fix).

Compares per-CB logit distributions and token outputs between the TT bfloat16
Code Predictor and CPU float32 reference to quantify precision loss and its
impact on token accuracy.
"""

import os
import time
import traceback

import torch
import torch.nn.functional as F
import numpy as np

os.environ["HF_MODEL"] = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

import ttnn
from models.demos.qwen3_tts.tt.generator import TTSGenerator
from models.demos.qwen3_tts.tt.code_predictor_cpu import CPUCodePredictor
from models.tt_transformers.tt.common import Mode


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
    # Build TT pipeline (with CPU Code Predictor as default — we'll also test TT CP)
    gen = TTSGenerator.build("Qwen/Qwen3-TTS-12Hz-1.7B-Base", mesh, max_seq_len=2560)
    args = gen.talker_args
    codec_embed_weight = gen.talker.codec_embed_weight

    # Build TT Code Predictor separately
    from models.demos.qwen3_tts.tt.model_config import CodePredictorModelArgs
    from models.demos.qwen3_tts.tt.code_predictor import CodePredictorTransformer

    cp_args = CodePredictorModelArgs(
        mesh_device=mesh,
        max_batch_size=1,
        max_seq_len=256,
        use_hf_rope=True,
    )
    cp_state_dict = cp_args.load_state_dict()
    tt_cp = CodePredictorTransformer(
        args=cp_args,
        dtype=ttnn.bfloat16,
        mesh_device=mesh,
        state_dict=cp_state_dict,
        weight_cache_path=cp_args.weight_cache_path(ttnn.bfloat16),
    )

    # CPU Code Predictor (float32 reference)
    cpu_cp = CPUCodePredictor.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    # Build input embeds
    input_embeds, trailing_text_hidden, tts_pad_embed = gen._build_input_embeds("こんにちは", "japanese")

    # Prefill (using the generator's talker)
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
    print(f"Prefill CB0: {cb0_token.item()}")
    print(f"Talker hidden shape: {talker_hidden.shape}, norm: {talker_hidden.float().norm():.4f}")

    # --- Compare TT vs CPU for N frames ---
    prefill_len = last_token_idx + 1
    N_FRAMES = 8

    total_cbs = 0
    total_match = 0
    per_cb_match = [0] * 16
    per_cb_total = [0] * 16

    for step in range(N_FRAMES):
        if cb0_token.item() == args.codec_eos_token_id:
            print(f"\nEOS at step {step}")
            break

        # TT Code Predictor (bfloat16)
        t0 = time.perf_counter()
        tt_frame = tt_cp.predict_codebooks(talker_hidden, cb0_token, codec_embed_weight)
        t_tt = time.perf_counter() - t0

        # CPU Code Predictor (float32)
        t0 = time.perf_counter()
        cpu_frame = cpu_cp.predict_codebooks(
            talker_hidden.float(), cb0_token,
            codec_embed_weight,
        )
        t_cpu = time.perf_counter() - t0

        # Compare
        tt_cbs = tt_frame[0].tolist()
        cpu_cbs = cpu_frame[0].tolist()
        match_count = sum(1 for a, b in zip(tt_cbs, cpu_cbs) if a == b)
        total_cbs += 16
        total_match += match_count

        for i in range(16):
            per_cb_total[i] += 1
            if tt_cbs[i] == cpu_cbs[i]:
                per_cb_match[i] += 1

        print(f"\n--- Frame {step} (CB0={cb0_token.item()}) ---")
        print(f"  TT  ({t_tt*1000:6.1f}ms): {tt_cbs}")
        print(f"  CPU ({t_cpu*1000:6.1f}ms): {cpu_cbs}")
        print(f"  Match: {match_count}/16")

        if match_count < 16:
            for i in range(16):
                if tt_cbs[i] != cpu_cbs[i]:
                    print(f"    CB{i}: TT={tt_cbs[i]:4d}, CPU={cpu_cbs[i]:4d}")

        # Advance using CPU predictions (known-good) to keep divergence from propagating
        cb0_emb = F.embedding(cb0_token.unsqueeze(-1), codec_embed_weight)
        cb_emb_sum = cb0_emb
        for cb_idx in range(args.num_code_groups - 1):
            cb_tok = cpu_frame[:, cb_idx + 1]
            cb_emb = F.embedding(cb_tok.unsqueeze(-1), cpu_cp.codec_embeddings[cb_idx])
            cb_emb_sum = cb_emb_sum + cb_emb

        decode_input = cb_emb_sum + trailing_text_hidden

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

    # Summary
    n_steps = min(N_FRAMES, step + 1) if cb0_token.item() == args.codec_eos_token_id else N_FRAMES
    print(f"\n{'='*60}")
    print(f"Precision Summary ({n_steps} frames)")
    print(f"{'='*60}")
    print(f"Overall token match: {total_match}/{total_cbs} ({total_match/total_cbs*100:.1f}%)")
    print(f"\nPer-codebook accuracy:")
    for i in range(16):
        if per_cb_total[i] > 0:
            pct = per_cb_match[i] / per_cb_total[i] * 100
            print(f"  CB{i:2d}: {per_cb_match[i]}/{per_cb_total[i]} ({pct:5.1f}%)")

except Exception as e:
    traceback.print_exc()
finally:
    ttnn.close_mesh_device(mesh)
