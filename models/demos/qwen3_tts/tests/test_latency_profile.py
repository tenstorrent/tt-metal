# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Latency profiling for Qwen3-TTS pipeline stages."""

import os
import time
import traceback

import torch
import torch.nn.functional as F
import numpy as np

os.environ["HF_MODEL"] = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

import ttnn
from models.demos.qwen3_tts.tt.generator import TTSGenerator
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
    gen = TTSGenerator.build("Qwen/Qwen3-TTS-12Hz-1.7B-Base", mesh, max_seq_len=2560)
    args = gen.talker_args
    codec_embed_weight = gen.talker.codec_embed_weight

    # --- Stage 1: Input embedding ---
    t_embed_start = time.perf_counter()
    input_embeds, trailing_text_hidden, tts_pad_embed = gen._build_input_embeds("こんにちは", "japanese")
    t_embed = time.perf_counter() - t_embed_start

    # --- Stage 2: Prefill ---
    B = 1
    norms = input_embeds.squeeze(0).norm(dim=-1)
    nonzero_mask = norms > 0
    last_token_idx = nonzero_mask.nonzero()[-1].item()

    t_prefill_prep_start = time.perf_counter()
    tokens_embd, rot_mats, rot_mats_local, tt_page_table, tt_chunk_page_table = (
        gen.talker.prepare_inputs_prefill(input_embeds, start_pos=0, last_token_idx=last_token_idx)
    )
    t_prefill_prep = time.perf_counter() - t_prefill_prep_start

    t_prefill_fwd_start = time.perf_counter()
    get_last_token = (last_token_idx // 32) * 32
    logits_tt, prefill_hidden_tt = gen.talker.ttnn_prefill_forward_with_hidden(
        tokens_embd, rot_mats_global=rot_mats, rot_mats_local=rot_mats_local,
        page_table=tt_page_table, chunk_page_table=tt_chunk_page_table,
        get_last_token=get_last_token, pre_projected=True,
    )
    ttnn.synchronize_device(mesh)
    t_prefill_fwd = time.perf_counter() - t_prefill_fwd_start

    t_prefill_post_start = time.perf_counter()
    logits = gen.talker.process_output_prefill(logits_tt.cpu(), last_token_idx=last_token_idx % 32)
    logits = logits.view(1, 1, args.vocab_size)
    prefill_hidden_torch = ttnn.to_torch(prefill_hidden_tt)
    last_in_block = last_token_idx % 32
    talker_hidden_torch = prefill_hidden_torch[:, :, last_in_block:last_in_block + 1, :args.dim]
    talker_hidden_torch = talker_hidden_torch.permute(0, 2, 1, 3).reshape(B, 1, args.dim)
    cb0_token = torch.argmax(logits[:, -1, :], dim=-1)
    t_prefill_post = time.perf_counter() - t_prefill_post_start

    # --- Stage 3: Decode loop (per-step timing) ---
    prefill_len = last_token_idx + 1
    all_frames = []
    decode_times = {
        "code_predictor": [],
        "embed_build": [],
        "decode_prep": [],
        "h2d": [],
        "talker_decode": [],
        "d2h_postproc": [],
        "total_step": [],
    }

    for step in range(500):
        if cb0_token.item() == args.codec_eos_token_id:
            print(f"EOS at step {step}")
            break

        t_step_start = time.perf_counter()

        # Code Predictor (CPU)
        t0 = time.perf_counter()
        frame_all_cb = gen.code_predictor.predict_codebooks(
            talker_hidden_torch, cb0_token, codec_embed_weight
        )
        t_cp = time.perf_counter() - t0
        decode_times["code_predictor"].append(t_cp)

        all_frames.append(frame_all_cb.unsqueeze(1))

        # Build next input embedding (CPU)
        t0 = time.perf_counter()
        cb0_emb = F.embedding(cb0_token.unsqueeze(-1), codec_embed_weight)
        cb_emb_sum = cb0_emb
        for cb_idx in range(args.num_code_groups - 1):
            cb_tok = frame_all_cb[:, cb_idx + 1]
            cb_emb = F.embedding(
                cb_tok.unsqueeze(-1),
                gen.code_predictor.codec_embeddings[cb_idx],
            )
            cb_emb_sum = cb_emb_sum + cb_emb
        decode_input = cb_emb_sum + trailing_text_hidden
        t_embed_build = time.perf_counter() - t0
        decode_times["embed_build"].append(t_embed_build)

        # Prepare decode inputs
        t0 = time.perf_counter()
        current_pos = torch.tensor([prefill_len + step], dtype=torch.int64)
        padded_pos = F.pad(current_pos, (0, args.max_batch_size - 1), value=0)
        dummy_tokens = torch.zeros(1, args.max_batch_size, dtype=torch.long)
        _, tt_pos, tt_rot_idxs, tt_pt = gen.talker.prepare_inputs_decode(dummy_tokens, padded_pos)
        t_decode_prep = time.perf_counter() - t0
        decode_times["decode_prep"].append(t_decode_prep)

        # H2D transfer
        t0 = time.perf_counter()
        decode_padded = torch.zeros(1, 1, 32, args.dim)
        decode_padded[0, 0, 0, :] = decode_input[0, 0, :]
        tt_decode = ttnn.from_torch(
            decode_padded, device=mesh, dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        decode_mem = args.get_residual_mem_config(Mode.DECODE, gen.talker.prefetcher)
        tt_decode = ttnn.to_memory_config(tt_decode, decode_mem)
        t_h2d = time.perf_counter() - t0
        decode_times["h2d"].append(t_h2d)

        # Talker decode forward (device)
        t0 = time.perf_counter()
        tt_logits, tt_hidden = gen.talker.ttnn_decode_forward_preembedded(
            tt_decode, tt_pos, rot_mat_idxs=tt_rot_idxs, page_table=tt_pt
        )
        ttnn.synchronize_device(mesh)
        t_talker = time.perf_counter() - t0
        decode_times["talker_decode"].append(t_talker)

        # D2H + post-process
        t0 = time.perf_counter()
        hidden_torch = ttnn.to_torch(tt_hidden)
        talker_hidden_torch = hidden_torch[:, :, :B, :args.dim].permute(0, 2, 1, 3).reshape(B, 1, args.dim)
        decode_logits = gen.talker.process_output_decode(tt_logits.cpu(), B=1)
        decode_logits = decode_logits[:, :, :args.vocab_size]
        cb0_token = torch.argmax(decode_logits[:, -1, :], dim=-1)
        t_d2h = time.perf_counter() - t0
        decode_times["d2h_postproc"].append(t_d2h)

        t_step = time.perf_counter() - t_step_start
        decode_times["total_step"].append(t_step)

    num_steps = len(decode_times["total_step"])

    # --- Stage 4: Vocoder ---
    if all_frames:
        all_codebooks = torch.cat(all_frames, dim=1)
    else:
        all_codebooks = torch.zeros(B, 0, 16, dtype=torch.long)

    t_vocoder_start = time.perf_counter()
    waveform = gen._decode_waveform(all_codebooks)
    t_vocoder = time.perf_counter() - t_vocoder_start

    # === Report ===
    duration = len(waveform) / 24000
    print(f"\n{'='*60}")
    print(f"Qwen3-TTS Latency Profile — 'こんにちは'")
    print(f"{'='*60}")
    print(f"Audio: {duration:.3f}s ({num_steps} frames)")
    print()
    print(f"--- One-time costs ---")
    print(f"  Input embedding:     {t_embed*1000:8.1f} ms")
    print(f"  Prefill prep:        {t_prefill_prep*1000:8.1f} ms")
    print(f"  Prefill forward:     {t_prefill_fwd*1000:8.1f} ms")
    print(f"  Prefill postproc:    {t_prefill_post*1000:8.1f} ms")
    print(f"  Vocoder:             {t_vocoder*1000:8.1f} ms")
    print()

    print(f"--- Per-step decode breakdown (mean over {num_steps} steps) ---")
    for key in ["code_predictor", "embed_build", "decode_prep", "h2d", "talker_decode", "d2h_postproc", "total_step"]:
        vals = decode_times[key]
        mean_ms = np.mean(vals) * 1000
        std_ms = np.std(vals) * 1000
        min_ms = np.min(vals) * 1000
        max_ms = np.max(vals) * 1000
        total_ms = np.sum(vals) * 1000
        pct = total_ms / (np.sum(decode_times["total_step"]) * 1000) * 100
        print(f"  {key:20s}: {mean_ms:7.1f} ±{std_ms:5.1f} ms  (min={min_ms:6.1f}, max={max_ms:7.1f})  total={total_ms:7.0f} ms  {pct:5.1f}%")

    total_decode_ms = np.sum(decode_times["total_step"]) * 1000
    total_e2e = t_embed + t_prefill_prep + t_prefill_fwd + t_prefill_post + total_decode_ms/1000 + t_vocoder
    print()
    print(f"--- Summary ---")
    print(f"  Total E2E:           {total_e2e*1000:8.1f} ms")
    print(f"  Total decode:        {total_decode_ms:8.1f} ms  ({total_decode_ms/total_e2e/10:5.1f}%)")
    print(f"  RTF:                 {total_e2e/duration:8.3f}")
    print(f"  Tokens/s (CB0):      {num_steps/total_decode_ms*1000:8.1f}")
    print()

    # Batch size analysis
    print(f"--- Batch size analysis ---")
    mean_talker_ms = np.mean(decode_times["talker_decode"]) * 1000
    mean_cp_ms = np.mean(decode_times["code_predictor"]) * 1000
    mean_overhead_ms = np.mean(decode_times["total_step"]) * 1000 - mean_talker_ms - mean_cp_ms
    print(f"  Mean Talker decode:    {mean_talker_ms:6.1f} ms/step")
    print(f"  Mean Code Predictor:   {mean_cp_ms:6.1f} ms/step (CPU)")
    print(f"  Mean overhead (rest):  {mean_overhead_ms:6.1f} ms/step")
    print(f"  Talker % of step:      {mean_talker_ms/(mean_talker_ms+mean_cp_ms+mean_overhead_ms)*100:5.1f}%")
    print(f"  CP % of step:          {mean_cp_ms/(mean_talker_ms+mean_cp_ms+mean_overhead_ms)*100:5.1f}%")
    print()
    print("  Batch throughput potential:")
    print("  If Talker decode scales sublinearly with batch (typical for LLMs),")
    print("  and CP runs on CPU independently per sample:")
    for bs in [1, 2, 4, 8]:
        # Talker decode: assume ~constant for small batch on single chip
        # CP: linear with batch (CPU, sequential per sample)
        # Overhead: ~constant
        est_talker = mean_talker_ms * 1.0  # near-constant for decode
        est_cp = mean_cp_ms * bs  # CPU sequential
        est_overhead = mean_overhead_ms * 1.0  # roughly constant
        est_step = est_talker + est_cp + est_overhead
        throughput = bs / est_step * 1000
        print(f"    B={bs}: est {est_step:6.1f} ms/step, throughput={throughput:5.1f} tok/s ({throughput/bs:.1f} tok/s/sample)")

except Exception as e:
    traceback.print_exc()
finally:
    ttnn.close_mesh_device(mesh)
