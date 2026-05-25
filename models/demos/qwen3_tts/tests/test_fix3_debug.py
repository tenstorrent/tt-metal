# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Debug: generate a few steps and print CB0 + CB1-15 per frame."""

import os
import traceback

import torch

os.environ["HF_MODEL"] = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

import ttnn
from models.demos.qwen3_tts.tt.generator import TTSGenerator
from loguru import logger

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

    # Monkey-patch to add per-step logging
    original_generate = gen._generate_and_predict

    def debug_generate(*args, **kwargs):
        # Run with max_new_tokens=25 and log each step
        return original_generate(*args, **kwargs)

    # Generate with detailed logging
    import math
    import numpy as np

    args = gen.talker_args
    input_embeds, trailing_text_hidden, tts_pad_embed = gen._build_input_embeds("こんにちは", "japanese")

    # Manually run prefill
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
    talker_hidden = prefill_hidden_torch[:, :, last_in_block:last_in_block+1, :args.dim]
    talker_hidden = talker_hidden.permute(0, 2, 1, 3).reshape(B, 1, args.dim)

    cb0_token = torch.argmax(logits[:, -1, :], dim=-1)  # greedy
    print(f"Step 0: CB0={cb0_token.item()}, hidden_norm={talker_hidden.float().norm():.4f}")

    # HF reference CB0 sequence: [1995, 215, 212, 1181, 462, 462, 251, 1109, 1974, 1300, 332, 1461, 1754, 308, 179, 10, 1475, 2150]
    hf_cb0s = [1995, 215, 212, 1181, 462, 462, 251, 1109, 1974, 1300, 332, 1461, 1754, 308, 179, 10, 1475, 2150]
    codec_embed_weight = gen.talker.codec_embed_weight

    from models.tt_transformers.tt.common import Mode

    prefill_len = last_token_idx + 1

    for step in range(25):
        if cb0_token.item() == args.codec_eos_token_id:
            print(f"EOS at step {step+1}!")
            break

        # Code Predictor
        frame = gen.code_predictor.predict_codebooks(talker_hidden, cb0_token, codec_embed_weight)
        print(f"  Frame {step}: CB0={frame[0,0].item()}, CB1-5={frame[0,1:6].tolist()}")

        # Compare with HF
        if step < len(hf_cb0s):
            hf_expected = hf_cb0s[step]
            match = "OK" if cb0_token.item() == hf_expected else f"MISMATCH (expected {hf_expected})"
            print(f"  HF comparison: {match}")

        # Build decode input
        cb0_emb = torch.nn.functional.embedding(cb0_token.unsqueeze(-1), codec_embed_weight)
        cb_emb_sum = cb0_emb
        for cb_idx in range(args.num_code_groups - 1):
            cb_tok = frame[:, cb_idx + 1]
            cb_emb = torch.nn.functional.embedding(
                cb_tok.unsqueeze(-1), gen.code_predictor.codec_embeddings[cb_idx]
            )
            cb_emb_sum = cb_emb_sum + cb_emb

        decode_input = cb_emb_sum + trailing_text_hidden
        print(f"  decode_input norm={decode_input.float().norm():.4f}")

        # Talker decode
        current_pos = torch.tensor([prefill_len + step], dtype=torch.int64)
        padded_pos = torch.nn.functional.pad(current_pos, (0, args.max_batch_size - 1), value=0)
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

        eos_logit = decode_logits[0, 0, args.codec_eos_token_id].item()
        max_logit = decode_logits[0, 0, :].max().item()
        print(f"Step {step+1}: CB0={cb0_token.item()}, EOS_logit={eos_logit:.2f}, max_logit={max_logit:.2f}, hidden_norm={talker_hidden.float().norm():.4f}")

except Exception as e:
    traceback.print_exc()
finally:
    ttnn.close_mesh_device(mesh)
