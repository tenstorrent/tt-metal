# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTFT (Time to First Token) benchmark for Qwen3.5-27B.

Measures prefill latency for various sequence lengths.
Prefill is done token-by-token (baseline) via the decode path,
matching the current e2e test behavior.

Run:
    HF_MODEL=~/models/Qwen3.5-27B-FP8 \
        pytest models/demos/qwen35_27b/tt/tests/test_ttft.py -v -s
"""

import os
import time

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.demos.qwen35_27b.tt.model import create_qwen35_model


def _get_model_path():
    return os.path.expanduser(os.environ.get("HF_MODEL", "~/models/Qwen3.5-27B-FP8"))


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "P150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 8))
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_ttft_baseline(mesh_device, reset_seeds, ensure_gc):
    """Measure TTFT using current token-by-token prefill (baseline).

    This times the full prefill phase: looping through prompt tokens
    one at a time via the decode forward path.
    """
    model_path = _get_model_path()
    batch_size = 32
    max_seq_len = 2048

    if mesh_device.get_num_devices() < 4:
        pytest.skip("Full model requires TP>=4")

    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    logger.info("Creating full Qwen3.5 model (64 layers)...")
    t0 = time.time()
    model = create_qwen35_model(
        mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        dtype=ttnn.bfloat8_b,
    )
    load_time = time.time() - t0
    logger.info(f"Model created in {load_time:.1f}s")

    args = model.args

    # Test with different prompt lengths
    prompts = {
        "short": "The capital of France is",
        "medium": "Explain the theory of relativity in simple terms. Albert Einstein proposed that space and time are intertwined, forming what we call spacetime. This revolutionary idea changed our understanding of physics",
    }

    for name, prompt in prompts.items():
        prompt_tokens = tokenizer.encode(prompt)
        seq_len = len(prompt_tokens)
        logger.info(f"\n{'='*60}")
        logger.info(f"Prompt '{name}': {seq_len} tokens")

        # Reset all states
        for layer in model.layers:
            if hasattr(layer.attention, "reset_state"):
                layer.attention.reset_state()
            if hasattr(layer.attention, "reset_state_inplace"):
                try:
                    layer.attention.reset_state_inplace()
                except Exception:
                    layer.attention.reset_state()

        # Warmup / compile run (first token)
        tok_batch = torch.full((batch_size,), prompt_tokens[0], dtype=torch.long)
        current_pos = torch.full((batch_size,), 0, dtype=torch.long)
        tt_tokens, tt_current_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        model.ttnn_decode_forward(tt_tokens, tt_current_pos, rot_mat_idxs=tt_rot_idxs)
        ttnn.synchronize_device(mesh_device)

        # Time the prefill (remaining tokens)
        t_prefill_start = time.time()

        for pos_idx in range(1, seq_len - 1):
            tok_batch = torch.full((batch_size,), prompt_tokens[pos_idx], dtype=torch.long)
            current_pos = torch.full((batch_size,), pos_idx, dtype=torch.long)
            tt_tokens, tt_current_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(tok_batch, current_pos)
            model.ttnn_decode_forward(tt_tokens, tt_current_pos, rot_mat_idxs=tt_rot_idxs)

        ttnn.synchronize_device(mesh_device)
        prefill_time = time.time() - t_prefill_start

        # First decode step (generates the first new token)
        t_first_token = time.time()
        tok_batch = torch.full((batch_size,), prompt_tokens[-1], dtype=torch.long)
        current_pos = torch.full((batch_size,), seq_len - 1, dtype=torch.long)
        tt_tokens, tt_current_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        tt_logits, _ = model.ttnn_decode_forward(tt_tokens, tt_current_pos, rot_mat_idxs=tt_rot_idxs)

        logits_torch = ttnn.to_torch(tt_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
        next_token = logits_torch[0, 0, 0, : args.vocab_size].argmax().item()
        first_token_time = time.time() - t_first_token

        total_ttft = prefill_time + first_token_time
        token_text = tokenizer.decode([next_token])

        logger.info(f"  Prefill time ({seq_len - 2} tokens, excl compile): {prefill_time*1000:.0f}ms")
        logger.info(f"  First token decode: {first_token_time*1000:.0f}ms")
        logger.info(f"  Total TTFT: {total_ttft*1000:.0f}ms")
        logger.info(f"  Per-token prefill: {prefill_time/(seq_len-2)*1000:.1f}ms/token")
        logger.info(f"  First token: '{token_text}'")

    logger.info("\nTTFT baseline measurement complete")


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "P150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 8))
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_ttft_batched_prefill(mesh_device, reset_seeds, ensure_gc):
    """Measure TTFT using batched prefill (optimized path).

    Uses ttnn_prefill_forward which calls forward_prefill on each layer:
    - Attention layers: flash attention over full sequence (B=1)
    - GDN layers: batched projections + sequential recurrence (B=1)
    Then replicates state to all 32 users and verifies decode works.
    """
    model_path = _get_model_path()
    batch_size = 32
    max_seq_len = 2048

    if mesh_device.get_num_devices() < 4:
        pytest.skip("Full model requires TP>=4")

    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    logger.info("Creating full Qwen3.5 model (64 layers)...")
    t0 = time.time()
    model = create_qwen35_model(
        mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        dtype=ttnn.bfloat8_b,
    )
    load_time = time.time() - t0
    logger.info(f"Model created in {load_time:.1f}s")

    args = model.args

    # Use a prompt that naturally tokenizes to a multiple of 32
    # "The capital of France is" = 5 tokens — too short for SDPA
    # Build a prompt that gives us exactly 128 tokens
    prompt = " ".join(
        ["The capital of France is a beautiful city."] * 4
        + ["It is known for its stunning architecture and rich cultural heritage."] * 4
        + ["Paris has been the center of French culture for centuries."] * 3
    )
    prompt_tokens = tokenizer.encode(prompt)
    # Truncate to nearest 128 (SDPA-friendly, tile-aligned)
    target_len = max(128, ((len(prompt_tokens)) // 128) * 128)
    if target_len > len(prompt_tokens):
        target_len = max(32, ((len(prompt_tokens)) // 32) * 32)
    prompt_tokens = prompt_tokens[:target_len]
    seq_len = len(prompt_tokens)
    logger.info(f"Prompt: {seq_len} tokens (truncated to tile boundary)")

    # ---- Reset all states ----
    for layer in model.layers:
        attn = layer.attention
        if hasattr(attn, "reset_state"):
            attn.reset_state()
        # Init B=1 prefill states for GDN layers
        if hasattr(attn, "_init_prefill_states"):
            attn._init_prefill_states()

    # ---- Prepare prefill inputs ----
    tokens_tensor = torch.tensor([prompt_tokens], dtype=torch.long)  # [1, seq_len]
    # get_last_token: framework slices [get_last_token : get_last_token+32]
    # Must satisfy: get_last_token + 32 <= seq_len, tile-aligned
    last_token_idx = ((seq_len - 1) // 32) * 32  # Start of tile containing last token

    # ---- Compile run (first prefill) ----
    logger.info("Compile run (framework prefill)...")
    t_compile = time.time()
    prefill_inputs = model.prepare_inputs_prefill(tokens_tensor)
    tt_embeds = prefill_inputs[0]
    tt_rot_global = prefill_inputs[1]

    tt_out = model.ttnn_prefill_forward(
        tt_embeds,
        rot_mats_global=tt_rot_global,
        get_last_token=last_token_idx,
    )
    ttnn.synchronize_device(mesh_device)
    compile_time = time.time() - t_compile
    logger.info(f"Compile run done in {compile_time:.1f}s")

    # Get first token from prefill
    logits_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    next_token = logits_torch[0, 0, 0, : args.vocab_size].argmax().item()
    token_text = tokenizer.decode([next_token])
    logger.info(f"First token (compile): '{token_text}'")

    # ---- Reset and re-run for timing ----
    for layer in model.layers:
        attn = layer.attention
        if hasattr(attn, "reset_state"):
            attn.reset_state()
        if hasattr(attn, "_init_prefill_states"):
            attn._init_prefill_states()

    logger.info("Timed prefill run...")
    prefill_inputs = model.prepare_inputs_prefill(tokens_tensor)
    tt_embeds = prefill_inputs[0]
    tt_rot_global = prefill_inputs[1]

    t_prefill = time.time()
    tt_out = model.ttnn_prefill_forward(
        tt_embeds,
        rot_mats_global=tt_rot_global,
        get_last_token=last_token_idx,
    )
    ttnn.synchronize_device(mesh_device)
    prefill_time = time.time() - t_prefill

    logits_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    next_token = logits_torch[0, 0, 0, : args.vocab_size].argmax().item()
    token_text = tokenizer.decode([next_token])

    logger.info(f"\n{'='*60}")
    logger.info(f"BATCHED PREFILL RESULTS ({seq_len} tokens):")
    logger.info(f"  Total prefill time: {prefill_time*1000:.0f}ms")
    logger.info(f"  Per-token: {prefill_time/seq_len*1000:.1f}ms/token")
    logger.info(f"  First token: '{token_text}'")
    logger.info(f"  vs baseline: ~{seq_len * 498:.0f}ms ({seq_len} * 498ms/token)")
    logger.info(f"{'='*60}")

    # ---- Compare with baseline token-by-token prefill ----
    logger.info("Running baseline comparison (token-by-token, same tokens)...")
    for layer in model.layers:
        attn = layer.attention
        if hasattr(attn, "reset_state"):
            attn.reset_state()

    # Use exact same prompt_tokens (no padding difference)
    for pos_idx in range(seq_len - 1):
        tok_batch = torch.full((batch_size,), prompt_tokens[pos_idx], dtype=torch.long)
        current_pos_t = torch.full((batch_size,), pos_idx, dtype=torch.long)
        tt_tok, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, current_pos_t)
        model.ttnn_decode_forward(tt_tok, tt_pos, rot_mat_idxs=tt_rot)

    tok_batch = torch.full((batch_size,), prompt_tokens[-1], dtype=torch.long)
    current_pos_t = torch.full((batch_size,), seq_len - 1, dtype=torch.long)
    tt_tok, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, current_pos_t)
    tt_logits_base, _ = model.ttnn_decode_forward(tt_tok, tt_pos, rot_mat_idxs=tt_rot)
    logits_base = ttnn.to_torch(tt_logits_base, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    base_token = logits_base[0, 0, 0, : args.vocab_size].argmax().item()
    logger.info(f"Baseline first token: '{tokenizer.decode([base_token])}'")
    logger.info(f"Batched first token:  '{token_text}'")
    if base_token == next_token:
        logger.info("MATCH! Prefill correctness verified")
    else:
        logger.info(f"MISMATCH: baseline={base_token} vs batched={next_token}")

    # ---- Replicate prefill state to batch for decode ----
    logger.info("Replicating prefill state to batch...")
    for layer in model.layers:
        attn = layer.attention
        if hasattr(attn, "replicate_kv_cache_to_batch"):
            attn.replicate_kv_cache_to_batch()
        if hasattr(attn, "replicate_prefill_state_to_batch"):
            attn.replicate_prefill_state_to_batch()

    # ---- Verify decode works after prefill ----
    logger.info("Verifying decode after prefill...")
    current_token = next_token
    generated_tokens = [current_token]

    for step in range(3):
        tok_batch = torch.full((batch_size,), current_token, dtype=torch.long)
        current_pos = torch.full((batch_size,), seq_len + step, dtype=torch.long)
        tt_tokens, tt_current_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        tt_logits, _ = model.ttnn_decode_forward(tt_tokens, tt_current_pos, rot_mat_idxs=tt_rot_idxs)

        logits_torch = ttnn.to_torch(tt_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
        next_token = logits_torch[0, 0, 0, : args.vocab_size].argmax().item()
        generated_tokens.append(next_token)
        current_token = next_token
        logger.info(f"  Decode step {step+1}: '{tokenizer.decode([next_token])}'")

    full_text = prompt + tokenizer.decode(generated_tokens, skip_special_tokens=True)
    logger.info(f"Full output: '{full_text}'")
    logger.info("TTFT batched prefill test complete")
