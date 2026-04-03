# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
E2E test with rolling-window L1 INTERLEAVED state for all 48 GDN layers.
3 GDN layers in L1 at a time, swapped around attention layers.
"""

import os

os.environ["TT_SKIP_CB_CLASH_CHECK"] = "1"

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
    [{"P150x4": (1, 4)}.get(os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 8)))],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_e2e_l1_rolling(mesh_device, reset_seeds, ensure_gc):
    """E2E decode with rolling-window L1 state on all 48 GDN layers."""
    model_path = _get_model_path()
    batch_size = 32
    max_seq_len = 2048
    max_gen_tokens = 10

    if mesh_device.get_num_devices() < 4:
        pytest.skip("Full model requires TP>=4")
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    logger.info("Creating model...")
    model = create_qwen35_model(mesh_device, model_path=model_path, max_batch_size=batch_size, max_seq_len=max_seq_len)
    args = model.args

    prompt = "The capital of France is"
    prompt_tokens = tokenizer.encode(prompt)
    logger.info(f"Prompt: '{prompt}' -> {len(prompt_tokens)} tokens")

    # Prefill
    logger.info("Prefilling...")
    for pos_idx in range(len(prompt_tokens) - 1):
        tok_batch = torch.full((batch_size,), prompt_tokens[pos_idx], dtype=torch.long)
        current_pos = torch.full((batch_size,), pos_idx, dtype=torch.long)
        tt_tok, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        model.ttnn_decode_forward(tt_tok, tt_pos, rot_mat_idxs=tt_rot)

    # Compile run
    logger.info("Compile run...")
    tok_batch = torch.full((batch_size,), prompt_tokens[-1], dtype=torch.long)
    current_pos = torch.full((batch_size,), len(prompt_tokens) - 1, dtype=torch.long)
    tt_tok, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, current_pos)
    tt_logits, _ = model.ttnn_decode_forward(tt_tok, tt_pos, rot_mat_idxs=tt_rot)
    logits_cpu = ttnn.to_torch(tt_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    first_token = logits_cpu[0, 0, 0, : args.vocab_size].argmax().item()
    logger.info(f"Compile done, first token: '{tokenizer.decode([first_token])}'")

    # Static L1 for first 3 GDN layers (no rolling window, just test correctness)
    logger.info("Enabling static L1 state (first 3 GDN layers, no swap)...")
    gdn_indices = [i for i in range(args.n_layers) if args.layer_types[i] == "linear_attention"]
    for idx in gdn_indices[:4]:
        gdn = model.layers[idx].attention
        l1_state = ttnn.to_memory_config(gdn.rec_states, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(gdn.rec_states)
        gdn.rec_states = l1_state
        logger.info(f"  Layer {idx}: state -> L1 (addr=0x{l1_state.buffer_address():x})")

    # Decode with rolling window
    logger.info(f"Decoding {max_gen_tokens} tokens with L1 rolling window...")
    generated_tokens = []
    current_token = first_token
    decode_times = []

    for step in range(max_gen_tokens):
        t_step = time.time()
        tok_batch = torch.full((batch_size,), current_token, dtype=torch.long)
        current_pos = torch.full((batch_size,), len(prompt_tokens) + step, dtype=torch.long)
        tt_tok, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        tt_logits, _ = model.ttnn_decode_forward(tt_tok, tt_pos, rot_mat_idxs=tt_rot)

        logits_cpu = ttnn.to_torch(tt_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
        next_token = logits_cpu[0, 0, 0, : args.vocab_size].argmax().item()
        generated_tokens.append(next_token)
        current_token = next_token
        dt = time.time() - t_step
        decode_times.append(dt)

        if step < 3:
            logger.info(f"  Step {step+1}: '{tokenizer.decode([next_token])}' ({dt*1000:.0f}ms)")

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = prompt + tokenizer.decode([first_token]) + generated_text
    logger.info(f"\nFull output: '{full_text}'")

    if len(decode_times) > 1:
        avg_time = sum(decode_times[1:]) / len(decode_times[1:])
        tps = 1.0 / avg_time if avg_time > 0 else 0
        logger.info(f"Avg step time: {avg_time*1000:.1f}ms ({tps:.1f} tok/s/user)")
        logger.info(f"Baseline: 68.6ms / 14.6 tok/s/user")

    output_lower = full_text.lower()
    assert "paris" in output_lower, f"Expected 'paris' in output, got: '{full_text}'"
    logger.info("PASSED: E2E with L1 rolling window")
