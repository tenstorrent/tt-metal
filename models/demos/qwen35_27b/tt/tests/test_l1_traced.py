# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Validation: L1 INTERLEAVED rec_states + traced decode.

Checks that:
1. L1 state survives trace capture
2. Traced execution produces correct output
3. Reports timing vs baseline 68.6ms/step

Run:
    tt-smi -r 0,1,2,3 && HF_MODEL=~/models/Qwen3.5-27B-FP8 \
    pytest models/demos/qwen35_27b/tt/tests/test_l1_traced.py -v -s
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
from models.tt_transformers.tt.common import copy_host_to_device


def _get_model_path():
    return os.path.expanduser(os.environ.get("HF_MODEL", "~/models/Qwen3.5-27B-FP8"))


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"P150x4": (1, 4)}.get(os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 8)))],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "num_command_queues": 2, "trace_region_size": 200_000_000}],
    indirect=True,
)
def test_l1_state_traced(mesh_device, reset_seeds, ensure_gc):
    """Traced decode with 4 GDN layers' rec_states in L1 INTERLEAVED."""
    model_path = _get_model_path()
    batch_size = 32
    max_seq_len = 2048
    max_gen_tokens = 15
    N_L1_LAYERS = 5

    if mesh_device.get_num_devices() < 4:
        pytest.skip("Need TP>=4")
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    logger.info("Creating model...")
    model = create_qwen35_model(mesh_device, model_path=model_path, max_batch_size=batch_size, max_seq_len=max_seq_len)
    args = model.args

    assert model._supports_on_device_sampling, "Need on-device sampling for tracing"

    # ---- Move first N GDN layers to L1 ----
    gdn_indices = [i for i in range(args.n_layers) if args.layer_types[i] == "linear_attention"]
    logger.info(f"Moving first {N_L1_LAYERS} of {len(gdn_indices)} GDN layers to L1...")

    for idx in gdn_indices[:N_L1_LAYERS]:
        gdn = model.layers[idx].attention
        if gdn.rec_states is None:
            gdn.reset_state()
        l1_state = ttnn.to_memory_config(gdn.rec_states, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(gdn.rec_states)
        gdn.rec_states = l1_state

    # Verify L1 placement
    for idx in gdn_indices[:N_L1_LAYERS]:
        gdn = model.layers[idx].attention
        assert gdn.rec_states.memory_config().buffer_type == ttnn.BufferType.L1, f"Layer {idx} not in L1!"
    logger.info("L1 state placement verified")

    # ---- Prefill ----
    prompt = "The capital of France is"
    prompt_tokens = tokenizer.encode(prompt)
    logger.info(f"Prompt: '{prompt}' -> {len(prompt_tokens)} tokens")

    for pos_idx in range(len(prompt_tokens) - 1):
        tok_batch = torch.full((batch_size,), prompt_tokens[pos_idx], dtype=torch.long)
        current_pos = torch.full((batch_size,), pos_idx, dtype=torch.long)
        tt_tok, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        model.ttnn_decode_forward(tt_tok, tt_pos, rot_mat_idxs=tt_rot)

    # ---- Compile run (with device sampling) ----
    logger.info("Compile run...")
    compile_pos = len(prompt_tokens) - 1
    tok_batch = torch.full((batch_size,), prompt_tokens[-1], dtype=torch.long)
    current_pos = torch.full((batch_size,), compile_pos, dtype=torch.long)
    tt_tok, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, current_pos)
    result = model.ttnn_decode_forward(tt_tok, tt_pos, rot_mat_idxs=tt_rot, sampling_on_device=True)
    tt_toks = result[0] if isinstance(result, tuple) else result
    toks_cpu = ttnn.to_torch(tt_toks, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    first_token = toks_cpu[0].flatten()[0].int().item()
    logger.info(f"Compile token: '{tokenizer.decode([first_token])}'")

    # ---- Reset GDN states for clean trace ----
    logger.info("Resetting GDN states...")
    for layer in model.layers:
        if hasattr(layer.attention, "reset_state_inplace"):
            layer.attention.reset_state_inplace()

    # Re-place first N in L1 (reset_state_inplace zeros them but keeps memory config)
    # Verify they're still in L1
    for idx in gdn_indices[:N_L1_LAYERS]:
        gdn = model.layers[idx].attention
        if gdn.rec_states.memory_config().buffer_type != ttnn.BufferType.L1:
            logger.warning(f"Layer {idx} fell out of L1 after reset, re-placing...")
            l1_state = ttnn.to_memory_config(gdn.rec_states, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(gdn.rec_states)
            gdn.rec_states = l1_state

    # Re-prefill
    logger.info("Re-prefilling for trace...")
    for pos_idx in range(len(prompt_tokens) - 1):
        tok_batch = torch.full((batch_size,), prompt_tokens[pos_idx], dtype=torch.long)
        current_pos = torch.full((batch_size,), pos_idx, dtype=torch.long)
        tt_tok, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        model.ttnn_decode_forward(tt_tok, tt_pos, rot_mat_idxs=tt_rot)

    # ---- Capture trace ----
    logger.info("Capturing trace...")
    trace_pos = len(prompt_tokens) - 1
    tok_batch = torch.full((batch_size,), prompt_tokens[-1], dtype=torch.long)
    current_pos_torch = torch.full((batch_size,), trace_pos, dtype=torch.long)

    host_inputs = model.prepare_decode_inputs_host(tok_batch, current_pos_torch)
    device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = model.ttnn_decode_forward(*device_inputs, sampling_on_device=True)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    logger.info("Trace captured!")

    tt_toks_trace = trace_output[0] if isinstance(trace_output, tuple) else trace_output
    ttnn.synchronize_device(mesh_device)
    toks_cpu = ttnn.to_torch(tt_toks_trace, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    current_token = toks_cpu[0].flatten()[0].int().item()
    logger.info(f"Trace capture token: '{tokenizer.decode([current_token])}'")

    # Verify L1 survived trace capture
    for idx in gdn_indices[:N_L1_LAYERS]:
        gdn = model.layers[idx].attention
        is_l1 = gdn.rec_states.memory_config().buffer_type == ttnn.BufferType.L1
        if not is_l1:
            logger.error(f"Layer {idx} state NOT in L1 after trace capture!")
        assert is_l1, f"Layer {idx} fell out of L1 during trace!"
    logger.info("L1 state survived trace capture")

    # ---- Traced decode loop ----
    logger.info(f"Decoding {max_gen_tokens} tokens with traced execution...")
    generated_tokens = [current_token]
    decode_times = []

    for step in range(max_gen_tokens - 1):
        t_step = time.time()

        pos = trace_pos + step + 1
        tok_batch = torch.full((batch_size,), current_token, dtype=torch.long)
        current_pos_torch = torch.full((batch_size,), pos, dtype=torch.long)

        host_inputs = model.prepare_decode_inputs_host(tok_batch, current_pos_torch)
        copy_host_to_device(host_tensors=host_inputs, device_tensors=device_inputs)

        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)

        toks_cpu = ttnn.to_torch(tt_toks_trace, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        next_token = toks_cpu[0].flatten()[0].int().item()

        generated_tokens.append(next_token)
        current_token = next_token

        dt = time.time() - t_step
        decode_times.append(dt)

        if step < 3:
            logger.info(f"  Step {step+1}: '{tokenizer.decode([next_token])}' ({dt*1000:.0f}ms)")

    ttnn.release_trace(mesh_device, trace_id)

    # Results
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = prompt + generated_text
    logger.info(f"\nFull output: '{full_text}'")

    if len(decode_times) > 1:
        steady_times = decode_times[1:]
        avg_time = sum(steady_times) / len(steady_times)
        tps = 1.0 / avg_time if avg_time > 0 else 0
        logger.info(f"Decode performance ({N_L1_LAYERS} GDN layers in L1):")
        logger.info(f"  Avg step time: {avg_time*1000:.1f}ms")
        logger.info(f"  Throughput: {tps:.1f} tok/s/user ({tps*batch_size:.0f} tok/s aggregate)")
        logger.info(f"  Baseline: 68.6ms / 14.6 tok/s/user")
        delta = 68.6 - avg_time * 1000
        logger.info(f"  Delta: {delta:+.1f}ms ({delta/68.6*100:+.1f}%)")

    output_lower = full_text.lower()
    assert "paris" in output_lower, f"Expected 'paris' in output, got: '{full_text}'"
    logger.info("PASSED: Correct traced decode with L1 state")
