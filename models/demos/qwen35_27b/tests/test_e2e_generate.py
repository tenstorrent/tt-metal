# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end text generation test for Qwen3.5-27B.

Validates that the full model (all 64 layers: 48 GDN + 16 full attention)
produces coherent output on TP=4 mesh.

Two test modes:
  - test_e2e_generate: No tracing, manual decode loop (correctness check)
  - test_e2e_generate_traced: Device sampling via model.ttnn_decode_forward(sampling_on_device=True)

Run:
    HF_MODEL=~/models/Qwen3.5-27B-FP8 \
        pytest models/demos/qwen35_27b/tt/tests/test_e2e_generate.py -v -s

Input length variants use pytest ids matching the Llama galaxy demo (``short``, ``long-4k-b1``, …).
Select one: ``pytest ... -k long-4k-b1``.
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


def _prompt_tokens_for_input_seq_len(tokenizer, target_len: int) -> list[int]:
    """Build a prompt of exactly ``target_len`` tokens; ends with the geography question for asserts."""
    suffix = "The capital of France is"
    suffix_toks = tokenizer.encode(suffix)
    if target_len < len(suffix_toks):
        return suffix_toks[-target_len:]
    filler_len = target_len - len(suffix_toks)
    filler_chunk = tokenizer.encode("The quick brown fox jumps over the lazy dog. ")
    if not filler_chunk:
        filler_chunk = suffix_toks[:1]
    prefix_toks: list[int] = []
    i = 0
    while len(prefix_toks) < filler_len:
        prefix_toks.append(filler_chunk[i % len(filler_chunk)])
        i += 1
    return prefix_toks + suffix_toks


def _max_seq_len_for_input(input_seq_len: int, max_gen_tokens: int) -> int:
    """Capacity for KV / positions: prefill length plus decode horizon."""
    need = input_seq_len + max_gen_tokens + 128
    for cap in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072):
        if cap >= need:
            return cap
    return 131072


# Input prefill lengths (pytest ids aligned with models/demos/llama3_70b_galaxy/demo/text_demo.py long-* cases).
E2E_INPUT_SEQ_LEN_PARAMS = (
    pytest.param(32, id="short"),
    pytest.param(4096, id="long-4k-b1"),
    pytest.param(8192, id="long-8k-b1"),
    pytest.param(16384, id="long-16k-b1"),
    pytest.param(32768, id="long-32k-b1"),
    pytest.param(65536, id="long-64k-b1"),
    pytest.param(131072, id="long-128k-b1"),
)


@torch.no_grad()
@pytest.mark.parametrize("input_seq_len", E2E_INPUT_SEQ_LEN_PARAMS)
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
def test_e2e_generate(mesh_device, reset_seeds, ensure_gc, input_seq_len):
    """Generate text with the full Qwen3.5-27B model and verify output quality.

    No tracing, no device sampling — straightforward correctness check.
    """
    model_path = _get_model_path()
    batch_size = 32
    max_gen_tokens = 20
    max_seq_len = _max_seq_len_for_input(input_seq_len, max_gen_tokens)

    if mesh_device.get_num_devices() < 4:
        pytest.skip("Full model requires TP>=4")

    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Create model
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

    prompt_tokens = _prompt_tokens_for_input_seq_len(tokenizer, input_seq_len)
    prompt = tokenizer.decode(prompt_tokens)
    logger.info(
        f"input_seq_len={input_seq_len}, actual {len(prompt_tokens)} tokens; " f"tail decode: {prompt_tokens[-8:]}"
    )

    # === PREFILL (decode one token at a time to fill KV cache / GDN state) ===
    logger.info("Prefilling...")
    t_prefill = time.time()

    for pos_idx in range(len(prompt_tokens) - 1):
        tok_batch = torch.full((batch_size,), prompt_tokens[pos_idx], dtype=torch.long)
        current_pos = torch.full((batch_size,), pos_idx, dtype=torch.long)

        tt_tokens, tt_current_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        tt_logits, _ = model.ttnn_decode_forward(
            tt_tokens,
            tt_current_pos,
            rot_mat_idxs=tt_rot_idxs,
        )

    logger.info(f"Prefill done in {time.time()-t_prefill:.1f}s ({len(prompt_tokens)-1} tokens)")

    # === DECODE ===
    logger.info(f"Decoding {max_gen_tokens} tokens...")
    generated_tokens = []
    current_token = prompt_tokens[-1]
    decode_times = []

    for step in range(max_gen_tokens):
        t_step = time.time()

        tok_batch = torch.full((batch_size,), current_token, dtype=torch.long)
        current_pos = torch.full((batch_size,), len(prompt_tokens) - 1 + step, dtype=torch.long)

        tt_tokens, tt_current_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        tt_logits, _ = model.ttnn_decode_forward(
            tt_tokens,
            tt_current_pos,
            rot_mat_idxs=tt_rot_idxs,
        )

        # Get logits to host
        logits_torch = ttnn.to_torch(tt_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
        # Slice to vocab size and take first user's logits
        logits_torch = logits_torch[0, 0, 0, : args.vocab_size]
        next_token = logits_torch.argmax().item()

        generated_tokens.append(next_token)
        current_token = next_token

        dt = time.time() - t_step
        decode_times.append(dt)

        token_text = tokenizer.decode([next_token])
        if step < 5:
            probs = torch.softmax(logits_torch.float(), dim=-1)
            topk = torch.topk(probs, k=5)
            top5 = ", ".join(f"'{tokenizer.decode([topk.indices[j].item()])}' ({topk.values[j]:.3f})" for j in range(5))
            logger.info(f"  Step {step+1}: top5=[{top5}] ({dt*1000:.0f}ms)")
        elif (step + 1) % 5 == 0:
            logger.info(f"  Step {step+1}: token={next_token} '{token_text}' ({dt*1000:.0f}ms)")

    # Decode results
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = prompt + generated_text
    logger.info(f"\nFull output: '{full_text}'")

    # Performance stats (skip first step which includes compile)
    if len(decode_times) > 1:
        avg_time = sum(decode_times[1:]) / len(decode_times[1:])
        tps = 1.0 / avg_time if avg_time > 0 else 0
        logger.info(f"Decode performance (excluding compile step):")
        logger.info(f"  Avg step time: {avg_time*1000:.1f}ms")
        logger.info(f"  Throughput: {tps:.1f} tok/s/user")
        logger.info(f"  First step (compile): {decode_times[0]*1000:.0f}ms")

    # Quality check
    output_lower = full_text.lower()
    assert "paris" in output_lower, f"Expected 'paris' in output, got: '{full_text}'"

    logger.info("PASSED: End-to-end generation produces correct output")


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
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "num_command_queues": 2, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("input_seq_len", E2E_INPUT_SEQ_LEN_PARAMS)
def test_e2e_generate_device_sampling(mesh_device, reset_seeds, ensure_gc, input_seq_len):
    """Generate text with on-device sampling (no host logits transfer).

    Uses model.ttnn_decode_forward(sampling_on_device=True) which:
    - Runs all_gather + untilize + argmax on device (force_argmax path)
    - Returns small token ID tensor instead of full logits
    - Eliminates 15.7MB PCIe logits transfer per step

    Dual CQ enabled (CQ0 for compute, CQ1 for IO).
    """
    model_path = _get_model_path()
    batch_size = 32
    max_gen_tokens = 30
    max_seq_len = _max_seq_len_for_input(input_seq_len, max_gen_tokens)

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

    # Verify on-device sampling is supported
    assert model._supports_on_device_sampling, (
        f"On-device sampling not supported: vocab_size={args.vocab_size}, "
        f"num_devices={args.num_devices}, per_device={args.vocab_size // args.num_devices}"
    )
    assert model.sampling is not None, "SamplingGenerator not initialized"
    logger.info(f"On-device sampling enabled: force_argmax={model.sampling.tt_sampling.force_argmax_sampling}")

    prompt_tokens = _prompt_tokens_for_input_seq_len(tokenizer, input_seq_len)
    prompt = tokenizer.decode(prompt_tokens)
    logger.info(f"input_seq_len={input_seq_len}, prefill tokens={len(prompt_tokens)}; tail: {prompt_tokens[-4:]}")

    # === PREFILL ===
    logger.info("Prefilling...")
    t_prefill = time.time()

    for pos_idx in range(len(prompt_tokens) - 1):
        tok_batch = torch.full((batch_size,), prompt_tokens[pos_idx], dtype=torch.long)
        current_pos = torch.full((batch_size,), pos_idx, dtype=torch.long)

        tt_tokens, tt_current_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        model.ttnn_decode_forward(
            tt_tokens,
            tt_current_pos,
            rot_mat_idxs=tt_rot_idxs,
        )

    logger.info(f"Prefill done in {time.time()-t_prefill:.1f}s ({len(prompt_tokens)-1} tokens)")

    # === DECODE with device sampling ===
    logger.info(f"Decoding {max_gen_tokens} tokens with on-device sampling...")
    generated_tokens = []
    current_token = prompt_tokens[-1]
    decode_times = []

    for step in range(max_gen_tokens):
        t_step = time.time()

        tok_batch = torch.full((batch_size,), current_token, dtype=torch.long)
        current_pos = torch.full((batch_size,), len(prompt_tokens) - 1 + step, dtype=torch.long)

        tt_tokens, tt_current_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        result = model.ttnn_decode_forward(
            tt_tokens,
            tt_current_pos,
            rot_mat_idxs=tt_rot_idxs,
            sampling_on_device=True,
        )

        # result is (tt_toks, tt_log_probs) when sampling_on_device
        if isinstance(result, tuple):
            tt_toks = result[0]
        else:
            tt_toks = result

        # Read small token tensor from device
        toks_cpu = ttnn.to_torch(tt_toks, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        # After all_gather, all devices have same result; take first device
        next_token = toks_cpu[0].flatten()[0].int().item()

        generated_tokens.append(next_token)
        current_token = next_token

        dt = time.time() - t_step
        decode_times.append(dt)

        token_text = tokenizer.decode([next_token])
        if step < 3:
            logger.info(f"  Step {step+1}: token={next_token} '{token_text}' ({dt*1000:.0f}ms)")
        elif (step + 1) % 5 == 0:
            if len(decode_times) > 1:
                avg = sum(decode_times[1:]) / len(decode_times[1:])
                tps = 1.0 / avg
            else:
                tps = 1.0 / dt
            logger.info(f"  Step {step+1}: '{token_text}' ({dt*1000:.0f}ms, {tps:.1f} tok/s/user)")

    # Decode results
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = prompt + generated_text
    logger.info(f"\nFull output: '{full_text}'")

    # Performance stats
    if len(decode_times) > 1:
        steady_times = decode_times[1:]
        avg_time = sum(steady_times) / len(steady_times)
        tps = 1.0 / avg_time if avg_time > 0 else 0
        logger.info(f"Decode performance (excluding compile step):")
        logger.info(f"  Avg step time: {avg_time*1000:.1f}ms")
        logger.info(f"  Throughput: {tps:.1f} tok/s/user ({tps*batch_size:.0f} tok/s aggregate)")
        logger.info(f"  First step (compile): {decode_times[0]*1000:.0f}ms")

    # Quality check
    output_lower = full_text.lower()
    assert "paris" in output_lower, f"Expected 'paris' in output, got: '{full_text}'"

    logger.info("PASSED: Device sampling produces correct output")


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
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "num_command_queues": 2, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("input_seq_len", E2E_INPUT_SEQ_LEN_PARAMS)
def test_e2e_generate_traced(mesh_device, reset_seeds, ensure_gc, input_seq_len):
    """Generate text with traced decode + on-device sampling.

    Captures the full decode graph (including GDN fused kernel) as a trace,
    then replays with ttnn.execute_trace() for zero Python dispatch overhead.
    """
    from models.tt_transformers.tt.common import copy_host_to_device

    model_path = _get_model_path()
    batch_size = 32
    max_gen_tokens = 30
    max_seq_len = _max_seq_len_for_input(input_seq_len, max_gen_tokens)

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

    assert model._supports_on_device_sampling, "On-device sampling not supported"
    assert model.sampling is not None, "SamplingGenerator not initialized"

    prompt_tokens = _prompt_tokens_for_input_seq_len(tokenizer, input_seq_len)
    prompt = tokenizer.decode(prompt_tokens)
    logger.info(f"input_seq_len={input_seq_len}, prefill tokens={len(prompt_tokens)}; tail: {prompt_tokens[-4:]}")

    # === PREFILL ===
    logger.info("Prefilling...")
    for pos_idx in range(len(prompt_tokens) - 1):
        tok_batch = torch.full((batch_size,), prompt_tokens[pos_idx], dtype=torch.long)
        current_pos = torch.full((batch_size,), pos_idx, dtype=torch.long)
        tt_tokens, tt_current_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        model.ttnn_decode_forward(tt_tokens, tt_current_pos, rot_mat_idxs=tt_rot_idxs)

    # === COMPILE RUN (with device sampling) ===
    logger.info("Compile run...")
    compile_pos = len(prompt_tokens) - 1
    tok_batch = torch.full((batch_size,), prompt_tokens[-1], dtype=torch.long)
    current_pos = torch.full((batch_size,), compile_pos, dtype=torch.long)
    tt_tokens, tt_current_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(tok_batch, current_pos)
    result = model.ttnn_decode_forward(
        tt_tokens,
        tt_current_pos,
        rot_mat_idxs=tt_rot_idxs,
        sampling_on_device=True,
    )
    tt_toks = result[0] if isinstance(result, tuple) else result
    toks_cpu = ttnn.to_torch(tt_toks, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    first_token = toks_cpu[0].flatten()[0].int().item()
    logger.info(f"Compile done, first token: '{tokenizer.decode([first_token])}'")

    # === RESET GDN STATES for clean trace ===
    logger.info("Resetting GDN states...")
    for layer in model.layers:
        if hasattr(layer.attention, "reset_state_inplace"):
            layer.attention.reset_state_inplace()

    # Re-prefill after reset
    logger.info("Re-prefilling for trace...")
    for pos_idx in range(len(prompt_tokens) - 1):
        tok_batch = torch.full((batch_size,), prompt_tokens[pos_idx], dtype=torch.long)
        current_pos = torch.full((batch_size,), pos_idx, dtype=torch.long)
        tt_tokens, tt_current_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        model.ttnn_decode_forward(tt_tokens, tt_current_pos, rot_mat_idxs=tt_rot_idxs)

    # === CAPTURE TRACE ===
    logger.info("Capturing trace...")
    # Prepare host inputs for the trace capture step
    trace_pos = len(prompt_tokens) - 1
    tok_batch = torch.full((batch_size,), prompt_tokens[-1], dtype=torch.long)
    current_pos_torch = torch.full((batch_size,), trace_pos, dtype=torch.long)

    host_inputs = model.prepare_decode_inputs_host(tok_batch, current_pos_torch)
    device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)
    # device_inputs = (tt_tokens, tt_current_pos, tt_rot_idxs, tt_page_table)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = model.ttnn_decode_forward(
        *device_inputs,
        sampling_on_device=True,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    logger.info("Trace captured!")

    # Read first traced token to verify
    tt_toks_trace = trace_output[0] if isinstance(trace_output, tuple) else trace_output
    ttnn.synchronize_device(mesh_device)
    toks_cpu = ttnn.to_torch(tt_toks_trace, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    current_token = toks_cpu[0].flatten()[0].int().item()
    logger.info(f"Trace capture token: '{tokenizer.decode([current_token])}'")

    # === TRACED DECODE LOOP ===
    logger.info(f"Decoding {max_gen_tokens} tokens with traced execution...")
    generated_tokens = [current_token]
    decode_times = []

    for step in range(max_gen_tokens - 1):
        t_step = time.time()

        pos = trace_pos + step + 1
        tok_batch = torch.full((batch_size,), current_token, dtype=torch.long)
        current_pos_torch = torch.full((batch_size,), pos, dtype=torch.long)

        # Update device inputs in-place
        host_inputs = model.prepare_decode_inputs_host(tok_batch, current_pos_torch)
        copy_host_to_device(host_tensors=host_inputs, device_tensors=device_inputs)

        # Execute trace
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)

        # Read token from device
        toks_cpu = ttnn.to_torch(tt_toks_trace, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        next_token = toks_cpu[0].flatten()[0].int().item()

        generated_tokens.append(next_token)
        current_token = next_token

        dt = time.time() - t_step
        decode_times.append(dt)

        token_text = tokenizer.decode([next_token])
        if step < 3:
            logger.info(f"  Step {step+1}: token={next_token} '{token_text}' ({dt*1000:.0f}ms)")
        elif (step + 1) % 5 == 0:
            avg = sum(decode_times[1:]) / max(len(decode_times[1:]), 1)
            tps = 1.0 / avg if avg > 0 else 0
            logger.info(f"  Step {step+1}: '{token_text}' ({dt*1000:.0f}ms, {tps:.1f} tok/s/user)")

    # Release trace
    ttnn.release_trace(mesh_device, trace_id)

    # Results
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = prompt + generated_text
    logger.info(f"\nFull output: '{full_text}'")

    if len(decode_times) > 1:
        steady_times = decode_times[1:]
        avg_time = sum(steady_times) / len(steady_times)
        tps = 1.0 / avg_time if avg_time > 0 else 0
        logger.info(f"Decode performance (excluding first trace step):")
        logger.info(f"  Avg step time: {avg_time*1000:.1f}ms")
        logger.info(f"  Throughput: {tps:.1f} tok/s/user ({tps*batch_size:.0f} tok/s aggregate)")
        logger.info(f"  First step: {decode_times[0]*1000:.0f}ms")

    output_lower = full_text.lower()
    assert "paris" in output_lower, f"Expected 'paris' in output, got: '{full_text}'"

    logger.info("PASSED: Traced decode with device sampling produces correct output")
