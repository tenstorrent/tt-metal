# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Profiling test for Qwen3.5-27B decode step breakdown.

Compares host-sampling vs device-sampling to quantify the D2H bottleneck.
"""

import os
import time

import pytest
import torch
from transformers import AutoTokenizer

import ttnn
from models.demos.qwen35_27b.tt.model import create_qwen35_model


def _get_model_path():
    return os.path.expanduser(os.environ.get("HF_MODEL", "~/models/Qwen3.5-27B-FP8"))


def _prefill_and_compile(model, mesh_device, tokenizer, prompt, batch_size, use_device_sampling=False):
    """Run prefill + compile step, return (current_token, prompt_tokens)."""
    args = model.args
    prompt_tokens = tokenizer.encode(prompt)

    # Prefill
    for pos_idx in range(len(prompt_tokens) - 1):
        tok_batch = torch.full((batch_size,), prompt_tokens[pos_idx], dtype=torch.long)
        current_pos = torch.full((batch_size,), pos_idx, dtype=torch.long)
        tt_tokens, tt_current_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        model.ttnn_decode_forward(tt_tokens, tt_current_pos, rot_mat_idxs=tt_rot_idxs)

    # Compile step
    current_token = prompt_tokens[-1]
    tok_batch = torch.full((batch_size,), current_token, dtype=torch.long)
    current_pos = torch.full((batch_size,), len(prompt_tokens) - 1, dtype=torch.long)
    tt_tokens, tt_current_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(tok_batch, current_pos)

    if use_device_sampling:
        result = model.ttnn_decode_forward(
            tt_tokens,
            tt_current_pos,
            rot_mat_idxs=tt_rot_idxs,
            sampling_on_device=True,
        )
        tt_toks = result[0] if isinstance(result, tuple) else result
        toks_cpu = ttnn.to_torch(tt_toks, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        next_token = toks_cpu[0].flatten()[0].int().item()
    else:
        tt_logits, _ = model.ttnn_decode_forward(tt_tokens, tt_current_pos, rot_mat_idxs=tt_rot_idxs)
        logits_torch = ttnn.to_torch(tt_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
        next_token = logits_torch[0, 0, 0, : args.vocab_size].argmax().item()

    return next_token, prompt_tokens


def _decode_step_host(model, mesh_device, current_token, pos, batch_size, args):
    """Single decode step with host sampling. Returns (next_token, prep_ms, fwd_ms, d2h_ms)."""
    tok_batch = torch.full((batch_size,), current_token, dtype=torch.long)
    current_pos = torch.full((batch_size,), pos, dtype=torch.long)

    t0 = time.perf_counter()
    tt_tokens, tt_current_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(tok_batch, current_pos)
    t1 = time.perf_counter()
    tt_logits, _ = model.ttnn_decode_forward(tt_tokens, tt_current_pos, rot_mat_idxs=tt_rot_idxs)
    ttnn.synchronize_device(mesh_device)
    t2 = time.perf_counter()
    logits_torch = ttnn.to_torch(tt_logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    next_token = logits_torch[0, 0, 0, : args.vocab_size].argmax().item()
    t3 = time.perf_counter()

    return next_token, (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000


def _decode_step_device(model, mesh_device, current_token, pos, batch_size):
    """Single decode step with device sampling. Returns (next_token, prep_ms, fwd_ms, d2h_ms)."""
    tok_batch = torch.full((batch_size,), current_token, dtype=torch.long)
    current_pos = torch.full((batch_size,), pos, dtype=torch.long)

    t0 = time.perf_counter()
    tt_tokens, tt_current_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(tok_batch, current_pos)
    t1 = time.perf_counter()
    result = model.ttnn_decode_forward(
        tt_tokens,
        tt_current_pos,
        rot_mat_idxs=tt_rot_idxs,
        sampling_on_device=True,
    )
    ttnn.synchronize_device(mesh_device)
    t2 = time.perf_counter()
    tt_toks = result[0] if isinstance(result, tuple) else result
    toks_cpu = ttnn.to_torch(tt_toks, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    next_token = toks_cpu[0].flatten()[0].int().item()
    t3 = time.perf_counter()

    return next_token, (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000


def _print_results(label, times, batch_size):
    """Print timing summary for a set of decode steps."""
    avg = [sum(t[i] for t in times) / len(times) for i in range(3)]
    avg_total = sum(avg)
    tps = 1000.0 / avg_total

    print("\n=== %s ===" % label)
    for i, (prep, fwd, d2h) in enumerate(times):
        total = prep + fwd + d2h
        print("  Step %d: total=%.1fms (prep=%.1f fwd=%.1f d2h=%.1f)" % (i + 1, total, prep, fwd, d2h))
    print("  ---")
    print("  Avg total:   %.1f ms/step" % avg_total)
    print("  Avg prep:    %.1f ms (%.0f%%)" % (avg[0], 100 * avg[0] / avg_total))
    print("  Avg forward: %.1f ms (%.0f%%)" % (avg[1], 100 * avg[1] / avg_total))
    print("  Avg D2H:     %.1f ms (%.0f%%)" % (avg[2], 100 * avg[2] / avg_total))
    print("  Throughput:  %.1f tok/s/user (batch=%d, %.0f tok/s aggregate)" % (tps, batch_size, tps * batch_size))
    return avg_total, avg


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "num_command_queues": 2, "trace_region_size": 200_000_000}],
    indirect=True,
)
def test_profile_decode(mesh_device, reset_seeds, ensure_gc):
    """Profile decode: host sampling vs device sampling side-by-side."""
    model_path = _get_model_path()
    batch_size = 32
    max_seq_len = 2048
    num_steps = 10

    if mesh_device.get_num_devices() < 4:
        pytest.skip("Full model requires TP>=4")

    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("\nCreating model...")
    model = create_qwen35_model(
        mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        dtype=ttnn.bfloat8_b,
    )
    args = model.args
    prompt = "The capital of France is"

    # ── Phase 1: Host sampling ──
    print("\n--- Phase 1: HOST SAMPLING (baseline) ---")
    current_token, prompt_tokens = _prefill_and_compile(model, mesh_device, tokenizer, prompt, batch_size)
    print("  Compile done, first token: %r" % tokenizer.decode([current_token]))

    # Reset GDN states for clean measurement
    for layer in model.layers:
        if hasattr(layer.attention, "reset_state"):
            layer.attention.reset_state()

    current_token, prompt_tokens = _prefill_and_compile(model, mesh_device, tokenizer, prompt, batch_size)

    host_times = []
    for step in range(num_steps):
        pos = len(prompt_tokens) - 1 + step
        next_token, prep, fwd, d2h = _decode_step_host(model, mesh_device, current_token, pos, batch_size, args)
        host_times.append((prep, fwd, d2h))
        current_token = next_token

    host_total, host_avg = _print_results("HOST SAMPLING (%d steps)" % num_steps, host_times, batch_size)

    # ── Phase 2: Device sampling ──
    print("\n--- Phase 2: DEVICE SAMPLING ---")
    assert model._supports_on_device_sampling, "Device sampling not supported"
    assert model.sampling is not None, "SamplingGenerator not initialized"

    # Reset GDN states
    for layer in model.layers:
        if hasattr(layer.attention, "reset_state"):
            layer.attention.reset_state()

    current_token, prompt_tokens = _prefill_and_compile(
        model,
        mesh_device,
        tokenizer,
        prompt,
        batch_size,
        use_device_sampling=True,
    )
    print("  Compile done, first token: %r" % tokenizer.decode([current_token]))

    device_times = []
    for step in range(num_steps):
        pos = len(prompt_tokens) - 1 + step
        next_token, prep, fwd, d2h = _decode_step_device(model, mesh_device, current_token, pos, batch_size)
        device_times.append((prep, fwd, d2h))
        current_token = next_token

    device_total, device_avg = _print_results("DEVICE SAMPLING (%d steps)" % num_steps, device_times, batch_size)

    # ── Comparison ──
    speedup = host_total / device_total if device_total > 0 else 0
    saved = host_total - device_total
    print("\n=== COMPARISON ===")
    print("  Host sampling:   %.1f ms/step (%.1f tok/s/user)" % (host_total, 1000.0 / host_total))
    print("  Device sampling: %.1f ms/step (%.1f tok/s/user)" % (device_total, 1000.0 / device_total))
    print("  Speedup:         %.1fx (saved %.0f ms/step)" % (speedup, saved))
    print("  D2H reduction:   %.1f ms -> %.1f ms" % (host_avg[2], device_avg[2]))
    print()
    print("  Next optimization: Tracing (eliminates ~%.0f ms Python dispatch)" % (device_avg[1] * 0.5))
