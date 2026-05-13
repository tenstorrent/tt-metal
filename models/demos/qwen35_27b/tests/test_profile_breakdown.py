# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Per-component decode step breakdown for Qwen3.5-27B.

Measures time spent in each section of the decode forward pass by
wrapping each layer with sync barriers. Calls the normal model forward
path so shapes/memory configs match exactly.

Results show GDN vs full-attention layer time, plus norm/LM-head overhead.
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


def _sync(mesh_device):
    ttnn.synchronize_device(mesh_device)
    return time.perf_counter()


def _wrap_layer(layer, mesh_device, timings, index):
    """Wrap a decoder layer's forward() with sync-barrier timing.

    Note: Python looks up dunder methods (__call__) on the type, not the instance,
    so we wrap forward() instead — LightweightModule.__call__ delegates to self.forward().
    """
    original_forward = layer.forward

    def timed_forward(*args, **kwargs):
        _sync(mesh_device)
        t0 = time.perf_counter()
        result = original_forward(*args, **kwargs)
        _sync(mesh_device)
        t1 = time.perf_counter()
        timings[index] = (t1 - t0) * 1000
        return result

    layer.forward = timed_forward
    layer._original_forward = original_forward
    return layer


def _unwrap_layer(layer):
    """Remove timing wrapper."""
    if hasattr(layer, "_original_forward"):
        layer.forward = layer._original_forward
        del layer._original_forward


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
def test_profile_breakdown(mesh_device, reset_seeds, ensure_gc):
    """Per-component timing breakdown of a single decode step."""
    model_path = _get_model_path()
    batch_size = 32
    max_seq_len = 2048

    if mesh_device.get_num_devices() < 4:
        pytest.skip("Full model requires TP>=4")

    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    logger.info("Creating model...")
    model = create_qwen35_model(
        mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        dtype=ttnn.bfloat8_b,
    )
    args = model.args
    prompt = "The capital of France is"

    # ── Prefill + compile (warmup) ──
    prompt_tokens = tokenizer.encode(prompt)
    for pos_idx in range(len(prompt_tokens) - 1):
        tok_batch = torch.full((batch_size,), prompt_tokens[pos_idx], dtype=torch.long)
        current_pos = torch.full((batch_size,), pos_idx, dtype=torch.long)
        tt_tokens, tt_current_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        model.ttnn_decode_forward(tt_tokens, tt_current_pos, rot_mat_idxs=tt_rot_idxs)

    # Compile step (ensures program cache is warm)
    tok_batch = torch.full((batch_size,), prompt_tokens[-1], dtype=torch.long)
    current_pos = torch.full((batch_size,), len(prompt_tokens) - 1, dtype=torch.long)
    tt_tokens, tt_current_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(tok_batch, current_pos)
    model.ttnn_decode_forward(tt_tokens, tt_current_pos, rot_mat_idxs=tt_rot_idxs)
    logger.info("Compile done.")

    # ── Install timing wrappers on all layers ──
    n_layers = len(model.layers)
    layer_timings = {}
    for i, layer in enumerate(model.layers):
        _wrap_layer(layer, mesh_device, layer_timings, i)

    # ── Run one decode step with timing ──
    pos = len(prompt_tokens)
    tok_batch = torch.full((batch_size,), prompt_tokens[-1], dtype=torch.long)
    current_pos_t = torch.full((batch_size,), pos, dtype=torch.long)

    t_prep0 = time.perf_counter()
    tt_tokens, tt_current_pos, tt_rot_idxs, _ = model.prepare_inputs_decode(tok_batch, current_pos_t)
    t_prep1 = time.perf_counter()

    # Full forward with sync barriers around it
    t_fwd0 = _sync(mesh_device)
    tt_logits, _ = model.ttnn_decode_forward(tt_tokens, tt_current_pos, rot_mat_idxs=tt_rot_idxs)
    t_fwd1 = _sync(mesh_device)

    # Remove wrappers
    for layer in model.layers:
        _unwrap_layer(layer)

    # ── Collect results ──
    prep_ms = (t_prep1 - t_prep0) * 1000
    total_fwd_ms = (t_fwd1 - t_fwd0) * 1000

    layer_times = [layer_timings.get(i, 0) for i in range(n_layers)]
    layer_type_list = args.layer_types if hasattr(args, "layer_types") else ["unknown"] * n_layers

    gdn_times = [t for t, lt in zip(layer_times, layer_type_list) if lt == "linear_attention"]
    attn_times = [t for t, lt in zip(layer_times, layer_type_list) if lt == "full_attention"]

    total_layers = sum(layer_times)
    total_gdn = sum(gdn_times)
    total_attn = sum(attn_times)
    overhead_ms = total_fwd_ms - total_layers  # embedding + norm + LM head + framework overhead

    # ── Print results ──
    print("\n" + "=" * 70)
    print("DECODE STEP BREAKDOWN (non-traced)")
    print("=" * 70)
    print(f"  Prepare inputs:      {prep_ms:7.1f} ms")
    print(f"  Total forward:       {total_fwd_ms:7.1f} ms")
    print(f"  ---")
    print(
        f"  GDN layers ({len(gdn_times)}):    {total_gdn:7.1f} ms  ({100*total_gdn/total_fwd_ms:.0f}%)  avg={total_gdn/max(len(gdn_times),1):.2f} ms/layer"
    )
    print(
        f"  Attn layers ({len(attn_times)}):   {total_attn:7.1f} ms  ({100*total_attn/total_fwd_ms:.0f}%)  avg={total_attn/max(len(attn_times),1):.2f} ms/layer"
    )
    print(f"  Overhead (emb+norm+lm): {overhead_ms:5.1f} ms  ({100*overhead_ms/total_fwd_ms:.0f}%)")
    print(f"  ---")
    print(
        f"  Throughput:          {1000.0/total_fwd_ms:.1f} tok/s/user  ({1000.0*batch_size/total_fwd_ms:.0f} tok/s aggregate)"
    )
    print()

    # Per-layer detail
    print("Per-layer times:")
    for i, (dt, lt) in enumerate(zip(layer_times, layer_type_list)):
        tag = "GDN " if lt == "linear_attention" else "ATTN"
        print(f"  Layer {i:2d} [{tag}]: {dt:6.2f} ms")

    # GDN analysis
    if gdn_times and attn_times:
        avg_gdn = total_gdn / len(gdn_times)
        avg_attn_layer = total_attn / len(attn_times)
        print()
        print("GDN analysis:")
        print(f"  Avg GDN layer:       {avg_gdn:.2f} ms")
        print(f"  Avg Attn layer:      {avg_attn_layer:.2f} ms")
        print(f"  GDN overhead/layer:  {avg_gdn - avg_attn_layer:.2f} ms")
        print(f"  GDN % of forward:    {100*total_gdn/total_fwd_ms:.1f}%")
        print(f"  Attn % of forward:   {100*total_attn/total_fwd_ms:.1f}%")
        print(f"  Non-layer % of fwd:  {100*overhead_ms/total_fwd_ms:.1f}%")
