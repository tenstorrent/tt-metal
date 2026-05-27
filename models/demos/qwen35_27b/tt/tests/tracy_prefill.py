#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Per-layer wall-clock timing for the Qwen3.5-27B prefill forward pass.

Uses pytest fixtures for device/fabric setup.

Usage (from tt-metal root):
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate

    MESH_DEVICE=P150x4 NUM_TOKENS=1024 N_LAYERS=4 \
    pytest models/demos/qwen35_27b/tt/tests/tracy_prefill.py -v -s
"""

import os
import time

import pytest
import torch

import ttnn
from models.demos.qwen35_27b.tt.model import create_qwen35_model

HF_MODEL = (
    "/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
)

NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "1024"))
N_LAYERS = int(os.environ.get("N_LAYERS", "4"))


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
    [{"fabric_config": True, "num_command_queues": 2}],
    indirect=True,
)
def test_tracy_prefill(mesh_device, reset_seeds, ensure_gc):
    os.environ.setdefault("HF_MODEL", HF_MODEL)
    max_seq = ((NUM_TOKENS + 127) // 128) * 128 + 128

    print(f"\nPrefill timing | ISL={NUM_TOKENS} | n_layers={N_LAYERS}")
    t0 = time.perf_counter()
    model = create_qwen35_model(
        mesh_device,
        model_path=HF_MODEL,
        max_batch_size=32,
        max_seq_len=max_seq,
        dtype=ttnn.bfloat8_b,
        n_layers=N_LAYERS,
    )
    print(f"Model loaded in {time.perf_counter()-t0:.1f}s")

    tok_ids = torch.arange(1, NUM_TOKENS + 1, dtype=torch.int32).reshape(1, 1, 1, NUM_TOKENS)
    tt_token_ids = ttnn.from_torch(
        tok_ids,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Warmup: compile all kernels
    print("Warmup (compile)...")
    model._reset_all_prefill_states(NUM_TOKENS)
    warmup_out = model._prefill_forward_device(tt_token_ids)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warmup_out)
    print("Warmup done.")

    # --- Pass 1: per-layer wall-clock totals ---
    print(f"\n--- Per-layer totals (ISL={NUM_TOKENS}) ---")
    os.environ["PREFILL_PROFILE"] = "1"
    model._reset_all_prefill_states(NUM_TOKENS)
    t = time.perf_counter()
    out = model._prefill_forward_device(tt_token_ids)
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = (time.perf_counter() - t) * 1000
    os.environ.pop("PREFILL_PROFILE", None)
    ttnn.deallocate(out)
    print(f"Total: {elapsed_ms:.1f} ms  (ISL={NUM_TOKENS}, n_layers={N_LAYERS})")

    # --- Pass 2: GDN sub-operation breakdown ---
    print(f"\n--- GDN sub-operation breakdown (ISL={NUM_TOKENS}) ---")
    os.environ["GDN_PROFILE"] = "1"
    model._reset_all_prefill_states(NUM_TOKENS)
    t = time.perf_counter()
    out2 = model._prefill_forward_device(tt_token_ids)
    ttnn.synchronize_device(mesh_device)
    elapsed_ms2 = (time.perf_counter() - t) * 1000
    os.environ.pop("GDN_PROFILE", None)
    ttnn.deallocate(out2)
    print(f"Total (GDN-profile run): {elapsed_ms2:.1f} ms  (ISL={NUM_TOKENS}, n_layers={N_LAYERS})")

    ttnn.deallocate(tt_token_ids)
