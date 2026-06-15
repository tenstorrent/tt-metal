# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stateful decode test for NemotronH-30B on QB TP=4 (4× Blackhole).

Exercises SSM state persistence, paged KV cache, and the end-to-end
generation loop on device.

Usage:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd) PYTHONPATH=ttnn:tools:.
    export LD_LIBRARY_PATH=build_Release/lib:/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
    pytest models/demos/nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16/tests/test_stateful_decode.py -v -s --timeout=0
"""

import os
import sys
import time

os.environ.setdefault("TT_METAL_HOME", "/home/ttuser/ssinghal/tt-metal")
_root = os.environ["TT_METAL_HOME"]
for p in (f"{_root}/ttnn", f"{_root}/tools", _root):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
import torch

import ttnn

N_GENERATED = 10  # tokens to generate beyond the single prefill token


@pytest.fixture(scope="module")
def mesh_device():
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import close_device_tp4, open_device_tp4

    dev = open_device_tp4()
    yield dev
    close_device_tp4(dev)


@pytest.fixture(scope="module")
def weight_cache():
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import WeightCache

    return WeightCache()


@pytest.mark.timeout(0)
def test_stateful_decode_shapes_and_finiteness(mesh_device, weight_cache):
    """Run N_GENERATED stateful decode steps; check shape and finiteness each step."""
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.kv_cache import allocate_decoder_state
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import nemotron_h_forward_stateful
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import _host_rep

    wc = weight_cache
    torch.manual_seed(42)
    token = torch.randint(0, 131072, (1, 1), dtype=torch.int32)
    ids_tt = ttnn.from_torch(
        token,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    state = allocate_decoder_state(mesh_device, B=1)
    ids_host = ttnn.from_torch(token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    pos_host = ttnn.from_torch(torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    print(f"\nRunning {N_GENERATED} stateful steps...")
    latencies = []
    for step in range(N_GENERATED):
        # Update position (outside forward)
        pos_cpu = torch.tensor([step], dtype=torch.int32)
        pos_host = ttnn.from_torch(pos_cpu, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(pos_host, state.current_pos)

        t0 = time.perf_counter()
        logits_tt = nemotron_h_forward_stateful(mesh_device, ids_tt, wc, state)
        ttnn.synchronize_device(mesh_device)
        latencies.append((time.perf_counter() - t0) * 1000)

        state.advance()

        logits_cpu = _host_rep(logits_tt, mesh_device, 1)
        assert logits_cpu.shape == (1, 1, 131072), f"step {step}: shape {logits_cpu.shape}"
        assert torch.isfinite(logits_cpu).all(), f"step {step}: non-finite logits"

        # Use greedy next token for next step
        next_tok = int(logits_cpu[0, 0].argmax())
        tok_cpu = torch.tensor([[next_tok]], dtype=torch.int32)
        ids_host = ttnn.from_torch(tok_cpu, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(ids_host, ids_tt)

        print(f"  step {step}: {latencies[-1]:.1f} ms  next_tok={next_tok}")

    avg_ms = sum(latencies) / len(latencies)
    print(f"\nStateful decode: avg {avg_ms:.1f} ms/tok over {N_GENERATED} steps (eager, no trace)")


@pytest.mark.timeout(0)
def test_generate_e2e(mesh_device, weight_cache):
    """End-to-end generate() call: tokenize → prefill → traced decode → text."""
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.generate import generate

    prompt = "The key insight of Mamba is that"
    print(f"\nPrompt: {repr(prompt)}")

    t0 = time.perf_counter()
    result = generate(
        prompt,
        mesh_device=mesh_device,
        wc=weight_cache,
        max_new_tokens=20,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nResult: {repr(result)}")
    print(f"Total time: {elapsed:.1f}s")

    assert result.startswith(prompt), "output does not start with prompt"
    assert len(result) > len(prompt), "no tokens generated"
