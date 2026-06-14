# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Performance and full-model correctness test for NemotronH-30B on QB TP=4.

Two tests:
  1. test_full_52layer_pcc — all 52 layers hidden-state PCC vs pure-PyTorch reference
  2. test_decode_latency   — B=1 S=1 decode, 2 warm-up + 5 timed runs; reports ms/tok

Usage:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd) PYTHONPATH=ttnn:tools:.
    export LD_LIBRARY_PATH=build_Release/lib:/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
    pytest models/demos/nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16/tests/test_perf.py -v -s
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

PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
N_LAYERS = 52
PCC_THRESHOLD = 0.99
N_WARMUP = 2
N_TIMED = 5


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a -= a.mean()
    b -= b.mean()
    denom = torch.sqrt((a**2).sum() * (b**2).sum())
    return ((a * b).sum() / denom).item() if denom.item() != 0.0 else 1.0


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


def _ref_forward_52(input_ids: torch.Tensor, wc) -> torch.Tensor:
    """Pure-PyTorch reference for all 52 layers (hidden states, no LM head)."""
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.reference.functional import (
        dense_attention,
        layer_norm,
        mamba2_layer,
        moe_experts,
        moe_gate,
        shared_expert,
    )

    B, S = input_ids.shape
    h = torch.nn.functional.embedding(input_ids, wc["backbone.embeddings.weight"])

    for li in range(N_LAYERS):
        lt = PATTERN[li]
        p = f"backbone.layers.{li}"

        if lt == "M":
            h = mamba2_layer(
                hidden_states=h,
                norm_weight=wc[f"{p}.norm.weight"],
                in_proj_weight=wc[f"{p}.mixer.in_proj.weight"],
                conv1d_weight=wc[f"{p}.mixer.conv1d.weight"],
                conv1d_bias=wc[f"{p}.mixer.conv1d.bias"],
                dt_bias=wc[f"{p}.mixer.dt_bias"],
                A_log=wc[f"{p}.mixer.A_log"],
                norm_mixer_weight=wc[f"{p}.mixer.norm.weight"],
                D=wc[f"{p}.mixer.D"],
                out_proj_weight=wc[f"{p}.mixer.out_proj.weight"],
            )
        elif lt == "E":
            residual = h
            normed = layer_norm(h, wc[f"{p}.norm.weight"])
            flat = normed.reshape(B * S, -1)
            topk_idx, topk_wts = moe_gate(
                flat,
                wc[f"{p}.mixer.gate.weight"],
                wc[f"{p}.mixer.gate.e_score_correction_bias"],
            )
            eu = [wc[f"{p}.mixer.experts.{e}.up_proj.weight"] for e in range(128)]
            ed = [wc[f"{p}.mixer.experts.{e}.down_proj.weight"] for e in range(128)]
            ex_out = moe_experts(flat, topk_idx, topk_wts, eu, ed).reshape(B, S, -1)
            sh_out = shared_expert(
                normed,
                wc[f"{p}.mixer.shared_experts.up_proj.weight"],
                wc[f"{p}.mixer.shared_experts.down_proj.weight"],
            )
            h = (residual + ex_out + sh_out).bfloat16()
        else:  # '*'
            h = dense_attention(
                h,
                norm_weight=wc[f"{p}.norm.weight"],
                wq=wc[f"{p}.mixer.q_proj.weight"],
                wk=wc[f"{p}.mixer.k_proj.weight"],
                wv=wc[f"{p}.mixer.v_proj.weight"],
                wo=wc[f"{p}.mixer.o_proj.weight"],
            )
    return h


def test_full_52layer_pcc(mesh_device, weight_cache):
    """Hidden-state PCC for all 52 layers (TTNN TP=4 vs pure-PyTorch reference)."""

    torch.manual_seed(42)
    input_ids = torch.randint(0, 131072, (1, 1), dtype=torch.long)  # S=1: Mamba2 decode kernel

    print(f"\nRunning reference forward (52 layers)...")
    ref_h = _ref_forward_52(input_ids, weight_cache)

    print(f"Running TTNN TP=4 forward (52 layers, hidden states)...")
    wc = weight_cache
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.dense_attention import dense_attention_forward
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.embedding import embedding_forward
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.mamba2_layer import mamba2_layer_forward
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import _moe_layer_forward

    B, S = input_ids.shape
    h = embedding_forward(mesh_device, input_ids, wc["backbone.embeddings.weight"])
    for li in range(N_LAYERS):
        lt = PATTERN[li]
        p = f"backbone.layers.{li}"
        if lt == "M":
            h, _ = mamba2_layer_forward(
                mesh_device,
                h,
                norm_weight=wc[f"{p}.norm.weight"],
                in_proj_weight=wc[f"{p}.mixer.in_proj.weight"],
                conv1d_weight=wc[f"{p}.mixer.conv1d.weight"],
                conv1d_bias=wc[f"{p}.mixer.conv1d.bias"],
                dt_bias=wc[f"{p}.mixer.dt_bias"],
                A_log=wc[f"{p}.mixer.A_log"],
                norm_mixer_weight=wc[f"{p}.mixer.norm.weight"],
                D=wc[f"{p}.mixer.D"],
                out_proj_weight=wc[f"{p}.mixer.out_proj.weight"],
            )
        elif lt == "E":
            h = _moe_layer_forward(mesh_device, h, li, wc)
        else:
            h = dense_attention_forward(
                mesh_device,
                h,
                norm_weight=wc[f"{p}.norm.weight"],
                wq=wc[f"{p}.mixer.q_proj.weight"],
                wk=wc[f"{p}.mixer.k_proj.weight"],
                wv=wc[f"{p}.mixer.v_proj.weight"],
                wo=wc[f"{p}.mixer.o_proj.weight"],
            )

    # h is ttnn.Tensor — bring to CPU for PCC comparison
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import _host_rep

    B = input_ids.shape[0]
    h_cpu = _host_rep(h, mesh_device, B)

    score = pcc(h_cpu, ref_h)
    print(f"\nFull-model hidden-state PCC (52 layers): {score:.6f}")
    assert score >= PCC_THRESHOLD, f"PCC {score:.6f} < {PCC_THRESHOLD}"


def test_decode_latency(mesh_device, weight_cache):
    """Decode latency: B=1, S=1 through all 52 layers + LM head on TP=4."""
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import nemotron_h_forward

    torch.manual_seed(7)
    input_ids = torch.randint(0, 131072, (1, 1), dtype=torch.long)

    print(f"\nWarming up ({N_WARMUP} runs, B=1 S=1)...")
    for _ in range(N_WARMUP):
        _ = nemotron_h_forward(mesh_device, input_ids, wc=weight_cache)

    print(f"Timing {N_TIMED} runs...")
    latencies = []
    for i in range(N_TIMED):
        t0 = time.perf_counter()
        logits = nemotron_h_forward(mesh_device, input_ids, wc=weight_cache)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        print(f"  run {i+1}: {latencies[-1]:.1f} ms  (logits shape {logits.shape})")

    avg_ms = sum(latencies) / len(latencies)
    min_ms = min(latencies)
    toks_per_sec = 1000.0 / avg_ms

    print(f"\nDecode latency B=1 S=1 (52 layers + LM head, TP=4):")
    print(f"  avg: {avg_ms:.1f} ms  min: {min_ms:.1f} ms  throughput: {toks_per_sec:.2f} tok/s")

    # Sanity: result is finite, correct shape
    assert logits.shape == (1, 1, 131072)
    assert torch.isfinite(logits).all(), "logits contain NaN/Inf"


@pytest.mark.timeout(0)
def test_decode_traced(mesh_device, weight_cache):
    """Traced decode: B=1, S=1 through all 52 layers + LM head on TP=4.

    Captures a TTNN trace on a fixed token, then times N_TIMED replay
    executions.  MoE routing is deterministic for a fixed input so the
    trace correctly represents one complete decode step.
    """
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.model import nemotron_h_forward_device
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.tp import _host_rep

    N_TRACED = 10

    torch.manual_seed(7)
    token_cpu = torch.randint(0, 131072, (1, 1), dtype=torch.int32)

    # Pre-allocate the persistent device input tensor (trace input).
    ids_tt = ttnn.from_torch(
        token_cpu,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    wc = weight_cache

    print("\nTrace warm-up (1 eager run to compile device kernels)...")
    _ = nemotron_h_forward_device(mesh_device, ids_tt, wc)
    ttnn.synchronize_device(mesh_device)

    print("Capturing trace...")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    logits_tt = nemotron_h_forward_device(mesh_device, ids_tt, wc)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    print("Trace captured.")

    # Verify shape before timing loop.
    logits_cpu = _host_rep(logits_tt, mesh_device, 1)
    assert logits_cpu.shape == (1, 1, 131072), f"unexpected logits shape {logits_cpu.shape}"
    assert torch.isfinite(logits_cpu).all(), "logits contain NaN/Inf after trace capture"

    print(f"Timing {N_TRACED} traced executions (same token)...")
    latencies = []
    ids_host = ttnn.from_torch(
        token_cpu,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    for i in range(N_TRACED):
        ttnn.copy_host_to_device_tensor(ids_host, ids_tt)
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        print(f"  run {i + 1}: {latencies[-1]:.1f} ms")

    ttnn.release_trace(mesh_device, trace_id)

    avg_ms = sum(latencies) / len(latencies)
    min_ms = min(latencies)
    print(f"\nTraced decode B=1 S=1 (52 layers + LM head, TP=4):")
    print(f"  avg: {avg_ms:.1f} ms  min: {min_ms:.1f} ms  throughput: {1000 / avg_ms:.2f} tok/s")
