# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Correctness + timing test for gdn_prefill_ttnn_opt.

Validates that the optimized kernel (pre-loop projections, per-token recurrence
only) produces output and final state matching the original gdn_prefill_ttnn,
then reports the wall-clock speedup.

Weight-free: uses hardcoded Qwen3.5-27B / P150x4 (TP=4) dimensions with
synthetic random inputs — no model checkpoint required.

Run:
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd)
    source python_env/bin/activate
    MESH_DEVICE=P150x4 pytest models/demos/qwen35_27b/tt/tests/test_gdn_prefill_ttnn_opt.py -v -s
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn

# ── Qwen3.5-27B GDN constants (from model_config.py) ──────────────────────
GDN_Nk = 16  # Total key heads
GDN_Nv = 48  # Total value heads
GDN_Dk = 128  # Key head dim
GDN_Dv = 128  # Value head dim
TP = 4  # Tensor-parallelism (P150x4)

Nk_TP = GDN_Nk // TP  # 4
Nv_TP = GDN_Nv // TP  # 12
Dk = GDN_Dk  # 128
Dv = GDN_Dv  # 128
repeat_factor = Nv_TP // Nk_TP  # 3
key_dim_tp = Nk_TP * Dk  # 512
value_dim_tp = Nv_TP * Dv  # 1536
qkv_dim_tp = 2 * key_dim_tp + value_dim_tp  # 2560
B = 1
num_pairs = B * Nv_TP  # 12
scale = Dk**-0.5


def _to_mesh(t, mesh_device):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _unshard(t):
    if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
    return t


def _compute_pcc(ref, test):
    r = ref.float().flatten()
    t = test.float().flatten()
    if r.numel() == 0:
        return 1.0
    vr = r - r.mean()
    vt = t - t.mean()
    num = (vr * vt).sum()
    den = (vr.norm() * vt.norm()) + 1e-12
    return (num / den).item()


def _make_kernel_params(mesh_device):
    """Create synthetic device tensors matching the shapes gdn.py supplies."""
    torch.manual_seed(7)
    # neg_exp_A: -exp(A_log) values — small negatives broadcast over N tokens
    neg_exp_A = _to_mesh(torch.full((1, 1, Nv_TP), -0.1, dtype=torch.bfloat16), mesh_device)
    # dt_bias: additive bias on 'a' input
    dt_bias = _to_mesh(torch.zeros(1, 1, Nv_TP, dtype=torch.bfloat16), mesh_device)
    # norm_w: RMS norm weight — not used by gdn_prefill_ttnn, pass dummy
    norm_w = _to_mesh(torch.ones(1, 1, value_dim_tp, dtype=torch.bfloat16), mesh_device)
    # scale_tt / rms_scale_tt / rms_eps_tt: scalar tiles
    scale_tt = _to_mesh(torch.full((1, 1, 1), scale, dtype=torch.bfloat16), mesh_device)
    rms_scale_tt = _to_mesh(torch.full((1, 1, 1), Dv**0.5, dtype=torch.bfloat16), mesh_device)
    rms_eps_tt = _to_mesh(torch.full((1, 1, 1), Dv * 1e-6, dtype=torch.bfloat16), mesh_device)
    return neg_exp_A, dt_bias, norm_w, scale_tt, rms_scale_tt, rms_eps_tt


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
@pytest.mark.parametrize("num_tokens", [32, 64])
def test_gdn_prefill_ttnn_opt_correctness(mesh_device, reset_seeds, ensure_gc, num_tokens):
    """gdn_prefill_ttnn_opt must match gdn_prefill_ttnn output and state (PCC > 0.99)."""
    from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op_ttnn import gdn_prefill_ttnn
    from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op_ttnn_v2 import gdn_prefill_ttnn_opt

    if mesh_device.get_num_devices() < 4:
        pytest.skip("P150x4 TP=4 required")

    N = num_tokens
    logger.info(f"N={N}, num_pairs={num_pairs}, Nk_TP={Nk_TP}, Nv_TP={Nv_TP}, Dk={Dk}, Dv={Dv}")

    # Kernel parameters (synthetic, weight-free)
    neg_exp_A, dt_bias, norm_w, scale_tt, rms_scale_tt, rms_eps_tt = _make_kernel_params(mesh_device)

    # Random inputs
    torch.manual_seed(42)
    conv_bf16 = torch.randn(1, N, qkv_dim_tp, dtype=torch.bfloat16) * 0.1
    a_bf16 = torch.randn(1, N, Nv_TP, dtype=torch.bfloat16) * 0.1
    b_bf16 = torch.randn(1, N, Nv_TP, dtype=torch.bfloat16) * 0.1

    common_kwargs = dict(
        num_pairs=num_pairs,
        num_tokens=N,
        Nv_TP=Nv_TP,
        Nk_TP=Nk_TP,
        repeat_factor=repeat_factor,
        key_dim_tp=key_dim_tp,
    )

    def _run(fn, label):
        conv = _unshard(_to_mesh(conv_bf16, mesh_device))
        a = _unshard(_to_mesh(a_bf16, mesh_device))
        b = _unshard(_to_mesh(b_bf16, mesh_device))
        st = _to_mesh(torch.zeros(num_pairs, Dk, Dv, dtype=torch.bfloat16), mesh_device)
        out = _to_mesh(torch.zeros(num_pairs * N, 1, Dv, dtype=torch.bfloat16), mesh_device)

        t0 = time.perf_counter()
        fn(conv, a, b, neg_exp_A, dt_bias, norm_w, scale_tt, rms_scale_tt, rms_eps_tt, st, out, **common_kwargs)
        ttnn.synchronize_device(mesh_device)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        out_cpu = (
            ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[: num_pairs * N]
            .float()
            .reshape(num_pairs, N, Dv)
        )
        st_cpu = ttnn.to_torch(st, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:num_pairs].float()

        for t in (conv, a, b, st, out):
            ttnn.deallocate(t)
        logger.info(f"  {label}: {elapsed_ms:.1f} ms")
        return out_cpu, st_cpu, elapsed_ms

    # Warmup both paths (compilation)
    logger.info("Warmup — original...")
    _run(gdn_prefill_ttnn, "warmup-orig")
    logger.info("Warmup — optimized...")
    _run(gdn_prefill_ttnn_opt, "warmup-opt")

    # Timed runs
    logger.info(f"Timed — original (N={N})...")
    ref_out, ref_state, t_orig = _run(gdn_prefill_ttnn, "orig")
    logger.info(f"Timed — optimized (N={N})...")
    opt_out, opt_state, t_opt = _run(gdn_prefill_ttnn_opt, "opt")

    out_pcc = _compute_pcc(ref_out, opt_out)
    state_pcc = _compute_pcc(ref_state, opt_state)
    speedup = t_orig / t_opt if t_opt > 0 else float("inf")

    logger.info(f"  Output PCC:  {out_pcc:.6f}")
    logger.info(f"  State  PCC:  {state_pcc:.6f}")
    logger.info(f"  Speedup: {speedup:.2f}× ({t_orig:.1f} ms → {t_opt:.1f} ms)")

    for t in (neg_exp_A, dt_bias, norm_w, scale_tt, rms_scale_tt, rms_eps_tt):
        ttnn.deallocate(t)

    assert out_pcc > 0.99, f"Output PCC {out_pcc:.6f} < 0.99"
    assert state_pcc > 0.99, f"State PCC {state_pcc:.6f} < 0.99"
    logger.info(f"PASS N={N}: out_pcc={out_pcc:.6f}, state_pcc={state_pcc:.6f}, speedup={speedup:.2f}×")
