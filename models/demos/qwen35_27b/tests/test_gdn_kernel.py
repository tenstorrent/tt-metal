# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
GDN fused kernel correctness test.

Compares the fused recurrence kernel output against the PyTorch reference
implementation (torch_recurrent_gated_delta_rule from HF transformers).

Tests the kernel in isolation: no conv1d, no projections, no all-reduce.
Just the raw recurrence: state decay + delta update + output computation.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_recurrence_fused_inplace


# Reference implementation from HF transformers (modeling_qwen3_next.py)
def ref_recurrence_single_step(q, k, v, g, beta, state):
    """Single-step GDN recurrence in PyTorch (float32).

    Args:
        q: [num_pairs, 1, Dk] — already L2-normed and scaled
        k: [num_pairs, 1, Dk] — already L2-normed
        v: [num_pairs, 1, Dv]
        g: [num_pairs, 1, 1] — log-space decay (negative values)
        beta: [num_pairs, 1, 1]
        state: [num_pairs, Dk, Dv] — recurrence state

    Returns:
        output: [num_pairs, 1, Dv]
        new_state: [num_pairs, Dk, Dv]
    """
    num_pairs, _, Dk = q.shape
    Dv = v.shape[-1]

    # Step 1: decay
    g_exp = g.exp().unsqueeze(-1)  # [num_pairs, 1, 1, 1] -> broadcast over state
    new_state = state * g_exp.squeeze(1)  # [num_pairs, Dk, Dv]

    # Step 2: kv_mem = k @ state (dot product over Dk dim)
    # k: [num_pairs, 1, Dk], state: [num_pairs, Dk, Dv]
    kv_mem = torch.bmm(k, new_state)  # [num_pairs, 1, Dv]

    # Step 3: delta = beta * (v - kv_mem)
    delta = beta * (v - kv_mem)  # [num_pairs, 1, Dv]

    # Step 4: state += outer(k, delta) = k^T @ delta
    # k^T: [num_pairs, Dk, 1], delta: [num_pairs, 1, Dv]
    new_state = new_state + torch.bmm(k.transpose(-2, -1), delta)

    # Step 5: output = q @ state
    output = torch.bmm(q, new_state)  # [num_pairs, 1, Dv]

    return output, new_state


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],  # Single device for kernel test
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("num_pairs", [10, 32, 384])
def test_gdn_kernel_correctness(mesh_device, reset_seeds, ensure_gc, num_pairs):
    """Test fused GDN kernel against PyTorch reference for a single decode step."""
    device = mesh_device

    Dk, Dv = 128, 128
    Kt, Vt = Dk // 32, Dv // 32  # tiles
    num_cores = min(10, num_pairs)  # up to 10 cores
    logger.info(f"Testing kernel: num_pairs={num_pairs}, Dk={Dk}, Dv={Dv}")

    # Create random inputs in float32 for reference
    torch.manual_seed(42)
    q_ref = torch.randn(num_pairs, 1, Dk, dtype=torch.float32) * 0.1
    k_ref = torch.randn(num_pairs, 1, Dk, dtype=torch.float32) * 0.1
    v_ref = torch.randn(num_pairs, 1, Dv, dtype=torch.float32) * 0.1
    g_ref = torch.randn(num_pairs, 1, 1, dtype=torch.float32) * 0.5 - 1.0  # negative values
    beta_ref = torch.sigmoid(torch.randn(num_pairs, 1, 1, dtype=torch.float32))
    state_ref = torch.randn(num_pairs, Dk, Dv, dtype=torch.float32) * 0.01

    # Run reference
    out_ref, state_new_ref = ref_recurrence_single_step(q_ref, k_ref, v_ref, g_ref, beta_ref, state_ref.clone())
    logger.info(f"Reference output: shape={out_ref.shape}, range=[{out_ref.min():.4f}, {out_ref.max():.4f}]")

    # Convert to bfloat16 for device
    def to_tt(t):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    q_tt = to_tt(q_ref)
    k_row_tt = to_tt(k_ref)
    k_col_tt = to_tt(k_ref.transpose(-2, -1))
    v_tt = to_tt(v_ref)
    g_tt = to_tt(g_ref)
    beta_tt = to_tt(beta_ref)
    state_tt = ttnn.from_torch(
        state_ref.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    output_tt = ttnn.from_torch(
        torch.zeros(num_pairs, 1, Dv, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Run fused kernel
    gdn_recurrence_fused_inplace(
        q_tt,
        k_row_tt,
        k_col_tt,
        v_tt,
        g_tt,
        beta_tt,
        state_tt,
        output_tt,
        num_cores=num_cores,
    )

    # Get results back to CPU
    out_tt_cpu = ttnn.to_torch(output_tt).float()
    state_tt_cpu = ttnn.to_torch(state_tt).float()

    logger.info(f"Kernel output: shape={out_tt_cpu.shape}, range=[{out_tt_cpu.min():.4f}, {out_tt_cpu.max():.4f}]")

    # Compare output
    out_diff = (out_ref - out_tt_cpu).abs()
    out_max_diff = out_diff.max().item()
    out_mean_diff = out_diff.mean().item()

    # PCC (Pearson correlation coefficient)
    out_ref_flat = out_ref.flatten()
    out_tt_flat = out_tt_cpu.flatten()
    pcc = torch.corrcoef(torch.stack([out_ref_flat, out_tt_flat]))[0, 1].item()

    logger.info(f"Output comparison:")
    logger.info(f"  Max diff: {out_max_diff:.6f}")
    logger.info(f"  Mean diff: {out_mean_diff:.6f}")
    logger.info(f"  PCC: {pcc:.6f}")

    # Compare state
    state_diff = (state_new_ref - state_tt_cpu).abs()
    state_max_diff = state_diff.max().item()
    state_pcc = torch.corrcoef(torch.stack([state_new_ref.flatten(), state_tt_cpu.flatten()]))[0, 1].item()

    logger.info(f"State comparison:")
    logger.info(f"  Max diff: {state_max_diff:.6f}")
    logger.info(f"  PCC: {state_pcc:.6f}")

    # Assert correctness (ttnn ops achieve PCC > 0.999 with bfloat16)
    assert pcc > 0.999, f"Output PCC too low: {pcc:.6f}"
    assert state_pcc > 0.999, f"State PCC too low: {state_pcc:.6f}"

    logger.info(f"PASSED: kernel matches reference (output PCC={pcc:.4f}, state PCC={state_pcc:.4f})")
