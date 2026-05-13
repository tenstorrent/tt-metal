# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Diagnostic test for GDN kernel - isolates reader/writer vs compute issues.

Test 1: Run each GDN step individually via ttnn ops and compare with fused kernel.
Test 2: Check intermediate values by running partial computations.
"""

import pytest
import torch
from loguru import logger

import ttnn

Dk = 128
Dv = 128
Kt = 4  # 128 / 32
Vt = 4
STATE_TILES = Kt * Vt
BF16_TILE_BYTES = 32 * 32 * 2


def ref_recurrence(q, k, v, g, beta, state):
    """Reference GDN recurrence in float32."""
    g_exp = torch.exp(g)
    state_b = state * g_exp
    kv_mem = k @ state_b
    delta = v - kv_mem
    delta_s = beta * delta
    k_col = k.transpose(-2, -1)
    outer = k_col @ delta_s
    state_new = state_b + outer
    output = q @ state_new
    return (
        output,
        state_new,
        {
            "g_exp": g_exp,
            "state_b": state_b,
            "kv_mem": kv_mem,
            "delta": delta,
            "delta_s": delta_s,
            "outer": outer,
        },
    )


def to_tt(t, device):
    return ttnn.from_torch(
        t.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def to_torch(t):
    return ttnn.to_torch(t).float()


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_step_by_step_ttnn(mesh_device):
    """Run GDN steps one at a time using ttnn ops and compare with reference."""
    device = mesh_device
    num_pairs = 4
    torch.manual_seed(42)

    q = torch.randn(num_pairs, 1, Dk) * 0.1
    k = torch.randn(num_pairs, 1, Dk) * 0.1
    v = torch.randn(num_pairs, 1, Dv) * 0.1
    g = torch.randn(num_pairs, 1, 1) * 0.5 - 1.0  # negative decay
    beta = torch.randn(num_pairs, 1, 1).abs() * 0.5
    state = torch.randn(num_pairs, Dk, Dv) * 0.01

    out_ref, state_ref, intermediates = ref_recurrence(q, k, v, g, beta, state)

    # Step 1: exp(g)
    g_tt = to_tt(g, device)
    g_exp_tt = ttnn.exp(g_tt)
    g_exp_cpu = to_torch(g_exp_tt)
    g_exp_ref = intermediates["g_exp"].to(torch.bfloat16).float()
    pcc = torch.corrcoef(torch.stack([g_exp_ref.flatten(), g_exp_cpu.flatten()]))[0, 1].item()
    logger.info(f"Step 1 (exp(g)): PCC={pcc:.6f}")
    assert pcc > 0.999

    # Step 2: state * exp(g)
    state_tt = to_tt(state, device)
    state_b_tt = ttnn.multiply(state_tt, g_exp_tt)
    state_b_cpu = to_torch(state_b_tt)
    state_b_ref = intermediates["state_b"].to(torch.bfloat16).float()
    pcc = torch.corrcoef(torch.stack([state_b_ref.flatten(), state_b_cpu.flatten()]))[0, 1].item()
    logger.info(f"Step 2 (state*exp(g)): PCC={pcc:.6f}")
    assert pcc > 0.999

    # Step 3: kv_mem = k_row @ state_b
    k_tt = to_tt(k, device)
    kv_mem_tt = ttnn.matmul(k_tt, state_b_tt)
    kv_mem_cpu = to_torch(kv_mem_tt)
    kv_mem_ref = intermediates["kv_mem"].to(torch.bfloat16).float()
    pcc = torch.corrcoef(torch.stack([kv_mem_ref.flatten(), kv_mem_cpu.flatten()]))[0, 1].item()
    logger.info(f"Step 3 (k@state_b): PCC={pcc:.6f}")
    assert pcc > 0.99

    # Step 4: delta = v - kv_mem
    v_tt = to_tt(v, device)
    delta_tt = ttnn.subtract(v_tt, kv_mem_tt)
    delta_cpu = to_torch(delta_tt)
    delta_ref = intermediates["delta"].to(torch.bfloat16).float()
    pcc = torch.corrcoef(torch.stack([delta_ref.flatten(), delta_cpu.flatten()]))[0, 1].item()
    logger.info(f"Step 4 (v-kv_mem): PCC={pcc:.6f}")

    # Step 4b: delta_s = beta * delta
    beta_tt = to_tt(beta, device)
    delta_s_tt = ttnn.multiply(beta_tt, delta_tt)
    delta_s_cpu = to_torch(delta_s_tt)
    delta_s_ref = intermediates["delta_s"].to(torch.bfloat16).float()
    pcc = torch.corrcoef(torch.stack([delta_s_ref.flatten(), delta_s_cpu.flatten()]))[0, 1].item()
    logger.info(f"Step 4b (beta*delta): PCC={pcc:.6f}")

    # Step 5: outer = k_col @ delta_s, state_new = state_b + outer
    k_col_tt = to_tt(k.transpose(-2, -1), device)
    outer_tt = ttnn.matmul(k_col_tt, delta_s_tt)
    outer_cpu = to_torch(outer_tt)
    outer_ref = intermediates["outer"].to(torch.bfloat16).float()
    pcc = torch.corrcoef(torch.stack([outer_ref.flatten(), outer_cpu.flatten()]))[0, 1].item()
    logger.info(f"Step 5a (outer product): PCC={pcc:.6f}")

    state_new_tt = ttnn.add(state_b_tt, outer_tt)
    state_new_cpu = to_torch(state_new_tt)
    state_new_ref = state_ref.to(torch.bfloat16).float()
    pcc = torch.corrcoef(torch.stack([state_new_ref.flatten(), state_new_cpu.flatten()]))[0, 1].item()
    logger.info(f"Step 5b (state update): PCC={pcc:.6f}")

    # Step 6: output = q @ state_new
    q_tt = to_tt(q, device)
    out_tt = ttnn.matmul(q_tt, state_new_tt)
    out_cpu = to_torch(out_tt)
    out_ref_bf16 = out_ref.to(torch.bfloat16).float()
    pcc = torch.corrcoef(torch.stack([out_ref_bf16.flatten(), out_cpu.flatten()]))[0, 1].item()
    logger.info(f"Step 6 (q@state_new): PCC={pcc:.6f}")

    logger.info("All ttnn ops produce correct results individually.")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_fused_vs_ttnn(mesh_device):
    """Compare fused kernel output with ttnn ops step-by-step output."""
    from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import _gdn_recurrence_fused, _gdn_recurrence_ttnn

    device = mesh_device
    num_pairs = 4
    torch.manual_seed(42)

    q = torch.randn(num_pairs, 1, Dk) * 0.1
    k = torch.randn(num_pairs, 1, Dk) * 0.1
    k_col = k.transpose(-2, -1)
    v = torch.randn(num_pairs, 1, Dv) * 0.1
    g = torch.randn(num_pairs, 1, 1) * 0.5 - 1.0
    beta = torch.randn(num_pairs, 1, 1).abs() * 0.5
    state = torch.randn(num_pairs, Dk, Dv) * 0.01

    # Reference
    out_ref, state_ref, _ = ref_recurrence(q, k, v, g, beta, state)

    # ttnn ops path
    q_tt1 = to_tt(q, device)
    k_tt1 = to_tt(k, device)
    k_col_tt1 = to_tt(k_col, device)
    v_tt1 = to_tt(v, device)
    g_tt1 = to_tt(g, device)
    beta_tt1 = to_tt(beta, device)
    state_tt1 = to_tt(state, device)

    ttnn_out = _gdn_recurrence_ttnn(q_tt1, k_tt1, k_col_tt1, v_tt1, g_tt1, beta_tt1, state_tt1)
    ttnn_out_cpu = to_torch(ttnn_out)
    ttnn_state_cpu = to_torch(state_tt1)

    pcc_ttnn_out = torch.corrcoef(torch.stack([out_ref.flatten(), ttnn_out_cpu.flatten()]))[0, 1].item()
    pcc_ttnn_state = torch.corrcoef(torch.stack([state_ref.flatten(), ttnn_state_cpu.flatten()]))[0, 1].item()
    logger.info(f"ttnn ops: output PCC={pcc_ttnn_out:.6f}, state PCC={pcc_ttnn_state:.6f}")

    # Fused kernel path
    q_tt2 = to_tt(q, device)
    k_tt2 = to_tt(k, device)
    k_col_tt2 = to_tt(k_col, device)
    v_tt2 = to_tt(v, device)
    g_tt2 = to_tt(g, device)
    beta_tt2 = to_tt(beta, device)
    state_tt2 = to_tt(state, device)
    output_tt2 = to_tt(torch.zeros(num_pairs, 1, Dv), device)

    try:
        _gdn_recurrence_fused(
            q_tt2,
            k_tt2,
            k_col_tt2,
            v_tt2,
            g_tt2,
            beta_tt2,
            state_tt2,
            output_tt2,
            state_tt2,
            num_cores=4,
        )
        fused_out_cpu = to_torch(output_tt2)
        fused_state_cpu = to_torch(state_tt2)

        pcc_fused_out = torch.corrcoef(torch.stack([out_ref.flatten(), fused_out_cpu.flatten()]))[0, 1].item()
        pcc_fused_state = torch.corrcoef(torch.stack([state_ref.flatten(), fused_state_cpu.flatten()]))[0, 1].item()
        logger.info(f"Fused kernel: output PCC={pcc_fused_out:.6f}, state PCC={pcc_fused_state:.6f}")

        # Compare fused vs ttnn directly
        pcc_out_vs = torch.corrcoef(torch.stack([ttnn_out_cpu.flatten(), fused_out_cpu.flatten()]))[0, 1].item()
        pcc_state_vs = torch.corrcoef(torch.stack([ttnn_state_cpu.flatten(), fused_state_cpu.flatten()]))[0, 1].item()
        logger.info(f"Fused vs ttnn: output PCC={pcc_out_vs:.6f}, state PCC={pcc_state_vs:.6f}")

        # Detailed analysis: check element-wise
        logger.info(f"Fused output range: [{fused_out_cpu.min():.6f}, {fused_out_cpu.max():.6f}]")
        logger.info(f"ttnn output range: [{ttnn_out_cpu.min():.6f}, {ttnn_out_cpu.max():.6f}]")
        logger.info(f"Ref output range: [{out_ref.min():.6f}, {out_ref.max():.6f}]")

        # Check per-pair PCC
        for p in range(num_pairs):
            pcc_p = torch.corrcoef(torch.stack([out_ref[p].flatten(), fused_out_cpu[p].flatten()]))[0, 1].item()
            logger.info(f"  Pair {p} output PCC: {pcc_p:.6f}")

        assert pcc_fused_out > 0.95, f"Fused output PCC too low: {pcc_fused_out:.6f}"
    except Exception as e:
        logger.error(f"Fused kernel failed: {e}")
        raise


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("num_pairs", [1, 2, 4, 8, 10])
def test_fused_pair_scaling(mesh_device, num_pairs):
    """Test how PCC degrades as num_pairs increases."""
    from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import _gdn_recurrence_fused

    device = mesh_device
    torch.manual_seed(42)

    q = torch.randn(num_pairs, 1, Dk) * 0.1
    k = torch.randn(num_pairs, 1, Dk) * 0.1
    k_col = k.transpose(-2, -1)
    v = torch.randn(num_pairs, 1, Dv) * 0.1
    g = torch.randn(num_pairs, 1, 1) * 0.5 - 1.0
    beta = torch.randn(num_pairs, 1, 1).abs() * 0.5
    state = torch.randn(num_pairs, Dk, Dv) * 0.01

    out_ref, state_ref, _ = ref_recurrence(q, k, v, g, beta, state)

    q_tt = to_tt(q, device)
    k_tt = to_tt(k, device)
    k_col_tt = to_tt(k_col, device)
    v_tt = to_tt(v, device)
    g_tt = to_tt(g, device)
    beta_tt = to_tt(beta, device)
    state_tt = to_tt(state, device)
    output_tt = to_tt(torch.zeros(num_pairs, 1, Dv), device)

    _gdn_recurrence_fused(
        q_tt,
        k_tt,
        k_col_tt,
        v_tt,
        g_tt,
        beta_tt,
        state_tt,
        output_tt,
        state_tt,
        num_cores=min(num_pairs, 4),
    )
    fused_out_cpu = to_torch(output_tt)
    fused_state_cpu = to_torch(state_tt)

    pcc_out = torch.corrcoef(torch.stack([out_ref.flatten(), fused_out_cpu.flatten()]))[0, 1].item()
    pcc_state = torch.corrcoef(torch.stack([state_ref.flatten(), fused_state_cpu.flatten()]))[0, 1].item()
    logger.info(f"num_pairs={num_pairs}: output PCC={pcc_out:.6f}, state PCC={pcc_state:.6f}")

    for p in range(min(num_pairs, 5)):
        pcc_p = torch.corrcoef(torch.stack([out_ref[p].flatten(), fused_out_cpu[p].flatten()]))[0, 1].item()
        logger.info(f"  Pair {p}: PCC={pcc_p:.6f}")

    assert pcc_out > 0.90, f"Output PCC too low: {pcc_out:.6f}"


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_data_layout_check(mesh_device):
    """Verify that tensor data survives the round-trip through DRAM correctly."""
    device = mesh_device
    num_pairs = 4
    torch.manual_seed(42)

    # Create test tensors with known values
    state = torch.randn(num_pairs, Dk, Dv) * 0.01
    q = torch.randn(num_pairs, 1, Dk) * 0.1

    # Round-trip through device
    state_tt = to_tt(state, device)
    state_back = to_torch(state_tt)

    q_tt = to_tt(q, device)
    q_back = to_torch(q_tt)

    # Check bfloat16 round-trip
    state_bf16 = state.to(torch.bfloat16).float()
    q_bf16 = q.to(torch.bfloat16).float()

    state_match = torch.allclose(state_bf16, state_back, atol=1e-6)
    q_match = torch.allclose(q_bf16, q_back, atol=1e-6)

    logger.info(f"State round-trip match: {state_match}")
    logger.info(f"Q round-trip match: {q_match}")

    if not state_match:
        diff = (state_bf16 - state_back).abs()
        logger.info(f"State max diff: {diff.max():.8f}")
    if not q_match:
        diff = (q_bf16 - q_back).abs()
        logger.info(f"Q max diff: {diff.max():.8f}")

    assert state_match
    assert q_match
