# Quick debug: test fused kernel with different num_cores configurations
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import _gdn_recurrence_fused

Dk = Dv = 128


def ref_recurrence(q, k, v, g, beta, state):
    g_exp = torch.exp(g)
    state_b = state * g_exp
    kv_mem = k @ state_b
    delta = v - kv_mem
    delta_s = beta * delta
    k_col = k.transpose(-2, -1)
    outer = k_col @ delta_s
    state_new = state_b + outer
    output = q @ state_new
    return output, state_new


def to_tt(t, device):
    return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def to_torch(t):
    return ttnn.to_torch(t).float()


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize(
    "num_pairs,num_cores",
    [
        (1, 1),
        (4, 1),  # 4 pairs on 1 core (loop 4x)
        (4, 4),  # 4 pairs on 4 cores (loop 1x each)
        (4, 2),  # 4 pairs on 2 cores (loop 2x each)
    ],
)
def test_core_vs_loop(mesh_device, num_pairs, num_cores):
    """Isolate: is the bug in multi-pair looping or core assignment?"""
    device = mesh_device
    torch.manual_seed(42)

    q = torch.randn(num_pairs, 1, Dk) * 0.1
    k = torch.randn(num_pairs, 1, Dk) * 0.1
    v = torch.randn(num_pairs, 1, Dv) * 0.1
    g = torch.randn(num_pairs, 1, 1) * 0.5 - 1.0
    beta = torch.randn(num_pairs, 1, 1).abs() * 0.5
    state = torch.randn(num_pairs, Dk, Dv) * 0.01

    out_ref, state_ref = ref_recurrence(q, k, v, g, beta, state)

    q_tt = to_tt(q, device)
    k_tt = to_tt(k, device)
    k_col_tt = to_tt(k.transpose(-2, -1), device)
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
        num_cores=num_cores,
    )
    fused_out = to_torch(output_tt)
    fused_state = to_torch(state_tt)

    pcc_out = torch.corrcoef(torch.stack([out_ref.flatten(), fused_out.flatten()]))[0, 1].item()
    pcc_state = torch.corrcoef(torch.stack([state_ref.flatten(), fused_state.flatten()]))[0, 1].item()
    logger.info(f"num_pairs={num_pairs}, num_cores={num_cores}: output PCC={pcc_out:.6f}, state PCC={pcc_state:.6f}")

    for p in range(num_pairs):
        pcc_p = torch.corrcoef(torch.stack([out_ref[p].flatten(), fused_out[p].flatten()]))[0, 1].item()
        state_pcc_p = torch.corrcoef(torch.stack([state_ref[p].flatten(), fused_state[p].flatten()]))[0, 1].item()
        logger.info(f"  Pair {p}: output PCC={pcc_p:.6f}, state PCC={state_pcc_p:.6f}")
