# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Per-op unit tests for the TTNN-native decode-path ops in
ttnn_decode_ops.py. Each test compares one TTNN op to its torch reference.

Bisects within `_use_ttnn_decode_path` to find which op makes the model output
gibberish when KERNEL=1. The kernel itself is already proven correct via
test_gdn_kernel_unit.py.

Run with:
  export TTNN_GDN_KERNEL=1
  export MESH_DEVICE=QB2
  pytest models/experimental/tt_symbiote/tests/test_gdn_ttnn_ops_unit.py -s
"""

import os

import pytest
import torch
import torch.nn.functional as F

import ttnn


MESH_DEVICE_MAP = {
    "T3K": (1, 8),
    "QB2": (1, 4),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
}


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / denom)


def _replicated(t, mesh_device):
    return ttnn.from_torch(
        t.to(torch.bfloat16).contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None,
    )


def _to_torch_replicated(t_ttnn, mesh_device):
    return ttnn.to_torch(t_ttnn, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]


# ============================================================
# Test 1: g/beta computation
# ============================================================
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_ttnn_compute_g_beta_matches_torch(mesh_device):
    """g = -exp(A_log) * softplus(a + dt_bias);  beta = sigmoid(b)."""
    from models.experimental.tt_symbiote.modules.ttnn_decode_ops import (
        ttnn_compute_g_beta,
    )

    torch.manual_seed(0)
    H = 32  # num_v_heads

    # Inputs: a, b as [1, 1, H]; A_log as [H]; dt_bias as [H].
    a = torch.randn(1, 1, H, dtype=torch.bfloat16)
    b = torch.randn(1, 1, H, dtype=torch.bfloat16)
    A_log = torch.empty(H).uniform_(0, 4).log()  # log of [0, 4]-uniform sample
    dt_bias = torch.ones(H, dtype=torch.bfloat16)

    # Torch reference (qwen_attention.py / modeling_qwen3_5_moe.py)
    ref_beta = torch.sigmoid(b.float())
    ref_g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())

    # TTNN
    A_log_neg_exp = -A_log.exp()  # precomputed, like the model does
    a_tt = _replicated(a, mesh_device)
    b_tt = _replicated(b, mesh_device)
    A_neg_exp_tt = _replicated(A_log_neg_exp, mesh_device)
    dt_bias_tt = _replicated(dt_bias, mesh_device)

    g_tt, beta_tt = ttnn_compute_g_beta(a_tt, b_tt, A_neg_exp_tt, dt_bias_tt)

    g_out = _to_torch_replicated(g_tt, mesh_device).float()
    beta_out = _to_torch_replicated(beta_tt, mesh_device).float()

    g_pcc = _pcc(g_out, ref_g)
    beta_pcc = _pcc(beta_out, ref_beta)
    print(f"\nG    pcc={g_pcc:.6f}  max_abs_err={(g_out-ref_g).abs().max():.4e}")
    print(f"BETA pcc={beta_pcc:.6f}  max_abs_err={(beta_out-ref_beta).abs().max():.4e}")
    print(f"ref_g[0,0,:6]    = {ref_g[0,0,:6].tolist()}")
    print(f"kernel_g[0,0,:6] = {g_out[0,0,:6].tolist()}")
    print(f"ref_beta[0,0,:6]    = {ref_beta[0,0,:6].tolist()}")
    print(f"kernel_beta[0,0,:6] = {beta_out[0,0,:6].tolist()}")

    assert g_pcc > 0.99, f"g pcc too low: {g_pcc}"
    assert beta_pcc > 0.99, f"beta pcc too low: {beta_pcc}"


# ============================================================
# Test 2: gated RMS norm
# ============================================================
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_ttnn_gated_rms_norm_matches_torch(mesh_device):
    """y = weight * rms_norm(x) * silu(gate)."""
    from models.experimental.tt_symbiote.modules.ttnn_decode_ops import (
        ttnn_gated_rms_norm,
    )

    torch.manual_seed(0)
    H = 32
    D = 128
    eps = 1e-6

    # Shapes match how the qwen_attention.py path calls this: [1, H, D].
    x = torch.randn(1, H, D, dtype=torch.bfloat16)
    gate = torch.randn(1, H, D, dtype=torch.bfloat16)
    weight = torch.randn(D, dtype=torch.bfloat16) * 0.1 + 1.0  # near-1 weights

    # Torch reference (Qwen3_5MoeRMSNormGated.forward).
    h = x.float()
    var = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(var + eps)
    h = weight.float() * h
    ref = (h * F.silu(gate.float())).to(torch.bfloat16)

    # TTNN
    x_tt = _replicated(x, mesh_device)
    gate_tt = _replicated(gate, mesh_device)
    weight_tt = _replicated(weight.unsqueeze(0), mesh_device)  # [1, D]

    y_tt = ttnn_gated_rms_norm(x_tt, gate_tt, weight_tt, eps=eps)
    y_out = _to_torch_replicated(y_tt, mesh_device).reshape(1, H, D)

    pcc = _pcc(y_out, ref)
    print(f"\nGATED_RMS_NORM  pcc={pcc:.6f}  max_abs_err={(y_out.float()-ref.float()).abs().max():.4e}")
    print(f"ref[0,0,:6]    = {ref[0,0,:6].tolist()}")
    print(f"kernel[0,0,:6] = {y_out[0,0,:6].tolist()}")

    assert pcc > 0.99, f"gated rms norm pcc too low: {pcc}"


# ============================================================
# Test 3: causal conv1d update (single decode step)
# ============================================================
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
def test_ttnn_causal_conv1d_update_matches_torch(mesh_device):
    """One decode step: TTNN shift-register conv1d vs torch_causal_conv1d_update.

    Mirrors what qwen_attention.py does at line 1944-1979.
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_causal_conv1d_update
    from models.experimental.tt_symbiote.modules.ttnn_decode_ops import (
        init_conv_slots_from_torch_state,
        ttnn_causal_conv1d_update_step,
        upload_conv1d_weights,
        upload_replicated,
    )

    torch.manual_seed(0)
    D = 1024  # conv_dim = key_dim*2 + value_dim — must be tile-aligned
    K = 4  # conv_kernel_size
    NUM_STEPS = 4

    # Pre-generate per-step inputs.
    tokens = [torch.randn(1, D, 1, dtype=torch.bfloat16) for _ in range(NUM_STEPS)]
    init_conv_state = torch.randn(1, D, K, dtype=torch.bfloat16) * 0.1
    weight = torch.randn(D, K, dtype=torch.bfloat16) * 0.1  # [dim, K] (depthwise)
    bias = torch.randn(D, dtype=torch.bfloat16) * 0.05

    # --- Torch reference, multi-step.
    ref_state = init_conv_state.clone()
    ref_outs = []
    for x in tokens:
        # torch_causal_conv1d_update mutates conv_state in place; clone in/out.
        y = torch_causal_conv1d_update(
            hidden_states=x,
            conv_state=ref_state,
            weight=weight,
            bias=bias,
            activation="silu",
        )
        ref_outs.append(y.clone())

    # --- TTNN, multi-step (shift register accumulates state).
    weights_per_k = upload_conv1d_weights(weight.unsqueeze(1), mesh_device)  # [D, 1, K]
    bias_tt = upload_replicated(bias.unsqueeze(0), mesh_device)  # [1, D]
    slots = init_conv_slots_from_torch_state(init_conv_state, mesh_device)

    ttnn_outs = []
    for x in tokens:
        # qwen_attention.py reshapes mixed_qkv_ttnn from [1, 1, conv_dim] to [1, conv_dim].
        x_2d = x.squeeze(-1)  # [1, D]
        x_tt = _replicated(x_2d, mesh_device)
        y_tt = ttnn_causal_conv1d_update_step(x_tt, slots, weights_per_k, bias_tt)
        y_torch = _to_torch_replicated(y_tt, mesh_device)  # [1, D]
        ttnn_outs.append(y_torch.unsqueeze(-1))  # [1, D, 1] to match ref

    # Per-step comparison.
    print(f"\nCausal conv1d update — per-step PCC:")
    print(f"{'step':>4}  {'pcc':>10}  {'max_abs_err':>12}")
    for i, (ko, ro) in enumerate(zip(ttnn_outs, ref_outs)):
        pcc = _pcc(ko, ro)
        mae = (ko.float() - ro.float()).abs().max().item()
        print(f"{i:>4}  {pcc:>10.6f}  {mae:>12.4e}")

    print(f"\nstep0 ref[0,:6,0]    = {ref_outs[0][0,:6,0].tolist()}")
    print(f"step0 ttnn[0,:6,0]   = {ttnn_outs[0][0,:6,0].tolist()}")
    print(f"\nstep3 ref[0,:6,0]    = {ref_outs[-1][0,:6,0].tolist()}")
    print(f"step3 ttnn[0,:6,0]   = {ttnn_outs[-1][0,:6,0].tolist()}")

    for i, (ko, ro) in enumerate(zip(ttnn_outs, ref_outs)):
        pcc = _pcc(ko, ro)
        assert pcc > 0.99, f"step {i}: conv1d pcc too low: {pcc}"
