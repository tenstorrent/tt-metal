# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Full fused GDN kernel (Phase A) correctness test — batched reader variant.

Passes conv_out in batched [1, B, qkv_dim_tp] format directly to the kernel.
The reader extracts Q/K/V per-pair via sub-tile row reads.

Compares the kernel output (recurrence only) against a PyTorch reference.
"""

import math

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_full_fused_inplace


def _l2_norm(x, eps=1e-6):
    """L2 normalize along last dim."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def ref_full_fused_step(
    conv_out,
    a,
    b,
    z,
    neg_exp_A,
    dt_bias,
    norm_w,
    scale,
    state,
    Nk_TP,
    Nv_TP,
    Dk,
    Dv,
    key_dim_tp,
):
    """Full fused GDN reference in PyTorch (float32).

    Args:
        conv_out: [1, B, qkv_dim_tp]
        a: [1, B, Nv_TP]
        b: [1, B, Nv_TP]
        z: [1, B, z_dim_tp] where z_dim_tp = Nv_TP * Dv
        neg_exp_A: [1, 1, Nv_TP]
        dt_bias: [1, 1, Nv_TP]
        norm_w: [1, 1, Dv]
        scale: float
        state: [num_pairs, Dk, Dv]
        Nk_TP, Nv_TP, Dk, Dv, key_dim_tp: architecture constants

    Returns:
        output: [num_pairs, 1, Dv]
        new_state: [num_pairs, Dk, Dv]
    """
    B = conv_out.shape[1]
    num_pairs = B * Nv_TP
    repeat_factor = Nv_TP // Nk_TP

    # Split Q/K/V from conv_out
    q_raw = conv_out[:, :, :key_dim_tp].reshape(B, Nk_TP, Dk)
    k_raw = conv_out[:, :, key_dim_tp : 2 * key_dim_tp].reshape(B, Nk_TP, Dk)
    v_raw = conv_out[:, :, 2 * key_dim_tp :].reshape(B, Nv_TP, Dv)

    # L2 norm Q and K
    q_normed = _l2_norm(q_raw) * scale
    k_normed = _l2_norm(k_raw)

    # GQA repeat
    q_exp = q_normed.repeat_interleave(repeat_factor, dim=1)  # [B, Nv_TP, Dk]
    k_exp = k_normed.repeat_interleave(repeat_factor, dim=1)  # [B, Nv_TP, Dk]

    # Reshape to [num_pairs, ...]
    q = q_exp.reshape(num_pairs, 1, Dk)
    k = k_exp.reshape(num_pairs, 1, Dk)
    v = v_raw.reshape(num_pairs, 1, Dv)

    # Gates
    beta = torch.sigmoid(b.reshape(num_pairs, 1, 1))
    softplus_val = torch.nn.functional.softplus(
        a.reshape(num_pairs, 1, 1) + dt_bias.expand(1, B, Nv_TP).reshape(num_pairs, 1, 1)
    )
    g = neg_exp_A.expand(1, B, Nv_TP).reshape(num_pairs, 1, 1) * softplus_val

    # Recurrence
    g_exp = g.squeeze(-1).exp()  # [num_pairs, 1]
    new_state = state * g_exp.unsqueeze(-1)  # decay
    kv_mem = torch.bmm(k, new_state)  # [num_pairs, 1, Dv]
    delta = beta * (v - kv_mem)
    new_state = new_state + torch.bmm(k.transpose(-2, -1), delta)
    rec_out = torch.bmm(q, new_state)  # [num_pairs, 1, Dv]

    output = rec_out  # [num_pairs, 1, Dv]

    return output, new_state


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("B", [1])
def test_gdn_full_fused_correctness(mesh_device, reset_seeds, ensure_gc, B):
    """Test full fused GDN kernel with batched conv_out reader."""
    device = mesh_device

    # Architecture constants (Qwen3.5-27B TP=4)
    Nk_TP = 4
    Nv_TP = 12
    Dk = 128
    Dv = 128
    repeat_factor = Nv_TP // Nk_TP
    key_dim_tp = Nk_TP * Dk  # 512
    qkv_dim_tp = 2 * key_dim_tp + Nv_TP * Dv  # 2*512 + 12*128 = 2560
    z_dim_tp = Nv_TP * Dv  # 1536
    value_dim_tp = Nv_TP * Dv  # 1536
    num_pairs = B * Nv_TP
    scale = Dk**-0.5

    logger.info(f"Testing full fused kernel (batched reader): B={B}, num_pairs={num_pairs}")

    torch.manual_seed(42)
    conv_out_ref = torch.randn(1, B, qkv_dim_tp, dtype=torch.float32) * 0.1
    a_ref = torch.randn(1, B, Nv_TP, dtype=torch.float32) * 0.5
    b_ref = torch.randn(1, B, Nv_TP, dtype=torch.float32) * 0.5
    z_ref = torch.randn(1, B, z_dim_tp, dtype=torch.float32) * 0.1
    neg_exp_A_ref = -torch.exp(torch.randn(1, 1, Nv_TP, dtype=torch.float32) * 0.5)
    dt_bias_ref = torch.randn(1, 1, Nv_TP, dtype=torch.float32) * 0.1
    norm_w_ref = torch.ones(1, 1, Dv, dtype=torch.float32) + torch.randn(1, 1, Dv, dtype=torch.float32) * 0.01
    state_ref = torch.randn(num_pairs, Dk, Dv, dtype=torch.float32) * 0.01

    # PyTorch reference
    out_ref, state_new_ref = ref_full_fused_step(
        conv_out_ref,
        a_ref,
        b_ref,
        z_ref,
        neg_exp_A_ref,
        dt_bias_ref,
        norm_w_ref,
        scale,
        state_ref.clone(),
        Nk_TP,
        Nv_TP,
        Dk,
        Dv,
        key_dim_tp,
    )
    logger.info(f"Reference output: shape={out_ref.shape}, range=[{out_ref.min():.4f}, {out_ref.max():.4f}]")

    # Convert to device tensors
    def to_tt(t):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    # conv_out stays in batched form [1, B, qkv_dim_tp]
    conv_out_tt = to_tt(conv_out_ref)

    # a, b: batched [1, B, Nv_TP] — reader extracts per-pair scalars
    a_tt = to_tt(a_ref)
    b_tt = to_tt(b_ref)

    # neg_exp_A, dt_bias: constant [1, 1, Nv_TP] — reader extracts per-pair scalars
    neg_exp_A_tt = to_tt(neg_exp_A_ref)
    dt_bias_tt = to_tt(dt_bias_ref)

    norm_w_tt = to_tt(norm_w_ref)
    state_tt = to_tt(state_ref)

    # Scalar constant tiles
    scale_tt = to_tt(torch.full((1, 1, 1), scale, dtype=torch.float32))
    rms_scale_tt = to_tt(torch.full((1, 1, 1), math.sqrt(Dv), dtype=torch.float32))
    rms_eps_tt = to_tt(torch.full((1, 1, 1), Dv * 1e-6, dtype=torch.float32))

    # Pre-allocated output [num_pairs, 1, Dv]
    output_tt = to_tt(torch.zeros(num_pairs, 1, Dv, dtype=torch.float32))

    # DRAM coherence: flush input tensors so kernel NOC reads see them
    for t in [conv_out_tt, a_tt, b_tt, neg_exp_A_tt, dt_bias_tt, state_tt]:
        _tmp = ttnn.add(t, 0.0)
        ttnn.deallocate(_tmp)

    # Run fused kernel
    gdn_full_fused_inplace(
        conv_out_tt,
        a_tt,
        b_tt,
        neg_exp_A_tt,
        dt_bias_tt,
        norm_w_tt,
        scale_tt,
        rms_scale_tt,
        rms_eps_tt,
        state_tt,
        output_tt,
        num_pairs=num_pairs,
        num_cores=min(40, num_pairs),
        Nv_TP=Nv_TP,
        Nk_TP=Nk_TP,
        repeat_factor=repeat_factor,
        key_dim_tp=key_dim_tp,
    )

    # Force DRAM coherence
    out_flushed = ttnn.add(output_tt, 0.0)
    state_flushed = ttnn.add(state_tt, 0.0)

    # Get results
    out_tt_cpu = ttnn.to_torch(out_flushed).float()
    state_tt_cpu = ttnn.to_torch(state_flushed).float()
    ttnn.deallocate(out_flushed)
    ttnn.deallocate(state_flushed)

    # Slice TT output to logical shape [num_pairs, 1, Dv] (remove tile padding)
    out_tt_cpu = out_tt_cpu[:num_pairs, :1, :Dv]
    logger.info(f"Kernel output: shape={out_tt_cpu.shape}, range=[{out_tt_cpu.min():.4f}, {out_tt_cpu.max():.4f}]")

    # Per-head PCC
    for p in range(num_pairs):
        ref_h = out_ref[p, 0, :]
        tt_h = out_tt_cpu[p, 0, :]
        if ref_h.std() > 0 and tt_h.std() > 0:
            h_pcc = torch.corrcoef(torch.stack([ref_h, tt_h]))[0, 1].item()
            logger.info(
                f"  Pair {p}: PCC={h_pcc:.6f}, ref=[{ref_h.min():.4f},{ref_h.max():.4f}], tt=[{tt_h.min():.4f},{tt_h.max():.4f}]"
            )
        else:
            logger.info(
                f"  Pair {p}: ZERO OUTPUT, ref=[{ref_h.min():.4f},{ref_h.max():.4f}], tt=[{tt_h.min():.4f},{tt_h.max():.4f}]"
            )

    # Compare output
    out_ref_flat = out_ref.flatten()
    out_tt_flat = out_tt_cpu.flatten()
    pcc = torch.corrcoef(torch.stack([out_ref_flat, out_tt_flat]))[0, 1].item()
    out_max_diff = (out_ref - out_tt_cpu).abs().max().item()

    logger.info(f"Output: PCC={pcc:.6f}, max_diff={out_max_diff:.6f}")

    # Compare state (slice to logical shape)
    state_tt_cpu = state_tt_cpu[:num_pairs, :Dk, :Dv]
    state_ref_flat = state_new_ref.flatten()
    state_tt_flat = state_tt_cpu.flatten()
    state_pcc = torch.corrcoef(torch.stack([state_ref_flat, state_tt_flat]))[0, 1].item()
    state_max_diff = (state_new_ref - state_tt_cpu).abs().max().item()

    logger.info(f"State: PCC={state_pcc:.6f}, max_diff={state_max_diff:.6f}")

    assert pcc > 0.999, f"Output PCC too low: {pcc:.6f}"
    assert state_pcc > 0.999, f"State PCC too low: {state_pcc:.6f}"

    logger.info(f"PASSED: full fused kernel matches reference (output PCC={pcc:.4f}, state PCC={state_pcc:.4f})")
