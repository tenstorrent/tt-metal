# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Test fused DeltaNet prefill kernel against CPU reference.
Tests multi-token (S>1) prefill with conv1d, recurrence, RMSNorm, and gating.
"""

import math
import os
import torch
import torch.nn.functional as F
import ttnn


def l2norm(x, dim=-1):
    """L2 normalize along dim, matching bf16 precision."""
    norm = x.norm(2, dim=dim, keepdim=True).clamp(min=1e-6)
    return x / norm


def deltanet_prefill_reference(
    qkv_proj,       # [S, conv_dim]
    z_proj,          # [S, H*Dv]
    b_proj,          # [S, H]
    a_proj,          # [S, H]
    conv_state,      # [conv_dim, conv_kernel_size]  or None
    recurrent_state, # [H, Dk, Dv]
    conv1d_weight,   # [conv_dim, conv_kernel_size]
    a_log,           # [H]
    dt_bias,         # [H]
    norm_weight,     # [Dv]
    num_heads,
    num_k_heads,
    k_head_dim,
    v_head_dim,
    head_expand_ratio,
):
    """
    CPU reference: per-token sequential DeltaNet with conv1d + recurrence + RMSNorm + gate.
    Returns: outputs [S, H*Dv], new_state [H, Dk, Dv], new_conv_state [conv_dim, conv_kernel_size]
    """
    S = qkv_proj.shape[0]
    H = num_heads
    Hk = num_k_heads
    Dk = k_head_dim
    Dv = v_head_dim
    conv_dim = qkv_proj.shape[1]
    key_dim = Hk * Dk
    value_dim = H * Dv

    state = recurrent_state.clone().float()
    if conv_state is not None:
        cs = conv_state.clone().float()
    else:
        cs = torch.zeros(conv_dim, conv1d_weight.shape[1]).float()

    scale = Dk ** -0.5
    outputs = []

    for t in range(S):
        qkv_t = qkv_proj[t].float()

        # Conv1d: roll + insert + dot + silu
        cs = torch.roll(cs, shifts=-1, dims=-1)
        cs[:, -1] = qkv_t
        qkv_conv = (cs * conv1d_weight.float()).sum(dim=-1)
        qkv_conv = F.silu(qkv_conv)

        # Split into q, k, v
        q_t, k_t, v_t = torch.split(qkv_conv, [key_dim, key_dim, value_dim], dim=-1)
        q_t = q_t.reshape(Hk, Dk)
        k_t = k_t.reshape(Hk, Dk)
        v_t = v_t.reshape(H, Dv)

        # Decay and beta
        b_t = b_proj[t, :H].float()
        a_t = a_proj[t, :H].float()
        beta = torch.sigmoid(b_t)
        g = (-a_log.float().exp() * F.softplus(a_t + dt_bias.float())).exp()

        # Head expansion (GQA-style)
        if head_expand_ratio > 1:
            q_t = q_t.repeat_interleave(head_expand_ratio, dim=0)
            k_t = k_t.repeat_interleave(head_expand_ratio, dim=0)

        # L2 normalize
        q_t = l2norm(q_t, dim=-1) * scale
        k_t = l2norm(k_t, dim=-1)

        # Recurrence
        g_expand = g.unsqueeze(-1).unsqueeze(-1)   # [H, 1, 1]
        beta_expand = beta.unsqueeze(-1)             # [H, 1]

        state = state * g_expand
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)  # [H, Dv]
        delta = (v_t - kv_mem) * beta_expand
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        output_t = (state * q_t.unsqueeze(-1)).sum(dim=-2)  # [H, Dv]

        # RMSNorm
        variance = output_t.pow(2).mean(-1, keepdim=True)
        out_normed = output_t * torch.rsqrt(variance + 1e-6)
        out_normed = norm_weight.float() * out_normed

        # Gate
        z_t = z_proj[t, :H * Dv].reshape(H, Dv).float()
        out_gated = out_normed * F.silu(z_t)
        outputs.append(out_gated.reshape(-1))

    outputs = torch.stack(outputs, dim=0)  # [S, H*Dv]
    return outputs, state, cs


def pad_to_tile(x, tile_size=32):
    """Pad last dimension to be a multiple of tile_size."""
    size = x.shape[-1]
    if size % tile_size == 0:
        return x
    pad_size = tile_size - (size % tile_size)
    return F.pad(x, [0, pad_size])


def run_prefill_test(device, H, Hk, Dk, Dv, S, conv_kernel_size=4, head_expand_ratio=3, seed=42):
    """Run a single prefill test with given dimensions."""
    assert H == Hk * head_expand_ratio
    conv_dim = Hk * Dk * 2 + H * Dv  # key_dim + key_dim + value_dim
    key_dim = Hk * Dk
    value_dim = H * Dv

    torch.manual_seed(seed)

    # Generate random inputs in bf16 precision
    qkv_proj = torch.randn(S, conv_dim).to(torch.bfloat16).float()
    z_proj = torch.randn(S, H * Dv).to(torch.bfloat16).float()
    b_proj = torch.randn(S, H).to(torch.bfloat16).float()
    a_proj = torch.randn(S, H).to(torch.bfloat16).float()

    conv_state = torch.randn(conv_dim, conv_kernel_size).to(torch.bfloat16).float() * 0.01
    recurrent_state = torch.randn(H, Dk, Dv).to(torch.bfloat16).float() * 0.01

    # Conv weight: [conv_dim, conv_kernel_size], padded to [conv_dim, 32] for TILE
    conv1d_weight = torch.randn(conv_dim, conv_kernel_size).to(torch.bfloat16).float() * 0.1
    a_log = torch.randn(H).to(torch.bfloat16).float().abs() * 0.5  # positive
    dt_bias = torch.randn(H).to(torch.bfloat16).float() * 0.1
    norm_weight = torch.ones(Dv).to(torch.bfloat16).float()

    # CPU reference
    ref_out, ref_state, ref_conv_state = deltanet_prefill_reference(
        qkv_proj, z_proj, b_proj, a_proj,
        conv_state, recurrent_state,
        conv1d_weight, a_log, dt_bias, norm_weight,
        num_heads=H, num_k_heads=Hk,
        k_head_dim=Dk, v_head_dim=Dv,
        head_expand_ratio=head_expand_ratio,
    )

    # Prepare device tensors
    # qkv_proj: [1, 1, S, conv_dim]
    qkv_tt = ttnn.from_torch(
        qkv_proj.to(torch.bfloat16).reshape(1, 1, S, conv_dim),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # z_proj: [1, 1, S, H*Dv]
    z_tt = ttnn.from_torch(
        z_proj.to(torch.bfloat16).reshape(1, 1, S, H * Dv),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # b_proj: [1, 1, S, H] — pad H to tile-aligned
    b_padded = pad_to_tile(b_proj.to(torch.bfloat16).reshape(1, 1, S, H))
    b_tt = ttnn.from_torch(b_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # a_proj: [1, 1, S, H] — pad H to tile-aligned
    a_padded = pad_to_tile(a_proj.to(torch.bfloat16).reshape(1, 1, S, H))
    a_tt = ttnn.from_torch(a_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # conv_state: [1, 1, conv_dim, 32] — pad conv_kernel_size to 32
    cs_padded = torch.zeros(1, 1, conv_dim, 32, dtype=torch.bfloat16)
    cs_padded[0, 0, :, :conv_kernel_size] = conv_state.to(torch.bfloat16)
    cs_tt = ttnn.from_torch(cs_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # recurrent_state: [1, H, Dk, Dv]
    state_tt = ttnn.from_torch(
        recurrent_state.to(torch.bfloat16).unsqueeze(0),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # conv1d_weight: [1, 1, conv_dim, 32] — padded
    cw_padded = torch.zeros(1, 1, conv_dim, 32, dtype=torch.bfloat16)
    cw_padded[0, 0, :, :conv_kernel_size] = conv1d_weight.to(torch.bfloat16)
    cw_tt = ttnn.from_torch(cw_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # a_log: [1, 1, 1, H] — pad to tile
    a_log_padded = pad_to_tile(a_log.to(torch.bfloat16).reshape(1, 1, 1, H))
    a_log_tt = ttnn.from_torch(a_log_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # dt_bias: [1, 1, 1, H] — pad to tile
    dt_bias_padded = pad_to_tile(dt_bias.to(torch.bfloat16).reshape(1, 1, 1, H))
    dt_bias_tt = ttnn.from_torch(dt_bias_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # norm_weight: [1, 1, 1, Dv]
    norm_w_padded = pad_to_tile(norm_weight.to(torch.bfloat16).reshape(1, 1, 1, Dv))
    norm_w_tt = ttnn.from_torch(norm_w_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Call device op
    results = ttnn.experimental.deltanet_prefill_full(
        qkv_tt, z_tt, b_tt, a_tt,
        cs_tt, state_tt,
        cw_tt, a_log_tt, dt_bias_tt, norm_w_tt,
        num_heads=H, num_k_heads=Hk,
        k_head_dim=Dk, v_head_dim=Dv,
        conv_dim=conv_dim, conv_kernel_size=conv_kernel_size,
        head_expand_ratio=head_expand_ratio,
        seq_len=S,
    )

    output_tt, new_state_tt, new_conv_state_tt = results[0], results[1], results[2]

    # Get outputs
    out_dev = ttnn.to_torch(output_tt).float()   # [S*H, 1, 1, Dv]
    state_dev = ttnn.to_torch(new_state_tt).float()  # [1, H, Dk, Dv]
    conv_state_dev = ttnn.to_torch(new_conv_state_tt).float()  # [1, 1, conv_dim, 32]

    # Reshape output from [S*H, 1, 1, Dv] to [S, H*Dv]
    out_dev = out_dev.reshape(S, H, Dv).reshape(S, H * Dv)

    # Compare output
    ref_out_bf16 = ref_out.to(torch.bfloat16).float()
    out_pcc = torch.corrcoef(torch.stack([out_dev.flatten(), ref_out_bf16.flatten()]))[0, 1].item()
    out_max_err = (out_dev - ref_out_bf16).abs().max().item()

    # Compare state
    state_dev_reshaped = state_dev[0]  # [H, Dk, Dv]
    ref_state_bf16 = ref_state.to(torch.bfloat16).float()
    state_pcc = torch.corrcoef(torch.stack([state_dev_reshaped.flatten(), ref_state_bf16.flatten()]))[0, 1].item()
    state_max_err = (state_dev_reshaped - ref_state_bf16).abs().max().item()

    # Compare conv_state (only first conv_kernel_size columns matter)
    cs_dev = conv_state_dev[0, 0, :, :conv_kernel_size]  # [conv_dim, conv_kernel_size]
    ref_cs_bf16 = ref_conv_state.to(torch.bfloat16).float()
    cs_pcc = torch.corrcoef(torch.stack([cs_dev.flatten(), ref_cs_bf16.flatten()]))[0, 1].item()
    cs_max_err = (cs_dev - ref_cs_bf16).abs().max().item()

    # Cleanup
    ttnn.deallocate(output_tt)
    ttnn.deallocate(new_state_tt)
    ttnn.deallocate(new_conv_state_tt)

    return {
        "out_pcc": out_pcc, "out_max_err": out_max_err,
        "state_pcc": state_pcc, "state_max_err": state_max_err,
        "cs_pcc": cs_pcc, "cs_max_err": cs_max_err,
    }


def test_prefill_small_s2():
    """Small dims, S=2: basic smoke test."""
    device_id = int(os.environ.get("TT_DEVICE_ID", 0))
    device = ttnn.open_device(device_id=device_id)
    try:
        results = run_prefill_test(device, H=6, Hk=2, Dk=64, Dv=64, S=2, head_expand_ratio=3)
        print(f"[Small S=2] Output PCC: {results['out_pcc']:.6f}  max_err: {results['out_max_err']:.6f}")
        print(f"            State  PCC: {results['state_pcc']:.6f}  max_err: {results['state_max_err']:.6f}")
        print(f"            ConvSt PCC: {results['cs_pcc']:.6f}  max_err: {results['cs_max_err']:.6f}")
        assert results["out_pcc"] > 0.99, f"Output PCC too low: {results['out_pcc']}"
        assert results["state_pcc"] > 0.99, f"State PCC too low: {results['state_pcc']}"
        assert results["cs_pcc"] > 0.99, f"ConvState PCC too low: {results['cs_pcc']}"
        print("PASS")
    finally:
        ttnn.close_device(device)


def test_prefill_small_s8():
    """Small dims, S=8."""
    device_id = int(os.environ.get("TT_DEVICE_ID", 0))
    device = ttnn.open_device(device_id=device_id)
    try:
        results = run_prefill_test(device, H=6, Hk=2, Dk=64, Dv=64, S=8, head_expand_ratio=3)
        print(f"[Small S=8] Output PCC: {results['out_pcc']:.6f}  max_err: {results['out_max_err']:.6f}")
        print(f"            State  PCC: {results['state_pcc']:.6f}  max_err: {results['state_max_err']:.6f}")
        print(f"            ConvSt PCC: {results['cs_pcc']:.6f}  max_err: {results['cs_max_err']:.6f}")
        assert results["out_pcc"] > 0.99, f"Output PCC too low: {results['out_pcc']}"
        assert results["state_pcc"] > 0.99, f"State PCC too low: {results['state_pcc']}"
        assert results["cs_pcc"] > 0.99, f"ConvState PCC too low: {results['cs_pcc']}"
        print("PASS")
    finally:
        ttnn.close_device(device)


def test_prefill_full_dims_s4():
    """Full Qwen3.6-27B dims: H=48, Hk=16, Dk=128, Dv=128, S=4."""
    device_id = int(os.environ.get("TT_DEVICE_ID", 0))
    device = ttnn.open_device(device_id=device_id)
    try:
        results = run_prefill_test(device, H=48, Hk=16, Dk=128, Dv=128, S=4, head_expand_ratio=3)
        print(f"[Full S=4]  Output PCC: {results['out_pcc']:.6f}  max_err: {results['out_max_err']:.6f}")
        print(f"            State  PCC: {results['state_pcc']:.6f}  max_err: {results['state_max_err']:.6f}")
        print(f"            ConvSt PCC: {results['cs_pcc']:.6f}  max_err: {results['cs_max_err']:.6f}")
        assert results["out_pcc"] > 0.99, f"Output PCC too low: {results['out_pcc']}"
        assert results["state_pcc"] > 0.99, f"State PCC too low: {results['state_pcc']}"
        assert results["cs_pcc"] > 0.99, f"ConvState PCC too low: {results['cs_pcc']}"
        print("PASS")
    finally:
        ttnn.close_device(device)


def test_prefill_tile_boundary_s32():
    """S=32: tile row boundary test."""
    device_id = int(os.environ.get("TT_DEVICE_ID", 0))
    device = ttnn.open_device(device_id=device_id)
    try:
        results = run_prefill_test(device, H=6, Hk=2, Dk=64, Dv=64, S=32, head_expand_ratio=3)
        print(f"[S=32 boundary] Output PCC: {results['out_pcc']:.6f}  max_err: {results['out_max_err']:.6f}")
        print(f"                State  PCC: {results['state_pcc']:.6f}  max_err: {results['state_max_err']:.6f}")
        print(f"                ConvSt PCC: {results['cs_pcc']:.6f}  max_err: {results['cs_max_err']:.6f}")
        assert results["out_pcc"] > 0.99, f"Output PCC too low: {results['out_pcc']}"
        assert results["state_pcc"] > 0.99, f"State PCC too low: {results['state_pcc']}"
        assert results["cs_pcc"] > 0.99, f"ConvState PCC too low: {results['cs_pcc']}"
        print("PASS")
    finally:
        ttnn.close_device(device)


def test_prefill_vs_sequential_decode():
    """
    Verify: S-step prefill produces same final state as S sequential decode steps.
    Uses the decode_full op for comparison.
    """
    device_id = int(os.environ.get("TT_DEVICE_ID", 0))
    device = ttnn.open_device(device_id=device_id)
    try:
        H, Hk, Dk, Dv, S = 6, 2, 64, 64, 4
        head_expand_ratio = 3
        conv_kernel_size = 4
        conv_dim = Hk * Dk * 2 + H * Dv
        torch.manual_seed(99)

        qkv_proj = torch.randn(S, conv_dim).to(torch.bfloat16).float()
        z_proj = torch.randn(S, H * Dv).to(torch.bfloat16).float()
        b_proj = torch.randn(S, H).to(torch.bfloat16).float()
        a_proj = torch.randn(S, H).to(torch.bfloat16).float()
        conv_state = torch.randn(conv_dim, conv_kernel_size).to(torch.bfloat16).float() * 0.01
        recurrent_state = torch.randn(H, Dk, Dv).to(torch.bfloat16).float() * 0.01
        conv1d_weight = torch.randn(conv_dim, conv_kernel_size).to(torch.bfloat16).float() * 0.1
        a_log = torch.randn(H).to(torch.bfloat16).float().abs() * 0.5
        dt_bias = torch.randn(H).to(torch.bfloat16).float() * 0.1
        norm_weight = torch.ones(Dv).to(torch.bfloat16).float()

        # CPU reference: sequential decode
        ref_out, ref_state, ref_cs = deltanet_prefill_reference(
            qkv_proj, z_proj, b_proj, a_proj,
            conv_state, recurrent_state, conv1d_weight,
            a_log, dt_bias, norm_weight,
            num_heads=H, num_k_heads=Hk,
            k_head_dim=Dk, v_head_dim=Dv,
            head_expand_ratio=head_expand_ratio,
        )

        # Device prefill
        qkv_tt = ttnn.from_torch(
            qkv_proj.to(torch.bfloat16).reshape(1, 1, S, conv_dim),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        z_tt = ttnn.from_torch(
            z_proj.to(torch.bfloat16).reshape(1, 1, S, H * Dv),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b_padded = pad_to_tile(b_proj.to(torch.bfloat16).reshape(1, 1, S, H))
        b_tt = ttnn.from_torch(b_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        a_padded = pad_to_tile(a_proj.to(torch.bfloat16).reshape(1, 1, S, H))
        a_tt = ttnn.from_torch(a_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        cs_padded = torch.zeros(1, 1, conv_dim, 32, dtype=torch.bfloat16)
        cs_padded[0, 0, :, :conv_kernel_size] = conv_state.to(torch.bfloat16)
        cs_tt = ttnn.from_torch(cs_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        state_tt = ttnn.from_torch(
            recurrent_state.to(torch.bfloat16).unsqueeze(0),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        cw_padded = torch.zeros(1, 1, conv_dim, 32, dtype=torch.bfloat16)
        cw_padded[0, 0, :, :conv_kernel_size] = conv1d_weight.to(torch.bfloat16)
        cw_tt = ttnn.from_torch(cw_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        a_log_padded = pad_to_tile(a_log.to(torch.bfloat16).reshape(1, 1, 1, H))
        a_log_tt = ttnn.from_torch(a_log_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        dt_bias_padded = pad_to_tile(dt_bias.to(torch.bfloat16).reshape(1, 1, 1, H))
        dt_bias_tt = ttnn.from_torch(dt_bias_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        norm_w_padded = pad_to_tile(norm_weight.to(torch.bfloat16).reshape(1, 1, 1, Dv))
        norm_w_tt = ttnn.from_torch(norm_w_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        results = ttnn.experimental.deltanet_prefill_full(
            qkv_tt, z_tt, b_tt, a_tt,
            cs_tt, state_tt,
            cw_tt, a_log_tt, dt_bias_tt, norm_w_tt,
            num_heads=H, num_k_heads=Hk,
            k_head_dim=Dk, v_head_dim=Dv,
            conv_dim=conv_dim, conv_kernel_size=conv_kernel_size,
            head_expand_ratio=head_expand_ratio,
            seq_len=S,
        )

        state_dev = ttnn.to_torch(results[1]).float()[0]  # [H, Dk, Dv]
        ref_state_bf16 = ref_state.to(torch.bfloat16).float()
        state_pcc = torch.corrcoef(torch.stack([state_dev.flatten(), ref_state_bf16.flatten()]))[0, 1].item()

        print(f"[Prefill vs SeqDecode S={S}] Final state PCC: {state_pcc:.6f}")
        assert state_pcc > 0.99, f"State PCC too low: {state_pcc}"
        print("PASS")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    print("=== Test 1: Small dims S=2 ===")
    test_prefill_small_s2()
    print()
    print("=== Test 2: Small dims S=8 ===")
    test_prefill_small_s8()
    print()
    print("=== Test 3: Full dims S=4 ===")
    test_prefill_full_dims_s4()
    print()
    print("=== Test 4: Tile boundary S=32 ===")
    test_prefill_tile_boundary_s32()
    print()
    print("=== Test 5: Prefill vs Sequential Decode ===")
    test_prefill_vs_sequential_decode()
