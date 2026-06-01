# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Test fused DeltaNet decode kernel against CPU reference.
"""

import torch
import ttnn


def deltanet_decode_reference(q, k, v, decay, beta, state):
    """CPU reference: per-head DeltaNet recurrence."""
    # q: [1, H, 1, Dk], k: [1, H, 1, Dk], v: [1, H, 1, Dv]
    # decay: [1, H, 1, 1], beta: [1, H, 1, 1], state: [1, H, Dk, Dv]
    S = state.clone()
    H = S.shape[1]

    for h in range(H):
        g = decay[0, h, 0, 0]
        b = beta[0, h, 0, 0]
        q_h = q[0, h, 0, :]      # [Dk]
        k_h = k[0, h, 0, :]      # [Dk]
        v_h = v[0, h, 0, :]      # [Dv]
        S_h = S[0, h]            # [Dk, Dv]

        # Step 1: scale
        S_h = S_h * g
        # Step 2: matvec k @ S
        mem = k_h @ S_h          # [Dv]
        # Step 3: delta
        delta = (v_h - mem) * b  # [Dv]
        # Step 4: rank-1 update
        S_h = S_h + k_h.unsqueeze(-1) * delta.unsqueeze(-2)
        # Step 5: output
        out_h = q_h @ S_h       # [Dv]

        S[0, h] = S_h

    out = torch.zeros(1, H, 1, v.shape[-1])
    for h in range(H):
        out[0, h, 0] = q[0, h, 0] @ S[0, h]

    # Recompute properly
    S2 = state.clone()
    outputs = []
    for h in range(H):
        g = decay[0, h, 0, 0]
        b = beta[0, h, 0, 0]
        S_h = S2[0, h] * g
        mem = k[0, h, 0] @ S_h
        delta = (v[0, h, 0] - mem) * b
        S_h = S_h + k[0, h, 0].unsqueeze(-1) * delta.unsqueeze(-2)
        out_h = q[0, h, 0] @ S_h
        outputs.append(out_h)
        S2[0, h] = S_h

    out = torch.stack(outputs, dim=0).unsqueeze(0).unsqueeze(2)  # [1, H, 1, Dv]
    return out, S2


def test_deltanet_decode_kernel():
    device = ttnn.open_device(device_id=0)
    try:
        H, Dk, Dv = 4, 64, 64  # small for quick test
        torch.manual_seed(42)

        q_t = torch.randn(1, H, 1, Dk).to(torch.bfloat16).float()
        k_t = torch.randn(1, H, 1, Dk).to(torch.bfloat16).float()
        v_t = torch.randn(1, H, 1, Dv).to(torch.bfloat16).float()
        decay_t = torch.rand(1, H, 1, 1).to(torch.bfloat16).float().clamp(0.5, 0.99)
        beta_t = torch.rand(1, H, 1, 1).to(torch.bfloat16).float().clamp(0.1, 0.9)
        state_t = torch.randn(1, H, Dk, Dv).to(torch.bfloat16).float() * 0.01

        # CPU reference
        ref_out, ref_state = deltanet_decode_reference(q_t, k_t, v_t, decay_t, beta_t, state_t)

        # Device kernel
        q_tt = ttnn.from_torch(q_t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        k_tt = ttnn.from_torch(k_t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        v_tt = ttnn.from_torch(v_t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        decay_tt = ttnn.from_torch(decay_t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        beta_tt = ttnn.from_torch(beta_t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        state_tt = ttnn.from_torch(state_t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        out_tt, new_state_tt = ttnn.experimental.deltanet_decode(
            q_tt, k_tt, v_tt, decay_tt, beta_tt, state_tt,
            num_heads=H, k_head_dim=Dk, v_head_dim=Dv,
        )

        out_dev = ttnn.to_torch(out_tt).float()
        state_dev = ttnn.to_torch(new_state_tt).float()

        # Compare
        ref_out_bf16 = ref_out.to(torch.bfloat16).float()
        ref_state_bf16 = ref_state.to(torch.bfloat16).float()

        out_pcc = torch.corrcoef(torch.stack([out_dev.flatten(), ref_out_bf16.flatten()]))[0, 1].item()
        state_pcc = torch.corrcoef(torch.stack([state_dev.flatten(), ref_state_bf16.flatten()]))[0, 1].item()

        out_max_err = (out_dev - ref_out_bf16).abs().max().item()
        state_max_err = (state_dev - ref_state_bf16).abs().max().item()

        print(f"Output  PCC: {out_pcc:.6f}  max_err: {out_max_err:.6f}")
        print(f"State   PCC: {state_pcc:.6f}  max_err: {state_max_err:.6f}")

        assert out_pcc > 0.99, f"Output PCC too low: {out_pcc}"
        assert state_pcc > 0.99, f"State PCC too low: {state_pcc}"
        print("PASS")

    finally:
        ttnn.close_device(device)


def test_deltanet_decode_kernel_full_dims():
    """Test with Qwen3.6-27B dimensions: H=32, Dk=128, Dv=128."""
    device = ttnn.open_device(device_id=0)
    try:
        H, Dk, Dv = 32, 128, 128
        torch.manual_seed(123)

        q_t = torch.randn(1, H, 1, Dk).to(torch.bfloat16).float()
        k_t = torch.randn(1, H, 1, Dk).to(torch.bfloat16).float()
        v_t = torch.randn(1, H, 1, Dv).to(torch.bfloat16).float()
        decay_t = torch.rand(1, H, 1, 1).to(torch.bfloat16).float().clamp(0.5, 0.99)
        beta_t = torch.rand(1, H, 1, 1).to(torch.bfloat16).float().clamp(0.1, 0.9)
        state_t = torch.randn(1, H, Dk, Dv).to(torch.bfloat16).float() * 0.01

        ref_out, ref_state = deltanet_decode_reference(q_t, k_t, v_t, decay_t, beta_t, state_t)

        q_tt = ttnn.from_torch(q_t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        k_tt = ttnn.from_torch(k_t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        v_tt = ttnn.from_torch(v_t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        decay_tt = ttnn.from_torch(decay_t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        beta_tt = ttnn.from_torch(beta_t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        state_tt = ttnn.from_torch(state_t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        out_tt, new_state_tt = ttnn.experimental.deltanet_decode(
            q_tt, k_tt, v_tt, decay_tt, beta_tt, state_tt,
            num_heads=H, k_head_dim=Dk, v_head_dim=Dv,
        )

        out_dev = ttnn.to_torch(out_tt).float()
        state_dev = ttnn.to_torch(new_state_tt).float()

        ref_out_bf16 = ref_out.to(torch.bfloat16).float()
        ref_state_bf16 = ref_state.to(torch.bfloat16).float()

        out_pcc = torch.corrcoef(torch.stack([out_dev.flatten(), ref_out_bf16.flatten()]))[0, 1].item()
        state_pcc = torch.corrcoef(torch.stack([state_dev.flatten(), ref_state_bf16.flatten()]))[0, 1].item()

        print(f"[Full dims H={H} Dk={Dk} Dv={Dv}]")
        print(f"Output  PCC: {out_pcc:.6f}")
        print(f"State   PCC: {state_pcc:.6f}")

        assert out_pcc > 0.99, f"Output PCC too low: {out_pcc}"
        assert state_pcc > 0.99, f"State PCC too low: {state_pcc}"
        print("PASS")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    print("=== Test 1: Small dims (H=4, Dk=64, Dv=64) ===")
    test_deltanet_decode_kernel()
    print()
    print("=== Test 2: Full dims (H=32, Dk=128, Dv=128) ===")
    test_deltanet_decode_kernel_full_dims()
