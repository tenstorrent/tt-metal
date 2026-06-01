import torch
import ttnn


def test_debug():
    device = ttnn.open_device(device_id=0)
    try:
        H, Dk, Dv = 1, 32, 32
        torch.manual_seed(0)

        q_t = torch.ones(1, H, 1, Dk).to(torch.bfloat16).float() * 0.1
        k_t = torch.ones(1, H, 1, Dk).to(torch.bfloat16).float() * 0.1
        v_t = torch.ones(1, H, 1, Dv).to(torch.bfloat16).float() * 0.5
        decay_t = torch.ones(1, H, 1, 1).to(torch.bfloat16).float() * 0.9
        beta_t = torch.ones(1, H, 1, 1).to(torch.bfloat16).float() * 0.5
        state_t = torch.eye(Dk, Dv).unsqueeze(0).unsqueeze(0).to(torch.bfloat16).float() * 0.1

        # CPU reference
        S = state_t[0, 0].clone()
        g = decay_t[0, 0, 0, 0]; b = beta_t[0, 0, 0, 0]
        q_h = q_t[0, 0, 0]; k_h = k_t[0, 0, 0]; v_h = v_t[0, 0, 0]
        S1 = S * g
        mem = k_h @ S1
        delta = (v_h - mem) * b
        S2 = S1 + k_h.unsqueeze(-1) * delta.unsqueeze(-2)
        out_ref = q_h @ S2

        # Device kernel
        q_tt = ttnn.from_torch(q_t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        k_tt = ttnn.from_torch(k_t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        v_tt = ttnn.from_torch(v_t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        decay_tt = ttnn.from_torch(decay_t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        beta_tt = ttnn.from_torch(beta_t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        state_tt = ttnn.from_torch(state_t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        out_tt, new_state_tt = ttnn.experimental.deltanet_decode(
            q_tt, k_tt, v_tt, decay_tt, beta_tt, state_tt,
            num_heads=H, k_head_dim=Dk, v_head_dim=Dv,
        )

        out_dev = ttnn.to_torch(out_tt).float()
        state_dev = ttnn.to_torch(new_state_tt).float()

        print("=== State comparison ===")
        S_ref = S2.unsqueeze(0).unsqueeze(0)
        print(f"Ref state shape: {S_ref.shape}")
        print(f"Dev state shape: {state_dev.shape}")

        # Print rows of state
        for r in [0, 1, 15, 16, 31]:
            print(f"  Row {r}: dev={state_dev[0,0,r,:4].tolist()} ref={S_ref[0,0,r,:4].tolist()}")

        # Check non-zero pattern
        dev_nz = (state_dev.abs() > 1e-4).sum().item()
        ref_nz = (S_ref.abs() > 1e-4).sum().item()
        print(f"Non-zero elements: dev={dev_nz} ref={ref_nz}")

        print(f"\n=== Output comparison ===")
        print(f"Dev out shape: {out_dev.shape}")
        print(f"Dev: {out_dev.flatten()[:8].tolist()}")
        print(f"Ref: {out_ref[:8].tolist()}")

        # Also test: manual matmul of q @ state_dev to verify
        q_flat = q_t[0, 0, 0]  # [32]
        manual_out = q_flat @ state_dev[0, 0]  # [32]
        print(f"\nManual q@state_dev: {manual_out[:8].tolist()}")
        print(f"This matches dev out? {torch.allclose(out_dev.flatten(), manual_out.flatten(), atol=0.01)}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_debug()
