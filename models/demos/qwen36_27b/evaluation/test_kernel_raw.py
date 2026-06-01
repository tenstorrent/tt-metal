# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test steps 1-5 only (raw output, no norm+gate) to isolate mixed-format compute."""

import torch
import ttnn
import time

H = 48
Hk = 16
Dk = 128
Dv = 128
expand = H // Hk
key_dim = Dk * Hk
conv_dim = key_dim * 2 + H * Dv
conv_k = 4
scale = Dk ** -0.5


def silu(x):
    return x / (1.0 + torch.exp(-x))


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def cpu_reference_raw(qkv_raw, b_raw, a_raw, conv_state, recurrent_state,
                      conv_weight, a_log, dt_bias):
    """CPU reference: steps 1-5 only (raw output, no norm+gate)."""
    conv_state_sq = conv_state.squeeze(0)
    conv_state_sq = torch.roll(conv_state_sq, shifts=-1, dims=-1)
    conv_state_sq[:, -1] = qkv_raw
    new_conv_state = conv_state_sq.unsqueeze(0).clone()
    qkv_conv = (conv_state_sq * conv_weight).sum(dim=-1)
    qkv_conv = torch.nn.functional.silu(qkv_conv)

    q_raw, k_raw, v_raw = torch.split(qkv_conv, [key_dim, key_dim, H * Dv], dim=-1)
    q = q_raw.reshape(1, Hk, Dk)
    k = k_raw.reshape(1, Hk, Dk)
    v = v_raw.reshape(1, H, Dv)

    q = q.repeat_interleave(expand, dim=1)
    k = k.repeat_interleave(expand, dim=1)

    q = l2norm(q, dim=-1).float() * scale
    k = l2norm(k, dim=-1).float()
    v = v.float()

    beta = torch.sigmoid(b_raw.float()[:H])
    g = -a_log[:H].exp() * torch.nn.functional.softplus(a_raw.float()[:H] + dt_bias[:H])
    decay = g.exp()

    S = recurrent_state[0].float()
    g_t = decay.unsqueeze(-1).unsqueeze(-1)
    beta_t = beta.unsqueeze(-1)
    q_t = q[0]
    k_t = k[0]
    v_t = v[0]

    S_mid = S * g_t
    kv_mem = (S_mid * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * beta_t
    S_new = S_mid + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    raw_output = (S_new * q_t.unsqueeze(-1)).sum(dim=-2)

    return raw_output.reshape(1, 1, 1, -1), S_new.unsqueeze(0), new_conv_state


def main():
    torch.manual_seed(42)

    device = ttnn.open_device(device_id=0)
    try:
        qkv_raw = torch.randn(conv_dim) * 0.1
        z_raw = torch.randn(H * Dv) * 0.1
        b_raw = torch.randn(H) * 0.1
        a_raw = torch.randn(H) * 0.1

        conv_state = torch.randn(1, conv_dim, conv_k) * 0.01
        recurrent_state = torch.randn(1, H, Dk, Dv) * 0.01
        conv_weight = torch.randn(conv_dim, conv_k) * 0.1
        a_log = torch.log(torch.rand(H) * 4 + 0.1)
        dt_bias = torch.ones(H) * 0.5
        norm_weight = torch.ones(Dv) * 1.0

        cpu_out, cpu_state, cpu_conv = cpu_reference_raw(
            qkv_raw, b_raw, a_raw,
            conv_state, recurrent_state,
            conv_weight, a_log, dt_bias,
        )
        print(f"CPU raw output: min={cpu_out.min():.4f} max={cpu_out.max():.4f} mean={cpu_out.mean():.4f}")

        def to_tt(t, shape, dtype=ttnn.bfloat16):
            t = t.reshape(shape).to(torch.bfloat16)
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

        qkv_tt = to_tt(qkv_raw, [1, 1, 1, conv_dim])
        z_tt = to_tt(z_raw, [1, 1, 1, H * Dv])
        b_tt = to_tt(b_raw, [1, 1, 1, H])
        a_tt = to_tt(a_raw, [1, 1, 1, H])

        conv_padded = torch.nn.functional.pad(conv_state, (0, 32 - conv_k))
        conv_state_tt = to_tt(conv_padded, [1, 1, conv_dim, 32])

        state_tt = ttnn.from_torch(
            recurrent_state.float().reshape(1, H, Dk, Dv),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device,
        )

        conv_w_padded = torch.nn.functional.pad(conv_weight, (0, 28))
        conv_w_tt = to_tt(conv_w_padded.unsqueeze(0), [1, 1, conv_dim, 32])

        a_log_tt = to_tt(a_log, [1, 1, 1, H])
        dt_bias_tt = to_tt(dt_bias, [1, 1, 1, H])
        norm_w_tt = to_tt(norm_weight, [1, 1, 1, Dv])

        print("Running fused kernel (steps 1-5 only, passthrough output)...", flush=True)
        t0 = time.time()
        output_tt, new_state_tt, new_conv_tt = ttnn.experimental.deltanet_decode_full(
            qkv_tt, z_tt, b_tt, a_tt,
            conv_state_tt, state_tt,
            conv_w_tt, a_log_tt, dt_bias_tt, norm_w_tt,
            num_heads=H, num_k_heads=Hk,
            k_head_dim=Dk, v_head_dim=Dv,
            conv_dim=conv_dim, conv_kernel_size=conv_k,
            head_expand_ratio=expand,
        )
        ttnn.synchronize_device(device)
        elapsed = time.time() - t0
        print(f"Kernel done in {elapsed:.2f}s")

        fused_out = ttnn.to_torch(output_tt).float().reshape(1, 1, 1, -1)
        fused_state = ttnn.to_torch(new_state_tt).float().reshape(1, H, Dk, Dv)

        print(f"\nFused raw output: min={fused_out.min():.4f} max={fused_out.max():.4f} mean={fused_out.mean():.4f}")
        print(f"CPU   raw output: min={cpu_out.min():.4f} max={cpu_out.max():.4f} mean={cpu_out.mean():.4f}")

        # Check if output is garbage (NaN, Inf, or extreme values)
        if torch.isnan(fused_out).any() or torch.isinf(fused_out).any():
            print("\nERROR: Output contains NaN/Inf!")
        elif fused_out.abs().max() > 1e10:
            print(f"\nERROR: Output has extreme values (max abs={fused_out.abs().max():.2e})")
        else:
            print("\nOutput values are in reasonable range.")

        # Per-head comparison
        cpu_flat = cpu_out.reshape(H, Dv)
        fused_flat = fused_out.reshape(H, Dv)

        print(f"\nPer-head raw output (first 4 heads):")
        for h in range(4):
            cpu_h = cpu_flat[h]
            fused_h = fused_flat[h]
            pcc = torch.corrcoef(torch.stack([cpu_h.flatten(), fused_h.flatten()]))[0, 1].item()
            mse = ((cpu_h - fused_h) ** 2).mean().item()
            print(f"  Head {h}: PCC={pcc:.6f}  MSE={mse:.6f}  "
                  f"cpu=[{cpu_h.min():.4f},{cpu_h.max():.4f}]  "
                  f"fused=[{fused_h.min():.4f},{fused_h.max():.4f}]")

        # State comparison
        cpu_s = cpu_state.float().reshape(H, Dk, Dv)
        fused_s = fused_state.reshape(H, Dk, Dv)
        print(f"\nState comparison (first 4 heads):")
        for h in range(4):
            cpu_sh = cpu_s[h].flatten()
            fused_sh = fused_s[h].flatten()
            pcc = torch.corrcoef(torch.stack([cpu_sh, fused_sh]))[0, 1].item()
            mse = ((cpu_sh - fused_sh) ** 2).mean().item()
            print(f"  Head {h}: PCC={pcc:.6f}  MSE={mse:.6f}")

        overall_pcc = torch.corrcoef(
            torch.stack([cpu_out.flatten(), fused_out.flatten()])
        )[0, 1].item()
        print(f"\nOverall raw output PCC: {overall_pcc:.6f}")

        state_pcc = torch.corrcoef(
            torch.stack([cpu_state.flatten(), fused_state.flatten()])
        )[0, 1].item()
        print(f"Overall state PCC: {state_pcc:.6f}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
