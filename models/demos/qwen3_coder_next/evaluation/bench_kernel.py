# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kernel-level benchmark: measure deltanet_decode_full latency per call."""

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

WARMUP = 3
ITERS = 20


def main():
    torch.manual_seed(42)
    device = ttnn.open_device(device_id=0)
    try:
        def to_tt(t, shape, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                t.reshape(shape).to(torch.bfloat16),
                dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
            )

        qkv_tt = to_tt(torch.randn(conv_dim) * 0.1, [1, 1, 1, conv_dim])
        z_tt = to_tt(torch.randn(H * Dv) * 0.1, [1, 1, 1, H * Dv])
        b_tt = to_tt(torch.randn(H) * 0.1, [1, 1, 1, H])
        a_tt = to_tt(torch.randn(H) * 0.1, [1, 1, 1, H])

        conv_padded = torch.nn.functional.pad(torch.randn(1, conv_dim, conv_k) * 0.01, (0, 32 - conv_k))
        conv_state_tt = to_tt(conv_padded, [1, 1, conv_dim, 32])

        state_tt = to_tt(torch.randn(1, H, Dk, Dv) * 0.01, [1, H, Dk, Dv])

        conv_w_padded = torch.nn.functional.pad(torch.randn(conv_dim, conv_k) * 0.1, (0, 28))
        conv_w_tt = to_tt(conv_w_padded.unsqueeze(0), [1, 1, conv_dim, 32])

        a_log_tt = to_tt(torch.log(torch.rand(H) * 4 + 0.1), [1, 1, 1, H])
        dt_bias_tt = to_tt(torch.ones(H) * 0.5, [1, 1, 1, H])
        norm_w_tt = to_tt(torch.ones(Dv), [1, 1, 1, Dv])

        # Warmup
        print(f"Warming up ({WARMUP} iters)...")
        for _ in range(WARMUP):
            out, new_state, new_conv = ttnn.experimental.deltanet_decode_full(
                qkv_tt, z_tt, b_tt, a_tt,
                conv_state_tt, state_tt,
                conv_w_tt, a_log_tt, dt_bias_tt, norm_w_tt,
                num_heads=H, num_k_heads=Hk,
                k_head_dim=Dk, v_head_dim=Dv,
                conv_dim=conv_dim, conv_kernel_size=conv_k,
                head_expand_ratio=expand,
            )
            ttnn.synchronize_device(device)
            # Feed state back for next iteration
            state_tt = new_state
            conv_state_tt = new_conv
            ttnn.deallocate(out)

        # Benchmark
        print(f"Benchmarking ({ITERS} iters)...")
        times = []
        for i in range(ITERS):
            t0 = time.perf_counter()
            out, new_state, new_conv = ttnn.experimental.deltanet_decode_full(
                qkv_tt, z_tt, b_tt, a_tt,
                conv_state_tt, state_tt,
                conv_w_tt, a_log_tt, dt_bias_tt, norm_w_tt,
                num_heads=H, num_k_heads=Hk,
                k_head_dim=Dk, v_head_dim=Dv,
                conv_dim=conv_dim, conv_kernel_size=conv_k,
                head_expand_ratio=expand,
            )
            ttnn.synchronize_device(device)
            elapsed = time.perf_counter() - t0
            times.append(elapsed * 1000)
            state_tt = new_state
            conv_state_tt = new_conv
            ttnn.deallocate(out)

        times.sort()
        p50 = times[len(times) // 2]
        p90 = times[int(len(times) * 0.9)]
        mean = sum(times) / len(times)
        print(f"\nResults ({ITERS} iterations):")
        print(f"  Mean:  {mean:.2f} ms")
        print(f"  P50:   {p50:.2f} ms")
        print(f"  P90:   {p90:.2f} ms")
        print(f"  Min:   {times[0]:.2f} ms")
        print(f"  Max:   {times[-1]:.2f} ms")
        print(f"  48 DeltaNet layers: {48 * p50:.1f} ms (p50 × 48)")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
