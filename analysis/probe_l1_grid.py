# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill SDPA configs that must build and run wherever main's do: head dims 64 to 512, q/k chunks 64 to 512,
fp32 DEST on and off, causal and not, bf16 and bfp8 K/V, the 8x8 grid (and 8x4 at d512). One line per config:
PROBE_L1 name status (ok or the first line of the error)."""
import itertools

import ttnn

CHUNKS = [(64, 64), (128, 128), (128, 256), (256, 128), (256, 256), (128, 512), (256, 512), (512, 256), (512, 512)]


def configs():
    for d in (64, 128, 256, 512):
        grids = ((8, 8), (8, 4)) if d == 512 else ((8, 8),)
        for (qc, kc), fp32, causal, kv_dt, grid in itertools.product(
            CHUNKS, (False, True), (True, False), ("bfloat16", "bfloat8_b"), grids
        ):
            yield d, qc, kc, fp32, causal, kv_dt, grid


def first_line(e):
    lines = [l.strip() for l in str(e).splitlines() if l.strip()]
    for l in lines:
        if not l.startswith(("TT_THROW", "TT_FATAL", "info:", "RuntimeError")):
            return l[:160]
    return lines[0][:160] if lines else type(e).__name__


def test_probe_l1_grid(device):
    for d, qc, kc, fp32, causal, kv_dt, grid in configs():
        s = max(16 * kc, 2048)
        name = f"d{d}_q{qc}_k{kc}_{'fp32' if fp32 else 'bf16dest'}_{'causal' if causal else 'full'}_{kv_dt}_g{grid[0]}x{grid[1]}"
        tensors = []
        try:
            q = ttnn.rand((1, 8, s, d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, seed=1)
            tensors.append(q)
            kv = []
            for seed in (2, 3):
                x = ttnn.rand((1, 2, s, d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, seed=seed)
                y = ttnn.typecast(x, getattr(ttnn, kv_dt))
                x.deallocate()
                tensors.append(y)
                kv.append(y)
            pc = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*grid), q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=False
            )
            ck = ttnn.init_device_compute_kernel_config(
                device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False
            )
            out = ttnn.transformer.scaled_dot_product_attention(
                q, kv[0], kv[1], is_causal=causal, scale=d**-0.5, program_config=pc, compute_kernel_config=ck
            )
            ttnn.synchronize_device(device)
            tensors.append(out)
            status = "ok"
        except Exception as e:
            status = "ERR " + first_line(e)
        for t in tensors:
            t.deallocate()
        print(f"PROBE_L1 {name} {status}", flush=True)
