# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Causal prefill chains against no chains (SDPA_PROBE_CHAIN_MODE=0 on the probe build) over grid sizes, fp32 DEST
and bf16 DEST, bf16 and bfp8 K/V; device time per call from traced replays. Lines start with PROBE_CG."""
import os
import statistics
import time

import pytest
import ttnn


def cases():
    for d, nh, nkv in ((128, 32, 8), (256, 16, 8), (512, 16, 4)):
        for grid in ((8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (11, 5), (6, 6), (4, 4), (11, 10)):
            if d == 512 and grid in ((6, 6), (4, 4)):
                continue
            yield (f"fp32_bf16kv_d{d}_g{grid[0]}x{grid[1]}", nh, nkv, d, 4096, grid, True, "bfloat16")
    for d, nh, nkv in ((64, 8, 2), (128, 32, 8), (256, 16, 8)):
        for grid in ((8, 4), (6, 6), (4, 4), (8, 8), (11, 10)):
            for kv in ("bfloat16", "bfloat8_b"):
                yield (f"bf16dest_{kv}_d{d}_g{grid[0]}x{grid[1]}", nh, nkv, d, 4096, grid, False, kv)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 4 * 1024 * 1024}], indirect=True)
def test_probe_chain_grid(device):
    mode = os.environ.get("SDPA_PROBE_CHAIN_MODE", "auto")
    full = device.compute_with_storage_grid_size()
    for name, nh, nkv, d, s, grid, fp32, kv in cases():
        if grid[0] > full.x or grid[1] > full.y:
            continue
        ck = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4 if fp32 else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False
        )
        q = ttnn.rand((1, nh, s, d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, seed=1)
        kvs = []
        for seed in (2, 3):
            x = ttnn.rand((1, nkv, s, d), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, seed=seed)
            kvs.append(ttnn.typecast(x, getattr(ttnn, kv)))
            x.deallocate()
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*grid), q_chunk_size=128, k_chunk_size=128, exp_approx_mode=False
        )

        def call():
            return ttnn.transformer.scaled_dot_product_attention(
                q, kvs[0], kvs[1], is_causal=True, scale=d**-0.5, program_config=pc, compute_kernel_config=ck
            )

        try:
            call().deallocate()
        except Exception as e:
            print(f"PROBE_CG mode={mode} name={name} ERR {str(e).splitlines()[0][:100]}", flush=True)
            for t in [q] + kvs:
                t.deallocate()
            continue
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        out = call()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        for _ in range(3):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        per = []
        for _ in range(7):
            t0 = time.perf_counter()
            for _ in range(10):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            per.append((time.perf_counter() - t0) / 10 * 1e6)
        ttnn.release_trace(device, tid)
        print(f"PROBE_CG mode={mode} name={name} us={statistics.median(per):.2f} min={min(per):.2f}", flush=True)
        out.deallocate()
        for t in [q] + kvs:
            t.deallocate()
