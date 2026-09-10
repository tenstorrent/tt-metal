#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Probe the ttnn ops the stage-07 on-device CFG + top-k sampling relies on (topk k=64 over the 16416-wide logits
window and over the 8192-wide padded depth-head logits, gather of both CFG rows at the conditional row's top
indices, one-hot row-copy / CFG matmuls). Prints agreement with torch and rough timings.

    with_hw_lock timeout 600 $MM3_PY $MM3_MODEL_DIR/scripts/probe_sampling_ops.py
"""

import time

import torch

import ttnn

TILE = 32


def main():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=20_000_000)
    dev.enable_program_cache()
    try:
        for width in (16416, 8192):
            torch.manual_seed(0)
            logits = (torch.randn(1, 1, TILE, width) * 4).to(torch.bfloat16)
            x = ttnn.from_torch(logits, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            # 1. row copy: rows 0 and 1 <- row 0 (cond), so a row-wise topk gives cond's indices in both rows.
            m = torch.zeros(1, 1, TILE, TILE)
            m[0, 0, 0, 0] = 1.0
            m[0, 0, 1, 0] = 1.0
            mt = ttnn.from_torch(m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            cc = ttnn.init_device_compute_kernel_config(
                dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
            )
            for i in range(3):
                t0 = time.perf_counter()
                xc = ttnn.matmul(
                    mt, x, compute_kernel_config=cc, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                vals, idx = ttnn.topk(xc, 64, dim=-1, largest=True, sorted=True)
                gathered = ttnn.gather(x, -1, idx)
                ttnn.synchronize_device(dev)
                dt = time.perf_counter() - t0
            v = ttnn.to_torch(vals).float()
            ix = ttnn.to_torch(idx).to(torch.int64)
            g = ttnn.to_torch(gathered).float()
            ref_v, ref_i = torch.topk(logits[0, 0, 0].float(), 64)
            print(f"width {width}: idx dtype {idx.dtype}, gathered dtype {gathered.dtype}, wall (3rd) {dt*1e3:.2f} ms")
            print("  row copy exact:", torch.equal(ttnn.to_torch(xc)[0, 0, 1].float(), logits[0, 0, 0].float()))
            print(
                "  topk values row0 match:",
                torch.equal(v[0, 0, 0], ref_v),
                " row1==row0 idx:",
                torch.equal(ix[0, 0, 0], ix[0, 0, 1]),
            )
            print("  topk indices match (as sets):", set(ix[0, 0, 0].tolist()) == set(ref_i.tolist()))
            print("  gathered row0 == cond@idx:", torch.equal(g[0, 0, 0], logits[0, 0, 0].float()[ix[0, 0, 0]]))
            print("  gathered row1 == uncond@idx:", torch.equal(g[0, 0, 1], logits[0, 0, 1].float()[ix[0, 0, 1]]))
            # 2. additive mask then topk (masked columns must never appear)
            mask = torch.zeros(1, 1, TILE, width)
            mask[..., :22] = -1e9
            mask[..., 23:27] = -1e9
            mask[..., width - 5 :] = -1e9
            maskt = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            xm = ttnn.add(xc, maskt)
            vals2, idx2 = ttnn.topk(xm, 64, dim=-1, largest=True, sorted=True)
            ix2 = ttnn.to_torch(idx2).to(torch.int64)[0, 0, 0]
            bad = [i for i in ix2.tolist() if i < 22 or 23 <= i < 27 or i >= width - 5]
            print("  masked columns in topk:", bad)
            # 3. CFG matmul in fp32 output
            cfg = torch.zeros(1, 1, TILE, TILE)
            cfg[0, 0, 0, 0], cfg[0, 0, 0, 1] = 1.5, -0.5
            cfgt = ttnn.from_torch(cfg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            try:
                gd = ttnn.matmul(
                    cfgt, x, compute_kernel_config=cc, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                gd_t = ttnn.to_torch(gd)[0, 0, 0].float()
                ref = 1.5 * logits[0, 0, 0].float() - 0.5 * logits[0, 0, 1].float()
                print("  fp32 CFG matmul max err:", float((gd_t - ref).abs().max()))
                g2 = ttnn.gather(gd, -1, idx)
                print(
                    "  gather fp32 ok:",
                    g2.dtype,
                    float((ttnn.to_torch(g2)[0, 0, 0].float() - ref[ix[0, 0, 0]]).abs().max()),
                )
            except Exception as e:  # noqa: BLE001
                print("  fp32 CFG matmul / gather failed:", repr(e)[:300])
            # 4. bf16 guided topk vs fp32 guided top-50 set inclusion
            gdb = ttnn.matmul(
                cfgt, x, compute_kernel_config=cc, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            _, idx3 = ttnn.topk(gdb, 64, dim=-1, largest=True, sorted=True)
            ix3 = set(ttnn.to_torch(idx3).to(torch.int64)[0, 0, 0].tolist())
            ref50 = set(torch.topk(1.5 * logits[0, 0, 0].float() - 0.5 * logits[0, 0, 1].float(), 50).indices.tolist())
            print("  fp32 guided top-50 within bf16 guided top-64:", ref50 <= ix3)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
