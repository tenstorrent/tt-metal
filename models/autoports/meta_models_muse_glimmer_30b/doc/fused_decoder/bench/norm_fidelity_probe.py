# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What does the RMSNorm compute-kernel uplift cost, and what does it buy?

The fused layer hands every RMSNorm an explicit
``HiFi4 / math_approx_mode=False / fp32_dest_acc_en=True / packer_l1_acc=True``
config.  ``ttnn.rms_norm``'s own default is
``HiFi4 / approx=True / fp32_dest_acc_en=False / packer_l1_acc=False``
(``rmsnorm.cpp:16-20``), which is what the functional layer used, so this is a
*fidelity* change and not a topology one -- the only one in the stage.

This measures it in isolation on both shapes the layer norms: the 8192-row
prefill chunk (interleaved, 110 cores) and a decode step (width-sharded on the
shipped 4x2 grid).  PCC is against a float64 torch reference, so a higher number
really is closer to the mathematical answer rather than closer to some other
BF16 rounding.

The stage-level counterpart is ``logs/norm_fidelity_control.log``, which runs
the whole graph with the norms on the op default.

Run under ``python -m tracy -r -p -v`` for device kernel time; the wall-clock
numbers it prints are also usable because the gap here is large.
"""
from __future__ import annotations

import time

import torch

import ttnn

HIDDEN = 6656
EPS = 1e-5
DECODE_ROWS = 32
NORM_GRID = (4, 2)  # choose_decode_norm_grid(6656, 11x10)
PREFILL_ROWS = 8192
ITERS = 20
ROUNDS = 3


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        configs = {
            "op default (functional)": None,
            "shipped uplift": ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        }
        gx, gy = NORM_GRID
        cores = gx * gy
        torch.manual_seed(0)
        weight = torch.randn(1, 1, 1, HIDDEN).to(torch.bfloat16) * 0.1 + 1.0
        tw = ttnn.from_torch(
            weight, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        for label, rows, sharded in (("prefill", PREFILL_ROWS, False), ("decode", DECODE_ROWS, True)):
            x = torch.randn(1, 1, rows, HIDDEN).to(torch.bfloat16)
            ref = (x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + EPS)) * weight.double()
            memcfg = ttnn.DRAM_MEMORY_CONFIG
            prg = None
            if sharded:
                memcfg = ttnn.create_sharded_memory_config(
                    shape=(rows, HIDDEN // cores),
                    core_grid=ttnn.CoreGrid(y=gy, x=gx),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                block_w = HIDDEN // cores // 32
                prg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=[gx, gy],
                    subblock_w=next(c for c in (4, 3, 2, 1) if block_w % c == 0),
                    block_h=max(rows // 32, 1),
                    block_w=block_w,
                    inplace=False,
                )
            tx = ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=memcfg)

            for name, ck in configs.items():
                print(f"GROUP {ITERS} {label} {name.split()[0]}", flush=True)

                def call():
                    return ttnn.rms_norm(
                        tx, epsilon=EPS, weight=tw, memory_config=memcfg, program_config=prg, compute_kernel_config=ck
                    )

                out = call()
                got = ttnn.to_torch(out).double().reshape(-1)
                want = ref.reshape(-1)
                pcc = torch.nn.functional.cosine_similarity(got - got.mean(), want - want.mean(), dim=0).item()
                rel = ((got - want).abs().max() / want.abs().max()).item()
                ttnn.deallocate(out)
                best = None
                for _ in range(ROUNDS):
                    ttnn.synchronize_device(mesh)
                    t0 = time.perf_counter()
                    for _ in range(ITERS):
                        ttnn.deallocate(call())
                    ttnn.synchronize_device(mesh)
                    dt = (time.perf_counter() - t0) / ITERS * 1e6
                    best = dt if best is None else min(best, dt)
                print(
                    f"NORM {label:8s} {name:24s} {best:9.2f} us/call   PCC(f64)={pcc:.9f}   max_rel_err={rel:.3e}",
                    flush=True,
                )
            ttnn.deallocate(tx)
        ttnn.deallocate(tw)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
