# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does passing ``dense_compute_kernel_config`` to the *decode* ``ttnn.linear`` matter?

Prefill pins the config that ``ttnn.linear`` selects by default for interleaved
BF16 (HiFi2 / approx=False / fp32_dest_acc_en=False / packer_l1_acc=True,
``matmul_device_operation.cpp:2749,2794-2800``), so the prefill before/after is
topology-only.  Decode takes a different auto-selected program config -- the
DRAM-sharded 1D one -- and this measures whether the compute config that comes
with it is the same thing.

Each projection is run at the decode shape (32 rows) three ways: with no
``compute_kernel_config`` (what the functional layer and the first fused
revision did), with the shipped explicit one, and against a float64 reference.
"""

from __future__ import annotations

import time

import torch

import ttnn

BATCH = 32
#: (label, K, N) -- the six decode dense projections
SHAPES = [
    ("wqkv    ", 6656, 4608),
    ("attn_gate", 6656, 4096),
    ("o_proj  ", 4096, 6656),
    ("mlp_gate", 6656, 19968),
    ("mlp_down", 19968, 6656),
]
ITERS = 30
ROUNDS = 3


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        ck = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        for label, k, n in SHAPES:
            torch.manual_seed(0)
            a = torch.randn(1, 1, BATCH, k).to(torch.bfloat16) * 0.1
            b = torch.randn(1, 1, k, n).to(torch.bfloat16) * 0.02
            ref = (a.double() @ b.double()).reshape(-1)
            ta = ttnn.from_torch(
                a, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            tb = ttnn.from_torch(
                b, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for name, cfg in (("auto (no config)", None), ("shipped explicit", ck)):

                def call():
                    return ttnn.linear(
                        ta, tb, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=cfg
                    )

                out = call()
                got = ttnn.to_torch(out).double().reshape(-1)
                pcc = torch.nn.functional.cosine_similarity(got - got.mean(), ref - ref.mean(), dim=0).item()
                rel = ((got - ref).abs().max() / ref.abs().max()).item()
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
                    f"DENSE {label} {name:18s} {best:9.2f} us/call  PCC(f64)={pcc:.9f}  max_rel_err={rel:.3e}",
                    flush=True,
                )
            for t in (ta, tb):
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
