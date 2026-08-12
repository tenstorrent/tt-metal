# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Do the two winning configs still win on a *tail* prefill chunk?

Rounds 1-2 selected ``o_proj -> M16 K4 N8`` and ``mlp_gate/up -> M8 K4 N16`` at
the full 8192-row chunk.  A prompt whose length is not a multiple of 8192 ends
in a shorter chunk, and ``_dense`` still routes anything at or above
``MINIMAL_MATMUL_MIN_ROWS = 3072`` to ``minimal_matmul``.  Round 1 already
showed ``o_proj`` losing ~10 % at 4096 rows, so this measures both winners at
the two representative tail heights to decide whether the shipped rule keys on
the exact 8192-row chunk or on every ``minimal_matmul`` call.

Run under ``python -m tracy -r -p -v``.
"""
from __future__ import annotations

import torch

import ttnn

ROWS = [4096, 6144]
CASES = [
    ("o_proj  ", 4096, 6656, [(16, 4, 8)]),
    ("mlp_gate", 6656, 19968, [(8, 4, 16)]),
]
REPS = 8


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        grid = mesh.compute_with_storage_grid_size()
        full_grid = ttnn.CoreCoord(grid.x, grid.y)
        ck = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        for m in ROWS:
            for label, k, n, cands in CASES:
                sh, sw = (2, 4) if n >= m else (4, 2)
                torch.manual_seed(0)
                ta = ttnn.from_torch(
                    torch.randn(1, 1, m, k).to(torch.bfloat16) * 0.1,
                    device=mesh,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                tb = ttnn.from_torch(
                    torch.randn(1, 1, k, n).to(torch.bfloat16) * 0.02,
                    device=mesh,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                def emit(tag, c):
                    print(f"GROUP {REPS} rows={m} {label} {tag}", flush=True)
                    for _ in range(REPS):
                        ttnn.deallocate(ttnn.experimental.minimal_matmul(ta, tb, compute_kernel_config=ck, config=c))
                    ttnn.synchronize_device(mesh)

                emit("default", None)
                for mb, kb, nb in cands:
                    emit(
                        f"M{mb}_K{kb}_N{nb}",
                        ttnn.MinimalMatmulConfig(
                            M_block_size=mb,
                            K_block_size=kb,
                            N_block_size=nb,
                            subblock_h=sh,
                            subblock_w=sw,
                            compute_with_storage_grid_size=full_grid,
                        ),
                    )
                    emit("default", None)
                for t in (ta, tb):
                    ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
