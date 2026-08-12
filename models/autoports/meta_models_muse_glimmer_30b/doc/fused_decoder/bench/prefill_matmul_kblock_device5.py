# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Round 5: the two remaining holes in the ``MinimalMatmulConfig`` sweep.

1. ``MINIMAL_MATMUL_BLOCKS`` applies the MLP gate/up config from
   ``MINIMAL_MATMUL_MIN_ROWS = 3072`` upward, but rounds 1-3 measured 4096,
   6144 and 8192 only.  ``o_proj``'s config was shown to *invert* below the full
   chunk, so the 3072 point is measured here rather than extrapolated.
2. The attention-gate sweep (round 4) covered ``K_block`` 4-16 but skipped the
   non-power-of-two values 14 and 18, which are legal candidates.

Run under ``python -m tracy -r -p -v``; summarize with
``summarize_device_probe.py``.
"""

from __future__ import annotations

import torch

import ttnn

REPS = 8
#: (label, rows, K, N, subblocks, candidates)
CASES = [
    ("mlp_gate", 3072, 6656, 19968, (2, 4), [(8, 4, 16), (8, 6, 16), (8, 8, 8)]),
    ("attn_gate", 8192, 6656, 4096, (4, 2), [(8, 14, 8), (8, 18, 8), (8, 7, 8), (8, 9, 8)]),
]


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
        for label, rows, k, n, (sh, sw), cands in CASES:
            torch.manual_seed(0)
            ta = ttnn.from_torch(
                torch.randn(1, 1, rows, k).to(torch.bfloat16) * 0.1,
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

            def cfg(mb, kb, nb):
                return ttnn.MinimalMatmulConfig(
                    M_block_size=mb,
                    K_block_size=kb,
                    N_block_size=nb,
                    subblock_h=sh,
                    subblock_w=sw,
                    compute_with_storage_grid_size=full_grid,
                )

            def emit(tag, c):
                print(f"GROUP {REPS} rows={rows} {label} {tag}", flush=True)
                for _ in range(REPS):
                    ttnn.deallocate(ttnn.experimental.minimal_matmul(ta, tb, compute_kernel_config=ck, config=c))
                ttnn.synchronize_device(mesh)

            emit("default", None)
            for mb, kb, nb in cands:
                tag = f"M{mb}_K{kb}_N{nb}"
                print(f"GROUP 1 rows={rows} {label} probe_{tag}", flush=True)
                try:
                    ttnn.deallocate(
                        ttnn.experimental.minimal_matmul(ta, tb, compute_kernel_config=ck, config=cfg(mb, kb, nb))
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"BLOCKED {label} M{mb} K{kb} N{nb}: {type(exc).__name__}: "
                        f"{' '.join(str(exc).split())[:200]}",
                        flush=True,
                    )
                    continue
                emit(tag, cfg(mb, kb, nb))
                emit("default", None)
            for t in (ta, tb):
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
