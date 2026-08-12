# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decide the ``MinimalMatmulConfig`` question on device kernel time, not wall clock.

``prefill_matmul_kblock_confirm.py`` showed the host-side A/B cannot resolve a
1-3 % gap: measuring the shipped default against *itself* in the same paired
loop reports -0.5 % to -10.8 %, because the second op of each pair is
systematically penalised.  Device kernel duration from the Tracy profiler is
the metric the committed perf reports use and is immune to that.

Run under::

    python -m tracy -r -p -v models/.../bench/prefill_matmul_kblock_device.py

Each (shape, config) pair is emitted as ``REPS`` consecutive
``MinimalMatmulDeviceOperation`` invocations; groups appear in the CSV in the
order printed to stdout, so the companion parser can label them.  The default
is re-emitted between every candidate, so drift shows up as a spread across
the default groups rather than as a fake candidate win.
"""
from __future__ import annotations

import torch

import ttnn

ROWS = [8192, 4096]
#: (label, K, N, default_subblocks, [(M_block, K_block, N_block), ...])
CASES = [
    ("wqkv    ", 6656, 4608, (4, 2), [(8, 13, 8), (8, 6, 8)]),
    ("o_proj  ", 4096, 6656, (4, 2), [(8, 13, 8), (16, 4, 8), (8, 4, 8)]),
    ("mlp_gate", 6656, 19968, (2, 4), [(8, 6, 16), (16, 6, 8), (8, 13, 8)]),
    ("mlp_down", 19968, 6656, (4, 2), [(8, 6, 8), (8, 16, 8), (8, 6, 16)]),
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
            for label, k, n, (sh, sw), cands in CASES:
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

                def emit(tag, cfg):
                    print(f"GROUP rows={m} {label} {tag}", flush=True)
                    for _ in range(REPS):
                        ttnn.deallocate(ttnn.experimental.minimal_matmul(ta, tb, compute_kernel_config=ck, config=cfg))
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
