# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Round 4: the attention-gate projection, the fifth prefill `minimal_matmul` shape.

Rounds 1-3 swept ``wqkv``, ``o_proj`` and the two MLP shapes but not
``attn_gate`` (``6656 x 4096``), which is its own dispatch in every prefill
chunk and costs 1,869 us of the 49,357 us `sliding` window -- the same order as
``o_proj`` (1,959 us), the shape a config was worth +2.80 % on.  This closes
that gap with the same method: device kernel time under Tracy, the full legal
``K_block`` range plus the ``M_block``/``N_block`` neighbourhood of the winners
found on the other shapes, and the op's own default re-measured between every
candidate.

``N = 4096 < M = 8192``, so the op's default subblocks are ``4x2``
(``minimal_matmul_program_factory.cpp:22-41``); only the blocking varies.

Run under ``python -m tracy -r -p -v``; summarize with
``summarize_device_probe.py``.
"""

from __future__ import annotations

import torch

import ttnn

LABEL = "attn_gate"
K, N = 6656, 4096
SUBBLOCKS = (4, 2)
ROWS = [8192, 4096]
#: K_block sweep (K is 208 tiles = 2^4 * 13), then the M/N neighbourhood.
K_BLOCKS = [4, 6, 8, 10, 12, 13, 16, 20, 26, 32, 52]
MN_CROSS = [(16, 4, 8), (16, 8, 8), (8, 4, 16), (8, 6, 16), (16, 4, 16), (8, 13, 8)]
REPS = 8


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        grid = mesh.compute_with_storage_grid_size()
        full_grid = ttnn.CoreCoord(grid.x, grid.y)
        sh, sw = SUBBLOCKS
        ck = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        for m in ROWS:
            torch.manual_seed(0)
            ta = ttnn.from_torch(
                torch.randn(1, 1, m, K).to(torch.bfloat16) * 0.1,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tb = ttnn.from_torch(
                torch.randn(1, 1, K, N).to(torch.bfloat16) * 0.02,
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
                print(f"GROUP {REPS} rows={m} {LABEL} {tag}", flush=True)
                for _ in range(REPS):
                    ttnn.deallocate(ttnn.experimental.minimal_matmul(ta, tb, compute_kernel_config=ck, config=c))
                ttnn.synchronize_device(mesh)

            def probe(c, tag):
                """One call, announced as its own group, so a rejection cannot truncate a group."""
                print(f"GROUP 1 rows={m} {LABEL} probe_{tag}", flush=True)
                try:
                    ttnn.deallocate(ttnn.experimental.minimal_matmul(ta, tb, compute_kernel_config=ck, config=c))
                    return True
                except Exception as exc:  # noqa: BLE001
                    return exc

            emit("default", None)
            candidates = [(8, kb, 8) for kb in K_BLOCKS] + MN_CROSS
            for mb, kb, nb in candidates:
                tag = f"M{mb}_K{kb}_N{nb}"
                ok = probe(cfg(mb, kb, nb), tag)
                if ok is not True:
                    print(
                        f"BLOCKED {LABEL} M{mb} K{kb} N{nb}: {type(ok).__name__}: "
                        f"{' '.join(str(ok).split())[:200]}",
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
