# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Round 2 of the device-time ``MinimalMatmulConfig`` sweep: widen around the winners.

Round 1 (``prefill_matmul_kblock_device.py``) found, on device kernel time at
the shipped 8192-row prefill chunk, that ``o_proj`` likes ``M16 K4 N8``
(+2.88 %) and the MLP gate/up shape likes ``M8 K6 N16`` (+2.19 %), while
``wqkv`` and ``mlp_down`` are best on the op's own default.  This round sweeps
the neighbourhood of both winners, and the same neighbourhood on the two shapes
that preferred the default, so a win is not missed just because round 1's
candidate list was drawn from the old wall-clock sweep.

It also records the exact error text for the ``K_block >= 20`` configs that the
op rejects, which is the op-contract blocker the optimization standard wants
instead of a bare "failed".

Run under ``python -m tracy -r -p -v``; groups are labelled on stdout in
emission order, parsed by the companion summary script.
"""

from __future__ import annotations

import torch

import ttnn

ROWS = 8192
#: (label, K, N, default_subblocks, candidates)
CASES = [
    ("wqkv    ", 6656, 4608, (4, 2), [(16, 4, 8), (16, 8, 8), (8, 6, 16), (8, 4, 16), (16, 4, 16)]),
    ("o_proj  ", 4096, 6656, (4, 2), [(16, 4, 8), (16, 8, 8), (16, 2, 8), (16, 4, 16), (16, 6, 8), (8, 4, 16)]),
    ("mlp_gate", 6656, 19968, (2, 4), [(8, 6, 16), (8, 8, 16), (8, 4, 16), (16, 6, 16), (8, 6, 24), (8, 12, 16)]),
    ("mlp_down", 19968, 6656, (4, 2), [(16, 4, 8), (16, 8, 8), (8, 4, 16), (16, 6, 16), (8, 12, 8)]),
]
#: Measured once each, purely to record the rejection text.
BLOCKED = [(8, 20, 8), (8, 26, 8), (8, 32, 8), (16, 8, 16)]
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
        for label, k, n, (sh, sw), cands in CASES:
            torch.manual_seed(0)
            ta = ttnn.from_torch(
                torch.randn(1, 1, ROWS, k).to(torch.bfloat16) * 0.1,
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
                print(f"GROUP {REPS} rows={ROWS} {label} {tag}", flush=True)
                for _ in range(REPS):
                    ttnn.deallocate(ttnn.experimental.minimal_matmul(ta, tb, compute_kernel_config=ck, config=c))
                ttnn.synchronize_device(mesh)

            def probe(c, tag):
                """One call outside any group, so a rejection cannot truncate a group.

                It still emits a device row when it succeeds, so it is announced
                as its own one-row group to keep the CSV parse aligned.
                """
                print(f"GROUP 1 rows={ROWS} {label} probe_{tag}", flush=True)
                try:
                    ttnn.deallocate(ttnn.experimental.minimal_matmul(ta, tb, compute_kernel_config=ck, config=c))
                    return True
                except Exception as exc:  # noqa: BLE001
                    return exc

            emit("default", None)
            for mb, kb, nb in cands:
                ok = probe(cfg(mb, kb, nb), f"M{mb}_K{kb}_N{nb}")
                if ok is not True:
                    print(
                        f"BLOCKED {label} M{mb} K{kb} N{nb}: {type(ok).__name__}: "
                        f"{' '.join(str(ok).split())[:220]}",
                        flush=True,
                    )
                    continue
                emit(f"M{mb}_K{kb}_N{nb}", cfg(mb, kb, nb))
                emit("default", None)
            for mb, kb, nb in BLOCKED:
                print(f"GROUP 1 rows={ROWS} {label} blockedprobe_M{mb}_K{kb}_N{nb}", flush=True)
                try:
                    ttnn.deallocate(
                        ttnn.experimental.minimal_matmul(ta, tb, compute_kernel_config=ck, config=cfg(mb, kb, nb))
                    )
                    print(f"UNBLOCKED {label} M{mb} K{kb} N{nb} ran after all", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"BLOCKED {label} M{mb} K{kb} N{nb}: {type(exc).__name__}: {exc}", flush=True)
            for t in (ta, tb):
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
