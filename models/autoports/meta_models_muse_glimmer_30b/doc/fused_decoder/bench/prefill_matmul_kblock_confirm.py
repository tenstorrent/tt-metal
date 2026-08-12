# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Confirm the ``K_block``/``N_block`` candidates from the sweep, A/B interleaved.

``prefill_matmul_kblock_probe.py`` measures every config once, in sequence, so a
1-3 % gap is the same size as the drift between the shipped default measured at
the top of a shape block and the same default measured again lower down (2.339
vs 2.333 ms on wqkv).  This re-measures only the surviving candidates against
the shipped no-``config=`` default, alternating A/B/A/B inside one loop so both
see the same thermal and dispatch conditions, over more rounds, at both prefill
chunk heights ``_dense`` can see (a full 8192-row chunk and a 4096-row tail;
``MINIMAL_MATMUL_MIN_ROWS = 3072`` gates anything shorter back to
``ttnn.linear``).

A candidate is only shipped if it wins at both heights by more than the paired
default-vs-default noise band this script also measures.
"""

from __future__ import annotations

import time

import torch

import ttnn
from models.common.utility_functions import comp_pcc

ROWS = [8192, 4096]
#: (label, K, N, default_subblocks, [(M_block, K_block, N_block), ...])
CASES = [
    ("wqkv    ", 6656, 4608, (4, 2), [(8, 13, 8), (8, 6, 8)]),
    ("o_proj  ", 4096, 6656, (4, 2), [(8, 13, 8), (16, 4, 8), (8, 4, 8)]),
    ("mlp_gate", 6656, 19968, (2, 4), [(8, 6, 16), (16, 6, 8), (8, 13, 8)]),
    ("mlp_down", 19968, 6656, (4, 2), [(8, 6, 8), (8, 16, 8), (8, 6, 16)]),
]
ITERS = 4
ROUNDS = 5


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
                a = torch.randn(1, 1, m, k).to(torch.bfloat16) * 0.1
                b = torch.randn(1, 1, k, n).to(torch.bfloat16) * 0.02
                ta = ttnn.from_torch(
                    a, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                tb = ttnn.from_torch(
                    b, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                ref = a.float() @ b.float()

                def default_fn():
                    return ttnn.experimental.minimal_matmul(ta, tb, compute_kernel_config=ck)

                def cand_fn(mb, kb, nb):
                    return ttnn.experimental.minimal_matmul(
                        ta,
                        tb,
                        compute_kernel_config=ck,
                        config=ttnn.MinimalMatmulConfig(
                            M_block_size=mb,
                            K_block_size=kb,
                            N_block_size=nb,
                            subblock_h=sh,
                            subblock_w=sw,
                            compute_with_storage_grid_size=full_grid,
                        ),
                    )

                def ab(tag, fn_b):
                    """Alternate default and candidate inside one loop."""
                    try:
                        out = fn_b()
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"AB rows={m} {label} {tag:16s} FAILED {type(exc).__name__}: {str(exc)[:120]}", flush=True
                        )
                        return
                    pcc = comp_pcc(ref, ttnn.to_torch(out).float(), 0.99)[1]
                    ttnn.deallocate(out)
                    ta_ms, tb_ms = [], []
                    for _ in range(ROUNDS):
                        for acc, fn in ((ta_ms, default_fn), (tb_ms, fn_b)):
                            ttnn.synchronize_device(mesh)
                            t0 = time.perf_counter()
                            for _ in range(ITERS):
                                ttnn.deallocate(fn())
                            ttnn.synchronize_device(mesh)
                            acc.append((time.perf_counter() - t0) / ITERS * 1e3)
                    da, db = min(ta_ms), min(tb_ms)
                    print(
                        f"AB rows={m} {label} {tag:16s} default {da:8.3f} ms  cand {db:8.3f} ms  "
                        f"{(da / db - 1) * 100:+6.2f} %  (default spread {max(ta_ms) - da:.3f}, "
                        f"cand spread {max(tb_ms) - db:.3f})  PCC={pcc}",
                        flush=True,
                    )

                # Paired noise band: the same op measured against itself.
                ab("control default", default_fn)
                for mb, kb, nb in cands:
                    ab(f"M{mb} K{kb} N{nb}", lambda mb=mb, kb=kb, nb=nb: cand_fn(mb, kb, nb))
                for t in (ta, tb):
                    ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
