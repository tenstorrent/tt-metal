# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Probe: does a larger ``MinimalMatmulConfig.K_block_size`` beat the default?

``prefill_matmul_probe.py`` swept ``M_block``/``N_block`` but only
``K_block in {2, 4}``, while ``determine_default_block_sizes``
(``minimal_matmul_program_factory.cpp:22-42``) hands the shipped no-``config=``
path ``M=K=N=8`` with ``subblock 2x4`` (``4x2`` when ``M > N``).  Every measured
pair in that sweep improved with larger K, so the sweep was truncated while the
K axis was still moving and could not, by construction, beat the default.

This probe closes that gap on the dominant prefill matmul (65 % of fused
prefill device time).  For each real projection shape at the shipped 8192-row
chunk it measures, under the shipped HiFi2 / ``fp32_dest_acc_en=False`` /
``packer_l1_acc=True`` policy:

* the shipped no-``config=`` default, as the control;
* that default replicated explicitly, to prove the config path reproduces it;
* ``K_block`` swept up from 4 to the full K-tile count, at the default
  ``M_block``/``N_block``/subblocks (both legal divisors of the tiled K and
  non-divisors, so any ragged-K blocker is recorded rather than assumed);
* an ``M_block``/``N_block`` cross at the best two K values, since larger K
  raises per-core L1 and may only pay off with a smaller M or N block.

Any config that the op rejects prints its exact error, which is the
op-contract blocker the optimization standard asks for.
"""

from __future__ import annotations

import time

import torch

import ttnn
from models.common.utility_functions import comp_pcc

TILE = 32

#: (label, M, K, N) at the shipped 8192-row prefill chunk.  ``mlp_up`` has the
#: same shape as ``mlp_gate`` and is not measured twice.
SHAPES = [
    ("wqkv    ", 8192, 6656, 4608),
    ("o_proj  ", 8192, 4096, 6656),
    ("mlp_gate", 8192, 6656, 19968),
    ("mlp_down", 8192, 19968, 6656),
]

#: K_block candidates in tiles.  Anything <= the shape's K-tile count is tried;
#: divisors of it are marked ``div`` in the output so the log distinguishes a
#: clean split from a ragged one.  208 = 2^4*13 (K=6656), 128 = 2^7 (K=4096),
#: 624 = 2^4*3*13 (K=19968).
K_BLOCKS = [4, 6, 8, 10, 12, 13, 16, 20, 24, 26, 32, 39, 48, 52, 64, 104, 128]
#: (M_block, N_block) cross applied at the best two K values.
MN_CROSS = [(4, 8), (8, 4), (4, 4), (16, 8), (8, 16), (16, 16)]
ITERS = 3
ROUNDS = 2


def default_subblocks(m, n):
    """``determine_default_block_sizes`` with ``fp32_dest_acc_en=False``."""
    return (2, 4) if n >= m else (4, 2)


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
        print(f"grid {grid.x}x{grid.y}  policy HiFi2 fp32_dest_acc_en=False packer_l1_acc=True", flush=True)
        for label, m, k, n in SHAPES:
            k_tiles = k // TILE
            sh, sw = default_subblocks(m, n)
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
            print(f"\n=== {label} M={m} K={k} ({k_tiles} tiles) N={n}  default subblock {sh}x{sw} ===", flush=True)

            def timed(tag, fn):
                try:
                    out = fn()
                except Exception as exc:  # noqa: BLE001
                    print(f"MM {label} {tag:34s} FAILED {type(exc).__name__}: {str(exc)[:150]}", flush=True)
                    return None
                pcc = comp_pcc(ref, ttnn.to_torch(out).float(), 0.99)[1]
                ttnn.deallocate(out)
                rounds = []
                for _ in range(ROUNDS):
                    ttnn.synchronize_device(mesh)
                    t0 = time.perf_counter()
                    for _ in range(ITERS):
                        ttnn.deallocate(fn())
                    ttnn.synchronize_device(mesh)
                    rounds.append((time.perf_counter() - t0) / ITERS * 1e3)
                dt = min(rounds)
                tflops = 2 * m * k * n / (dt * 1e-3) / 1e12
                print(f"MM {label} {tag:34s} {dt:9.3f} ms  {tflops:7.1f} TFLOPs  PCC={pcc}", flush=True)
                return dt

            def cfg(mb, kb, nb):
                return ttnn.MinimalMatmulConfig(
                    M_block_size=mb,
                    K_block_size=kb,
                    N_block_size=nb,
                    subblock_h=sh,
                    subblock_w=sw,
                    compute_with_storage_grid_size=full_grid,
                )

            base = timed(
                "default no-config (shipped)",
                lambda: ttnn.experimental.minimal_matmul(ta, tb, compute_kernel_config=ck),
            )
            timed(
                "explicit M8 K8 N8 (= default)",
                lambda: ttnn.experimental.minimal_matmul(ta, tb, compute_kernel_config=ck, config=cfg(8, 8, 8)),
            )

            results = {}
            for kb in K_BLOCKS:
                if kb > k_tiles:
                    continue
                mark = "div" if k_tiles % kb == 0 else "rag"
                dt = timed(
                    f"M8 K{kb} N8 [{mark}]",
                    lambda kb=kb: ttnn.experimental.minimal_matmul(
                        ta, tb, compute_kernel_config=ck, config=cfg(8, kb, 8)
                    ),
                )
                if dt is not None:
                    results[kb] = dt

            best_ks = sorted(results, key=results.get)[:2]
            print(f"--- {label} best K_block by time: {best_ks} ---", flush=True)
            for kb in best_ks:
                for mb, nb in MN_CROSS:
                    if mb % sh or nb % sw:
                        continue
                    timed(
                        f"M{mb} K{kb} N{nb}",
                        lambda mb=mb, kb=kb, nb=nb: ttnn.experimental.minimal_matmul(
                            ta, tb, compute_kernel_config=ck, config=cfg(mb, kb, nb)
                        ),
                    )
            if base is not None and results:
                bk = best_ks[0]
                print(
                    f"SUMMARY {label} default {base:.3f} ms vs best K_block={bk} {results[bk]:.3f} ms "
                    f"({(base / results[bk] - 1) * 100:+.2f} %)",
                    flush=True,
                )
            for t in (ta, tb):
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
