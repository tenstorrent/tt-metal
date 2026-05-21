#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Tile-sparse matmul benchmark: measures latency and DRAM bandwidth savings
# from the K-block zero-fill path added in Step 8.
#
# Usage:
#   python tests/ttnn/benchmark/python/tile_sparse_matmul_benchmark.py
#   python tests/ttnn/benchmark/python/tile_sparse_matmul_benchmark.py --shape 512,512,512
#   python tests/ttnn/benchmark/python/tile_sparse_matmul_benchmark.py --sparsity 0.75
#
# Sparsity model used here: uniform K-row sparsity — a fraction of K-tile rows
# in B are entirely zero.  This is the best-case scenario for the current
# implementation because:
#   - global_k_active_mask bits for inactive K-rows = 0  → sender skips A reads
#   - per_core_k_masks bits for inactive K-rows = 0       → writer skips B reads

import argparse
import time

import torch
import ttnn

TILE_SIZE = 32
TILE_BYTES_BF16 = TILE_SIZE * TILE_SIZE * 2  # 2 048 bytes per bf16 tile


# ── helpers ──────────────────────────────────────────────────────────────────


def _median(lst):
    s = sorted(lst)
    return s[len(s) // 2]


def make_k_row_sparse_mask(k_tiles: int, n_tiles: int, sparsity: float, seed: int = 0) -> torch.Tensor:
    """Return a uint8 mask [k_tiles, n_tiles] where `sparsity` fraction of K-rows are 0."""
    torch.manual_seed(seed)
    mask = torch.ones((k_tiles, n_tiles), dtype=torch.uint8)
    n_sparse = int(round(k_tiles * sparsity))
    if n_sparse:
        mask[torch.randperm(k_tiles)[:n_sparse], :] = 0
    return mask


def apply_mask_to_b(b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out K-tile rows of b that are masked off."""
    k_tiles = b.shape[0] // TILE_SIZE
    b_out = b.clone()
    for ki in range(k_tiles):
        if mask[ki, 0].item() == 0:
            b_out[ki * TILE_SIZE : (ki + 1) * TILE_SIZE, :] = 0.0
    return b_out


# ── core benchmark ────────────────────────────────────────────────────────────


def run_one(device, m: int, k: int, n: int, sparsity: float, iters: int, warmup: int) -> dict:
    """Benchmark dense vs sparse matmul for given shape and sparsity."""
    torch.manual_seed(42)
    a_th = torch.randn((m, k), dtype=torch.bfloat16)
    b_th = torch.randn((k, n), dtype=torch.bfloat16)

    k_tiles = k // TILE_SIZE
    n_tiles = n // TILE_SIZE
    mask = make_k_row_sparse_mask(k_tiles, n_tiles, sparsity)
    b_sparse_th = apply_mask_to_b(b_th, mask)

    active_k_rows = int(mask[:, 0].sum().item())
    active_fraction = active_k_rows / k_tiles

    # device tensors
    a_t = ttnn.from_torch(
        a_th, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    b_dense_t = ttnn.from_torch(
        b_th, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    b_sparse_t = ttnn.from_torch(
        b_sparse_th, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    mask_t = ttnn.from_torch(mask, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT)

    # ── dense warmup + time ──
    for _ in range(warmup):
        ttnn.tile_sparse_matmul(a_t, b_dense_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(device)

    dense_ms_list = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ttnn.tile_sparse_matmul(a_t, b_dense_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(device)
        dense_ms_list.append((time.perf_counter() - t0) * 1e3)

    # ── sparse warmup + time ──
    for _ in range(warmup):
        ttnn.tile_sparse_matmul(a_t, b_sparse_t, sparsity_mask_b=mask_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(device)

    sparse_ms_list = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ttnn.tile_sparse_matmul(a_t, b_sparse_t, sparsity_mask_b=mask_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(device)
        sparse_ms_list.append((time.perf_counter() - t0) * 1e3)

    dense_ms = _median(dense_ms_list)
    sparse_ms = _median(sparse_ms_list)

    flops = 2 * m * k * n
    dense_gf = flops / (dense_ms * 1e-3) / 1e9
    sparse_gf = flops / (sparse_ms * 1e-3) / 1e9

    # estimated DRAM reads (tiles × bytes/tile)
    # dense: all Mt*Kt A-tiles + all Kt*Nt B-tiles per dispatch
    # sparse: active_fraction of each
    a_tiles = (m // TILE_SIZE) * k_tiles
    b_tiles = k_tiles * n_tiles
    bytes_dense = (a_tiles + b_tiles) * TILE_BYTES_BF16
    bytes_sparse = (a_tiles + b_tiles) * active_fraction * TILE_BYTES_BF16
    saved_pct = 100.0 * (1.0 - bytes_sparse / bytes_dense)

    return dict(
        m=m,
        k=k,
        n=n,
        sparsity=sparsity,
        active_k_rows=active_k_rows,
        k_tiles=k_tiles,
        active_fraction=active_fraction,
        dense_ms=dense_ms,
        sparse_ms=sparse_ms,
        speedup=dense_ms / sparse_ms,
        dense_gflops=dense_gf,
        sparse_gflops=sparse_gf,
        est_bytes_dense=bytes_dense,
        est_bytes_sparse=bytes_sparse,
        est_dram_saved_pct=saved_pct,
    )


# ── display ───────────────────────────────────────────────────────────────────


def print_result(r: dict, verbose: bool = True):
    tag = f"({r['m']:4d},{r['k']:4d},{r['n']:4d}) sp={r['sparsity']:.0%}"
    print(
        f"  {tag}  dense={r['dense_ms']:6.2f}ms  sparse={r['sparse_ms']:6.2f}ms"
        f"  speedup={r['speedup']:.3f}x  est_DRAM_saved={r['est_dram_saved_pct']:.0f}%"
    )
    if verbose:
        print(
            f"           dense={r['dense_gflops']:.1f} GFLOPS  sparse={r['sparse_gflops']:.1f} GFLOPS"
            f"  active_K={r['active_k_rows']}/{r['k_tiles']}"
        )


def print_table(results: list[dict]):
    """Print a summary table."""
    print()
    print(f"{'Shape':>22}  {'sparsity':>8}  {'dense ms':>9}  {'sparse ms':>9}  " f"{'speedup':>8}  {'DRAM saved':>10}")
    print("-" * 82)
    for r in results:
        shape = f"({r['m']},{r['k']},{r['n']})"
        print(
            f"{shape:>22}  {r['sparsity']:>8.0%}  {r['dense_ms']:>9.2f}  {r['sparse_ms']:>9.2f}  "
            f"{r['speedup']:>8.3f}x  {r['est_dram_saved_pct']:>9.0f}%"
        )


# ── main ──────────────────────────────────────────────────────────────────────


SWEEP_SHAPES = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]
SWEEP_SPARSITIES = [0.0, 0.25, 0.5, 0.75, 0.90]


def main():
    parser = argparse.ArgumentParser(description="Tile-sparse matmul benchmark")
    parser.add_argument("--shape", default=None, help="Single M,K,N shape (default: sweep standard shapes)")
    parser.add_argument(
        "--sparsity", type=float, default=None, help="Single sparsity (default: sweep 0%%,25%%,...,90%%)"
    )
    parser.add_argument("--iterations", type=int, default=100, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations (JIT compile)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run verbose output")
    args = parser.parse_args()

    shapes = [tuple(int(x) for x in args.shape.split(","))] if args.shape else SWEEP_SHAPES
    sparsities = [args.sparsity] if args.sparsity is not None else SWEEP_SPARSITIES

    device = ttnn.open_device(device_id=0)
    print(f"=== Tile-Sparse Matmul Benchmark ===")
    print(f"iterations={args.iterations}  warmup={args.warmup}")
    print(f"Note: uses K-row sparsity (entire K-tile rows of B zeroed)\n")

    all_results = []
    try:
        for m, k, n in shapes:
            k_tiles = k // TILE_SIZE
            if k_tiles > 32:
                print(f"  SKIP ({m},{k},{n}): Kt={k_tiles} > 32 — sparse kernels require num_k_blocks ≤ 32")
                continue
            for sp in sparsities:
                r = run_one(device, m, k, n, sp, args.iterations, args.warmup)
                print_result(r, verbose=not args.quiet)
                all_results.append(r)
    finally:
        ttnn.close_device(device)

    if len(all_results) > 1:
        print_table(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
