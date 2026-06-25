# SPDX-License-Identifier: Apache-2.0
"""MLP matmul A/B: reference 1D-mcast (M=32) vs matmul_decode 32x32 (M=32) vs
matmul_decode 16x32 (M=16). 2 layers/chip, ALL weights resident in L1 at init.

MLP shapes (Gemma-300M denoise expert): gate 1024->4096, up 1024->4096, down 4096->1024.
Weights bf8_b (model dtype). Per-config trace runs all 6 matmuls (3 shapes x 2 layers)."""
import argparse
import statistics
import time

import torch
import ttnn

N_WARMUP, N_ITER = 3, 10
LAYERS = 2
# (name, K, N)
MLP_SHAPES = [("gate", 1024, 4096), ("up", 1024, 4096), ("down", 4096, 1024)]


def b_l1_interleaved(device, k, n):
    return ttnn.from_torch(
        torch.randn(k, n).bfloat16(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def b_l1_dram(device, k, n):
    """Reference model config: DRAM-interleaved B (the 1D-mcast path reads from DRAM)."""
    return ttnn.from_torch(
        torch.randn(k, n).bfloat16(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def b_width_sharded(device, k, n):
    """Full-width matmul_decode B: width-shard over n//32 cores (needs n//32 <= grid)."""
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    nbc = n // 32
    while nbc > max_cores or n % (nbc * 32) != 0:
        nbc //= 2
    cfg = ttnn.create_sharded_memory_config(
        (k, n // nbc),
        core_grid=ttnn.num_cores_to_corerangeset(nbc, grid, True),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.from_torch(
        torch.randn(k, n).bfloat16(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg
    )


def b_partial(device, k, n, k_blocks, n_blocks):
    """Partial-width matmul_decode B (K-split): reshape [k_blocks,kc,n]->permute->[kc, n*k_blocks],
    width-shard over k_blocks*n_blocks cores, each holding (kc, nc). For wide N (>120 tiles)."""
    kc, nc = k // k_blocks, n // n_blocks
    num = k_blocks * n_blocks
    bt = torch.randn(k, n).bfloat16()
    br = bt.reshape(k_blocks, kc, n).permute(1, 0, 2).reshape(kc, n * k_blocks)
    grid = device.compute_with_storage_grid_size()
    cfg = ttnn.create_sharded_memory_config(
        (kc, nc),
        core_grid=ttnn.num_cores_to_corerangeset(num, grid, True),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.from_torch(br, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)


# matmul_decode mode per shape: full-width if N//32 <= 120, else partial (K-split).
# gate/up N=4096 -> 128 tiles > 120 -> partial (k_blocks=2, n_blocks=32 => 64 cores).
# down  N=1024 -> 32 tiles -> full-width.
def md_is_partial(n):
    return (n // 32) > 120


def a_l1(device, m, k):
    return ttnn.from_torch(
        torch.randn(m, k).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )


def a_width_sharded(device, m, k, tile):
    grid = device.compute_with_storage_grid_size()
    cfg = ttnn.create_sharded_memory_config(
        (m, k // 2),
        core_grid=ttnn.num_cores_to_corerangeset(2, grid, True),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.from_torch(
        torch.randn(m, k).bfloat16(), layout=ttnn.TILE_LAYOUT, tile=tile, device=device, memory_config=cfg
    )


def ref_pcfg(device, m, k, n):
    grid = device.compute_with_storage_grid_size()
    total = grid.x * grid.y
    m_t, k_t, n_t = (m + 31) // 32, k // 32, n // 32
    nc = min(n_t, total)
    while n_t % nc != 0:  # need num_cores to divide n_tiles cleanly
        nc -= 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        in0_block_w=k_t,
        per_core_M=m_t,
        per_core_N=n_t // nc,
        out_subblock_h=1,
        out_subblock_w=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def time_trace(device, run_fn):
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    run_fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    for _ in range(N_WARMUP):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    times = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        times.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(device, tid)
    return statistics.mean(times), min(times)


def bench_shape(dev, cfg, k, n):
    """Time a 2-layer (2-matmul) trace for one shape under one config.
    cfg in {'ref','md32','md16'}. Weights for both layers resident in L1 (md) /
    DRAM (ref = model's actual config) at init; freed by caller scope."""
    if cfg == "ref":
        w = [b_l1_dram(dev, k, n) for _ in range(LAYERS)]
        a = a_l1(dev, 32, k)
        pc = ref_pcfg(dev, 32, k, n)
        run = lambda: [
            ttnn.linear(a, w[l], program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG) for l in range(LAYERS)
        ]
    else:
        m = 32 if cfg == "md32" else 16
        tile = ttnn.Tile((m, 32))
        partial = md_is_partial(n)
        w = [(b_partial(dev, k, n, 1, 64) if partial else b_width_sharded(dev, k, n)) for _ in range(LAYERS)]
        a = a_width_sharded(dev, m, k, tile)
        run = lambda: [ttnn.matmul_decode(a, w[l], partial_width_sharded=partial) for l in range(LAYERS)]
    avg, mn = time_trace(dev, run)
    for t in w:
        ttnn.deallocate(t)
    ttnn.deallocate(a)
    return avg, mn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=1)
    args = ap.parse_args()
    dev = ttnn.CreateDevice(device_id=args.device_id, l1_small_size=24576, trace_region_size=134_217_728)

    print(f"\n2 layers/chip, MLP, bf8_b weights L1-resident@init (ref=DRAM model cfg), device {args.device_id}")
    print(f"{'shape':<18}{'reference':>12}{'md 32x32':>12}{'md 16x32':>12}   notes")
    print("=" * 72)
    tot = {"ref": 0.0, "md32": 0.0, "md16": 0.0}
    for nm, k, n in MLP_SHAPES:
        row = {}
        for cfg in ("ref", "md32", "md16"):
            try:
                avg, _ = bench_shape(dev, cfg, k, n)
                row[cfg] = avg
                tot[cfg] += avg
            except Exception as e:
                row[cfg] = None
                print(f"    [{nm} {cfg} SKIP: {str(e)[:60]}]")
        note = "partial K-split (N=4096>120c)" if md_is_partial(n) else "full-width (32c)"

        def f(v):
            return f"{v:9.1f}us" if v is not None else "    SKIP "

        print(f"{nm+f' ({k}x{n})':<18}{f(row['ref']):>12}{f(row['md32']):>12}{f(row['md16']):>12}   {note}")
    print("=" * 72)
    print(f"{'TOTAL (2 layers)':<18}{tot['ref']:9.1f}us{tot['md32']:11.1f}us{tot['md16']:11.1f}us")
    if tot["ref"] > 0:
        print(f"  vs reference:  md32 = {tot['ref']/tot['md32']:.2f}x   md16 = {tot['ref']/tot['md16']:.2f}x")
    print("=" * 72)
    ttnn.CloseDevice(dev)


if __name__ == "__main__":
    main()
