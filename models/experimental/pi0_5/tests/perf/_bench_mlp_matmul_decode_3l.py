# SPDX-License-Identifier: Apache-2.0
"""3 MLP layers chained forward pass: gate/gelu/up/mul/down × 3 layers.
All 9 weight tensors (gate+up+down × 3) resident in L1 at init.
MD 32×32 tile. Checks for nan/inf, then times via trace.

MLP shapes (Gemma-300M denoise expert): gate/up 1024->4096, down 4096->1024.
Run:
  python _bench_mlp_matmul_decode_3l.py --device-id 23
"""
import argparse
import statistics
import time

import torch
import ttnn

LAYERS = 3
N_WARMUP, N_ITER = 3, 10
M = 32  # action horizon (32-row tile)
K_HID = 1024  # hidden dim
K_INT = 4096  # intermediate dim (gate/up output, down input)


def _n_cores(n, maxc):
    c = n // 32
    while c > maxc:
        c //= 2
    return c


def _wsh_cfg(grid, rows, cols, nc):
    return ttnn.create_sharded_memory_config(
        (rows, cols // nc),
        core_grid=ttnn.num_cores_to_corerangeset(nc, grid, True),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _a_wsh_cfg(grid, K, nc_a=2):
    return _wsh_cfg(grid, M, K, nc_a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=23)
    args = ap.parse_args()

    dev = ttnn.CreateDevice(device_id=args.device_id, l1_small_size=24576, trace_region_size=134_217_728)
    grid = dev.compute_with_storage_grid_size()
    maxc = grid.x * grid.y
    tile32 = ttnn.Tile((32, 32))

    nc_gu = _n_cores(K_INT, maxc)  # gate/up: N=4096 → 64 cores on P150
    nc_dn = _n_cores(K_HID, maxc)  # down: N=1024 → 32 cores

    # K_blocks=1 for all: each B shard holds the full K height → full_width_sharded factory.
    # partial_width_sharded=True with K_blocks=1 hits an OOB slab read → inf output.
    partial_gu = False
    partial_dn = False

    print(f"grid={grid.x}×{grid.y}={maxc}  nc_gu={nc_gu} partial_gu={partial_gu}  nc_dn={nc_dn}")

    # Pre-load all 3×gate, 3×up, 3×down weights into L1
    def mk_weight_gu(K, N, nc):
        cfg = _wsh_cfg(grid, K, N, nc)
        return ttnn.from_torch(
            torch.randn(K, N).bfloat16(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=cfg
        )

    gate_w = [mk_weight_gu(K_HID, K_INT, nc_gu) for _ in range(LAYERS)]
    up_w = [mk_weight_gu(K_HID, K_INT, nc_gu) for _ in range(LAYERS)]
    down_w = [mk_weight_gu(K_INT, K_HID, nc_dn) for _ in range(LAYERS)]

    # Starting activation (interleaved L1, 32×32 tile)
    x_il = ttnn.from_torch(
        torch.randn(M, K_HID).bfloat16(),
        layout=ttnn.TILE_LAYOUT,
        tile=tile32,
        device=dev,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    cfg_a_gu = _a_wsh_cfg(grid, K_HID)  # [M, K_HID/2] per core, 2 cores
    cfg_a_dn = _a_wsh_cfg(grid, K_INT)  # [M, K_INT/2] per core, 2 cores

    def run():
        """3-layer MLP forward pass. x_il → layer0 → layer1 → layer2 → output."""
        x = x_il
        for l in range(LAYERS):
            _a = ttnn.to_memory_config(x, cfg_a_gu)
            _gate = ttnn.matmul_decode(_a, gate_w[l], partial_width_sharded=partial_gu)
            _gate = ttnn.gelu(_gate, fast_and_approximate_mode=True, memory_config=_gate.memory_config())
            _up = ttnn.matmul_decode(_a, up_w[l], partial_width_sharded=partial_gu)
            ttnn.deallocate(_a)
            _h = ttnn.multiply(_gate, _up, memory_config=_gate.memory_config())
            ttnn.deallocate(_gate)
            ttnn.deallocate(_up)
            _h_il = ttnn.sharded_to_interleaved(_h, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(_h)
            _a_dn = ttnn.to_memory_config(_h_il, cfg_a_dn)
            ttnn.deallocate(_h_il)
            _out = ttnn.matmul_decode(_a_dn, down_w[l], partial_width_sharded=partial_dn)
            ttnn.deallocate(_a_dn)
            x_new = ttnn.sharded_to_interleaved(_out, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(_out)
            if l > 0:
                ttnn.deallocate(x)  # free intermediate (never free x_il)
            x = x_new
        return x

    # Correctness check before timing: verify no nan/inf
    print("Checking forward pass for nan/inf...")
    out = run()
    ttnn.synchronize_device(dev)
    out_t = ttnn.to_torch(out).float()
    has_nan = out_t.isnan().any().item()
    has_inf = out_t.isinf().any().item()
    max_abs = out_t.abs().max().item()
    print(f"  output shape={list(out_t.shape)}  nan={has_nan}  inf={has_inf}  max_abs={max_abs:.4f}")
    if has_nan or has_inf:
        print("FAIL: nan/inf in output")
        for t in gate_w + up_w + down_w:
            ttnn.deallocate(t)
        ttnn.deallocate(x_il)
        ttnn.CloseDevice(dev)
        return
    print("  OK")
    ttnn.deallocate(out)

    # Trace timing
    for _ in range(N_WARMUP):
        ttnn.deallocate(run())
    ttnn.synchronize_device(dev)

    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    _ = run()
    ttnn.end_trace_capture(dev, tid, cq_id=0)

    for _ in range(N_WARMUP):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)

    times = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        times.append((time.perf_counter() - t0) * 1e6)

    ttnn.release_trace(dev, tid)

    avg = statistics.mean(times)
    mn = min(times)
    print(f"\n3-layer MLP forward pass (MD 32×32, L1 weights), device {args.device_id}")
    print(f"  avg={avg:.1f}µs  min={mn:.1f}µs  per-layer avg={avg/LAYERS:.1f}µs")

    for t in gate_w + up_w + down_w:
        ttnn.deallocate(t)
    ttnn.deallocate(x_il)
    ttnn.CloseDevice(dev)


if __name__ == "__main__":
    main()
