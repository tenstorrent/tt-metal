# SPDX-License-Identifier: Apache-2.0
"""Full MD MLP flow device-kernel timing (all 9 ops: 2×shard_a, gate/up/down matmul_decode,
gelu, multiply, 2×S2I). Mirrors exactly what GemmaMLPTTNN does with PI0_MD_DENOISE=1.

Run under tracy:
  python -m tracy -p -r -n mlp_full_md32 _bench_mlp_full_flow_tracy.py --config md32 --device-id 23
  python -m tracy -p -r -n mlp_full_md16 _bench_mlp_full_flow_tracy.py --config md16 --device-id 23

Ops captured per replay:
  0  to_memory_config  x→a (I→Wsh 2c)       shard activation for gate/up
  1  MatmulDecode      gate (1024→4096)
  2  Gelu              (fast, in-place sharded)
  3  MatmulDecode      up   (1024→4096)
  4  multiply          gate_gelu * up (sharded)
  5  sharded_to_interleaved  h→h_il
  6  to_memory_config  h_il→a_down (I→Wsh 2c)  shard for down
  7  MatmulDecode      down (4096→1024)
  8  sharded_to_interleaved  out→out_il
"""
import argparse
import torch
import ttnn

MLP_SHAPES = [("gate", 1024, 4096), ("up", 1024, 4096), ("down", 4096, 1024)]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["md32", "md16"], required=True)
    ap.add_argument("--device-id", type=int, default=23)
    args = ap.parse_args()

    dev = ttnn.CreateDevice(device_id=args.device_id, l1_small_size=24576, trace_region_size=134_217_728)
    grid = dev.compute_with_storage_grid_size()
    maxc = grid.x * grid.y

    m_tile = 32 if args.config == "md32" else 16
    M = m_tile  # activation height matches tile height
    tile = ttnn.Tile((m_tile, 32))

    # Build weights (L1 width-sharded)
    weights = {}
    partials = {}
    for nm, K, N in MLP_SHAPES:
        nc = _n_cores(N, maxc)
        partials[nm] = nc < N // 32
        cfg = _wsh_cfg(grid, K, N, nc)
        weights[nm] = ttnn.from_torch(
            torch.randn(K, N).bfloat16(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=cfg
        )

    # Activation shard config (2 cores, width-sharded)
    def a_wsh_cfg(K):
        return _wsh_cfg(grid, M, K, 2)

    # Input activation (interleaved L1) — used as source for shard_a
    x_il = ttnn.from_torch(
        torch.randn(M, 1024).bfloat16(),
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        device=dev,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    def run():
        # op 0: shard x for gate/up
        _a = ttnn.to_memory_config(x_il, a_wsh_cfg(1024))
        # op 1: gate matmul_decode
        _gate = ttnn.matmul_decode(_a, weights["gate"], partial_width_sharded=partials["gate"])
        # op 2: gelu (fast, in-place on sharded output)
        _gate = ttnn.gelu(_gate, fast_and_approximate_mode=True, memory_config=_gate.memory_config())
        # op 3: up matmul_decode
        _up = ttnn.matmul_decode(_a, weights["up"], partial_width_sharded=partials["up"])
        ttnn.deallocate(_a)
        # op 4: multiply gate_gelu * up (stays sharded)
        _h = ttnn.multiply(_gate, _up, memory_config=_gate.memory_config())
        ttnn.deallocate(_gate)
        ttnn.deallocate(_up)
        # op 5: S2I — interleaved L1 output (feeds down shard_a)
        _h_il = ttnn.sharded_to_interleaved(_h, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(_h)
        # op 6: shard h_il for down
        _a_down = ttnn.to_memory_config(_h_il, a_wsh_cfg(4096))
        ttnn.deallocate(_h_il)
        # op 7: down matmul_decode
        _out = ttnn.matmul_decode(_a_down, weights["down"], partial_width_sharded=partials["down"])
        ttnn.deallocate(_a_down)
        # op 8: S2I — final interleaved output
        _out_il = ttnn.sharded_to_interleaved(_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(_out)
        return _out_il

    # warmup
    run()
    ttnn.synchronize_device(dev)

    # trace capture
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    run()
    ttnn.end_trace_capture(dev, tid, cq_id=0)

    # warmup replays
    for _ in range(3):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)

    # profiling replay
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)

    ttnn.release_trace(dev, tid)
    ttnn.CloseDevice(dev)
    print(f"[done] config={args.config} m_tile={m_tile} device={args.device_id}")


if __name__ == "__main__":
    main()
