# SPDX-License-Identifier: Apache-2.0
"""Device-kernel duration of every denoise matmul under ttnn.matmul_decode (M=32, L1 weights).
One op per shape, all in one trace -> tracy device-kernel. Compare vs reference MatmulDeviceOperation.
Shapes: modulation, wqkv, o_proj (attention) + gate, up, down (MLP)."""
import argparse
import torch
import ttnn

# (name, K, N)  -- M=32 for all (action-horizon padded)
SHAPES = [
    ("modulation", 1024, 6144),
    ("wqkv", 1024, 2560),
    ("o_proj", 2048, 1024),
    ("gate", 1024, 4096),
    ("up", 1024, 4096),
    ("down", 4096, 1024),
]
M = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=1)
    args = ap.parse_args()
    dev = ttnn.CreateDevice(device_id=args.device_id, l1_small_size=24576, trace_region_size=134_217_728)
    grid = dev.compute_with_storage_grid_size()
    maxc = grid.x * grid.y

    def out_cores(n):  # op caps output at N//32 halved until <= maxc
        c = n // 32
        while c > maxc:
            c //= 2
        return c

    ws, as_, partials = [], [], []
    for nm, k, n in SHAPES:
        n_tiles = n // 32
        partial = n_tiles > maxc
        partials.append(partial)
        if partial:
            n_blocks = out_cores(n)  # k_blocks=1 -> N_blocks = out cores
            kc, nc, num = k, n // n_blocks, n_blocks
            br = torch.randn(k, n).bfloat16()  # k_blocks=1 => reshape is identity
            bcfg = ttnn.create_sharded_memory_config(
                (kc, nc),
                core_grid=ttnn.num_cores_to_corerangeset(num, grid, True),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        else:
            nbc = n // 32
            br = torch.randn(k, n).bfloat16()
            bcfg = ttnn.create_sharded_memory_config(
                (k, n // nbc),
                core_grid=ttnn.num_cores_to_corerangeset(nbc, grid, True),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        ws.append(ttnn.from_torch(br, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=bcfg))
        acfg = ttnn.create_sharded_memory_config(
            (M, k // 2),
            core_grid=ttnn.num_cores_to_corerangeset(2, grid, True),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        as_.append(
            ttnn.from_torch(
                torch.randn(M, k).bfloat16(),
                layout=ttnn.TILE_LAYOUT,
                tile=ttnn.Tile((32, 32)),
                device=dev,
                memory_config=acfg,
            )
        )

    def run():
        for a, w, p in zip(as_, ws, partials):
            ttnn.matmul_decode(a, w, partial_width_sharded=p)

    run()
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    run()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    ttnn.release_trace(dev, tid)
    ttnn.CloseDevice(dev)
    print("[done] shape order:", [s[0] for s in SHAPES], "partial:", partials)


if __name__ == "__main__":
    main()
