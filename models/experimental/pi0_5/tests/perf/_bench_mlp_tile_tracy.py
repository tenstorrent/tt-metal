# SPDX-License-Identifier: Apache-2.0
"""Device-kernel timing for MLP matmul shapes under 3 configs:
  ref   -- model's actual 1D-mcast kernel (DRAM weights, build_matmul_pcfg)
  md32  -- matmul_decode tile 32x32, L1-sharded B
  md16  -- matmul_decode tile 16x32, L1-sharded B

One trace per run (gate->up->down, 1 layer). ReadDeviceProfiler after the
profiling replay. Run under tracy to capture device-kernel durations:

  python -m tracy -p -r <csv> _bench_mlp_tile_tracy.py --config ref  --device-id 23
  python -m tracy -p -r <csv> _bench_mlp_tile_tracy.py --config md32 --device-id 23
  python -m tracy -p -r <csv> _bench_mlp_tile_tracy.py --config md16 --device-id 23

Then parse each CSV:
  python _parse_ops_perop.py <csv>
"""
import argparse
import torch
import ttnn

from models.experimental.pi0_5.tt.ttnn_gemma import build_matmul_pcfg

# Gemma-300M denoise MLP shapes: (name, K, N)
MLP_SHAPES = [("gate", 1024, 4096), ("up", 1024, 4096), ("down", 4096, 1024)]
M = 32  # padded action horizon


def _md_n_cores(n, maxc):
    c = n // 32
    while c > maxc:
        c //= 2
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["ref", "md32", "md16"], required=True)
    ap.add_argument("--device-id", type=int, default=23)
    args = ap.parse_args()

    dev = ttnn.CreateDevice(device_id=args.device_id, l1_small_size=24576, trace_region_size=134_217_728)
    grid = dev.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    maxc = gx * gy

    cfg = args.config
    m_tiles = M // 32  # = 1

    ws, acts = [], []

    if cfg == "ref":
        # DRAM-interleaved weights, model's actual build_matmul_pcfg (1D mcast for M=1).
        # Use ttnn.matmul with pre-allocated outputs so trace captures real work.
        pcfgs = []
        outs = []
        for nm, K, N in MLP_SHAPES:
            k_t, n_t = K // 32, N // 32
            act_kw = {"activation": (ttnn.UnaryOpType.GELU, True)} if nm == "gate" else {}
            pc = build_matmul_pcfg(m_tiles, k_t, n_t, gx, gy, **act_kw)
            pcfgs.append(pc)
            w = ttnn.from_torch(
                torch.randn(K, N).bfloat16(),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ws.append(w)
            outs.append(
                ttnn.from_torch(
                    torch.zeros(M, N).bfloat16(),
                    layout=ttnn.TILE_LAYOUT,
                    device=dev,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            )
        a = ttnn.from_torch(
            torch.randn(M, MLP_SHAPES[0][1]).bfloat16(),
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        a_down = ttnn.from_torch(
            torch.randn(M, MLP_SHAPES[2][0]).bfloat16(),
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )

        def run():
            for i, (nm, K, N) in enumerate(MLP_SHAPES):
                act = a_down if nm == "down" else a
                kw = {"program_config": pcfgs[i]} if pcfgs[i] else {}
                ttnn.linear(
                    act,
                    ws[i],
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=ckc,
                    **kw,
                )

    else:
        m_tile = 32 if cfg == "md32" else 16
        tile = ttnn.Tile((m_tile, 32))

        for nm, K, N in MLP_SHAPES:
            nc = _md_n_cores(N, maxc)
            partial = nc < N // 32
            if partial:
                # K-split: k_blocks=1, reshape identity; n_blocks=nc
                bcfg = ttnn.create_sharded_memory_config(
                    (K, N // nc),
                    core_grid=ttnn.num_cores_to_corerangeset(nc, grid, True),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
            else:
                bcfg = ttnn.create_sharded_memory_config(
                    (K, N // nc),
                    core_grid=ttnn.num_cores_to_corerangeset(nc, grid, True),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
            w = ttnn.from_torch(
                torch.randn(K, N).bfloat16(),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=bcfg,
            )
            ws.append(w)

        # One activation tensor per K size (gate/up share K=1024, down has K=4096)
        def make_a(K):
            acfg = ttnn.create_sharded_memory_config(
                (m_tile, K // 2),
                core_grid=ttnn.num_cores_to_corerangeset(2, grid, True),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            return ttnn.from_torch(
                torch.randn(m_tile, K).bfloat16(), layout=ttnn.TILE_LAYOUT, tile=tile, device=dev, memory_config=acfg
            )

        a_gate_up = make_a(1024)
        a_down_t = make_a(4096)
        acts = [a_gate_up, a_down_t]

        partials = [_md_n_cores(N, maxc) < N // 32 for _, _, N in MLP_SHAPES]

        def run():
            for i, (nm, K, N) in enumerate(MLP_SHAPES):
                a = a_down_t if nm == "down" else a_gate_up
                ttnn.matmul_decode(a, ws[i], partial_width_sharded=partials[i])

    # warmup
    for _ in range(5):
        run()
    ttnn.synchronize_device(dev)

    if cfg == "ref":
        # ttnn.linear inside a trace yields no device profiler data; run directly.
        run()
        ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)
    else:
        # capture trace
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
    print(f"[done] config={cfg} device={args.device_id}")


if __name__ == "__main__":
    main()
