import argparse

import ttnn
import torch

# Matrix dimensions to benchmark
MATRIX_SIZES = [
    128,
    256,
    512,
    1024,
    # 2048,
    # 4096,
    # 8192,
]

# Data types to benchmark (name, ttnn dtype) from ttnn.types float32/bfloat16/bfloat8_b/bfloat4_b
DATA_TYPES = [
    ("bfloat4_b", ttnn.bfloat4_b),
    ("bfloat8_b", ttnn.bfloat8_b),
    ("bfloat16", ttnn.bfloat16),
    ("float32", ttnn.float32),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="NextAI TTNN matmul benchmark")
    parser.add_argument(
        "--cores",
        type=int,
        default=1,
        help="The number you specify here will be nxn core grid",
    )
    parser.add_argument(
        "--memory_config",
        type=str,
        default="l1",
        choices=["l1", "dram"],
        help="Determines which memory will be used (L1 or DRAM)",
    )
    return parser.parse_args()


def run_benchmark_for_size(
    device, size, core_grid, dtype, memory_config, weights_memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    """Run one benchmark for a given matrix size and dtype; returns (a, b, c) tensors for validation."""
    m = k = n = size
    core_size = core_grid.x
    print("Memory Config = ", memory_config)
    a = ttnn.from_torch(
        torch.rand(m, k),
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
    )
    weights_data = torch.rand(k, n).repeat(
        core_size * core_size, 1
    )  # Replicate weights across cores for better performance.
    print("Weights Memory Config = ", weights_memory_config, " with shape ", weights_data.shape)
    b = ttnn.from_torch(
        weights_data,
        dtype=dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=weights_memory_config,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
    )
    config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(core_grid.x, core_grid.y),
        in0_block_w=((k // 32) // core_size),
        out_subblock_h=1,
        out_subblock_w=min(4, ((n // 32) // core_size)),
        per_core_M=((m // 32) // core_size),
        per_core_N=((n // 32) // core_size),
        transpose_mcast=False,
    )
    c = ttnn.experimental.fast_matmul(a, b, dtype=dtype)
    ttnn.deallocate(c)
    c = ttnn.experimental.fast_matmul(a, b, dtype=dtype)
    ttnn.deallocate(c)
    ttnn.synchronize_device(device)
    return (a, b, c)


def main():
    args = parse_args()

    core_grid = ttnn.CoreGrid(y=args.cores, x=args.cores)
    memory_config = ttnn.DRAM_MEMORY_CONFIG

    device_id = 0
    device = ttnn.open_device(device_id=device_id)
    for dtype_name, dtype in DATA_TYPES:
        for size in MATRIX_SIZES:
            try:
                print(
                    f"Benchmarking: dtype({dtype_name}) Matrix({size}x{size}x{size}) cores={args.cores}x{args.cores} memory={args.memory_config}"
                )
                if args.memory_config == "l1":
                    memory_config = ttnn.create_sharded_memory_config(
                        shape=(size // args.cores, size // args.cores),
                        core_grid=core_grid,
                        strategy=ttnn.ShardStrategy.HEIGHT,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )
                    weights_memory_config = ttnn.create_sharded_memory_config(
                        shape=(size, size),  # Replicate weights across cores for better performance.
                        core_grid=core_grid,
                        strategy=ttnn.ShardStrategy.HEIGHT,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )
                run_benchmark_for_size(
                    device,
                    size,
                    core_grid,
                    dtype=dtype,
                    memory_config=memory_config,
                    weights_memory_config=weights_memory_config,
                )
            except RuntimeError:
                print(f"  -> OOM (RuntimeError), skipped")
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
