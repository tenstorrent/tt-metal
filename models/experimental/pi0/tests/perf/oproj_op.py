"""SigLIP attention output-projection (O-proj) matmul — Python wrapper.

Same kernel as QKV + FC1 (qkv_matmul_kernel.cpp — encoder-shape matmul with
matmul_block, SUBBLOCK_H=1). Shape: M=256, K=1152, N=1152. Tile-aligned.

Decomposition: 36 cores × 1 N-tile-per-core = 36 N-tiles ✓.
Core grid: 6×6 (reuse QKV layout).
"""
import ttnn

from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor


class SigLIPOprojMatmulOp:
    KERNEL_SOURCE = "models/experimental/pi0/tests/perf/qkv_matmul_kernel.cpp"

    M = 256
    K = 1152
    N = 1152
    TILE = 32
    M_TILES = M // TILE  # 8
    K_TILES = K // TILE  # 36
    N_TILES = N // TILE  # 36

    @staticmethod
    def op(activation_tensor, weight_tensor, output_tensor, num_cores: int = 36, math_fidelity=ttnn.MathFidelity.HiFi2):
        n_tiles_per_core = SigLIPOprojMatmulOp.N_TILES // num_cores
        assert (
            n_tiles_per_core * num_cores == SigLIPOprojMatmulOp.N_TILES
        ), f"num_cores ({num_cores}) must evenly divide N_TILES ({SigLIPOprojMatmulOp.N_TILES})"

        act_cb, weights_cb, out_cb = 0, 1, 2
        act_tiles = SigLIPOprojMatmulOp.M_TILES * SigLIPOprojMatmulOp.K_TILES  # 288
        weights_tiles = SigLIPOprojMatmulOp.K_TILES * n_tiles_per_core  # 36

        ncrisc_named_compile_time_args = [
            ("act_cb", act_cb),
            ("weights_cb", weights_cb),
            ("act_tiles", act_tiles),
            ("weights_tiles", weights_tiles),
        ]
        trisc_named_compile_time_args = [
            ("act_cb", act_cb),
            ("weights_cb", weights_cb),
            ("out_cb", out_cb),
            ("m_tiles", SigLIPOprojMatmulOp.M_TILES),
            ("k_tiles", SigLIPOprojMatmulOp.K_TILES),
            ("n_tiles_per_core", n_tiles_per_core),
            ("act_tiles", act_tiles),
            ("weights_tiles", weights_tiles),
        ]
        core_grid = output_tensor.memory_config().shard_spec.grid

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source=SigLIPOprojMatmulOp.KERNEL_SOURCE,
            core_ranges=core_grid,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            brisc_named_compile_time_args=[],
            trisc_named_compile_time_args=trisc_named_compile_time_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=math_fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=True,
            ),
        )
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=[
                ttnn.cb_descriptor_from_sharded_tensor(act_cb, activation_tensor),
                ttnn.cb_descriptor_from_sharded_tensor(weights_cb, weight_tensor),
                ttnn.cb_descriptor_from_sharded_tensor(out_cb, output_tensor),
            ],
            semaphores=[],
        )
        ttnn.generic_op([activation_tensor, weight_tensor, output_tensor], program_descriptor)
        return output_tensor


def build_tensors_for_oproj_test(device, w_torch, x_torch, num_cores=36):
    import torch

    M, K, N = SigLIPOprojMatmulOp.M, SigLIPOprojMatmulOp.K, SigLIPOprojMatmulOp.N
    assert x_torch.shape == (M, K)
    assert w_torch.shape == (K, N)

    grid_cols = 6
    grid_rows = num_cores // grid_cols
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_cols - 1, grid_rows - 1))})

    act_per_core = x_torch.unsqueeze(0).repeat(num_cores, 1, 1).reshape(num_cores * M, K)
    act_shard = ttnn.ShardSpec(core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR)
    act_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, act_shard)
    activation_tt = ttnn.from_torch(
        act_per_core,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=act_mem,
    )

    n_per_core = N // num_cores  # 1152/36 = 32
    w_shard = ttnn.ShardSpec(core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    w_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, w_shard)
    weight_tt = ttnn.from_torch(
        w_torch,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=w_mem,
    )

    out_shard = ttnn.ShardSpec(core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard)
    output_tt = ttnn.from_torch(
        torch.zeros(M, N, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem,
    )

    return activation_tt, weight_tt, output_tt
