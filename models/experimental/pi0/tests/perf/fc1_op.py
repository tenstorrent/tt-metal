"""SigLIP MLP FC1 matmul micro-op — Python wrapper.

Same kernel as QKV (qkv_matmul_kernel.cpp — encoder-shape matmul with
matmul_block, SUBBLOCK_H=1, validated to PCC 0.999989, 23.6 µs at QKV shape).
Just different shapes + core grid.

Shape: M=256, K=1152, N=4304 → pad N to 4320 (next 32-multiple = 135 tiles).
Decomposition: 27 cores × 5 N-tiles-per-core = 135 N-tiles ✓.
Core grid: 9 cols × 3 rows = 27 cores (fits in BH's 14×10 logical Tensix grid).

Activation HEIGHT_SHARDED with FULL K replicated × 27 cores (same pattern as QKV).
Weight WIDTH_SHARDED with shard (K=1152, n_per_core=160 padded cols).
Output WIDTH_SHARDED matching the weight partition.
"""
import ttnn

from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor


class SigLIPFC1MatmulOp:
    """SigLIP MLP FC1 (hidden→intermediate) on a single chip with L1-resident bfp8 weights."""

    KERNEL_SOURCE = "models/experimental/pi0/tests/perf/qkv_matmul_kernel.cpp"

    # SigLIP-So400m/14 MLP FC1
    M = 256
    K = 1152
    N_LOGICAL = 4304  # actual cols
    N_PADDED = 4320  # padded to next multiple of 32 for tile alignment
    TILE = 32
    M_TILES = M // TILE  # 8
    K_TILES = K // TILE  # 36
    N_TILES = N_PADDED // TILE  # 135

    @staticmethod
    def op(
        activation_tensor,
        weight_tensor,
        output_tensor,
        num_cores: int = 27,
        math_fidelity=ttnn.MathFidelity.HiFi2,
    ):
        n_tiles_per_core = SigLIPFC1MatmulOp.N_TILES // num_cores
        assert n_tiles_per_core * num_cores == SigLIPFC1MatmulOp.N_TILES, (
            f"num_cores ({num_cores}) must evenly divide N_TILES " f"({SigLIPFC1MatmulOp.N_TILES})"
        )

        act_cb = 0
        weights_cb = 1
        out_cb = 2

        act_tiles = SigLIPFC1MatmulOp.M_TILES * SigLIPFC1MatmulOp.K_TILES  # 8 * 36 = 288
        weights_tiles = SigLIPFC1MatmulOp.K_TILES * n_tiles_per_core  # 36 * 5 = 180

        ncrisc_named_compile_time_args = [
            ("act_cb", act_cb),
            ("weights_cb", weights_cb),
            ("act_tiles", act_tiles),
            ("weights_tiles", weights_tiles),
        ]
        brisc_named_compile_time_args = []
        trisc_named_compile_time_args = [
            ("act_cb", act_cb),
            ("weights_cb", weights_cb),
            ("out_cb", out_cb),
            ("m_tiles", SigLIPFC1MatmulOp.M_TILES),
            ("k_tiles", SigLIPFC1MatmulOp.K_TILES),
            ("n_tiles_per_core", n_tiles_per_core),
            ("act_tiles", act_tiles),
            ("weights_tiles", weights_tiles),
        ]

        core_grid = output_tensor.memory_config().shard_spec.grid

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source=SigLIPFC1MatmulOp.KERNEL_SOURCE,
            core_ranges=core_grid,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            brisc_named_compile_time_args=brisc_named_compile_time_args,
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

        ttnn.generic_op(
            [activation_tensor, weight_tensor, output_tensor],
            program_descriptor,
        )
        return output_tensor


def build_tensors_for_fc1_test(device, w_torch_padded, x_torch, num_cores=27):
    """Construct ttnn tensors for SigLIPFC1MatmulOp.

    Args:
        device: ttnn device
        w_torch_padded: torch.Tensor (K=1152, N=4320) — FC1 weight transposed and
            padded with 16 zero cols on the right. Will be bfp8 on device.
        x_torch: torch.Tensor (M=256, K=1152) bf16 activation
        num_cores: 27 default (9×3 grid, 5 N-tiles per core)
    """
    import torch

    M = SigLIPFC1MatmulOp.M
    K = SigLIPFC1MatmulOp.K
    N = SigLIPFC1MatmulOp.N_PADDED
    assert x_torch.shape == (M, K), f"x shape {x_torch.shape}"
    assert w_torch_padded.shape == (K, N), f"w shape {w_torch_padded.shape}"

    # Core grid: 9 cols × 3 rows = 27 cores.
    grid_cols = 9
    grid_rows = num_cores // grid_cols
    assert grid_cols * grid_rows == num_cores, f"num_cores={num_cores} must factor as 9x{num_cores//9}"
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_cols - 1, grid_rows - 1))})

    # Activation: HEIGHT_SHARDED, FULL (M, K) replicated × num_cores.
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

    # Weight: WIDTH_SHARDED, (K, n_per_core) per core.
    n_per_core = N // num_cores  # 4320 / 27 = 160
    w_shard = ttnn.ShardSpec(core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    w_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, w_shard)
    weight_tt = ttnn.from_torch(
        w_torch_padded,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=w_mem,
    )

    # Output: WIDTH_SHARDED, (M, n_per_core) per core.
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
