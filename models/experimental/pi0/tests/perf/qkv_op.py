"""SigLIP fused-QKV matmul micro-op — Python wrapper.

First-draft. Mirrors models/demos/deepseek_v3_b1/micro_ops/kn_sliced_matmul/op.py
but for SigLIP encoder shape (M=256, K=1152, N=3456) with M-tile-row outer loop
in the kernel.

Decomposition (matches qkv_matmul_kernel.cpp):
  - Activation `(M=256, K=1152)` bf16, HEIGHT_SHARDED L1, full M×K replicated per core
  - Weight `(K=1152, N=3456)` bfp8, WIDTH_SHARDED L1, N partitioned across cores
  - Output `(M=256, N=3456)` bf16, WIDTH_SHARDED L1, same N partition as weight

Each core computes:
  out_local[256, N_per_core] = activation[256, 1152] @ weight_local[1152, N_per_core]

No all-reduce (N-sharding produces final values). Weights stay L1-resident.
"""
import ttnn

from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor


class SigLIPQKVMatmulOp:
    """SigLIP fused-QKV matmul on a single chip with L1-resident weights."""

    KERNEL_SOURCE = "models/experimental/pi0/tests/perf/qkv_matmul_kernel.cpp"

    # SigLIP-So400m/14 layer-0 fused QKV
    M = 256
    K = 1152
    N = 3456
    TILE = 32
    M_TILES = M // TILE  # 8
    K_TILES = K // TILE  # 36
    N_TILES = N // TILE  # 108

    @staticmethod
    def op(
        activation_tensor,  # (M, K) bf16, HEIGHT_SHARDED L1, full K replicated per core
        weight_tensor,  # (K, N) bfp8, WIDTH_SHARDED L1, (K, N/num_cores) per core
        output_tensor,  # (M, N) bf16, WIDTH_SHARDED L1, (M, N/num_cores) per core
        num_cores: int = 36,
        math_fidelity=ttnn.MathFidelity.HiFi2,
    ):
        """Execute the matmul. Tensors must already be on device with the
        memory configs described above. Output is written in-place to
        `output_tensor`."""
        n_tiles_per_core = SigLIPQKVMatmulOp.N_TILES // num_cores
        assert (
            n_tiles_per_core * num_cores == SigLIPQKVMatmulOp.N_TILES
        ), f"num_cores ({num_cores}) must evenly divide N_TILES ({SigLIPQKVMatmulOp.N_TILES})"

        # CB indices
        act_cb = 0
        weights_cb = 1
        out_cb = 2

        # Tile counts for sharded buffer setup (per core)
        act_tiles = SigLIPQKVMatmulOp.M_TILES * SigLIPQKVMatmulOp.K_TILES  # 8 * 36 = 288
        weights_tiles = SigLIPQKVMatmulOp.K_TILES * n_tiles_per_core  # 36 * 3 = 108

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
            ("m_tiles", SigLIPQKVMatmulOp.M_TILES),
            ("k_tiles", SigLIPQKVMatmulOp.K_TILES),
            ("n_tiles_per_core", n_tiles_per_core),
            ("act_tiles", act_tiles),
            ("weights_tiles", weights_tiles),
        ]

        core_grid = output_tensor.memory_config().shard_spec.grid

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source=SigLIPQKVMatmulOp.KERNEL_SOURCE,
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


def build_tensors_for_test(device, w_torch, b_torch, x_torch, num_cores=36):
    """Construct ttnn tensors with the right shard specs for SigLIPQKVMatmulOp.op.

    Args:
        device: ttnn device
        w_torch: torch.Tensor (K=1152, N=3456) — weight, will be bfp8 on device
        b_torch: torch.Tensor (N=3456,) — bias (currently unused; reserved for fused-bias kernel)
        x_torch: torch.Tensor (M=256, K=1152) bf16 activation
        num_cores: how many cores to spread N over

    Returns:
        (activation_tt, weight_tt, output_tt)
    """
    import torch

    M = SigLIPQKVMatmulOp.M
    K = SigLIPQKVMatmulOp.K
    N = SigLIPQKVMatmulOp.N
    assert x_torch.shape == (M, K), f"x shape {x_torch.shape}"
    assert w_torch.shape == (K, N), f"w shape {w_torch.shape}"

    # Core grid: 6×6 = 36 cores (for default num_cores=36).
    # The kernel doesn't depend on the specific layout, only on num_cores
    # evenly dividing N_TILES.
    grid_cols = 6 if num_cores == 36 else 4 if num_cores == 32 else num_cores
    grid_rows = num_cores // grid_cols
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_cols - 1, grid_rows - 1))})

    # Activation: HEIGHT_SHARDED with FULL K and FULL M on every core (replicated).
    # This matches kn_sliced_matmul's activation layout — each core has a copy
    # of the full activation buffer.
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

    # Weight: WIDTH_SHARDED so each core owns its N-slice (K rows, N/num_cores cols).
    n_per_core = N // num_cores
    w_shard = ttnn.ShardSpec(core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    w_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, w_shard)
    weight_tt = ttnn.from_torch(
        w_torch,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=w_mem,
    )

    # Output: WIDTH_SHARDED, same partition as weight.
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
