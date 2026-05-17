"""Generic encoder-shape matmul wrapper.

Reuses qkv_matmul_kernel.cpp with arbitrary (M, K, N) shapes and selectable
M-parallel or N-parallel sharding. Used for SDPA's Q@K^T and attn@V matmuls
which need M-parallel sharding (output feeds into row-wise softmax).

The underlying kernel computes per core:
    output_tile = activation @ weight
What changes with sharding is which dimension each core slices.

  N-parallel (default, QKV pattern):
    activation: full (M, K) replicated per core
    weight:     (K, N/num_cores) per core
    output:     (M, N/num_cores) per core
    Each core produces a vertical N-strip of the result.

  M-parallel (for SDPA):
    activation: (M/num_cores, K) per core
    weight:     full (K, N) replicated per core
    output:     (M/num_cores, N) per core
    Each core produces a horizontal M-strip.
"""
import torch
import ttnn

from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor


KERNEL_SOURCE = "models/experimental/pi0/tests/perf/qkv_matmul_kernel.cpp"
TILE = 32
TILE_BYTES_BF16 = TILE * TILE * 2


def run_encoder_matmul(
    activation_tensor,
    weight_tensor,
    output_tensor,
    M,
    K,
    N,
    parallel: str = "M",  # "M" or "N"
    num_cores: int = 8,
    math_fidelity=ttnn.MathFidelity.HiFi4,
):
    """Run a generic encoder matmul on the given pre-built tensors."""
    assert M % TILE == 0 and K % TILE == 0 and N % TILE == 0
    M_TILES = M // TILE
    K_TILES = K // TILE
    N_TILES = N // TILE

    if parallel == "M":
        assert M_TILES % num_cores == 0, f"M_TILES={M_TILES} not divisible by num_cores={num_cores}"
        m_tiles_per_core = M_TILES // num_cores
        n_tiles_per_core = N_TILES
    elif parallel == "N":
        assert N_TILES % num_cores == 0, f"N_TILES={N_TILES} not divisible by num_cores={num_cores}"
        m_tiles_per_core = M_TILES
        n_tiles_per_core = N_TILES // num_cores
    else:
        raise ValueError(f"parallel must be 'M' or 'N', got {parallel}")

    act_cb, weights_cb, out_cb = 0, 1, 2
    act_tiles = m_tiles_per_core * K_TILES
    weights_tiles = K_TILES * n_tiles_per_core

    ncrisc_ct = [
        ("act_cb", act_cb),
        ("weights_cb", weights_cb),
        ("act_tiles", act_tiles),
        ("weights_tiles", weights_tiles),
    ]
    trisc_ct = [
        ("act_cb", act_cb),
        ("weights_cb", weights_cb),
        ("out_cb", out_cb),
        ("m_tiles", m_tiles_per_core),
        ("k_tiles", K_TILES),
        ("n_tiles_per_core", n_tiles_per_core),
        ("act_tiles", act_tiles),
        ("weights_tiles", weights_tiles),
    ]
    core_grid = output_tensor.memory_config().shard_spec.grid

    unified_kernel = UnifiedKernelDescriptor(
        kernel_source=KERNEL_SOURCE,
        core_ranges=core_grid,
        ncrisc_named_compile_time_args=ncrisc_ct,
        brisc_named_compile_time_args=[],
        trisc_named_compile_time_args=trisc_ct,
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


def build_matmul_tensors_m_parallel(
    device, x_torch, w_torch, M, K, N, num_cores=8, act_dtype=ttnn.bfloat16, weight_dtype=ttnn.bfloat16
):
    """Build the 3 sharded tensors for an M-parallel matmul.

    activation: (M, K) HEIGHT_SHARDED by M across `num_cores` cores
    weight:     (K, N) replicated across all `num_cores` cores
    output:     (M, N) HEIGHT_SHARDED by M across `num_cores`
    """
    assert x_torch.shape == (M, K)
    assert w_torch.shape == (K, N)
    assert M % num_cores == 0
    m_per_core = M // num_cores

    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})

    # Activation HEIGHT_SHARDED.
    act_shard = ttnn.ShardSpec(core_grid, (m_per_core, K), ttnn.ShardOrientation.ROW_MAJOR)
    act_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, act_shard)
    activation_tt = ttnn.from_torch(
        x_torch,
        dtype=act_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=act_mem,
    )

    # Weight: replicated full (K, N) across all cores using HEIGHT_SHARDED with K-replication.
    # We stack `num_cores` copies of the K x N weight vertically so each core sees the full matrix.
    w_stacked = w_torch.unsqueeze(0).repeat(num_cores, 1, 1).reshape(num_cores * K, N)
    w_shard = ttnn.ShardSpec(core_grid, (K, N), ttnn.ShardOrientation.ROW_MAJOR)
    w_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, w_shard)
    weight_tt = ttnn.from_torch(
        w_stacked,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=w_mem,
    )

    # Output HEIGHT_SHARDED by M.
    out_shard = ttnn.ShardSpec(core_grid, (m_per_core, N), ttnn.ShardOrientation.ROW_MAJOR)
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard)
    output_tt = ttnn.from_torch(
        torch.zeros(M, N, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem,
    )

    return activation_tt, weight_tt, output_tt
