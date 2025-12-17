# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.utils.config_helpers import create_sharded_norm_config
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# =============================================================================
# Reference tests for distributed RMSNorm operations
#
# The distributed RMSNorm is split into two phases:
#   1. rms_norm_pre_all_gather: Computes local mean(x^2) statistics
#   2. rms_norm_post_all_gather: Applies normalization using gathered stats
#
# Test cases based on model usage:
#   - Input: (1, 1, 32, 896), L1 WIDTH_SHARDED, grid=(4,7)
#   - HiFi4 compute kernel
#   - LayerNormShardedMultiCoreProgramConfig
# =============================================================================


def reference_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Reference RMSNorm implementation in PyTorch."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon) * gamma


# =============================================================================
# Test: rms_norm_pre_all_gather
# =============================================================================


def test_rmsnorm_pre_all_gather(device):
    """
    Test rms_norm_pre_all_gather operation.

    This computes the local mean(x^2) statistics that will be gathered across devices.
    The output is a stats tensor containing the partial sums.

    Test configuration based on model usage:
    - Input: (1, 1, 32, 896), L1 WIDTH_SHARDED, grid=(4,7)
    - HiFi4 compute kernel with math_approx_mode=True
    - LayerNormShardedMultiCoreProgramConfig
    """
    torch.manual_seed(1234)

    inp_shape = (1, 1, 32, 896)
    grid = ttnn.CoreGrid(x=4, y=7)

    logger.info(f"Testing rms_norm_pre_all_gather: shape={inp_shape}, grid={grid}")

    # Compute kernel config - HiFi4 as specified in model
    kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Create input tensor
    inp = torch.randn(inp_shape).bfloat16().float()

    # Compute expected partial sum(x^2)
    expected_sum_x2 = inp.pow(2).sum(dim=-1, keepdim=True)

    # Create L1 width-sharded config matching model usage
    # grid=(4,7) = 28 cores, shard_width = 896 / 28 = 32
    num_cores = grid.num_cores
    shard_width = inp_shape[-1] // num_cores  # 896 / 28 = 32
    shard_height = inp_shape[-2]  # 32

    in_mem_config = ttnn.create_sharded_memory_config_(
        shape=(shard_height, shard_width),
        core_grid=ttnn.num_cores_to_corerangeset(
            num_cores,
            ttnn.CoreCoord(grid.x, grid.y),
            row_wise=True,
        ),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Use create_sharded_norm_config to generate program config
    program_config = create_sharded_norm_config(
        grid=grid,
        dim=inp_shape[-1],  # 896
        tile_padded_batch_rows=inp_shape[-2],  # 32
    )

    # Convert to TT tensor
    tt_inp = ttnn.from_torch(
        inp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_inp = ttnn.to_memory_config(tt_inp, in_mem_config)

    # Run pre_all_gather
    tt_stats = ttnn.rms_norm_pre_all_gather(
        tt_inp,
        compute_kernel_config=kernel_config,
        program_config=program_config,
        dtype=ttnn.bfloat16,
    )

    # Get output
    tt_stats_cpu = ttnn.to_torch(tt_stats)

    # The output contains the partial sum(x^2) in the first position
    tt_sum_x2 = tt_stats_cpu[..., 0:1]

    # Compare
    passing, output_str = comp_pcc(expected_sum_x2, tt_sum_x2, pcc=0.99)
    logger.info(f"Result: {output_str}")

    assert passing, "rms_norm_pre_all_gather test failed"


# =============================================================================
# Test: rms_norm_post_all_gather
# =============================================================================


def run_test_rmsnorm_post_all_gather(
    device,
    inp_shape,
    n_devices,
    input_dtype,
    output_dtype,
    use_sharded,
):
    """
    Test rms_norm_post_all_gather operation.

    This applies the RMSNorm using the gathered statistics from all devices.
    It mirrors the distributed layernorm test pattern.
    """
    torch.manual_seed(1234)

    logger.info(f"Testing rms_norm_post_all_gather: shape={inp_shape}, n_devices={n_devices}")
    logger.info(f"input_dtype={input_dtype}, output_dtype={output_dtype}, use_sharded={use_sharded}")

    # Compute kernel config - HiFi4 for accuracy
    kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Create full input tensor and gamma weights
    full_inp = torch.randn(inp_shape).bfloat16().float()
    full_gamma = torch.rand(inp_shape[-1]).bfloat16().float() * 2 - 1

    # Split across simulated devices
    inp_chunked = full_inp.chunk(n_devices, dim=-1)
    gamma_chunked = full_gamma.chunk(n_devices, dim=-1)

    # Compute partial statistics for each chunk (sum of x^2)
    partial_sum_x2 = [chunk.pow(2).sum(dim=-1, keepdim=True) for chunk in inp_chunked]

    # Create the gathered stats tensor as the post_all_gather op expects
    # Stats tensor contains sum(x^2) values from all devices, padded to tile width
    stats_tiles = torch.zeros(inp_shape[:-1] + (32 * n_devices,))
    for idx, sum_x2 in enumerate(partial_sum_x2):
        stats_tiles[..., idx * 32 : idx * 32 + 1] = sum_x2

    epsilon = 1e-6

    # Reference output (full RMSNorm)
    ref_out = reference_rmsnorm(full_inp, full_gamma, epsilon)
    ref_chunks = ref_out.chunk(n_devices, dim=-1)

    # Memory configs
    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    if use_sharded:
        grid_x, grid_y = 4, 7
        num_cores = grid_x * grid_y
        shard_width = ttnn.core.roundup(inp_shape[-1] // n_devices // num_cores, ttnn.TILE_SIZE)
        shard_height = ttnn.core.roundup(inp_shape[-2], ttnn.TILE_SIZE)

        in_mem_config = ttnn.create_sharded_memory_config_(
            shape=(shard_height, shard_width),
            core_grid=ttnn.num_cores_to_corerangeset(
                num_cores,
                ttnn.CoreCoord(grid_x, grid_y),
                row_wise=True,
            ),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            subblock_w=1,
            block_h=ttnn.core.divup(shard_height, ttnn.TILE_SIZE),
            block_w=ttnn.core.divup(shard_width, ttnn.TILE_SIZE),
            inplace=False,
        )
    else:
        in_mem_config = dram_memcfg
        program_config = ttnn.LayerNormDefaultProgramConfig()

    all_pass = True

    # Test post_all_gather for each simulated device's chunk
    for d in range(n_devices):
        # Prepare input chunk
        tt_inp = ttnn.from_torch(
            inp_chunked[d],
            dtype=input_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram_memcfg,
        )
        if use_sharded:
            tt_inp = ttnn.to_memory_config(tt_inp, in_mem_config)

        # Prepare gamma weights (ROW_MAJOR layout as per model)
        tt_gamma = ttnn.from_torch(
            gamma_chunked[d].reshape(1, 1, -1, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=dram_memcfg,
        )

        # Prepare gathered stats
        tt_stats = ttnn.from_torch(
            stats_tiles,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram_memcfg,
        )

        # Run post_all_gather
        tt_out = ttnn.rms_norm_post_all_gather(
            tt_inp,
            tt_stats,
            epsilon=epsilon,
            weight=tt_gamma,
            compute_kernel_config=kernel_config,
            program_config=program_config,
            dtype=output_dtype,
        )

        # Get output and compare
        tt_out_cpu = ttnn.to_torch(tt_out)

        passing, output_str = comp_pcc(ref_chunks[d], tt_out_cpu, pcc=0.99)
        logger.info(f"Device {d}: {output_str}")
        all_pass = all_pass and passing

    assert all_pass, "rms_norm_post_all_gather test failed"


def run_test_rmsnorm_distributed_e2e(
    device,
    inp_shape,
    n_devices,
    input_dtype,
    output_dtype,
    use_sharded,
):
    """
    End-to-end test of distributed RMSNorm (pre_all_gather + post_all_gather).

    This simulates the full distributed RMSNorm flow as used in the model.
    """
    torch.manual_seed(1234)

    logger.info(f"Testing distributed RMSNorm E2E: shape={inp_shape}, n_devices={n_devices}")

    # Compute kernel config
    kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Create full input tensor and gamma weights
    full_inp = torch.randn(inp_shape).bfloat16().float()
    full_gamma = torch.rand(inp_shape[-1]).bfloat16().float() * 2 - 1

    # Split across simulated devices
    inp_chunked = full_inp.chunk(n_devices, dim=-1)
    gamma_chunked = full_gamma.chunk(n_devices, dim=-1)

    epsilon = 1e-6

    # Reference output
    ref_out = reference_rmsnorm(full_inp, full_gamma, epsilon)
    ref_chunks = ref_out.chunk(n_devices, dim=-1)

    # Memory configs
    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    if use_sharded:
        grid_x, grid_y = 4, 7
        num_cores = grid_x * grid_y
        shard_width = ttnn.core.roundup(inp_shape[-1] // n_devices // num_cores, ttnn.TILE_SIZE)
        shard_height = ttnn.core.roundup(inp_shape[-2], ttnn.TILE_SIZE)

        in_mem_config = ttnn.create_sharded_memory_config_(
            shape=(shard_height, shard_width),
            core_grid=ttnn.num_cores_to_corerangeset(
                num_cores,
                ttnn.CoreCoord(grid_x, grid_y),
                row_wise=True,
            ),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            subblock_w=1,
            block_h=ttnn.core.divup(shard_height, ttnn.TILE_SIZE),
            block_w=ttnn.core.divup(shard_width, ttnn.TILE_SIZE),
            inplace=False,
        )
    else:
        in_mem_config = dram_memcfg
        program_config = ttnn.LayerNormDefaultProgramConfig()

    # Step 1: Run pre_all_gather on each chunk and collect stats
    all_stats = []
    tt_inputs = []

    for d in range(n_devices):
        tt_inp = ttnn.from_torch(
            inp_chunked[d],
            dtype=input_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram_memcfg,
        )
        if use_sharded:
            tt_inp = ttnn.to_memory_config(tt_inp, in_mem_config)
        tt_inputs.append(tt_inp)

        # Run pre_all_gather
        tt_stats = ttnn.rms_norm_pre_all_gather(
            tt_inp,
            compute_kernel_config=kernel_config,
            program_config=program_config,
            dtype=ttnn.bfloat16,
        )
        all_stats.append(ttnn.to_torch(tt_stats))

    # Step 2: Simulate all-gather by concatenating stats
    # In real distributed setting, this would be done via CCL
    gathered_stats = torch.cat(all_stats, dim=-1)

    all_pass = True

    # Step 3: Run post_all_gather on each chunk
    for d in range(n_devices):
        # Recreate input tensor (or reuse if still on device)
        tt_inp = ttnn.from_torch(
            inp_chunked[d],
            dtype=input_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram_memcfg,
        )
        if use_sharded:
            tt_inp = ttnn.to_memory_config(tt_inp, in_mem_config)

        tt_gamma = ttnn.from_torch(
            gamma_chunked[d].reshape(1, 1, -1, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=dram_memcfg,
        )

        tt_gathered_stats = ttnn.from_torch(
            gathered_stats,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram_memcfg,
        )

        # Run post_all_gather
        tt_out = ttnn.rms_norm_post_all_gather(
            tt_inp,
            tt_gathered_stats,
            epsilon=epsilon,
            weight=tt_gamma,
            compute_kernel_config=kernel_config,
            program_config=program_config,
            dtype=output_dtype,
        )

        tt_out_cpu = ttnn.to_torch(tt_out)

        passing, output_str = comp_pcc(ref_chunks[d], tt_out_cpu, pcc=0.99)
        logger.info(f"Device {d}: {output_str}")
        all_pass = all_pass and passing

    assert all_pass, "Distributed RMSNorm E2E test failed"


@pytest.mark.parametrize(
    "inp_shape",
    [
        (1, 1, 32, 896),  # From model spec
        (1, 1, 32, 7168),  # Full hidden size
        (1, 1, 128, 8192),  # Larger sequence
    ],
    ids=["32x896", "32x7168", "128x8192"],
)
@pytest.mark.parametrize(
    "n_devices",
    [1, 4, 8],
    ids=["1dev", "4dev", "8dev"],
)
@pytest.mark.parametrize(
    "input_dtype, output_dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat16),
    ],
    ids=["bf16_bf16"],
)
@pytest.mark.parametrize(
    "use_sharded",
    [False, True],
    ids=["interleaved", "sharded"],
)
def test_rmsnorm_post_all_gather(
    device,
    inp_shape,
    n_devices,
    input_dtype,
    output_dtype,
    use_sharded,
):
    # Skip if input can't be evenly divided
    if inp_shape[-1] % n_devices != 0:
        pytest.skip(f"Input width {inp_shape[-1]} not divisible by {n_devices} devices")

    # Skip sharded tests for very small per-device widths
    per_device_width = inp_shape[-1] // n_devices
    if use_sharded and per_device_width < 32:
        pytest.skip(f"Per-device width {per_device_width} too small for sharding")

    run_test_rmsnorm_post_all_gather(
        device,
        inp_shape,
        n_devices,
        input_dtype,
        output_dtype,
        use_sharded,
    )


# =============================================================================
# Test: Distributed RMSNorm End-to-End
# =============================================================================


@pytest.mark.parametrize(
    "inp_shape",
    [
        (1, 1, 32, 896),  # Model decode shape
        (1, 1, 32, 7168),  # Full hidden size
    ],
    ids=["32x896", "32x7168"],
)
@pytest.mark.parametrize(
    "n_devices",
    [4, 8],
    ids=["4dev", "8dev"],
)
@pytest.mark.parametrize(
    "input_dtype, output_dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat16),
    ],
    ids=["bf16_bf16"],
)
@pytest.mark.parametrize(
    "use_sharded",
    [False, True],
    ids=["interleaved", "sharded"],
)
def test_rmsnorm_distributed_e2e(
    device,
    inp_shape,
    n_devices,
    input_dtype,
    output_dtype,
    use_sharded,
):
    # Skip if input can't be evenly divided
    if inp_shape[-1] % n_devices != 0:
        pytest.skip(f"Input width {inp_shape[-1]} not divisible by {n_devices} devices")

    # Skip sharded tests for very small per-device widths
    per_device_width = inp_shape[-1] // n_devices
    if use_sharded and per_device_width < 32:
        pytest.skip(f"Per-device width {per_device_width} too small for sharding")

    run_test_rmsnorm_distributed_e2e(
        device,
        inp_shape,
        n_devices,
        input_dtype,
        output_dtype,
        use_sharded,
    )
