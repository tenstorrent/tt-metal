# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.tests.unit.utils import run_test
from models.demos.deepseek_v3.utils.config_helpers import COMPUTE_KERNEL_CONFIG_LOFI, create_sharded_norm_config
from tests.ttnn.utils_for_testing import assert_with_pcc

# =============================================================================
# Reference tests for distributed RMSNorm operations
#
# The distributed RMSNorm is split into two phases:
#   1. rms_norm_pre_all_gather: Computes local mean(x^2) statistics
#   2. rms_norm_post_all_gather: Applies normalization using gathered stats
#
# Test cases based on model usage:
#   - Input: (1, 1, 32, 896), L1 WIDTH_SHARDED, grid=(4,7)
#   - Stats: (1, 1, 32, 256), L1 WIDTH_SHARDED, grid=(1,1)
#   - Weight: (1, 1, 28, 32), DRAM, ROW_MAJOR layout
#   - HiFi4 compute kernel
#   - LayerNormShardedMultiCoreProgramConfig
# =============================================================================


def reference_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Reference RMSNorm implementation in PyTorch."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon) * gamma


# =============================================================================
# Test: rms_norm_pre_all_gather
# =============================================================================


@pytest.mark.requires_device(["N150", "N300", "T3K", "TG", "DUAL", "QUAD"])
def test_rmsnorm_pre_all_gather_single_device(device):
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
    assert_with_pcc(expected_sum_x2, tt_sum_x2, pcc=0.99)


@pytest.mark.requires_device(["N150", "N300", "T3K", "TG", "DUAL", "QUAD"])
@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 90112, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_rmsnorm_pre_all_gather_mesh_device(mesh_device, enable_trace, device_params):
    """
    Mesh device test for rms_norm_pre_all_gather operation.

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

    logger.info(f"Testing rms_norm_pre_all_gather on mesh: shape={inp_shape}, grid={grid}")

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
        inp,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_inp = ttnn.to_memory_config(tt_inp, in_mem_config)

    def run_op():
        tt_stats = ttnn.rms_norm_pre_all_gather(
            tt_inp,
            compute_kernel_config=kernel_config,
            program_config=program_config,
            dtype=ttnn.bfloat16,
        )
        return tt_stats

    def check_op(tt_output):
        # The output contains the partial sum(x^2) in the first position
        tt_sum_x2 = tt_output[..., 0:1]
        assert_with_pcc(expected_sum_x2, tt_sum_x2, pcc=0.99)

    run_test(mesh_device, run_op, check_op, enable_trace)


# =============================================================================
# Test: rms_norm_post_all_gather
# =============================================================================


def test_rmsnorm_post_all_gather(device):
    """
    Test rms_norm_post_all_gather operation.

    This applies the RMSNorm using the gathered statistics from all devices.
    The test simulates one device's computation, with stats already gathered from 8 devices.

    Test configuration based on model usage:
    - Input X: (1, 1, 32, 896), L1 WIDTH_SHARDED, grid=(4,7)
    - Stats: (1, 1, 32, 256), L1 WIDTH_SHARDED, grid=(1,1) - gathered from 8 devices
    - Weight: (1, 1, 28, 32), DRAM, ROW_MAJOR layout (896/32 = 28 sticks)
    - HiFi4 compute kernel with math_approx_mode=True
    - LayerNormShardedMultiCoreProgramConfig
    - epsilon: 1e-6
    - dtype: bfloat16
    """
    torch.manual_seed(1234)

    # Model dimensions
    inp_shape = (1, 1, 32, 896)  # Per-device input shape
    stats_shape = (1, 1, 32, 256)  # Gathered stats from 8 devices (32 * 8 = 256)
    weight_shape = (1, 1, 28, 32)  # Gamma weights (896 / 32 = 28 sticks)
    n_devices = 8  # Number of devices stats were gathered from
    full_hidden_size = inp_shape[-1] * n_devices  # 896 * 8 = 7168
    grid = ttnn.CoreGrid(x=4, y=7)
    epsilon = 1e-6

    logger.info(f"Testing rms_norm_post_all_gather:")
    logger.info(f"  Input: {inp_shape}, Stats: {stats_shape}, Weight: {weight_shape}")

    # Compute kernel config - HiFi4 as specified in model
    kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Create the full input tensor (simulating full hidden size across all devices)
    # This device's input is just one chunk of the full tensor
    full_inp = torch.randn((1, 1, 32, full_hidden_size)).bfloat16().float()
    full_gamma = torch.rand(full_hidden_size).bfloat16().float() * 2 - 1

    # This device's local input and gamma (first chunk, simulating device 0)
    inp = full_inp[..., : inp_shape[-1]]  # (1, 1, 32, 896)
    gamma = full_gamma[: inp_shape[-1]]  # (896,)

    # Compute partial statistics for each device's chunk (sum of x^2)
    # These would have been computed by pre_all_gather on each device, then gathered
    inp_chunked = full_inp.chunk(n_devices, dim=-1)
    partial_sum_x2 = [chunk.pow(2).sum(dim=-1, keepdim=True) for chunk in inp_chunked]

    # Create the gathered stats tensor as the post_all_gather op expects
    # Stats tensor: (1, 1, 32, 256) with each device's sum(x^2) at position [idx * 32]
    stats = torch.zeros(stats_shape).bfloat16().float()
    for idx, sum_x2 in enumerate(partial_sum_x2):
        stats[..., idx * 32 : idx * 32 + 1] = sum_x2

    # Reference output: RMSNorm using global statistics
    # output = x * rsqrt(global_mean_x2 + eps) * gamma
    ref_out = reference_rmsnorm(full_inp, full_gamma, epsilon)
    ref_out_local = ref_out[..., : inp_shape[-1]]  # This device's expected output

    # Create L1 width-sharded config for input
    # grid=(4,7) = 28 cores, shard_width = 896 / 28 = 32, shard_height = 32
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

    # Create L1 width-sharded config for stats - (1, 1, 32, 256) on single core
    # Matches model: core_grid=ttnn.CoreGrid(y=1, x=1)
    stats_mem_config = ttnn.create_sharded_memory_config(
        shape=[1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE * n_devices],  # (1, 1, 32, 256)
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.WIDTH,
    )

    # Use create_sharded_norm_config to generate program config
    program_config = create_sharded_norm_config(
        grid=grid,
        dim=inp_shape[-1],  # 896
        tile_padded_batch_rows=inp_shape[-2],  # 32
    )

    # Prepare input - shape (1, 1, 32, 896), L1 WIDTH_SHARDED
    tt_inp = ttnn.from_torch(
        inp,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_inp = ttnn.to_memory_config(tt_inp, in_mem_config)

    # Prepare gamma weights - shape (1, 1, 28, 32) in ROW_MAJOR layout, DRAM
    # Matches model: memory_config=ttnn.DRAM_MEMORY_CONFIG
    tt_gamma = ttnn.from_torch(
        gamma.reshape(weight_shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Prepare gathered stats - shape (1, 1, 32, 256) in TILE layout, L1 WIDTH_SHARDED
    tt_stats = ttnn.from_torch(
        stats,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_stats = ttnn.to_memory_config(tt_stats, stats_mem_config)

    # Run post_all_gather
    tt_out = ttnn.rms_norm_post_all_gather(
        tt_inp,
        tt_stats,
        epsilon=epsilon,
        weight=tt_gamma,
        compute_kernel_config=kernel_config,
        program_config=program_config,
        dtype=ttnn.bfloat16,
    )

    # Get output and compare
    tt_out_cpu = ttnn.to_torch(tt_out)

    assert_with_pcc(ref_out_local, tt_out_cpu, pcc=0.99)


@pytest.mark.requires_device(["TG", "DUAL", "QUAD"])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 90112, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_rmsnorm_distributed_mesh_device(mesh_device, enable_trace, device_params):
    """
    End-to-end mesh device test for distributed RMSNorm.

    This test runs the full distributed RMSNorm pipeline:
    1. rms_norm_pre_all_gather: Each device computes local sum(x^2) stats
    2. all_gather: Gather stats from all devices along cluster_axis=1
    3. rms_norm_post_all_gather: Normalize using the gathered global stats

    Test configuration based on model usage:
    - Input: sharded across 8 devices along width, each device has (1, 1, 32, 896)
    - Full hidden size: 7168 = 896 * 8
    - Stats gathered along cluster_axis=1
    - HiFi4 compute kernel with math_approx_mode=True
    - epsilon: 1e-6
    """
    torch.manual_seed(1234)

    # Model dimensions
    n_devices_row = mesh_device.shape[1]  # 8 devices along cluster_axis=1
    per_device_width = 896
    full_hidden_size = per_device_width * n_devices_row  # 7168
    inp_shape_per_device = (1, 1, 32, per_device_width)
    inp_shape_full = (1, 1, 32, full_hidden_size)

    grid = ttnn.CoreGrid(x=4, y=7)
    epsilon = 1e-6

    logger.info(f"Testing distributed RMSNorm end-to-end on mesh:")
    logger.info(f"  Full input: {inp_shape_full}, per-device: {inp_shape_per_device}")
    logger.info(f"  Devices along cluster_axis=1: {n_devices_row}")

    # Compute kernel config - HiFi4 as specified in model
    kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Create the full input tensor and weights
    full_inp = torch.randn(inp_shape_full).bfloat16().float()
    full_gamma = torch.rand(full_hidden_size).bfloat16().float() * 2 - 1

    # Reference output: RMSNorm using global statistics
    ref_out = reference_rmsnorm(full_inp, full_gamma, epsilon)

    # Create L1 width-sharded config for input
    num_cores = grid.num_cores
    shard_width = per_device_width // num_cores  # 896 / 28 = 32
    shard_height = inp_shape_per_device[-2]  # 32

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

    # Create L1 width-sharded config for gathered stats
    # After all_gather: (1, 1, 32, 32 * n_devices_row) = (1, 1, 32, 256)
    stats_mem_config = ttnn.create_sharded_memory_config(
        shape=[1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE * n_devices_row],
        core_grid=ttnn.CoreGrid(y=1, x=1),
        strategy=ttnn.ShardStrategy.WIDTH,
    )

    # Program config for RMSNorm ops
    program_config = create_sharded_norm_config(
        grid=grid,
        dim=per_device_width,
        tile_padded_batch_rows=shard_height,
    )

    # Shard input across devices along width dimension (cluster_axis=1)
    # Each device gets a chunk of the hidden dimension
    tt_inp = ttnn.from_torch(
        full_inp,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=mesh_device.shape),
    )
    tt_inp = ttnn.to_memory_config(tt_inp, in_mem_config)

    # Shard gamma weights across devices
    tt_gamma = ttnn.from_torch(
        full_gamma.reshape(1, 1, full_hidden_size // 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=mesh_device.shape),
    )

    def run_op():
        # Step 1: Pre-all-gather - each device computes local sum(x^2)
        tt_stats = ttnn.rms_norm_pre_all_gather(
            tt_inp,
            compute_kernel_config=kernel_config,
            program_config=program_config,
            dtype=ttnn.bfloat16,
        )

        # Step 2: All-gather stats along cluster_axis=1
        tt_gathered_stats = ttnn.all_gather(
            tt_stats,
            dim=3,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            memory_config=stats_mem_config,
        )
        ttnn.deallocate(tt_stats)

        # Step 3: Post-all-gather - normalize using gathered global stats
        tt_out = ttnn.rms_norm_post_all_gather(
            tt_inp,
            tt_gathered_stats,
            epsilon=epsilon,
            weight=tt_gamma,
            compute_kernel_config=kernel_config,
            program_config=program_config,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(tt_gathered_stats)

        return tt_out

    def check_outputs(tt_output):
        # tt_output is sharded across devices, each device has its chunk
        coords = list(tt_output.tensor_topology().mesh_coords())
        view = mesh_device.get_view() if ttnn.using_distributed_env() else None
        device_tensors = ttnn.get_device_tensors(tt_output)

        # Compare each local device's output with its corresponding chunk of reference.
        for coord, tt_device_out in zip(coords, device_tensors):
            if view is not None and not view.is_local(coord):
                continue
            col_idx = coord[1]  # Column index in mesh (cluster_axis=1)
            tt_output_torch = ttnn.to_torch(tt_device_out)
            ref_chunk = ref_out[..., col_idx * per_device_width : (col_idx + 1) * per_device_width]
            assert_with_pcc(ref_chunk, tt_output_torch, pcc=0.99)

    # Run without trace or with trace based on enable_trace flag
    if not enable_trace:
        tt_output = run_op()
        check_outputs(tt_output)
    else:
        # Compile the op
        tt_output = run_op()
        tt_output.deallocate(True)

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        tt_output = run_op()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

        # Execute trace
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        ttnn.release_trace(mesh_device, trace_id)

        check_outputs(tt_output)


# =============================================================================
# Test: Non-distributed RMSNorm (ttnn.rms_norm)
# =============================================================================


@pytest.mark.parametrize(
    "inp_shape, weight_shape",
    [
        ((1, 1, 32, 1536), (1, 1, 48, 32)),  # 1536 / 32 = 48 sticks
        ((1, 1, 32, 512), (1, 1, 16, 32)),  # 512 / 32 = 16 sticks
    ],
    ids=["32x1536", "32x512"],
)
@pytest.mark.requires_device(["N150", "N300", "T3K", "TG", "DUAL", "QUAD"])
def test_rmsnorm_single_device(device, inp_shape, weight_shape):
    """
    Test non-distributed RMSNorm operation (ttnn.rms_norm).

    Test configuration:
    - Input: bfloat16, DRAM INTERLEAVED, TILE layout
    - Weight: bfloat16, DRAM INTERLEAVED, ROW_MAJOR layout
    - LoFi compute kernel (math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True)
    - LayerNormDefaultProgramConfig
    - epsilon: 1e-6
    """
    torch.manual_seed(1234)

    epsilon = 1e-6

    logger.info(f"Testing rms_norm: inp_shape={inp_shape}, weight_shape={weight_shape}")

    # Create input and weight tensors
    inp = torch.randn(inp_shape).bfloat16().float()
    gamma = torch.rand(inp_shape[-1]).bfloat16().float() * 2 - 1

    # Reference output
    ref_out = reference_rmsnorm(inp, gamma, epsilon)

    # Prepare input - DRAM INTERLEAVED, TILE layout
    tt_inp = ttnn.from_torch(
        inp,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Prepare gamma weights - DRAM INTERLEAVED, ROW_MAJOR layout
    tt_gamma = ttnn.from_torch(
        gamma.reshape(weight_shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Run rms_norm with LoFi compute kernel and default program config
    tt_out = ttnn.rms_norm(
        tt_inp,
        epsilon=epsilon,
        weight=tt_gamma,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
        program_config=ttnn.LayerNormDefaultProgramConfig(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Get output and compare
    tt_out_cpu = ttnn.to_torch(tt_out)

    assert_with_pcc(ref_out, tt_out_cpu, pcc=0.99)


@pytest.mark.parametrize(
    "inp_shape, weight_shape",
    [
        ((1, 1, 32, 1536), (1, 1, 48, 32)),  # 1536 / 32 = 48 sticks
        ((1, 1, 32, 512), (1, 1, 16, 32)),  # 512 / 32 = 16 sticks
    ],
    ids=["32x1536", "32x512"],
)
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 90112, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
@pytest.mark.requires_device(["T3K", "TG", "DUAL", "QUAD"])
def test_rmsnorm_mesh_device(mesh_device, inp_shape, weight_shape, enable_trace, device_params):
    """
    Mesh device test for non-distributed RMSNorm operation (ttnn.rms_norm).

    Test configuration:
    - Input: bfloat16, DRAM INTERLEAVED, TILE layout
    - Weight: bfloat16, DRAM INTERLEAVED, ROW_MAJOR layout
    - LoFi compute kernel (math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True)
    - LayerNormDefaultProgramConfig
    - epsilon: 1e-6
    """
    torch.manual_seed(1234)

    epsilon = 1e-6

    logger.info(f"Testing rms_norm on mesh: inp_shape={inp_shape}, weight_shape={weight_shape}")

    # Create input and weight tensors
    inp = torch.randn(inp_shape).bfloat16().float()
    gamma = torch.rand(inp_shape[-1]).bfloat16().float() * 2 - 1

    # Reference output
    ref_out = reference_rmsnorm(inp, gamma, epsilon)

    # Prepare input - DRAM INTERLEAVED, TILE layout
    tt_inp = ttnn.from_torch(
        inp,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Prepare gamma weights - DRAM INTERLEAVED, ROW_MAJOR layout
    tt_gamma = ttnn.from_torch(
        gamma.reshape(weight_shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        tt_out = ttnn.rms_norm(
            tt_inp,
            epsilon=epsilon,
            weight=tt_gamma,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            program_config=ttnn.LayerNormDefaultProgramConfig(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return tt_out

    def check_op(tt_output):
        assert_with_pcc(ref_out, tt_output, pcc=0.99)

    run_test(mesh_device, run_op, check_op, enable_trace)
