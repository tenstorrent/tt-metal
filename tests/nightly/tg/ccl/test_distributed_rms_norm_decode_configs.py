# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test DistributedRMSNorm with actual decode configurations from DeepSeek V3 model.
Based on test_rms_fuse_deepseek from tests/ttnn/unit_tests/operations/ccl/test_minimals.py
"""

import torch
import pytest
from loguru import logger
import ttnn
from tracy import signpost

from models.common.utility_functions import skip_for_blackhole
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from models.perf.benchmarking_utils import BenchmarkProfiler


def get_torch_rms(x, dim, gamma, eps):
    """PyTorch reference implementation of RMS normalization."""
    return x * torch.rsqrt(x.pow(2).mean([-i for i in range(1, len(dim) + 1)], keepdim=True) + eps) * gamma


def run_distributed_rms_norm_decode_impl(
    mesh_device,
    num_devices,
    batch_size_per_row,
    seq_len,
    hidden_size,
    epsilon,
    input_shard_grid,
    output_shard_grid,
    topology,
    num_iters=20,
    trace_mode=False,
    warmup_iters=20,
    input_dtype=ttnn.bfloat16,
    output_dtype=ttnn.bfloat16,
    profiler=None,
):
    """
    Test DistributedRMSNorm with actual decode configuration.

    This matches the configuration used in:
    - mtp.py: hidden_norm, token_norm, head_norm
    - decoder_block_2d_base.py: mla_norm, mlp_norm
    - row_batched_model.py: norm
    """
    if profiler is None:
        profiler = BenchmarkProfiler()

    # Create subdevice for CCL operations
    ccl_sub_device_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 7))})
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    torch.manual_seed(1234)

    # Calculate sharding parameters matching DistributedRMSNorm.decode_model_config
    num_cores = input_shard_grid.num_cores()
    total_cores = num_cores * num_devices

    # Shard width per core (each core gets a slice of hidden_size dimension)
    shard_width_per_core = ttnn.core.roundup(
        hidden_size // total_cores,
        ttnn.TILE_SIZE,
    )

    # Input shape: (1, 1, seq_len, hidden_size)
    # seq_len is typically 8 or 32 for decode (in tiles)
    # batch_size_per_row affects memory config but not tensor shape
    input_shape = (1, 1, seq_len, hidden_size)

    # Shard height matches the actual decode config: roundup(batch_size_per_row, TILE_SIZE)
    # For batch_size_per_row in [8, 32], this rounds up to 32
    shard_height = ttnn.core.roundup(batch_size_per_row, ttnn.TILE_SIZE)

    # Input memory config matching decode_model_config
    input_memory_config = ttnn.create_sharded_memory_config(
        shape=(shard_height, shard_width_per_core),
        core_grid=input_shard_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Layer norm program config
    # block_h is always 1 for decode (32 rows = 1 tile)
    layer_norm_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(4, 8),
        subblock_w=1,
        block_h=1,
        block_w=shard_width_per_core // ttnn.TILE_SIZE,
        inplace=False,
    )

    # Create semaphore handles for each iteration
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, input_shard_grid, 0) for _ in range(num_iters)]

    # Stats buffer configuration - single core sharding so stats.padded_shape[-1] = num_devices * TILE_WIDTH
    # This is critical for correct RMS scaling across devices
    ag_memory_config = ttnn.create_sharded_memory_config(
        shape=(32, 32),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Create persistent stats tensor
    ag_shape = [1, 1, 32, num_devices]
    stats_tensor = torch.zeros(ag_shape, dtype=torch.bfloat16)
    tt_stats = ttnn.from_torch(
        stats_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ag_memory_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(3, None), mesh_shape=list(ttnn.MeshShape(num_devices, 1))
        ),
    )

    # Output memory config
    if output_shard_grid is None:
        output_shard_grid = input_shard_grid

    output_memory_config = ttnn.create_sharded_memory_config(
        shape=(shard_height, shard_width_per_core),
        core_grid=output_shard_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Create test data
    input_tensor_torch = []
    gamma_torch = []
    input_tensor = []
    gamma_tensor = []
    tt_out_array = []

    for i in range(num_iters):
        input_tensor_torch.append(torch.randn(input_shape))
        gamma_torch.append(torch.randn((1, 1, 1, hidden_size)))

        input_tensor.append(
            ttnn.as_tensor(
                input_tensor_torch[i],
                dtype=input_dtype,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device=mesh_device, dims=(3, None), mesh_shape=list(ttnn.MeshShape(num_devices, 1))
                ),
                layout=ttnn.TILE_LAYOUT,
                memory_config=input_memory_config,
            )
        )

        gamma_tensor.append(
            ttnn.as_tensor(
                gamma_torch[i].reshape([1, 1, hidden_size // 32, 32]),
                dtype=ttnn.bfloat16,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device=mesh_device, dims=(2, None), mesh_shape=list(ttnn.MeshShape(num_devices, 1))
                ),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )

    # Compile run
    logger.info("Compiling model")
    for i in range(num_iters):
        tt_out = ttnn.fused_rms_minimal(
            input_tensor[i],
            layer_norm_config,
            0,  # cluster_axis=0 for row-wise sharding (dims=(3, None))
            mesh_device,
            ccl_semaphore_handles[i],
            topology=topology,
            memory_config=output_memory_config,
            epsilon=epsilon,
            dtype=output_dtype,
            weight=gamma_tensor[i],
            residual_input_tensor=None,
            stats=tt_stats,
            use_noc1_only=False,
        )
        tt_out_array.append(tt_out)

    ttnn.synchronize_device(mesh_device)

    # PCC check on compile output
    logger.info("Checking PCC for compile run")
    for i in range(num_iters):
        tt_out_torch = ttnn.to_torch(
            tt_out_array[i],
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=(num_devices, 1)),
        )[0].unsqueeze(0)

        ref_lnorm = get_torch_rms(input_tensor_torch[i], [3], gamma_torch[i], epsilon)
        passing, output = comp_pcc(tt_out_torch, ref_lnorm, 0.999)
        logger.info(f"Iteration {i}: {output}")
        assert passing, f"PCC check failed for iteration {i}"

    if trace_mode:
        logger.info("Running in trace mode")

        # Warmup trace
        if warmup_iters > 0:
            logger.info("Capturing warmup trace")
            trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            for i in range(warmup_iters):
                tt_out = ttnn.fused_rms_minimal(
                    input_tensor[i % num_iters],
                    layer_norm_config,
                    0,
                    mesh_device,
                    ccl_semaphore_handles[i % num_iters],
                    topology=topology,
                    memory_config=output_memory_config,
                    epsilon=epsilon,
                    dtype=output_dtype,
                    weight=gamma_tensor[i % num_iters],
                    residual_input_tensor=None,
                    stats=tt_stats,
                    use_noc1_only=False,
                )
                tt_out.deallocate(True)
            ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
            ttnn.synchronize_device(mesh_device)

        # Main trace
        logger.info("Capturing main trace")
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for i in range(num_iters):
            tt_out = ttnn.fused_rms_minimal(
                input_tensor[i],
                layer_norm_config,
                0,
                mesh_device,
                ccl_semaphore_handles[i],
                topology=topology,
                memory_config=output_memory_config,
                epsilon=epsilon,
                dtype=output_dtype,
                weight=gamma_tensor[i],
                residual_input_tensor=None,
                stats=tt_stats,
                use_noc1_only=False,
            )
            if i != num_iters - 1:
                tt_out.deallocate(True)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        # Execute traces with signposts
        logger.info("Executing traces with signposts")
        profiler.start("rms-trace-warmup")
        if warmup_iters > 0:
            ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
            ttnn.release_trace(mesh_device, trace_id_warmup)
        profiler.end("rms-trace-warmup")

        signpost("start")
        profiler.start("rms-trace")
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        ttnn.release_trace(mesh_device, trace_id)
        profiler.end("rms-trace")
        signpost("stop")

        time_taken = profiler.get_duration("rms-trace") - profiler.get_duration("rms-trace-warmup")
        logger.info(f"Trace execution time: {time_taken}ms")

    mesh_device.reset_sub_device_stall_group()


@skip_for_blackhole("This is a wormhole test")
@pytest.mark.parametrize(
    "batch_size_per_row, hidden_size, input_shard_grid, output_shard_grid, config_name",
    [
        # MTP hidden_norm, token_norm, head_norm configuration with batch_size_per_row=8, seq_len=32
        (
            8,
            896 * 8,  # 7168 (DeepSeek V3 hidden_size)
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))}),
            None,
            "mtp_norms_batch8",
        ),
        # MTP norms with batch_size_per_row=32
        (
            32,
            896 * 8,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))}),
            None,
            "mtp_norms_batch32",
        ),
        # Decoder block mla_norm, mlp_norm configuration
        (
            8,
            896 * 8,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))}),
            "decoder_block_norms_batch8",
        ),
        # Row batched model norm configuration
        (
            32,
            896 * 8,
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))}),
            "row_batched_model_norm",
        ),
    ],
)
@pytest.mark.parametrize("num_iters", [20])
@pytest.mark.parametrize(
    "input_dtype, output_dtype", [(ttnn.bfloat16, ttnn.bfloat16), (ttnn.bfloat8_b, ttnn.bfloat8_b)]
)
@pytest.mark.parametrize("topology", [ttnn.Topology.Linear])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", [8, 32])
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_distributed_rms_norm_decode_configs(
    mesh_device,
    batch_size_per_row,
    seq_len,
    hidden_size,
    input_shard_grid,
    output_shard_grid,
    config_name,
    num_iters,
    input_dtype,
    output_dtype,
    topology,
    function_level_defaults,
):
    """Test DistributedRMSNorm with actual decode configurations from DeepSeek V3."""
    # DeepSeek V3 config values
    epsilon = 1e-6  # rms_norm_eps from DeepSeek V3 config
    num_devices = 8  # Using 8 devices in column (mesh shape is 8x4)

    logger.info(f"Testing configuration: {config_name}")
    logger.info(f"  batch_size_per_row: {batch_size_per_row}")
    logger.info(f"  seq_len: {seq_len}")
    logger.info(f"  hidden_size: {hidden_size}")
    logger.info(f"  num_devices: {num_devices}")
    logger.info(f"  input_dtype: {input_dtype}, output_dtype: {output_dtype}")

    run_distributed_rms_norm_decode_impl(
        mesh_device=mesh_device,
        num_devices=num_devices,
        batch_size_per_row=batch_size_per_row,
        seq_len=seq_len,
        hidden_size=hidden_size,
        epsilon=epsilon,
        input_shard_grid=input_shard_grid,
        output_shard_grid=output_shard_grid,
        topology=topology,
        num_iters=num_iters,
        trace_mode=True,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
    )
