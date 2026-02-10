# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize(
    "op_name, input_shape, src_mem_config, dst_mem_config_dict",
    [
        (
            "kv_nope_to_interleaved",
            [1, 1, 32, 512],
            {
                "memory_layout": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                "buffer_type": ttnn.BufferType.L1,
                "shard_spec": {
                    "grid": (8, 2),
                    "shape": [32, 32],
                    "orientation": ttnn.ShardOrientation.ROW_MAJOR,
                },
            },
            {"memory_config": ttnn.L1_MEMORY_CONFIG},
        ),
        (
            "kv_rope_reshard",
            [1, 32, 1, 64],
            {"memory_config": ttnn.L1_MEMORY_CONFIG},
            {
                "memory_config": {
                    "memory_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "buffer_type": ttnn.BufferType.L1,
                    "shard_spec": {
                        "grid": (4, 8),
                        "shape": [32, 64],
                        "orientation": ttnn.ShardOrientation.ROW_MAJOR,
                    },
                }
            },
        ),
        (
            "kv_rope_out_reshard",
            [1, 32, 1, 64],
            {
                "memory_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                "buffer_type": ttnn.BufferType.L1,
                "shard_spec": {
                    "grid": (4, 8),
                    "shape": [32, 64],
                    "orientation": ttnn.ShardOrientation.ROW_MAJOR,
                },
            },
            {"memory_config": ttnn.L1_MEMORY_CONFIG},
        ),
        (
            "kvpe_reshard",
            [1, 4, 32, 576],  # After all-gather, shape is [1, 4, 1(32), 576]
            {"memory_config": ttnn.L1_MEMORY_CONFIG},
            {
                "memory_config": {
                    "memory_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "buffer_type": ttnn.BufferType.L1,
                    "shard_spec": {
                        "grid": (1, 4),
                        "shape": [32, 576],
                        "orientation": ttnn.ShardOrientation.ROW_MAJOR,
                    },
                }
            },
        ),
        (
            "q_rope_out_reshard",
            [1, 32, 16, 64],
            {
                "memory_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                "buffer_type": ttnn.BufferType.L1,
                "shard_spec": {
                    "grid": (8, 4),  # 32 cores
                    "shape": [32, 64],
                    "orientation": ttnn.ShardOrientation.ROW_MAJOR,
                },
            },
            {"memory_config": ttnn.L1_MEMORY_CONFIG},
        ),
        (
            "flash_mla_reshard",
            [1, 4, 128, 576],
            {"memory_config": ttnn.L1_MEMORY_CONFIG},
            {
                "memory_config": {
                    "memory_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    "buffer_type": ttnn.BufferType.L1,
                    "shard_spec": {
                        "grid": (8, 8),  # 64 cores: min(32/8 * 16, 70) = 64
                        "shape": [32, 576],
                        "orientation": ttnn.ShardOrientation.ROW_MAJOR,
                    },
                }
            },
        ),
        (
            "flash_mla_out_reshard",
            [1, 4, 128, 512],
            {
                "memory_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                "buffer_type": ttnn.BufferType.L1,
                "shard_spec": {
                    "grid": (8, 8),  # 64 cores: same as flash_mla_reshard
                    "shape": [32, 512],
                    "orientation": ttnn.ShardOrientation.ROW_MAJOR,
                },
            },
            {"memory_config": ttnn.L1_MEMORY_CONFIG},
        ),
    ],
    ids=[
        "kv_nope_to_interleaved",
        "kv_rope_reshard",
        "kv_rope_out_reshard",
        "kvpe_reshard",
        "q_rope_out_reshard",
        "flash_mla_reshard",
        "flash_mla_out_reshard",
    ],
)
@pytest.mark.parametrize("warmup_iters", [10])
@pytest.mark.parametrize("num_iters", [100])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "trace_region_size": 550912,
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_deepseek_v3_mla_to_memory_config_trace_mode(
    device,
    batch_size,
    op_name,
    input_shape,
    src_mem_config,
    dst_mem_config_dict,
    warmup_iters,
    num_iters,
):
    """
    Test the to_memory_config operations from mla1d.py with trace mode.

    These operations convert tensors between different memory layouts:
    1. kv_nope_to_interleaved (line 1148): Width sharded → L1 interleaved
    2. kv_rope_reshard (line 1155-1156): L1 interleaved → Width sharded
    3. kv_rope_out_reshard (line 1167): Width sharded → L1 interleaved
    4. kvpe_reshard (line 1187): L1 interleaved → Height sharded
    5. q_rope_out_reshard (line 1259): Width sharded → L1 interleaved
    6. flash_mla_reshard (line 1273): L1 interleaved → Height sharded
    7. flash_mla_out_reshard (line 1301): Height sharded → L1 interleaved

    Configuration:
    - Warmup iterations: 10
    - Test iterations: 100
    - Trace mode: Enabled
    - Various memory layouts: Width/Height sharded, L1 interleaved
    """
    torch.manual_seed(0)

    # Create random tensor for input
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)

    # Golden output - to_memory_config doesn't change values, just layout
    torch_output_tensor = torch_input_tensor.clone()

    # Helper function to create memory config from dict
    def create_memory_config(config_dict):
        if isinstance(config_dict, dict):
            if "memory_layout" in config_dict:
                # Create sharded memory config
                shard_spec_dict = config_dict["shard_spec"]
                num_cores_y, num_cores_x = shard_spec_dict["grid"]
                num_cores = num_cores_x * num_cores_y

                # Use ttnn.num_cores_to_corerangeset to avoid dispatch cores
                grid_size = device.compute_with_storage_grid_size()
                shard_grid_set = ttnn.num_cores_to_corerangeset(num_cores, grid_size, row_wise=True)

                shard_spec = ttnn.ShardSpec(
                    shard_grid_set,
                    shard_spec_dict["shape"],
                    shard_spec_dict["orientation"],
                )
                return ttnn.MemoryConfig(
                    config_dict["memory_layout"],
                    config_dict["buffer_type"],
                    shard_spec,
                )
            else:
                # It's a nested dict with "memory_config" key
                return create_memory_config(config_dict["memory_config"])
        else:
            # It's already a memory config object
            return config_dict

    # Create ttnn tensor with source memory config
    if isinstance(src_mem_config, dict) and "memory_config" in src_mem_config:
        # Simple L1 interleaved case
        tt_input_tensor = ttnn.from_torch(
            torch_input_tensor,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=src_mem_config["memory_config"],
        )
    else:
        # Create with default config first, then convert to sharded
        tt_input_tensor = ttnn.from_torch(
            torch_input_tensor,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # Convert to the source memory config
        src_mem_cfg = create_memory_config(src_mem_config)
        tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, src_mem_cfg)

    # Get destination memory config
    if "memory_config" in dst_mem_config_dict:
        if isinstance(dst_mem_config_dict["memory_config"], dict):
            dst_mem_cfg = create_memory_config(dst_mem_config_dict["memory_config"])
        else:
            dst_mem_cfg = dst_mem_config_dict["memory_config"]
    else:
        dst_mem_cfg = create_memory_config(dst_mem_config_dict)

    # Compile run
    logger.info(f"Compiling to_memory_config operation: {op_name}")
    logger.info(f"  Input shape: {input_shape}")
    logger.info(f"  Source memory config: {src_mem_config}")
    logger.info(f"  Destination memory config: {dst_mem_config_dict}")

    tt_output_tensor = ttnn.to_memory_config(tt_input_tensor, dst_mem_cfg)
    ttnn.synchronize_device(device)

    # Capture warmup trace
    logger.info(f"Capturing warmup trace with {warmup_iters} iterations")
    trace_id_warmup = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(warmup_iters):
        tt_output_tensor = ttnn.to_memory_config(tt_input_tensor, dst_mem_cfg)
    ttnn.end_trace_capture(device, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(device)

    # Capture main trace
    logger.info(f"Capturing main trace with {num_iters} iterations")
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for i in range(num_iters):
        tt_output_tensor = ttnn.to_memory_config(tt_input_tensor, dst_mem_cfg)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)

    # Execute warmup trace
    logger.info("Executing warmup trace")
    profiler = BenchmarkProfiler()
    profiler.start("warmup")
    ttnn.execute_trace(device, trace_id_warmup, blocking=False)
    ttnn.release_trace(device, trace_id_warmup)
    profiler.end("warmup")
    ttnn.synchronize_device(device)

    # Execute main trace with signposts
    logger.info("Executing main trace")
    signpost("start")
    profiler.start("main")
    ttnn.execute_trace(device, trace_id, blocking=False)
    ttnn.release_trace(device, trace_id)
    profiler.end("main")
    signpost("stop")
    ttnn.synchronize_device(device)

    # Verify correctness
    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    torch_output_from_tt = ttnn.to_torch(tt_output_tensor)

    assert (
        torch_output_from_tt.shape == torch_output_tensor.shape
    ), f"Shape mismatch: {torch_output_from_tt.shape} != {torch_output_tensor.shape}"

    assert_equal(torch_output_tensor, torch_output_from_tt)

    logger.info(f"✓ Trace mode {op_name} test passed with correct output")
