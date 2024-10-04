# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from ttnn import experimental as ttl
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor, ReplicateTensorToMesh
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from models.utility_functions import skip_for_grayskull, skip_for_wormhole_b0
from models.utility_functions import nearest_32
import math

shard_spec_48_cores_grid = ttnn.CoreRangeSet(
    {
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(7, 5),
        ),
    }
)
shard_spec_36_cores_grid = ttnn.CoreRangeSet(
    {
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(5, 5),
        ),
    }
)
shard_spec_32_cores_grid = ttnn.CoreRangeSet(
    {
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(7, 3),
        ),
    }
)
shard_spec_28_cores_grid = ttnn.CoreRangeSet(
    {
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(6, 3),
        ),
    }
)
shard_spec_24_cores_grid = ttnn.CoreRangeSet(
    {
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(7, 2),
        ),
    }
)
shard_spec_20_cores_grid = ttnn.CoreRangeSet(
    {
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(4, 3),
        ),
    }
)
shard_spec_16_cores_grid = ttnn.CoreRangeSet(
    {
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(7, 1),
        ),
    }
)
shard_spec_12_cores_grid = ttnn.CoreRangeSet(
    {
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(5, 1),
        ),
    }
)
shard_spec_8_cores_grid = ttnn.CoreRangeSet(
    {
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(7, 0),
        ),
    }
)
shard_spec_1_cores_grid = ttnn.CoreRangeSet(
    {
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(0, 0),
        ),
    }
)

TILE_SIZE = 32
shard_height = TILE_SIZE  # Decode mode only
cluster_size = (4, 8)
hidden_size = 8192

hidden_size_tg = hidden_size // cluster_size[0]
hidden_size_48_pad = (hidden_size + (cluster_size[0] * 48 * TILE_SIZE * 2 // 3)) // cluster_size[0]
hidden_size_24_pad = (hidden_size + (cluster_size[0] * 24 * TILE_SIZE // 3)) // cluster_size[0]
hidden_size_12_pad = (hidden_size + (cluster_size[0] * 12 * TILE_SIZE * 2 // 3)) // cluster_size[0]

shard_width_hidden_dim_across_48_cores = hidden_size_48_pad // 48
shard_width_hidden_dim_across_36_cores = hidden_size_24_pad // 36
shard_width_hidden_dim_across_32_cores = hidden_size_tg // 32
shard_width_hidden_dim_across_24_cores = hidden_size_24_pad // 24
shard_width_hidden_dim_across_16_cores = hidden_size_tg // 16
shard_width_hidden_dim_across_12_cores = hidden_size_12_pad // 12
shard_width_hidden_dim_across_8_cores = hidden_size_tg // 8


def run_prefetch_matmul_on_t3000_impl(
    t3k_mesh_device,
    input_shape,
    input_dtype,
    layout,
    # Matmul params
    N,
    weight_shard_dim,
    core_grid,
    matmul_config,
    matmul_weights_dtype,
    max_in0_block_w,
    math_fidelity,
    # Memory configs
    mem_config_input,
    mem_config_weights,
    mem_config_mm,
    num_iters=1,
    enable_trace=False,
):
    devices = t3k_mesh_device.get_devices()
    num_devices = len(devices)

    ##### Create input tensor for the all gather #####
    _, _, M, K = input_shape

    # Add padding to create a dummy tensor to overcome PCC issue for in-place CB
    padded_K = K * (mem_config_input.shard_spec.num_cores() - 1)
    input_tensor_padding = torch.randn([1, 1, M, padded_K]).float()
    _ = ttnn.as_tensor(
        input_tensor_padding,
        dtype=input_dtype,
        layout=layout,
        device=t3k_mesh_device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_24_cores_grid,
                [
                    M,
                    padded_K // mem_config_input.shard_spec.num_cores(),
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        ),
        mesh_mapper=ReplicateTensorToMesh(t3k_mesh_device),
    )

    input_tensor = torch.randn(input_shape).float()
    tt_input_tensor = ttnn.as_tensor(
        input_tensor,
        dtype=input_dtype,
        layout=layout,
        device=t3k_mesh_device,
        memory_config=mem_config_input,
        mesh_mapper=ReplicateTensorToMesh(t3k_mesh_device),
    )
    logger.info(f"Input tensor shape: {tt_input_tensor.shape}")

    ##### Config for the weight matrix #####
    if mem_config_weights == "dram_sharded_ff1":
        device = t3k_mesh_device.get_device(0)
        weight_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
                )
            }
        )
        shard_shape = (K, nearest_32(N // 12))  # padded cols to divide by 12
        shard_spec = ttnn.ShardSpec(weight_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
        mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

        mem_config_weights = mem_config
    elif mem_config_weights == "dram_sharded_ff2":
        device = t3k_mesh_device.get_device(0)
        weight_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
                )
            }
        )
        shard_shape = (K, nearest_32(N // 12))  # padded cols to divide by 12
        shard_spec = ttnn.ShardSpec(weight_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
        mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

        mem_config_weights = mem_config
    elif mem_config_weights == "dram":
        mem_config_weights = ttnn.DRAM_MEMORY_CONFIG
    elif mem_config_weights == "l1":
        mem_config_weights = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_24_cores_grid,
                [
                    K,
                    N // mem_config_input.shard_spec.num_cores(),
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    else:
        raise ValueError(f"Unsupported mem_config_weights: {mem_config_weights}")

    ##### Create the weight matrix for the matmul #####
    weights_tensor = torch.randn([1, 1, K, N * num_devices]).float()
    weight_tt = ttnn.as_tensor(
        weights_tensor,
        dtype=matmul_weights_dtype,
        layout=layout,
        device=t3k_mesh_device,
        memory_config=mem_config_weights,
        mesh_mapper=ShardTensorToMesh(t3k_mesh_device, dim=weight_shard_dim),
    )
    logger.info(f"Weight tensor shape: {weight_tt.shape}")

    ##### Configs for ttnn.matmul #####
    if matmul_config == "matmul_1d":
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=core_grid,
            in0_block_w=min(max_in0_block_w, K // 32 // core_grid[0]),  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=max(1, input_shape[2] // 32 // core_grid[1]),  # M / TILE_HEIGHT / Grid_Size
            per_core_N=1
            if mem_config_input.is_sharded()
            else max(1, N // 32 // core_grid[0]),  # N / TILE_WIDTH / Grid_Size
            mcast_in0=True,
            fused_activation=None,  # ttnn.UnaryOpType.SILU,
            fuse_batch=True,
        )
    elif matmul_config == "matmul_1d_ff1":
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=core_grid,
            in0_block_w=max(1, math.ceil(K / 32)),  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=5,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=shard_height // 32,  # M / TILE_HEIGHT / Grid_Size
            per_core_N=max(1, math.ceil(N / 32 / (core_grid[0] * core_grid[1]))),  # N / TILE_WIDTH / Grid_Size
            mcast_in0=True,
            gather_in0=True,
            fused_activation=None,  # ttnn.UnaryOpType.SILU,
            fuse_batch=True,
        )
    elif matmul_config == "matmul_dram_sharded_ff1":
        program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=K
            // mem_config_input.shard_spec.num_cores()
            // 32,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            per_core_M=M // 32,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=math.ceil(
                N / mem_config_input.shard_spec.num_cores() / 32
            ),  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
            fused_activation=None,
        )
    elif matmul_config == "matmul_dram_sharded_ff2":
        program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=K
            // mem_config_input.shard_spec.num_cores()
            // 32,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            per_core_M=M // 32,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=math.ceil(
                N / mem_config_input.shard_spec.num_cores() / 32
            ),  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
            fused_activation=None,
        )
    elif matmul_config == "matmul_qkv":
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 5),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    elif matmul_config == "matmul_do":
        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,  # (32 x 8k) x (8k x 1k) = (32 x 1k)
            out_subblock_h=1,
            out_subblock_w=1,  # TODO: Maximize
            per_core_M=shard_height // 32,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    elif matmul_config == "matmul_2d":
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=core_grid,
            in0_block_w=min(max_in0_block_w, K // 32 // core_grid[0]),  # how much inner dim you take each time
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=max(1, input_shape[2] // 32 // core_grid[1]),  # M / TILE_HEIGHT / Grid_Size
            per_core_N=max(1, N // 32 // core_grid[0]),  # N / TILE_WIDTH / Grid_Size
            transpose_mcast=False,
            fused_activation=None,  # ttnn.UnaryOpType.SILU,
            fuse_batch=False,
        )
    elif matmul_config == "auto":
        program_config = None
    else:
        raise ValueError(f"Unsupported matmul_config: {matmul_config}")

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=math_fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    ##### Perform the torch ops #####
    matmul_output = torch.matmul(input_tensor, weights_tensor)

    ##### Perform the TT ops #####
    def run_op():
        if core_grid is None or isinstance(core_grid, tuple):
            return ttnn.matmul(
                tt_input_tensor,
                weight_tt,
                memory_config=mem_config_mm,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
            )
        else:
            return ttnn.matmul(
                tt_input_tensor,
                weight_tt,
                core_grid=core_grid,
                memory_config=mem_config_mm,
                compute_kernel_config=compute_kernel_config,
            )

    if enable_trace:
        # Compile the op
        tt_matmul_out_tensor = run_op()
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(t3k_mesh_device, cq_id=0)
        for i in range(num_iters):
            tt_matmul_out_tensor = run_op()
        ttnn.end_trace_capture(t3k_mesh_device, trace_id, cq_id=0)

        logger.info(f"Done capturing trace")

        # Execute trace
        ttnn.execute_trace(t3k_mesh_device, trace_id, cq_id=0, blocking=False)
        logger.info(f"Done executing trace")

        # Synchronize the devices
        for d in devices:
            ttnn.synchronize_device(d)
    else:
        for i in range(num_iters):
            tt_matmul_out_tensor = run_op()

            # Synchronize the devices
            for d in devices:
                ttnn.synchronize_device(d)

            logger.info(f"Done iteration {i}")

    print("Checking outputs for Matmul")
    tt_mm_out = ttnn.from_device(tt_matmul_out_tensor)
    tt_mm_out = ttnn.to_torch(tt_mm_out, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=weight_shard_dim))

    eq, output = comp_pcc(tt_mm_out, matmul_output)
    logger.info(f"Output: {output}")
    assert eq


@pytest.mark.parametrize(
    "layout",
    [
        ttnn.TILE_LAYOUT,
    ],
)
@pytest.mark.parametrize(
    "input_dtype, matmul_weights_dtype, math_fidelity",
    [
        (ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi),
        # (ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.HiFi2),
        # (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),
        # (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
        # (ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi),
        # (ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.MathFidelity.HiFi2),
        # (ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),
        # (ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
    ],
)
@pytest.mark.parametrize(
    "matmul_config, input_shape, N, weight_shard_dim, core_grid, max_in0_block_w, mem_config_input, mem_config_weights, mem_config_mm",
    [
        # (  # FF1/3 matmul1d
        #     "matmul_1d_ff1",
        #     [1, 1, 32, hidden_size_tg],
        #     3584,
        #     3,
        #     (8, 1),
        #     4,
        #     ttnn.MemoryConfig(
        #         ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #         ttnn.BufferType.L1,
        #         ttnn.ShardSpec(
        #             shard_spec_1_cores_grid,
        #             [
        #                 shard_height,
        #                 hidden_size_tg,
        #             ],
        #             ttnn.ShardOrientation.ROW_MAJOR,
        #             False,
        #         ),
        #     ),
        #     "dram",
        #     ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        # ),
        # (  # FF1/3 matmul1d
        #     "matmul_1d_ff1",
        #     [1, 1, 32, hidden_size_tg],
        #     3584,
        #     3,
        #     (8, 1),
        #     4,
        #     ttnn.MemoryConfig(
        #         ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #         ttnn.BufferType.L1,
        #         ttnn.ShardSpec(
        #             shard_spec_8_cores_grid,
        #             [
        #                 shard_height,
        #                 shard_width_hidden_dim_across_8_cores,
        #             ],
        #             ttnn.ShardOrientation.ROW_MAJOR,
        #             False,
        #         ),
        #     ),
        #     "dram",
        #     ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        # ),
        # (  # FF1/3 matmul1d
        #     "matmul_1d_ff1",
        #     [1, 1, 32, hidden_size_12_pad],
        #     3584,
        #     3,
        #     (6, 2),
        #     4,
        #     ttnn.MemoryConfig(
        #         ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #         ttnn.BufferType.L1,
        #         ttnn.ShardSpec(
        #             shard_spec_12_cores_grid,
        #             [
        #                 shard_height,
        #                 shard_width_hidden_dim_across_12_cores,
        #             ],
        #             ttnn.ShardOrientation.ROW_MAJOR,
        #             False,
        #         ),
        #     ),
        #     "dram",
        #     ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        # ),
        # (  # FF1/3 matmul1d
        #     "matmul_1d_ff1",
        #     [1, 1, 32, hidden_size_tg],
        #     3584,
        #     3,
        #     (8, 2),
        #     4,
        #     ttnn.MemoryConfig(
        #         ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #         ttnn.BufferType.L1,
        #         ttnn.ShardSpec(
        #             shard_spec_16_cores_grid,
        #             [
        #                 shard_height,
        #                 shard_width_hidden_dim_across_16_cores,
        #             ],
        #             ttnn.ShardOrientation.ROW_MAJOR,
        #             False,
        #         ),
        #     ),
        #     "dram",
        #     ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        # ),
        (  # FF1/3 matmul1d
            "matmul_1d_ff1",
            [1, 1, 32, hidden_size_24_pad],
            3584 + 256,
            3,
            (8, 3),
            4,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    shard_spec_24_cores_grid,
                    [
                        shard_height,
                        shard_width_hidden_dim_across_24_cores,
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            ),
            "l1",
            ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        ),
        # (  # FF1/3 matmul1d
        #     "matmul_1d_ff1",
        #     [1, 1, 32, hidden_size_tg],
        #     3584,
        #     3,
        #     (8, 4),
        #     4,
        #     ttnn.MemoryConfig(
        #         ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #         ttnn.BufferType.L1,
        #         ttnn.ShardSpec(
        #             shard_spec_32_cores_grid,
        #             [
        #                 shard_height,
        #                 shard_width_hidden_dim_across_32_cores,
        #             ],
        #             ttnn.ShardOrientation.ROW_MAJOR,
        #             False,
        #         ),
        #     ),
        #     "dram",
        #     ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        # ),
        # (  # FF1/3 matmul1d
        #     "matmul_1d_ff1",
        #     [1, 1, 32, hidden_size_tg],
        #     3584,
        #     3,
        #     (6, 6),
        #     4,
        #     ttnn.MemoryConfig(
        #         ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #         ttnn.BufferType.L1,
        #         ttnn.ShardSpec(
        #             shard_spec_36_cores_grid,
        #             [
        #                 shard_height,
        #                 shard_width_hidden_dim_across_36_cores,
        #             ],
        #             ttnn.ShardOrientation.ROW_MAJOR,
        #             False,
        #         ),
        #     ),
        #     "dram",
        #     ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        # ),
        # (  # FF1/3 matmul1d
        #     "matmul_1d_ff1",
        #     [1, 1, 32, hidden_size_tg],
        #     3584,
        #     3,
        #     (8, 6),
        #     4,
        #     ttnn.MemoryConfig(
        #         ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #         ttnn.BufferType.L1,
        #         ttnn.ShardSpec(
        #             shard_spec_48_cores_grid,
        #             [
        #                 shard_height,
        #                 shard_width_hidden_dim_across_48_cores,
        #             ],
        #             ttnn.ShardOrientation.ROW_MAJOR,
        #             False,
        #         ),
        #     ),
        #     "dram",
        #     ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        # ),
        # (  # FF1/3 dram sharded
        #     "matmul_dram_sharded_ff1",
        #     [1, 1, 32, hidden_size_tg],
        #     3584,
        #     3,
        #     None,
        #     4,
        #     ttnn.MemoryConfig(
        #         ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #         ttnn.BufferType.L1,
        #         ttnn.ShardSpec(
        #             shard_spec_8_cores_grid,
        #             [
        #                 shard_height,
        #                 shard_width_hidden_dim_across_8_cores,
        #             ],
        #             ttnn.ShardOrientation.ROW_MAJOR,
        #             False,
        #         ),
        #     ),
        #     "dram_sharded_ff1",
        #     ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        # ),
        # (  # FF1/3 dram sharded
        #     "matmul_dram_sharded_ff1",
        #     [1, 1, 32, hidden_size_12_pad],
        #     3584,
        #     3,
        #     None,
        #     4,
        #     ttnn.MemoryConfig(
        #         ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #         ttnn.BufferType.L1,
        #         ttnn.ShardSpec(
        #             shard_spec_12_cores_grid,
        #             [
        #                 shard_height,
        #                 shard_width_hidden_dim_across_12_cores,
        #             ],
        #             ttnn.ShardOrientation.ROW_MAJOR,
        #             False,
        #         ),
        #     ),
        #     "dram_sharded_ff1",
        #     ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        # ),
        # (  # FF1/3 dram sharded
        #     "matmul_dram_sharded_ff1",
        #     [1, 1, 32, hidden_size_tg],
        #     3584,
        #     3,
        #     None,
        #     4,
        #     ttnn.MemoryConfig(
        #         ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #         ttnn.BufferType.L1,
        #         ttnn.ShardSpec(
        #             shard_spec_16_cores_grid,
        #             [
        #                 shard_height,
        #                 shard_width_hidden_dim_across_16_cores,
        #             ],
        #             ttnn.ShardOrientation.ROW_MAJOR,
        #             False,
        #         ),
        #     ),
        #     "dram_sharded_ff1",
        #     ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        # ),
        # (  # FF1/3 dram sharded
        #     "matmul_dram_sharded_ff1",
        #     [1, 1, 32, hidden_size_24_pad],
        #     3584,
        #     3,
        #     None,
        #     4,
        #     ttnn.MemoryConfig(
        #         ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #         ttnn.BufferType.L1,
        #         ttnn.ShardSpec(
        #             shard_spec_24_cores_grid,
        #             [
        #                 shard_height,
        #                 shard_width_hidden_dim_across_24_cores,
        #             ],
        #             ttnn.ShardOrientation.ROW_MAJOR,
        #             False,
        #         ),
        #     ),
        #     "dram_sharded_ff1",
        #     ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        # ),
        # (  # FF1/3 dram sharded
        #     "matmul_dram_sharded_ff1",
        #     [1, 1, 32, hidden_size_tg],
        #     3584,
        #     3,
        #     None,
        #     4,
        #     ttnn.MemoryConfig(
        #         ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #         ttnn.BufferType.L1,
        #         ttnn.ShardSpec(
        #             shard_spec_32_cores_grid,
        #             [
        #                 shard_height,
        #                 shard_width_hidden_dim_across_32_cores,
        #             ],
        #             ttnn.ShardOrientation.ROW_MAJOR,
        #             False,
        #         ),
        #     ),
        #     "dram_sharded_ff1",
        #     ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        # ),
        # (  # FF1/3 dram sharded
        #     "matmul_dram_sharded_ff1",
        #     [1, 1, 32, hidden_size_tg],
        #     3584,
        #     3,
        #     None,
        #     4,
        #     ttnn.MemoryConfig(
        #         ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #         ttnn.BufferType.L1,
        #         ttnn.ShardSpec(
        #             shard_spec_36_cores_grid,
        #             [
        #                 shard_height,
        #                 shard_width_hidden_dim_across_36_cores,
        #             ],
        #             ttnn.ShardOrientation.ROW_MAJOR,
        #             False,
        #         ),
        #     ),
        #     "dram_sharded_ff1",
        #     ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        #     ttnn.bfloat16,
        #     ttnn.bfloat4_b,
        # ),
        # (  # FF1/3 dram sharded
        #     "matmul_dram_sharded_ff1",
        #     [1, 1, 32, hidden_size_tg],
        #     3584,
        #     3,
        #     None,
        #     4,
        #     ttnn.MemoryConfig(
        #         ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #         ttnn.BufferType.L1,
        #         ttnn.ShardSpec(
        #             shard_spec_48_cores_grid,
        #             [
        #                 shard_height,
        #                 shard_width_hidden_dim_across_48_cores,
        #             ],
        #             ttnn.ShardOrientation.ROW_MAJOR,
        #             False,
        #         ),
        #     ),
        #     "dram_sharded_ff1",
        #     ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        #     ttnn.bfloat16,
        #     ttnn.bfloat4_b,
        # ),
        # (  # FF2 Decode
        #     "matmul_dram_sharded_ff2",
        #     [1, 1, 32, (1024 * 28) // 8], # 28k / 8 devices, for 24 pad: + (8 * 32 * 24 // 3), for
        #     hidden_size // 4,
        #     3,
        #     None,
        #     4,
        #     ttnn.MemoryConfig(
        #         ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #         ttnn.BufferType.L1,
        #         ttnn.ShardSpec(
        #             shard_spec_8_cores_grid,
        #             [
        #                 shard_height,
        #                 (1024 * 28) // 8 // 8, # 8k / # devices / # cores
        #             ],
        #             ttnn.ShardOrientation.ROW_MAJOR,
        #             False,
        #         ),
        #     ),
        #     "dram_sharded_ff2",
        #     ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        #     ttnn.bfloat16,
        #     ttnn.bfloat8_b,
        # ),
        # (  # QKV Decode
        #     "matmul_qkv",
        #     [1, 1, 32, hidden_size_tg],
        #     1024 * 10 // 8,
        #     3,
        #     None,
        #     4,
        #     ttnn.MemoryConfig(
        #         ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #         ttnn.BufferType.L1,
        #         ttnn.ShardSpec(
        #             shard_spec_32_cores_grid,
        #             [
        #                 shard_height,
        #                 8192 // 32 // 4,
        #             ],
        #             ttnn.ShardOrientation.ROW_MAJOR,
        #             False,
        #         ),
        #     ),
        #     "dram",
        #     ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        #     ttnn.bfloat16,
        #     ttnn.bfloat8_b,
        # ),
        # (  # DO Decode
        #     "matmul_do",
        #     [1, 1, 32, hidden_size_tg],
        #     8192 // 8,
        #     3,
        #     ttnn.CoreGrid(y=4, x=8),
        #     4,
        #     ttnn.DRAM_MEMORY_CONFIG,
        #     "dram",
        #     ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        #     ttnn.bfloat16,
        #     ttnn.bfloat8_b,
        # ),
    ],
    ids=[
        # "ff1_matmul1d_1",
        # "ff1_matmul1d_8",
        # "ff1_matmul1d_12",
        # "ff1_matmul1d_16",
        "ff1_matmul1d_24",
        # "ff1_matmul1d_32",
        # "ff1_matmul1d_36",
        # "ff1_matmul1d_48",
        # "ff1_dram_sharded_8",
        # "ff1_dram_sharded_12",
        # "ff1_dram_sharded_16",
        # "ff1_dram_sharded_24",
        # "ff1_dram_sharded_32",
        # "ff1_dram_sharded_36",
        # "ff1_dram_sharded_48",
        # "ff2_dram_sharded",
        # "qkv_dram_sharded",
        # "do_matmul_1d",
    ],
)
@pytest.mark.parametrize(
    "enable_async",
    [
        True,
        # False,
    ],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 90112}], indirect=True)
def test_prefetch_matmul_on_t3000(
    t3k_mesh_device,
    input_shape,
    input_dtype,
    layout,
    N,
    weight_shard_dim,
    core_grid,
    matmul_config,
    matmul_weights_dtype,
    max_in0_block_w,
    math_fidelity,
    mem_config_input,
    mem_config_weights,
    mem_config_mm,
    use_program_cache,
    function_level_defaults,
    enable_async,
):
    run_prefetch_matmul_on_t3000_impl(
        t3k_mesh_device,
        input_shape,
        input_dtype,
        layout,
        N,
        weight_shard_dim,
        core_grid,
        matmul_config,
        matmul_weights_dtype,
        max_in0_block_w,
        math_fidelity,
        mem_config_input,
        mem_config_weights,
        mem_config_mm,
    )
