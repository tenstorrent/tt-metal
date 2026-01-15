# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.tests.unit.utils import run_test
from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_HIFI2,
    COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
    COMPUTE_KERNEL_CONFIG_LOFI,
    dram_sharded_weight_config,
    get_activation_sharding_core_counts_for_dram_matmul,
    get_dram_sharded_matmul_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

# =============================================================================
# Reference tests for matmuls with no program config
# All of the following are using INTERLEAVED memory layout
#
# Test cases (see test_matmul_interleaved):
#   - InputA: (1, 1, 32, 896),   InputB: (1, 1, 896, 1536)   - HiFi2, DRAM
#   - InputA: (1, 1, 32, 1536),  InputB: (1, 1, 1536, 3072)  - HiFi2, DRAM
#   - InputA: (1, 16, 32, 128),  InputB: (1, 16, 128, 512)   - HiFi2, DRAM, batched
#   - InputA: (1, 1, 32, 896),   InputB: (1, 1, 896, 576)    - HiFi2, DRAM
#   - InputA: (1, 16, 32, 512),  InputB: (1, 16, 512, 128)   - HiFi2, DRAM, batched
#   - InputA: (1, 1, 32, 16384), InputB: (1, 1, 16384, 896)  - HiFi2, DRAM
#   - InputA: (1, 1, 32, 7168),  InputB: (1, 1, 7168, 256)   - HiFi2, L1, fp32_acc
#   - InputA: (1, 8, 128, 7168), InputB: (1, 8, 7168, 2048)  - LoFi, L1, batched
#   - InputA: (1, 8, 128, 2048), InputB: (1, 8, 2048, 7168)  - LoFi, L1, batched
# =============================================================================


def run_test_matmul_dram_sharded(
    device,
    M,
    K,
    N,
    in0_dtype,
    in1_dtype,
    out_dtype,
):
    """
    Test matmul with DRAM sharded weights using MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig.

    This test mirrors the model's approach in MLP.decode_model_config:
    - InputA (activation) is L1 width sharded
    - InputB (weights) is DRAM width sharded
    - Uses the same helper functions as the model
    """
    # Get device grid info (same as model does)
    max_num_cores = device.compute_with_storage_grid_size().x * device.compute_with_storage_grid_size().y
    dram_grid_size = device.dram_grid_size()

    # Calculate core counts using the same function as the model
    # Input tensor shards across K dimension, output across N dimension
    input_num_cores = max(get_activation_sharding_core_counts_for_dram_matmul(K, max_num_cores))
    output_num_cores = max(get_activation_sharding_core_counts_for_dram_matmul(N, max_num_cores))

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]

    logger.info(f"Testing M={M}, K={K}, N={N}")
    logger.info(
        f"max_num_cores={max_num_cores}, input_num_cores={input_num_cores}, output_num_cores={output_num_cores}"
    )

    # Get program config using the model's helper function
    program_config = get_dram_sharded_matmul_config(M, K, N, input_num_cores, output_num_cores)
    logger.info(
        f"Program config: in0_block_w={program_config.in0_block_w}, per_core_M={program_config.per_core_M}, per_core_N={program_config.per_core_N}"
    )

    # Get weight memory config using the model's helper function
    in1_mem_config = dram_sharded_weight_config(K, N, dram_grid_size)
    logger.info(f"Weight memory config: {in1_mem_config}")

    # Create input activation memory config (L1 width sharded) - same as model
    in0_shard_shape = (
        ttnn.core.roundup(M, ttnn.TILE_SIZE),
        ttnn.core.roundup(K // input_num_cores, ttnn.TILE_SIZE),
    )
    in0_mem_config = ttnn.create_sharded_memory_config_(
        shape=in0_shard_shape,
        core_grid=ttnn.num_cores_to_corerangeset(
            input_num_cores,
            ttnn.CoreCoord(device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y),
            row_wise=True,
        ),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    logger.info(f"Activation input memory config: {in0_mem_config}")

    # Output memory config (L1 width sharded) - same as model
    out_shard_shape = (
        ttnn.core.roundup(M, ttnn.TILE_SIZE),
        ttnn.core.roundup(N // output_num_cores, ttnn.TILE_SIZE),
    )
    out_mem_config = ttnn.create_sharded_memory_config_(
        shape=out_shard_shape,
        core_grid=ttnn.num_cores_to_corerangeset(
            output_num_cores,
            ttnn.CoreCoord(device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y),
            row_wise=True,
        ),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    # Interleaved config for initial tensor creation and final comparison
    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )

    # Create input tensors
    torch.manual_seed(1234)
    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    # Convert activation to TT tensor and shard to L1
    in0_t = ttnn.from_torch(
        in0, dtype=in0_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=interleaved_mem_config
    )
    in0_t = ttnn.to_memory_config(in0_t, in0_mem_config)

    # Convert weight to TT tensor with DRAM sharding (using model's memory config)
    in1_t = ttnn.from_torch(in1, dtype=in1_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in1_mem_config)

    # Run linear (same as model uses ttnn.linear)
    output_t = ttnn.linear(
        in0_t,
        in1_t,
        program_config=program_config,
        memory_config=out_mem_config,
        dtype=out_dtype,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
    )

    # Convert back to interleaved for comparison
    output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)

    # Reference computation
    pt_out = in0 @ in1

    # Get output and compare
    tt_out = ttnn.to_torch(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


# =============================================================================
# Reference testcases the DRAM sharded matmuls
# All inputs are width sharded. Activations are in L1, weights are in DRAM.
#
# Test cases (see test_matmul_dram_sharded):
#   - InputA: (1, 1, 32, 7168), InputB: (1, 1, 7168, 2304) - in0_block_w=4, per_core_M=1, per_core_N=1
#   - InputA: (1, 1, 32, 2304), InputB: (1, 1, 2304, 7168) - in0_block_w=1, per_core_M=1, per_core_N=8
#   - InputA: (1, 1, 32, 7168), InputB: (1, 1, 7168, 256)  - in0_block_w=4, per_core_M=1, per_core_N=1
#   - InputA: (1, 1, 32, 256),  InputB: (1, 1, 256, 7168)  - in0_block_w=1, per_core_M=1, per_core_N=8
#   - InputA: (1, 1, 32, 7168), InputB: (1, 1, 7168, 4064) - in0_block_w=4, per_core_M=1, per_core_N=127
# =============================================================================


@pytest.mark.parametrize(
    "M, K, N",
    [
        # InputA: (1, 1, 32, 7168), InputB: (1, 1, 7168, 2304) - MLP w1/w3 style
        (32, 7168, 2304),
        # InputA: (1, 1, 32, 2304), InputB: (1, 1, 2304, 7168) - MLP w2 style
        (32, 2304, 7168),
        # InputA: (1, 1, 32, 7168), InputB: (1, 1, 7168, 256) - smaller output
        (32, 7168, 256),
        # InputA: (1, 1, 32, 256), InputB: (1, 1, 256, 7168) - smaller input
        (32, 256, 7168),
        # InputA: (1, 1, 32, 7168), InputB: (1, 1, 7168, 4064) - LM head style
        (32, 7168, 4064),
    ],
    ids=[
        "32x7168x2304_mlp_w1_style",
        "32x2304x7168_mlp_w2_style",
        "32x7168x256_small_output",
        "32x256x7168_small_input",
        "32x7168x4064_lm_head_style",
    ],
)
@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat16),
    ],
    ids=["bf16_bf4b_bf16"],
)
@pytest.mark.requires_device(["N150", "N300", "T3K", "TG", "DUAL", "QUAD"])
def test_matmul_dram_sharded_single_device(
    device,
    M,
    K,
    N,
    in0_dtype,
    in1_dtype,
    out_dtype,
):
    run_test_matmul_dram_sharded(
        device,
        M,
        K,
        N,
        in0_dtype,
        in1_dtype,
        out_dtype,
    )


@pytest.mark.parametrize("mesh_device", [(8, 8)], indirect=True)
@pytest.mark.parametrize(
    "M, K, N",
    [
        # InputA: (1, 1, 32, 7168), InputB: (1, 1, 7168, 2304) - MLP w1/w3 style
        (32, 7168, 2304),
        # InputA: (1, 1, 32, 2304), InputB: (1, 1, 2304, 7168) - MLP w2 style
        (32, 2304, 7168),
        # InputA: (1, 1, 32, 7168), InputB: (1, 1, 7168, 256) - smaller output
        (32, 7168, 256),
        # InputA: (1, 1, 32, 256), InputB: (1, 1, 256, 7168) - smaller input
        (32, 256, 7168),
        # InputA: (1, 1, 32, 7168), InputB: (1, 1, 7168, 4064) - LM head style
        (32, 7168, 4064),
    ],
    ids=[
        "32x7168x2304_mlp_w1_style",
        "32x2304x7168_mlp_w2_style",
        "32x7168x256_small_output",
        "32x256x7168_small_input",
        "32x7168x4064_lm_head_style",
    ],
)
@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, out_dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat16),
    ],
    ids=["bf16_bf4b_bf16"],
)
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 90112, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
def test_matmul_dram_sharded_mesh_device(
    mesh_device,
    M,
    K,
    N,
    in0_dtype,
    in1_dtype,
    out_dtype,
    enable_trace,
    device_params,
):
    """
    Mesh device test for matmul with DRAM sharded weights.
    """
    # Get device grid info from mesh_device
    grid = mesh_device.compute_with_storage_grid_size()
    max_num_cores = grid.x * grid.y
    dram_grid_size = mesh_device.dram_grid_size()

    input_num_cores = max(get_activation_sharding_core_counts_for_dram_matmul(K, max_num_cores))
    output_num_cores = max(get_activation_sharding_core_counts_for_dram_matmul(N, max_num_cores))

    in0_shape = [1, 1, M, K]
    in1_shape = [1, 1, K, N]

    program_config = get_dram_sharded_matmul_config(M, K, N, input_num_cores, output_num_cores)
    in1_mem_config = dram_sharded_weight_config(K, N, dram_grid_size)

    in0_shard_shape = (
        ttnn.core.roundup(M, ttnn.TILE_SIZE),
        ttnn.core.roundup(K // input_num_cores, ttnn.TILE_SIZE),
    )
    in0_mem_config = ttnn.create_sharded_memory_config_(
        shape=in0_shard_shape,
        core_grid=ttnn.num_cores_to_corerangeset(
            input_num_cores,
            ttnn.CoreCoord(grid.x, grid.y),
            row_wise=True,
        ),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    out_shard_shape = (
        ttnn.core.roundup(M, ttnn.TILE_SIZE),
        ttnn.core.roundup(N // output_num_cores, ttnn.TILE_SIZE),
    )
    out_mem_config = ttnn.create_sharded_memory_config_(
        shape=out_shard_shape,
        core_grid=ttnn.num_cores_to_corerangeset(
            output_num_cores,
            ttnn.CoreCoord(grid.x, grid.y),
            row_wise=True,
        ),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    interleaved_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        buffer_type=ttnn.BufferType.DRAM,
    )

    torch.manual_seed(1234)
    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    pt_out = in0 @ in1

    in0_t = ttnn.from_torch(
        in0,
        dtype=in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=interleaved_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    in0_t = ttnn.to_memory_config(in0_t, in0_mem_config)

    in1_t = ttnn.from_torch(
        in1,
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=in1_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        output_t = ttnn.linear(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=out_mem_config,
            dtype=out_dtype,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
        )
        return ttnn.sharded_to_interleaved(output_t, interleaved_mem_config)

    def check_op(tt_output):
        passing, output = comp_pcc(pt_out, tt_output)
        logger.info(output)
        assert passing

    run_test(mesh_device, run_op, check_op, enable_trace)


def run_test_matmul_interleaved(
    device,
    in0_shape,
    in1_shape,
    in0_dtype,
    in1_dtype,
    out_dtype,
    in0_mem_config,
    in1_mem_config,
    out_mem_config,
    compute_kernel_config,
):
    """
    Test matmul with interleaved memory configs (no DRAM sharded program config).

    Uses the same pre-defined compute kernel configs from config_helpers.py as the model.
    """
    logger.info(f"Testing in0_shape={in0_shape}, in1_shape={in1_shape}")
    logger.info(f"in0_dtype={in0_dtype}, in1_dtype={in1_dtype}, out_dtype={out_dtype}")
    logger.info(f"in0_mem={in0_mem_config}, in1_mem={in1_mem_config}, out_mem={out_mem_config}")
    logger.info(f"compute_kernel_config={compute_kernel_config}")

    torch.manual_seed(1234)
    # Create input tensors
    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    # Convert to TT tensors
    in0_t = ttnn.from_torch(in0, dtype=in0_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_mem_config)
    in1_t = ttnn.from_torch(in1, dtype=in1_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in1_mem_config)

    # Run matmul - no explicit program_config, let it auto-select
    # Uses pre-defined compute kernel configs from config_helpers.py (same as model)
    output_t = ttnn.matmul(
        in0_t,
        in1_t,
        memory_config=out_mem_config,
        dtype=out_dtype,
        compute_kernel_config=compute_kernel_config,
    )

    # Reference computation
    pt_out = in0 @ in1

    # Get output and compare
    tt_out = ttnn.to_torch(output_t)

    passing, output = comp_pcc(pt_out, tt_out)
    logger.info(output)
    assert passing


# Pre-defined memory configs for interleaved tests
DRAM_INTERLEAVED = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    buffer_type=ttnn.BufferType.DRAM,
)
L1_INTERLEAVED = ttnn.MemoryConfig(
    memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    buffer_type=ttnn.BufferType.L1,
)


@pytest.mark.parametrize(
    "in0_shape, in1_shape, in0_dtype, in1_dtype, in0_mem_config, in1_mem_config, out_mem_config, compute_kernel_config",
    [
        # HiFi2 FP16 cases with DRAM interleaved (uses COMPUTE_KERNEL_CONFIG_HIFI2_FP16)
        # InputA: (1, 1, 32, 896), InputB: (1, 1, 896, 1536)
        (
            (1, 1, 32, 896),
            (1, 1, 896, 1536),
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
        ),
        # InputA: (1, 1, 32, 1536), InputB: (1, 1, 1536, 3072)
        (
            (1, 1, 32, 1536),
            (1, 1, 1536, 3072),
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
        ),
        # InputA: (1, 16, 32, 128), InputB: (1, 16, 128, 512) - batched
        (
            (1, 16, 32, 128),
            (1, 16, 128, 512),
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
        ),
        # InputA: (1, 1, 32, 896), InputB: (1, 1, 896, 576)
        (
            (1, 1, 32, 896),
            (1, 1, 896, 576),
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
        ),
        # InputA: (1, 16, 32, 512), InputB: (1, 16, 512, 128) - batched
        (
            (1, 16, 32, 512),
            (1, 16, 512, 128),
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
        ),
        # InputA: (1, 1, 32, 16384), InputB: (1, 1, 16384, 896)
        (
            (1, 1, 32, 16384),
            (1, 1, 16384, 896),
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
        ),
        # HiFi2 with L1 interleaved input and fp32_dest_acc_en (uses COMPUTE_KERNEL_CONFIG_HIFI2)
        # InputA: (1, 1, 32, 7168), InputB: (1, 1, 7168, 256)
        (
            (1, 1, 32, 7168),
            (1, 1, 7168, 256),
            ttnn.bfloat16,
            ttnn.bfloat16,
            L1_INTERLEAVED,
            DRAM_INTERLEAVED,
            L1_INTERLEAVED,
            COMPUTE_KERNEL_CONFIG_HIFI2,
        ),
        # LoFi cases with L1 interleaved - batched (uses COMPUTE_KERNEL_CONFIG_LOFI)
        # InputA: (1, 8, 128, 7168), InputB: (1, 8, 7168, 2048)
        (
            (1, 8, 128, 7168),
            (1, 8, 7168, 2048),
            ttnn.bfloat16,
            ttnn.bfloat4_b,
            L1_INTERLEAVED,
            DRAM_INTERLEAVED,
            L1_INTERLEAVED,
            COMPUTE_KERNEL_CONFIG_LOFI,
        ),
        # InputA: (1, 8, 128, 2048), InputB: (1, 8, 2048, 7168)
        (
            (1, 8, 128, 2048),
            (1, 8, 2048, 7168),
            ttnn.bfloat16,
            ttnn.bfloat4_b,
            L1_INTERLEAVED,
            DRAM_INTERLEAVED,
            L1_INTERLEAVED,
            COMPUTE_KERNEL_CONFIG_LOFI,
        ),
    ],
    ids=[
        "32x896x1536_HiFi2_FP16_DRAM",
        "32x1536x3072_HiFi2_FP16_DRAM",
        "16x32x128x512_HiFi2_FP16_DRAM_batched",
        "32x896x576_HiFi2_FP16_DRAM",
        "16x32x512x128_HiFi2_FP16_DRAM_batched",
        "32x16384x896_HiFi2_FP16_DRAM",
        "32x7168x256_HiFi2_L1_fp32acc",
        "8x128x7168x2048_LoFi_L1_batched",
        "8x128x2048x7168_LoFi_L1_batched",
    ],
)
@pytest.mark.parametrize(
    "out_dtype",
    [ttnn.bfloat16],
    ids=["out_bf16"],
)
def test_matmul_interleaved_single_device(
    device,
    in0_shape,
    in1_shape,
    in0_dtype,
    in1_dtype,
    in0_mem_config,
    in1_mem_config,
    out_mem_config,
    compute_kernel_config,
    out_dtype,
):
    run_test_matmul_interleaved(
        device,
        in0_shape,
        in1_shape,
        in0_dtype,
        in1_dtype,
        out_dtype,
        in0_mem_config,
        in1_mem_config,
        out_mem_config,
        compute_kernel_config,
    )


@pytest.mark.parametrize(
    "in0_shape, in1_shape, in0_dtype, in1_dtype, in0_mem_config, in1_mem_config, out_mem_config, compute_kernel_config",
    [
        # HiFi2 FP16 cases with DRAM interleaved (uses COMPUTE_KERNEL_CONFIG_HIFI2_FP16)
        # InputA: (1, 1, 32, 896), InputB: (1, 1, 896, 1536)
        (
            (1, 1, 32, 896),
            (1, 1, 896, 1536),
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
        ),
        # InputA: (1, 1, 32, 1536), InputB: (1, 1, 1536, 3072)
        (
            (1, 1, 32, 1536),
            (1, 1, 1536, 3072),
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
        ),
        # InputA: (1, 16, 32, 128), InputB: (1, 16, 128, 512) - batched
        (
            (1, 16, 32, 128),
            (1, 16, 128, 512),
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
        ),
        # InputA: (1, 1, 32, 896), InputB: (1, 1, 896, 576)
        (
            (1, 1, 32, 896),
            (1, 1, 896, 576),
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
        ),
        # InputA: (1, 16, 32, 512), InputB: (1, 16, 512, 128) - batched
        (
            (1, 16, 32, 512),
            (1, 16, 512, 128),
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
        ),
        # InputA: (1, 1, 32, 16384), InputB: (1, 1, 16384, 896)
        (
            (1, 1, 32, 16384),
            (1, 1, 16384, 896),
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            DRAM_INTERLEAVED,
            COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
        ),
        # HiFi2 with L1 interleaved input and fp32_dest_acc_en (uses COMPUTE_KERNEL_CONFIG_HIFI2)
        # InputA: (1, 1, 32, 7168), InputB: (1, 1, 7168, 256)
        (
            (1, 1, 32, 7168),
            (1, 1, 7168, 256),
            ttnn.bfloat16,
            ttnn.bfloat16,
            L1_INTERLEAVED,
            DRAM_INTERLEAVED,
            L1_INTERLEAVED,
            COMPUTE_KERNEL_CONFIG_HIFI2,
        ),
        # LoFi cases with L1 interleaved - batched (uses COMPUTE_KERNEL_CONFIG_LOFI)
        # InputA: (1, 8, 128, 7168), InputB: (1, 8, 7168, 2048)
        (
            (1, 8, 128, 7168),
            (1, 8, 7168, 2048),
            ttnn.bfloat16,
            ttnn.bfloat4_b,
            L1_INTERLEAVED,
            DRAM_INTERLEAVED,
            L1_INTERLEAVED,
            COMPUTE_KERNEL_CONFIG_LOFI,
        ),
        # InputA: (1, 8, 128, 2048), InputB: (1, 8, 2048, 7168)
        (
            (1, 8, 128, 2048),
            (1, 8, 2048, 7168),
            ttnn.bfloat16,
            ttnn.bfloat4_b,
            L1_INTERLEAVED,
            DRAM_INTERLEAVED,
            L1_INTERLEAVED,
            COMPUTE_KERNEL_CONFIG_LOFI,
        ),
    ],
    ids=[
        "32x896x1536_HiFi2_FP16_DRAM",
        "32x1536x3072_HiFi2_FP16_DRAM",
        "16x32x128x512_HiFi2_FP16_DRAM_batched",
        "32x896x576_HiFi2_FP16_DRAM",
        "16x32x512x128_HiFi2_FP16_DRAM_batched",
        "32x16384x896_HiFi2_FP16_DRAM",
        "32x7168x256_HiFi2_L1_fp32acc",
        "8x128x7168x2048_LoFi_L1_batched",
        "8x128x2048x7168_LoFi_L1_batched",
    ],
)
@pytest.mark.parametrize(
    "out_dtype",
    [ttnn.bfloat16],
    ids=["out_bf16"],
)
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.parametrize(
    "device_params", [{"trace_region_size": 90112, "fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True
)
@pytest.mark.requires_device(["T3K", "TG", "DUAL", "QUAD"])
def test_matmul_interleaved_mesh_device(
    mesh_device,
    in0_shape,
    in1_shape,
    in0_dtype,
    in1_dtype,
    in0_mem_config,
    in1_mem_config,
    out_mem_config,
    compute_kernel_config,
    out_dtype,
    enable_trace,
    device_params,
):
    """
    Mesh device test for matmul with interleaved memory configs.
    """
    torch.manual_seed(1234)
    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()
    pt_out = in0 @ in1

    in0_t = ttnn.from_torch(
        in0,
        dtype=in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=in0_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    in1_t = ttnn.from_torch(
        in1,
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=in1_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run_op():
        output_t = ttnn.matmul(
            in0_t,
            in1_t,
            memory_config=out_mem_config,
            dtype=out_dtype,
            compute_kernel_config=compute_kernel_config,
        )
        return output_t

    def check_op(tt_output):
        passing, output = comp_pcc(pt_out, tt_output)
        logger.info(output)
        assert passing

    run_test(mesh_device, run_op, check_op, enable_trace)
