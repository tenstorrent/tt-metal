# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for fused MoE operation (routed expert + shared expert).

Runs both MoE routed expert and shared expert on the same input,
validates each independently, and verifies the combined MoE output.

Run:
    pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_moe.py -v -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.fused_ops.down_proj.op import DownProj
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp, MoeRoutedExpertOp
from models.demos.deepseek_v3_b1.fused_ops.shared_expert.op import SharedExpertOp
from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import shuffle_tensor_tiles


# ============================================================================
# Helper: create DRAM streaming expert weight tensors
# ============================================================================
def create_expert_matmul_tensors(
    device,
    K,
    N,
    num_experts,
    compute_core_grid,
    num_cores,
    tile_h=1,
    tile_w=32,
    dtype=ttnn.bfloat4_b,
    seed=0,
    mesh_mapper=None,
):
    """
    Create DRAM streaming matmul weight and output tensors.

    Returns:
        Tuple of:
        - weights_tensor: First expert tensor (kernel uses base addr + offset for others)
        - output_tensor: Output tensor for matmul result
        - expert_weights_for_validation: Dict of expert weights for golden validation
        - expert_tensors: List of all expert tensors (must be kept alive to prevent deallocation)
    """
    num_banks = device.dram_grid_size().x

    # Pad N to be divisible by num_banks * tile_w
    N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    per_core_N = N_padded // num_banks

    logger.info(f"DRAM MM: K={K}, N={N}, N_padded={N_padded}, per_core_N={per_core_N}, num_experts={num_experts}")

    # DRAM shard spec for weights
    in1_shard_shape = [K, per_core_N]
    in1_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in1_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)

    # Upload experts as separate contiguous tensors
    logger.info(f"Uploading {num_experts} experts as separate contiguous tensors...")
    expert_tensors = []
    expert_weights_for_validation = {}

    for expert_idx in range(num_experts):
        torch.manual_seed(seed + expert_idx)
        expert_weights = torch.randn(1, 1, K, N_padded).clamp(-2, 2).bfloat16()

        expert_weights_for_validation[expert_idx] = expert_weights.clone()

        expert_shuffled = shuffle_tensor_tiles(expert_weights.reshape(1, K, N_padded), tile_w, num_banks)
        expert_shuffled = expert_shuffled.reshape(1, 1, K, N_padded)

        expert_t = ttnn.from_torch(
            expert_shuffled.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=in1_memory_config,
            mesh_mapper=mesh_mapper,
        )
        expert_tensors.append(expert_t)

        del expert_shuffled

        if (expert_idx + 1) % 32 == 0:
            logger.info(f"  Uploaded {expert_idx + 1}/{num_experts} experts")

    logger.info(f"All experts uploaded.")

    # Use first expert tensor for the op
    weights_tensor = expert_tensors[0]

    # Output tensor - WIDTH_SHARDED in L1
    out_tile = ttnn.Tile([tile_h, tile_w])
    out_shard_shape = [tile_h, per_core_N]
    out_shard_spec = ttnn.ShardSpec(compute_core_grid, out_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    out_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard_spec)

    output_tensor = ttnn.from_torch(
        torch.zeros(1, 1, tile_h, N_padded).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_memory_config,
        tile=out_tile,
        mesh_mapper=mesh_mapper,
    )

    return weights_tensor, output_tensor, expert_weights_for_validation, expert_tensors


# ============================================================================
# Helper: create all shared-expert tensors
# ============================================================================
def create_shared_expert_tensors(device, M, K_gate, mcast_grid):
    """
    Create all tensors needed by SharedExpertOp.

    Args:
        device: TT device
        M: Batch dimension (1)
        K_gate: Gate/Up input dimension (7168)
        mcast_grid: CoreRangeSet for mcast destination (same as routed input mcast)

    Returns:
        dict with all ttnn tensors, torch tensors, and validation data.
    """
    k_parallel = 8
    n_parallel = 8
    K_down = n_parallel * 32  # 256
    N_per_core = 64
    N = N_per_core * DownProj.NUM_MATMUL_CORES  # 7168
    k_per_core = (K_gate // 32) // k_parallel
    weights_dtype = ttnn.bfloat8_b

    a_tile = ttnn.Tile([M, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([M, 32])

    # Core grids
    a_cores_list, b_cores_list = SharedExpertOp.build_ab_grids()
    compute_cores_list = a_cores_list + b_cores_list
    mcast_gather_core = DownProj.MCAST_GATHER_CORE
    sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_gather_core, mcast_gather_core)])
    matmul_core_grid = DownProj.build_matmul_core_grid()

    # Create torch data
    torch.manual_seed(100)  # Different seed from routed expert
    torch_activation = torch.randn((M, K_gate), dtype=torch.bfloat16)
    torch_gate_weights = torch.randn((K_gate, K_down), dtype=torch.bfloat16)
    torch_up_weights = torch.randn((K_gate, K_down), dtype=torch.bfloat16)
    torch_down_weights = torch.randn((K_down, N), dtype=torch.bfloat16)
    torch_bias = torch.randn((M, N), dtype=torch.bfloat16)

    # ── Activation tensor ──
    act_shard = ttnn.ShardSpec(sender_core_grid, (M, K_gate), ttnn.ShardOrientation.ROW_MAJOR)
    act_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, act_shard)
    ttnn_activation = ttnn.from_torch(
        torch_activation,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=act_mem,
        tile=a_tile,
    )

    # ── Gate/Up weights (stacked, HEIGHT_SHARDED on 128 compute cores) ──
    weight_shards = []
    for i in range(len(a_cores_list)):
        k_idx = i // n_parallel
        n_idx = i % n_parallel
        k_start = k_idx * k_per_core * 32
        k_end = k_start + k_per_core * 32
        n_start, n_end = n_idx * 32, (n_idx + 1) * 32
        weight_shards.append(torch_gate_weights[k_start:k_end, n_start:n_end])
    for i in range(len(b_cores_list)):
        k_idx = i // n_parallel
        n_idx = i % n_parallel
        k_start = k_idx * k_per_core * 32
        k_end = k_start + k_per_core * 32
        n_start, n_end = n_idx * 32, (n_idx + 1) * 32
        weight_shards.append(torch_up_weights[k_start:k_end, n_start:n_end])
    torch_gate_up_stacked = torch.cat(weight_shards, dim=0)

    compute_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in compute_cores_list])
    gu_shard = ttnn.ShardSpec(compute_core_grid, (k_per_core * 32, 32), ttnn.ShardOrientation.ROW_MAJOR)
    gu_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gu_shard)
    ttnn_gate_up_weights = ttnn.from_torch(
        torch_gate_up_stacked,
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gu_mem,
        tile=b_tile,
    )

    # ── Down proj weights ──
    down_shard = ttnn.ShardSpec(matmul_core_grid, (K_down, N_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    down_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, down_shard)
    ttnn_down_weights = ttnn.from_torch(
        torch_down_weights,
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=down_mem,
        tile=b_tile,
    )

    # ── Bias ──
    bias_shard = ttnn.ShardSpec(sender_core_grid, (M, N), ttnn.ShardOrientation.ROW_MAJOR)
    bias_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, bias_shard)
    ttnn_bias = ttnn.from_torch(
        torch_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=bias_mem, tile=out_tile
    )

    # ── Output ──
    out_shard = ttnn.ShardSpec(sender_core_grid, (M, N), ttnn.ShardOrientation.ROW_MAJOR)
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard)
    ttnn_output = ttnn.from_torch(
        torch.zeros((M, N), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=out_mem,
        tile=out_tile,
    )

    # ── Residual mcast destination tensor (on full mcast grid, same as routed input mcast) ──
    residual_mcast_dst_shard = ttnn.ShardSpec(mcast_grid, (M, N), ttnn.ShardOrientation.ROW_MAJOR)
    residual_mcast_dst_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, residual_mcast_dst_shard
    )
    ttnn_residual_mcast_dst = ttnn.from_torch(
        torch.zeros((M, N), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=residual_mcast_dst_mem,
        tile=out_tile,
    )

    # ── Down mcast destination tensor (gated reduce output [1, K_down] → all 130 cores) ──
    down_mcast_dst_shard = ttnn.ShardSpec(mcast_grid, (M, K_down), ttnn.ShardOrientation.ROW_MAJOR)
    down_mcast_dst_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, down_mcast_dst_shard
    )
    ttnn_down_mcast_dst = ttnn.from_torch(
        torch.zeros((M, K_down), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=down_mcast_dst_mem,
        tile=a_tile,
    )

    return {
        # TTNN tensors
        "ttnn_activation": ttnn_activation,
        "ttnn_gate_up_weights": ttnn_gate_up_weights,
        "ttnn_down_weights": ttnn_down_weights,
        "ttnn_bias": ttnn_bias,
        "ttnn_output": ttnn_output,
        "ttnn_residual_mcast_dst": ttnn_residual_mcast_dst,
        "ttnn_down_mcast_dst": ttnn_down_mcast_dst,
        # Params
        "k_parallel": k_parallel,
        "n_parallel": n_parallel,
        # Torch tensors for golden
        "torch_activation": torch_activation,
        "torch_gate_weights": torch_gate_weights,
        "torch_up_weights": torch_up_weights,
        "torch_down_weights": torch_down_weights,
        "torch_bias": torch_bias,
        # Dimensions
        "N": N,
    }


# ============================================================================
# Helper: create all routed-expert tensors
# ============================================================================
def create_routed_expert_tensors(device, use_hardcoded_expert_index):
    """
    Create all tensors needed for MoE routed expert test.
    Directly extracted from the working inline test setup.

    Returns:
        dict with all ttnn tensors, torch tensors, expert dicts, and dimensions.
    """
    # MoE router: [1, 7168] x [7168, 256] with 8 cores
    M = 1
    K = 7168
    N_per_core = 32
    num_cores = 8
    N = N_per_core * num_cores  # 256 total output width (routing matmul)

    # DRAM matmul + SiLU parameters
    gate_proj_K = K  # Same K as routing matmul (7168)
    gate_proj_N = 2048  # Expert output width

    # num_experts: 1 for hardcoded (only expert 0), 256 for dynamic
    num_experts = 1 if use_hardcoded_expert_index else 256

    # Tile definitions
    tile_1x32 = ttnn.Tile([1, 32])
    tile_32x32 = ttnn.Tile([32, 32])  # For weights
    tile_16x16 = ttnn.Tile([16, 16])  # For gate 16x16 tensors

    # Gate parameters (must match op.py)
    gate_eps = 1e-20
    gate_scaling_factor = 2.5

    # Create input, weights, and gate tensors
    torch.manual_seed(0)
    torch_input = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gate_mm_weights = torch.randn((K, N), dtype=torch.bfloat16)
    torch_bias = torch.randn(
        (1, 8, 32), dtype=torch.bfloat16
    )  # Gate bias (batch=1, 8, 32) - matches golden expectation
    # Expert indices 0-255, transposed as expected by gate
    torch_indices = torch.arange(N, dtype=torch.int32).reshape(16, 16).T.contiguous().to(torch.uint16)

    # Define core grid for compute (first column, 8 cores)
    compute_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, num_cores - 1))])

    # Input tensor: sharded on sender core OUTSIDE the compute grid
    device_grid_size = device.compute_with_storage_grid_size()
    input_core = ttnn.CoreCoord(device_grid_size.x - 1, 9)
    input_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(input_core, input_core)])
    input_shard_spec = ttnn.ShardSpec(
        input_core_grid,
        (M, K),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tile_1x32,
    )

    # Get optimal DRAM bank cores for DRAM streaming matmul + SiLU
    gate_proj_noc = ttnn.NOC.NOC_0
    gate_proj_worker_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(gate_proj_noc)
    gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in gate_proj_worker_cores])
    num_gate_proj_cores = len(gate_proj_worker_cores)

    # Mcast output tensor: sharded on rectangular grid from (0,0) to sender core
    mcast_output_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), input_core)])
    mcast_output_shard_spec = ttnn.ShardSpec(
        mcast_output_core_grid,
        (M, K),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mcast_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, mcast_output_shard_spec
    )
    ttnn_mcast_output = ttnn.from_torch(
        torch.zeros((M, K), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mcast_output_mem_config,
        tile=tile_1x32,
    )

    # Gate matmul weights: width-sharded across 8 cores
    gate_mm_weights_shard_spec = ttnn.ShardSpec(
        compute_core_grid,
        (K, N_per_core),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    gate_mm_weights_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, gate_mm_weights_shard_spec
    )

    ttnn_gate_mm_weights = ttnn.from_torch(
        torch_gate_mm_weights,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_mm_weights_mem_config,
        tile=tile_32x32,
    )

    # Gate matmul output: width-sharded across compute cores
    gate_mm_output_shard_spec = ttnn.ShardSpec(
        compute_core_grid,
        (M, N_per_core),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    gate_mm_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, gate_mm_output_shard_spec
    )

    ttnn_gate_mm_output = ttnn.from_torch(
        torch.zeros((M, N), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_mm_output_mem_config,
        tile=tile_1x32,
    )

    # Gate input tensor: [16, 16] on sender core
    gate_input_shard_spec = ttnn.ShardSpec(
        input_core_grid,
        (16, 16),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    gate_input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gate_input_shard_spec
    )

    torch_gate_input_zeros = torch.zeros((16, 16), dtype=torch.bfloat16)
    ttnn_gate_input = ttnn.from_torch(
        torch_gate_input_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_input_mem_config,
        tile=tile_16x16,
    )

    # Gate bias tensor
    torch_bias_reshaped = torch_bias.reshape(16, 16)
    torch_bias_transposed = torch.transpose(torch_bias_reshaped, 0, 1).contiguous()
    ttnn_gate_bias = ttnn.from_torch(
        torch_bias_transposed,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_input_mem_config,
        tile=tile_16x16,
    )

    # Gate indices tensor
    ttnn_gate_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_input_mem_config,
        tile=tile_16x16,
    )

    # Gate output scores tensor [1, 16] on sender core
    tile_1x16 = ttnn.Tile((1, 16))
    gate_output_shard_spec = ttnn.ShardSpec(
        input_core_grid,
        (1, 16),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    gate_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gate_output_shard_spec
    )

    gate_output_scores_tensor = ttnn.from_torch(
        torch.zeros((1, 16), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_output_mem_config,
        tile=tile_1x16,
    )

    # Gate output indices tensor [1, 16] on sender core
    gate_output_indices_tensor = ttnn.from_torch(
        torch.zeros((1, 16), dtype=torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_output_mem_config,
        tile=tile_1x16,
    )

    # Expert index tensor [1, 16] on mcast grid
    expert_index_shard_spec = ttnn.ShardSpec(
        mcast_output_core_grid,
        (1, 16),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    expert_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, expert_index_shard_spec
    )
    expert_index_tensor = ttnn.from_torch(
        torch.zeros((1, 16), dtype=torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=expert_index_mem_config,
        tile=tile_1x16,
    )

    # Expert scale tensor [1, 16] on mcast grid
    expert_scale_tensor = ttnn.from_torch(
        torch.zeros((1, 16), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=expert_index_mem_config,
        tile=tile_1x16,
    )

    (
        gate_proj_weights,
        gate_proj_output,
        expert_weights_dict,
        gate_proj_expert_tensors,
    ) = create_expert_matmul_tensors(
        device=device,
        K=gate_proj_K,
        N=gate_proj_N,
        num_experts=num_experts,
        compute_core_grid=gate_proj_core_ranges,
        num_cores=num_gate_proj_cores,
        tile_h=M,
        tile_w=32,
        dtype=ttnn.bfloat4_b,
        seed=0,
    )

    # up_proj matmul tensors
    (
        up_proj_weights,
        up_proj_mm_out_tensor,
        up_proj_weights_dict,
        up_proj_expert_tensors,
    ) = create_expert_matmul_tensors(
        device=device,
        K=gate_proj_K,
        N=gate_proj_N,
        num_experts=num_experts,
        compute_core_grid=gate_proj_core_ranges,
        num_cores=num_gate_proj_cores,
        tile_h=M,
        tile_w=32,
        dtype=ttnn.bfloat4_b,
        seed=256,
    )

    # Fused output tensor
    num_banks = device.dram_grid_size().x
    gate_proj_N_padded = ((gate_proj_N + num_banks * 32 - 1) // (num_banks * 32)) * (num_banks * 32)
    fused_output_tensor = ttnn.from_torch(
        torch.zeros([1, 1, M, gate_proj_N_padded]).bfloat16().float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=up_proj_mm_out_tensor.memory_config(),
        tile=up_proj_mm_out_tensor.get_tile(),
    )

    # down_proj tensors
    down_proj_K = gate_proj_N
    down_proj_N = K

    down_proj_gather_shard_spec = ttnn.ShardSpec(
        input_core_grid,
        (M, gate_proj_N_padded),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    down_proj_gather_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, down_proj_gather_shard_spec
    )
    down_proj_gather_output_tensor = ttnn.from_torch(
        torch.zeros([M, gate_proj_N_padded]).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=down_proj_gather_mem_config,
        tile=tile_1x32,
    )

    down_proj_mcast_shard_spec = ttnn.ShardSpec(
        mcast_output_core_grid,
        (M, gate_proj_N_padded),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    down_proj_mcast_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, down_proj_mcast_shard_spec
    )
    down_proj_mcast_output_tensor = ttnn.from_torch(
        torch.zeros([M, gate_proj_N_padded]).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=down_proj_mcast_mem_config,
        tile=tile_1x32,
    )

    # down_proj expert weights and output
    (
        down_proj_weights,
        down_proj_output,
        down_proj_weights_dict,
        down_proj_expert_tensors,
    ) = create_expert_matmul_tensors(
        device=device,
        K=down_proj_K,
        N=down_proj_N,
        num_experts=num_experts,
        compute_core_grid=gate_proj_core_ranges,
        num_cores=num_gate_proj_cores,
        tile_h=M,
        tile_w=32,
        dtype=ttnn.bfloat4_b,
        seed=512,
    )

    # fused_add tensor
    down_proj_N_padded = ((down_proj_N + num_banks * 32 - 1) // (num_banks * 32)) * (num_banks * 32)
    per_core_down_proj_N = down_proj_N_padded // num_banks

    torch.manual_seed(1024)
    fused_add_torch = torch.randn([1, 1, 1, down_proj_N_padded]).bfloat16().float()

    fused_add_replicated = fused_add_torch.repeat(1, 1, num_gate_proj_cores, 1)

    fused_add_shard_spec = ttnn.ShardSpec(
        gate_proj_core_ranges,
        (1, down_proj_N_padded),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    fused_add_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, fused_add_shard_spec
    )
    fused_add_tensor = ttnn.from_torch(
        fused_add_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=fused_add_mem_config,
        tile=tile_1x32,
    )

    # Final output tensor
    final_output_width_per_core = 32 * 32
    final_output_total_width = final_output_width_per_core * num_gate_proj_cores

    final_output_shard_spec = ttnn.ShardSpec(
        gate_proj_core_ranges,
        (1, final_output_width_per_core),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    final_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, final_output_shard_spec
    )
    final_output_tensor = ttnn.from_torch(
        torch.zeros([1, 1, 1, final_output_total_width]).bfloat16().float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=final_output_mem_config,
        tile=tile_1x32,
    )

    # ── Tensor-backed working buffers for DRAM matmul CBs ──
    def _create_matmul_working_buf(dev, weights_tensor, core_ranges, num_cores, num_subblocks_k):
        """Create a tensor-backed working buffer for DRAM streaming matmul."""
        w_tile = weights_tensor.get_tile()
        w_shard = weights_tensor.memory_config().shard_spec.shape
        Kt = w_shard[0] // w_tile.tile_shape[0]
        subblock_k = Kt // num_subblocks_k
        num_in1_buffers = 3 * num_subblocks_k
        in1_CB_tiles = subblock_k * num_in1_buffers
        tile_h = w_tile.tile_shape[0]
        tile_w = w_tile.tile_shape[1]
        shard_h = in1_CB_tiles * tile_h
        shard_w = tile_w
        shard_spec = ttnn.ShardSpec(core_ranges, (shard_h, shard_w), ttnn.ShardOrientation.ROW_MAJOR)
        mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
        buf_tensor = ttnn.from_torch(
            torch.zeros(shard_h, shard_w * num_cores, dtype=torch.bfloat16),
            dtype=weights_tensor.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=mem_config,
            tile=w_tile,
        )
        return buf_tensor

    # gate_proj and up_proj share the same working buffer (identical shape, sequential execution)
    gate_up_proj_in1_buf_tensor = _create_matmul_working_buf(
        device, gate_proj_weights, gate_proj_core_ranges, num_gate_proj_cores, num_subblocks_k=4
    )
    down_proj_in1_buf_tensor = _create_matmul_working_buf(
        device, down_proj_weights, gate_proj_core_ranges, num_gate_proj_cores, num_subblocks_k=2
    )

    # Scalar working buffer (16x16 tile, bfloat16)
    tile_16x16_buf = ttnn.Tile([16, 16])
    scalar_shard = ttnn.ShardSpec(gate_proj_core_ranges, (16, 16), ttnn.ShardOrientation.ROW_MAJOR)
    scalar_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, scalar_shard)
    mul_scalar_buf_tensor = ttnn.from_torch(
        torch.zeros(16, 16 * num_gate_proj_cores, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=scalar_mem,
        tile=tile_16x16_buf,
    )

    return {
        # TTNN tensors for op()
        "ttnn_input": ttnn_input,
        "ttnn_mcast_output": ttnn_mcast_output,
        "ttnn_gate_mm_weights": ttnn_gate_mm_weights,
        "ttnn_gate_mm_output": ttnn_gate_mm_output,
        "ttnn_gate_input": ttnn_gate_input,
        "ttnn_gate_bias": ttnn_gate_bias,
        "ttnn_gate_indices": ttnn_gate_indices,
        "gate_output_scores_tensor": gate_output_scores_tensor,
        "gate_output_indices_tensor": gate_output_indices_tensor,
        "expert_index_tensor": expert_index_tensor,
        "expert_scale_tensor": expert_scale_tensor,
        "gate_proj_weights": gate_proj_weights,
        "gate_proj_output": gate_proj_output,
        "up_proj_weights": up_proj_weights,
        "up_proj_mm_out_tensor": up_proj_mm_out_tensor,
        "fused_output_tensor": fused_output_tensor,
        "down_proj_gather_output_tensor": down_proj_gather_output_tensor,
        "down_proj_mcast_output_tensor": down_proj_mcast_output_tensor,
        "down_proj_weights": down_proj_weights,
        "down_proj_output": down_proj_output,
        "fused_add_tensor": fused_add_tensor,
        "final_output_tensor": final_output_tensor,
        # Tensor-backed working buffers (gate_proj and up_proj share one buffer)
        "gate_proj_in1_buf_tensor": gate_up_proj_in1_buf_tensor,
        "up_proj_in1_buf_tensor": gate_up_proj_in1_buf_tensor,
        "down_proj_in1_buf_tensor": down_proj_in1_buf_tensor,
        "mul_scalar_buf_tensor": mul_scalar_buf_tensor,
        # Keep-alive references (prevent garbage collection)
        "gate_proj_expert_tensors": gate_proj_expert_tensors,
        "up_proj_expert_tensors": up_proj_expert_tensors,
        "down_proj_expert_tensors": down_proj_expert_tensors,
        # Torch tensors for golden
        "torch_input": torch_input,
        "torch_gate_mm_weights": torch_gate_mm_weights,
        "torch_bias": torch_bias,
        "expert_weights_dict": expert_weights_dict,
        "up_proj_weights_dict": up_proj_weights_dict,
        "down_proj_weights_dict": down_proj_weights_dict,
        "fused_add_torch": fused_add_torch,
        # Constants for golden
        "gate_eps": gate_eps,
        "gate_scaling_factor": gate_scaling_factor,
        # Dimensions for output extraction
        "num_gate_proj_cores": num_gate_proj_cores,
        "final_output_width_per_core": final_output_width_per_core,
        "per_core_down_proj_N": per_core_down_proj_N,
    }


# ============================================================================
# Helper: extract valid data from padded routed expert output
# ============================================================================
def extract_routed_expert_output(
    output_final_torch, num_gate_proj_cores, final_output_width_per_core, per_core_down_proj_N
):
    """Extract valid data from padded final output tensor."""
    result_valid = []
    for i in range(num_gate_proj_cores):
        start_idx = i * final_output_width_per_core
        end_idx = start_idx + per_core_down_proj_N
        result_valid.append(output_final_torch[..., start_idx:end_idx])
    return torch.cat(result_valid, dim=-1)


# ============================================================================
# Test: Fused MoE (routed expert + shared expert)
# ============================================================================
@pytest.mark.parametrize(
    "use_hardcoded_expert_index",
    [
        True,
    ],
)
def test_moe_fused(device, use_hardcoded_expert_index):
    """Test fused MoE: run both routed expert and shared expert, validate combined output."""

    device_grid = device.compute_with_storage_grid_size()
    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for 13x10")

    M = 1
    K = 7168

    logger.info(f"Testing fused MoE: K={K}, use_hardcoded_expert_index={use_hardcoded_expert_index}")

    # ── Phase 1: Fused routed expert + shared gate/up matmul ──
    logger.info("Phase 1: Running fused routed expert + shared gate/up matmul...")
    r = create_routed_expert_tensors(device, use_hardcoded_expert_index)
    mcast_grid = r["ttnn_mcast_output"].memory_config().shard_spec.grid
    s = create_shared_expert_tensors(device, M, K, mcast_grid)

    num_iterations = 1
    for iteration in range(num_iterations):
        ttnn_result_scores, ttnn_result_indices, ttnn_result_final = MoeOp.op(
            r["ttnn_input"],
            r["ttnn_mcast_output"],
            r["ttnn_gate_mm_weights"],
            r["ttnn_gate_mm_output"],
            r["ttnn_gate_input"],
            r["ttnn_gate_bias"],
            r["ttnn_gate_indices"],
            r["gate_output_scores_tensor"],
            r["gate_output_indices_tensor"],
            r["expert_index_tensor"],
            r["expert_scale_tensor"],
            r["gate_proj_weights"],
            r["gate_proj_output"],
            r["up_proj_weights"],
            r["up_proj_mm_out_tensor"],
            r["fused_output_tensor"],
            r["down_proj_gather_output_tensor"],
            r["down_proj_mcast_output_tensor"],
            r["down_proj_weights"],
            r["down_proj_output"],
            r["fused_add_tensor"],
            r["final_output_tensor"],
            r["gate_proj_in1_buf_tensor"],
            r["up_proj_in1_buf_tensor"],
            r["down_proj_in1_buf_tensor"],
            r["mul_scalar_buf_tensor"],
            # Shared expert tensors
            shared_gate_up_weights_tensor=s["ttnn_gate_up_weights"],
            shared_bias_tensor=s["ttnn_bias"],
            shared_residual_mcast_dst_tensor=s["ttnn_residual_mcast_dst"],
            shared_down_mcast_dst_tensor=s["ttnn_down_mcast_dst"],
            shared_down_weights_tensor=s["ttnn_down_weights"],
            shared_k_parallel=s["k_parallel"],
            shared_n_parallel=s["n_parallel"],
            use_hardcoded_expert_index=use_hardcoded_expert_index,
        )
    ttnn.synchronize_device(device)
    logger.info(f"Fused routed+shared gate/up: {num_iterations} iterations completed")

    # Read back routed expert results
    output_scores_torch = ttnn.to_torch(ttnn_result_scores)
    output_indices_torch = ttnn.to_torch(ttnn_result_indices).to(torch.int64)
    output_final_torch = ttnn.to_torch(ttnn_result_final)

    output_final_valid = extract_routed_expert_output(
        output_final_torch,
        r["num_gate_proj_cores"],
        r["final_output_width_per_core"],
        r["per_core_down_proj_N"],
    )

    # Compute routed expert golden (uses only torch tensors, no device)
    torch_expected_scores, torch_expected_indices, torch_expected_final = MoeRoutedExpertOp.golden(
        r["torch_input"],
        r["torch_gate_mm_weights"],
        r["torch_bias"],
        gate_proj_weights_dict=r["expert_weights_dict"],
        up_proj_weights_dict=r["up_proj_weights_dict"],
        down_proj_weights_dict=r["down_proj_weights_dict"],
        fused_add_tensor=r["fused_add_torch"],
        eps=r["gate_eps"],
        scaling_factor=r["gate_scaling_factor"],
        use_hardcoded_expert_index=use_hardcoded_expert_index,
    )

    # Verify routed expert gate
    output_indices_top8 = output_indices_torch[0, :8]
    output_scores_top8 = output_scores_torch[0, :8]
    sorted_output_indices, sort_idx = torch.sort(output_indices_top8.to(torch.int64), dim=-1)
    sorted_output_scores = torch.gather(output_scores_top8, dim=-1, index=sort_idx)

    sorted_expected_indices, sort_idx_expected = torch.sort(torch_expected_indices.squeeze(0).to(torch.int64), dim=-1)
    sorted_expected_scores = torch.gather(torch_expected_scores.squeeze(0).bfloat16(), dim=-1, index=sort_idx_expected)

    assert torch.equal(sorted_output_indices, sorted_expected_indices), "Routed expert: gate indices mismatch"
    assert torch.allclose(
        sorted_output_scores, sorted_expected_scores, atol=1e-2, rtol=1e-4
    ), "Routed expert: gate scores mismatch"

    passing, pcc_routed = comp_pcc(torch_expected_final, output_final_valid, 0.97)
    logger.info(f"Routed expert PCC: {pcc_routed}")
    assert passing, f"Routed expert PCC check failed: {pcc_routed}"

    # ── Phase 2: Shared expert ──
    logger.info("Phase 2: Running shared expert...")

    for iteration in range(num_iterations):
        ttnn_shared_result = SharedExpertOp.op(
            s["ttnn_activation"],
            s["ttnn_gate_up_weights"],
            s["ttnn_down_weights"],
            s["ttnn_bias"],
            s["ttnn_output"],
            s["k_parallel"],
            s["n_parallel"],
        )
    ttnn.synchronize_device(device)
    logger.info(f"Shared expert: {num_iterations} iterations completed")

    # Read back shared expert results and save torch tensors for golden
    output_shared_torch = ttnn.to_torch(ttnn_shared_result)
    s_torch_activation = s["torch_activation"]
    s_torch_gate_weights = s["torch_gate_weights"]
    s_torch_up_weights = s["torch_up_weights"]
    s_torch_down_weights = s["torch_down_weights"]
    s_torch_bias = s["torch_bias"]

    # Deallocate shared expert device tensors
    del s, ttnn_shared_result

    # Validate shared expert
    torch_expected_shared = SharedExpertOp.golden(
        s_torch_activation.float(),
        s_torch_gate_weights.float(),
        s_torch_up_weights.float(),
        s_torch_down_weights.float(),
        s_torch_bias.float(),
    ).bfloat16()

    passing, pcc_shared = comp_pcc(torch_expected_shared, output_shared_torch, 0.97)
    logger.info(f"Shared expert PCC: {pcc_shared}")
    assert passing, f"Shared expert PCC check failed: {pcc_shared}"

    logger.info(f"Fused MoE test PASSED! (routed={pcc_routed}, shared={pcc_shared})")
