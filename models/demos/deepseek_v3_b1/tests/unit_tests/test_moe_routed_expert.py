# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test for MoE Routed Expert fused operation.

Tests the fused operation:
1. Input: [1, 7168] tensor on sender core (outside compute grid)
2. Mcast from sender to 8 compute cores
3. Each compute core: [1, 7168] @ [7168, 32] -> [1, 32] + sigmoid
4. Gather outputs back to sender core -> [1, 256] = [16, 16]
5. Gate: top-8 expert selection with normalized scores
6. Mcast expert indices to compute cores
7. DRAM streaming matmul + SiLU with indexed expert weights
8. Output: expert computation result [1, N] on compute cores
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, skip_for_wormhole_b0
from models.demos.deepseek_v3_b1.fused_ops.moe_routed_expert.op import MoeRoutedExpert
from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import shuffle_tensor_tiles


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

    This helper is designed to be reused for multiple DRAM matmuls in the fused kernel.

    Args:
        device: TT device or MeshDevice
        K: K dimension (input width)
        N: N dimension (output width, will be padded to num_banks)
        num_experts: Number of expert weight matrices
        compute_core_grid: CoreRangeSet for compute cores
        num_cores: Number of compute cores
        tile_h: Tile height (default 1)
        tile_w: Tile width (default 32)
        dtype: Data type for weights (default bfloat4_b)
        seed: Random seed for weight generation
        mesh_mapper: Optional mesh mapper for multi-device (e.g., ttnn.ReplicateTensorToMesh)

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
    # Keep unshuffled weights for validation
    logger.info(f"Uploading {num_experts} experts as separate contiguous tensors...")
    expert_tensors = []
    expert_weights_for_validation = {}  # Store unshuffled weights for validation

    for expert_idx in range(num_experts):
        torch.manual_seed(seed + expert_idx)
        expert_weights = torch.randn(1, 1, K, N_padded).clamp(-2, 2).bfloat16()

        # Store unshuffled weights for validation
        expert_weights_for_validation[expert_idx] = expert_weights.clone()

        # Shuffle tiles for this expert
        expert_shuffled = shuffle_tensor_tiles(expert_weights.reshape(1, K, N_padded), tile_w, num_banks)
        expert_shuffled = expert_shuffled.reshape(1, 1, K, N_padded)

        # Upload to DRAM
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


@pytest.mark.parametrize("use_hardcoded_expert_index", [True, pytest.param(False, marks=pytest.mark.skip_post_commit)])
def test_moe_routed_expert(device, use_hardcoded_expert_index):
    """Test MoE routed expert fused operation"""

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

    logger.info(f"Testing MoE routed expert: [{M}, {K}] x [{K}, {N}] with {num_cores} cores")
    logger.info(f"DRAM matmul + SiLU: [{M}, {gate_proj_K}] x [{gate_proj_K}, {gate_proj_N}] with {num_experts} experts")

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
    # Same location as pre_sdpa mcast sender: (device_grid_x - 1, 9)
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
    logger.info(f"Created input tensor with shard shape ({M}, {K}) on core ({input_core.x}, {input_core.y})")

    # Get optimal DRAM bank cores for DRAM streaming matmul + SiLU
    gate_proj_noc = ttnn.NOC.NOC_0
    gate_proj_worker_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(gate_proj_noc)
    gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in gate_proj_worker_cores])
    num_gate_proj_cores = len(gate_proj_worker_cores)

    # Mcast output tensor: sharded on rectangular grid from (0,0) to sender core
    # This rectangle includes all cores that need the input (routing matmul + DRAM matmul + sender)
    mcast_output_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), input_core)])
    mcast_output_shard_spec = ttnn.ShardSpec(
        mcast_output_core_grid,
        (M, K),  # Each core gets full input [1, K]
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
    logger.info(
        f"Created mcast output tensor with shard shape ({M}, {K}) on {mcast_output_core_grid.num_cores()} cores"
    )

    # Gate matmul weights: width-sharded across 8 cores
    # Each core gets [K, N_per_core] = [7168, 32]
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
    logger.info(f"Created gate matmul weights tensor with shard shape ({K}, {N_per_core}) on {num_cores} cores")

    # Gate matmul output: width-sharded across compute cores
    # Each core produces [1, N_per_core] = [1, 32]
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
    logger.info(f"Created gate matmul output tensor with shard shape ({M}, {N_per_core}) on {num_cores} cores")

    # Gate input tensor: sharded on sender core (gathered from compute cores)
    # [16, 16] = 256 elements on single core (receives gathered matmul output)
    gate_input_shard_spec = ttnn.ShardSpec(
        input_core_grid,  # Same core as input (sender core)
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
    logger.info(f"Created gate input tensor with shard shape (16, 16) on sender core ({input_core.x}, {input_core.y})")

    # Gate bias tensor: sharded on sender core (transposed as expected by gate)
    # Reshape from (1, 8, 32) to (16, 16) and transpose (matches unit test pattern)
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
    logger.info(f"Created gate bias tensor with shard shape (16, 16) on sender core")

    # Gate indices tensor: sharded on sender core (uint16 indices, already transposed)
    ttnn_gate_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_input_mem_config,
        tile=tile_16x16,
    )
    logger.info(f"Created gate indices tensor with shard shape (16, 16) on sender core")

    # Gate output scores tensor: sharded on sender core [1, 16]
    tile_1x16 = ttnn.Tile((1, 16))
    gate_output_shard_spec = ttnn.ShardSpec(
        input_core_grid,
        (1, 16),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    gate_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gate_output_shard_spec
    )

    # Gate output scores tensor [1, 16] on sender core
    gate_output_scores_tensor = ttnn.from_torch(
        torch.zeros((1, 16), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_output_mem_config,
        tile=tile_1x16,
    )
    logger.info(f"Created gate output scores tensor [1, 16] on sender core")

    # Gate output indices tensor [1, 16] on sender core
    gate_output_indices_tensor = ttnn.from_torch(
        torch.zeros((1, 16), dtype=torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_output_mem_config,
        tile=tile_1x16,
    )
    logger.info(f"Created gate output indices tensor [1, 16] on sender core")

    # ========== DRAM Streaming Matmul + SiLU Tensors ==========
    # The gate output indices determine which expert to use (we'll validate after op runs)

    # Expert index tensor [1, 16] on mcast grid (receives mcasted indices)
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
    logger.info(f"Created expert index tensor [1, 16] on mcast grid")

    # Expert scale tensor [1, 16] on mcast grid (receives mcasted scale)
    expert_scale_tensor = ttnn.from_torch(
        torch.zeros((1, 16), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=expert_index_mem_config,
        tile=tile_1x16,
    )
    logger.info(f"Created expert scale tensor [1, 16] on mcast grid")

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
        compute_core_grid=gate_proj_core_ranges,  # Use optimal DRAM bank cores
        num_cores=num_gate_proj_cores,
        tile_h=M,
        tile_w=32,
        dtype=ttnn.bfloat4_b,
        seed=0,
    )
    logger.info(
        f"Created DRAM matmul + SiLU tensors: weights={gate_proj_weights.shape}, output={gate_proj_output.shape}"
    )

    # ========== up_proj Matmul Tensors (with fused mul using gate_proj output) ==========
    # up_proj computes: up_proj_mm_result * gate_proj_output (element-wise mul)
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
        seed=256,  # Different seed for different weights
    )
    logger.info(
        f"Created up_proj matmul tensors: weights={up_proj_weights.shape}, mm_out={up_proj_mm_out_tensor.shape}"
    )

    # Final fused output tensor: silu(gate_proj) * up_proj
    # Same shape and memory config as up_proj_mm_out_tensor
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
    logger.info(f"Created fused_output_tensor for intermediate result: silu(gate_proj) * up_proj")

    # ========== down_proj Tensors ==========
    # down_proj: [1, gate_proj_N] x [gate_proj_N, K] -> [1, K]
    down_proj_K = gate_proj_N  # Input dimension = fused output width (2048)
    down_proj_N = K  # Output dimension = original input width (7168)

    # down_proj_gather_output: gathered fused output on sender core
    # Shape: [1, gate_proj_N_padded] on sender core
    down_proj_gather_shard_spec = ttnn.ShardSpec(
        input_core_grid,  # Sender core
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
    logger.info(f"Created down_proj_gather_output_tensor on sender core")

    # down_proj_mcast_output: mcasted fused output on mcast grid
    # Same shape as gather output, but sharded on mcast grid
    down_proj_mcast_shard_spec = ttnn.ShardSpec(
        mcast_output_core_grid,  # Mcast grid
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
    logger.info(f"Created down_proj_mcast_output_tensor on gate_proj cores")

    # down_proj expert weights and output
    (
        down_proj_weights,
        down_proj_output,
        down_proj_weights_dict,
        down_proj_expert_tensors,
    ) = create_expert_matmul_tensors(
        device=device,
        K=down_proj_K,  # 2048 (fused output width)
        N=down_proj_N,  # 7168 (original input width)
        num_experts=num_experts,
        compute_core_grid=gate_proj_core_ranges,  # Same cores as gate_proj/up_proj
        num_cores=num_gate_proj_cores,
        tile_h=M,
        tile_w=32,
        dtype=ttnn.bfloat4_b,
        seed=512,  # Different seed for different weights
    )
    logger.info(f"Created down_proj tensors: weights={down_proj_weights.shape}, output={down_proj_output.shape}")

    # ========== fused_add Tensor (for eltwise_add after down_proj) ==========
    # fused_add is replicated on all gate_proj cores via HEIGHT_SHARDED
    # Each core has the full [1, down_proj_N_padded] tensor
    down_proj_N_padded = ((down_proj_N + num_banks * 32 - 1) // (num_banks * 32)) * (num_banks * 32)
    per_core_down_proj_N = down_proj_N_padded // num_banks

    # Create fused_add torch tensor [1, 1, 1, down_proj_N_padded]
    torch.manual_seed(1024)  # Different seed for fused_add
    fused_add_torch = torch.randn([1, 1, 1, down_proj_N_padded]).bfloat16().float()

    # Replicate for HEIGHT_SHARDING: [1, 1, num_gate_proj_cores, down_proj_N_padded]
    fused_add_replicated = fused_add_torch.repeat(1, 1, num_gate_proj_cores, 1)

    # HEIGHT_SHARDED memory config (replicated on all gate_proj cores)
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
    logger.info(f"Created fused_add tensor: HEIGHT_SHARDED [1, {down_proj_N_padded}] on {num_gate_proj_cores} cores")

    # ========== Final Output Tensor (down_proj + fused_add) ==========
    # WIDTH_SHARDED with padded output per core (32x32 tile size = 1024 elements)
    final_output_width_per_core = 32 * 32  # 1024 elements (padded for 32x32 tile)
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
    logger.info(
        f"Created final_output tensor: WIDTH_SHARDED [1, {final_output_width_per_core}] per core (padded) on {num_gate_proj_cores} cores"
    )

    # Run fused operation
    num_iterations = 100
    logger.info(f"Running MoE routed expert fused operation for {num_iterations} iterations...")
    for iteration in range(num_iterations):
        ttnn_result_scores, ttnn_result_indices, ttnn_result_final = MoeRoutedExpert.op(
            ttnn_input,
            ttnn_mcast_output,
            ttnn_gate_mm_weights,
            ttnn_gate_mm_output,
            ttnn_gate_input,
            ttnn_gate_bias,
            ttnn_gate_indices,
            gate_output_scores_tensor,
            gate_output_indices_tensor,
            expert_index_tensor,
            expert_scale_tensor,
            gate_proj_weights,
            gate_proj_output,
            up_proj_weights,
            up_proj_mm_out_tensor,
            fused_output_tensor,
            down_proj_gather_output_tensor,
            down_proj_mcast_output_tensor,
            down_proj_weights,
            down_proj_output,
            fused_add_tensor,
            final_output_tensor,
            use_hardcoded_expert_index=use_hardcoded_expert_index,
        )
    ttnn.synchronize_device(device)
    logger.info(f"All {num_iterations} iterations completed")

    # Convert back to torch for comparison
    output_scores_torch = ttnn.to_torch(ttnn_result_scores)
    output_indices_torch = ttnn.to_torch(ttnn_result_indices).to(torch.int64)
    output_final_torch = ttnn.to_torch(ttnn_result_final)

    # Extract valid data from padded final output
    # Final output is [1, 1, 1, final_output_total_width] with 1024 per core (padded)
    # We need first per_core_down_proj_N (896) elements from each 1024-element chunk
    result_valid = []
    for i in range(num_gate_proj_cores):
        start_idx = i * final_output_width_per_core
        end_idx = start_idx + per_core_down_proj_N
        result_valid.append(output_final_torch[..., start_idx:end_idx])
    output_final_valid = torch.cat(result_valid, dim=-1)  # [1, 1, 1, down_proj_N_padded]

    # Also read back intermediate matmul outputs for debugging
    output_gate_proj_torch = ttnn.to_torch(gate_proj_output)
    output_up_proj_mm_torch = ttnn.to_torch(up_proj_mm_out_tensor)
    output_fused_torch = ttnn.to_torch(fused_output_tensor)

    # Compute golden reference (includes gate + expert matmuls + fused mul + down_proj + fused_add)
    (
        torch_expected_scores,
        torch_expected_indices,
        torch_expected_final,
    ) = MoeRoutedExpert.golden(
        torch_input,
        torch_gate_mm_weights,
        torch_bias,
        gate_proj_weights_dict=expert_weights_dict,
        up_proj_weights_dict=up_proj_weights_dict,
        down_proj_weights_dict=down_proj_weights_dict,
        fused_add_tensor=fused_add_torch,
        eps=gate_eps,
        scaling_factor=gate_scaling_factor,
        use_hardcoded_expert_index=use_hardcoded_expert_index,
    )

    # ========== Verify Outputs ==========
    # Verify gate: sort by indices to handle tie-breaking differences
    output_indices_top8 = output_indices_torch[0, :8]
    output_scores_top8 = output_scores_torch[0, :8]
    sorted_output_indices, sort_idx = torch.sort(output_indices_top8.to(torch.int64), dim=-1)
    sorted_output_scores = torch.gather(output_scores_top8, dim=-1, index=sort_idx)

    sorted_expected_indices, sort_idx_expected = torch.sort(torch_expected_indices.squeeze(0).to(torch.int64), dim=-1)
    sorted_expected_scores = torch.gather(torch_expected_scores.squeeze(0).bfloat16(), dim=-1, index=sort_idx_expected)

    assert torch.equal(sorted_output_indices, sorted_expected_indices), "Gate indices mismatch"
    assert torch.allclose(sorted_output_scores, sorted_expected_scores, atol=1e-2, rtol=1e-4), "Gate scores mismatch"

    # Compute expected intermediate outputs for debugging
    if use_hardcoded_expert_index:
        selected_expert_idx = 0
        logger.info(f"Using expert index: {selected_expert_idx} (hardcoded for testing)")
    else:
        selected_expert_idx = int(torch_expected_indices[0, 0].item())
        logger.info(f"Using expert index: {selected_expert_idx} (from gate output)")

    # gate_proj expected: input @ weights + SiLU
    gate_proj_weights_torch = expert_weights_dict[selected_expert_idx]
    input_for_expert = torch_input.reshape(1, 1, 1, -1).float()
    torch_expected_gate_proj = input_for_expert @ gate_proj_weights_torch.float()
    torch_expected_gate_proj = torch.nn.functional.silu(torch_expected_gate_proj)

    # up_proj expected: input @ weights (no activation)
    up_proj_weights_torch = up_proj_weights_dict[selected_expert_idx]
    torch_expected_up_proj = input_for_expert @ up_proj_weights_torch.float()

    # Get expert scale (first score for the selected expert)
    expert_scale = torch_expected_scores[0, 0].float()
    logger.info(f"Expert scale: {expert_scale.item()}")

    # fused expected: silu(gate_proj) * up_proj * expert_scale
    torch_expected_fused = torch_expected_gate_proj * torch_expected_up_proj * expert_scale

    # Verify gate_proj matmul + SiLU
    passing, pcc_output = comp_pcc(torch_expected_gate_proj, output_gate_proj_torch, 0.98)
    logger.info(f"gate_proj (matmul + SiLU): {pcc_output}")

    # Verify up_proj matmul (no SiLU)
    passing, pcc_output = comp_pcc(torch_expected_up_proj, output_up_proj_mm_torch, 0.98)
    logger.info(f"up_proj (matmul only): {pcc_output}")

    # Verify fused output: silu(gate_proj) * up_proj * expert_scale
    passing, pcc_output = comp_pcc(torch_expected_fused, output_fused_torch, 0.98)
    logger.info(f"fused output (silu(gate_proj) * up_proj * expert_scale): {pcc_output}")

    # Compute expected down_proj for intermediate validation
    down_proj_weights_torch = down_proj_weights_dict[selected_expert_idx]
    torch_expected_down_proj = torch_expected_fused @ down_proj_weights_torch.float()

    # Verify down_proj output (intermediate check)
    output_down_proj_torch = ttnn.to_torch(down_proj_output)
    passing, pcc_output = comp_pcc(torch_expected_down_proj, output_down_proj_torch, 0.97)
    logger.info(f"down_proj output: {pcc_output}")

    # Verify final output (down_proj + fused_add)
    passing, pcc_output = comp_pcc(torch_expected_final, output_final_valid, 0.97)
    logger.info(f"final output (down_proj + fused_add): {pcc_output}")
    assert passing, f"final output PCC check failed: {pcc_output}"

    logger.info("MoE routed expert test passed!")


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": 573440})],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
@pytest.mark.parametrize("use_hardcoded_expert_index", [True, pytest.param(False, marks=pytest.mark.skip_post_commit)])
def test_moe_routed_expert_with_reduce(bh_2d_mesh_device, use_hardcoded_expert_index):
    """
    Test MoE routed expert fused operation with reduce_to_one on 4x2 mesh.

    This tests the full fused operation:
    - Each of 8 devices runs moe_routed_expert
    - Results are reduced (summed) across all devices to ROOT1
    - Final output on ROOT1 contains sum of all 8 device outputs
    """
    # Validate mesh size
    num_devices = 8
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip(
            f"Test requires {num_devices} devices, mesh has {bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1]}"
        )

    # Create 4x2 submesh
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    logger.info(f"Created submesh with shape: {submesh.shape}")

    # MoE router parameters
    M = 1
    K = 7168
    N_per_core = 32
    num_cores = 8
    N = N_per_core * num_cores  # 256 total output width

    # DRAM matmul + SiLU parameters
    gate_proj_K = K
    gate_proj_N = 2048

    # num_experts: 8 for hardcoded (one per device), 256 for dynamic
    num_experts = 8 if use_hardcoded_expert_index else 256

    # Tile definitions
    tile_1x32 = ttnn.Tile([1, 32])
    tile_32x32 = ttnn.Tile([32, 32])
    tile_16x16 = ttnn.Tile([16, 16])
    tile_1x16 = ttnn.Tile([1, 16])

    logger.info(f"Testing MoE routed expert with reduce on {num_devices}-device mesh")

    # Gate parameters
    gate_eps = 1e-20
    gate_scaling_factor = 2.5

    # Create PyTorch tensors (same for all devices since we replicate)
    torch.manual_seed(0)
    torch_input = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gate_mm_weights = torch.randn((K, N), dtype=torch.bfloat16)
    torch_bias = torch.randn((1, 8, 32), dtype=torch.bfloat16)
    torch_indices = torch.arange(N, dtype=torch.int32).reshape(16, 16).T.contiguous().to(torch.uint16)

    # Get device info from mesh for grid setup
    device_grid_size = submesh.compute_with_storage_grid_size()

    # Define core grids (same for all devices)
    compute_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, num_cores - 1))])
    input_core = ttnn.CoreCoord(device_grid_size.x - 1, 9)
    input_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(input_core, input_core)])

    # Get optimal DRAM bank cores
    gate_proj_noc = ttnn.NOC.NOC_0
    gate_proj_worker_cores = submesh.get_optimal_dram_bank_to_logical_worker_assignment(gate_proj_noc)
    gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in gate_proj_worker_cores])
    num_gate_proj_cores = len(gate_proj_worker_cores)

    # Mcast output core grid
    mcast_output_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), input_core)])

    # Mesh mapper for replication
    mesh_mapper = ttnn.ReplicateTensorToMesh(submesh)

    # ========== Create Tensors with Mesh Replication ==========

    # Input tensor
    input_shard_spec = ttnn.ShardSpec(input_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_mem_config,
        tile=tile_1x32,
        mesh_mapper=mesh_mapper,
    )

    # Mcast output tensor
    mcast_output_shard_spec = ttnn.ShardSpec(mcast_output_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR)
    mcast_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, mcast_output_shard_spec
    )
    ttnn_mcast_output = ttnn.from_torch(
        torch.zeros((M, K), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=mcast_output_mem_config,
        tile=tile_1x32,
        mesh_mapper=mesh_mapper,
    )

    # Gate matmul weights
    gate_mm_weights_shard_spec = ttnn.ShardSpec(compute_core_grid, (K, N_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    gate_mm_weights_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, gate_mm_weights_shard_spec
    )
    ttnn_gate_mm_weights = ttnn.from_torch(
        torch_gate_mm_weights,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=gate_mm_weights_mem_config,
        tile=tile_32x32,
        mesh_mapper=mesh_mapper,
    )

    # Gate matmul output
    gate_mm_output_shard_spec = ttnn.ShardSpec(compute_core_grid, (M, N_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    gate_mm_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, gate_mm_output_shard_spec
    )
    ttnn_gate_mm_output = ttnn.from_torch(
        torch.zeros((M, N), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=gate_mm_output_mem_config,
        tile=tile_1x32,
        mesh_mapper=mesh_mapper,
    )

    # Gate input tensor (16x16)
    gate_input_shard_spec = ttnn.ShardSpec(input_core_grid, (16, 16), ttnn.ShardOrientation.ROW_MAJOR)
    gate_input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gate_input_shard_spec
    )
    ttnn_gate_input = ttnn.from_torch(
        torch.zeros((16, 16), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=gate_input_mem_config,
        tile=tile_16x16,
        mesh_mapper=mesh_mapper,
    )

    # Gate bias tensor
    torch_bias_reshaped = torch_bias.reshape(16, 16)
    torch_bias_transposed = torch.transpose(torch_bias_reshaped, 0, 1).contiguous()
    ttnn_gate_bias = ttnn.from_torch(
        torch_bias_transposed,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=gate_input_mem_config,
        tile=tile_16x16,
        mesh_mapper=mesh_mapper,
    )

    # Gate indices tensor
    ttnn_gate_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=gate_input_mem_config,
        tile=tile_16x16,
        mesh_mapper=mesh_mapper,
    )

    # Gate output tensors (1x16)
    gate_output_shard_spec = ttnn.ShardSpec(input_core_grid, (1, 16), ttnn.ShardOrientation.ROW_MAJOR)
    gate_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gate_output_shard_spec
    )
    gate_output_scores_tensor = ttnn.from_torch(
        torch.zeros((1, 16), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=gate_output_mem_config,
        tile=tile_1x16,
        mesh_mapper=mesh_mapper,
    )
    gate_output_indices_tensor = ttnn.from_torch(
        torch.zeros((1, 16), dtype=torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=gate_output_mem_config,
        tile=tile_1x16,
        mesh_mapper=mesh_mapper,
    )

    # Expert index and scale tensors
    expert_index_shard_spec = ttnn.ShardSpec(mcast_output_core_grid, (1, 16), ttnn.ShardOrientation.ROW_MAJOR)
    expert_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, expert_index_shard_spec
    )
    expert_index_tensor = ttnn.from_torch(
        torch.zeros((1, 16), dtype=torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=expert_index_mem_config,
        tile=tile_1x16,
        mesh_mapper=mesh_mapper,
    )
    expert_scale_tensor = ttnn.from_torch(
        torch.zeros((1, 16), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=expert_index_mem_config,
        tile=tile_1x16,
        mesh_mapper=mesh_mapper,
    )

    # ========== DRAM Matmul Tensors ==========
    (
        gate_proj_weights,
        gate_proj_output,
        expert_weights_for_validation,
        gate_proj_expert_tensors,
    ) = create_expert_matmul_tensors(
        device=submesh,
        K=gate_proj_K,
        N=gate_proj_N,
        num_experts=num_experts,
        compute_core_grid=gate_proj_core_ranges,
        num_cores=num_gate_proj_cores,
        tile_h=M,
        tile_w=32,
        dtype=ttnn.bfloat4_b,
        seed=0,
        mesh_mapper=mesh_mapper,
    )
    logger.info(f"Created gate_proj tensors: weights={gate_proj_weights.shape}, output={gate_proj_output.shape}")

    # up_proj tensors
    (
        up_proj_weights,
        up_proj_mm_out_tensor,
        up_proj_weights_for_validation,
        up_proj_expert_tensors,
    ) = create_expert_matmul_tensors(
        device=submesh,
        K=gate_proj_K,
        N=gate_proj_N,
        num_experts=num_experts,
        compute_core_grid=gate_proj_core_ranges,
        num_cores=num_gate_proj_cores,
        tile_h=M,
        tile_w=32,
        dtype=ttnn.bfloat4_b,
        seed=256,
        mesh_mapper=mesh_mapper,
    )
    logger.info(f"Created up_proj tensors: weights={up_proj_weights.shape}, mm_out={up_proj_mm_out_tensor.shape}")

    # Fused output tensor (same shape as up_proj_mm_out)
    num_banks = submesh.dram_grid_size().x
    gate_proj_N_padded = ((gate_proj_N + num_banks * 32 - 1) // (num_banks * 32)) * (num_banks * 32)
    per_core_gate_proj_N = gate_proj_N_padded // num_banks
    out_tile = ttnn.Tile([M, 32])
    out_shard_shape = [M, per_core_gate_proj_N]
    out_shard_spec = ttnn.ShardSpec(gate_proj_core_ranges, out_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    out_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard_spec)
    fused_output_tensor = ttnn.from_torch(
        torch.zeros([1, 1, M, gate_proj_N_padded]).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=out_memory_config,
        tile=out_tile,
        mesh_mapper=mesh_mapper,
    )

    # down_proj tensors
    down_proj_K = gate_proj_N
    down_proj_N = K
    down_proj_N_padded = ((down_proj_N + num_banks * 32 - 1) // (num_banks * 32)) * (num_banks * 32)
    per_core_down_proj_N = down_proj_N_padded // num_banks

    # down_proj gather output
    down_proj_gather_shard_spec = ttnn.ShardSpec(
        input_core_grid, (M, gate_proj_N_padded), ttnn.ShardOrientation.ROW_MAJOR
    )
    down_proj_gather_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, down_proj_gather_shard_spec
    )
    down_proj_gather_output_tensor = ttnn.from_torch(
        torch.zeros([M, gate_proj_N_padded]).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=down_proj_gather_mem_config,
        tile=tile_1x32,
        mesh_mapper=mesh_mapper,
    )

    # down_proj mcast output
    down_proj_mcast_shard_spec = ttnn.ShardSpec(
        mcast_output_core_grid, (M, gate_proj_N_padded), ttnn.ShardOrientation.ROW_MAJOR
    )
    down_proj_mcast_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, down_proj_mcast_shard_spec
    )
    down_proj_mcast_output_tensor = ttnn.from_torch(
        torch.zeros([M, gate_proj_N_padded]).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=down_proj_mcast_mem_config,
        tile=tile_1x32,
        mesh_mapper=mesh_mapper,
    )

    # down_proj expert weights and output
    (
        down_proj_weights,
        down_proj_output,
        down_proj_weights_for_validation,
        down_proj_expert_tensors,
    ) = create_expert_matmul_tensors(
        device=submesh,
        K=down_proj_K,
        N=down_proj_N,
        num_experts=num_experts,
        compute_core_grid=gate_proj_core_ranges,
        num_cores=num_gate_proj_cores,
        tile_h=M,
        tile_w=32,
        dtype=ttnn.bfloat4_b,
        seed=512,
        mesh_mapper=mesh_mapper,
    )
    logger.info(f"Created down_proj tensors: weights={down_proj_weights.shape}, output={down_proj_output.shape}")

    # fused_add tensor
    torch.manual_seed(1024)
    fused_add_torch = torch.randn([1, 1, 1, down_proj_N_padded]).bfloat16().float()
    fused_add_replicated = fused_add_torch.repeat(1, 1, num_gate_proj_cores, 1)
    fused_add_shard_spec = ttnn.ShardSpec(
        gate_proj_core_ranges, (1, down_proj_N_padded), ttnn.ShardOrientation.ROW_MAJOR
    )
    fused_add_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, fused_add_shard_spec
    )
    fused_add_tensor = ttnn.from_torch(
        fused_add_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=fused_add_mem_config,
        tile=tile_1x32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # Final output tensor
    # Must match down_proj_output shape: width_per_core = per_core_down_proj_N, total = down_proj_N_padded
    final_output_width_per_core = per_core_down_proj_N
    final_output_total_width = down_proj_N_padded
    final_output_shard_spec = ttnn.ShardSpec(
        gate_proj_core_ranges, (1, final_output_width_per_core), ttnn.ShardOrientation.ROW_MAJOR
    )
    final_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, final_output_shard_spec
    )
    final_output_tensor = ttnn.from_torch(
        torch.zeros([1, 1, 1, final_output_total_width]).bfloat16().float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=final_output_mem_config,
        tile=tile_1x32,
        mesh_mapper=mesh_mapper,
    )

    # ========== ReduceToOne Tensors and Semaphores ==========
    # Root coordinate (row 1, col 1)
    root_coord = (1, 1)

    # Mesh mapper for reduce tensors
    reduce_mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], submesh.shape)
    reduce_mesh_mapper = ttnn.create_mesh_mapper(submesh, reduce_mesh_mapper_config)

    # Create 3 intermediate tensors for 3 reduction rounds
    # Same shape as final_output_tensor (which is the input to reduce)
    intermediate_tensors = []
    for _ in range(3):
        intermediate_data = torch.zeros([4, 2, final_output_total_width], dtype=torch.bfloat16)
        intermediate_tensor = ttnn.from_torch(
            intermediate_data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=final_output_mem_config,
            tile=tile_1x32,
            mesh_mapper=reduce_mesh_mapper,
        )
        intermediate_tensors.append(intermediate_tensor)
    logger.info(f"Created 3 intermediate tensors for reduce rounds")

    # Create reduce output tensor (single-core sharded on each device)
    # Only ROOT1 device will have the final reduced result
    compute_grid = submesh.compute_with_storage_grid_size()
    reduce_output_core = ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1)
    reduce_output_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(reduce_output_core, reduce_output_core)})
    reduce_output_shard_spec = ttnn.ShardSpec(
        reduce_output_shard_grid,
        (1, final_output_total_width),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    reduce_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, reduce_output_shard_spec
    )
    reduce_output_data = torch.zeros([4, 2, final_output_total_width], dtype=torch.bfloat16)
    reduce_output_tensor = ttnn.from_torch(
        reduce_output_data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=reduce_output_mem_config,
        tile=tile_1x32,
        mesh_mapper=reduce_mesh_mapper,
    )
    logger.info(f"Created reduce output tensor on core {reduce_output_core}")

    # Create 4 semaphores for reduce_to_one (round1, round2, round3, exit)
    num_cores = compute_grid.x * compute_grid.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, row_wise=True)
    reduce_semaphores = [ttnn.create_global_semaphore(submesh, available_cores, 0) for _ in range(4)]
    logger.info(f"Created 4 global semaphores for reduce synchronization")

    # ========== Run Operation ==========

    logger.info("Running moe routed expert operation...")

    for iteration in range(num_iterations):
        ttnn_result_scores, ttnn_result_indices, ttnn_result_reduce = MoeRoutedExpert.op(
            ttnn_input,
            ttnn_mcast_output,
            ttnn_gate_mm_weights,
            ttnn_gate_mm_output,
            ttnn_gate_input,
            ttnn_gate_bias,
            ttnn_gate_indices,
            gate_output_scores_tensor,
            gate_output_indices_tensor,
            expert_index_tensor,
            expert_scale_tensor,
            gate_proj_weights,
            gate_proj_output,
            up_proj_weights,
            up_proj_mm_out_tensor,
            fused_output_tensor,
            down_proj_gather_output_tensor,
            down_proj_mcast_output_tensor,
            down_proj_weights,
            down_proj_output,
            fused_add_tensor,
            final_output_tensor,
            # ReduceToOne parameters
            reduce_intermediate_tensors=intermediate_tensors,
            reduce_output_tensor=reduce_output_tensor,
            reduce_semaphores=reduce_semaphores,
            reduce_root_coord=ttnn.MeshCoordinate(root_coord),
            use_hardcoded_expert_index=use_hardcoded_expert_index,
        )
    ttnn.synchronize_device(submesh)

    # ========== Verify Results ==========
    # Get actual gate output indices and scores from device
    device_gate_indices = ttnn.to_torch(
        gate_output_indices_tensor, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0)
    )
    device_gate_scores = ttnn.to_torch(gate_output_scores_tensor, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))

    # Compute expected output for each device, then sum
    expected_final_outputs = []
    for device_idx in range(num_devices):
        chip_id = device_idx  # Row-major order

        if use_hardcoded_expert_index:
            actual_expert_idx = chip_id
            actual_expert_scale = device_gate_scores[0].flatten()[chip_id].float()
        else:
            actual_expert_idx = int(device_gate_indices[0].flatten()[chip_id].item())
            actual_expert_scale = device_gate_scores[0].flatten()[chip_id].float()

        torch_expected_scores, torch_expected_indices, torch_expected_final = MoeRoutedExpert.golden(
            torch_input,
            torch_gate_mm_weights,
            torch_bias,
            gate_proj_weights_dict=expert_weights_for_validation,
            up_proj_weights_dict=up_proj_weights_for_validation,
            down_proj_weights_dict=down_proj_weights_for_validation,
            fused_add_tensor=fused_add_torch,
            eps=gate_eps,
            scaling_factor=gate_scaling_factor,
            use_hardcoded_expert_index=True,  # Always use hardcoded since we have the actual expert
            hardcoded_expert_index=actual_expert_idx,
            explicit_expert_scale=actual_expert_scale,
        )
        expected_final_outputs.append(torch_expected_final)

    # Expected reduce output = sum of all device outputs (using golden_reduce)
    expected_reduce_output = MoeRoutedExpert.golden_reduce(expected_final_outputs)

    # Get actual reduce output from ROOT1 device
    reduce_output_torch = ttnn.to_torch(
        ttnn_result_reduce,
        mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0),
    )

    # ROOT1 is at row 1, col 1 -> device_idx = 1*2 + 1 = 3
    root_device_idx = root_coord[0] * submesh.shape[1] + root_coord[1]
    reduce_output_root = reduce_output_torch[root_device_idx]

    # Verify reduce output
    passing, pcc_output = comp_pcc(expected_reduce_output, reduce_output_root, 0.97)
    logger.info(f"Reduce output PCC: {pcc_output}")

    if not passing:
        # Debug: print some values
        logger.error(f"Expected reduce (first 8): {expected_reduce_final.flatten()[:8]}")
        logger.error(f"Actual reduce (first 8): {reduce_output_valid.flatten()[:8]}")
        diff = torch.abs(expected_reduce_final.flatten() - reduce_output_valid.flatten())
        logger.error(f"Max diff: {diff.max()}, Mean diff: {diff.mean()}")

    assert passing, f"Reduce output PCC check failed: {pcc_output}"
    logger.info("MoE routed expert with reduce test passed!")
