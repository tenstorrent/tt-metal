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

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
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
):
    """
    Create DRAM streaming matmul weight and output tensors.

    This helper is designed to be reused for multiple DRAM matmuls in the fused kernel.

    Args:
        device: TT device
        K: K dimension (input width)
        N: N dimension (output width, will be padded to num_banks)
        num_experts: Number of expert weight matrices
        compute_core_grid: CoreRangeSet for compute cores
        num_cores: Number of compute cores
        tile_h: Tile height (default 1)
        tile_w: Tile width (default 32)
        dtype: Data type for weights (default bfloat4_b)
        seed: Random seed for weight generation

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
    )

    return weights_tensor, output_tensor, expert_weights_for_validation, expert_tensors


def test_moe_routed_expert(device):
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

    # Testing mode: when True, hardcode expert index 0 and create only 1 expert
    use_hardcoded_expert_index = False
    num_experts = 1 if use_hardcoded_expert_index else 256

    # Tile definitions
    tile_1x32 = ttnn.Tile([1, 32])
    tile_32x32 = ttnn.Tile([32, 32])  # For weights
    tile_16x16 = ttnn.Tile([16, 16])  # For gate 16x16 tensors
    tile_1x16 = ttnn.Tile([1, 16])  # For gate output tensors

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
    logger.info("Running MoE routed expert fused operation...")
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
