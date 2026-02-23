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
from models.common.utility_functions import comp_pcc, skip_for_wormhole_b0
from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights
from models.demos.deepseek_v3_b1.fused_ops.down_proj.op import DownProj
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp
from models.demos.deepseek_v3_b1.fused_ops.shared_expert.op import SharedExpertOp


# ============================================================================
# Helper: create all shared-expert tensors
# ============================================================================
def create_shared_expert_tensors(device, M, K_gate, mcast_grid, mesh_mapper=None):
    """
    Create all tensors needed by SharedExpertOp.

    Args:
        device: TT device or mesh device
        M: Batch dimension (1)
        K_gate: Gate/Up input dimension (7168)
        mcast_grid: CoreRangeSet for mcast destination (same as routed input mcast)
        mesh_mapper: Optional mesh mapper for multi-device replication

    Returns:
        dict with all ttnn tensors, torch tensors, and validation data.
    """
    k_parallel = 8
    n_parallel = 8
    K_down = n_parallel * 32  # 256
    N_per_core = 64
    N = N_per_core * DownProj.NUM_MATMUL_CORES  # 7168

    a_tile = ttnn.Tile([M, 32])
    out_tile = ttnn.Tile([M, 32])

    # Core grids
    compute_cores_list = sum(SharedExpertOp.build_ab_grids(), [])
    mcast_gather_core = DownProj.MCAST_GATHER_CORE
    sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_gather_core, mcast_gather_core)])
    matmul_core_grid = DownProj.build_matmul_core_grid()

    # Create torch data — generate full TP-width weights with unique data per shard
    bdw = BlitzDecodeWeights(device)
    moe_tp = bdw.moe_tp
    K_down_full = K_down * moe_tp

    torch.manual_seed(100)  # Different seed from routed expert
    torch_activation = torch.randn((M, K_gate), dtype=torch.bfloat16)
    torch_gate_weights = torch.randn((K_gate, K_down_full), dtype=torch.bfloat16)
    torch_up_weights = torch.randn((K_gate, K_down_full), dtype=torch.bfloat16)
    torch_down_weights = torch.randn((K_down_full, N), dtype=torch.bfloat16)
    torch_bias = torch.randn((M, N), dtype=torch.bfloat16)

    from_torch_kwargs = {"mesh_mapper": mesh_mapper} if mesh_mapper else {}

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
        **from_torch_kwargs,
    )

    compute_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in compute_cores_list])

    gate_ov, up_ov, ttnn_down_weights = bdw.get_tt_moe_shared_expert_weights(
        torch_gate_weights, torch_up_weights, torch_down_weights
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
        **from_torch_kwargs,
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
        **from_torch_kwargs,
    )

    # ── Output mcast destination tensor (shared expert output [1, N] → all 130 cores) ──
    # This will replace fused_add_tensor: mcast writes shared expert output into add_cb_in1
    num_mcast_cores = len(ttnn.corerange_to_cores(mcast_grid))
    output_mcast_dst_shard = ttnn.ShardSpec(mcast_grid, (M, N), ttnn.ShardOrientation.ROW_MAJOR)
    output_mcast_dst_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_mcast_dst_shard
    )
    ttnn_output_mcast_dst = ttnn.from_torch(
        torch.zeros((M * num_mcast_cores, N), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mcast_dst_mem,
        tile=out_tile,
        **from_torch_kwargs,
    )

    # ── Tensor-backed CB tensors ──

    # CB 30/31: Gate/Up gather destination (64 tiles each on sender core)
    total_gather_tiles = k_parallel * n_parallel  # 64
    ag_dummy_shape = (total_gather_tiles, 32)
    ag_dummy_shard_spec = ttnn.ShardSpec(sender_core_grid, ag_dummy_shape, ttnn.ShardOrientation.ROW_MAJOR)
    ag_dummy_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ag_dummy_shard_spec)
    ttnn_ag_gather_dst = ttnn.from_torch(
        torch.zeros(total_gather_tiles, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ag_dummy_mem,
        tile=a_tile,
        **from_torch_kwargs,
    )
    ttnn_bg_gather_dst = ttnn.from_torch(
        torch.zeros(total_gather_tiles, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ag_dummy_mem,
        tile=a_tile,
        **from_torch_kwargs,
    )

    # Determine face-view tile for intermed/mcast_src CBs
    from models.demos.deepseek_v3_b1.fused_ops.face_view_utils import FACE_HEIGHT, FACE_WIDTH, can_use_face_view

    use_face_view = can_use_face_view(M, 32, k_parallel, n_parallel)
    assert use_face_view, "Expected face_view=True for M=1, tile_w=32, k_parallel=8, n_parallel=8"
    face_tile = ttnn.Tile([FACE_HEIGHT, FACE_WIDTH])

    # CB 29: Gate/Up matmul output (1 tile per core on 128 compute cores)
    gu_out_shard = ttnn.ShardSpec(compute_core_grid, (M, 32), ttnn.ShardOrientation.ROW_MAJOR)
    gu_out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gu_out_shard)
    num_compute_cores = len(compute_cores_list)
    ttnn_gu_out = ttnn.from_torch(
        torch.zeros((M * num_compute_cores, 32), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gu_out_mem,
        tile=a_tile,
        **from_torch_kwargs,
    )

    # CB 32: Gated reduce intermediate (2 face tiles on sender core)
    intermed_shard = ttnn.ShardSpec(sender_core_grid, (2 * FACE_HEIGHT, FACE_WIDTH), ttnn.ShardOrientation.ROW_MAJOR)
    intermed_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, intermed_shard)
    ttnn_intermed = ttnn.from_torch(
        torch.zeros((2 * FACE_HEIGHT, FACE_WIDTH), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=intermed_mem,
        tile=face_tile,
        **from_torch_kwargs,
    )

    # CB 33: Gated reduce output / down mcast source (1 face tile on sender core)
    mcast_src_shard = ttnn.ShardSpec(sender_core_grid, (FACE_HEIGHT, FACE_WIDTH), ttnn.ShardOrientation.ROW_MAJOR)
    mcast_src_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, mcast_src_shard)
    ttnn_down_mcast_src = ttnn.from_torch(
        torch.zeros((FACE_HEIGHT, FACE_WIDTH), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mcast_src_mem,
        tile=face_tile,
        **from_torch_kwargs,
    )

    # CB 36: Down proj matmul output (N_per_core/32 tiles of [1,32] per core on 112 matmul cores)
    down_matmul_out_shard = ttnn.ShardSpec(matmul_core_grid, (M, N_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    down_matmul_out_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, down_matmul_out_shard
    )
    num_matmul_cores = DownProj.NUM_MATMUL_CORES
    ttnn_down_matmul_out = ttnn.from_torch(
        torch.zeros((M, N_per_core * num_matmul_cores), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=down_matmul_out_mem,
        tile=out_tile,
        **from_torch_kwargs,
    )

    # CB 37: Residual add output (same shape as down matmul output on 112 matmul cores)
    residual_add_out_shard = ttnn.ShardSpec(matmul_core_grid, (M, N_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    residual_add_out_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, residual_add_out_shard
    )
    ttnn_residual_add_out = ttnn.from_torch(
        torch.zeros((M, N_per_core * num_matmul_cores), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=residual_add_out_mem,
        tile=out_tile,
        **from_torch_kwargs,
    )

    return {
        # TTNN tensors
        "ttnn_activation": ttnn_activation,
        "shared_gate_weights_overlapped": gate_ov,
        "shared_up_weights_overlapped": up_ov,
        "ttnn_down_weights": ttnn_down_weights,
        "ttnn_output": ttnn_output,
        "ttnn_down_mcast_dst": ttnn_down_mcast_dst,
        "ttnn_output_mcast_dst": ttnn_output_mcast_dst,
        # Params
        "k_parallel": k_parallel,
        "n_parallel": n_parallel,
        "moe_tp": moe_tp,
        "K_down": K_down,
        # Tensor-backed CB tensors
        "ttnn_ag_gather_dst": ttnn_ag_gather_dst,
        "ttnn_bg_gather_dst": ttnn_bg_gather_dst,
        "ttnn_gu_out": ttnn_gu_out,
        "ttnn_intermed": ttnn_intermed,
        "ttnn_down_mcast_src": ttnn_down_mcast_src,
        "ttnn_down_matmul_out": ttnn_down_matmul_out,
        "ttnn_residual_add_out": ttnn_residual_add_out,
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
def create_routed_expert_tensors(device, use_hardcoded_expert_index, mesh_mapper=None):
    """
    Create all tensors needed for MoE routed expert test.
    Directly extracted from the working inline test setup.

    Args:
        device: TT device or mesh device
        use_hardcoded_expert_index: Whether to use hardcoded expert index (1 expert vs 256)
        mesh_mapper: Optional mesh mapper for multi-device replication

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

    # num_experts: for hardcoded, need one per device in mesh; for dynamic, need all 256
    num_experts = device.get_num_devices() if use_hardcoded_expert_index else 256

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

    from_torch_kwargs = {"mesh_mapper": mesh_mapper} if mesh_mapper else {}

    # ── Residual mcast source tensor (raw input on sender core, RMSNorm input) ──
    residual_mcast_src_shard = ttnn.ShardSpec(input_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR)
    residual_mcast_src_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, residual_mcast_src_shard
    )
    ttnn_residual_mcast_src = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=residual_mcast_src_mem,
        tile=tile_1x32,
        **from_torch_kwargs,
    )

    # ── RMSNorm gamma weights [1, K] on sender core ──
    torch_rmsnorm_gamma = torch.randn(1, K, dtype=torch.bfloat16).float()
    rmsnorm_gamma_shard = ttnn.ShardSpec(input_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR)
    rmsnorm_gamma_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, rmsnorm_gamma_shard
    )
    ttnn_rmsnorm_gamma = ttnn.from_torch(
        torch_rmsnorm_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=rmsnorm_gamma_mem,
        tile=tile_1x32,
        **from_torch_kwargs,
    )

    # ── RMSNorm output [M, K] on sender core (L1 backing for compute output) ──
    ttnn_rmsnorm_output = ttnn.from_torch(
        torch.zeros(M, K, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        tile=tile_1x32,
        **from_torch_kwargs,
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
        **from_torch_kwargs,
    )

    # ── Residual mcast destination tensor (on full mcast grid, populated by residual mcast) ──
    residual_mcast_dst_shard = ttnn.ShardSpec(mcast_output_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR)
    residual_mcast_dst_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, residual_mcast_dst_shard
    )
    ttnn_residual_mcast_dst = ttnn.from_torch(
        torch.zeros(M, K, dtype=torch.bfloat16).float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=residual_mcast_dst_mem,
        tile=tile_1x32,
        **from_torch_kwargs,
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
        **from_torch_kwargs,
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
        **from_torch_kwargs,
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
        **from_torch_kwargs,
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
        **from_torch_kwargs,
    )

    # Gate indices tensor
    ttnn_gate_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_input_mem_config,
        tile=tile_16x16,
        **from_torch_kwargs,
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
        **from_torch_kwargs,
    )

    # Gate output indices tensor [1, 16] on sender core
    gate_output_indices_tensor = ttnn.from_torch(
        torch.zeros((1, 16), dtype=torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_output_mem_config,
        tile=tile_1x16,
        **from_torch_kwargs,
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
        **from_torch_kwargs,
    )

    # Expert scale tensor [1, 16] on mcast grid
    expert_scale_tensor = ttnn.from_torch(
        torch.zeros((1, 16), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=expert_index_mem_config,
        tile=tile_1x16,
        **from_torch_kwargs,
    )

    # ── Compute dimensions for expert DRAM matmul ──
    num_banks = device.dram_grid_size().x
    tile_w = 32
    gate_proj_N_padded = ((gate_proj_N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    down_proj_K = gate_proj_N
    down_proj_N = K
    down_proj_N_padded = ((down_proj_N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    per_core_gate_N = gate_proj_N_padded // num_banks

    # ── Generate expert weights for validation ──
    def _gen_experts(num_exp, K_dim, N_padded, seed):
        stacked = torch.zeros(num_exp, K_dim, N_padded, dtype=torch.bfloat16)
        validation = {}
        for i in range(num_exp):
            torch.manual_seed(seed + i)
            w = torch.randn(1, 1, K_dim, N_padded).clamp(-2, 2).bfloat16()
            validation[i] = w.clone()
            stacked[i] = w.reshape(K_dim, N_padded)
            if (i + 1) % 32 == 0:
                logger.info(f"  Generated {i + 1}/{num_exp} experts (seed={seed})")
        return stacked, validation

    gate_stacked, expert_weights_dict = _gen_experts(num_experts, gate_proj_K, gate_proj_N_padded, seed=0)
    up_stacked, up_proj_weights_dict = _gen_experts(num_experts, gate_proj_K, gate_proj_N_padded, seed=256)
    down_stacked, down_proj_weights_dict = _gen_experts(num_experts, down_proj_K, down_proj_N_padded, seed=512)

    # ── Upload expert weights via BlitzDecodeWeights ──
    bdw = BlitzDecodeWeights(device)
    gate_proj_expert_tensors, up_proj_expert_tensors, down_proj_expert_tensors = bdw.get_tt_moe_routed_expert_weights(
        gate_stacked, up_stacked, down_stacked
    )
    gate_proj_weights = gate_proj_expert_tensors[0]
    up_proj_weights = up_proj_expert_tensors[0]
    down_proj_weights = down_proj_expert_tensors[0]

    # ── Create matmul output tensors (WIDTH_SHARDED in L1) ──
    def _create_dram_mm_output(N_pad, per_core_N_val):
        out_tile = ttnn.Tile([M, tile_w])
        out_shard = ttnn.ShardSpec(gate_proj_core_ranges, [M, per_core_N_val], ttnn.ShardOrientation.ROW_MAJOR)
        out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, out_shard)
        return ttnn.from_torch(
            torch.zeros(1, 1, M, N_pad).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=out_mem,
            tile=out_tile,
            **from_torch_kwargs,
        )

    gate_proj_output = _create_dram_mm_output(gate_proj_N_padded, per_core_gate_N)
    up_proj_mm_out_tensor = _create_dram_mm_output(gate_proj_N_padded, per_core_gate_N)

    # Fused output tensor (same layout as gate/up output)
    fused_output_tensor = ttnn.from_torch(
        torch.zeros([1, 1, M, gate_proj_N_padded]).bfloat16().float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=up_proj_mm_out_tensor.memory_config(),
        tile=up_proj_mm_out_tensor.get_tile(),
        **from_torch_kwargs,
    )

    # down_proj intermediate tensors
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
        **from_torch_kwargs,
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
        **from_torch_kwargs,
    )

    per_core_down_proj_N = down_proj_N_padded // num_banks
    down_proj_output = _create_dram_mm_output(down_proj_N_padded, per_core_down_proj_N)

    # fused_add tensor
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
        **from_torch_kwargs,
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
        **from_torch_kwargs,
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
            **from_torch_kwargs,
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
        **from_torch_kwargs,
    )

    return {
        # TTNN tensors for op()
        "ttnn_rmsnorm_output": ttnn_rmsnorm_output,
        "ttnn_residual_mcast_src": ttnn_residual_mcast_src,
        "ttnn_residual_mcast_dst": ttnn_residual_mcast_dst,
        "ttnn_rmsnorm_gamma": ttnn_rmsnorm_gamma,
        "torch_rmsnorm_gamma": torch_rmsnorm_gamma,
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
        "final_output_total_width": final_output_total_width,
        "final_output_mem_config": final_output_mem_config,
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
    [True, pytest.param(False, marks=pytest.mark.skip_post_commit)],
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

    num_iterations = 100
    ttnn_result_scores, ttnn_result_indices, ttnn_result_final = MoeOp.op(
        r["ttnn_rmsnorm_output"],
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
        s["ttnn_output_mcast_dst"],
        r["final_output_tensor"],
        r["gate_proj_in1_buf_tensor"],
        r["down_proj_in1_buf_tensor"],
        r["mul_scalar_buf_tensor"],
        # RMSNorm gamma weights (sender core)
        rmsnorm_gamma_tensor=r["ttnn_rmsnorm_gamma"],
        # Shared expert tensors
        shared_residual_mcast_src_tensor=r["ttnn_residual_mcast_src"],
        shared_gate_weights_overlapped=s["shared_gate_weights_overlapped"],
        shared_up_weights_overlapped=s["shared_up_weights_overlapped"],
        shared_residual_mcast_dst_tensor=r["ttnn_residual_mcast_dst"],
        shared_down_mcast_dst_tensor=s["ttnn_down_mcast_dst"],
        shared_down_weights_tensor=s["ttnn_down_weights"],
        shared_output_tensor=s["ttnn_output"],
        # Shared expert tensor-backed CB tensors
        shared_ag_gather_dst_tensor=s["ttnn_ag_gather_dst"],
        shared_bg_gather_dst_tensor=s["ttnn_bg_gather_dst"],
        shared_gu_out_tensor=s["ttnn_gu_out"],
        shared_intermed_tensor=s["ttnn_intermed"],
        shared_down_mcast_src_tensor=s["ttnn_down_mcast_src"],
        shared_down_matmul_out_tensor=s["ttnn_down_matmul_out"],
        shared_residual_add_out_tensor=s["ttnn_residual_add_out"],
        shared_k_parallel=s["k_parallel"],
        shared_n_parallel=s["n_parallel"],
        use_hardcoded_expert_index=use_hardcoded_expert_index,
        num_iterations=num_iterations,
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

    # Compute fused MoE golden (routed + shared expert + eltwise add)
    torch_expected_scores, torch_expected_indices, torch_expected_final = MoeOp.golden(
        r["torch_input"],
        r["torch_gate_mm_weights"],
        r["torch_bias"],
        shared_gate_weights=s["torch_gate_weights"],
        shared_up_weights=s["torch_up_weights"],
        shared_down_weights=s["torch_down_weights"],
        gate_proj_weights_dict=r["expert_weights_dict"],
        up_proj_weights_dict=r["up_proj_weights_dict"],
        down_proj_weights_dict=r["down_proj_weights_dict"],
        eps=r["gate_eps"],
        scaling_factor=r["gate_scaling_factor"],
        use_hardcoded_expert_index=use_hardcoded_expert_index,
        rmsnorm_gamma=r["torch_rmsnorm_gamma"],
        rmsnorm_epsilon=1e-6,
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

    passing, pcc = comp_pcc(torch_expected_final, output_final_valid, 0.97)
    logger.info(f"Fused MoE PCC: {pcc}")
    assert passing, f"Fused MoE PCC check failed: {pcc}"

    logger.info(f"Fused MoE test PASSED! (PCC={pcc})")


# ============================================================================
# Test: Fused MoE with reduce_to_one on 4x2 mesh
# ============================================================================
@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D})],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
@pytest.mark.parametrize("use_hardcoded_expert_index", [True, pytest.param(False, marks=pytest.mark.skip_post_commit)])
def test_moe_fused_with_reduce(bh_2d_mesh_device, use_hardcoded_expert_index):
    """
    Test fused MoE with reduce_to_one on 4x2 mesh.

    Each of 8 devices runs the full fused MoE (routed + shared expert),
    then results are reduced (summed) across all devices to ROOT1.
    """
    num_devices = 8
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip(
            f"Test requires {num_devices} devices, mesh has "
            f"{bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1]}"
        )

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    logger.info(f"Created submesh with shape: {submesh.shape}")

    device_grid = submesh.compute_with_storage_grid_size()
    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for 13x10")

    M = 1
    K = 7168

    logger.info(f"Testing fused MoE with reduce: K={K}")

    # ── Create MoE tensors (replicated across mesh) ──
    mesh_mapper = ttnn.ReplicateTensorToMesh(submesh)
    r = create_routed_expert_tensors(submesh, use_hardcoded_expert_index, mesh_mapper=mesh_mapper)
    mcast_grid = r["ttnn_mcast_output"].memory_config().shard_spec.grid
    s = create_shared_expert_tensors(submesh, M, K, mcast_grid, mesh_mapper=mesh_mapper)

    # ── ReduceToOne tensors and semaphores ──
    root_coord = (1, 1)

    # Reduce mesh mapper (2D shard across 4x2 mesh)
    reduce_mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], submesh.shape)
    reduce_mesh_mapper = ttnn.create_mesh_mapper(submesh, reduce_mesh_mapper_config)

    tile_1x32 = ttnn.Tile([1, 32])
    final_output_total_width = r["final_output_total_width"]
    final_output_mem_config = r["final_output_mem_config"]

    # 3 intermediate tensors for 3 reduction rounds (same shape as final_output)
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
    logger.info("Created 3 intermediate tensors for reduce rounds")

    # Reduce output tensor (single-core sharded on each device)
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

    # 4 global semaphores for reduce synchronization (round1, round2, round3, exit)
    num_cores = compute_grid.x * compute_grid.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, row_wise=True)
    ttnn.synchronize_device(submesh)
    reduce_semaphores = [ttnn.create_global_semaphore(submesh, available_cores, 0) for _ in range(4)]
    ttnn.synchronize_device(submesh)
    logger.info("Created 4 global semaphores for reduce synchronization")

    # ── Run fused MoE op with reduce (looping inside kernel) ──
    num_iterations = 100
    ttnn_result_scores, ttnn_result_indices, ttnn_result_reduce = MoeOp.op(
        r["ttnn_rmsnorm_output"],
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
        s["ttnn_output_mcast_dst"],  # fused_add_tensor (shared expert output)
        r["final_output_tensor"],
        r["gate_proj_in1_buf_tensor"],
        r["down_proj_in1_buf_tensor"],
        r["mul_scalar_buf_tensor"],
        # RMSNorm gamma
        rmsnorm_gamma_tensor=r["ttnn_rmsnorm_gamma"],
        # Shared expert tensors
        shared_residual_mcast_src_tensor=r["ttnn_residual_mcast_src"],
        shared_gate_weights_overlapped=s["shared_gate_weights_overlapped"],
        shared_up_weights_overlapped=s["shared_up_weights_overlapped"],
        shared_residual_mcast_dst_tensor=r["ttnn_residual_mcast_dst"],
        shared_down_mcast_dst_tensor=s["ttnn_down_mcast_dst"],
        shared_down_weights_tensor=s["ttnn_down_weights"],
        shared_output_tensor=s["ttnn_output"],
        # Shared expert tensor-backed CB tensors
        shared_ag_gather_dst_tensor=s["ttnn_ag_gather_dst"],
        shared_bg_gather_dst_tensor=s["ttnn_bg_gather_dst"],
        shared_gu_out_tensor=s["ttnn_gu_out"],
        shared_intermed_tensor=s["ttnn_intermed"],
        shared_down_mcast_src_tensor=s["ttnn_down_mcast_src"],
        shared_down_matmul_out_tensor=s["ttnn_down_matmul_out"],
        shared_residual_add_out_tensor=s["ttnn_residual_add_out"],
        shared_k_parallel=s["k_parallel"],
        shared_n_parallel=s["n_parallel"],
        use_hardcoded_expert_index=use_hardcoded_expert_index,
        num_iterations=num_iterations,
        # ReduceToOne parameters
        reduce_intermediate_tensors=intermediate_tensors,
        reduce_output_tensor=reduce_output_tensor,
        reduce_semaphores=reduce_semaphores,
        reduce_root_coord=ttnn.MeshCoordinate(root_coord),
    )
    ttnn.synchronize_device(submesh)
    logger.info(f"Fused MoE with reduce: {num_iterations} iterations completed")

    # ── Verify results ──
    # Read gate scores/indices from device (needed for per-device golden)
    device_gate_indices = ttnn.to_torch(ttnn_result_indices, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))
    device_gate_scores = ttnn.to_torch(ttnn_result_scores, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))

    # Compute expected output for each device, then sum
    # Each device uses a different hardcoded expert index (chip_id)
    # and a different TP shard of shared expert weights
    K_down = s["K_down"]
    expected_final_outputs = []
    for device_idx in range(num_devices):
        chip_id = device_idx

        if use_hardcoded_expert_index:
            actual_expert_idx = chip_id
            actual_expert_scale = device_gate_scores[0].flatten()[chip_id].float()
        else:
            actual_expert_idx = int(device_gate_indices[0].flatten()[chip_id].item())
            actual_expert_scale = device_gate_scores[0].flatten()[chip_id].float()

        shared_gate_shard = s["torch_gate_weights"][:, device_idx * K_down : (device_idx + 1) * K_down]
        shared_up_shard = s["torch_up_weights"][:, device_idx * K_down : (device_idx + 1) * K_down]
        shared_down_shard = s["torch_down_weights"][device_idx * K_down : (device_idx + 1) * K_down, :]

        _, _, torch_expected_final = MoeOp.golden(
            r["torch_input"],
            r["torch_gate_mm_weights"],
            r["torch_bias"],
            shared_gate_weights=shared_gate_shard,
            shared_up_weights=shared_up_shard,
            shared_down_weights=shared_down_shard,
            gate_proj_weights_dict=r["expert_weights_dict"],
            up_proj_weights_dict=r["up_proj_weights_dict"],
            down_proj_weights_dict=r["down_proj_weights_dict"],
            eps=r["gate_eps"],
            scaling_factor=r["gate_scaling_factor"],
            use_hardcoded_expert_index=True,
            hardcoded_expert_index=actual_expert_idx,
            explicit_expert_scale=actual_expert_scale,
            rmsnorm_gamma=r["torch_rmsnorm_gamma"],
            rmsnorm_epsilon=1e-6,
        )
        expected_final_outputs.append(torch_expected_final)
        logger.info(
            f"Device {device_idx}: expert_idx={actual_expert_idx}, "
            f"expert_scale={actual_expert_scale:.4f}, "
            f"output range=[{torch_expected_final.min():.4f}, {torch_expected_final.max():.4f}]"
        )

    # Expected reduce output = sum of all per-device outputs
    expected_reduce_output = sum(expected_final_outputs)

    # Get actual reduce output from ROOT1 device
    reduce_output_torch = ttnn.to_torch(
        ttnn_result_reduce,
        mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0),
    )

    # ROOT1 is at row 1, col 1 -> device_idx = 1*2 + 1 = 3
    root_device_idx = root_coord[0] * submesh.shape[1] + root_coord[1]
    reduce_output_root = reduce_output_torch[root_device_idx]

    # Extract valid portion (remove per-core padding)
    reduce_output_valid = extract_routed_expert_output(
        reduce_output_root.unsqueeze(0),
        r["num_gate_proj_cores"],
        r["final_output_width_per_core"],
        r["per_core_down_proj_N"],
    )

    # Verify reduce output
    passing, pcc_output = comp_pcc(expected_reduce_output.flatten(), reduce_output_valid.flatten(), 0.97)
    logger.info(f"Reduce output PCC: {pcc_output}")
    assert passing, f"Reduce output PCC check failed: {pcc_output}"

    logger.info("Fused MoE with reduce test PASSED!")
