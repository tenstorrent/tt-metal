# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for fused MLP operation (dense MLP + shared expert, no routing).

Runs dense MLP (single expert, no routing) and shared expert on the same input,
validates the combined MLP output via PCC check.

Run:
    pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_mlp.py -v -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, skip_for_wormhole_b0
from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights
from models.demos.deepseek_v3_b1.fused_ops.mlp.op import MlpOp
from models.demos.deepseek_v3_b1.tests.unit_tests.test_moe import (
    create_shared_expert_tensors,
    extract_routed_expert_output,
)


# ============================================================================
# Helper: create all MLP dense expert tensors (no routing)
# ============================================================================
def create_mlp_tensors(device, mesh_mapper=None):
    """
    Create all tensors needed for MLP dense expert test.
    Same as create_routed_expert_tensors but without routing-specific tensors
    (gate MM, gate input/bias/indices, gate output scores/indices,
     expert index/scale, mul scalar buffer).

    Args:
        device: TT device or mesh device
        mesh_mapper: Optional mesh mapper for multi-device replication

    Returns:
        dict with all ttnn tensors, torch tensors, expert dicts, and dimensions.
    """
    M = 1
    K = 7168

    # DRAM matmul parameters (same as MoE routed expert)
    gate_proj_K = K
    gate_proj_N = 2048

    # Always 1 expert for MLP (no routing)
    num_experts = 1

    # Tile definitions
    tile_1x32 = ttnn.Tile([1, 32])

    # Create input tensors
    torch.manual_seed(0)
    torch_input = torch.randn((M, K), dtype=torch.bfloat16)

    # Device grid and sender core
    device_grid_size = device.compute_with_storage_grid_size()
    input_core = ttnn.CoreCoord(device_grid_size.x - 1, 9)
    input_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(input_core, input_core)])
    input_shard_spec = ttnn.ShardSpec(
        input_core_grid,
        (M, K),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    # ── Residual mcast source tensor (raw input on sender core, RMSNorm input) ──
    residual_mcast_src_shard = ttnn.ShardSpec(input_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR)
    residual_mcast_src_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, residual_mcast_src_shard
    )
    from_torch_kwargs = {"mesh_mapper": mesh_mapper} if mesh_mapper else {}

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

    # Get optimal DRAM bank cores for DRAM streaming matmul
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

    # ── Residual mcast destination tensor (on full mcast grid) ──
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

    # ── Compute dimensions for expert DRAM matmul ──
    num_banks = device.dram_grid_size().x
    tile_w = 32
    gate_proj_N_padded = ((gate_proj_N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    down_proj_K = gate_proj_N
    down_proj_N = K
    down_proj_N_padded = ((down_proj_N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    per_core_gate_N = gate_proj_N_padded // num_banks
    per_core_down_proj_N = down_proj_N_padded // num_banks

    # ── Generate expert weights for validation ──
    def _gen_experts(num_exp, K_dim, N_padded, seed):
        stacked = torch.zeros(num_exp, K_dim, N_padded, dtype=torch.bfloat16)
        validation = {}
        for i in range(num_exp):
            torch.manual_seed(seed + i)
            w = torch.randn(1, 1, K_dim, N_padded).clamp(-2, 2).bfloat16()
            validation[i] = w.clone()
            stacked[i] = w.reshape(K_dim, N_padded)
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

    down_proj_output = _create_dram_mm_output(down_proj_N_padded, per_core_down_proj_N)

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

    # No mul_scalar_buf_tensor needed for MLP (no expert scale)

    return {
        # TTNN tensors for op()
        "ttnn_rmsnorm_output": ttnn_rmsnorm_output,
        "ttnn_residual_mcast_src": ttnn_residual_mcast_src,
        "ttnn_residual_mcast_dst": ttnn_residual_mcast_dst,
        "ttnn_rmsnorm_gamma": ttnn_rmsnorm_gamma,
        "ttnn_mcast_output": ttnn_mcast_output,
        "gate_proj_weights": gate_proj_weights,
        "gate_proj_output": gate_proj_output,
        "up_proj_weights": up_proj_weights,
        "up_proj_mm_out_tensor": up_proj_mm_out_tensor,
        "fused_output_tensor": fused_output_tensor,
        "down_proj_gather_output_tensor": down_proj_gather_output_tensor,
        "down_proj_mcast_output_tensor": down_proj_mcast_output_tensor,
        "down_proj_weights": down_proj_weights,
        "down_proj_output": down_proj_output,
        "final_output_tensor": final_output_tensor,
        # Tensor-backed working buffers
        "gate_proj_in1_buf_tensor": gate_up_proj_in1_buf_tensor,
        "down_proj_in1_buf_tensor": down_proj_in1_buf_tensor,
        # Keep-alive references (prevent garbage collection)
        "gate_proj_expert_tensors": gate_proj_expert_tensors,
        "up_proj_expert_tensors": up_proj_expert_tensors,
        "down_proj_expert_tensors": down_proj_expert_tensors,
        # Torch tensors for golden
        "torch_input": torch_input,
        "torch_rmsnorm_gamma": torch_rmsnorm_gamma,
        "expert_weights_dict": expert_weights_dict,
        "up_proj_weights_dict": up_proj_weights_dict,
        "down_proj_weights_dict": down_proj_weights_dict,
        # Dimensions for output extraction
        "num_gate_proj_cores": num_gate_proj_cores,
        "final_output_width_per_core": final_output_width_per_core,
        "final_output_total_width": final_output_total_width,
        "final_output_mem_config": final_output_mem_config,
        "per_core_down_proj_N": per_core_down_proj_N,
        "gate_proj_core_ranges": gate_proj_core_ranges,
    }


# ============================================================================
# Test: Fused MLP (dense MLP + shared expert)
# ============================================================================
def test_mlp_fused(device):
    """Test fused MLP: run dense MLP + shared expert, validate combined output."""

    device_grid = device.compute_with_storage_grid_size()
    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for 13x10")

    M = 1
    K = 7168

    logger.info(f"Testing fused MLP: K={K}")

    # ── Create MLP tensors (no routing) ──
    r = create_mlp_tensors(device)
    mcast_grid = r["ttnn_mcast_output"].memory_config().shard_spec.grid
    s = create_shared_expert_tensors(device, M, K, mcast_grid)

    # ── Run fused MLP op (looping inside kernel) ──
    num_iterations = 100
    ttnn_result_final = MlpOp.op(
        r["ttnn_rmsnorm_output"],
        r["ttnn_mcast_output"],
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
        num_iterations=num_iterations,
    )
    ttnn.synchronize_device(device)
    logger.info(f"Fused MLP: {num_iterations} iterations completed (looped inside kernel)")

    # ── Read back and validate ──
    output_final_torch = ttnn.to_torch(ttnn_result_final)

    output_final_valid = extract_routed_expert_output(
        output_final_torch,
        r["num_gate_proj_cores"],
        r["final_output_width_per_core"],
        r["per_core_down_proj_N"],
    )

    # Compute golden (no routing, no expert scale)
    torch_expected = MlpOp.golden(
        r["torch_input"],
        shared_gate_weights=s["torch_gate_weights"],
        shared_up_weights=s["torch_up_weights"],
        shared_down_weights=s["torch_down_weights"],
        gate_proj_weights=r["expert_weights_dict"][0],
        up_proj_weights=r["up_proj_weights_dict"][0],
        down_proj_weights=r["down_proj_weights_dict"][0],
        rmsnorm_gamma=r["torch_rmsnorm_gamma"],
        rmsnorm_epsilon=1e-6,
    )

    # PCC check (no gate indices/scores since no routing)
    passing, pcc = comp_pcc(torch_expected, output_final_valid, 0.97)
    logger.info(f"Fused MLP PCC: {pcc}")
    assert passing, f"Fused MLP PCC check failed: {pcc}"

    logger.info(f"Fused MLP test PASSED! (PCC={pcc})")


# ============================================================================
# Test: Fused MLP with reduce_to_one on 4x2 mesh
# ============================================================================
@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D})],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
def test_mlp_fused_with_reduce(bh_2d_mesh_device):
    """
    Test fused MLP with reduce_to_one on 4x2 mesh.

    Each of 8 devices runs the full fused MLP (dense MLP + shared expert),
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

    logger.info(f"Testing fused MLP with reduce: K={K}")

    # ── Create MLP tensors (replicated across mesh) ──
    mesh_mapper = ttnn.ReplicateTensorToMesh(submesh)
    r = create_mlp_tensors(submesh, mesh_mapper=mesh_mapper)
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

    # ── Run fused MLP op with reduce (looping inside kernel) ──
    num_iterations = 100
    ttnn_result_reduce = MlpOp.op(
        r["ttnn_rmsnorm_output"],
        r["ttnn_mcast_output"],
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
        num_iterations=num_iterations,
        # ReduceToOne parameters
        reduce_intermediate_tensors=intermediate_tensors,
        reduce_output_tensor=reduce_output_tensor,
        reduce_semaphores=reduce_semaphores,
        reduce_root_coord=ttnn.MeshCoordinate(root_coord),
    )
    ttnn.synchronize_device(submesh)
    logger.info(f"Fused MLP with reduce: {num_iterations} iterations completed")

    # ── Verify results ──
    # Compute per-device golden with per-device TP shards of shared expert weights
    K_down = s["K_down"]
    expected_final_outputs = []
    for device_idx in range(num_devices):
        shared_gate_shard = s["torch_gate_weights"][:, device_idx * K_down : (device_idx + 1) * K_down]
        shared_up_shard = s["torch_up_weights"][:, device_idx * K_down : (device_idx + 1) * K_down]
        shared_down_shard = s["torch_down_weights"][device_idx * K_down : (device_idx + 1) * K_down, :]

        device_expected = MlpOp.golden(
            r["torch_input"],
            shared_gate_weights=shared_gate_shard,
            shared_up_weights=shared_up_shard,
            shared_down_weights=shared_down_shard,
            gate_proj_weights=r["expert_weights_dict"][0],
            up_proj_weights=r["up_proj_weights_dict"][0],
            down_proj_weights=r["down_proj_weights_dict"][0],
            rmsnorm_gamma=r["torch_rmsnorm_gamma"],
            rmsnorm_epsilon=1e-6,
        )
        expected_final_outputs.append(device_expected)

    # Expected reduce output = sum of all per-device outputs (each with unique TP shard)
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

    logger.info("Fused MLP with reduce test PASSED!")
