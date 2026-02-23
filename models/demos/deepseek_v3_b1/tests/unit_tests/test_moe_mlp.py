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

    compute_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in compute_cores_list])

    gate_ov, up_ov, ttnn_down_weights = bdw.get_tt_moe_shared_expert_weights(
        torch_gate_weights, torch_up_weights, torch_down_weights
    )

    return {
        # TTNN tensors
        "shared_gate_weights_overlapped": gate_ov,
        "shared_up_weights_overlapped": up_ov,
        "ttnn_down_weights": ttnn_down_weights,
        # Params
        "k_parallel": k_parallel,
        "n_parallel": n_parallel,
        "moe_tp": moe_tp,
        "K_down": K_down,
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
def create_routed_expert_tensors(
    device, use_hardcoded_expert_index=False, mesh_mapper=None, create_final_output=True, enable_routing=True
):
    """
    Create all tensors needed for MoE routed expert test.

    When enable_routing=False, skips routing-specific tensors (gate MM weights,
    gate bias/indices, gate output scores/indices) and uses a single expert.

    Args:
        device: TT device or mesh device
        use_hardcoded_expert_index: Whether to use hardcoded expert index (routing only)
        mesh_mapper: Optional mesh mapper for multi-device replication
        create_final_output: If True, create final_output_tensor
        enable_routing: If True, create routing tensors. If False, skip them.

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

    # num_experts: 1 when no routing, otherwise per-device or all 256
    if not enable_routing:
        num_experts = 1
    elif use_hardcoded_expert_index:
        num_experts = device.get_num_devices()
    else:
        num_experts = 256

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

    # Get optimal DRAM bank cores for DRAM streaming matmul + SiLU
    gate_proj_noc = ttnn.NOC.NOC_0
    gate_proj_worker_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(gate_proj_noc)
    gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in gate_proj_worker_cores])
    num_gate_proj_cores = len(gate_proj_worker_cores)

    # Routing tensors (only when enable_routing=True)
    ttnn_gate_mm_weights = None
    ttnn_gate_bias = None
    ttnn_gate_indices = None
    gate_output_scores_tensor = None
    gate_output_indices_tensor = None

    if enable_routing:
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

        # Gate bias and indices tensors: [16, 16] on sender core
        gate_input_shard_spec = ttnn.ShardSpec(
            input_core_grid,
            (16, 16),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        gate_input_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gate_input_shard_spec
        )

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

    # ── Compute dimensions for expert DRAM matmul ──
    num_banks = device.dram_grid_size().x
    tile_w = 32
    gate_proj_N_padded = ((gate_proj_N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    down_proj_K = gate_proj_N
    down_proj_N = K
    down_proj_N_padded = ((down_proj_N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
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
    final_output_tensor = None
    if create_final_output:
        final_output_tensor = ttnn.from_torch(
            torch.zeros([1, 1, 1, final_output_total_width]).bfloat16().float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=final_output_mem_config,
            tile=tile_1x32,
            **from_torch_kwargs,
        )

    return {
        # TTNN tensors for op()
        "ttnn_residual_mcast_src": ttnn_residual_mcast_src,
        "ttnn_rmsnorm_gamma": ttnn_rmsnorm_gamma,
        "torch_rmsnorm_gamma": torch_rmsnorm_gamma,
        "ttnn_gate_mm_weights": ttnn_gate_mm_weights,
        "ttnn_gate_bias": ttnn_gate_bias,
        "ttnn_gate_indices": ttnn_gate_indices,
        "gate_output_scores_tensor": gate_output_scores_tensor,
        "gate_output_indices_tensor": gate_output_indices_tensor,
        "gate_proj_weights": gate_proj_weights,
        "up_proj_weights": up_proj_weights,
        "down_proj_weights": down_proj_weights,
        "final_output_tensor": final_output_tensor,
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
    sender_core = r["ttnn_residual_mcast_src"].memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
    s = create_shared_expert_tensors(device, M, K, mcast_grid)

    # ── Create sdpa_kv_cache_buffer for CB memory overlap ──
    kv_cache_shard_height = 256
    kvpe_dim = 576
    num_mcast_cores = len(ttnn.corerange_to_cores(mcast_grid))
    kv_cache_shard_spec = ttnn.ShardSpec(mcast_grid, (kv_cache_shard_height, kvpe_dim), ttnn.ShardOrientation.ROW_MAJOR)
    sdpa_kv_cache_buffer = ttnn.from_torch(
        torch.zeros((kv_cache_shard_height * num_mcast_cores, kvpe_dim), dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, kv_cache_shard_spec
        ),
    )

    # ── Create sdpa_out_interm_buffer for overflow CBs (26, 30, 31) ──
    device_grid_size = device.compute_with_storage_grid_size()
    sdpa_out_interm_shard_height = 40
    sdpa_out_interm_shard_width = 544
    full_device_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )
    num_full_cores = device_grid_size.x * device_grid_size.y
    sdpa_out_interm_shard_spec = ttnn.ShardSpec(
        full_device_grid,
        (sdpa_out_interm_shard_height, sdpa_out_interm_shard_width),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sdpa_out_interm_buffer = ttnn.from_torch(
        torch.zeros((sdpa_out_interm_shard_height * num_full_cores, sdpa_out_interm_shard_width), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            sdpa_out_interm_shard_spec,
        ),
        tile=ttnn.Tile([8, 32]),
    )

    num_iterations = 100
    ttnn_result_scores, ttnn_result_indices, ttnn_result_final = MoeOp.op(
        r["ttnn_residual_mcast_src"],
        r["ttnn_gate_mm_weights"],
        r["ttnn_gate_bias"],
        r["ttnn_gate_indices"],
        r["gate_output_scores_tensor"],
        r["gate_output_indices_tensor"],
        r["gate_proj_weights"],
        r["up_proj_weights"],
        r["down_proj_weights"],
        r["final_output_tensor"],
        r["ttnn_rmsnorm_gamma"],
        # Shared expert tensors
        shared_gate_weights_overlapped=s["shared_gate_weights_overlapped"],
        shared_up_weights_overlapped=s["shared_up_weights_overlapped"],
        shared_down_weights_tensor=s["ttnn_down_weights"],
        shared_k_parallel=s["k_parallel"],
        shared_n_parallel=s["n_parallel"],
        use_hardcoded_expert_index=use_hardcoded_expert_index,
        sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
        sdpa_out_interm_buffer=sdpa_out_interm_buffer,
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
        shared_gate_weights=s["torch_gate_weights"],
        shared_up_weights=s["torch_up_weights"],
        shared_down_weights=s["torch_down_weights"],
        gate_proj_weights_dict=r["expert_weights_dict"],
        up_proj_weights_dict=r["up_proj_weights_dict"],
        down_proj_weights_dict=r["down_proj_weights_dict"],
        rmsnorm_gamma=r["torch_rmsnorm_gamma"],
        rmsnorm_epsilon=1e-6,
        routing_weights_tensor=r["torch_gate_mm_weights"],
        bias_tensor=r["torch_bias"],
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
    r = create_routed_expert_tensors(
        submesh, use_hardcoded_expert_index, mesh_mapper=mesh_mapper, create_final_output=False
    )
    sender_core = r["ttnn_residual_mcast_src"].memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
    s = create_shared_expert_tensors(submesh, M, K, mcast_grid, mesh_mapper=mesh_mapper)

    # ── Create SDPA buffers for CB memory overlap (required by fused MoE) ──
    kv_cache_shard_height = 256
    kvpe_dim = 576
    num_mcast_cores = len(ttnn.corerange_to_cores(mcast_grid))
    kv_cache_shard_spec = ttnn.ShardSpec(mcast_grid, (kv_cache_shard_height, kvpe_dim), ttnn.ShardOrientation.ROW_MAJOR)
    sdpa_kv_cache_buffer = ttnn.from_torch(
        torch.zeros((kv_cache_shard_height * num_mcast_cores, kvpe_dim), dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, kv_cache_shard_spec
        ),
    )

    sdpa_out_interm_shard_height = 40
    sdpa_out_interm_shard_width = 544
    full_device_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid.x - 1, device_grid.y - 1))}
    )
    num_full_cores = device_grid.x * device_grid.y
    sdpa_out_interm_shard_spec = ttnn.ShardSpec(
        full_device_grid,
        (sdpa_out_interm_shard_height, sdpa_out_interm_shard_width),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sdpa_out_interm_buffer = ttnn.from_torch(
        torch.zeros((sdpa_out_interm_shard_height * num_full_cores, sdpa_out_interm_shard_width), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            sdpa_out_interm_shard_spec,
        ),
        tile=ttnn.Tile([8, 32]),
    )

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
        r["ttnn_residual_mcast_src"],
        r["ttnn_gate_mm_weights"],
        r["ttnn_gate_bias"],
        r["ttnn_gate_indices"],
        r["gate_output_scores_tensor"],
        r["gate_output_indices_tensor"],
        r["gate_proj_weights"],
        r["up_proj_weights"],
        r["down_proj_weights"],
        r["final_output_tensor"],
        r["ttnn_rmsnorm_gamma"],
        # Shared expert tensors
        shared_gate_weights_overlapped=s["shared_gate_weights_overlapped"],
        shared_up_weights_overlapped=s["shared_up_weights_overlapped"],
        shared_down_weights_tensor=s["ttnn_down_weights"],
        shared_k_parallel=s["k_parallel"],
        shared_n_parallel=s["n_parallel"],
        use_hardcoded_expert_index=use_hardcoded_expert_index,
        sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
        sdpa_out_interm_buffer=sdpa_out_interm_buffer,
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
            shared_gate_weights=shared_gate_shard,
            shared_up_weights=shared_up_shard,
            shared_down_weights=shared_down_shard,
            gate_proj_weights_dict=r["expert_weights_dict"],
            up_proj_weights_dict=r["up_proj_weights_dict"],
            down_proj_weights_dict=r["down_proj_weights_dict"],
            rmsnorm_gamma=r["torch_rmsnorm_gamma"],
            rmsnorm_epsilon=1e-6,
            routing_weights_tensor=r["torch_gate_mm_weights"],
            bias_tensor=r["torch_bias"],
            eps=r["gate_eps"],
            scaling_factor=r["gate_scaling_factor"],
            use_hardcoded_expert_index=True,
            hardcoded_expert_index=actual_expert_idx,
            explicit_expert_scale=actual_expert_scale,
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


# ============================================================================
# Test: Fused MoE with enable_routing=False (dense MLP mode)
# ============================================================================
def test_mlp(device):
    """Test MoeOp with enable_routing=False: same as MLP, no routing logic."""

    device_grid = device.compute_with_storage_grid_size()
    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for 13x10")

    M = 1
    K = 7168

    logger.info(f"Testing MoeOp with enable_routing=False: K={K}")

    # ── Create MLP tensors (no routing) ──
    r = create_routed_expert_tensors(device, enable_routing=False)
    sender_core = r["ttnn_residual_mcast_src"].memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
    s = create_shared_expert_tensors(device, M, K, mcast_grid)

    # ── Create SDPA buffers for CB memory overlap ──
    kv_cache_shard_height = 256
    kvpe_dim = 576
    num_mcast_cores = len(ttnn.corerange_to_cores(mcast_grid))
    kv_cache_shard_spec = ttnn.ShardSpec(mcast_grid, (kv_cache_shard_height, kvpe_dim), ttnn.ShardOrientation.ROW_MAJOR)
    sdpa_kv_cache_buffer = ttnn.from_torch(
        torch.zeros((kv_cache_shard_height * num_mcast_cores, kvpe_dim), dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, kv_cache_shard_spec
        ),
    )

    device_grid_size = device.compute_with_storage_grid_size()
    sdpa_out_interm_shard_height = 40
    sdpa_out_interm_shard_width = 544
    full_device_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )
    num_full_cores = device_grid_size.x * device_grid_size.y
    sdpa_out_interm_shard_spec = ttnn.ShardSpec(
        full_device_grid,
        (sdpa_out_interm_shard_height, sdpa_out_interm_shard_width),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sdpa_out_interm_buffer = ttnn.from_torch(
        torch.zeros((sdpa_out_interm_shard_height * num_full_cores, sdpa_out_interm_shard_width), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            sdpa_out_interm_shard_spec,
        ),
        tile=ttnn.Tile([8, 32]),
    )

    # ── Run MoeOp with enable_routing=False ──
    num_iterations = 100
    ttnn_result_final = MoeOp.op(
        r["ttnn_residual_mcast_src"],
        # No routing tensors
        gate_proj_weights_tensor=r["gate_proj_weights"],
        up_proj_weights_tensor=r["up_proj_weights"],
        down_proj_weights_tensor=r["down_proj_weights"],
        final_output_tensor=r["final_output_tensor"],
        rmsnorm_gamma_tensor=r["ttnn_rmsnorm_gamma"],
        shared_gate_weights_overlapped=s["shared_gate_weights_overlapped"],
        shared_up_weights_overlapped=s["shared_up_weights_overlapped"],
        shared_down_weights_tensor=s["ttnn_down_weights"],
        shared_k_parallel=s["k_parallel"],
        shared_n_parallel=s["n_parallel"],
        enable_routing=False,
        sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
        sdpa_out_interm_buffer=sdpa_out_interm_buffer,
        num_iterations=num_iterations,
    )
    ttnn.synchronize_device(device)
    logger.info(f"MoeOp no-routing: {num_iterations} iterations completed")

    # ── Read back and validate ──
    output_final_torch = ttnn.to_torch(ttnn_result_final)

    output_final_valid = extract_routed_expert_output(
        output_final_torch,
        r["num_gate_proj_cores"],
        r["final_output_width_per_core"],
        r["per_core_down_proj_N"],
    )

    # Compute golden (no routing, no expert scale)
    _, _, torch_expected = MoeOp.golden(
        r["torch_input"],
        shared_gate_weights=s["torch_gate_weights"],
        shared_up_weights=s["torch_up_weights"],
        shared_down_weights=s["torch_down_weights"],
        gate_proj_weights_dict=r["expert_weights_dict"],
        up_proj_weights_dict=r["up_proj_weights_dict"],
        down_proj_weights_dict=r["down_proj_weights_dict"],
        rmsnorm_gamma=r["torch_rmsnorm_gamma"],
        rmsnorm_epsilon=1e-6,
        enable_routing=False,
    )

    passing, pcc = comp_pcc(torch_expected, output_final_valid, 0.97)
    logger.info(f"MoeOp no-routing PCC: {pcc}")
    assert passing, f"MoeOp no-routing PCC check failed: {pcc}"

    logger.info(f"MoeOp no-routing test PASSED! (PCC={pcc})")


# ============================================================================
# Test: Fused MLP (enable_routing=False) with reduce_to_one on 4x2 mesh
# ============================================================================
@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D})],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
def test_mlp_with_reduce(bh_2d_mesh_device):
    """
    Test MoeOp with enable_routing=False and reduce_to_one on 4x2 mesh.

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

    logger.info(f"Testing MoeOp no-routing with reduce: K={K}")

    # ── Create MLP tensors (replicated across mesh) ──
    mesh_mapper = ttnn.ReplicateTensorToMesh(submesh)
    r = create_routed_expert_tensors(submesh, mesh_mapper=mesh_mapper, create_final_output=False, enable_routing=False)
    sender_core = r["ttnn_residual_mcast_src"].memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
    s = create_shared_expert_tensors(submesh, M, K, mcast_grid, mesh_mapper=mesh_mapper)

    # ── Create SDPA buffers for CB memory overlap ──
    kv_cache_shard_height = 256
    kvpe_dim = 576
    num_mcast_cores = len(ttnn.corerange_to_cores(mcast_grid))
    kv_cache_shard_spec = ttnn.ShardSpec(mcast_grid, (kv_cache_shard_height, kvpe_dim), ttnn.ShardOrientation.ROW_MAJOR)
    sdpa_kv_cache_buffer = ttnn.from_torch(
        torch.zeros((kv_cache_shard_height * num_mcast_cores, kvpe_dim), dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, kv_cache_shard_spec
        ),
    )

    sdpa_out_interm_shard_height = 40
    sdpa_out_interm_shard_width = 544
    full_device_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid.x - 1, device_grid.y - 1))}
    )
    num_full_cores = device_grid.x * device_grid.y
    sdpa_out_interm_shard_spec = ttnn.ShardSpec(
        full_device_grid,
        (sdpa_out_interm_shard_height, sdpa_out_interm_shard_width),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sdpa_out_interm_buffer = ttnn.from_torch(
        torch.zeros((sdpa_out_interm_shard_height * num_full_cores, sdpa_out_interm_shard_width), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            sdpa_out_interm_shard_spec,
        ),
        tile=ttnn.Tile([8, 32]),
    )

    # ── ReduceToOne tensors and semaphores ──
    root_coord = (1, 1)

    reduce_mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], submesh.shape)
    reduce_mesh_mapper = ttnn.create_mesh_mapper(submesh, reduce_mesh_mapper_config)

    tile_1x32 = ttnn.Tile([1, 32])
    final_output_total_width = r["final_output_total_width"]
    final_output_mem_config = r["final_output_mem_config"]

    # 3 intermediate tensors for 3 reduction rounds
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

    # 4 global semaphores for reduce synchronization
    num_cores = compute_grid.x * compute_grid.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, row_wise=True)
    ttnn.synchronize_device(submesh)
    reduce_semaphores = [ttnn.create_global_semaphore(submesh, available_cores, 0) for _ in range(4)]
    ttnn.synchronize_device(submesh)
    logger.info("Created 4 global semaphores for reduce synchronization")

    # ── Run MoeOp with enable_routing=False and reduce ──
    num_iterations = 100
    ttnn_result_reduce = MoeOp.op(
        r["ttnn_residual_mcast_src"],
        # No routing tensors
        gate_proj_weights_tensor=r["gate_proj_weights"],
        up_proj_weights_tensor=r["up_proj_weights"],
        down_proj_weights_tensor=r["down_proj_weights"],
        final_output_tensor=r["final_output_tensor"],
        rmsnorm_gamma_tensor=r["ttnn_rmsnorm_gamma"],
        shared_gate_weights_overlapped=s["shared_gate_weights_overlapped"],
        shared_up_weights_overlapped=s["shared_up_weights_overlapped"],
        shared_down_weights_tensor=s["ttnn_down_weights"],
        shared_k_parallel=s["k_parallel"],
        shared_n_parallel=s["n_parallel"],
        enable_routing=False,
        sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
        sdpa_out_interm_buffer=sdpa_out_interm_buffer,
        num_iterations=num_iterations,
        reduce_intermediate_tensors=intermediate_tensors,
        reduce_output_tensor=reduce_output_tensor,
        reduce_semaphores=reduce_semaphores,
        reduce_root_coord=ttnn.MeshCoordinate(root_coord),
    )
    ttnn.synchronize_device(submesh)
    logger.info(f"MoeOp no-routing with reduce: {num_iterations} iterations completed")

    # ── Verify results ──
    # Compute per-device golden with per-device TP shards of shared expert weights
    K_down = s["K_down"]
    expected_final_outputs = []
    for device_idx in range(num_devices):
        shared_gate_shard = s["torch_gate_weights"][:, device_idx * K_down : (device_idx + 1) * K_down]
        shared_up_shard = s["torch_up_weights"][:, device_idx * K_down : (device_idx + 1) * K_down]
        shared_down_shard = s["torch_down_weights"][device_idx * K_down : (device_idx + 1) * K_down, :]

        _, _, device_expected = MoeOp.golden(
            r["torch_input"],
            shared_gate_weights=shared_gate_shard,
            shared_up_weights=shared_up_shard,
            shared_down_weights=shared_down_shard,
            gate_proj_weights_dict=r["expert_weights_dict"],
            up_proj_weights_dict=r["up_proj_weights_dict"],
            down_proj_weights_dict=r["down_proj_weights_dict"],
            rmsnorm_gamma=r["torch_rmsnorm_gamma"],
            rmsnorm_epsilon=1e-6,
            enable_routing=False,
        )
        expected_final_outputs.append(device_expected)

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

    logger.info("MoeOp no-routing with reduce test PASSED!")
