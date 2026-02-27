# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for fused MoE operation (routed expert + shared expert).

Runs both MoE routed expert and shared expert on the same input,
validates each independently, and verifies the combined MoE output.

Run:
    pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_moe.py -v -s
"""

from typing import Any, NamedTuple

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, skip_for_wormhole_b0
from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights
from models.demos.deepseek_v3_b1.fused_ops.down_proj.op import DownProj
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp
from models.demos.deepseek_v3_b1.fused_ops.shared_expert.op import SharedExpertOp
from models.demos.deepseek_v3_b1.prepare_weights import (
    create_gate_bias_tensor,
    create_gate_indices_tensor,
    prepare_attention_weights,
    prepare_routed_expert_weights,
    prepare_shared_expert_weights,
)


# ============================================================================
# Return types for tensor helpers
# ============================================================================
class SharedExpertTensors(NamedTuple):
    shared_gate_weights_overlapped: Any
    shared_up_weights_overlapped: Any
    ttnn_down_weights: Any
    k_parallel: int
    n_parallel: int
    moe_tp: int
    K_down: int
    torch_activation: Any
    torch_gate_weights: Any
    torch_up_weights: Any
    torch_down_weights: Any
    torch_bias: Any
    N: int


class RoutedExpertTensors(NamedTuple):
    ttnn_residual_mcast_src: Any
    ttnn_rmsnorm_gamma: Any
    torch_rmsnorm_gamma: Any
    ttnn_gate_mm_weights: Any
    ttnn_gate_bias: Any
    ttnn_gate_indices: Any
    gate_output_scores_tensor: Any
    gate_output_indices_tensor: Any
    gate_proj_weights: Any
    up_proj_weights: Any
    down_proj_weights: Any
    final_output_tensor: Any
    gate_proj_expert_tensors: Any
    up_proj_expert_tensors: Any
    down_proj_expert_tensors: Any
    torch_input: Any
    torch_gate_mm_weights: Any
    torch_bias: Any
    expert_weights_dict: Any
    up_proj_weights_dict: Any
    down_proj_weights_dict: Any
    gate_eps: float
    gate_scaling_factor: float
    num_gate_proj_cores: int
    final_output_width_per_core: int
    per_core_down_proj_N: int
    final_output_total_width: int
    final_output_mem_config: Any


# ============================================================================
# Constants (namespaced by usage)
# ============================================================================
class RoutedExpert:
    M = 1
    K = 7168
    N_PER_CORE = 32  # routing matmul width per core
    NUM_CORES = 8  # routing matmul cores
    GATE_PROJ_N = 2048
    GATE_EPS = 1e-20
    GATE_SCALING_FACTOR = 2.5
    TILE_W = 32  # for padding math
    FINAL_OUTPUT_WIDTH_PER_CORE = 32 * 32  # 1024
    INPUT_CORE_Y = 9  # for ttnn.CoreCoord(device_grid_size.x - 1, INPUT_CORE_Y)
    SEED = 0
    GATE_PROJ_EXPERT_SEED = 0
    UP_PROJ_EXPERT_SEED = 256
    DOWN_PROJ_EXPERT_SEED = 512


class SharedExpert:
    K_PARALLEL = 8
    N_PARALLEL = 8
    N_PER_CORE = 64  # N = N_PER_CORE * DownProj.NUM_MATMUL_CORES in helper
    SEED = 100


class SDPA:
    KV_CACHE_SHARD_HEIGHT = 256
    KVPE_DIM = 576
    OUT_INTERM_SHARD_HEIGHT = 40
    OUT_INTERM_SHARD_WIDTH = 544


class Tiles:
    TILE_1x32 = ttnn.Tile([1, 32])
    TILE_32x32 = ttnn.Tile([32, 32])
    TILE_16x16 = ttnn.Tile([16, 16])
    TILE_8x32 = ttnn.Tile([8, 32])


class TestConfig:
    NUM_ITERATIONS = 100
    NUM_DEVICES_4x2 = 8  # for 4x2 mesh tests
    REDUCE_ROOT_COORD = (1, 1)
    REDUCE_NUM_ROUNDS = 3
    REDUCE_NUM_SEMAPHORES = 4


# Layer index used when building state dict for prepare_*_expert_weights (HF key convention)
SHARED_EXPERT_LAYER_IDX = 4
ROUTED_EXPERT_LAYER_IDX = 4


# ============================================================================
# Helper: create all shared-expert tensors
# ============================================================================
def create_shared_expert_tensors(device, M, K_gate, mcast_grid, mesh_mapper=None, *, get_reference_model_state_dict):
    """
    Create all tensors needed by SharedExpertOp.

    The state_dict used for weight preparation is always the one returned by
    get_reference_model_state_dict (same layer/seed as routed path in fused tests).

    Args:
        device: TT device or mesh device
        M: Batch dimension (1)
        K_gate: Gate/Up input dimension (7168)
        mcast_grid: CoreRangeSet for mcast destination (same as routed input mcast)
        mesh_mapper: Optional mesh mapper for multi-device replication
        get_reference_model_state_dict: Required callable from conftest; returns state dict
            for prepare_shared_expert_weights.

    Returns:
        SharedExpertTensors with all ttnn tensors, torch tensors, and validation data.
    """
    k_parallel = SharedExpert.K_PARALLEL
    n_parallel = SharedExpert.N_PARALLEL
    K_down = SharedExpert.N_PARALLEL * 32  # 256
    N = SharedExpert.N_PER_CORE * DownProj.NUM_MATMUL_CORES  # 7168

    # Core grids
    compute_cores_list = sum(SharedExpertOp.build_ab_grids(), [])
    mcast_gather_core = DownProj.MCAST_GATHER_CORE
    sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_gather_core, mcast_gather_core)])

    bdw = BlitzDecodeWeights(device)
    moe_tp = bdw.moe_tp
    K_down_full = K_down * moe_tp

    gate_key = f"model.layers.{SHARED_EXPERT_LAYER_IDX}.mlp.shared_experts.gate_proj.weight"
    up_key = f"model.layers.{SHARED_EXPERT_LAYER_IDX}.mlp.shared_experts.up_proj.weight"
    down_key = f"model.layers.{SHARED_EXPERT_LAYER_IDX}.mlp.shared_experts.down_proj.weight"

    state_dict = get_reference_model_state_dict(
        layer_idx=SHARED_EXPERT_LAYER_IDX,
        is_moe=True,
        seed=RoutedExpert.SEED,
        include_global=False,
    )
    # Expected HF shapes (out_features, in_features): gate/up (GATE_PROJ_N, K), down (K, GATE_PROJ_N)
    expected_gate_up = (RoutedExpert.GATE_PROJ_N, RoutedExpert.K)
    expected_down = (RoutedExpert.K, RoutedExpert.GATE_PROJ_N)
    assert (
        state_dict[gate_key].shape == expected_gate_up
    ), f"shared gate_proj: expected {expected_gate_up}, got {state_dict[gate_key].shape}"
    assert (
        state_dict[up_key].shape == expected_gate_up
    ), f"shared up_proj: expected {expected_gate_up}, got {state_dict[up_key].shape}"
    assert (
        state_dict[down_key].shape == expected_down
    ), f"shared down_proj: expected {expected_down}, got {state_dict[down_key].shape}"
    torch_gate_weights = state_dict[gate_key].T.contiguous()
    torch_up_weights = state_dict[up_key].T.contiguous()
    torch_down_weights = state_dict[down_key].T.contiguous()
    # Reference state dict has full logical (2048); slice to per-TP for single-device golden
    if moe_tp == 1 and torch_gate_weights.shape[1] == K_down_full * 8:
        torch_gate_weights = torch_gate_weights[:, :K_down_full].contiguous()
        torch_up_weights = torch_up_weights[:, :K_down_full].contiguous()
        torch_down_weights = torch_down_weights[:K_down_full, :].contiguous()
    torch.manual_seed(RoutedExpert.SEED)
    torch_activation = torch.randn((M, K_gate), dtype=torch.bfloat16)
    torch_bias = torch.randn((M, N), dtype=torch.bfloat16)

    shared_weights = prepare_shared_expert_weights(
        bdw, state_dict, layer_idx=SHARED_EXPERT_LAYER_IDX, is_moe=True, move_to_device=True
    )

    return SharedExpertTensors(
        shared_gate_weights_overlapped=shared_weights.shared_gate_proj,
        shared_up_weights_overlapped=shared_weights.shared_up_proj,
        ttnn_down_weights=shared_weights.shared_down_proj,
        k_parallel=k_parallel,
        n_parallel=n_parallel,
        moe_tp=moe_tp,
        K_down=K_down,
        torch_activation=torch_activation,
        torch_gate_weights=torch_gate_weights,
        torch_up_weights=torch_up_weights,
        torch_down_weights=torch_down_weights,
        torch_bias=torch_bias,
        N=N,
    )


# ============================================================================
# Helper: create all routed-expert tensors
# ============================================================================
def create_routed_expert_tensors(
    device,
    use_hardcoded_expert_index=False,
    mesh_mapper=None,
    create_final_output=True,
    enable_routing=True,
    *,
    get_reference_model_state_dict,
):
    """
    Create all tensors needed for MoE routed expert test.

    The state_dict from get_reference_model_state_dict is never mutated. Gate weight
    and bias are always read from it for both device preparation and golden reference.

    When enable_routing=False, skips routing-specific tensors (gate MM weights,
    gate bias/indices, gate output scores/indices) and uses a single expert.

    Args:
        device: TT device or mesh device
        use_hardcoded_expert_index: Whether to use hardcoded expert index (routing only)
        mesh_mapper: Optional mesh mapper for multi-device replication
        create_final_output: If True, create final_output_tensor
        enable_routing: If True, create routing tensors. If False, skip them.
        get_reference_model_state_dict: Required callable from conftest; returns
            state dict (read-only when using HF weights).

    Returns:
        RoutedExpertTensors with all ttnn tensors, torch tensors, expert dicts, and dimensions.
    """
    # MoE router: [1, 7168] x [7168, 256] with 8 cores
    M = RoutedExpert.M
    K = RoutedExpert.K
    N_per_core = RoutedExpert.N_PER_CORE
    num_cores = RoutedExpert.NUM_CORES
    N = N_per_core * num_cores  # 256 total output width (routing matmul)

    # DRAM matmul + SiLU parameters
    gate_proj_K = K
    gate_proj_N = RoutedExpert.GATE_PROJ_N

    # num_experts: 1 when no routing, otherwise per-device or all 256
    if not enable_routing:
        num_experts = 1
    elif use_hardcoded_expert_index:
        num_experts = device.get_num_devices()
    else:
        num_experts = 256

    # Gate parameters (must match op.py)
    gate_eps = RoutedExpert.GATE_EPS
    gate_scaling_factor = RoutedExpert.GATE_SCALING_FACTOR

    # ── Build state dict from reference fixture (gate weight/bias/rmsnorm_gamma all from state dict) ──
    bdw = BlitzDecodeWeights(device)
    layer_key = f"model.layers.{ROUTED_EXPERT_LAYER_IDX}"
    state_dict = get_reference_model_state_dict(
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=num_experts,
        include_global=False,
    )

    # Create input tensor; gate weight/bias/rmsnorm_gamma are always read from state dict
    torch.manual_seed(RoutedExpert.SEED)
    torch_input = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gate_mm_weights = None
    torch_bias = None

    # Define core grid for compute (first column, 8 cores)
    compute_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, num_cores - 1))])

    # Input tensor: sharded on sender core OUTSIDE the compute grid
    device_grid_size = device.compute_with_storage_grid_size()
    input_core = ttnn.CoreCoord(device_grid_size.x - 1, RoutedExpert.INPUT_CORE_Y)
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
        tile=Tiles.TILE_1x32,
        **from_torch_kwargs,
    )

    # ── RMSNorm gamma weights [1, K] from state dict (post_attention_layernorm) ──
    ffn_norm_key = f"{layer_key}.post_attention_layernorm.weight"
    torch_rmsnorm_gamma = state_dict[ffn_norm_key].reshape(1, K).to(torch.bfloat16).float()
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
        tile=Tiles.TILE_1x32,
        **from_torch_kwargs,
    )
    # Get optimal DRAM bank cores for DRAM streaming matmul + SiLU
    gate_proj_noc = ttnn.NOC.NOC_0
    gate_proj_worker_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(gate_proj_noc)
    gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in gate_proj_worker_cores])
    num_gate_proj_cores = len(gate_proj_worker_cores)

    # Build attention-side overlapped tensors from state dict via prepare_weights.
    attn = prepare_attention_weights(
        bdw, state_dict, layer_idx=ROUTED_EXPERT_LAYER_IDX, is_moe=True, move_to_device=True
    )
    ttnn_gate_mm_weights = attn.gate_mm
    ttnn_rmsnorm_gamma = attn.ffn_norm
    compute_core_grid = ttnn_gate_mm_weights.core_range_set

    # Routing tensors (only when enable_routing=True)
    if not enable_routing:
        ttnn_gate_mm_weights = None
    ttnn_gate_bias = None
    ttnn_gate_indices = None
    gate_output_scores_tensor = None
    gate_output_indices_tensor = None

    # ── Compute dimensions for expert DRAM matmul (down_proj padding for output extraction) ──
    num_banks = device.dram_grid_size().x
    tile_w = RoutedExpert.TILE_W
    down_proj_K = gate_proj_N
    down_proj_N = K
    down_proj_N_padded = ((down_proj_N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    per_core_down_proj_N = down_proj_N_padded // num_banks
    # Expected HF expert shapes: gate/up (GATE_PROJ_N, K), down (K, GATE_PROJ_N)
    expected_gate_up = (RoutedExpert.GATE_PROJ_N, RoutedExpert.K)
    expected_down = (RoutedExpert.K, RoutedExpert.GATE_PROJ_N)
    expert0_prefix = f"{layer_key}.mlp.experts.0."
    assert state_dict[f"{expert0_prefix}gate_proj.weight"].shape == expected_gate_up
    assert state_dict[f"{expert0_prefix}up_proj.weight"].shape == expected_gate_up
    assert state_dict[f"{expert0_prefix}down_proj.weight"].shape == expected_down
    if enable_routing:
        gate_key = f"{layer_key}.mlp.gate.weight"
        bias_key = f"{layer_key}.mlp.gate.e_score_correction_bias"
        torch_gate_mm_weights = state_dict[gate_key].T.contiguous()
        torch_bias = state_dict[bias_key].reshape(1, 8, 32).contiguous().to(torch.bfloat16)
    expert_weights_dict = {}
    up_proj_weights_dict = {}
    down_proj_weights_dict = {}
    for e in range(num_experts):
        # HF layout: gate/up (out,in)=(2048,7168), down (7168,2048); golden wants (1,1,K,N)
        w_g = state_dict[f"{layer_key}.mlp.experts.{e}.gate_proj.weight"].T.contiguous()
        expert_weights_dict[e] = w_g.reshape(1, 1, gate_proj_K, gate_proj_N)
        w_u = state_dict[f"{layer_key}.mlp.experts.{e}.up_proj.weight"].T.contiguous()
        up_proj_weights_dict[e] = w_u.reshape(1, 1, gate_proj_K, gate_proj_N)
        w_d = state_dict[f"{layer_key}.mlp.experts.{e}.down_proj.weight"].T.contiguous()
        down_proj_weights_dict[e] = w_d.reshape(1, 1, down_proj_K, down_proj_N)

    routed_weights = prepare_routed_expert_weights(
        bdw,
        state_dict,
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        is_moe=True,
        num_routed_experts=num_experts,
        move_to_device=True,
    )
    gate_proj_expert_tensors = routed_weights.routed_gate_proj
    up_proj_expert_tensors = routed_weights.routed_up_proj
    down_proj_expert_tensors = routed_weights.routed_down_proj
    gate_proj_weights = gate_proj_expert_tensors[0]
    up_proj_weights = up_proj_expert_tensors[0]
    down_proj_weights = down_proj_expert_tensors[0]

    if enable_routing:
        # Gate bias/indices from prepare_weights helpers.
        raw_bias = state_dict[f"model.layers.{ROUTED_EXPERT_LAYER_IDX}.mlp.gate.e_score_correction_bias"]
        ttnn_gate_bias = create_gate_bias_tensor(raw_bias, device, input_core_grid, mesh_mapper=mesh_mapper)
        ttnn_gate_indices = create_gate_indices_tensor(device, input_core_grid, mesh_mapper=mesh_mapper)
        # Gate output buffers (scores and indices on sender core)
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
        gate_output_indices_tensor = ttnn.from_torch(
            torch.zeros((1, 16), dtype=torch.uint16),
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=gate_output_mem_config,
            tile=tile_1x16,
            **from_torch_kwargs,
        )

    # Final output tensor
    final_output_width_per_core = RoutedExpert.FINAL_OUTPUT_WIDTH_PER_CORE
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
            tile=Tiles.TILE_1x32,
            **from_torch_kwargs,
        )

    return RoutedExpertTensors(
        ttnn_residual_mcast_src=ttnn_residual_mcast_src,
        ttnn_rmsnorm_gamma=ttnn_rmsnorm_gamma,
        torch_rmsnorm_gamma=torch_rmsnorm_gamma,
        ttnn_gate_mm_weights=ttnn_gate_mm_weights,
        ttnn_gate_bias=ttnn_gate_bias,
        ttnn_gate_indices=ttnn_gate_indices,
        gate_output_scores_tensor=gate_output_scores_tensor,
        gate_output_indices_tensor=gate_output_indices_tensor,
        gate_proj_weights=gate_proj_weights,
        up_proj_weights=up_proj_weights,
        down_proj_weights=down_proj_weights,
        final_output_tensor=final_output_tensor,
        gate_proj_expert_tensors=gate_proj_expert_tensors,
        up_proj_expert_tensors=up_proj_expert_tensors,
        down_proj_expert_tensors=down_proj_expert_tensors,
        torch_input=torch_input,
        torch_gate_mm_weights=torch_gate_mm_weights,
        torch_bias=torch_bias,
        expert_weights_dict=expert_weights_dict,
        up_proj_weights_dict=up_proj_weights_dict,
        down_proj_weights_dict=down_proj_weights_dict,
        gate_eps=gate_eps,
        gate_scaling_factor=gate_scaling_factor,
        num_gate_proj_cores=num_gate_proj_cores,
        final_output_width_per_core=final_output_width_per_core,
        per_core_down_proj_N=per_core_down_proj_N,
        final_output_total_width=final_output_total_width,
        final_output_mem_config=final_output_mem_config,
    )


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
@pytest.mark.parametrize("reconfig_moe_cbs", [True, False])
@pytest.mark.parametrize("noc_mode", [ttnn.NOC_MODE.DM_DYNAMIC_NOC])
@pytest.mark.requires_grid_size((13, 10))
def test_moe_fused(device, use_hardcoded_expert_index, reconfig_moe_cbs, noc_mode, get_reference_model_state_dict):
    """Test fused MoE: run both routed expert and shared expert, validate combined output."""

    M = RoutedExpert.M
    K = RoutedExpert.K

    logger.info(f"Testing fused MoE: K={K}, use_hardcoded_expert_index={use_hardcoded_expert_index}")

    # ── Phase 1: Fused routed expert + shared gate/up matmul ──
    logger.info("Phase 1: Running fused routed expert + shared gate/up matmul...")
    r = create_routed_expert_tensors(
        device, use_hardcoded_expert_index, get_reference_model_state_dict=get_reference_model_state_dict
    )
    sender_core = r.ttnn_residual_mcast_src.memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
    s = create_shared_expert_tensors(
        device, M, K, mcast_grid, get_reference_model_state_dict=get_reference_model_state_dict
    )

    # ── Create sdpa_kv_cache_buffer for CB memory overlap ──
    kv_cache_shard_height = SDPA.KV_CACHE_SHARD_HEIGHT
    kvpe_dim = SDPA.KVPE_DIM
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
    sdpa_out_interm_shard_height = SDPA.OUT_INTERM_SHARD_HEIGHT
    sdpa_out_interm_shard_width = SDPA.OUT_INTERM_SHARD_WIDTH
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
        tile=Tiles.TILE_8x32,
    )

    moe_semaphores = MoeOp.create_semaphores(device)
    num_iterations = TestConfig.NUM_ITERATIONS
    ttnn_result_scores, ttnn_result_indices, ttnn_result_final = MoeOp.op(
        r.ttnn_residual_mcast_src,
        r.ttnn_gate_mm_weights,
        r.ttnn_gate_bias,
        r.ttnn_gate_indices,
        r.gate_output_scores_tensor,
        r.gate_output_indices_tensor,
        r.gate_proj_weights,
        r.up_proj_weights,
        r.down_proj_weights,
        r.final_output_tensor,
        r.ttnn_rmsnorm_gamma,
        # Shared expert tensors
        shared_gate_weights_overlapped=s.shared_gate_weights_overlapped,
        shared_up_weights_overlapped=s.shared_up_weights_overlapped,
        shared_down_weights_tensor=s.ttnn_down_weights,
        shared_k_parallel=s.k_parallel,
        shared_n_parallel=s.n_parallel,
        use_hardcoded_expert_index=use_hardcoded_expert_index,
        sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
        sdpa_out_interm_buffer=sdpa_out_interm_buffer,
        num_iterations=num_iterations,
        reconfig_moe_cbs=reconfig_moe_cbs,
        semaphores=moe_semaphores,
        noc_mode=noc_mode,
    )
    ttnn.synchronize_device(device)
    logger.info(f"Fused routed+shared gate/up: {num_iterations} iterations completed (reconfig={reconfig_moe_cbs})")

    # Read back routed expert results
    output_scores_torch = ttnn.to_torch(ttnn_result_scores)
    output_indices_torch = ttnn.to_torch(ttnn_result_indices).to(torch.int64)
    output_final_torch = ttnn.to_torch(ttnn_result_final)

    output_final_valid = extract_routed_expert_output(
        output_final_torch,
        r.num_gate_proj_cores,
        r.final_output_width_per_core,
        r.per_core_down_proj_N,
    )

    # Compute fused MoE golden (routed + shared expert + eltwise add)
    torch_expected_scores, torch_expected_indices, torch_expected_final = MoeOp.golden(
        r.torch_input,
        shared_gate_weights=s.torch_gate_weights,
        shared_up_weights=s.torch_up_weights,
        shared_down_weights=s.torch_down_weights,
        gate_proj_weights_dict=r.expert_weights_dict,
        up_proj_weights_dict=r.up_proj_weights_dict,
        down_proj_weights_dict=r.down_proj_weights_dict,
        rmsnorm_gamma=r.torch_rmsnorm_gamma,
        rmsnorm_epsilon=1e-6,
        routing_weights_tensor=r.torch_gate_mm_weights,
        bias_tensor=r.torch_bias,
        eps=r.gate_eps,
        scaling_factor=r.gate_scaling_factor,
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
        sorted_output_scores, sorted_expected_scores, atol=2e-2, rtol=1e-4
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
@pytest.mark.parametrize("reconfig_moe_cbs", [True, False])
@pytest.mark.parametrize("noc_mode", [ttnn.NOC_MODE.DM_DYNAMIC_NOC])
@pytest.mark.requires_grid_size((13, 10))
def test_moe_fused_with_reduce(
    bh_2d_mesh_device, use_hardcoded_expert_index, reconfig_moe_cbs, noc_mode, get_reference_model_state_dict
):
    """
    Test fused MoE with reduce_to_one on 4x2 mesh.

    Each of 8 devices runs the full fused MoE (routed + shared expert),
    then results are reduced (summed) across all devices to ROOT1.
    """
    num_devices = TestConfig.NUM_DEVICES_4x2
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip(
            f"Test requires {num_devices} devices, mesh has "
            f"{bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1]}"
        )

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    logger.info(f"Created submesh with shape: {submesh.shape}")

    M = RoutedExpert.M
    K = RoutedExpert.K

    logger.info(f"Testing fused MoE with reduce: K={K}")

    # ── Create MoE tensors (replicated across mesh) ──
    mesh_mapper = ttnn.ReplicateTensorToMesh(submesh)
    r = create_routed_expert_tensors(
        submesh,
        use_hardcoded_expert_index,
        mesh_mapper=mesh_mapper,
        create_final_output=False,
        get_reference_model_state_dict=get_reference_model_state_dict,
    )
    sender_core = r.ttnn_residual_mcast_src.memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
    s = create_shared_expert_tensors(
        submesh,
        M,
        K,
        mcast_grid,
        mesh_mapper=mesh_mapper,
        get_reference_model_state_dict=get_reference_model_state_dict,
    )

    # ── Create SDPA buffers for CB memory overlap (required by fused MoE) ──
    device_grid_size = submesh.compute_with_storage_grid_size()
    kv_cache_shard_height = SDPA.KV_CACHE_SHARD_HEIGHT
    kvpe_dim = SDPA.KVPE_DIM
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

    sdpa_out_interm_shard_height = SDPA.OUT_INTERM_SHARD_HEIGHT
    sdpa_out_interm_shard_width = SDPA.OUT_INTERM_SHARD_WIDTH
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
        device=submesh,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            sdpa_out_interm_shard_spec,
        ),
        tile=Tiles.TILE_8x32,
    )

    # ── ReduceToOne tensors and semaphores ──
    root_coord = TestConfig.REDUCE_ROOT_COORD

    # Reduce mesh mapper (2D shard across 4x2 mesh)
    reduce_mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], submesh.shape)
    reduce_mesh_mapper = ttnn.create_mesh_mapper(submesh, reduce_mesh_mapper_config)

    tile_1x32 = Tiles.TILE_1x32
    final_output_total_width = r.final_output_total_width
    final_output_mem_config = r.final_output_mem_config

    # 3 intermediate tensors for 3 reduction rounds (same shape as final_output)
    intermediate_tensors = []
    for _ in range(TestConfig.REDUCE_NUM_ROUNDS):
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
    reduce_semaphores = [
        ttnn.create_global_semaphore(submesh, available_cores, 0) for _ in range(TestConfig.REDUCE_NUM_SEMAPHORES)
    ]
    ttnn.synchronize_device(submesh)
    logger.info("Created 4 global semaphores for reduce synchronization")

    # ── Run fused MoE op with reduce (looping inside kernel) ──
    moe_semaphores = MoeOp.create_semaphores(submesh)
    num_iterations = TestConfig.NUM_ITERATIONS
    ttnn_result_scores, ttnn_result_indices, ttnn_result_reduce = MoeOp.op(
        r.ttnn_residual_mcast_src,
        r.ttnn_gate_mm_weights,
        r.ttnn_gate_bias,
        r.ttnn_gate_indices,
        r.gate_output_scores_tensor,
        r.gate_output_indices_tensor,
        r.gate_proj_weights,
        r.up_proj_weights,
        r.down_proj_weights,
        r.final_output_tensor,
        r.ttnn_rmsnorm_gamma,
        # Shared expert tensors
        shared_gate_weights_overlapped=s.shared_gate_weights_overlapped,
        shared_up_weights_overlapped=s.shared_up_weights_overlapped,
        shared_down_weights_tensor=s.ttnn_down_weights,
        shared_k_parallel=s.k_parallel,
        shared_n_parallel=s.n_parallel,
        use_hardcoded_expert_index=use_hardcoded_expert_index,
        sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
        sdpa_out_interm_buffer=sdpa_out_interm_buffer,
        num_iterations=num_iterations,
        reconfig_moe_cbs=reconfig_moe_cbs,
        # ReduceToOne parameters
        reduce_intermediate_tensors=intermediate_tensors,
        reduce_output_tensor=reduce_output_tensor,
        reduce_semaphores=reduce_semaphores,
        reduce_root_coord=ttnn.MeshCoordinate(root_coord),
        semaphores=moe_semaphores,
        noc_mode=noc_mode,
    )
    ttnn.synchronize_device(submesh)
    logger.info(f"Fused MoE with reduce: {num_iterations} iterations completed (reconfig={reconfig_moe_cbs})")

    # ── Verify results ──
    # Read gate scores/indices from device (needed for per-device golden)
    device_gate_indices = ttnn.to_torch(ttnn_result_indices, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))
    device_gate_scores = ttnn.to_torch(ttnn_result_scores, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))

    # Compute expected output for each device, then sum
    # Each device uses a different hardcoded expert index (chip_id)
    # and a different TP shard of shared expert weights
    K_down = s.K_down
    expected_final_outputs = []
    for device_idx in range(num_devices):
        chip_id = device_idx

        if use_hardcoded_expert_index:
            actual_expert_idx = chip_id
            actual_expert_scale = device_gate_scores[0].flatten()[chip_id].float()
        else:
            actual_expert_idx = int(device_gate_indices[0].flatten()[chip_id].item())
            actual_expert_scale = device_gate_scores[0].flatten()[chip_id].float()

        shared_gate_shard = s.torch_gate_weights[:, device_idx * K_down : (device_idx + 1) * K_down]
        shared_up_shard = s.torch_up_weights[:, device_idx * K_down : (device_idx + 1) * K_down]
        shared_down_shard = s.torch_down_weights[device_idx * K_down : (device_idx + 1) * K_down, :]

        _, _, torch_expected_final = MoeOp.golden(
            r.torch_input,
            shared_gate_weights=shared_gate_shard,
            shared_up_weights=shared_up_shard,
            shared_down_weights=shared_down_shard,
            gate_proj_weights_dict=r.expert_weights_dict,
            up_proj_weights_dict=r.up_proj_weights_dict,
            down_proj_weights_dict=r.down_proj_weights_dict,
            rmsnorm_gamma=r.torch_rmsnorm_gamma,
            rmsnorm_epsilon=1e-6,
            routing_weights_tensor=r.torch_gate_mm_weights,
            bias_tensor=r.torch_bias,
            eps=r.gate_eps,
            scaling_factor=r.gate_scaling_factor,
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
        r.num_gate_proj_cores,
        r.final_output_width_per_core,
        r.per_core_down_proj_N,
    )

    # Verify reduce output
    passing, pcc_output = comp_pcc(expected_reduce_output.flatten(), reduce_output_valid.flatten(), 0.97)
    logger.info(f"Reduce output PCC: {pcc_output}")
    assert passing, f"Reduce output PCC check failed: {pcc_output}"

    logger.info("Fused MoE with reduce test PASSED!")


# ============================================================================
# Test: Fused MoE with enable_routing=False (dense MLP mode)
# ============================================================================
@pytest.mark.parametrize("reconfig_moe_cbs", [True, False])
@pytest.mark.parametrize("noc_mode", [ttnn.NOC_MODE.DM_DYNAMIC_NOC])
@pytest.mark.requires_grid_size((13, 10))
def test_mlp(device, reconfig_moe_cbs, noc_mode, get_reference_model_state_dict):
    """Test MoeOp with enable_routing=False: same as MLP, no routing logic."""

    M = RoutedExpert.M
    K = RoutedExpert.K

    logger.info(f"Testing MoeOp with enable_routing=False: K={K}")

    # ── Create MLP tensors (no routing) ──
    r = create_routed_expert_tensors(
        device, enable_routing=False, get_reference_model_state_dict=get_reference_model_state_dict
    )
    sender_core = r.ttnn_residual_mcast_src.memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
    s = create_shared_expert_tensors(
        device, M, K, mcast_grid, get_reference_model_state_dict=get_reference_model_state_dict
    )

    # ── Create SDPA buffers for CB memory overlap ──
    kv_cache_shard_height = SDPA.KV_CACHE_SHARD_HEIGHT
    kvpe_dim = SDPA.KVPE_DIM
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
    sdpa_out_interm_shard_height = SDPA.OUT_INTERM_SHARD_HEIGHT
    sdpa_out_interm_shard_width = SDPA.OUT_INTERM_SHARD_WIDTH
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
        tile=Tiles.TILE_8x32,
    )

    # ── Run MoeOp with enable_routing=False ──
    moe_semaphores = MoeOp.create_semaphores(device)
    num_iterations = TestConfig.NUM_ITERATIONS
    ttnn_result_final = MoeOp.op(
        r.ttnn_residual_mcast_src,
        # No routing tensors
        gate_proj_weights_tensor=r.gate_proj_weights,
        up_proj_weights_tensor=r.up_proj_weights,
        down_proj_weights_tensor=r.down_proj_weights,
        final_output_tensor=r.final_output_tensor,
        rmsnorm_gamma_tensor=r.ttnn_rmsnorm_gamma,
        shared_gate_weights_overlapped=s.shared_gate_weights_overlapped,
        shared_up_weights_overlapped=s.shared_up_weights_overlapped,
        shared_down_weights_tensor=s.ttnn_down_weights,
        shared_k_parallel=s.k_parallel,
        shared_n_parallel=s.n_parallel,
        enable_routing=False,
        sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
        sdpa_out_interm_buffer=sdpa_out_interm_buffer,
        num_iterations=num_iterations,
        reconfig_moe_cbs=reconfig_moe_cbs,
        semaphores=moe_semaphores,
        noc_mode=noc_mode,
    )
    ttnn.synchronize_device(device)
    logger.info(f"MoeOp no-routing: {num_iterations} iterations completed (reconfig={reconfig_moe_cbs})")

    # ── Read back and validate ──
    output_final_torch = ttnn.to_torch(ttnn_result_final)

    output_final_valid = extract_routed_expert_output(
        output_final_torch,
        r.num_gate_proj_cores,
        r.final_output_width_per_core,
        r.per_core_down_proj_N,
    )

    # Compute golden (no routing, no expert scale)
    _, _, torch_expected = MoeOp.golden(
        r.torch_input,
        shared_gate_weights=s.torch_gate_weights,
        shared_up_weights=s.torch_up_weights,
        shared_down_weights=s.torch_down_weights,
        gate_proj_weights_dict=r.expert_weights_dict,
        up_proj_weights_dict=r.up_proj_weights_dict,
        down_proj_weights_dict=r.down_proj_weights_dict,
        rmsnorm_gamma=r.torch_rmsnorm_gamma,
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
@pytest.mark.parametrize("reconfig_moe_cbs", [True, False])
@pytest.mark.parametrize("noc_mode", [ttnn.NOC_MODE.DM_DYNAMIC_NOC])
@pytest.mark.requires_grid_size((13, 10))
def test_mlp_with_reduce(bh_2d_mesh_device, reconfig_moe_cbs, noc_mode, get_reference_model_state_dict):
    """
    Test MoeOp with enable_routing=False and reduce_to_one on 4x2 mesh.

    Each of 8 devices runs the full fused MLP (dense MLP + shared expert),
    then results are reduced (summed) across all devices to ROOT1.
    """

    num_devices = TestConfig.NUM_DEVICES_4x2
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip(
            f"Test requires {num_devices} devices, mesh has "
            f"{bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1]}"
        )

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((4, 2)))
    logger.info(f"Created submesh with shape: {submesh.shape}")

    M = RoutedExpert.M
    K = RoutedExpert.K

    logger.info(f"Testing MoeOp no-routing with reduce: K={K}")

    # ── Create MLP tensors (replicated across mesh) ──
    mesh_mapper = ttnn.ReplicateTensorToMesh(submesh)
    r = create_routed_expert_tensors(
        submesh,
        mesh_mapper=mesh_mapper,
        create_final_output=False,
        enable_routing=False,
        get_reference_model_state_dict=get_reference_model_state_dict,
    )
    sender_core = r.ttnn_residual_mcast_src.memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
    s = create_shared_expert_tensors(
        submesh,
        M,
        K,
        mcast_grid,
        mesh_mapper=mesh_mapper,
        get_reference_model_state_dict=get_reference_model_state_dict,
    )

    # ── Create SDPA buffers for CB memory overlap ──
    device_grid_size = submesh.compute_with_storage_grid_size()
    kv_cache_shard_height = SDPA.KV_CACHE_SHARD_HEIGHT
    kvpe_dim = SDPA.KVPE_DIM
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

    sdpa_out_interm_shard_height = SDPA.OUT_INTERM_SHARD_HEIGHT
    sdpa_out_interm_shard_width = SDPA.OUT_INTERM_SHARD_WIDTH
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
        device=submesh,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            sdpa_out_interm_shard_spec,
        ),
        tile=Tiles.TILE_8x32,
    )

    # ── ReduceToOne tensors and semaphores ──
    root_coord = TestConfig.REDUCE_ROOT_COORD

    reduce_mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], submesh.shape)
    reduce_mesh_mapper = ttnn.create_mesh_mapper(submesh, reduce_mesh_mapper_config)

    tile_1x32 = Tiles.TILE_1x32
    final_output_total_width = r.final_output_total_width
    final_output_mem_config = r.final_output_mem_config

    # 3 intermediate tensors for 3 reduction rounds
    intermediate_tensors = []
    for _ in range(TestConfig.REDUCE_NUM_ROUNDS):
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
    reduce_semaphores = [
        ttnn.create_global_semaphore(submesh, available_cores, 0) for _ in range(TestConfig.REDUCE_NUM_SEMAPHORES)
    ]
    ttnn.synchronize_device(submesh)
    logger.info("Created 4 global semaphores for reduce synchronization")

    # ── Run MoeOp with enable_routing=False and reduce ──
    moe_semaphores = MoeOp.create_semaphores(submesh)
    num_iterations = TestConfig.NUM_ITERATIONS
    ttnn_result_reduce = MoeOp.op(
        r.ttnn_residual_mcast_src,
        # No routing tensors
        gate_proj_weights_tensor=r.gate_proj_weights,
        up_proj_weights_tensor=r.up_proj_weights,
        down_proj_weights_tensor=r.down_proj_weights,
        final_output_tensor=r.final_output_tensor,
        rmsnorm_gamma_tensor=r.ttnn_rmsnorm_gamma,
        shared_gate_weights_overlapped=s.shared_gate_weights_overlapped,
        shared_up_weights_overlapped=s.shared_up_weights_overlapped,
        shared_down_weights_tensor=s.ttnn_down_weights,
        shared_k_parallel=s.k_parallel,
        shared_n_parallel=s.n_parallel,
        enable_routing=False,
        sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
        sdpa_out_interm_buffer=sdpa_out_interm_buffer,
        num_iterations=num_iterations,
        reconfig_moe_cbs=reconfig_moe_cbs,
        reduce_intermediate_tensors=intermediate_tensors,
        reduce_output_tensor=reduce_output_tensor,
        reduce_semaphores=reduce_semaphores,
        reduce_root_coord=ttnn.MeshCoordinate(root_coord),
        semaphores=moe_semaphores,
        noc_mode=noc_mode,
    )
    ttnn.synchronize_device(submesh)
    logger.info(f"MoeOp no-routing with reduce: {num_iterations} iterations completed (reconfig={reconfig_moe_cbs})")

    # ── Verify results ──
    # Compute per-device golden with per-device TP shards of shared expert weights
    K_down = s.K_down
    expected_final_outputs = []
    for device_idx in range(num_devices):
        shared_gate_shard = s.torch_gate_weights[:, device_idx * K_down : (device_idx + 1) * K_down]
        shared_up_shard = s.torch_up_weights[:, device_idx * K_down : (device_idx + 1) * K_down]
        shared_down_shard = s.torch_down_weights[device_idx * K_down : (device_idx + 1) * K_down, :]

        _, _, device_expected = MoeOp.golden(
            r.torch_input,
            shared_gate_weights=shared_gate_shard,
            shared_up_weights=shared_up_shard,
            shared_down_weights=shared_down_shard,
            gate_proj_weights_dict=r.expert_weights_dict,
            up_proj_weights_dict=r.up_proj_weights_dict,
            down_proj_weights_dict=r.down_proj_weights_dict,
            rmsnorm_gamma=r.torch_rmsnorm_gamma,
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
        r.num_gate_proj_cores,
        r.final_output_width_per_core,
        r.per_core_down_proj_N,
    )

    # Verify reduce output
    passing, pcc_output = comp_pcc(expected_reduce_output.flatten(), reduce_output_valid.flatten(), 0.97)
    logger.info(f"Reduce output PCC: {pcc_output}")
    assert passing, f"Reduce output PCC check failed: {pcc_output}"

    logger.info("MoeOp no-routing with reduce test PASSED!")
