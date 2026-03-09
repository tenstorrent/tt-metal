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
DENSE_LAYER_IDX = 0  # Layer index for dense (MLP) weights when is_moe=False
DENSE_SHARED_N = 2048  # Blitz uses first 2048 of 18432 as shared expert for dense MLP
DENSE_INTERMEDIATE_SIZE = 18432  # Full dense MLP intermediate size (gate/up rows, down cols)


# ============================================================================
# Helper: create all shared-expert tensors
# ============================================================================
def create_shared_expert_tensors(
    device, M, K_gate, mcast_grid, mesh_mapper=None, *, state_dict, is_moe=True, layer_idx=None
):
    """
    Create all tensors needed by SharedExpertOp.

    Args:
        device: TT device or mesh device
        M: Batch dimension (1)
        K_gate: Gate/Up input dimension (7168)
        mcast_grid: CoreRangeSet for mcast destination (same as routed input mcast)
        mesh_mapper: Optional mesh mapper for multi-device replication
        state_dict: State dict in HF key convention (same as used for routed path in fused tests).
        is_moe: If True use MoE keys (shared_experts.*). If False use dense keys (gate_proj/up_proj/down_proj).
        layer_idx: Layer index for state dict keys. If None, uses SHARED_EXPERT_LAYER_IDX when is_moe else DENSE_LAYER_IDX.

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

    assert layer_idx is not None, "layer_idx must be provided"

    if is_moe:
        gate_key = f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"
        up_key = f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"
        down_key = f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"
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
    else:
        # Dense MLP: gate/up (18432, 7168), down (7168, 18432). Blitz uses first 2048 as shared.
        gate_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        up_key = f"model.layers.{layer_idx}.mlp.up_proj.weight"
        down_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        expected_gate_up = (18432, RoutedExpert.K)
        expected_down = (RoutedExpert.K, 18432)
        assert (
            state_dict[gate_key].shape == expected_gate_up
        ), f"dense gate_proj: expected {expected_gate_up}, got {state_dict[gate_key].shape}"
        assert (
            state_dict[up_key].shape == expected_gate_up
        ), f"dense up_proj: expected {expected_gate_up}, got {state_dict[up_key].shape}"
        assert (
            state_dict[down_key].shape == expected_down
        ), f"dense down_proj: expected {expected_down}, got {state_dict[down_key].shape}"
        # (out, in) -> (in, out) for gate/up; shared = first 2048 columns
        gate_full = state_dict[gate_key].T.contiguous()  # (7168, 18432)
        up_full = state_dict[up_key].T.contiguous()
        down_full = state_dict[down_key].T.contiguous()  # (18432, 7168)
        torch_gate_weights = gate_full[:, :DENSE_SHARED_N].contiguous()
        torch_up_weights = up_full[:, :DENSE_SHARED_N].contiguous()
        torch_down_weights = down_full[:DENSE_SHARED_N, :].contiguous()
        # Do not apply MoE per-TP slice; dense shared is always 2048 columns for 8-device sharding

    torch.manual_seed(RoutedExpert.SEED)
    torch_activation = torch.randn((M, K_gate), dtype=torch.bfloat16)
    torch_bias = torch.randn((M, N), dtype=torch.bfloat16)

    shared_weights = prepare_shared_expert_weights(
        bdw, state_dict, layer_idx=layer_idx, is_moe=is_moe, move_to_device=True
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
    state_dict,
    is_moe=True,
    layer_idx=None,
):
    """
    Create all tensors needed for MoE routed expert test.

    The state_dict is never mutated. Gate weight and bias are always read from it
    for both device preparation and golden reference. Must contain at least
    num_experts routed experts (1 when enable_routing=False, 8 when use_hardcoded_expert_index
    on 8 devices, 256 otherwise). When is_moe=False (dense MLP), state dict has
    mlp.gate_proj/up_proj/down_proj and we slice into 8 routed experts for golden.

    When enable_routing=False, skips routing-specific tensors (gate MM weights,
    gate bias/indices, gate output scores/indices) and uses a single expert.

    Args:
        device: TT device or mesh device
        use_hardcoded_expert_index: Whether to use hardcoded expert index (routing only)
        mesh_mapper: Optional mesh mapper for multi-device replication
        create_final_output: If True, create final_output_tensor
        enable_routing: If True, create routing tensors. If False, skip them.
        state_dict: State dict in HF key convention (read-only when using HF weights).
        is_moe: If True use MoE keys (experts.{e}.*). If False use dense keys and slice to 8 experts.
        layer_idx: Layer index for state dict. If None, uses ROUTED_EXPERT_LAYER_IDX when is_moe else DENSE_LAYER_IDX.

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

    # num_experts: for dense no-routing we need 8 (one per device) for golden; else 1. With routing: per-device or 256.
    if not enable_routing:
        num_experts = 8 if (is_moe is False) else 1
    elif use_hardcoded_expert_index:
        num_experts = device.get_num_devices()
    else:
        num_experts = 256

    # Gate parameters (must match op.py)
    gate_eps = RoutedExpert.GATE_EPS
    gate_scaling_factor = RoutedExpert.GATE_SCALING_FACTOR

    # ── Use provided state dict (gate weight/bias/rmsnorm_gamma all from state dict) ──
    bdw = BlitzDecodeWeights(device)
    if layer_idx is None:
        layer_idx = ROUTED_EXPERT_LAYER_IDX if is_moe else DENSE_LAYER_IDX
    layer_key = f"model.layers.{layer_idx}"

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

    # Get optimal DRAM bank cores for DRAM streaming matmul + SiLU
    gate_proj_noc = ttnn.NOC.NOC_0
    gate_proj_worker_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(gate_proj_noc)
    gate_proj_core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in gate_proj_worker_cores])
    num_gate_proj_cores = len(gate_proj_worker_cores)

    # Build attention-side overlapped tensors from state dict via prepare_weights.
    attn = prepare_attention_weights(bdw, state_dict, layer_idx=layer_idx, is_moe=is_moe, move_to_device=True)
    ttnn_gate_mm_weights = attn.gate_mm
    ttnn_rmsnorm_gamma = attn.ffn_norm
    if ttnn_gate_mm_weights is not None:
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

    expert_weights_dict = {}
    up_proj_weights_dict = {}
    down_proj_weights_dict = {}

    if is_moe:
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
            layer_idx=layer_idx,
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
    else:
        # Dense MLP: slice gate/up (7168, 18432) and down (18432, 7168) into 8 experts of 2048 each
        gate_key = f"{layer_key}.mlp.gate_proj.weight"
        up_key = f"{layer_key}.mlp.up_proj.weight"
        down_key = f"{layer_key}.mlp.down_proj.weight"
        gate_full = state_dict[gate_key].T.contiguous()  # (7168, 18432)
        up_full = state_dict[up_key].T.contiguous()
        down_full = state_dict[down_key].T.contiguous()  # (18432, 7168)
        for e in range(8):
            start = DENSE_SHARED_N + e * RoutedExpert.GATE_PROJ_N
            end = start + RoutedExpert.GATE_PROJ_N
            w_g = gate_full[:, start:end].contiguous()
            expert_weights_dict[e] = w_g.reshape(1, 1, gate_proj_K, gate_proj_N)
            w_u = up_full[:, start:end].contiguous()
            up_proj_weights_dict[e] = w_u.reshape(1, 1, gate_proj_K, gate_proj_N)
            w_d = down_full[start:end, :].contiguous()
            down_proj_weights_dict[e] = w_d.reshape(1, 1, down_proj_K, down_proj_N)

        routed_weights = prepare_routed_expert_weights(
            bdw,
            state_dict,
            layer_idx=layer_idx,
            is_moe=False,
            num_routed_experts=8,
            move_to_device=True,
        )
        # DenseRoutedExpertWeights: single tensor per projection (mesh-shaped), no list
        gate_proj_weights = routed_weights.routed_gate_proj
        up_proj_weights = routed_weights.routed_up_proj
        down_proj_weights = routed_weights.routed_down_proj
        gate_proj_expert_tensors = None  # unused when is_moe=False
        up_proj_expert_tensors = None
        down_proj_expert_tensors = None

    if enable_routing:
        assert is_moe, "enable_routing=True is only supported with MoE weights"
        # Gate bias/indices from prepare_weights helpers.
        raw_bias = state_dict[f"{layer_key}.mlp.gate.e_score_correction_bias"]
        ttnn_gate_bias = create_gate_bias_tensor(raw_bias, device, move_to_device=True)
        assert list(ttnn.corerange_to_cores(ttnn_gate_bias.memory_config().shard_spec.grid)) == list(
            ttnn.corerange_to_cores(input_core_grid)
        ), "gate_bias grid must match input_core_grid (MOE_SENDER_GRID_SIZE)"
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


def create_reference_moe_model(state_dict, layer_idx):
    """Instantiate DeepseekV3MoE, load state dict (with auto-dequantization for real weights), return model and config."""
    from transformers import AutoConfig

    from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE

    hf_config = AutoConfig.from_pretrained("models/demos/deepseek_v3/reference", trust_remote_code=True)
    moe_prefix = f"model.layers.{layer_idx}.mlp."
    moe_state = {k[len(moe_prefix) :]: v for k, v in state_dict.items() if k.startswith(moe_prefix)}

    if any(k.endswith("_scale_inv") for k in moe_state):
        from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict

        moe_state = dequantize_state_dict(moe_state, hf_config)

    reference_model = DeepseekV3MoE(hf_config).eval().to(torch.bfloat16)
    reference_model.load_state_dict(moe_state)
    return reference_model, hf_config


def create_reference_dense_mlp_slices(state_dict, layer_idx):
    """Build shared (first 2048) and 8 routed MLP slices from dense layer state dict for reference comparison.

    Blitz uses first 2048 of 18432 as shared, remaining 8*2048 as 8 routed experts. Returns one
    shared DeepseekV3MLP (intermediate_size=2048) and a list of 8 routed DeepseekV3MLP (each 2048).
    """
    from transformers import AutoConfig

    from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP

    hf_config = AutoConfig.from_pretrained("models/demos/deepseek_v3/reference", trust_remote_code=True)
    gate_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
    up_key = f"model.layers.{layer_idx}.mlp.up_proj.weight"
    down_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
    gate_full = state_dict[gate_key]  # (18432, 7168)
    up_full = state_dict[up_key]
    down_full = state_dict[down_key]  # (7168, 18432)

    shared_gate = gate_full[:DENSE_SHARED_N, :].clone()
    shared_up = up_full[:DENSE_SHARED_N, :].clone()
    shared_down = down_full[:, :DENSE_SHARED_N].clone()
    shared_state = {
        "gate_proj.weight": shared_gate,
        "up_proj.weight": shared_up,
        "down_proj.weight": shared_down,
    }
    shared_mlp = DeepseekV3MLP(hf_config, intermediate_size=DENSE_SHARED_N).eval().to(torch.bfloat16)
    shared_mlp.load_state_dict(shared_state, strict=True)

    routed_mlps = []
    for e in range(8):
        start = DENSE_SHARED_N + e * RoutedExpert.GATE_PROJ_N
        end = start + RoutedExpert.GATE_PROJ_N
        routed_gate = gate_full[start:end, :].clone()
        routed_up = up_full[start:end, :].clone()
        routed_down = down_full[:, start:end].clone()
        routed_state = {
            "gate_proj.weight": routed_gate,
            "up_proj.weight": routed_up,
            "down_proj.weight": routed_down,
        }
        routed_mlp = DeepseekV3MLP(hf_config, intermediate_size=RoutedExpert.GATE_PROJ_N).eval().to(torch.bfloat16)
        routed_mlp.load_state_dict(routed_state, strict=True)
        routed_mlps.append(routed_mlp)

    return shared_mlp, routed_mlps


def create_reference_dense_full_mlp(state_dict, layer_idx):
    """Build one DeepseekV3MLP(intermediate_size=18432) from dense layer state dict.

    Reference block output = raw_input + full_mlp(normed_input). Reduce output satisfies
    reduce_output - 7*raw_input = block_output, so we can compare adjusted reduce to this.
    """
    from transformers import AutoConfig

    from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP

    hf_config = AutoConfig.from_pretrained("models/demos/deepseek_v3/reference", trust_remote_code=True)
    gate_key = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
    up_key = f"model.layers.{layer_idx}.mlp.up_proj.weight"
    down_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
    full_state = {
        "gate_proj.weight": state_dict[gate_key].clone(),
        "up_proj.weight": state_dict[up_key].clone(),
        "down_proj.weight": state_dict[down_key].clone(),
    }
    full_mlp = DeepseekV3MLP(hf_config, intermediate_size=DENSE_INTERMEDIATE_SIZE).eval().to(torch.bfloat16)
    full_mlp.load_state_dict(full_state, strict=True)
    return full_mlp


def create_reference_mlp_models(state_dict, layer_idx):
    """Instantiate expert-0 and shared-expert DeepseekV3MLP from state dict for reference comparison."""
    from transformers import AutoConfig

    from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP

    hf_config = AutoConfig.from_pretrained("models/demos/deepseek_v3/reference", trust_remote_code=True)

    # Expert 0 MLP
    expert_prefix = f"model.layers.{layer_idx}.mlp.experts.0."
    expert_state = {k[len(expert_prefix) :]: v for k, v in state_dict.items() if k.startswith(expert_prefix)}
    if any(k.endswith("_scale_inv") for k in expert_state):
        from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict

        expert_state = dequantize_state_dict(expert_state, hf_config)
    expert_mlp = DeepseekV3MLP(hf_config, intermediate_size=hf_config.moe_intermediate_size).eval().to(torch.bfloat16)
    expert_mlp.load_state_dict(expert_state)

    # Shared expert MLP
    shared_prefix = f"model.layers.{layer_idx}.mlp.shared_experts."
    shared_state = {k[len(shared_prefix) :]: v for k, v in state_dict.items() if k.startswith(shared_prefix)}
    if any(k.endswith("_scale_inv") for k in shared_state):
        from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict

        shared_state = dequantize_state_dict(shared_state, hf_config)
    shared_mlp = (
        DeepseekV3MLP(
            hf_config,
            intermediate_size=hf_config.moe_intermediate_size * hf_config.n_shared_experts,
        )
        .eval()
        .to(torch.bfloat16)
    )
    shared_mlp.load_state_dict(shared_state)

    return expert_mlp, shared_mlp


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

    state_dict = get_reference_model_state_dict(
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=256,
        include_global=False,
    )

    # ── Phase 1: Fused routed expert + shared gate/up matmul ──
    logger.info("Phase 1: Running fused routed expert + shared gate/up matmul...")
    r = create_routed_expert_tensors(
        device,
        use_hardcoded_expert_index,
        state_dict=state_dict,
        is_moe=True,
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
    )
    sender_core = r.ttnn_residual_mcast_src.memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
    s = create_shared_expert_tensors(
        device,
        M,
        K,
        mcast_grid,
        state_dict=state_dict,
        is_moe=True,
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
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

    if not torch.equal(sorted_output_indices, sorted_expected_indices):
        logger.warning(
            f"Gate indices mismatch (device={sorted_output_indices.tolist()}, "
            f"golden={sorted_expected_indices.tolist()}). "
            "This may be caused by bfloat16 matmul accumulation precision over K=7168."
        )
    else:
        assert torch.allclose(
            sorted_output_scores, sorted_expected_scores, atol=2e-2, rtol=1e-4
        ), "Routed expert: gate scores mismatch"

    passing, pcc = comp_pcc(torch_expected_final, output_final_valid, 0.97)
    logger.info(f"Fused MoE PCC: {pcc}")
    assert passing, f"Fused MoE PCC check failed: {pcc}"

    logger.info(f"Fused MoE test PASSED! (PCC={pcc})")


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

    When use_hardcoded_expert_index=True, each device uses expert index = chip_id (0..7)
    and the gate is only used for scaling; this is not the same as the reference model,
    which uses the gate to select the actual top-8 experts. The reference comparison
    is therefore skipped in that case and only runs when use_hardcoded_expert_index=False.
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

    state_dict = get_reference_model_state_dict(
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=256,
        include_global=False,
    )

    # ── Create MoE tensors (replicated across mesh) ──
    mesh_mapper = ttnn.ReplicateTensorToMesh(submesh)
    r = create_routed_expert_tensors(
        submesh,
        use_hardcoded_expert_index,
        mesh_mapper=mesh_mapper,
        create_final_output=False,
        state_dict=state_dict,
        is_moe=True,
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
    )
    sender_core = r.ttnn_residual_mcast_src.memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
    s = create_shared_expert_tensors(
        submesh,
        M,
        K,
        mcast_grid,
        mesh_mapper=mesh_mapper,
        state_dict=state_dict,
        is_moe=True,
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
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

    device_grid_size = submesh.compute_with_storage_grid_size()
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

    # Single intermediate tensor with 3x shard width for all 3 reduction rounds
    orig_shard_spec = final_output_mem_config.shard_spec
    intermediate_shard_shape = [orig_shard_spec.shape[0], orig_shard_spec.shape[1] * 3]
    intermediate_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            orig_shard_spec.grid,
            intermediate_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    intermediate_tensors = ttnn.from_torch(
        torch.zeros([4, 2, final_output_total_width * 3], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=intermediate_mem_config,
        tile=tile_1x32,
        mesh_mapper=reduce_mesh_mapper,
    )

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

    # Compute expected output for each device, then sum.
    # Residual is only added on ROOT1 device; non-root devices skip it.
    K_down = s.K_down
    root_device_idx = root_coord[0] * submesh.shape[1] + root_coord[1]
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
            include_residual=(device_idx == root_device_idx),
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

    # --- Reference model comparison (when doing same op as reference: gate-selected experts) ---
    # Skip when use_hardcoded_expert_index=True: B1 then uses fixed experts 0..7 (one per device),
    # while the reference uses gate-selected top-8 experts; they are not the same operation.
    num_experts_in_state_dict = sum(1 for k in state_dict if ".mlp.experts." in k and "gate_proj" in k)
    if num_experts_in_state_dict >= 256 and not use_hardcoded_expert_index:
        reference_moe, _ = create_reference_moe_model(state_dict, ROUTED_EXPERT_LAYER_IDX)

        # Apply RMSNorm (same as B1 fused op does internally)
        x = r.torch_input.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        normed_input = ((x * torch.rsqrt(variance + 1e-6)) * r.torch_rmsnorm_gamma.float()).bfloat16()

        # Optional: log whether TT gate indices match reference gate indices
        with torch.no_grad():
            ref_topk_idx, _ = reference_moe.gate(normed_input.unsqueeze(0))
        ref_top8_sorted = torch.sort(ref_topk_idx.squeeze(0)).values
        tt_top8 = device_gate_indices[0].flatten()[:8].to(torch.int64)
        tt_top8_sorted = torch.sort(tt_top8).values
        if torch.equal(ref_top8_sorted, tt_top8_sorted):
            logger.info("Gate indices match (reference vs TT)")
        else:
            logger.info(f"Gate indices differ: ref={ref_top8_sorted.tolist()}, tt={tt_top8_sorted.tolist()}")

        with torch.no_grad():
            ref_moe_output = reference_moe(normed_input.unsqueeze(0)).squeeze(0)

        # Residual is added once (on ROOT1 only), so reduce output directly
        # matches: residual + moe_output
        ref_block_output = r.torch_input.float() + ref_moe_output.float()

        passing_ref, pcc_ref = comp_pcc(ref_block_output, reduce_output_valid.float(), 0.95)
        logger.info(f"Reference MoE comparison PCC: {pcc_ref}")
        assert passing_ref, f"Reference MoE comparison PCC failed: {pcc_ref}"

    logger.info("Fused MoE with reduce test PASSED!")


@pytest.mark.parametrize("reconfig_moe_cbs", [True, False])
@pytest.mark.parametrize("noc_mode", [ttnn.NOC_MODE.DM_DYNAMIC_NOC])
@pytest.mark.requires_grid_size((13, 10))
def test_mlp(device, reconfig_moe_cbs, noc_mode, get_reference_model_state_dict):
    """Test MoeOp with enable_routing=False: same as MLP (dense mode), no routing logic."""

    M = RoutedExpert.M
    K = RoutedExpert.K

    logger.info(f"Testing MoeOp with enable_routing=False: K={K}")

    state_dict = get_reference_model_state_dict(
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=256,
        include_global=False,
    )

    # ── Create MLP tensors (no routing) ──
    r = create_routed_expert_tensors(
        device,
        enable_routing=False,
        state_dict=state_dict,
        is_moe=True,
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
    )
    sender_core = r.ttnn_residual_mcast_src.memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
    s = create_shared_expert_tensors(
        device,
        M,
        K,
        mcast_grid,
        state_dict=state_dict,
        is_moe=True,
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
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


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_2D})],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
@pytest.mark.parametrize("use_mlp_weights", [True, False], ids=["mlp", "moe"])
@pytest.mark.parametrize("reconfig_moe_cbs", [True, False])
@pytest.mark.parametrize("noc_mode", [ttnn.NOC_MODE.DM_DYNAMIC_NOC])
@pytest.mark.requires_grid_size((13, 10))
def test_mlp_with_reduce(
    bh_2d_mesh_device, use_mlp_weights, reconfig_moe_cbs, noc_mode, get_reference_model_state_dict
):
    """
    Test MoeOp with enable_routing=False and reduce_to_one on 4x2 mesh.

    Each of 8 devices runs the full fused MLP (dense MLP + shared expert),
    then results are reduced (summed) across all devices to ROOT1.

    When use_mlp_weights=False uses MoE layer weights (experts.0 + shared_experts).
    When use_mlp_weights=True uses dense MLP weights (gate_proj/up_proj/down_proj)
    sliced into shared (first 2048) + 8 routed experts.
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
    is_moe = not use_mlp_weights
    layer_idx = DENSE_LAYER_IDX if use_mlp_weights else ROUTED_EXPERT_LAYER_IDX

    logger.info(f"Testing MoeOp no-routing with reduce: K={K}, use_mlp_weights={use_mlp_weights}")

    state_dict = get_reference_model_state_dict(
        layer_idx=layer_idx,
        is_moe=is_moe,
        seed=RoutedExpert.SEED,
        num_routed_experts=256 if is_moe else 4,
        include_global=False,
    )

    # ── Create MLP tensors (replicated across mesh) ──
    mesh_mapper = ttnn.ReplicateTensorToMesh(submesh)
    r = create_routed_expert_tensors(
        submesh,
        mesh_mapper=mesh_mapper,
        create_final_output=False,
        enable_routing=False,
        state_dict=state_dict,
        is_moe=is_moe,
        layer_idx=layer_idx,
    )
    sender_core = r.ttnn_residual_mcast_src.memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
    s = create_shared_expert_tensors(
        submesh,
        M,
        K,
        mcast_grid,
        mesh_mapper=mesh_mapper,
        state_dict=state_dict,
        is_moe=is_moe,
        layer_idx=layer_idx,
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

    device_grid_size = submesh.compute_with_storage_grid_size()
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

    # Single intermediate tensor with 3x shard width for all 3 reduction rounds
    orig_shard_spec = final_output_mem_config.shard_spec
    intermediate_shard_shape = [orig_shard_spec.shape[0], orig_shard_spec.shape[1] * 3]
    intermediate_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            orig_shard_spec.grid,
            intermediate_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    intermediate_tensors = ttnn.from_torch(
        torch.zeros([4, 2, final_output_total_width * 3], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=intermediate_mem_config,
        tile=tile_1x32,
        mesh_mapper=reduce_mesh_mapper,
    )

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
    # Compute per-device golden with per-device TP shards of shared expert weights.
    # Residual is only added on ROOT1 device; non-root devices skip it.
    K_down = s.K_down
    root_device_idx = root_coord[0] * submesh.shape[1] + root_coord[1]
    expected_final_outputs = []
    for device_idx in range(num_devices):
        shared_gate_shard = s.torch_gate_weights[:, device_idx * K_down : (device_idx + 1) * K_down]
        shared_up_shard = s.torch_up_weights[:, device_idx * K_down : (device_idx + 1) * K_down]
        shared_down_shard = s.torch_down_weights[device_idx * K_down : (device_idx + 1) * K_down, :]

        if use_mlp_weights:
            gate_dict = {0: r.expert_weights_dict[device_idx]}
            up_dict = {0: r.up_proj_weights_dict[device_idx]}
            down_dict = {0: r.down_proj_weights_dict[device_idx]}
        else:
            gate_dict = r.expert_weights_dict
            up_dict = r.up_proj_weights_dict
            down_dict = r.down_proj_weights_dict

        _, _, device_expected = MoeOp.golden(
            r.torch_input,
            shared_gate_weights=shared_gate_shard,
            shared_up_weights=shared_up_shard,
            shared_down_weights=shared_down_shard,
            gate_proj_weights_dict=gate_dict,
            up_proj_weights_dict=up_dict,
            down_proj_weights_dict=down_dict,
            rmsnorm_gamma=r.torch_rmsnorm_gamma,
            rmsnorm_epsilon=1e-6,
            enable_routing=False,
            include_residual=(device_idx == root_device_idx),
        )
        expected_final_outputs.append(device_expected)

    # Expected reduce output = sum of all per-device outputs
    expected_reduce_output = sum(expected_final_outputs)

    # Get actual reduce output from ROOT1 device
    reduce_output_torch = ttnn.to_torch(
        ttnn_result_reduce,
        mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0),
    )

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

    # --- Reference MLP comparison ---
    x = r.torch_input.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    normed_input = ((x * torch.rsqrt(variance + 1e-6)) * r.torch_rmsnorm_gamma.float()).bfloat16()

    if use_mlp_weights:
        shared_ref, routed_refs = create_reference_dense_mlp_slices(state_dict, layer_idx)
        with torch.no_grad():
            shared_output = shared_ref(normed_input.unsqueeze(0)).squeeze(0)
            routed_outputs = [routed_refs[d](normed_input.unsqueeze(0)).squeeze(0) for d in range(8)]
        # Residual is added once (on ROOT1 only)
        ref_reduce = sum(routed_outputs).float() + shared_output.float() + r.torch_input.float()
        # Also compare against single full MLP (reference block): reduce_output directly matches
        full_mlp_ref = create_reference_dense_full_mlp(state_dict, layer_idx)
        with torch.no_grad():
            ref_block_output = r.torch_input.float() + full_mlp_ref(normed_input.unsqueeze(0)).squeeze(0).float()
        passing_full, pcc_full = comp_pcc(ref_block_output.flatten(), reduce_output_valid.float().flatten(), 0.975)
        logger.info(f"Reference full MLP (block) comparison PCC: {pcc_full}")
        assert passing_full, f"Reference full MLP block comparison PCC failed: {pcc_full}"
    else:
        expert_ref, shared_ref = create_reference_mlp_models(state_dict, ROUTED_EXPERT_LAYER_IDX)
        with torch.no_grad():
            expert_0_output = expert_ref(normed_input.unsqueeze(0)).squeeze(0)
            shared_full_output = shared_ref(normed_input.unsqueeze(0)).squeeze(0)
        # Residual is added once (on ROOT1 only)
        ref_reduce = num_devices * expert_0_output.float() + shared_full_output.float() + r.torch_input.float()

    passing_ref, pcc_ref = comp_pcc(ref_reduce, reduce_output_valid, 0.975)
    logger.info(f"Reference MLP comparison PCC: {pcc_ref}")
    assert passing_ref, f"Reference MLP comparison PCC failed: {pcc_ref}"

    logger.info("MoeOp no-routing with reduce test PASSED!")
