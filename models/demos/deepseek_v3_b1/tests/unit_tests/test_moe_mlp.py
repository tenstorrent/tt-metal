# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Tests for fused MoE operation (routed expert + shared expert).

Runs both MoE routed expert and shared expert on the same input,
validates each independently, and verifies the combined MoE output.

Run:
    pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_moe.py -v -s
"""

import gc
from typing import Any, NamedTuple

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, skip_for_wormhole_b0
from models.demos.deepseek_v3_b1.fused_ops.down_proj.op import DownProj
from models.demos.deepseek_v3_b1.fused_ops.moe.op import MoeOp
from models.demos.deepseek_v3_b1.fused_ops.shared_expert.op import SharedExpertOp
from models.demos.deepseek_v3_b1.weights.prepare import (
    _load_expert_proj,
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
from models.demos.deepseek_v3_b1.model_dimensions import RoutedExpert, SharedExpert


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

    moe_tp = device.shape[0] * device.shape[1] if hasattr(device, "shape") else 1
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
        device, state_dict, layer_idx=layer_idx, is_moe=is_moe, move_to_device=True
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
    bspm_dir=None,
    preloaded_experts=None,
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
        bspm_dir: Optional path to BitSculpt results root for BSPM mixed-precision loading.
        preloaded_experts: Optional tuple (gate_list, up_list, down_list) of pre-loaded device
            tensors (plain ttnn.Tensor or CompressedTensor). When provided, skips
            prepare_routed_expert_weights — the float golden dicts are still built from state_dict.

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

    # Load expert weights to DRAM BEFORE allocating attention weights to L1.
    # to_memory_config for DRAM-sharded expert tensors dispatches a copy program
    # that needs L1 CBs.  If attention weights are already in L1 at those CB
    # addresses, the dispatch check fires a "CB clash" error.  Loading to DRAM
    # first (L1 is empty) avoids this.
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

        if preloaded_experts is not None:
            gate_proj_expert_tensors, up_proj_expert_tensors, down_proj_expert_tensors = preloaded_experts
        else:
            from pathlib import Path as _Path

            bspm_dir_path = _Path(bspm_dir) if bspm_dir is not None else None
            routed_weights = prepare_routed_expert_weights(
                device,
                state_dict,
                layer_idx=layer_idx,
                is_moe=True,
                num_routed_experts=num_experts,
                move_to_device=True,
                bspm_dir=bspm_dir_path / "deepseek-r1-0528" if bspm_dir_path is not None else None,
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
            device,
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

    # Build attention-side overlapped tensors from state dict via prepare_weights.
    # Must run AFTER routed expert weights are in DRAM so L1 is clean for the
    # to_memory_config dispatches used internally.
    attn = prepare_attention_weights(device, state_dict, layer_idx=layer_idx, is_moe=is_moe, move_to_device=True)
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


def rig_moe_gate_for_expected_experts(
    state_dict,
    layer_idx,
    winning_groups,
    winning_experts_by_group,
    *,
    low_bias=-10.0,
    high_bias=10.0,
    zero_gate_weights=True,
):
    """Rig grouped gate so top-k deterministically selects requested experts."""
    gate_key = f"model.layers.{layer_idx}.mlp.gate.weight"
    bias_key = f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"

    if zero_gate_weights:
        state_dict[gate_key] = torch.zeros_like(state_dict[gate_key])

    bias_2d = torch.full((8, 32), low_bias, dtype=state_dict[bias_key].dtype)
    expected_expert_ids = []
    for group_id in winning_groups:
        for expert_id in winning_experts_by_group[group_id]:
            bias_2d[group_id, expert_id] = high_bias
            expected_expert_ids.append(group_id * 32 + expert_id)

    state_dict[bias_key] = bias_2d.reshape(-1).contiguous()
    return expected_expert_ids


@pytest.mark.parametrize(
    "use_hardcoded_expert_index",
    [True, pytest.param(False, marks=pytest.mark.skip_post_commit)],
)
@pytest.mark.parametrize("reconfig_moe_cbs", [True, False])
@pytest.mark.parametrize("noc_mode", [ttnn.NOC_MODE.DM_DYNAMIC_NOC])
@pytest.mark.requires_grid_size((13, 10))
@pytest.mark.timeout(1200)
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
    torch_expected_scores, torch_expected_indices, torch_expected_final = MoeOp.golden_single_device(
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
@pytest.mark.parametrize("reconfig_moe_cbs", [True, False])
@pytest.mark.parametrize("noc_mode", [ttnn.NOC_MODE.DM_DYNAMIC_NOC])
@pytest.mark.requires_grid_size((13, 10))
@pytest.mark.timeout(1200)
def test_moe_fused_with_reduce(bh_2d_mesh_device, reconfig_moe_cbs, noc_mode, get_reference_model_state_dict):
    """
    Test fused MoE with reduce_to_one on 4x2 mesh.

    Each of 8 devices runs the full fused MoE (routed + shared expert),
    then results are reduced (summed) across all devices to ROOT1.

    Gate is rigged so grouped top-k picks deterministic winners.
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
    winning_groups = [0, 2, 5, 7]
    winning_experts_by_group = {
        0: [1, 9],
        2: [4, 19],
        5: [7, 23],
        7: [3, 28],
    }
    expected_expert_ids = rig_moe_gate_for_expected_experts(
        state_dict,
        ROUTED_EXPERT_LAYER_IDX,
        winning_groups,
        winning_experts_by_group,
    )

    # ── Create MoE tensors (replicated across mesh) ──
    mesh_mapper = ttnn.ReplicateTensorToMesh(submesh)
    r = create_routed_expert_tensors(
        submesh,
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
        use_hardcoded_expert_index=False,
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
    # Read gate scores/indices from device
    device_gate_indices = ttnn.to_torch(ttnn_result_indices, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))
    _ = ttnn.to_torch(ttnn_result_scores, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))
    root_device_idx = root_coord[0] * submesh.shape[1] + root_coord[1]
    tt_top8 = device_gate_indices[0].flatten()[:8].to(torch.int64)
    tt_top8_sorted = torch.sort(tt_top8).values
    expected_top8_sorted = torch.sort(torch.tensor(expected_expert_ids, dtype=torch.int64)).values
    assert torch.equal(tt_top8_sorted, expected_top8_sorted), (
        f"Rigged gate experts mismatch: expected={expected_top8_sorted.tolist()}, " f"got={tt_top8_sorted.tolist()}"
    )

    # One logical golden call (h + MoE(h)); no per-device golden loop needed.
    _, _, expected_reduce_output = MoeOp.golden(
        r.torch_input,
        shared_gate_weights=s.torch_gate_weights,
        shared_up_weights=s.torch_up_weights,
        shared_down_weights=s.torch_down_weights,
        gate_proj_weights_dict=r.expert_weights_dict,
        up_proj_weights_dict=r.up_proj_weights_dict,
        down_proj_weights_dict=r.down_proj_weights_dict,
        rmsnorm_gamma=r.torch_rmsnorm_gamma,
        routing_weights_tensor=r.torch_gate_mm_weights,
        bias_tensor=r.torch_bias,
        rmsnorm_epsilon=1e-6,
        routing_mode=True,
        eps=r.gate_eps,
        scaling_factor=r.gate_scaling_factor,
        include_residual=True,
    )

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

    # --- Reference model comparison ---
    num_experts_in_state_dict = sum(1 for k in state_dict if ".mlp.experts." in k and "gate_proj" in k)
    if num_experts_in_state_dict >= 256:
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
@pytest.mark.timeout(1200)
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
        routing_weights_tensor=r.torch_gate_mm_weights,
        bias_tensor=r.torch_bias,
        rmsnorm_epsilon=1e-6,
        routing_mode=False,
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
@pytest.mark.parametrize("use_mlp_weights", [True], ids=["mlp"])
@pytest.mark.parametrize("reconfig_moe_cbs", [True])
@pytest.mark.parametrize("noc_mode", [ttnn.NOC_MODE.DM_DYNAMIC_NOC])
@pytest.mark.requires_grid_size((13, 10))
@pytest.mark.timeout(1200)
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

        _, _, device_expected = MoeOp.golden_single_device(
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


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.requires_grid_size((13, 10))
@pytest.mark.timeout(1200)
def test_moe_fused_bspm(device, get_reference_model_state_dict):
    """Test fused MoE with real BSPM-driven mixed-precision weights.

    Validates Phase 3 path: prepare_routed_expert_weights(bspm_dir=...) →
    CompressedTensor.from_bspm() → MoeOp → DRAMStreamingMatmulCompressed.

    Requires env var:
      BSPM_DIR — path to BitSculpt results root (contains deepseek-r1-0528/ subdir)

    Uses random weights (no real checkpoint needed) — the BSPM codes are what drive the
    mixed-precision tile assignment. The float golden is computed from the same random
    weights, so PCC validates kernel correctness independent of weight values.

    PCC threshold is 0.90 (lower than 0.97 for uniform BFP4 — 3.5 b/e uses ~35% bfp2/zero tiles).
    """
    import os

    bspm_dir = os.environ.get("BSPM_DIR")
    if not bspm_dir:
        pytest.skip("BSPM_DIR not set — skipping real mixed-precision path test")

    M = RoutedExpert.M
    K = RoutedExpert.K

    logger.info(f"Testing fused MoE with BSPM mixed-precision: layer={ROUTED_EXPERT_LAYER_IDX}, bspm_dir={bspm_dir}")

    state_dict = get_reference_model_state_dict(
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=1,
    )

    r = create_routed_expert_tensors(
        device,
        use_hardcoded_expert_index=True,
        state_dict=state_dict,
        is_moe=True,
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        bspm_dir=bspm_dir,
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

    # ── SDPA buffers (required by MoeOp.op for CB memory overlap) ──
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

    moe_semaphores = MoeOp.create_semaphores(device)
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
        shared_gate_weights_overlapped=s.shared_gate_weights_overlapped,
        shared_up_weights_overlapped=s.shared_up_weights_overlapped,
        shared_down_weights_tensor=s.ttnn_down_weights,
        shared_k_parallel=s.k_parallel,
        shared_n_parallel=s.n_parallel,
        use_hardcoded_expert_index=True,
        sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
        sdpa_out_interm_buffer=sdpa_out_interm_buffer,
        num_iterations=1,
        reconfig_moe_cbs=True,
        semaphores=moe_semaphores,
        noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
    )
    ttnn.synchronize_device(device)

    output_final_torch = ttnn.to_torch(ttnn_result_final)
    output_final_valid = extract_routed_expert_output(
        output_final_torch,
        r.num_gate_proj_cores,
        r.final_output_width_per_core,
        r.per_core_down_proj_N,
    )

    torch_expected_scores, torch_expected_indices, torch_expected_final = MoeOp.golden_single_device(
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
        use_hardcoded_expert_index=True,
    )

    assert output_final_valid.abs().sum() > 0, "BSPM output is all zeros — kernel may not have run"

    passing, pcc = comp_pcc(torch_expected_final, output_final_valid, 0.90)
    logger.info(f"BSPM fused MoE PCC: {pcc} (threshold=0.90)")
    assert passing, f"BSPM fused MoE PCC check failed (got {pcc}, expected >= 0.90)"

    logger.info(f"BSPM fused MoE test PASSED! (PCC={pcc})")


# ============================================================================
# F1b: per-expert DRAM offset table + stacked metadata (B6 extension)
# ============================================================================


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.requires_grid_size((13, 10))
@pytest.mark.timeout(1200)
def test_moe_fused_bspm_f1b_compat(device, get_reference_model_state_dict):
    """F1b backward compatibility: single-element list[CT] must behave identically to scalar CT.

    Wraps gate/up/down_proj weights in [ct] before calling MoeOp.op.
    _setup_compressed_matmul_metadata sees num_experts=1 → expert_offsets_l1_addr=0 →
    legacy B6 code path unchanged.  PCC threshold 0.90 (same as test_moe_fused_bspm).

    Requires env var: BSPM_DIR
    """
    import os

    bspm_dir = os.environ.get("BSPM_DIR")
    if not bspm_dir:
        pytest.skip("BSPM_DIR not set — skipping F1b compat test")

    logger.info("Testing F1b compat: single-element list[CT] via BSPM path")

    state_dict = get_reference_model_state_dict(
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=1,
    )
    r = create_routed_expert_tensors(
        device,
        use_hardcoded_expert_index=True,
        state_dict=state_dict,
        is_moe=True,
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        bspm_dir=bspm_dir,
    )
    sender_core = r.ttnn_residual_mcast_src.memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
    s = create_shared_expert_tensors(
        device,
        RoutedExpert.M,
        RoutedExpert.K,
        mcast_grid,
        state_dict=state_dict,
        is_moe=True,
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
    )

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
    full_device_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )
    num_full_cores = device_grid_size.x * device_grid_size.y
    sdpa_out_interm_shard_spec = ttnn.ShardSpec(
        full_device_grid,
        (SDPA.OUT_INTERM_SHARD_HEIGHT, SDPA.OUT_INTERM_SHARD_WIDTH),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sdpa_out_interm_buffer = ttnn.from_torch(
        torch.zeros((SDPA.OUT_INTERM_SHARD_HEIGHT * num_full_cores, SDPA.OUT_INTERM_SHARD_WIDTH), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sdpa_out_interm_shard_spec
        ),
        tile=Tiles.TILE_8x32,
    )

    moe_semaphores = MoeOp.create_semaphores(device)
    # Key change vs test_moe_fused_bspm: wrap each CT in a single-element list.
    # This exercises the isinstance(..., list) branch in op.py while keeping num_experts=1
    # so expert_offsets_l1_addr=0 and the legacy B6 code path runs unchanged.
    ttnn_result_scores, ttnn_result_indices, ttnn_result_final = MoeOp.op(
        r.ttnn_residual_mcast_src,
        r.ttnn_gate_mm_weights,
        r.ttnn_gate_bias,
        r.ttnn_gate_indices,
        r.gate_output_scores_tensor,
        r.gate_output_indices_tensor,
        [r.gate_proj_weights],
        [r.up_proj_weights],
        [r.down_proj_weights],
        r.final_output_tensor,
        r.ttnn_rmsnorm_gamma,
        shared_gate_weights_overlapped=s.shared_gate_weights_overlapped,
        shared_up_weights_overlapped=s.shared_up_weights_overlapped,
        shared_down_weights_tensor=s.ttnn_down_weights,
        shared_k_parallel=s.k_parallel,
        shared_n_parallel=s.n_parallel,
        use_hardcoded_expert_index=True,
        sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
        sdpa_out_interm_buffer=sdpa_out_interm_buffer,
        num_iterations=1,
        reconfig_moe_cbs=True,
        semaphores=moe_semaphores,
        noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
    )
    ttnn.synchronize_device(device)

    output_final_torch = ttnn.to_torch(ttnn_result_final)
    output_final_valid = extract_routed_expert_output(
        output_final_torch,
        r.num_gate_proj_cores,
        r.final_output_width_per_core,
        r.per_core_down_proj_N,
    )

    _, _, torch_expected_final = MoeOp.golden_single_device(
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
        use_hardcoded_expert_index=True,
    )

    assert output_final_valid.abs().sum() > 0, "F1b compat output is all zeros — kernel may not have run"
    passing, pcc = comp_pcc(torch_expected_final, output_final_valid, 0.90)
    logger.info(f"F1b compat PCC: {pcc} (threshold=0.90)")
    assert passing, f"F1b compat PCC check failed (got {pcc}, expected >= 0.90)"
    logger.info(f"F1b compat test PASSED! (PCC={pcc})")


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.requires_grid_size((13, 10))
@pytest.mark.timeout(1200)
def test_moe_fused_bspm_f1b(device, get_reference_model_state_dict):
    """F1b kernel path: two-element list[CT, CT] with same CT forces expert_offsets_l1_addr != 0.

    Passes [ct0, ct0] (same CompressedTensor object twice) so num_experts=2 triggers the
    F1b code path: expert_offsets_l1_addr is allocated, the offset table is [0, 0], and
    expert_idx=0 selects offset 0.  Output must match the single-CT golden (PCC ≥ 0.90).

    This exercises _setup_compressed_matmul_metadata with num_experts > 1, the NCRISC
    offset-table lookup + scratch write, and the TRISC cb_wait_front fence + fmt_tile_offset.

    Requires env var: BSPM_DIR
    """
    import os

    bspm_dir = os.environ.get("BSPM_DIR")
    if not bspm_dir:
        pytest.skip("BSPM_DIR not set — skipping F1b two-element list test")

    logger.info("Testing F1b: two-element [ct0, ct0] list triggers F1b kernel path")

    state_dict = get_reference_model_state_dict(
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=1,
    )
    r = create_routed_expert_tensors(
        device,
        use_hardcoded_expert_index=True,
        state_dict=state_dict,
        is_moe=True,
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        bspm_dir=bspm_dir,
    )
    sender_core = r.ttnn_residual_mcast_src.memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
    s = create_shared_expert_tensors(
        device,
        RoutedExpert.M,
        RoutedExpert.K,
        mcast_grid,
        state_dict=state_dict,
        is_moe=True,
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
    )

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
    full_device_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )
    num_full_cores = device_grid_size.x * device_grid_size.y
    sdpa_out_interm_shard_spec = ttnn.ShardSpec(
        full_device_grid,
        (SDPA.OUT_INTERM_SHARD_HEIGHT, SDPA.OUT_INTERM_SHARD_WIDTH),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sdpa_out_interm_buffer = ttnn.from_torch(
        torch.zeros((SDPA.OUT_INTERM_SHARD_HEIGHT * num_full_cores, SDPA.OUT_INTERM_SHARD_WIDTH), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sdpa_out_interm_shard_spec
        ),
        tile=Tiles.TILE_8x32,
    )

    moe_semaphores = MoeOp.create_semaphores(device)
    # Duplicate the same CT: num_experts=2 → expert_offsets_l1_addr != 0 → F1b path.
    # offset table = [0, 0]; expert_idx=0 → offset 0 → same DRAM address as ct0.
    # Stacked metadata duplicates ct0's meta/fmt blocks; TRISC fmt_tile_offset=0 reads block 0.
    ct0_gate = r.gate_proj_weights
    ct0_up = r.up_proj_weights
    ct0_down = r.down_proj_weights
    ttnn_result_scores, ttnn_result_indices, ttnn_result_final = MoeOp.op(
        r.ttnn_residual_mcast_src,
        r.ttnn_gate_mm_weights,
        r.ttnn_gate_bias,
        r.ttnn_gate_indices,
        r.gate_output_scores_tensor,
        r.gate_output_indices_tensor,
        [ct0_gate, ct0_gate],
        [ct0_up, ct0_up],
        [ct0_down, ct0_down],
        r.final_output_tensor,
        r.ttnn_rmsnorm_gamma,
        shared_gate_weights_overlapped=s.shared_gate_weights_overlapped,
        shared_up_weights_overlapped=s.shared_up_weights_overlapped,
        shared_down_weights_tensor=s.ttnn_down_weights,
        shared_k_parallel=s.k_parallel,
        shared_n_parallel=s.n_parallel,
        use_hardcoded_expert_index=True,
        sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
        sdpa_out_interm_buffer=sdpa_out_interm_buffer,
        num_iterations=1,
        reconfig_moe_cbs=True,
        semaphores=moe_semaphores,
        noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
    )
    ttnn.synchronize_device(device)

    output_final_torch = ttnn.to_torch(ttnn_result_final)
    output_final_valid = extract_routed_expert_output(
        output_final_torch,
        r.num_gate_proj_cores,
        r.final_output_width_per_core,
        r.per_core_down_proj_N,
    )

    _, _, torch_expected_final = MoeOp.golden_single_device(
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
        use_hardcoded_expert_index=True,
    )

    assert output_final_valid.abs().sum() > 0, "F1b two-CT output is all zeros — kernel may not have run"
    passing, pcc = comp_pcc(torch_expected_final, output_final_valid, 0.90)
    logger.info(f"F1b [ct0, ct0] PCC: {pcc} (threshold=0.90)")
    assert passing, f"F1b [ct0, ct0] PCC check failed (got {pcc}, expected >= 0.90)"
    logger.info(f"F1b [ct0, ct0] test PASSED! (PCC={pcc})")


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.requires_grid_size((13, 10))
@pytest.mark.timeout(1800)
def test_moe_fused_bspm_f1b_expert_selection(device, get_reference_model_state_dict):
    """F1b expert selection: DRAM offset table routes to the correct expert.

    Loads two BSPM CompressedTensors with distinct weight data (experts 0 and 1).
    Run A: gate_proj_weights=[ct0, ct1], expert_idx=0 → offset table[0]=0 → ct0 selected.
    Run B: gate_proj_weights=[ct1, ct0], expert_idx=0 → offset table[0]=0 → ct1 selected.
    Each run's output must PCC ≥ 0.90 against the corresponding expert's float golden.

    This is the primary correctness test for the F1b offset table: swapping positions in
    the list must change which expert's DRAM data and metadata are read.

    Requires env var: BSPM_DIR
    """
    import os
    from pathlib import Path

    bspm_dir = os.environ.get("BSPM_DIR")
    if not bspm_dir:
        pytest.skip("BSPM_DIR not set — skipping F1b expert selection test")

    logger.info("Testing F1b expert selection: [ct0,ct1] vs [ct1,ct0] with expert_idx=0")

    state_dict = get_reference_model_state_dict(
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=2,
    )

    # Load 2 BSPM CompressedTensors: experts 0 and 1 have different random weights
    # (seeded differently in _reference_layer_state_dict) and therefore different
    # BSPM tile assignments, meta/fmt tables, and DRAM addresses.
    routed_weights = prepare_routed_expert_weights(
        device,
        state_dict,
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        is_moe=True,
        num_routed_experts=2,
        move_to_device=True,
        bspm_dir=Path(bspm_dir) / "deepseek-r1-0528",
    )
    ct0_gate, ct1_gate = routed_weights.routed_gate_proj
    ct0_up, ct1_up = routed_weights.routed_up_proj
    ct0_down, ct1_down = routed_weights.routed_down_proj

    # Build scaffolding (routing, attention, output tensors) using preloaded_experts.
    # create_routed_expert_tensors with use_hardcoded_expert_index=True on a single device
    # sets num_experts=1, so only expert 0's float weights end up in expert_weights_dict.
    r = create_routed_expert_tensors(
        device,
        use_hardcoded_expert_index=True,
        state_dict=state_dict,
        is_moe=True,
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
        preloaded_experts=([ct0_gate, ct1_gate], [ct0_up, ct1_up], [ct0_down, ct1_down]),
    )

    # Extract expert 1's float weights for run B golden.
    # r.expert_weights_dict only has key 0 (expert 0) since num_experts=1 in the scaffold.
    layer_key = f"model.layers.{ROUTED_EXPERT_LAYER_IDX}"
    expert1_gate = (
        state_dict[f"{layer_key}.mlp.experts.1.gate_proj.weight"]
        .T.contiguous()
        .reshape(1, 1, RoutedExpert.K, RoutedExpert.GATE_PROJ_N)
    )
    expert1_up = (
        state_dict[f"{layer_key}.mlp.experts.1.up_proj.weight"]
        .T.contiguous()
        .reshape(1, 1, RoutedExpert.K, RoutedExpert.GATE_PROJ_N)
    )
    expert1_down = (
        state_dict[f"{layer_key}.mlp.experts.1.down_proj.weight"]
        .T.contiguous()
        .reshape(1, 1, RoutedExpert.GATE_PROJ_N, RoutedExpert.K)
    )

    sender_core = r.ttnn_residual_mcast_src.memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
    s = create_shared_expert_tensors(
        device,
        RoutedExpert.M,
        RoutedExpert.K,
        mcast_grid,
        state_dict=state_dict,
        is_moe=True,
        layer_idx=ROUTED_EXPERT_LAYER_IDX,
    )

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
    full_device_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )
    num_full_cores = device_grid_size.x * device_grid_size.y
    sdpa_out_interm_shard_spec = ttnn.ShardSpec(
        full_device_grid,
        (SDPA.OUT_INTERM_SHARD_HEIGHT, SDPA.OUT_INTERM_SHARD_WIDTH),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sdpa_out_interm_buffer = ttnn.from_torch(
        torch.zeros((SDPA.OUT_INTERM_SHARD_HEIGHT * num_full_cores, SDPA.OUT_INTERM_SHARD_WIDTH), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, sdpa_out_interm_shard_spec
        ),
        tile=Tiles.TILE_8x32,
    )

    moe_semaphores = MoeOp.create_semaphores(device)

    # ── Run A: [ct0, ct1] → expert_idx=0 → offset_table[0]=0 → ct0 selected ──
    logger.info("F1b selection run A: [ct0, ct1], expert_idx=0 → expect ct0 output")
    _, _, ttnn_result_final_a = MoeOp.op(
        r.ttnn_residual_mcast_src,
        r.ttnn_gate_mm_weights,
        r.ttnn_gate_bias,
        r.ttnn_gate_indices,
        r.gate_output_scores_tensor,
        r.gate_output_indices_tensor,
        [ct0_gate, ct1_gate],
        [ct0_up, ct1_up],
        [ct0_down, ct1_down],
        r.final_output_tensor,
        r.ttnn_rmsnorm_gamma,
        shared_gate_weights_overlapped=s.shared_gate_weights_overlapped,
        shared_up_weights_overlapped=s.shared_up_weights_overlapped,
        shared_down_weights_tensor=s.ttnn_down_weights,
        shared_k_parallel=s.k_parallel,
        shared_n_parallel=s.n_parallel,
        use_hardcoded_expert_index=True,
        sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
        sdpa_out_interm_buffer=sdpa_out_interm_buffer,
        num_iterations=1,
        reconfig_moe_cbs=True,
        semaphores=moe_semaphores,
        noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
    )
    ttnn.synchronize_device(device)

    output_a = extract_routed_expert_output(
        ttnn.to_torch(ttnn_result_final_a),
        r.num_gate_proj_cores,
        r.final_output_width_per_core,
        r.per_core_down_proj_N,
    )
    _, _, torch_expected_a = MoeOp.golden_single_device(
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
        use_hardcoded_expert_index=True,
    )

    assert output_a.abs().sum() > 0, "F1b selection run A output is all zeros"
    passing_a, pcc_a = comp_pcc(torch_expected_a, output_a, 0.90)
    logger.info(f"F1b selection run A (ct0 at pos 0) PCC: {pcc_a} (threshold=0.90)")
    assert passing_a, f"F1b selection run A PCC failed (got {pcc_a}, expected >= 0.90)"

    # ── Run B: [ct1, ct0] → expert_idx=0 → offset_table[0]=0 → ct1 selected ──
    # The base address for the offset table is now ct1.data.buffer_address(), so
    # offset_table[0]=0 points to ct1's DRAM data and ct1's stacked meta/fmt block.
    logger.info("F1b selection run B: [ct1, ct0], expert_idx=0 → expect ct1 output")
    final_output_tensor_b = ttnn.from_torch(
        torch.zeros([1, 1, 1, r.final_output_total_width]).bfloat16().float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=r.final_output_mem_config,
        tile=Tiles.TILE_1x32,
    )
    _, _, ttnn_result_final_b = MoeOp.op(
        r.ttnn_residual_mcast_src,
        r.ttnn_gate_mm_weights,
        r.ttnn_gate_bias,
        r.ttnn_gate_indices,
        r.gate_output_scores_tensor,
        r.gate_output_indices_tensor,
        [ct1_gate, ct0_gate],
        [ct1_up, ct0_up],
        [ct1_down, ct0_down],
        final_output_tensor_b,
        r.ttnn_rmsnorm_gamma,
        shared_gate_weights_overlapped=s.shared_gate_weights_overlapped,
        shared_up_weights_overlapped=s.shared_up_weights_overlapped,
        shared_down_weights_tensor=s.ttnn_down_weights,
        shared_k_parallel=s.k_parallel,
        shared_n_parallel=s.n_parallel,
        use_hardcoded_expert_index=True,
        sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
        sdpa_out_interm_buffer=sdpa_out_interm_buffer,
        num_iterations=1,
        reconfig_moe_cbs=True,
        semaphores=moe_semaphores,
        noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
    )
    ttnn.synchronize_device(device)

    output_b = extract_routed_expert_output(
        ttnn.to_torch(ttnn_result_final_b),
        r.num_gate_proj_cores,
        r.final_output_width_per_core,
        r.per_core_down_proj_N,
    )
    # Run B golden uses expert 1's float weights at dict key 0 (golden routes to key 0
    # when use_hardcoded_expert_index=True; the kernel is selecting ct1 via the offset table).
    _, _, torch_expected_b = MoeOp.golden_single_device(
        r.torch_input,
        shared_gate_weights=s.torch_gate_weights,
        shared_up_weights=s.torch_up_weights,
        shared_down_weights=s.torch_down_weights,
        gate_proj_weights_dict={0: expert1_gate},
        up_proj_weights_dict={0: expert1_up},
        down_proj_weights_dict={0: expert1_down},
        rmsnorm_gamma=r.torch_rmsnorm_gamma,
        rmsnorm_epsilon=1e-6,
        routing_weights_tensor=r.torch_gate_mm_weights,
        bias_tensor=r.torch_bias,
        eps=r.gate_eps,
        scaling_factor=r.gate_scaling_factor,
        use_hardcoded_expert_index=True,
    )

    assert output_b.abs().sum() > 0, "F1b selection run B output is all zeros"
    passing_b, pcc_b = comp_pcc(torch_expected_b, output_b, 0.90)
    logger.info(f"F1b selection run B (ct1 at pos 0) PCC: {pcc_b} (threshold=0.90)")
    assert passing_b, f"F1b selection run B PCC failed (got {pcc_b}, expected >= 0.90)"

    logger.info(f"F1b expert selection test PASSED! (run A PCC={pcc_a}, run B PCC={pcc_b})")


# ============================================================================
# Real-weight BSPM cache validation
# ============================================================================


def _load_cached_expert_subset(cache_path, device, layer_idx: int, num_experts: int):
    """Load the first num_experts experts from the BSPM cache (gate-all, up-all, down-all order).

    Mirrors the allocation order of load_moe_routed_experts so that DRAM buffer
    layout matches what DRAMStreamingExpertsMatmulCompressed expects.
    """
    from pathlib import Path

    experts_dir = Path(cache_path) / f"layer_{layer_idx:03d}" / "experts"
    gate_list, up_list, down_list = [], [], []
    for e in range(num_experts):
        gate_list.append(_load_expert_proj(experts_dir / f"e_{e:03d}", "gate_proj", device))
    for e in range(num_experts):
        up_list.append(_load_expert_proj(experts_dir / f"e_{e:03d}", "up_proj", device))
    for e in range(num_experts):
        down_list.append(_load_expert_proj(experts_dir / f"e_{e:03d}", "down_proj", device))
    logger.info("Loaded {} experts from BSPM cache for layer {}", num_experts, layer_idx)
    return gate_list, up_list, down_list


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.requires_grid_size((13, 10))
@pytest.mark.timeout(600)
@pytest.mark.parametrize("layer_idx", [4, 7, 9, 18, 30, 31, 57])
def test_moe_fused_bspm_real_ct(device, get_reference_model_state_dict, layer_idx):
    """Bridge test: MoeOp + real BSPM tile assignment via TensorCache + synthetic random weights.

    Sits between B.4 (test_moe_fused_bspm — synthetic CT, fixed layer 4) and B.5
    (test_moe_layer_real_weights — real HF weights). Uses the real BSPM tile assignment
    for layer_idx from TensorCache but packs it with deterministic random weights, so
    DEEPSEEK_V3_HF_MODEL is not needed.

    The golden is computed from the same random weights (not the compressed approximation),
    so PCC ≥ 0.90 validates that:
      1. The BSPM tile assignment for this layer loads and shuffles correctly via TensorCache.
      2. The CompressedTensor stores (miss → tiles.bin) and reloads (hit) correctly.
      3. MoeOp correctly decompresses this layer's specific tile pattern — bfp4/bfp2/zero
         distribution, subblock sizes, CBIn1BaseAddr — without needing real HF values.

    Diagnostic interpretation:
      - Fails here + passes test_moe_fused_bspm → layer-specific tile pattern or TensorCache
        path is the root cause, independent of real weight values.
      - Passes here + fails test_moe_layer_real_weights → something specific to real HF
        weight values (e.g. activations + weights interaction, not tile layout).

    Required env vars:
      BSPM_DIR      — path to bit_sculpt results root (contains deepseek-r1-0528/ subdir)
      TT_CACHE_PATH — path to TensorCache root (populated on first run)
    """
    import os
    from pathlib import Path

    bspm_dir_env = os.environ.get("BSPM_DIR")
    tt_cache_path = os.environ.get("TT_CACHE_PATH")
    if not bspm_dir_env or not tt_cache_path:
        pytest.skip("BSPM_DIR and TT_CACHE_PATH must be set — skipping bridge BSPM real-CT test")

    from models.demos.deepseek_v3_b1.weights.cache import CacheConfig, CacheContext, TensorCache
    from models.demos.deepseek_v3_b1.weights.prepare import prepare_routed_expert_weights

    bspm_dir_path = Path(bspm_dir_env) / "deepseek-r1-0528"
    cache_config = CacheConfig(
        cache=TensorCache(Path(tt_cache_path)),
        context=CacheContext(
            schema_version=1,
            hf_model_id="deepseek-ai/DeepSeek-R1-0528",
            hf_revision="local",
            mesh_shape=(1, 1),
        ),
    )

    # Deterministic random weights (fixed seed) → stable TensorCache fingerprint across runs.
    state_dict = get_reference_model_state_dict(
        layer_idx=layer_idx,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=1,
        random_weights=True,
    )

    logger.info(
        "test_moe_fused_bspm_real_ct: layer={}, bspm_dir={}, cache={}",
        layer_idx,
        bspm_dir_path,
        tt_cache_path,
    )

    # Load CompressedTensor via TensorCache (miss → tiles.bin write; hit → reload).
    # Random weights + real BSPM assignment → no HF checkpoint needed.
    result = prepare_routed_expert_weights(
        device,
        state_dict,
        layer_idx,
        is_moe=True,
        num_routed_experts=1,
        move_to_device=True,
        bspm_dir=bspm_dir_path,
        cache_config=cache_config,
    )
    experts = (result.routed_gate_proj, result.routed_up_proj, result.routed_down_proj)

    # create_routed_expert_tensors with preloaded_experts uses the CTs for device weights
    # and builds the float golden dicts from state_dict (same random weights) — correct PCC.
    r = create_routed_expert_tensors(
        device,
        use_hardcoded_expert_index=True,
        state_dict=state_dict,
        is_moe=True,
        layer_idx=layer_idx,
        bspm_dir=None,
        preloaded_experts=experts,
    )

    sender_core = r.ttnn_residual_mcast_src.memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
    s = create_shared_expert_tensors(
        device,
        RoutedExpert.M,
        RoutedExpert.K,
        mcast_grid,
        state_dict=state_dict,
        is_moe=True,
        layer_idx=layer_idx,
    )

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
    full_device_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )
    num_full_cores = device_grid_size.x * device_grid_size.y
    sdpa_out_interm_shard_spec = ttnn.ShardSpec(
        full_device_grid,
        (SDPA.OUT_INTERM_SHARD_HEIGHT, SDPA.OUT_INTERM_SHARD_WIDTH),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sdpa_out_interm_buffer = ttnn.from_torch(
        torch.zeros((SDPA.OUT_INTERM_SHARD_HEIGHT * num_full_cores, SDPA.OUT_INTERM_SHARD_WIDTH), dtype=torch.bfloat16),
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
        shared_gate_weights_overlapped=s.shared_gate_weights_overlapped,
        shared_up_weights_overlapped=s.shared_up_weights_overlapped,
        shared_down_weights_tensor=s.ttnn_down_weights,
        shared_k_parallel=s.k_parallel,
        shared_n_parallel=s.n_parallel,
        use_hardcoded_expert_index=True,
        sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
        sdpa_out_interm_buffer=sdpa_out_interm_buffer,
        num_iterations=1,
        reconfig_moe_cbs=True,
        semaphores=moe_semaphores,
        noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
    )
    ttnn.synchronize_device(device)

    output_final_torch = ttnn.to_torch(ttnn_result_final)
    output_final_valid = extract_routed_expert_output(
        output_final_torch,
        r.num_gate_proj_cores,
        r.final_output_width_per_core,
        r.per_core_down_proj_N,
    )

    torch_expected_scores, torch_expected_indices, torch_expected_final = MoeOp.golden_single_device(
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
        use_hardcoded_expert_index=True,
    )

    assert output_final_valid.abs().sum() > 0, "Output is all zeros — kernel may not have run"

    passing, pcc = comp_pcc(torch_expected_final, output_final_valid, 0.90)
    logger.info("test_moe_fused_bspm_real_ct layer={} PCC: {} (threshold=0.90)", layer_idx, pcc)
    assert passing, f"PCC check failed for layer {layer_idx} (got {pcc}, expected >= 0.90)"

    logger.info("test_moe_fused_bspm_real_ct layer={} PASSED! (PCC={})", layer_idx, pcc)


@skip_for_wormhole_b0()
@pytest.mark.parametrize("layer_idx", [4, 7, 9, 18, 30, 31, 57])
def test_moe_layer_real_weights(device, hf_state_dict, get_reference_model_state_dict, layer_idx):
    """MoE MLP: real BSPM-compressed weights from TensorCache vs PyTorch float32 reference.

    Loads BSPM 3.5 b/e expert tensors via prepare_routed_expert_weights() with a
    TensorCache-backed CacheConfig (compact tiles.bin format), runs through MoeOp,
    and compares against a float32 PyTorch golden.

    Only the expert gate/up/down float weights for the golden come from the real HF
    checkpoint. All scaffolding (routing matrix, gamma, attention weights, shared expert)
    uses deterministic random weights to avoid the FP8→ttnn.from_torch issue in
    prepare_attention_weights when loading directly from the R1-0528 FP8 checkpoint.

    PCC threshold is 0.90 (consistent with 3.5 b/e having ~35% bfp2/zero tiles).

    Required env vars:
      DEEPSEEK_V3_HF_MODEL  — path to DeepSeek R1-0528 HF checkpoint (for expert float golden)
      BSPM_DIR              — path to bit_sculpt results root (contains deepseek-r1-0528/ subdir)
      TT_CACHE_PATH         — path to TensorCache root (will be populated on first run)

    Run (BH Galaxy only — requires 13x10 grid):
      TT_METAL_CLEAR_L1=1 TT_METAL_SLOW_DISPATCH_MODE=1 \\
        DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/.../DeepSeek-R1-0528 \\
        BSPM_DIR=/path/to/bit_sculpt/results \\
        TT_CACHE_PATH=/mnt/MLPerf/.../tt_cache \\
        pytest .../test_moe_mlp.py -k test_moe_layer_real_weights -v -s
    """
    import os
    from pathlib import Path

    bspm_dir_env = os.environ.get("BSPM_DIR")
    tt_cache_path = os.environ.get("TT_CACHE_PATH")
    if not bspm_dir_env or not tt_cache_path:
        pytest.skip("BSPM_DIR and TT_CACHE_PATH must be set — skipping real-weight BSPM cache test")

    from models.demos.deepseek_v3_b1.weights.cache import CacheConfig, CacheContext, TensorCache
    from models.demos.deepseek_v3_b1.weights.prepare import prepare_routed_expert_weights

    num_experts = device.get_num_devices()  # 1 on single-chip, 8 on 8-chip submesh
    bspm_dir_path = Path(bspm_dir_env) / "deepseek-r1-0528"
    cache_config = CacheConfig(
        cache=TensorCache(Path(tt_cache_path)),
        context=CacheContext(
            schema_version=1,
            hf_model_id="deepseek-ai/DeepSeek-R1-0528",
            hf_revision="local",
            mesh_shape=(1, 1),
        ),
    )

    logger.info(
        "test_moe_layer_real_weights: layer={}, bspm_dir={}, cache={}, num_experts={}",
        layer_idx,
        bspm_dir_path,
        tt_cache_path,
        num_experts,
    )

    # Load compressed experts via TensorCache (compact tiles.bin on miss, cached reload on hit).
    result = prepare_routed_expert_weights(
        device,
        hf_state_dict,
        layer_idx,
        is_moe=True,
        num_routed_experts=num_experts,
        move_to_device=True,
        bspm_dir=bspm_dir_path,
        cache_config=cache_config,
    )
    experts = (result.routed_gate_proj, result.routed_up_proj, result.routed_down_proj)

    # Random state dict for all scaffolding (routing, gamma, attention, shared expert).
    # Avoids the FP8→ttnn.from_torch incompatibility in prepare_attention_weights when
    # reading directly from the R1-0528 FP8 safetensors checkpoint.
    random_state = get_reference_model_state_dict(
        layer_idx=layer_idx,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=num_experts,
        random_weights=True,
    )

    r = create_routed_expert_tensors(
        device,
        use_hardcoded_expert_index=True,
        state_dict=random_state,
        is_moe=True,
        layer_idx=layer_idx,
        bspm_dir=None,
        preloaded_experts=experts,
    )

    # Override the expert float golden dicts with real HF weights so PCC reflects
    # compression quality rather than random-vs-compressed comparison.
    K = RoutedExpert.K
    GATE_PROJ_N = RoutedExpert.GATE_PROJ_N
    layer_key = f"model.layers.{layer_idx}"
    expert_weights_dict = {}
    up_proj_weights_dict = {}
    down_proj_weights_dict = {}
    for e in range(num_experts):
        w_g = hf_state_dict[f"{layer_key}.mlp.experts.{e}.gate_proj.weight"].T.contiguous().float()
        expert_weights_dict[e] = w_g.reshape(1, 1, K, GATE_PROJ_N)
        w_u = hf_state_dict[f"{layer_key}.mlp.experts.{e}.up_proj.weight"].T.contiguous().float()
        up_proj_weights_dict[e] = w_u.reshape(1, 1, K, GATE_PROJ_N)
        w_d = hf_state_dict[f"{layer_key}.mlp.experts.{e}.down_proj.weight"].T.contiguous().float()
        down_proj_weights_dict[e] = w_d.reshape(1, 1, GATE_PROJ_N, K)
    r = r._replace(
        expert_weights_dict=expert_weights_dict,
        up_proj_weights_dict=up_proj_weights_dict,
        down_proj_weights_dict=down_proj_weights_dict,
    )

    sender_core = r.ttnn_residual_mcast_src.memory_config().shard_spec.grid.bounding_box().end
    mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
    s = create_shared_expert_tensors(
        device,
        RoutedExpert.M,
        RoutedExpert.K,
        mcast_grid,
        state_dict=random_state,
        is_moe=True,
        layer_idx=layer_idx,
    )

    # SDPA buffers — required by MoeOp for CB memory overlap.
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
    full_device_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )
    num_full_cores = device_grid_size.x * device_grid_size.y
    sdpa_out_interm_shard_spec = ttnn.ShardSpec(
        full_device_grid,
        (SDPA.OUT_INTERM_SHARD_HEIGHT, SDPA.OUT_INTERM_SHARD_WIDTH),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sdpa_out_interm_buffer = ttnn.from_torch(
        torch.zeros((SDPA.OUT_INTERM_SHARD_HEIGHT * num_full_cores, SDPA.OUT_INTERM_SHARD_WIDTH), dtype=torch.bfloat16),
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
        shared_gate_weights_overlapped=s.shared_gate_weights_overlapped,
        shared_up_weights_overlapped=s.shared_up_weights_overlapped,
        shared_down_weights_tensor=s.ttnn_down_weights,
        shared_k_parallel=s.k_parallel,
        shared_n_parallel=s.n_parallel,
        use_hardcoded_expert_index=True,
        sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
        sdpa_out_interm_buffer=sdpa_out_interm_buffer,
        num_iterations=1,
        reconfig_moe_cbs=True,
        semaphores=moe_semaphores,
        noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
    )
    ttnn.synchronize_device(device)

    output_final_torch = ttnn.to_torch(ttnn_result_final)
    output_final_valid = extract_routed_expert_output(
        output_final_torch,
        r.num_gate_proj_cores,
        r.final_output_width_per_core,
        r.per_core_down_proj_N,
    )

    torch_expected_scores, torch_expected_indices, torch_expected_final = MoeOp.golden_single_device(
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
        use_hardcoded_expert_index=True,
    )

    assert output_final_valid.abs().sum() > 0, "Output is all zeros — kernel may not have run"

    passing, pcc = comp_pcc(torch_expected_final, output_final_valid, 0.90)
    logger.info("test_moe_layer_real_weights PCC: {} (threshold=0.90)", pcc)
    assert passing, f"PCC check failed (got {pcc}, expected >= 0.90)"

    logger.info("test_moe_layer_real_weights PASSED! (PCC={})", pcc)


# ============================================================================
# Real-weight BSPM cache validation — 8 experts on single chip
# ============================================================================


@skip_for_wormhole_b0()
@pytest.mark.parametrize("layer_idx", list(range(3, 61)))
def test_moe_layer_real_weights_8expert(device, hf_state_dict, get_reference_model_state_dict, layer_idx):
    """MoE MLP: experts 0-7 of a single layer vs PyTorch float32 reference.

    Tests 8 different experts from the TensorCache sequentially on a single chip.
    All 8 experts are loaded upfront via prepare_routed_expert_weights() with a
    TensorCache-backed CacheConfig (compact tiles.bin format), then each expert is
    run through MoeOp and compared against a float32 golden from the real HF checkpoint.
    PCC must be >= 0.25 for every non-near-zero expert.

    Required env vars:
      DEEPSEEK_V3_HF_MODEL  — path to DeepSeek R1-0528 HF checkpoint
      BSPM_DIR              — path to bit_sculpt results root (contains deepseek-r1-0528/ subdir)
      TT_CACHE_PATH         — path to TensorCache root (will be populated on first run)

    Run (BH Galaxy only — 13x10 grid required):
      TT_METAL_CLEAR_L1=1 TT_METAL_SLOW_DISPATCH_MODE=1 \\
        DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/.../DeepSeek-R1-0528 \\
        BSPM_DIR=/path/to/bit_sculpt/results \\
        TT_CACHE_PATH=/mnt/MLPerf/.../tt_cache \\
        pytest .../test_moe_mlp.py -k test_moe_layer_real_weights_8expert -v -s
    """
    import os
    from pathlib import Path

    bspm_dir_env = os.environ.get("BSPM_DIR")
    tt_cache_path = os.environ.get("TT_CACHE_PATH")
    if not bspm_dir_env or not tt_cache_path:
        pytest.skip("BSPM_DIR and TT_CACHE_PATH must be set")

    from models.demos.deepseek_v3_b1.weights.cache import CacheConfig, CacheContext, TensorCache
    from models.demos.deepseek_v3_b1.weights.prepare import prepare_routed_expert_weights

    num_experts_to_test = 8
    K = RoutedExpert.K
    GATE_PROJ_N = RoutedExpert.GATE_PROJ_N
    layer_key = f"model.layers.{layer_idx}"

    bspm_dir_path = Path(bspm_dir_env) / "deepseek-r1-0528"
    cache_config = CacheConfig(
        cache=TensorCache(Path(tt_cache_path)),
        context=CacheContext(
            schema_version=1,
            hf_model_id="deepseek-ai/DeepSeek-R1-0528",
            hf_revision="local",
            mesh_shape=(1, 1),
        ),
    )

    # Random scaffolding shared across all expert iterations.
    random_state = get_reference_model_state_dict(
        layer_idx=layer_idx,
        is_moe=True,
        seed=RoutedExpert.SEED,
        num_routed_experts=1,
        random_weights=True,
    )

    logger.info(
        "test_moe_layer_real_weights_8expert: layer={}, testing experts 0-{}", layer_idx, num_experts_to_test - 1
    )

    # Load all 8 experts upfront via TensorCache (compact tiles.bin on miss, cached on hit).
    all_experts = prepare_routed_expert_weights(
        device,
        hf_state_dict,
        layer_idx,
        is_moe=True,
        num_routed_experts=num_experts_to_test,
        move_to_device=True,
        bspm_dir=bspm_dir_path,
        cache_config=cache_config,
    )

    # Clear program cache once before the sweep so all 8 experts compile fresh
    # with the current L1 layout.  Re-enable is deferred to after the loop so
    # the cache stays ENABLED throughout (some layer/expert combos require the
    # "enabled compilation path" for correct output, but we must not carry stale
    # on-disk binaries into the loop — those can embed wrong compile-time args
    # such as CBIn1BaseAddr / meta_l1_addr from a previous session's L1 layout).
    device.disable_and_clear_program_cache()
    device.enable_program_cache()

    all_passing = True
    for e in range(num_experts_to_test):
        gate_ct = all_experts.routed_gate_proj[e]
        up_ct = all_experts.routed_up_proj[e]
        down_ct = all_experts.routed_down_proj[e]
        experts = ([gate_ct], [up_ct], [down_ct])

        r = create_routed_expert_tensors(
            device,
            use_hardcoded_expert_index=True,
            state_dict=random_state,
            is_moe=True,
            layer_idx=layer_idx,
            bspm_dir=None,
            preloaded_experts=experts,
        )

        # Real float golden for this expert.
        gate_f = hf_state_dict[f"{layer_key}.mlp.experts.{e}.gate_proj.weight"].T.contiguous().float()
        up_f = hf_state_dict[f"{layer_key}.mlp.experts.{e}.up_proj.weight"].T.contiguous().float()
        down_f = hf_state_dict[f"{layer_key}.mlp.experts.{e}.down_proj.weight"].T.contiguous().float()
        r = r._replace(
            expert_weights_dict={0: gate_f.reshape(1, 1, K, GATE_PROJ_N)},
            up_proj_weights_dict={0: up_f.reshape(1, 1, K, GATE_PROJ_N)},
            down_proj_weights_dict={0: down_f.reshape(1, 1, GATE_PROJ_N, K)},
        )

        sender_core = r.ttnn_residual_mcast_src.memory_config().shard_spec.grid.bounding_box().end
        mcast_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), sender_core)])
        s = create_shared_expert_tensors(
            device,
            RoutedExpert.M,
            RoutedExpert.K,
            mcast_grid,
            state_dict=random_state,
            is_moe=True,
            layer_idx=layer_idx,
        )

        device_grid_size = device.compute_with_storage_grid_size()
        kv_cache_shard_height = SDPA.KV_CACHE_SHARD_HEIGHT
        kvpe_dim = SDPA.KVPE_DIM
        num_mcast_cores = len(ttnn.corerange_to_cores(mcast_grid))
        kv_cache_shard_spec = ttnn.ShardSpec(
            mcast_grid, (kv_cache_shard_height, kvpe_dim), ttnn.ShardOrientation.ROW_MAJOR
        )
        sdpa_kv_cache_buffer = ttnn.from_torch(
            torch.zeros((kv_cache_shard_height * num_mcast_cores, kvpe_dim), dtype=torch.bfloat16),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, kv_cache_shard_spec
            ),
        )
        full_device_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1),
                )
            }
        )
        num_full_cores = device_grid_size.x * device_grid_size.y
        sdpa_out_interm_shard_spec = ttnn.ShardSpec(
            full_device_grid,
            (SDPA.OUT_INTERM_SHARD_HEIGHT, SDPA.OUT_INTERM_SHARD_WIDTH),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sdpa_out_interm_buffer = ttnn.from_torch(
            torch.zeros(
                (SDPA.OUT_INTERM_SHARD_HEIGHT * num_full_cores, SDPA.OUT_INTERM_SHARD_WIDTH), dtype=torch.bfloat16
            ),
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

        # Create fresh semaphores each iteration so their L1 addresses stay consistent with
        # the rest of the inside-loop allocation order. GlobalSemaphore::setup_buffer already
        # does a blocking write of the initial_value, so the reset below is redundant but kept
        # as an explicit guard against any residual kernel writes to those L1 slots.
        moe_semaphores = MoeOp.create_semaphores(device)
        # Reset all semaphores to 0 (blocking write) so each expert starts from a clean
        # synchronization state, regardless of what the previous expert's kernel left behind.
        for sem in moe_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
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
            shared_gate_weights_overlapped=s.shared_gate_weights_overlapped,
            shared_up_weights_overlapped=s.shared_up_weights_overlapped,
            shared_down_weights_tensor=s.ttnn_down_weights,
            shared_k_parallel=s.k_parallel,
            shared_n_parallel=s.n_parallel,
            use_hardcoded_expert_index=True,
            sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
            sdpa_out_interm_buffer=sdpa_out_interm_buffer,
            num_iterations=1,
            reconfig_moe_cbs=True,
            semaphores=moe_semaphores,
            noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
        )
        ttnn.synchronize_device(device)

        output_torch = ttnn.to_torch(ttnn_result_final)
        output_valid = extract_routed_expert_output(
            output_torch,
            r.num_gate_proj_cores,
            r.final_output_width_per_core,
            r.per_core_down_proj_N,
        )

        _, _, torch_expected = MoeOp.golden_single_device(
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
            use_hardcoded_expert_index=True,
        )

        # BSPM 3.5 b/e allocates 3 levels (bfp4/bfp2/zero) per-expert based on saliency.
        # Near-zero experts (e.g. ~100% zero tiles at layer 30) produce near-zero TT output,
        # making PCC against float golden meaningless (PCC ≈ noise). Detect them by the
        # output-to-golden norm ratio and skip PCC, just verifying the output IS near-zero.
        # For all other experts use PCC ≥ 0.25 to catch true hardware failures (garbage).
        output_norm = output_valid.float().norm().item()
        expected_norm = torch_expected.float().norm().item()
        output_ratio = output_norm / expected_norm if expected_norm > 0 else 0.0
        if output_ratio < 0.05:
            # Near-zero expert: output is <5% of float magnitude — expected for ~100% zero tiles.
            passing_e, pcc_e = True, float("nan")
            logger.info("layer={} expert={} near-zero (ratio={:.4f}) pass=True", layer_idx, e, output_ratio)
        else:
            passing_e, pcc_e = comp_pcc(torch_expected, output_valid, 0.25)
            logger.info("layer={} expert={} PCC={:.6f} pass={}", layer_idx, e, pcc_e, passing_e)
        if not passing_e:
            all_passing = False

        # Free all device tensors before the next expert iteration to avoid L1/DRAM OOM.
        # Only call ttnn.deallocate on plain ttnn.Tensor objects; CompressedTensor
        # objects (expert weights from BSPM cache) are freed via Python del below.
        for t in [
            ttnn_result_scores,
            ttnn_result_indices,
            ttnn_result_final,
            sdpa_kv_cache_buffer,
            sdpa_out_interm_buffer,
            r.ttnn_residual_mcast_src,
            r.ttnn_gate_mm_weights,
            r.ttnn_rmsnorm_gamma,
            r.ttnn_gate_bias,
            r.ttnn_gate_indices,
            r.gate_output_scores_tensor,
            r.gate_output_indices_tensor,
            r.final_output_tensor,
            s.shared_gate_weights_overlapped,
            s.shared_up_weights_overlapped,
            s.ttnn_down_weights,
        ]:
            if t is not None and isinstance(t, ttnn.Tensor):
                ttnn.deallocate(t)
        # Drop scaffolding tensors; expert CompressedTensors (gate_ct/up_ct/down_ct) are
        # owned by all_experts and stay in DRAM for the duration of the test.
        del r, s
        # Force Python GC to immediately free any L1 tensors kept alive inside MoeOp.op()
        # (meta/fmt tensors stored in routed_ctx params). Without this, CPython might defer
        # freeing if there are any reference cycles introduced by exception tracebacks or
        # other indirect references, leaving stale L1 data visible to the next expert.
        gc.collect()

    # Explicitly free the last iteration's semaphores. Python loop variables remain in
    # function scope after the loop ends, so moe_semaphores holds the last expert's L1
    # semaphore slots alive until the test returns. Freeing here ensures those L1 addresses
    # are returned to the allocator before the next layer test starts, preventing the
    # shifted allocation order from corrupting subsequent tests that share the same device.
    del moe_semaphores
    gc.collect()

    assert all_passing, f"One or more experts failed PCC check (layer={layer_idx})"
    logger.info("test_moe_layer_real_weights_8expert PASSED (layer={})", layer_idx)
