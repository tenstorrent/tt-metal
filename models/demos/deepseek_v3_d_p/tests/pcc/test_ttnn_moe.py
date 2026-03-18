"""
PCC test for TtMinimalMoe module.

Tests that TTNN TtMinimalMoe produces matching outputs to TorchMinimalMoE reference.
This validates the full MoE pipeline: dispatch → routed experts → combine → split → add shared.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.demos.deepseek_v3_d_p.tests.pcc.test_moe import TorchMinimalMoE
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    create_fabric_router_config,
    extract_mesh_config,
    get_combine_output_mesh_composer,
    get_dispatch_output_mesh_composer,
    get_gate_outputs,
    get_routed_expert_output_mesh_composer,
    get_tp_mesh_composer,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_minimal_moe import TtMinimalMoe
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    log_combine_mismatch_details,
    log_per_chip_statistics,
    validate_combine_output,
    validate_dispatch_buffer,
    validate_dispatch_buffer_pcc,
    validate_dispatch_metadata,
)
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_validation_results
from tests.ttnn.utils_for_testing import comp_pcc


def create_torch_expert_weights(
    num_experts: int,
    emb_dim: int,
    hidden_dim: int,
    seed: int = 42,
) -> list[dict]:
    """
    Create random weights for torch experts.

    Args:
        num_experts: Number of experts to create weights for
        emb_dim: Embedding dimension
        hidden_dim: Hidden/intermediate dimension
        seed: Random seed

    Returns:
        List of dicts with gate_proj, up_proj, down_proj per expert
    """
    torch.manual_seed(seed)
    weights_list = []
    for _ in range(num_experts):
        weights = {
            "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
            "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
        }
        weights_list.append(weights)
    return weights_list


def create_shared_expert_weights(
    emb_dim: int,
    hidden_dim: int,
    seed: int = 123,
) -> tuple[dict, dict]:
    """
    Create random weights for shared expert in both formats.

    Returns:
        Tuple of (torch_weights, ttnn_weights):
        - torch_weights: HF format (out_features, in_features) for TorchExpert
        - ttnn_weights: TTNN format (in_features, out_features) for TtSharedExpert

    Args:
        emb_dim: Embedding dimension
        hidden_dim: Hidden/intermediate dimension
        seed: Random seed

    Returns:
        Tuple of (torch_weights, ttnn_weights)
    """
    torch.manual_seed(seed)

    # HF format: (out_features, in_features)
    torch_weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
    }

    # TTNN format: (in_features, out_features) - transpose of HF
    ttnn_weights = {
        "gate_proj": torch_weights["gate_proj"].T.contiguous(),
        "up_proj": torch_weights["up_proj"].T.contiguous(),
        "down_proj": torch_weights["down_proj"].T.contiguous(),
    }

    return torch_weights, ttnn_weights


@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, hidden_dim, num_routed_experts, num_experts_per_tok, capacity_factor",
    [
        # Smaller config to fit in L1 memory
        # emb_dim must be divisible by TP factor (4) and tile size (32)
        # seq_len must fit in L1 with emb_dim after sharding
        (256, 2048, 7 * 1024, 16, 4, 2),
    ],
    ids=["small-config"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="linear"),
            id="linear-8",
        ),
        # 2D mesh configurations - dispatch along axis 0, shared expert CCL along axis 1
        pytest.param(
            (2, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-4x2"),
            id="mesh-2x4",
        ),
        pytest.param(
            (4, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="mesh-4x2",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_ttnn_moe(
    mesh_device,
    device_params,
    seq_len_per_chip,
    emb_dim,
    hidden_dim,
    num_routed_experts,
    num_experts_per_tok,
    capacity_factor,
    num_links,
    topology,
):
    """
    Test TtMinimalMoe PCC against TorchMinimalMoE reference.

    This test verifies the full MoE pipeline:
    1. Token dispatch to expert buffers
    2. Routed expert FFN computation
    3. Expert output combining
    4. Split connection (weighted sum)
    5. Addition with shared expert output
    """
    num_devices = mesh_device.get_num_devices()

    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    logger.debug(f"\n{'='*60}")
    logger.debug("TtMinimalMoe PCC Test")
    logger.debug(f"{'='*60}")
    logger.debug(f"mesh_shape={mesh_device.shape}, num_devices={num_devices}")
    logger.debug(f"dispatch_group_size={dispatch_group_size}, num_dispatch_groups={num_dispatch_groups}")

    signpost(
        f"TtMinimalMoe PCC test - mesh {mesh_device.shape}, seq_len={seq_len_per_chip}, "
        f"emb_dim={emb_dim}, experts={num_routed_experts}"
    )

    # Compute configuration constants
    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, num_routed_experts, num_experts_per_tok, num_devices, dispatch_group_size, capacity_factor
    )
    logger.debug(f"experts_per_chip={experts_per_chip}, metadata_len={metadata_len}")
    logger.debug(f"max_dispatched_tokens_per_expert={max_dispatched_tokens_per_expert}")

    total_experts = num_routed_experts

    # ========================================
    # Step 1: Create weights for both torch and TTNN
    # ========================================
    logger.debug("Creating expert weights...")
    all_routed_weights = create_torch_expert_weights(total_experts, emb_dim, hidden_dim, seed=42)
    shared_weights_torch, shared_weights_ttnn = create_shared_expert_weights(emb_dim, hidden_dim, seed=123)

    # Pass all routed expert weights - TtRoutedExpert distributes them across devices
    # (each device gets unique expert weights, not replicated)
    ttnn_routed_weights = all_routed_weights

    # ========================================
    # Step 2: Generate test inputs
    # ========================================
    logger.debug("Generating test inputs...")
    x, weights, indices = initialize_test_inputs(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=emb_dim,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seed=42,
        num_dispatch_groups=num_dispatch_groups,
    )
    logger.debug(f"Input shapes: x={x.shape}, weights={weights.shape}, indices={indices.shape}")

    # Compute gate outputs (offsets and token counts)
    expert_offsets, expert_token_counts, _ = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
    )

    # Create expert dispatch table
    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )

    # ========================================
    # Step 3: Run TorchMinimalMoE reference with intermediates
    # ========================================
    logger.debug("Running TorchMinimalMoE reference...")
    torch_moe = TorchMinimalMoE(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=emb_dim,
        expert_dispatch_table=expert_dispatch_table,
        num_dispatch_groups=num_dispatch_groups,
    )

    # Override weights with our controlled random weights
    for i, expert in enumerate(torch_moe.routed_experts):
        expert.gate_proj = torch.nn.Parameter(all_routed_weights[i]["gate_proj"].float())
        expert.up_proj = torch.nn.Parameter(all_routed_weights[i]["up_proj"].float())
        expert.down_proj = torch.nn.Parameter(all_routed_weights[i]["down_proj"].float())

    torch_moe.shared_expert_module.expert.gate_proj = torch.nn.Parameter(shared_weights_torch["gate_proj"].float())
    torch_moe.shared_expert_module.expert.up_proj = torch.nn.Parameter(shared_weights_torch["up_proj"].float())
    torch_moe.shared_expert_module.expert.down_proj = torch.nn.Parameter(shared_weights_torch["down_proj"].float())

    torch_output, torch_intermediates = torch_moe(
        x.float(),
        weights.float(),
        indices,
        expert_offsets,
        expert_token_counts,
        return_intermediates=True,
    )
    logger.debug(f"Torch output shape: {torch_output.shape}")
    logger.debug(f"Torch output stats - min: {torch_output.min():.4f}, max: {torch_output.max():.4f}")

    # ========================================
    # Step 4: Create TTNN tensors
    # ========================================
    logger.debug("Creating TTNN tensors...")

    # For 2D mesh: shard x along dispatch_group_size (dim 0) across axis 0 AND emb_dim (dim -1) across axis 1
    # This supports both dispatch (SP along axis 0) and shared expert (TP along axis 1)
    mesh_mapper_2d_input = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, -1),  # Shard dim 0 across axis 0, shard dim -1 across axis 1
    )

    tt_x = ttnn.from_torch(
        x, mesh_mapper=mesh_mapper_2d_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    logger.debug(f"tt_x.shape: {tt_x.shape}")

    # Weights and indices: shard on dispatch axis, replicate on TP axis
    mesh_mapper_sp_only = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, None),  # Shard dim 0 across axis 0, replicate on axis 1
    )

    tt_weights = ttnn.from_torch(
        weights,
        mesh_mapper=mesh_mapper_sp_only,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    tt_indices = ttnn.from_torch(
        indices, mesh_mapper=mesh_mapper_sp_only, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.int32
    )

    # Expert offsets and dispatch table
    tt_expert_offsets = TtDispatchModule.shard_expert_offsets(mesh_device, expert_offsets)
    tt_expert_dispatch_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, expert_dispatch_table, sp_axis)

    # Expert token counts
    mesh_mapper_2d = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(1, 0),  # Shard tensor dim 1 across mesh rows, tensor dim 0 across mesh cols
    )
    tt_expert_token_counts = ttnn.from_torch(
        expert_token_counts,
        mesh_mapper=mesh_mapper_2d,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    # ========================================
    # Step 5: Create TtMinimalMoe and run forward
    # ========================================
    logger.debug("Creating TtMinimalMoe...")
    tt_moe = TtMinimalMoe(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
        routed_expert_weights=ttnn_routed_weights,
        shared_expert_weights=shared_weights_ttnn,
        # activations_dtype=ttnn.bfloat8_b,
        # weights_dtype=ttnn.bfloat4_b,
        activations_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
    )

    logger.debug("Running TtMinimalMoe forward pass...")
    tt_output, tt_intermediates = tt_moe(
        tt_x,
        tt_weights,
        tt_indices,
        tt_expert_offsets,
        tt_expert_dispatch_table,
        tt_expert_token_counts,
        return_intermediates=True,
    )
    ttnn.synchronize_device(mesh_device)
    logger.debug(f"TTNN output shape: {tt_output.shape}")

    # ========================================
    # Step 6: Compare intermediates
    # ========================================
    logger.debug(f"\n{'='*60}")
    logger.debug("Comparing intermediate outputs...")
    logger.debug(f"{'='*60}")

    all_passed = True
    validation_results = []

    # Dense tensor checks with PCC (shared_output, final_output)
    # fmt: off
    dense_checks = [
        ("shared_output", tt_intermediates.shared_output, torch_intermediates.shared_output, get_tp_mesh_composer(mesh_device), 0.97),
        ("final_output", tt_output, torch_output, get_tp_mesh_composer(mesh_device), 0.95),
    ]
    # fmt: on

    for name, tt_tensor, torch_tensor, composer, threshold in dense_checks:
        if tt_tensor is None:
            logger.warning(f"[{name}] SKIPPED - TTNN tensor is None (component not yet enabled)")
            continue
        # Convert TTNN to torch
        logger.debug(f"📊 {name} {tt_tensor.shape=} {torch_tensor.shape=}")
        tt_host = ttnn.to_torch(tt_tensor, mesh_composer=composer, dtype=torch.bfloat16)
        logger.debug(f"📊 {name} {tt_tensor.shape=} {tt_host.shape=} {torch_tensor.shape=}")

        # Check shapes match
        if tt_host.shape != torch_tensor.shape:
            logger.error(f"[{name}] FAILED - Shape mismatch: TTNN {tt_host.shape} vs Torch {torch_tensor.shape}")
            all_passed = False
            continue

        # Check for NaN/Inf
        if torch.isnan(tt_host).any():
            logger.error(f"[{name}] FAILED - TTNN tensor contains NaN")
            all_passed = False
            continue
        if torch.isinf(tt_host).any():
            logger.error(f"[{name}] FAILED - TTNN tensor contains Inf")
            all_passed = False
            continue

        # Compute PCC
        _, pcc = comp_pcc(torch_tensor.float(), tt_host.float())
        if pcc >= threshold:
            logger.info(f"[{name}] PASSED - PCC: {pcc:.6f} (threshold: {threshold})")
        else:
            logger.error(f"[{name}] FAILED - PCC: {pcc:.6f} below threshold {threshold}")
            all_passed = False

    # Sparse tensor validation for dispatched_buffer
    # This is sparse: only valid data up to expert_token_counts per (dispatch_group, chip, expert)
    # Using validate_dispatch_buffer for exact match validation (dispatch doesn't modify data)
    if tt_intermediates.dispatched_buffer is not None:
        name = "dispatched_buffer"
        tt_tensor = tt_intermediates.dispatched_buffer
        torch_tensor = torch_intermediates.dispatched_buffer
        composer = get_dispatch_output_mesh_composer(mesh_device)

        logger.debug(f"📊 {name} {tt_tensor.shape=} {torch_tensor.shape=}")
        tt_host = ttnn.to_torch(tt_tensor, mesh_composer=composer, dtype=torch.bfloat16)
        logger.debug(f"📊 {name} {tt_tensor.shape=} {tt_host.shape=} {torch_tensor.shape=}")

        # Validate using sparse-aware comparison (only compares valid slots)
        result = validate_dispatch_buffer(
            torch_tensor.to(torch.bfloat16),
            tt_host,
            expert_token_counts,
            expert_dispatch_table,
            num_dispatch_groups,
            dispatch_group_size,
            experts_per_chip,
            verbose=True,
        )
        result.name = name
        validation_results.append(result)

        if result.passed:
            logger.info(f"[{name}] PASSED - {result.matches}/{result.total} slots matched")
        else:
            logger.error(f"[{name}] FAILED - {result.matches}/{result.total} slots matched")
            result.log_mismatches(limit=5)
            all_passed = False
    else:
        logger.warning("[dispatched_buffer] SKIPPED - TTNN tensor is None (component not yet enabled)")

    if tt_intermediates.metadata is not None:
        name = "dispatch_metadata"
        tt_tensor = tt_intermediates.metadata
        torch_tensor = torch_intermediates.metadata
        composer = get_dispatch_output_mesh_composer(mesh_device)

        logger.debug(f"📊 {name} {tt_tensor.shape=} {torch_tensor.shape=}")
        tt_host = ttnn.to_torch(tt_tensor, mesh_composer=composer)
        logger.debug(f"📊 {name} {tt_tensor.shape=} {tt_host.shape=} {torch_tensor.shape=}")

        # Validate using sparse-aware comparison (only compares valid slots)
        result = validate_dispatch_metadata(
            torch_tensor,
            tt_host,
            expert_token_counts,
            expert_dispatch_table,
            num_dispatch_groups,
            dispatch_group_size,
            experts_per_chip,
            verbose=True,
        )
        result.name = name
        validation_results.append(result)

        if result.passed:
            logger.info(f"[{name}] PASSED - {result.matches}/{result.total} slots matched")
        else:
            logger.error(f"[{name}] FAILED - {result.matches}/{result.total} slots matched")
            result.log_mismatches(limit=5)
            all_passed = False
    else:
        logger.warning("[dispatched_buffer] SKIPPED - TTNN tensor is None (component not yet enabled)")

    # Sparse PCC validation for expert_outputs
    # This is sparse and has numerical differences from quantized matmul, so use PCC not allclose
    if tt_intermediates.expert_outputs is not None:
        name = "expert_outputs"
        tt_tensor = tt_intermediates.expert_outputs
        torch_tensor = torch_intermediates.expert_outputs
        composer = get_routed_expert_output_mesh_composer(mesh_device)

        logger.debug(f"📊 {name} {tt_tensor.shape=} {torch_tensor.shape=}")
        tt_host = ttnn.to_torch(tt_tensor, mesh_composer=composer, dtype=torch.bfloat16)
        logger.debug(f"📊 {name} {tt_tensor.shape=} {tt_host.shape=} {torch_tensor.shape=}")

        # Validate using sparse-aware PCC comparison (only compares valid slots)
        result = validate_dispatch_buffer_pcc(
            torch_tensor.to(torch.bfloat16),
            tt_host,
            expert_token_counts,
            expert_dispatch_table,
            num_dispatch_groups,
            dispatch_group_size,
            experts_per_chip,
            pcc_threshold=0.93,
            verbose=True,
        )
        result.name = name
        validation_results.append(result)

        if result.passed:
            logger.info(f"[{name}] PASSED - {result.matches}/{result.total} slots matched (PCC >= 0.93)")
        else:
            logger.error(f"[{name}] FAILED - {result.matches}/{result.total} slots matched")
            result.log_mismatches(limit=5)
            all_passed = False
    else:
        logger.warning("[expert_outputs] SKIPPED - TTNN tensor is None (component not yet enabled)")

    # Validate combined_output (before reduce step)
    if tt_intermediates.combined_output is not None:
        name = "combined_output"
        logger.debug(f"📊 {name} tt shape: {tt_intermediates.combined_output.shape}")
        logger.debug(f"📊 {name} torch shape: {torch_intermediates.combined_output.shape}")

        # Convert TTNN combined_output to torch
        combine_mesh_composer = get_combine_output_mesh_composer(mesh_device)
        tt_combined_torch = ttnn.to_torch(
            tt_intermediates.combined_output,
            mesh_composer=combine_mesh_composer,
            dtype=torch.bfloat16,
        )
        logger.debug(f"📊 {name} tt_combined_torch shape: {tt_combined_torch.shape}")

        # Validate using EP-rank aware comparison with PCC
        combine_pcc = 0.997
        combine_result = validate_combine_output(
            torch_intermediates.combined_output,
            tt_combined_torch,
            indices,
            num_dispatch_groups,
            num_routed_experts,
            # use_pcc=False,
            use_pcc=True,
            pcc_threshold=combine_pcc,
            verbose=True,
            expert_dispatch_table=expert_dispatch_table,
            expert_token_counts=expert_token_counts,
            experts_per_chip=experts_per_chip,
        )

        log_validation_results(
            results=[combine_result],
            num_dispatch_groups=num_dispatch_groups,
            dispatch_group_size=dispatch_group_size,
            title="Combined Output Validation",
        )

        if combine_result.passed:
            logger.info(
                f"[{name}] PASSED - {combine_result.matches}/{combine_result.total} slots matched (PCC >= {combine_pcc})"
            )
        else:
            logger.error(f"[{name}] FAILED - {combine_result.matches}/{combine_result.total} slots matched")
            log_combine_mismatch_details(
                combine_result.mismatches, torch_intermediates.combined_output, tt_combined_torch, use_pcc=True
            )
            log_per_chip_statistics(
                combine_result.mismatches, dispatch_group_size, seq_len_per_chip, num_experts_per_tok
            )
            all_passed = False
    else:
        logger.warning("[combined_output] SKIPPED - TTNN tensor is None")

    # routed_output is a dense tensor (after reduce over topk), use standard PCC
    if tt_intermediates.routed_output is not None:
        name = "routed_output"
        pcc_threshold = 0.90

        logger.debug(f"📊 {name} {tt_intermediates.routed_output.shape=} {torch_intermediates.routed_output.shape=}")
        tt_routed = ttnn.to_torch(
            tt_intermediates.routed_output,
            mesh_composer=get_tp_mesh_composer(mesh_device),
            dtype=torch.bfloat16,
        )
        logger.debug(f"📊 {name} {tt_routed.shape=} {torch_intermediates.routed_output.shape=}")

        # Check shapes match
        if tt_routed.shape != torch_intermediates.routed_output.shape:
            logger.error(
                f"[{name}] FAILED - Shape mismatch: TTNN {tt_routed.shape} vs Torch {torch_intermediates.routed_output.shape}"
            )
            all_passed = False
        else:
            # Check for NaN/Inf
            if torch.isnan(tt_routed).any():
                logger.error(f"[{name}] FAILED - TTNN tensor contains NaN")
                all_passed = False
            elif torch.isinf(tt_routed).any():
                logger.error(f"[{name}] FAILED - TTNN tensor contains Inf")
                all_passed = False
            else:
                _, pcc = comp_pcc(torch_intermediates.routed_output.float(), tt_routed.float())
                if pcc >= pcc_threshold:
                    logger.info(f"[{name}] PASSED - PCC: {pcc:.6f} (threshold: {pcc_threshold})")
                else:
                    logger.error(f"[{name}] FAILED - PCC: {pcc:.6f} below threshold {pcc_threshold}")
                    all_passed = False
    else:
        logger.warning("[routed_output] SKIPPED - TTNN tensor is None (component not yet enabled)")

    # Log validation summary
    if validation_results:
        log_validation_results(
            results=validation_results,
            num_dispatch_groups=num_dispatch_groups,
            dispatch_group_size=dispatch_group_size,
            title="Sparse Tensor Validation Results",
        )

    # # ========================================
    # # Step 7: Compare final output
    # # ========================================
    # logger.debug(f"\n{'='*60}")
    # logger.debug("Comparing final output...")
    # logger.debug(f"{'='*60}")

    # tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=mesh_composer_2d, dtype=torch.bfloat16)
    # logger.debug(f"TTNN final output shape: {tt_output_torch.shape}")
    # logger.debug(f"Torch final output shape: {torch_output.shape}")
    # logger.debug(f"TTNN output stats - min: {tt_output_torch.min():.4f}, max: {tt_output_torch.max():.4f}")
    # logger.debug(f"Torch output stats - min: {torch_output.min():.4f}, max: {torch_output.max():.4f}")

    # # Check for NaN/Inf
    # assert not torch.isnan(tt_output_torch).any(), "TTNN output contains NaN"
    # assert not torch.isinf(tt_output_torch).any(), "TTNN output contains Inf"

    # # Shape check
    # assert (
    #     tt_output_torch.shape == torch_output.shape
    # ), f"Shape mismatch: TTNN {tt_output_torch.shape} vs Torch {torch_output.shape}"

    # # Compute PCC for final output
    # # Note: With only shared expert enabled, this won't match torch's full pipeline output
    # # The comparison will show the delta caused by missing routed expert path
    # _, final_pcc = comp_pcc(torch_output, tt_output_torch.float())
    # logger.debug(f"Final output PCC: {final_pcc:.6f}")
    logger.debug("Note: Final PCC expected to be low until full pipeline is enabled")

    # Assert intermediate checks passed
    assert all_passed, "One or more intermediate comparisons failed"

    logger.debug(f"\n{'='*60}")
    logger.debug("TtMinimalMoe PCC Test PASSED!")
    logger.debug(f"{'='*60}")
