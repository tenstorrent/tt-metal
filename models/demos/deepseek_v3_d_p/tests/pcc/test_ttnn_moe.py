"""
PCC test for TtMoe module.

Tests that TTNN TtMoe produces matching outputs to TorchMoe reference.
This validates the full MoE pipeline: dispatch → routed experts → combine → split → add shared.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_blackhole, profiler
from models.demos.deepseek_v3_d_p.reference.tt.moe.moe import TorchMoe
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    create_fabric_router_config,
    extract_mesh_config,
    get_ep_mesh_composer,
    get_gate_outputs,
    get_tp_mesh_composer,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
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
    "seq_len_per_chip, emb_dim, hidden_dim, num_routed_experts, num_experts_per_tok, capacity_factor, run_pcc_check",
    [
        # Smaller config to fit in L1 memory
        # emb_dim must be divisible by TP factor (4) and tile size (32)
        # seq_len must fit in L1 with emb_dim after sharding
        # (3200, 2048, 7 * 1024, 64, 2, 2, False),  # Profiling mode
        (3200, 2048, 7 * 1024, 64, 2, 2, False),  # PCC validation mode
        (3200, 2048, 7 * 1024, 64, 2, 2, True),  # PCC validation mode
    ],
    # ids=["small-config"],
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
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
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
    run_pcc_check,
    num_links,
    topology,
):
    """
    Test TtMoe PCC against TorchMoe reference.

    This test verifies the full MoE pipeline:
    1. Token dispatch to expert buffers
    2. Routed expert FFN computation
    3. Expert output combining
    4. Split connection (weighted sum)
    5. Addition with shared expert output
    """
    profiler.clear()
    profiler.start("test_ttnn_moe")

    num_devices = mesh_device.get_num_devices()

    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    logger.debug(f"\n{'='*60}")
    logger.debug("TtMoe PCC Test")
    logger.debug(f"{'='*60}")
    logger.debug(f"mesh_shape={mesh_device.shape}, num_devices={num_devices}")
    logger.debug(f"dispatch_group_size={dispatch_group_size}, num_dispatch_groups={num_dispatch_groups}")

    signpost(
        f"TtMoe PCC test - mesh {mesh_device.shape}, seq_len={seq_len_per_chip}, "
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
    if run_pcc_check:
        profiler.start("torch_weights_creation")
        logger.debug("Creating expert weights...")
        all_routed_weights = create_torch_expert_weights(total_experts, emb_dim, hidden_dim, seed=42)
        shared_weights_torch, shared_weights_ttnn = create_shared_expert_weights(emb_dim, hidden_dim, seed=123)
        # Pass all routed expert weights - TtRoutedExpert distributes them across devices
        # (each device gets unique expert weights, not replicated)
        ttnn_routed_weights = all_routed_weights
        profiler.end("torch_weights_creation")
    else:
        # When run_pcc_check=False, pass None to use internal random weight allocation
        all_routed_weights = None
        shared_weights_torch = None
        shared_weights_ttnn = None
        ttnn_routed_weights = None

    # ========================================
    # Step 2: Generate test inputs
    # ========================================
    profiler.start("torch_input_creation")
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
        skip_x_initialization=not run_pcc_check,  # Skip x initialization when not validating to save time
    )
    logger.debug(
        f"Input shapes: x={x.shape if x is not None else None}, weights={weights.shape}, indices={indices.shape}"
    )

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
    profiler.end("torch_input_creation")

    # ========================================
    # Step 3: Run TorchMoe reference with intermediates
    # ========================================
    if run_pcc_check:
        profiler.start("torch_forward")
        logger.debug("Running TorchMoe reference...")
        torch_moe = TorchMoe(
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
            routed_expert_weights=all_routed_weights,
            shared_expert_weights=shared_weights_torch,
        )

        torch_output, torch_intermediates = torch_moe(
            x.float(),
            weights.float(),
            indices,
            expert_offsets,
            expert_token_counts,
            return_intermediates=True,
        )
        profiler.end("torch_forward")
        logger.debug(f"Torch output shape: {torch_output.shape}")
        logger.debug(f"Torch output stats - min: {torch_output.min():.4f}, max: {torch_output.max():.4f}")

    # ========================================
    # Step 4: Create TTNN tensors
    # ========================================
    profiler.start("ttnn_input_creation")
    logger.debug("Creating TTNN tensors...")

    mesh_rows, mesh_cols = mesh_device.shape

    # For 2D mesh: shard x along dispatch_group_size (dim 0) across axis 0 AND emb_dim (dim -1) across axis 1
    # This supports both dispatch (SP along axis 0) and shared expert (TP along axis 1)
    if run_pcc_check:
        mesh_mapper_2d_input = ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(0, -1),  # Shard dim 0 across axis 0, shard dim -1 across axis 1
        )
        tt_x = ttnn.from_torch(
            x, mesh_mapper=mesh_mapper_2d_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
        )
    else:
        # Device-only allocation for x (large tensor) - no host-to-device transfer
        per_device_x_shape = (dispatch_group_size // mesh_rows, seq_len_per_chip, emb_dim // mesh_cols)
        tt_x = ttnn.empty(per_device_x_shape, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
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
    ttnn.synchronize_device(mesh_device)
    profiler.end("ttnn_input_creation")

    # ========================================
    # Step 5: Create TtMoe and run forward
    # ========================================
    profiler.start("tt_moe_creation")
    logger.debug("Creating TtMoe...")
    tt_moe = TtMoe(
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
        combine_output_buffer_memory_config=ttnn.L1_MEMORY_CONFIG if is_blackhole() else ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_moe_creation")

    profiler.start("tt_forward")
    logger.debug("Running TtMoe forward pass...")
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
    profiler.end("tt_forward")
    logger.debug(f"TTNN output shape: {tt_output.shape}")

    # Early return when run_pcc_check=False (profiling mode)
    if not run_pcc_check:
        profiler.end("test_ttnn_moe")
        logger.debug("run_pcc_check=False, skipping PCC validation")
        for key in profiler.times:
            logger.debug(f"{key}: {profiler.get(key) * 1000:.2f} ms")
        return

    # ========================================
    # Step 6: Compare intermediates
    # ========================================
    profiler.start("pcc_validation")
    logger.debug(f"\n{'='*60}")
    logger.debug("Comparing intermediate outputs...")
    logger.debug(f"{'='*60}")

    all_passed = True
    validation_results = []

    # Dense tensor checks with PCC
    # fmt: off
    dense_checks = [
        ("shared_output", tt_intermediates.shared_output, torch_intermediates.shared_output, get_tp_mesh_composer(mesh_device), 0.97),
        ("routed_output", tt_intermediates.routed_output, torch_intermediates.routed_output, get_tp_mesh_composer(mesh_device), 0.90),
        ("final_output", tt_output, torch_output, get_tp_mesh_composer(mesh_device), 0.95),
    ]
    # fmt: on

    for name, tt_tensor, torch_tensor, composer, threshold in dense_checks:
        if tt_tensor is None:
            logger.warning(f"[{name}] validation SKIPPED")
            continue
        tt_host = ttnn.to_torch(tt_tensor, mesh_composer=composer, dtype=torch.bfloat16)
        _, pcc = comp_pcc(torch_tensor.float(), tt_host.float())
        if pcc >= threshold:
            logger.info(f"[{name}] PASSED - PCC: {pcc:.6f} (threshold: {threshold})")
        else:
            logger.error(f"[{name}] FAILED - PCC: {pcc:.6f} below threshold {threshold}")
            all_passed = False

    # Sparse tensor validation using slot-aware comparisons
    # fmt: off
    sparse_checks = [
        ("dispatched_buffer", tt_intermediates.dispatched_buffer, torch_intermediates.dispatched_buffer,
         get_ep_mesh_composer(mesh_device), torch.bfloat16, validate_dispatch_buffer, {}),
        ("dispatch_metadata", tt_intermediates.metadata, torch_intermediates.metadata,
         get_ep_mesh_composer(mesh_device), None, validate_dispatch_metadata, {}),
        ("expert_outputs", tt_intermediates.expert_outputs, torch_intermediates.expert_outputs,
         get_ep_mesh_composer(mesh_device), torch.bfloat16, validate_dispatch_buffer_pcc, {"pcc_threshold": 0.93}),
    ]
    # fmt: on

    for name, tt_tensor, torch_tensor, composer, dtype, validate_fn, extra_kwargs in sparse_checks:
        if tt_tensor is None:
            logger.warning(f"[{name}] validation SKIPPED")
            continue
        tt_host = (
            ttnn.to_torch(tt_tensor, mesh_composer=composer, dtype=dtype)
            if dtype
            else ttnn.to_torch(tt_tensor, mesh_composer=composer)
        )
        torch_ref = torch_tensor.to(dtype) if dtype else torch_tensor
        result = validate_fn(
            torch_ref,
            tt_host,
            expert_token_counts,
            expert_dispatch_table,
            num_dispatch_groups,
            dispatch_group_size,
            experts_per_chip,
            verbose=True,
            **extra_kwargs,
        )
        result.name = name
        validation_results.append(result)
        if result.passed:
            logger.info(f"[{name}] PASSED - {result.matches}/{result.total} slots matched")
        else:
            logger.error(f"[{name}] FAILED - {result.matches}/{result.total} slots matched")
            result.log_mismatches(limit=5)
            all_passed = False

    # Validate combined_output (before reduce step)
    if tt_intermediates.combined_output is not None:
        name = "combined_output"
        logger.debug(f"📊 {name} tt shape: {tt_intermediates.combined_output.shape}")
        logger.debug(f"📊 {name} torch shape: {torch_intermediates.combined_output.shape}")

        # Convert TTNN combined_output to torch
        combine_mesh_composer = get_ep_mesh_composer(mesh_device)
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

    # Log validation summary
    if validation_results:
        log_validation_results(
            results=validation_results,
            num_dispatch_groups=num_dispatch_groups,
            dispatch_group_size=dispatch_group_size,
            title="Sparse Tensor Validation Results",
        )

    logger.debug("Note: Final PCC expected to be low until full pipeline is enabled")
    profiler.end("pcc_validation")

    # Assert intermediate checks passed
    assert all_passed, "One or more intermediate comparisons failed"

    profiler.end("test_ttnn_moe")
    logger.debug(f"\n{'='*60}")
    logger.debug("TtMoe PCC Test PASSED!")
    logger.debug(f"{'='*60}")
    for key in profiler.times:
        logger.debug(f"{key}: {profiler.get(key) * 1000:.2f} ms")
