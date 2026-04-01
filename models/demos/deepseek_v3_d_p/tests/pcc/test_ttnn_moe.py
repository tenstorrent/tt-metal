# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
PCC test for TtMoe module with integrated gate.

Tests that TTNN TtMoe produces matching outputs to TorchMoe reference.
This validates the full MoE pipeline:
Gate → Dispatch → Routed Experts → Combine → Split → Add Shared.
"""

import random

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from conftest import is_galaxy
from models.common.utility_functions import profiler
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.tt.moe.moe import TorchMoe
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    create_fabric_router_config,
    create_gate_weights,
    create_shared_expert_weights,
    create_torch_expert_weights,
    extract_mesh_config,
    get_ep_mesh_composer,
    get_tp_mesh_composer,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    compare_recall,
    log_combine_mismatch_details,
    log_per_chip_statistics,
    validate_combine_output,
    validate_composed,
    validate_dispatch_buffer,
    validate_dispatch_buffer_pcc,
    validate_dispatch_metadata,
)
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_validation_results
from tests.ttnn.utils_for_testing import comp_pcc


@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, hidden_dim, num_routed_experts, num_experts_per_tok, capacity_factor, run_pcc_check",
    [
        # fmt: off
        pytest.param(3200, DeepSeekV3Config.EMB_SIZE, DeepSeekV3Config.MOE_INTERMEDIATE_SIZE, 256, 8, 2, False), # skip PCC validation
        pytest.param(3200, DeepSeekV3Config.EMB_SIZE, DeepSeekV3Config.MOE_INTERMEDIATE_SIZE, 256, 8, 2, True),  # run PCC validation
        pytest.param(3200, DeepSeekV3Config.EMB_SIZE, DeepSeekV3Config.MOE_INTERMEDIATE_SIZE, 256, 8, 2, True, marks=pytest.mark.skipif(not is_galaxy(), reason="Requires Galaxy")),
        # fmt: on
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="linear"),
            id="linear-8",
        ),
        pytest.param(
            (4, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
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
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
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

    Both TtMoe and TorchMoe create their gate internally from gate_weights
    and run forward(x) end-to-end. Validation compares intermediates directly.
    """
    profiler.clear()
    profiler.start("test_ttnn_moe")

    random.seed(42)
    torch.manual_seed(42)

    num_devices = mesh_device.get_num_devices()
    mesh_config = extract_mesh_config(mesh_device)
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups
    n_sp_devices, n_tp_devices = mesh_device.shape

    logger.debug(f"\n{'='*60}")
    logger.debug("TtMoe PCC Test")
    logger.debug(f"{'='*60}")
    logger.debug(f"mesh_shape={mesh_device.shape}, num_devices={num_devices}")
    logger.debug(f"dispatch_group_size={dispatch_group_size}, num_dispatch_groups={num_dispatch_groups}")

    signpost(
        f"TtMoe PCC test - mesh {mesh_device.shape}, seq_len={seq_len_per_chip}, "
        f"emb_dim={emb_dim}, experts={num_routed_experts}"
    )

    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, num_routed_experts, num_experts_per_tok, num_devices, dispatch_group_size, capacity_factor
    )
    logger.debug(f"experts_per_chip={experts_per_chip}, metadata_len={metadata_len}")
    logger.debug(f"max_dispatched_tokens_per_expert={max_dispatched_tokens_per_expert}")

    # ========================================
    # Step 1: Create weights
    # ========================================
    if run_pcc_check:
        profiler.start("weights_creation")
        all_routed_weights = create_torch_expert_weights(num_routed_experts, emb_dim, hidden_dim, seed=42)
        shared_expert_weights = create_shared_expert_weights(emb_dim, hidden_dim, seed=123)
        profiler.end("weights_creation")
    else:
        all_routed_weights = None
        shared_expert_weights = None

    gate_weights = create_gate_weights(num_routed_experts, emb_dim)

    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )

    # ========================================
    # Step 2: Create input tensor
    # ========================================
    profiler.start("input_creation")
    mesh_rows, mesh_cols = mesh_device.shape

    if run_pcc_check:
        x = torch.randn(dispatch_group_size, seq_len_per_chip, emb_dim, dtype=torch.bfloat16)
        tt_x = ttnn.from_torch(
            x,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(0, -1)),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.bfloat16,
        )
    else:
        x = None
        per_device_x_shape = (dispatch_group_size // mesh_rows, seq_len_per_chip, emb_dim // mesh_cols)
        tt_x = ttnn.empty(per_device_x_shape, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    profiler.end("input_creation")

    # ========================================
    # Step 3: Run TorchMoe reference with intermediates
    # ========================================
    if run_pcc_check:
        profiler.start("torch_moe_creation")
        torch_moe = TorchMoe(
            dispatch_group_size=dispatch_group_size,
            experts_per_chip=experts_per_chip,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            metadata_len=metadata_len,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            seq_len_per_chip=seq_len_per_chip,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            expert_dispatch_table=expert_dispatch_table,
            num_dispatch_groups=num_dispatch_groups,
            routed_expert_weights=all_routed_weights,
            shared_expert_weights=shared_expert_weights,
            gate_weights=gate_weights,
        )
        profiler.end("torch_moe_creation")

        profiler.start("torch_forward")
        torch_output, torch_intermediates = torch_moe(x, return_intermediates=True)
        profiler.end("torch_forward")
        logger.debug(f"Torch output stats - min: {torch_output.min():.4f}, max: {torch_output.max():.4f}")

    # ========================================
    # Step 4: TtMoe forward
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
        num_links=num_links,
        topology=topology,
        routed_expert_weights=all_routed_weights,
        shared_expert_weights=shared_expert_weights,
        activations_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        gate_weights=gate_weights,
    )
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_moe_creation")

    profiler.start("tt_forward")
    logger.debug("Running TtMoe forward pass...")

    tt_output, tt_intermediates = tt_moe(tt_x, return_intermediates=True)
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_forward")

    # Early return when run_pcc_check=False (profiling mode)
    if not run_pcc_check:
        profiler.end("test_ttnn_moe")
        logger.debug("run_pcc_check=False, skipping PCC validation")
        for key in profiler.times:
            logger.debug(f"{key}: {profiler.get(key) * 1000:.2f} ms")
        return

    # ========================================
    # Step 5: Validate
    # ========================================
    profiler.start("pcc_validation")
    logger.debug("Comparing intermediate outputs...")

    all_passed = True
    validation_results = []

    # Gate recall: compare TtMoe gate indices vs TorchMoe gate indices
    tt_indices = ttnn.to_torch(
        tt_intermediates.gate_indices,
        mesh_composer=get_tp_mesh_composer(mesh_device),
        dtype=torch.int32,
    )
    recall_result = validate_composed(
        tt_indices.view(1, n_sp_devices, seq_len_per_chip, -1),
        torch_intermediates.gate_indices.view(1, n_sp_devices, seq_len_per_chip, -1),
        1,
        n_sp_devices,
        compare_recall(0.999),
        name="gate_indices_recall",
        broadcast_groups=n_tp_devices,
    )
    log_validation_results(
        results=[recall_result],
        num_dispatch_groups=n_tp_devices,
        dispatch_group_size=n_sp_devices,
        title="Gate Recall Validation",
    )

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

    expert_token_counts = torch_intermediates.expert_token_counts

    for name, tt_tensor, torch_tensor, composer, dtype, validate_fn, extra_kwargs in sparse_checks:
        if tt_tensor is None or torch_tensor is None:
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
    if tt_intermediates.combined_output is not None and torch_intermediates.combined_output is not None:
        name = "combined_output"
        logger.debug(f"  {name} tt shape: {tt_intermediates.combined_output.shape}")
        logger.debug(f"  {name} torch shape: {torch_intermediates.combined_output.shape}")

        tt_combined_torch = ttnn.to_torch(
            tt_intermediates.combined_output,
            mesh_composer=get_ep_mesh_composer(mesh_device),
            dtype=torch.bfloat16,
        )

        combine_pcc = 0.997
        combine_result = validate_combine_output(
            torch_intermediates.combined_output,
            tt_combined_torch,
            tt_indices,
            num_dispatch_groups,
            num_routed_experts,
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

    assert all_passed, "One or more intermediate comparisons failed"
    recall_result.assert_passed("Gate recall validation failed")

    profiler.end("test_ttnn_moe")
    logger.debug(f"\n{'='*60}")
    logger.debug("TtMoe PCC Test PASSED!")
    logger.debug(f"{'='*60}")
    for key in profiler.times:
        logger.debug(f"{key}: {profiler.get(key) * 1000:.2f} ms")
