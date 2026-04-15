# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test for parallel dispatch+combine on independent submeshes with dynamic lifecycle.

Splits an 8-device mesh into two 4x1 submeshes and runs the full
dispatch→combine round-trip on each with different inputs, then tears down
the submeshes and runs again on the full 8x1 mesh, validating:
  1. Fabric isolation — each submesh's CCL stays within its 4 devices
  2. Correctness — both submeshes produce correct output vs torch reference
  3. Parallel execution — ops dispatched to both submeshes execute concurrently on HW
  4. Cross-contamination — different seeds per submesh detect data leaks
  5. Dynamic lifecycle — submeshes can be created, destroyed, and the parent mesh reused
"""

from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.tt.moe.combine import TorchCombineModule
from models.demos.deepseek_v3_d_p.reference.tt.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    create_fabric_router_config,
    extract_mesh_config,
    get_dispatch_input_mesh_mapper,
    get_ep_mesh_composer,
    get_gate_outputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    compare_exact,
    log_combine_mismatch_details,
    validate_combine_output,
    validate_composed,
    validate_roundtrip_output,
)
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_expert_dispatch_table, log_validation_results


@dataclass
class SubmeshContext:
    """Holds per-submesh state for deferred validation after both submeshes finish."""

    label: str
    mesh_device: ttnn.MeshDevice
    x: torch.Tensor
    indices: torch.Tensor
    tt_output: ttnn.Tensor
    torch_output: torch.Tensor
    num_dispatch_groups: int
    num_routed_experts: int
    dispatch_group_size: int
    expert_dispatch_table: torch.Tensor
    expert_token_counts: torch.Tensor
    experts_per_chip: int
    num_experts_per_tok: int


def setup_and_dispatch_roundtrip(
    mesh_device,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    capacity_factor,
    num_links,
    topology,
    seed,
    label,
):
    """
    Set up inputs and run dispatch+combine on a single submesh.

    Returns SubmeshContext with tt_output still on device (not yet synchronized).
    """
    num_devices = mesh_device.get_num_devices()

    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    logger.info(
        f"[{label}] mesh_shape={mesh_device.shape}, {num_devices=}, {dispatch_group_size=}, {num_dispatch_groups=}"
    )

    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, num_routed_experts, num_experts_per_tok, num_devices, dispatch_group_size, capacity_factor
    )
    logger.debug(f"[{label}] {experts_per_chip=}, {metadata_len=}, {max_dispatched_tokens_per_expert=}")

    # Generate inputs with submesh-specific seed
    x, weights, indices = initialize_test_inputs(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seed=seed,
        num_dispatch_groups=num_dispatch_groups,
    )
    logger.debug(f"[{label}] Input shapes: {x.shape=}, {weights.shape=}, {indices.shape=} (seed={seed})")

    # Place inputs on submesh
    mesh_mapper_dispatch_inputs = get_dispatch_input_mesh_mapper(mesh_device, sp_axis)

    tt_x = ttnn.from_torch(
        x,
        mesh_mapper=mesh_mapper_dispatch_inputs,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    tt_weights = ttnn.from_torch(
        weights,
        mesh_mapper=mesh_mapper_dispatch_inputs,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    tt_indices = ttnn.from_torch(
        indices,
        mesh_mapper=mesh_mapper_dispatch_inputs,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    # Expert dispatch table
    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )
    log_expert_dispatch_table(
        expert_dispatch_table=expert_dispatch_table,
        num_dispatch_groups=num_dispatch_groups,
        dispatch_group_size=dispatch_group_size,
        num_routed_experts=num_routed_experts,
    )

    # On-device routing setup
    tt_moe_routing_setup = TtMoERoutingSetup(
        mesh_device=mesh_device, expert_dispatch_table=expert_dispatch_table, num_links=num_links
    )
    tt_dispatch_offsets, tt_expert_token_counts, _ = tt_moe_routing_setup(
        ttnn_top_k_experts_indices=indices,
        num_routed_experts=num_routed_experts,
        seq_len_per_chip=seq_len_per_chip,
        num_experts_per_tok=num_experts_per_tok,
    )

    # Torch reference gate outputs
    expert_offsets, expert_token_counts, _ = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )

    # Validate routing setup
    ep_composer = get_ep_mesh_composer(mesh_device)
    host_offsets = ttnn.to_torch(ttnn.unsqueeze_to_4D(tt_dispatch_offsets), mesh_composer=ep_composer).squeeze(2)
    host_token_counts = ttnn.to_torch(ttnn.unsqueeze_to_4D(tt_expert_token_counts), mesh_composer=ep_composer).squeeze(
        2
    )

    offsets_result = validate_composed(
        host_offsets.int(),
        expert_offsets.int(),
        num_dispatch_groups,
        dispatch_group_size,
        compare_exact,
        name="expert_offsets",
    )
    counts_result = validate_composed(
        host_token_counts.int(),
        expert_token_counts.int(),
        num_dispatch_groups,
        dispatch_group_size,
        compare_exact,
        name="expert_token_counts",
    )
    log_validation_results(
        results=[offsets_result, counts_result],
        num_dispatch_groups=num_dispatch_groups,
        dispatch_group_size=dispatch_group_size,
        title=f"Routing Setup Validation ({label})",
    )
    offsets_result.assert_passed(f"[{label}] Dispatch offsets mismatch")
    counts_result.assert_passed(f"[{label}] Expert token counts mismatch")

    # TTNN dispatch
    logger.debug(f"[{label}] Running TTNN dispatch...")
    tt_dispatch_module = TtDispatchModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
    )

    tt_expert_dispatch_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, expert_dispatch_table, sp_axis)
    tt_dispatched_buffer, tt_metadata = tt_dispatch_module(
        tt_x, tt_weights, tt_indices, tt_dispatch_offsets, tt_expert_dispatch_table
    )
    logger.debug(f"[{label}] Dispatch dispatched (non-blocking)")

    # Torch reference dispatch+combine
    torch_dispatch_module = TorchDispatchModule(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        num_dispatch_groups=num_dispatch_groups,
        expert_dispatch_table=expert_dispatch_table,
    )
    torch_dispatched_buffer, torch_dispatched_metadata = torch_dispatch_module(x, weights, indices, expert_offsets)

    torch_combine_module = TorchCombineModule(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        num_dispatch_groups=num_dispatch_groups,
    )
    torch_output = torch_combine_module(torch_dispatched_buffer, torch_dispatched_metadata, expert_token_counts)

    # TTNN combine
    logger.debug(f"[{label}] Running TTNN combine...")
    tt_combine_module = TtCombineModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
        init_zeros=True,
    )
    tt_output = tt_combine_module(tt_dispatched_buffer, tt_metadata, tt_expert_token_counts)
    logger.debug(f"[{label}] Combine dispatched (non-blocking)")

    return SubmeshContext(
        label=label,
        mesh_device=mesh_device,
        x=x,
        indices=indices,
        tt_output=tt_output,
        torch_output=torch_output,
        num_dispatch_groups=num_dispatch_groups,
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        expert_dispatch_table=expert_dispatch_table,
        expert_token_counts=expert_token_counts,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
    )


def validate_roundtrip(ctx: SubmeshContext):
    """Synchronize, read back, and validate a submesh's dispatch+combine output."""
    ttnn.synchronize_device(ctx.mesh_device)
    logger.info(f"[{ctx.label}] Device synchronized, validating...")

    ep_composer = get_ep_mesh_composer(ctx.mesh_device)
    y = ttnn.to_torch(ctx.tt_output, mesh_composer=ep_composer, dtype=torch.bfloat16)
    y = y.squeeze(-4)

    # Combine validation (TTNN vs torch combine reference)
    combine_result = validate_combine_output(
        ctx.torch_output,
        y,
        ctx.indices,
        ctx.num_dispatch_groups,
        ctx.num_routed_experts,
        verbose=True,
        expert_dispatch_table=ctx.expert_dispatch_table,
        expert_token_counts=ctx.expert_token_counts,
        experts_per_chip=ctx.experts_per_chip,
    )

    log_validation_results(
        results=[combine_result],
        num_dispatch_groups=ctx.num_dispatch_groups,
        dispatch_group_size=ctx.dispatch_group_size,
        title=f"Combine Validation ({ctx.label})",
    )

    # Round-trip validation (x == combine(dispatch(x)))
    result = validate_roundtrip_output(
        ctx.x,
        y,
        ctx.indices,
        ctx.num_dispatch_groups,
        ctx.num_routed_experts,
    )

    log_validation_results(
        results=[result],
        num_dispatch_groups=ctx.num_dispatch_groups,
        dispatch_group_size=ctx.dispatch_group_size,
        title=f"Roundtrip Validation ({ctx.label})",
    )

    if not result.passed:
        x_expanded = ctx.x.unsqueeze(2).expand(-1, -1, ctx.num_experts_per_tok, -1)
        log_combine_mismatch_details(result.mismatches, x_expanded, y)

    return result


@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, num_routed_experts, num_experts_per_tok, capacity_factor",
    [
        (3200, 7168, 64, 2, 2),
    ],
    ids=["3200-avg"],
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
            id="linear-8-1link",
        ),
        pytest.param(
            (8, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 1), topology="linear"),
            id="linear-8-2link",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_submesh_parallel_dispatch_combine(
    mesh_device,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    capacity_factor,
    num_links,
    topology,
):
    """Test dispatch+combine on two independent 4x1 submeshes, then tear them down and run on full 8x1 mesh."""

    assert mesh_device.get_num_devices() == 8, f"Expected 8 devices, got {mesh_device.get_num_devices()}"
    logger.info(f"Parent mesh shape: {mesh_device.shape}")
    ttnn.visualize_mesh_device(mesh_device)

    # Split into two (4,1) submeshes
    submeshes = mesh_device.create_submeshes(ttnn.MeshShape(4, 1))
    assert len(submeshes) == 2, f"Expected 2 submeshes, got {len(submeshes)}"
    submesh_a, submesh_b = submeshes[0], submeshes[1]

    logger.info(f"Submesh A: shape={submesh_a.shape}, devices={submesh_a.get_num_devices()}")
    logger.info(f"Submesh B: shape={submesh_b.shape}, devices={submesh_b.get_num_devices()}")

    # Phase 1: Setup and dispatch to both submeshes (non-blocking ops give HW parallelism)
    ctx_a = setup_and_dispatch_roundtrip(
        submesh_a,
        seq_len_per_chip,
        emb_dim,
        num_routed_experts,
        num_experts_per_tok,
        capacity_factor,
        num_links,
        topology,
        seed=42,
        label="submesh_a",
    )

    ctx_b = setup_and_dispatch_roundtrip(
        submesh_b,
        seq_len_per_chip,
        emb_dim,
        num_routed_experts,
        num_experts_per_tok,
        capacity_factor,
        num_links,
        topology,
        seed=123,
        label="submesh_b",
    )

    # Phase 2: Synchronize and validate each submesh
    result_a = validate_roundtrip(ctx_a)
    result_b = validate_roundtrip(ctx_b)

    result_a.assert_passed("Submesh A round-trip mismatch")
    result_b.assert_passed("Submesh B round-trip mismatch")

    logger.info("Both submeshes passed dispatch+combine round-trip independently!")

    # Phase 3: Tear down 4x1 submeshes and run on full 8x1 parent mesh
    logger.info("Phase 3: Synchronizing and releasing 4x1 submeshes...")
    ttnn.synchronize_device(submesh_a)
    ttnn.synchronize_device(submesh_b)

    # Close submeshes explicitly via the same API the fixture teardown uses.
    # This ensures CQ state is properly torn down before reusing the parent mesh.
    ttnn.close_mesh_device(submesh_a)
    ttnn.close_mesh_device(submesh_b)
    # del ctx_a, ctx_b
    # del submesh_a, submesh_b, submeshes

    logger.info("Submeshes closed. Running dispatch+combine on full 8x1 parent mesh...")
    ttnn.visualize_mesh_device(mesh_device)

    ctx_full = setup_and_dispatch_roundtrip(
        mesh_device,
        seq_len_per_chip,
        emb_dim,
        num_routed_experts,
        num_experts_per_tok,
        capacity_factor,
        num_links,
        topology,
        seed=777,
        label="full_8x1",
    )

    result_full = validate_roundtrip(ctx_full)
    result_full.assert_passed("Full 8x1 mesh round-trip mismatch after submesh teardown")

    logger.info("Full 8x1 parent mesh passed dispatch+combine after submesh teardown!")
