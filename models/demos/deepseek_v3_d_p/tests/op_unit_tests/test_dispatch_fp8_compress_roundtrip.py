# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
End-to-end PCC test for the FP8-compressed dispatch path.

Two dispatched buffers are compared:
  1. Reference: dispatch the BFLOAT16 tokens directly.
  2. FP8 path:  per_token_cast_to_fp8 -> dispatch the FP8 tokens -> masked per_token_cast_back.

Both paths use identical routing, so the dispatched buffers are row-aligned; the FP8 path's
decompressed buffer must match the BF16 reference within FP8 quantization error.

per_token_cast_back gathers each row's scale by the metadata token_idx, which is the source-local
token index — unambiguous only when every source device in a dispatch group holds the same tokens.
The test therefore replicates the per-source-device slices so the token_idx scale lookup is correct.
"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tests.pcc.mesh_configs import ALL_MESH_CONFIGS
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    extract_mesh_config,
    get_dispatch_input_mesh_mapper,
    get_ep_mesh_composer,
    get_expert_token_counts_mesh_mapper,
    get_gate_outputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import assert_output_shape, validate_dispatch_buffer_pcc
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_expert_dispatch_table, log_validation_results


@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, num_routed_experts, num_experts_per_tok, dispatch_buffer_capacity_factor",
    [
        pytest.param(640, 7168, 64, 8, 4, id="pcc"),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    ALL_MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
def test_ttnn_dispatch_fp8_compress_roundtrip(
    mesh_device,
    seq_len_per_chip,
    emb_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_buffer_capacity_factor,
    num_links,
    topology,
):
    """Compare a BF16 dispatch against the compress -> dispatch(fp8) -> decompress round-trip."""
    if not is_blackhole():
        pytest.skip("per_token_cast_to_fp8 / per_token_cast_back require Blackhole")

    num_devices = mesh_device.get_num_devices()

    torch.manual_seed(42)

    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    if sp_axis != 0:
        pytest.skip("per_token_cast_back counter_offset assumes dispatch groups along rows (sp_axis=0)")

    logger.debug(f"Testing with {mesh_device.shape=}, {num_devices=} {dispatch_group_size=} {num_dispatch_groups=}")
    ttnn.visualize_mesh_device(mesh_device)

    signpost(
        f"DispatchFp8Roundtrip {mesh_device=} {num_devices=} {dispatch_group_size=} {num_dispatch_groups=} "
        f"{seq_len_per_chip=} {emb_dim=} {num_routed_experts=} {num_experts_per_tok=} {num_links=} {topology=}"
    )

    (
        experts_per_chip,
        metadata_len,
        max_dispatch_buffer_token_size,
        max_dispatched_tokens_per_expert,
    ) = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        num_devices,
        dispatch_group_size,
        dispatch_buffer_capacity_factor,
    )
    logger.debug(
        f"{experts_per_chip=}, {metadata_len=}, {max_dispatch_buffer_token_size=}, {max_dispatched_tokens_per_expert=}"
    )

    # Initialize inputs using helper function
    x, weights, indices = initialize_test_inputs(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        num_dispatch_groups=num_dispatch_groups,
    )

    logger.debug(f"Input shapes: {x.shape=}, {weights.shape=}, {indices.shape=}")

    # x and indices: sharded across SP axis, replicated across EP ranks
    mesh_mapper_replicated = get_dispatch_input_mesh_mapper(mesh_device, sp_axis)

    tt_x = ttnn.from_torch(
        x, mesh_mapper=mesh_mapper_replicated, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )

    tt_weights = ttnn.from_torch(
        weights,
        mesh_mapper=mesh_mapper_replicated,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )

    tt_indices = ttnn.from_torch(
        indices,
        mesh_mapper=mesh_mapper_replicated,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint16,
    )

    # Create expert dispatch table
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

    tt_dispatch_module = TtDispatchModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatch_buffer_token_size=max_dispatch_buffer_token_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
        fp8_output=False,
    )

    # Compute gate outputs (offsets and token counts) before dispatch
    expert_offsets, expert_token_counts, expert_region_offsets, _ = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )

    # Forward pass through TTNN dispatch
    tt_expert_offsets = TtDispatchModule.shard_expert_offsets(mesh_device, expert_offsets)
    tt_expert_dispatch_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, expert_dispatch_table, sp_axis)

    # Reference: dispatch the bf16 tokens.
    tt_dispatched_ref, _ = tt_dispatch_module(tt_x, tt_weights, tt_indices, tt_expert_offsets, tt_expert_dispatch_table)

    # FP8 path: compress -> dispatch(fp8 row-major) -> masked decompress back to bf16.
    tt_fp8, tt_scale = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(tt_x)
    tt_dispatched_fp8, tt_metadata = tt_dispatch_module(
        tt_fp8, tt_weights, tt_indices, tt_expert_offsets, tt_expert_dispatch_table
    )

    # tt_scale is sharded across the dispatch group (each device holds only its own tokens' scales).
    # A dispatched buffer holds tokens from every source device in the group, so all-gather the scale
    # along the dispatch-group axis: every device then holds all ISL tokens' scales, which
    # per_token_cast_back indexes by the metadata token_idx.
    tt_scale = ttnn.all_gather(tt_scale, dim=1, cluster_axis=sp_axis, num_links=num_links, topology=topology)

    tt_expert_token_counts = ttnn.from_torch(
        expert_token_counts,
        mesh_mapper=get_expert_token_counts_mesh_mapper(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )
    tt_expert_region_offsets = ttnn.from_torch(
        expert_region_offsets,
        mesh_mapper=get_expert_token_counts_mesh_mapper(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    tt_dispatched_decompressed = ttnn.experimental.deepseek_prefill.per_token_cast_back(
        tt_dispatched_fp8,
        tt_scale,
        output_dtype=ttnn.bfloat16,
        expert_token_counts=tt_expert_token_counts,
        expert_region_offsets=tt_expert_region_offsets,
        metadata=tt_metadata,
        experts_per_chip=experts_per_chip,
        dispatch_group_size=dispatch_group_size,
    )

    # Convert TTNN outputs to torch for comparison
    mesh_composer = get_ep_mesh_composer(mesh_device)
    ref_buffer = ttnn.to_torch(tt_dispatched_ref, mesh_composer=mesh_composer, dtype=torch.float32)
    decompressed_buffer = ttnn.to_torch(tt_dispatched_decompressed, mesh_composer=mesh_composer, dtype=torch.float32)

    assert_output_shape(ref_buffer, num_dispatch_groups, dispatch_group_size, "reference buffer")
    assert_output_shape(decompressed_buffer, num_dispatch_groups, dispatch_group_size, "decompressed buffer")

    # FP8 quantizes the buffer (~3-bit mantissa), so allclose is too tight — compare valid expert
    # regions with PCC, matching the dispatch/combine fp8 tests.
    buffer_result = validate_dispatch_buffer_pcc(
        ref_buffer,
        decompressed_buffer,
        expert_region_offsets,
        expert_token_counts,
        expert_dispatch_table,
        num_dispatch_groups,
        dispatch_group_size,
        experts_per_chip,
    )

    log_validation_results(
        results=[buffer_result],
        num_dispatch_groups=num_dispatch_groups,
        dispatch_group_size=dispatch_group_size,
        title="Dispatch FP8 Compress/Decompress Validation Results",
    )

    assert buffer_result.passed, "BF16 dispatch vs FP8 compress/decompress buffer mismatch! Check logs for details."
    logger.debug("✅ FP8 compress/dispatch/decompress matches BF16 dispatch reference!")
