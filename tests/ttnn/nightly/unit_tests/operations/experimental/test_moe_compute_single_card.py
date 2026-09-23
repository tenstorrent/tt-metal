# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Single-card MoE compute test (1x1 mesh, cluster_axis=None). Runs on both WH
and BH; other arches are skipped at fixture time. Three tests use a (1, 4) mesh with
cluster_axis=0: an axis of extent 1 has no neighbours, so the ordinary fabric (FullCcl)
combine degenerates to a local combine at every device without touching the fabric; the
mesh_device fixture skips them on a single card.

This test exercises both paths of `ttnn.experimental.moe_compute` on a single device:
  - `compute_only=True`: bypasses the fused selective_reduce_combine stage entirely.
    Returns 5 tensors; matmul_output (slot 4) is the final output.
  - `compute_only=False` (FullLocal): runs the fused local combine stage without CCL/fabric.
    Returns 6 tensors; combine_output (slot 5) is the final output.

It is the hermetic dev/regression net for the MoE compute kernels (tilize + matmul +
activation [+ combine]) without requiring a 6U Galaxy host or working CCL-on-BH.

Validation points (all using the 6U helpers verbatim — no logic duplication):
  - Output 0 (per_expert_total_tokens)
  - Output 1 (expert_activation)
  - Output 2 (e_t)
  - Output 4 (matmul_output) — final output in compute_only mode
  - Output 5 (combine_output) — final output in FullLocal mode, validated only when
    compute_only=False
"""

import os
import pytest
import random
import torch
import ttnn
from loguru import logger

from ttnn.operations.ccl import MoEActivationFunction

from ttnn.experimental.moe_compute_utils import (
    prepare_w0_w1_tensor_for_moe_compute,
    prepare_w0_w1_tensor_with_bias,
    prepare_w2_tensor_for_moe_compute,
    prepare_w2_tensor_with_bias,
    get_weight_core_shard_maps,
    get_weight_mem_configs,
    auto_output_width_shard_dim,
    effective_matmul_ring_size,
)

# Reuse 6U test helpers verbatim. The intent is that this single-card test
# never duplicates compute logic — same goldens, same validators.
from tests.nightly.tg.ccl.moe.test_moe_compute_6U import (
    create_torch_w0,
    create_torch_w1,
    create_torch_w2,
    compute_e_t_golden,
    compute_expert_activation_golden,
    compute_matmul_golden,
    compute_combine_golden,
    compute_selective_tilize_golden,
    create_sharded_memory_config,
    gen_expert_mapping,
    gen_sparse_buffer_and_indices,
    tt_to_torch_dtype,
    validate_activation,
    validate_e_t,
    validate_matmul,
    validate_combine,
    validate_combine_torch,
    validate_per_expert_tokens,
    _get_base_pcc_threshold,
)


def _build_quantized_weight_tensors_cpu_prepare(
    mesh_device,
    torch_w0,
    torch_w1,
    torch_w2,
    torch_b0,
    torch_b1,
    torch_b2,
    num_layers,
    experts_per_device,
    hidden_size,
    N,
    has_bias,
    w0_w1_shard_map,
    w2_shard_map,
    w0_w1_mem_config,
    w2_mem_config,
):
    """Upload prepared weights as ``bfloat4_b`` HEIGHT_SHARDED device tensors.

    With bias, direct ``from_torch(..., bfloat4_b)`` segfaults in ``pack_as_bfp4_tiles`` for
    large DeepSeek-shaped tensors; upload bf16 then ``typecast`` on device instead. Full 6U
    flow (on-device prepare → ``quantize_weights_via_host``) OOMs on single-card WH DRAM.
    """

    def _upload_bf16_then_typecast(torch_tensor, mem_config):
        tt_bf16 = ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        tt_b4 = ttnn.typecast(tt_bf16, dtype=ttnn.bfloat4_b)
        ttnn.deallocate(tt_bf16)
        return tt_b4

    def _upload_bfloat4_direct(torch_tensor, mem_config):
        return ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.bfloat4_b,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    upload_fn = _upload_bf16_then_typecast if has_bias else _upload_bfloat4_direct

    if has_bias:
        torch_w0_w1_reordered = prepare_w0_w1_tensor_with_bias(
            torch_w0, torch_w1, torch_b0, torch_b1, num_layers, experts_per_device, hidden_size, N, w0_w1_shard_map
        )
    else:
        torch_w0_w1_reordered = prepare_w0_w1_tensor_for_moe_compute(
            torch_w0, torch_w1, num_layers, experts_per_device, hidden_size, N, w0_w1_shard_map
        )
    tt_w0_w1 = upload_fn(torch_w0_w1_reordered, w0_w1_mem_config)
    del torch_w0_w1_reordered

    if has_bias:
        torch_w2_reordered = prepare_w2_tensor_with_bias(
            torch_w2, torch_b2, num_layers, experts_per_device, N, hidden_size, w2_shard_map, w0_w1_shard_map
        )
    else:
        torch_w2_reordered = prepare_w2_tensor_for_moe_compute(
            torch_w2, num_layers, experts_per_device, N, hidden_size, w2_shard_map, w0_w1_shard_map
        )
    tt_w2 = upload_fn(torch_w2_reordered, w2_mem_config)
    del torch_w2_reordered

    return tt_w0_w1, tt_w2


@torch.no_grad()
def _run_moe_compute_single_card_test(
    mesh_device,
    mesh_shape,
    experts_per_device,
    tokens_per_device,
    selected_experts_k,
    N,
    hidden_size,
    output_height_shard_dim,
    output_width_shard_dim,
    dtype,
    activation_type,
    has_bias=False,
    compute_only=True,
    skip_on_ci=False,
    matmul_xfail_on_bh=False,
    op_cluster_axis=None,
    expect_error=None,
    ccl_knobs=False,
    local_output_memory_config=None,
):
    """
    Single-card MoE compute test body. The op is called with cluster_axis=op_cluster_axis:
    None (the 1x1 FullLocal path, fused local combine) or 0, an axis of extent 1 on every mesh
    shape used here. With an explicit axis the op takes its local output path: no combine
    kernels, dm1 writes the final [k, T, H] tensor directly, and the fabric is never consulted
    (this fixture never enables one).

    On a 1xN mesh with op_cluster_axis=0 every device writes one local output over the same
    replicated token set and its own expert shard (global expert e lives on device
    e // experts_per_device), and the outputs are the per-device partials the caller sums.
    That path needs the conftest ``expect_error`` fixture for its sharded-input rejection check.

    ccl_knobs=True passes the Galaxy-style CCL arguments (topology, num_links, a mux core range
    and a cross-device GlobalSemaphore) with the explicit axis; on an axis of extent 1 they are
    accepted and unused, so the result must match the plain call. The mux range is threaded
    through the core-placement helpers as the op does.

    local_output_memory_config (explicit axis only): after the interleaved runs the op is run
    twice more with a preallocated output in that memory config (a program-cache miss, then the
    hit with a fresh tensor), slot 5 must be that tensor and every device's rows must be
    bitwise equal to the interleaved output of the same inputs: the writer addresses the
    output through a TensorAccessor built from the given buffer, one token row per page.

    The matmul ring size is auto-detected from the live DRAM-bank count (12 on WH, 7/8 on
    BH) — the same ``effective_matmul_ring_size(mesh_device)`` the public op uses — and is used
    to pack the weights so host tensor layout matches the op's ring-aware width-parallel
    auto-derivation.
    """
    arch = mesh_device.arch()
    if arch not in (ttnn.device.Arch.WORMHOLE_B0, ttnn.device.Arch.BLACKHOLE):
        pytest.skip(f"MoE compute single-card test: arch {arch} is not supported (only WH and BH).")

    if arch == ttnn.device.Arch.BLACKHOLE and skip_on_ci:
        # Matmul output fails PCC on BH; runs locally for regression, skipped in CI pending fix.
        # https://github.com/tenstorrent/tt-metal/issues/50038
        pytest.skip(
            "MoE compute single-card test fails PCC on BH; skipped in CI pending fix "
            "(https://github.com/tenstorrent/tt-metal/issues/50038)."
        )

    torch.manual_seed(2003)
    random.seed(2003)

    # Single device, no CCL: cluster_axis is None.
    cluster_axis = None
    num_layers = 1

    # Derived dims (mirrors run_moe_compute_test in test_moe_compute_6U.py).
    num_devices = mesh_shape[0] * mesh_shape[1]
    multi_device_local = num_devices > 1
    if multi_device_local:
        assert op_cluster_axis == 0 and mesh_shape[0] == 1, "a multi-device run needs cluster_axis=0 on a 1xN mesh"
        assert expect_error is not None, "a multi-device run needs the expect_error fixture"
    num_dispatch_devices = num_devices  # cluster_axis is None
    num_replicated_devices = num_devices // num_dispatch_devices
    total_tokens = tokens_per_device * num_dispatch_devices

    experts = experts_per_device * num_devices
    experts_per_cluster = experts // num_replicated_devices

    logger.info(f"Single-card MoE compute test:")
    logger.info(f"  mesh_shape: {mesh_shape}")
    logger.info(f"  cluster_axis: {cluster_axis}")
    logger.info(f"  compute_only: {compute_only}")
    logger.info(f"  op cluster_axis: {op_cluster_axis}")
    logger.info(f"  num_devices: {num_devices}")
    logger.info(f"  tokens_per_device: {tokens_per_device}, total_tokens: {total_tokens}")
    logger.info(f"  experts: {experts}, experts_per_device: {experts_per_device}")
    logger.info(f"  selected_experts_k: {selected_experts_k}")
    logger.info(f"  hidden_size: {hidden_size}, N: {N}")
    logger.info(f"  output_height_shard_dim: {output_height_shard_dim}")
    logger.info(f"  output_width_shard_dim: {output_width_shard_dim}")

    #########################################
    # CREATE TILIZE INPUT TENSORS AND GOLDENS
    #########################################

    # Drain tilize core: use dynamic core placement API to get the drain core
    # instead of hardcoding per-arch coordinates. This works on both WH and BH
    # and adapts to harvested grids (when supported).
    # The mux range of the ccl_knobs variant: unused on an axis of extent 1, but the op still
    # places every worker group around it, so the placement helpers must see the same range.
    mux_core_range_set = (
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 3))]) if ccl_knobs else None
    )
    placement_kwargs = {"mux_core_range_set": mux_core_range_set} if ccl_knobs else {}
    drain_core_coord = ttnn.experimental.get_moe_tilize_drain_core(
        mesh_device,
        output_height_shard_dim,
        output_width_shard_dim,
        hidden_size,
        **placement_kwargs,
    )
    tilize_drain_core = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(drain_core_coord.x, drain_core_coord.y),
                ttnn.CoreCoord(drain_core_coord.x, drain_core_coord.y),
            )
        }
    )

    expert_mapping = gen_expert_mapping(
        num_devices, num_replicated_devices, cluster_axis, experts, experts_per_cluster, experts_per_device
    )
    expert_mapping_mem_config = ttnn.L1_MEMORY_CONFIG
    tt_expert_mapping = ttnn.from_torch(
        expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=expert_mapping_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    sparse_mem_config = ttnn.L1_MEMORY_CONFIG
    expert_indices_shard_shape = [total_tokens, selected_experts_k]
    expert_indices_mem_config = create_sharded_memory_config(tilize_drain_core, expert_indices_shard_shape, ttnn.uint16)
    expert_scores_shard_shape = [total_tokens, selected_experts_k]
    expert_scores_mem_config = create_sharded_memory_config(tilize_drain_core, expert_scores_shard_shape, dtype)

    # Generate test data.
    sparse_buffer, expert_indices, expert_scores, original_tokens = gen_sparse_buffer_and_indices(
        tokens_per_device,
        hidden_size,
        experts,
        selected_experts_k,
        mesh_shape,
        cluster_axis,
        dtype=tt_to_torch_dtype(dtype),
    )

    # Goldens.
    tilize_golden_output, expert_token_counts = compute_selective_tilize_golden(
        sparse_buffer, expert_indices, expert_scores, expert_mapping, mesh_shape, cluster_axis
    )
    logger.info(f"  expert_token_counts:\n{expert_token_counts}")

    golden_activation, _ = compute_expert_activation_golden(
        expert_indices, expert_scores, expert_mapping, mesh_shape, cluster_axis
    )

    golden_e_t, _ = compute_e_t_golden(expert_indices, expert_mapping, mesh_shape, cluster_axis)

    # Stack the (one) layer along the L dim so compute_matmul_golden gets shape (L, D, E/D, T, H).
    tilize_golden_outputs = tilize_golden_output.unsqueeze(0)

    # Sparse buffer / indices / scores tensors. The 1x1 tests shard the per-device stack on
    # dim 0 (one slice per device). The multi-device local combine replicates one copy of the
    # dense token set instead: the op requires a replicated input topology, and each device
    # reads only the rows routed to its own experts (the golden reads the same rows).
    if multi_device_local:
        token_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        tilize_input = original_tokens.reshape(1, total_tokens, hidden_size)
        token_copies = 1
    else:
        token_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        tilize_input = sparse_buffer
        token_copies = num_devices

    def upload_tilize_input(torch_input, mesh_mapper):
        return ttnn.from_torch(
            torch_input,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=sparse_mem_config,
            mesh_mapper=mesh_mapper,
        )

    tt_sparse_buffer = upload_tilize_input(tilize_input, token_mesh_mapper)

    expert_indices_flat = expert_indices.reshape(total_tokens, selected_experts_k)
    expert_indices_replicated = expert_indices_flat.unsqueeze(0).repeat(token_copies, 1, 1)
    tt_expert_indices = ttnn.from_torch(
        expert_indices_replicated,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=expert_indices_mem_config,
        mesh_mapper=token_mesh_mapper,
    )

    expert_scores_flat = expert_scores.reshape(total_tokens, selected_experts_k)
    expert_scores_replicated = expert_scores_flat.unsqueeze(0).repeat(token_copies, 1, 1)
    tt_expert_scores = ttnn.from_torch(
        expert_scores_replicated,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dtype,
        memory_config=expert_scores_mem_config,
        mesh_mapper=token_mesh_mapper,
    )

    #########################################
    # CREATE MATMUL INPUT TENSORS
    #########################################

    w0_w1_shard_map, w2_shard_map, dram_core_range_set = get_weight_core_shard_maps(mesh_device, hidden_size, N)

    torch_w0 = create_torch_w0(num_layers, experts_per_device, hidden_size, N)
    torch_w1 = create_torch_w1(num_layers, experts_per_device, hidden_size, N)
    torch_w2 = create_torch_w2(num_layers, experts_per_device, N, hidden_size)

    # Bias tensors (mirrors test_moe_compute_6U.run_moe_compute_test bias block).
    # Use the same _bias_std and PyTorch shape conventions so the prepare-with-bias
    # functions emit byte-identical tile padding to the 1x16 reference.
    torch_b0 = torch_b1 = torch_b2 = None
    if has_bias:
        _bias_std = 0.12
        torch_b0 = (torch.randn(num_layers, experts_per_device, N, dtype=torch.float32) * _bias_std).to(torch.bfloat16)
        torch_b1 = (torch.randn(num_layers, experts_per_device, N, dtype=torch.float32) * _bias_std).to(torch.bfloat16)
        torch_b2 = (torch.randn(num_layers, experts_per_device, hidden_size, dtype=torch.float32) * _bias_std).to(
            torch.bfloat16
        )

    matmul_goldens = compute_matmul_golden(
        tilize_golden_outputs,
        torch_w0,
        torch_w1,
        torch_w2,
        num_layers,
        experts,
        num_devices,
        tokens_per_device,
        hidden_size,
        torch_b0=torch_b0,
        torch_b1=torch_b1,
        torch_b2=torch_b2,
        activation_type=activation_type,
    )

    w0_w1_mem_config, w2_mem_config, _, _ = get_weight_mem_configs(
        num_layers,
        experts_per_device,
        hidden_size,
        N,
        w0_w1_shard_map,
        w2_shard_map,
        dram_core_range_set,
        has_bias=has_bias,
    )

    # CPU prepare → bf16 HEIGHT_SHARDED → on-device typecast (see helper docstring).
    tt_w0_w1, tt_w2 = _build_quantized_weight_tensors_cpu_prepare(
        mesh_device,
        torch_w0,
        torch_w1,
        torch_w2,
        torch_b0,
        torch_b1,
        torch_b2,
        num_layers,
        experts_per_device,
        hidden_size,
        N,
        has_bias,
        w0_w1_shard_map,
        w2_shard_map,
        w0_w1_mem_config,
        w2_mem_config,
    )

    #########################################
    # RUN OP
    #########################################
    logger.info(f"\n========== Running op (compute_only={compute_only}) ==========")

    def create_combine_output_tensor(memory_config=None):
        torch_combine_output = torch.zeros([selected_experts_k, total_tokens, hidden_size], dtype=torch.bfloat16)
        memory_config_kwargs = {} if memory_config is None else {"memory_config": memory_config}
        return ttnn.from_torch(
            torch_combine_output,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=token_mesh_mapper if multi_device_local else ttnn.ShardTensorToMesh(mesh_device, dim=1),
            **memory_config_kwargs,
        )

    output_shard_cores = ttnn.experimental.get_moe_combine_cores(
        mesh_device, output_height_shard_dim, output_width_shard_dim, hidden_size, **placement_kwargs
    )
    if ccl_knobs:
        # Galaxy-style CCL arguments. On an axis of extent 1 the combine has no neighbours, so the
        # op opens no link, launches no mux worker and never touches the semaphore.
        combine_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in output_shard_cores])
        ccl_kwargs = dict(
            topology=ttnn.Topology.Linear,
            num_links=2,
            mux_core_range_set=mux_core_range_set,
            optional_cross_device_semaphore=ttnn.create_global_semaphore(mesh_device, combine_core_range_set, 0),
        )
    else:
        # cluster_axis=None: required for compute_only and for the implicit 1x1 FullLocal call,
        # and topology/num_links/mux/semaphore must be None there. cluster_axis=0 names an axis
        # of extent 1 (1x1 or 1xN): the fabric path with no neighbours, run as a local combine at
        # every mesh coordinate; the CCL arguments may be None.
        ccl_kwargs = dict(topology=None, num_links=None, mux_core_range_set=None, optional_cross_device_semaphore=None)

    def run_moe_compute_once(optional_combine_output_tensor, tilize_input_tensor=None, output_memory_config=None):
        return ttnn.experimental.moe_compute(
            tt_sparse_buffer if tilize_input_tensor is None else tilize_input_tensor,
            tt_expert_indices,
            tt_expert_scores,
            tt_expert_mapping,
            tt_w0_w1,
            tt_w2,
            layer_id=layer_id,
            output_height_shard_dim=output_height_shard_dim,
            intermediate_size=N,
            has_bias=has_bias,
            cluster_axis=op_cluster_axis,
            output_memory_config=output_memory_config,
            optional_output_tensor=optional_combine_output_tensor,
            activation_type=activation_type,
            compute_only=compute_only,
            **ccl_kwargs,
        )

    def per_device_bf16_bits(tensor):
        # The raw bf16 bit patterns of every device's rows: the bitwise comparison of two output
        # placements must not go through a float compare (NaN != NaN, -0.0 == 0.0).
        return [
            ttnn.to_torch(device_tensor, mesh_composer=None).contiguous().view(torch.int16)
            for device_tensor in ttnn.get_device_tensors(tensor)
        ]

    def deallocate_l1_moe_compute_outputs(output_tensors):
        # Slots 3 and 4 share a backing buffer; deallocating slot 4 releases the shared L1 output.
        for tensor in (output_tensors[0], output_tensors[1], output_tensors[2], output_tensors[4]):
            ttnn.deallocate(tensor)

    # For the fused combine (compute_only=False), pre-allocate the optional combine output tensor.
    combine_goldens = None
    tt_combine_output_tensor = None
    if not compute_only:
        combine_goldens = compute_combine_golden(
            num_layers,
            experts,
            total_tokens,
            hidden_size,
            selected_experts_k,
            mesh_shape,
            matmul_goldens,
            [golden_activation],  # compute_combine_golden expects a per-layer list
            cluster_axis=-1,  # single-device: get_cluster_dims uses -1 for "no replication axis"
        )
        tt_combine_output_tensor = create_combine_output_tensor()

    layer_id = 0

    def assert_replicated_outputs(output_tensors):
        # A combine axis of extent 1 on a multi-device mesh leaves one full-width partial per
        # device and shards no tensor dimension, so every output keeps the replicated input
        # topology (the op's compute_output_topologies) instead of an expert-sharded placement
        # from the weights.
        input_topology = tt_sparse_buffer.tensor_topology()
        for slot, tensor in enumerate(output_tensors):
            placements = tensor.tensor_topology().placements()
            assert all(
                isinstance(placement, ttnn.PlacementReplicate) for placement in placements
            ), f"output slot {slot} placements {placements} are not all replicated"
            assert tensor.tensor_topology() == input_topology, f"output slot {slot} topology differs from the input"

    def validate_combine_output(tt_combine_output, pcc_threshold):
        if not multi_device_local:
            # validate_combine uses cluster_axis for the mesh composer; on 1x1, dim=1 is correct.
            return validate_combine(
                layer_id,
                mesh_device,
                cluster_axis=1,  # single device: either axis works
                tt_combine_output=tt_combine_output,
                combine_goldens=combine_goldens,
                pcc_threshold=pcc_threshold,
            )
        # Every device holds one full-width partial over its own experts: compare each device's
        # output against the golden rows its experts own (the caller sums the partials).
        output_ref, output_data_map = combine_goldens
        slot_owner = expert_mapping[0].long()[expert_indices_flat.long()].transpose(0, 1)  # [K, total_tokens]
        all_passed = True
        for device_idx, device_tensor in enumerate(ttnn.get_device_tensors(tt_combine_output)):
            device_data_map = output_data_map * (slot_owner == device_idx).unsqueeze(0)
            passed = validate_combine_torch(
                layer_id, ttnn.to_torch(device_tensor, mesh_composer=None), (output_ref, device_data_map), pcc_threshold
            )
            logger.info(f"Combine Output Tensor device {device_idx}: {'PASSED' if passed else 'FAILED'}")
            all_passed = all_passed and passed
        return all_passed

    outputs = run_moe_compute_once(tt_combine_output_tensor)

    # ===================================================================
    # TRIPWIRE: output count must match the mode.
    # - compute_only=True: 5 tensors (matmul_output is the final output, no combine).
    # - compute_only=False (FullLocal): 6 tensors (slot 5 = combine output).
    # ===================================================================
    expected_n = 5 if compute_only else 6
    assert (
        len(outputs) == expected_n
    ), f"compute_only={compute_only} must return {expected_n} tensors. Got {len(outputs)}."
    if multi_device_local:
        assert_replicated_outputs(outputs)

    if compute_only:
        (
            per_expert_total_tokens_output_tensor,
            expert_activation_output_tensor,
            e_t_output_tensor,
            tilize_output_tensor,  # slot 3
            matmul_output_tensor,  # slot 4 -- final output
        ) = outputs
        combine_output_tensor = None
    else:
        (
            per_expert_total_tokens_output_tensor,
            expert_activation_output_tensor,
            e_t_output_tensor,
            tilize_output_tensor,  # slot 3
            matmul_output_tensor,  # slot 4
            combine_output_tensor,  # slot 5 -- final output in FullLocal
        ) = outputs

    # Move outputs to DRAM for validation (host readback).
    per_expert_total_tokens_output_tensor = ttnn.to_memory_config(
        per_expert_total_tokens_output_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    expert_activation_output_tensor = ttnn.to_memory_config(
        expert_activation_output_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    e_t_output_tensor = ttnn.to_memory_config(e_t_output_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    matmul_output_tensor = ttnn.to_memory_config(matmul_output_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if combine_output_tensor is not None:
        combine_output_tensor = ttnn.to_memory_config(combine_output_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    #########################################
    # VALIDATE
    #########################################
    logger.info(f"\n========== Validation ==========")
    logger.info(f"Per expert total tokens tensor shape: {per_expert_total_tokens_output_tensor.shape}")
    logger.info(f"Expert activation tensor shape: {expert_activation_output_tensor.shape}")
    logger.info(f"E-T tensor shape: {e_t_output_tensor.shape}")
    logger.info(f"Matmul output tensor shape: {matmul_output_tensor.shape}")

    all_core_grid = mesh_device.compute_with_storage_grid_size()
    all_core_range_set = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(all_core_grid.x - 1, all_core_grid.y - 1),
            ),
        }
    )
    worker_mcast_bbox = ttnn.experimental.get_moe_worker_mcast_bounding_box(
        mesh_device, output_height_shard_dim, output_width_shard_dim, hidden_size, **placement_kwargs
    )

    base_pcc_threshold = _get_base_pcc_threshold(activation_type, has_bias)
    if has_bias:
        # with_bias weights use bf16 upload + on-device typecast (pack_as_bfp4 segfault workaround).
        base_pcc_threshold = min(base_pcc_threshold, 0.982)
    else:
        base_pcc_threshold = min(base_pcc_threshold, 0.984)

    per_expert_tokens_all_passed = validate_per_expert_tokens(
        mesh_device,
        experts_per_device,
        num_devices,
        per_expert_total_tokens_output_tensor,
        expert_token_counts,
        worker_mcast_bbox,
    )

    activation_all_passed = validate_activation(
        mesh_device,
        experts_per_device,
        num_devices,
        expert_activation_output_tensor,
        golden_activation,
    )

    e_t_all_passed = validate_e_t(
        mesh_device,
        total_tokens,
        experts_per_device,
        num_devices,
        e_t_output_tensor,
        golden_e_t,
    )

    # The local output path (an explicit cluster_axis of extent 1) writes the final tensor directly
    # and never stages the expert outputs in the combine cores' L1, so slot 4 holds whatever the
    # shared buffer last carried; the final output below is the validated artifact there.
    local_output_path = not compute_only and op_cluster_axis is not None
    if local_output_path:
        matmul_all_passed = True
        logger.info("Matmul Output Tensor: not staged on the local output path (slot 4 undefined), skipped")
    else:
        matmul_all_passed = validate_matmul(
            layer_id,
            experts_per_device,
            all_core_range_set,
            output_shard_cores,
            output_height_shard_dim,
            output_width_shard_dim,
            total_tokens,
            hidden_size,
            expert_token_counts,
            matmul_goldens,
            matmul_output_tensor,
            mesh_device,
            base_pcc_threshold,
            has_bias=has_bias,
        )

    combine_all_passed = True
    if not compute_only:
        combine_all_passed = validate_combine_output(combine_output_tensor, base_pcc_threshold)

    logger.info(f"\n========== Asserts ==========")
    logger.info(f"Per Expert Total Tokens: {'PASSED' if per_expert_tokens_all_passed else 'FAILED'}")
    logger.info(f"Expert Activation: {'PASSED' if activation_all_passed else 'FAILED'}")
    logger.info(f"E-T Tensor: {'PASSED' if e_t_all_passed else 'FAILED'}")
    logger.info(f"Matmul Output Tensor: {'PASSED' if matmul_all_passed else 'FAILED'}")
    if not compute_only:
        logger.info(f"Combine Output Tensor: {'PASSED' if combine_all_passed else 'FAILED'}")

    assert per_expert_tokens_all_passed, "Per expert total tokens tensor verification failed!"
    assert activation_all_passed, "Expert activation tensor verification failed!"
    assert e_t_all_passed, "E-T tensor verification failed!"
    # matmul-output PCC is broken on Blackhole by #50038 (separate); xfail it there so the metadata
    # regression above still runs on BH. Only fires when matmul actually fails.
    if not matmul_all_passed and matmul_xfail_on_bh and arch == ttnn.device.Arch.BLACKHOLE:
        pytest.xfail(
            "moe_compute matmul-output PCC on Blackhole is tracked by "
            "https://github.com/tenstorrent/tt-metal/issues/50038 (independent of #50669); "
            "E-T / activation / per-expert-tokens metadata asserted above."
        )
    assert matmul_all_passed, "Matmul output tensor verification failed!"
    if not compute_only:
        assert combine_all_passed, "Combine output tensor verification failed!"

        # Read before the deallocations below: to_memory_config returns slot 5 itself when it is
        # already DRAM interleaved, so combine_output_tensor may alias tt_combine_output_tensor.
        interleaved_bits = (
            per_device_bf16_bits(combine_output_tensor) if local_output_memory_config is not None else None
        )

        # Exercise the cached-program path with a fresh optional output tensor. This catches stale
        # FullLocal combine runtime arguments, especially output addresses patched on cache hit.
        deallocate_l1_moe_compute_outputs(outputs)
        ttnn.deallocate(tt_combine_output_tensor)
        ttnn.synchronize_device(mesh_device)

        logger.info(f"\n========== Running op cache hit (compute_only={compute_only}) ==========")
        cache_hit_outputs = run_moe_compute_once(create_combine_output_tensor())
        assert (
            len(cache_hit_outputs) == expected_n
        ), f"compute_only={compute_only} cache hit must return {expected_n} tensors. Got {len(cache_hit_outputs)}."
        if multi_device_local:
            assert_replicated_outputs(cache_hit_outputs)

        cache_hit_combine_output_tensor = ttnn.to_memory_config(
            cache_hit_outputs[5], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        cache_hit_combine_all_passed = validate_combine_output(cache_hit_combine_output_tensor, base_pcc_threshold)
        logger.info(f"Combine Output Tensor Cache Hit: {'PASSED' if cache_hit_combine_all_passed else 'FAILED'}")
        assert cache_hit_combine_all_passed, "Combine output tensor cache-hit verification failed!"

        deallocate_l1_moe_compute_outputs(cache_hit_outputs)
        ttnn.deallocate(cache_hit_outputs[5])

        if local_output_memory_config is not None:
            # Same inputs and routing, the output in another placement: the writer addresses the
            # [k, T, H] output through a TensorAccessor built from the given buffer (one token row
            # per page for an interleaved or height-sharded row-major tensor), so every device's
            # rows must be bit-identical to the interleaved run above. The first call is a new
            # program (the output spec is in the hash); the second is its cache hit with a fresh
            # optional tensor, so the override must patch the caller-provided address.
            assert local_output_path, "local_output_memory_config needs the explicit-axis (local output) path"
            for pass_name in ("cache miss", "cache hit"):
                logger.info(f"\n========== Running op, output {local_output_memory_config} ({pass_name}) ==========")
                placed_output_tensor = create_combine_output_tensor(memory_config=local_output_memory_config)
                placed_outputs = run_moe_compute_once(
                    placed_output_tensor, output_memory_config=local_output_memory_config
                )
                assert len(placed_outputs) == expected_n, f"expected {expected_n} tensors, got {len(placed_outputs)}"
                if multi_device_local:
                    assert_replicated_outputs(placed_outputs)
                placed_output = placed_outputs[5]
                assert (
                    placed_output.buffer_address() == placed_output_tensor.buffer_address()
                ), "slot 5 must be the caller's optional_output_tensor"
                placed_memory_config = placed_output.memory_config()
                assert placed_memory_config.memory_layout == local_output_memory_config.memory_layout, (
                    f"output memory layout {placed_memory_config.memory_layout} != "
                    f"{local_output_memory_config.memory_layout}"
                )
                assert placed_memory_config.buffer_type == local_output_memory_config.buffer_type, (
                    f"output buffer type {placed_memory_config.buffer_type} != "
                    f"{local_output_memory_config.buffer_type}"
                )
                assert (
                    placed_memory_config == local_output_memory_config
                ), f"output memory config {placed_memory_config} != {local_output_memory_config}"
                placed_bits = per_device_bf16_bits(placed_output)
                assert len(placed_bits) == len(interleaved_bits)
                for device_idx, (placed, interleaved) in enumerate(zip(placed_bits, interleaved_bits)):
                    assert (
                        placed.shape == interleaved.shape
                    ), f"device {device_idx}: {placed.shape} != {interleaved.shape}"
                    mismatches = int((placed != interleaved).sum())
                    assert mismatches == 0, (
                        f"device {device_idx}: {mismatches} of {placed.numel()} bf16 values differ between the "
                        f"{local_output_memory_config.memory_layout} and the interleaved local output ({pass_name})"
                    )
                logger.info(
                    f"Local output {local_output_memory_config.memory_layout} {local_output_memory_config.buffer_type} "
                    f"({pass_name}): bitwise equal to the interleaved output on {len(placed_bits)} device(s)"
                )
                deallocate_l1_moe_compute_outputs(placed_outputs)
                ttnn.deallocate(placed_output)
                ttnn.synchronize_device(mesh_device)

        if multi_device_local:
            # A per-device (sharded) token set is rejected: every device must see the whole
            # replicated set. The check runs on cache hits too, so the cached program above
            # does not bypass it.
            tt_sharded_input = upload_tilize_input(sparse_buffer, ttnn.ShardTensorToMesh(mesh_device, dim=0))
            with expect_error(RuntimeError, r"fully replicated input topology"):
                run_moe_compute_once(create_combine_output_tensor(), tilize_input_tensor=tt_sharded_input)
            ttnn.deallocate(tt_sharded_input)


# DeepSeek-shaped workload mirrored on a single WH card so kernels see the same dims.
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW,
            "trace_region_size": 500000,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("compute_only", [True, False], ids=["compute_only", "fused_local"])
@pytest.mark.parametrize("has_bias", [False, True], ids=["no_bias", "with_bias"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_single_card_deepseek(mesh_device, mesh_shape, has_bias, compute_only, is_ci_env, is_ci_v2_env):
    """Single-card MoE compute on a 1x1 mesh, DeepSeek-shaped workload (hidden=7168).

    Runs in both compute_only mode (5 outputs, matmul is final) and fused-local mode
    (6 outputs, combine is final). The matmul ring size is auto-detected from the live
    DRAM-bank count (12 on WH, 7/8 on BH); the op no longer exposes a bh_ring_size knob.
    The width-shard dim must match the op's ring-aware derivation, so it is auto-derived.
    """
    hidden_size = 7168
    N = 2048
    ring_n = effective_matmul_ring_size(mesh_device)
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=16,
        tokens_per_device=32,
        selected_experts_k=8,
        N=N,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size, matmul_ring_size=ring_n),
        dtype=ttnn.bfloat16,
        activation_type=MoEActivationFunction.SILU,
        has_bias=has_bias,
        compute_only=compute_only,
        skip_on_ci=is_ci_env or is_ci_v2_env,
    )


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "trace_region_size": 500000,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("compute_only", [True, False], ids=["compute_only", "fused_local"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_single_card_gpt_oss(mesh_device, mesh_shape, compute_only, is_ci_env, is_ci_v2_env):
    """Single-card MoE compute on a 1x1 mesh, GPT-OSS-shaped workload (hidden=N=2880, SWIGLU+bias).

    Runs in both compute_only mode (5 outputs, matmul is final) and fused-local mode
    (6 outputs, combine is final). The matmul ring size is auto-detected from the live
    DRAM-bank count (12 on WH, 7/8 on BH); the op no longer exposes a bh_ring_size knob.
    """
    hidden_size = 2880
    ring_n = effective_matmul_ring_size(mesh_device)
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=16,
        tokens_per_device=32,
        selected_experts_k=4,
        N=hidden_size,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size, matmul_ring_size=ring_n),
        dtype=ttnn.bfloat16,
        activation_type=MoEActivationFunction.SWIGLU,
        has_bias=True,
        compute_only=compute_only,
        skip_on_ci=is_ci_env or is_ci_v2_env,
    )


# Regression sweep for tt-metal#50669 (correct output for non-tile-aligned token counts). Small hidden/N
# keep bf4 weight-prep fast; configs vary tilize_num_cores (largest divisor of hidden/32 <= 4): 512->4,
# 1344->3, 320->2, with activation/bias/k<E variety. Real model shapes are covered at tokens=32 above.
_MOE_50669_SWEEP_CONFIGS = [
    dict(
        name="c4_silu",
        experts_per_device=4,
        selected_experts_k=4,
        N=256,
        hidden_size=512,
        activation_type=MoEActivationFunction.SILU,
        has_bias=False,
    ),
    dict(
        name="c3_swiglu_bias",
        experts_per_device=4,
        selected_experts_k=4,
        N=256,
        hidden_size=1344,
        activation_type=MoEActivationFunction.SWIGLU,
        has_bias=True,
    ),
    dict(
        name="c2_partial",
        experts_per_device=8,
        selected_experts_k=4,
        N=256,
        hidden_size=320,
        activation_type=MoEActivationFunction.SILU,
        has_bias=False,
    ),
]

_MOE_50669_SWEEP_TOKENS = [1, 2, 3, 6, 16, 32, 48, 63, 64]


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 500000}],
    indirect=True,
)
@pytest.mark.parametrize("tokens_per_device", _MOE_50669_SWEEP_TOKENS)
@pytest.mark.parametrize("cfg", _MOE_50669_SWEEP_CONFIGS, ids=lambda c: c["name"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_single_card_nontile_tokens_sweep(mesh_device, mesh_shape, cfg, tokens_per_device):
    """Regression for tt-metal#50669: correctness across non-tile-aligned token counts / configs."""
    arch = mesh_device.arch()
    ring_n = effective_matmul_ring_size(mesh_device)
    N = max(cfg["N"], 32 * ring_n)
    worker_grid = mesh_device.compute_with_storage_grid_size()
    # The c4 sweep needs 12 matmul + 16 combine + 4 tilize cores on WH. Even after padding N to the
    # ring width, that layout cannot be placed on the harvested wh_n300_civ2 7x8 worker grid.
    if arch == ttnn.device.Arch.WORMHOLE_B0 and cfg["name"] == "c4_silu" and worker_grid.x <= 7 and worker_grid.y <= 8:
        pytest.xfail(
            f"moe_compute cannot place the c4_silu core layout on the {worker_grid.x}x{worker_grid.y} "
            "Wormhole worker grid; https://github.com/tenstorrent/tt-metal/issues/52246"
        )
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=cfg["experts_per_device"],
        tokens_per_device=tokens_per_device,
        selected_experts_k=cfg["selected_experts_k"],
        N=N,
        hidden_size=cfg["hidden_size"],
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(cfg["hidden_size"], matmul_ring_size=ring_n),
        dtype=ttnn.bfloat16,
        activation_type=cfg["activation_type"],
        has_bias=cfg["has_bias"],
        matmul_xfail_on_bh=True,  # metadata asserted on BH; matmul (#50038) xfailed
    )


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW, "trace_region_size": 500000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_single_card_full_local_b1(mesh_device, mesh_shape):
    """Regression for tt-metal#52371: B=1 dense token-map stride in FullLocal mode."""
    hidden_size = 2048
    ring_n = effective_matmul_ring_size(mesh_device)
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=16,
        tokens_per_device=1,
        selected_experts_k=8,
        N=512,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size, matmul_ring_size=ring_n),
        dtype=ttnn.bfloat16,
        activation_type=MoEActivationFunction.SILU,
        has_bias=False,
        compute_only=False,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW, "trace_region_size": 500000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_single_card_explicit_axis0(mesh_device, mesh_shape):
    """cluster_axis=0 on a 1x1 mesh names an axis of extent 1: the fabric path degenerates to the
    local combine and matches the implicit cluster_axis=None call."""
    hidden_size = 2048
    ring_n = effective_matmul_ring_size(mesh_device)
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=16,
        tokens_per_device=1,
        selected_experts_k=8,
        N=512,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size, matmul_ring_size=ring_n),
        dtype=ttnn.bfloat16,
        activation_type=MoEActivationFunction.SILU,
        has_bias=False,
        compute_only=False,
        op_cluster_axis=0,
    )


def _height_sharded_rows_memory_config(buffer_type, num_rows, hidden_size, num_cores):
    """[num_rows, hidden_size] row-major HEIGHT_SHARDED over num_cores cores in a row of the grid,
    whole token rows per shard: every page is one 2 x H byte row, the same page rule as the
    interleaved output. A DRAM grid is bank coordinates (bank_id, 0); the last shard may be
    partial when num_cores does not divide num_rows."""
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))])
    rows_per_shard = -(-num_rows // num_cores)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        buffer_type,
        ttnn.ShardSpec(grid, [rows_per_shard, hidden_size], ttnn.ShardOrientation.ROW_MAJOR),
    )


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW, "trace_region_size": 500000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@pytest.mark.parametrize("output_buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM], ids=["l1", "dram"])
def test_moe_compute_single_card_explicit_axis0_height_sharded_output(mesh_device, mesh_shape, output_buffer_type):
    """The local output path (cluster_axis=0 on a 1x1 mesh) with the [k, T, H] output HEIGHT_SHARDED
    row-major, whole token rows per shard (k*T = 64 rows over 8 cores, 8 rows each): the writer
    addresses the output through a TensorAccessor built from the given buffer, one row per page,
    so the result must be bitwise equal to the interleaved output of the same inputs and seed.
    L1 shards on 8 worker cores; DRAM shards on the live DRAM banks (8 on an unharvested
    Blackhole, 7 on a harvested one, 12 on Wormhole)."""
    hidden_size = 2048
    tokens_per_device = 8
    selected_experts_k = 8
    ring_n = effective_matmul_ring_size(mesh_device)
    num_shard_cores = 8 if output_buffer_type == ttnn.BufferType.L1 else min(8, ring_n)
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=16,
        tokens_per_device=tokens_per_device,
        selected_experts_k=selected_experts_k,
        N=512,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size, matmul_ring_size=ring_n),
        dtype=ttnn.bfloat16,
        activation_type=MoEActivationFunction.SILU,
        has_bias=False,
        compute_only=False,
        op_cluster_axis=0,
        local_output_memory_config=_height_sharded_rows_memory_config(
            output_buffer_type, selected_experts_k * tokens_per_device, hidden_size, num_shard_cores
        ),
    )


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW, "trace_region_size": 500000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 4), (1, 4))], indirect=["mesh_device"])
@pytest.mark.parametrize("ccl_knobs", [False, True], ids=["no_ccl_knobs", "galaxy_ccl_knobs"])
def test_moe_compute_multi_device_local_axis(mesh_device, mesh_shape, expect_error, ccl_knobs):
    """cluster_axis=0 on a 1x4 mesh names an axis of extent 1: the local output path. No combine
    kernels run; moe_compute's writer puts each expert's token rows straight into the final
    output at every device, on a mesh opened without a fabric config. The replicated token set
    and identical routing metadata go to every device, each device computes its own expert
    shard, every device's partial matches the golden rows its experts own, all outputs keep the
    replicated input topology, and a dim-0-sharded input is rejected. With ccl_knobs the Galaxy-style topology / num_links / mux
    range / cross-device semaphore are passed too and must be accepted and unused. The
    mesh_device fixture skips this on machines with fewer than four devices."""
    if mesh_device.get_num_devices() < 2:
        pytest.skip("multi-device local combine needs at least two devices")
    hidden_size = 2048
    ring_n = effective_matmul_ring_size(mesh_device)
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=16,
        tokens_per_device=8,
        selected_experts_k=8,
        N=512,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size, matmul_ring_size=ring_n),
        dtype=ttnn.bfloat16,
        activation_type=MoEActivationFunction.SILU,
        has_bias=False,
        compute_only=False,
        op_cluster_axis=0,
        expect_error=expect_error,
        ccl_knobs=ccl_knobs,
    )


def _minimal_rejection_inputs(mesh_device):
    """Replicated inputs with valid shapes for argument-validation tests. The op must reject
    the bad argument combination before any kernel launch, so the weights are placeholders."""
    hidden_size = 7168
    tokens_per_device = 32
    experts = 8
    selected_experts_k = 8

    sparse = torch.zeros(1, tokens_per_device, hidden_size, dtype=torch.bfloat16)
    indices = torch.zeros(1, tokens_per_device, selected_experts_k, dtype=torch.uint16)
    scores = torch.zeros(1, tokens_per_device, selected_experts_k, dtype=torch.bfloat16)
    mapping = torch.zeros(1, experts, dtype=torch.uint16)

    tt_sparse = ttnn.from_torch(
        sparse,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_indices = ttnn.from_torch(
        indices,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_scores = ttnn.from_torch(
        scores,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_mapping = ttnn.from_torch(
        mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    # Dummy weights: rank 6 like the real packed weights so the device op's rank checks pass
    # and a rejection test can reach the path-specific validation; the values never run.
    dummy_weight = torch.zeros(1, 1, 1, 1, 32, 32, dtype=torch.bfloat16)
    tt_w0_w1 = ttnn.from_torch(
        dummy_weight,
        device=mesh_device,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_w2 = ttnn.from_torch(
        dummy_weight,
        device=mesh_device,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return (tt_sparse, tt_indices, tt_scores, tt_mapping, tt_w0_w1, tt_w2)


def _call_moe_compute_for_rejection(mesh_device, **overrides):
    kwargs = dict(
        layer_id=0,
        output_height_shard_dim=4,
        intermediate_size=2048,
        has_bias=False,
        cluster_axis=None,
        topology=None,
        num_links=None,
        mux_core_range_set=None,
        optional_output_tensor=None,
        optional_cross_device_semaphore=None,
        activation_type=MoEActivationFunction.SILU,
        compute_only=False,
    )
    kwargs.update(overrides)
    return ttnn.experimental.moe_compute(*_minimal_rejection_inputs(mesh_device), **kwargs)


# Minimal sanity check that compute_only=True with conflicting CCL kwargs is rejected.
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_compute_only_rejects_cluster_axis(mesh_device, mesh_shape, expect_error):
    """compute_only=True with cluster_axis set must raise (loud rejection per spec)."""
    with expect_error(RuntimeError, r"compute_only.*cluster_axis"):
        _call_moe_compute_for_rejection(mesh_device, compute_only=True, cluster_axis=1)


@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_local_axis_rejects_shared_experts(mesh_device, mesh_shape, expect_error):
    """cluster_axis over an axis of extent 1 (the local output path) leaves one partial per device
    that the caller sums, so it has no shared-expert path; the 1x1 cluster_axis=None call keeps
    shared experts. The rejection fires at the op boundary before any fabric lookup."""
    with expect_error(RuntimeError, r"mesh axis of extent 1 writes a local output.*shared experts"):
        _call_moe_compute_for_rejection(mesh_device, cluster_axis=0, num_shared_experts_per_device=1)


@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_local_axis_rejects_width_sharded_output(mesh_device, mesh_shape, expect_error):
    """The local output path writes one token row per TensorAccessor page. A WIDTH_SHARDED (or
    BLOCK / ND sharded) row-major [k, T, H] output has (1, shard width) pages, so a row would span
    several pages and a core's slice would need a column-page split that is not implemented
    (the combine writer has the same limitation); the op says so before any kernel launch.
    Shapes are those of _minimal_rejection_inputs (k=8, T=32, H=7168)."""
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))])
    width_sharded = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [8 * 32, 7168 // 16], ttnn.ShardOrientation.ROW_MAJOR),
    )
    with expect_error(RuntimeError, r"does not split a row across column pages"):
        _call_moe_compute_for_rejection(mesh_device, cluster_axis=0, output_memory_config=width_sharded)


@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 4), (1, 4))], indirect=["mesh_device"])
def test_moe_compute_axis_extent_selects_path(mesh_device, mesh_shape, expect_error):
    """On a 1x4 mesh opened without a fabric config, cluster_axis=0 (extent 1) never consults the
    fabric: with topology=None the fabric topology lookup is skipped and an explicit num_links=0
    reaches the op's own num_links check. cluster_axis=1 (extent 4) is the fabric combine: the
    same call stops at the fabric topology lookup ("un-initialized fabric context") on this
    fixture, or at the num_links check when a fabric is up."""
    if mesh_device.get_num_devices() < 2:
        pytest.skip("needs a mesh with an axis of extent > 1")
    with expect_error(RuntimeError, r"num_links must be greater than 0"):
        _call_moe_compute_for_rejection(mesh_device, cluster_axis=0, topology=None, num_links=0)
    with expect_error(RuntimeError, r"num_links must be greater than 0|un-initialized fabric context"):
        _call_moe_compute_for_rejection(mesh_device, cluster_axis=1, topology=None, num_links=0)
