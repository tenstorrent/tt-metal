# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Single-card MoE compute test (1x1 mesh, cluster_axis=None). Runs on both WH
and BH; other arches are skipped at fixture time. A few tests use a (1, 4) mesh with
cluster_axis=0: an axis of extent 1 has nothing to combine, so the op takes its local output
path at every device (no combine kernels, no fabric) and returns one partial per device that
writes only the rows of its own experts; the mesh_device fixture skips them on a single card.

This test exercises both paths of `ttnn.experimental.moe_compute` on a single device:
  - `compute_only=True`: bypasses the fused selective_reduce_combine stage entirely.
    Returns 5 tensors; matmul_output (slot 4) is the final output.
  - `compute_only=False` (SingleDevice): runs the fused local combine stage without CCL/fabric.
    Returns 6 tensors; combine_output (slot 5) is the final output.

It is the hermetic dev/regression net for the MoE compute kernels (tilize + matmul +
activation [+ combine]) without requiring a 6U Galaxy host or working CCL-on-BH.

Validation points (all using the 6U helpers verbatim — no logic duplication):
  - Output 0 (per_expert_total_tokens)
  - Output 1 (expert_activation)
  - Output 2 (e_t)
  - Output 4 (matmul_output) — final output in compute_only mode
  - Output 5 (combine_output) — final output in SingleDevice mode, validated only when
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
    decode_packed_token_lists,
    token_list_header_words,
    token_list_page_words,
    token_list_segment_starts,
)

# Reuse 6U test helpers verbatim. The intent is that this single-card test
# never duplicates compute logic — same goldens, same validators.
from tests.nightly.tg.ccl.moe.test_moe_compute_6U import (
    create_torch_w0,
    create_torch_w1,
    create_torch_w2,
    compute_e_t_golden,
    compute_e_t_k_slot_golden,
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
    validate_packed_token_lists,
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
    check_writes_owned_rows_only=False,
    num_shared_experts_per_device=0,
):
    """
    Single-card MoE compute test body. The op is called with cluster_axis=op_cluster_axis:
    None (the 1x1 SingleDevice path, fused local combine) or 0, an axis of extent 1 on every mesh
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

    check_writes_owned_rows_only (1xN local output only): the output contract is that a device
    writes the rows of the experts it holds and leaves every other row of [k, T, H] as the buffer
    held it, like the combine. Checked on an op-allocated output (owned rows only), on a caller
    tensor pre-filled with a sentinel, and on that same tensor reused after a routing change
    (its other rows then hold the previous routing's results, bit for bit).

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
    if num_shared_experts_per_device:
        assert num_shared_experts_per_device == 1
        assert num_devices == 1 and not compute_only and op_cluster_axis is None
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
        experts - num_shared_experts_per_device,
        selected_experts_k,
        mesh_shape,
        cluster_axis,
        dtype=tt_to_torch_dtype(dtype),
    )
    if num_shared_experts_per_device:
        # A 1x1 shared expert has TP factor one and the same packed weights as a routed
        # expert. Route every token to the final expert with a nonzero score, so the
        # control exercises its computation rather than only admitting the argument.
        expert_indices[..., -1] = experts - 1
        expert_scores[..., -1] = 1.0

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
        # cluster_axis=None: required for compute_only and for the implicit 1x1 SingleDevice call,
        # and topology/num_links/mux/semaphore must be None there. cluster_axis=0 names an axis
        # of extent 1 (1x1 or 1xN): the fabric path with no neighbours, run as a local combine at
        # every mesh coordinate; the CCL arguments may be None.
        ccl_kwargs = dict(topology=None, num_links=None, mux_core_range_set=None, optional_cross_device_semaphore=None)
    if num_shared_experts_per_device:
        ccl_kwargs["num_shared_experts_per_device"] = num_shared_experts_per_device

    def run_moe_compute_once(
        optional_combine_output_tensor,
        tilize_input_tensor=None,
        output_memory_config=None,
        expert_indices_tensor=None,
        expert_scores_tensor=None,
    ):
        return ttnn.experimental.moe_compute(
            tt_sparse_buffer if tilize_input_tensor is None else tilize_input_tensor,
            tt_expert_indices if expert_indices_tensor is None else expert_indices_tensor,
            tt_expert_scores if expert_scores_tensor is None else expert_scores_tensor,
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

    def owner_of_slots(indices_flat):
        # [K, total_tokens]: the device whose experts own output slot (k, t).
        return expert_mapping[0].long()[indices_flat.long()].transpose(0, 1)

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
        slot_owner = owner_of_slots(expert_indices_flat)
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
    # - compute_only=False (SingleDevice): 6 tensors (slot 5 = combine output).
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
            combine_output_tensor,  # slot 5 -- final output in SingleDevice
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

    # The local output path (an explicit cluster_axis of extent 1) writes the final tensor directly
    # and never stages the expert outputs in the combine cores' L1, so slot 4 holds whatever the
    # shared buffer last carried; the final output below is the validated artifact there. It builds
    # no activation rows (slot 1 is a placeholder) and keeps the expert -> token lists packed (slot 2).
    local_output_path = not compute_only and op_cluster_axis is not None
    if local_output_path:
        activation_all_passed = True
        logger.info("Expert Activation Tensor: not produced on the local output path (placeholder), skipped")
        e_t_all_passed = validate_packed_token_lists(
            mesh_device,
            experts_per_device,
            num_devices,
            e_t_output_tensor,
            golden_e_t,
            compute_e_t_k_slot_golden(expert_indices, expert_mapping, mesh_shape, cluster_axis),
        )
    else:
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
        # SingleDevice combine runtime arguments, especially output addresses patched on cache hit.
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

        if check_writes_owned_rows_only:
            # The output contract on a 1xN mesh: a device writes the rows of the experts it holds
            # and leaves every other row of [k, T, H] as the buffer held it. Owned rows are checked
            # against the golden; the other rows, when their previous content is known, bitwise (an
            # int16 view: NaN != NaN and -0.0 == 0.0 cannot hide a write) against that content.
            assert (
                multi_device_local and local_output_path
            ), "check_writes_owned_rows_only needs the 1xN local output path"

            def assert_partial_output(tt_output, slot_owner, goldens, what, previous_bits=None):
                output_ref, output_data_map = goldens
                for device_idx, device_tensor in enumerate(ttnn.get_device_tensors(tt_output)):
                    rows = ttnn.to_torch(device_tensor, mesh_composer=None)
                    assert rows.shape == (
                        selected_experts_k,
                        total_tokens,
                        hidden_size,
                    ), f"{what}: device {device_idx} output shape {tuple(rows.shape)}"
                    owned = slot_owner == device_idx  # [K, total_tokens]
                    num_owned = int(owned.sum())
                    num_non_owned = int((~owned).sum())
                    assert num_owned > 0 and num_non_owned > 0, (
                        f"{what}: device {device_idx} owns {num_owned} of {owned.numel()} slots; the check needs "
                        "both owned and non-owned rows"
                    )
                    if previous_bits is not None:
                        non_owned_bits = rows.contiguous().view(torch.int16)[~owned]
                        changed = int((non_owned_bits != previous_bits[device_idx][~owned]).sum())
                        assert changed == 0, (
                            f"{what}: device {device_idx} changed {changed} bf16 values in the {num_non_owned} rows "
                            "its experts do not own"
                        )
                    passed = validate_combine_torch(
                        layer_id, rows, (output_ref, output_data_map * owned.unsqueeze(0)), base_pcc_threshold
                    )
                    assert passed, f"{what}: device {device_idx} owned rows do not match the golden"
                    other_rows = "unchanged" if previous_bits is not None else "not checked"
                    logger.info(
                        f"{what}: device {device_idx} owned rows ({num_owned}) match the golden, "
                        f"{num_non_owned} other rows {other_rows}"
                    )

            slot_owner = owner_of_slots(expert_indices_flat)

            logger.info("\n========== Running op, op-allocated output ==========")
            allocated_outputs = run_moe_compute_once(None)
            assert len(allocated_outputs) == expected_n, f"expected {expected_n} tensors, got {len(allocated_outputs)}"
            assert_partial_output(allocated_outputs[5], slot_owner, combine_goldens, "Op-allocated output")
            deallocate_l1_moe_compute_outputs(allocated_outputs)
            ttnn.deallocate(allocated_outputs[5])

            logger.info("\n========== Running op, caller tensor pre-filled with 1.0 ==========")
            sentinel_tensor = ttnn.from_torch(
                torch.ones([selected_experts_k, total_tokens, hidden_size], dtype=torch.bfloat16),
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=token_mesh_mapper,
            )
            sentinel_bits = per_device_bf16_bits(sentinel_tensor)
            sentinel_outputs = run_moe_compute_once(sentinel_tensor)
            assert (
                sentinel_outputs[5].buffer_address() == sentinel_tensor.buffer_address()
            ), "slot 5 must be the caller's optional_output_tensor"
            assert_partial_output(
                sentinel_outputs[5], slot_owner, combine_goldens, "Sentinel-filled caller tensor", sentinel_bits
            )
            deallocate_l1_moe_compute_outputs(sentinel_outputs)

            # A routing change: new tokens, indices and scores over the same weights, run through the
            # cached program into the same caller tensor, which still holds the previous results.
            torch.manual_seed(2004)
            random.seed(2004)
            sparse_buffer_b, expert_indices_b, expert_scores_b, original_tokens_b = gen_sparse_buffer_and_indices(
                tokens_per_device,
                hidden_size,
                experts,
                selected_experts_k,
                mesh_shape,
                cluster_axis,
                dtype=tt_to_torch_dtype(dtype),
            )
            tilize_golden_b, _ = compute_selective_tilize_golden(
                sparse_buffer_b, expert_indices_b, expert_scores_b, expert_mapping, mesh_shape, cluster_axis
            )
            activation_b, _ = compute_expert_activation_golden(
                expert_indices_b, expert_scores_b, expert_mapping, mesh_shape, cluster_axis
            )
            matmul_goldens_b = compute_matmul_golden(
                tilize_golden_b.unsqueeze(0),
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
            combine_goldens_b = compute_combine_golden(
                num_layers,
                experts,
                total_tokens,
                hidden_size,
                selected_experts_k,
                mesh_shape,
                matmul_goldens_b,
                [activation_b],
                cluster_axis=-1,
            )
            expert_indices_b_flat = expert_indices_b.reshape(total_tokens, selected_experts_k)
            slot_owner_b = owner_of_slots(expert_indices_b_flat)
            assert not torch.equal(
                slot_owner_b, slot_owner
            ), "the routing change must move at least one output slot to another device"
            tt_input_b = upload_tilize_input(original_tokens_b.reshape(1, total_tokens, hidden_size), token_mesh_mapper)
            tt_indices_b = ttnn.from_torch(
                expert_indices_b_flat.unsqueeze(0),
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint16,
                memory_config=expert_indices_mem_config,
                mesh_mapper=token_mesh_mapper,
            )
            tt_scores_b = ttnn.from_torch(
                expert_scores_b.reshape(total_tokens, selected_experts_k).unsqueeze(0),
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=dtype,
                memory_config=expert_scores_mem_config,
                mesh_mapper=token_mesh_mapper,
            )
            logger.info("\n========== Running op, caller tensor reused after a routing change ==========")
            previous_bits = per_device_bf16_bits(sentinel_tensor)
            reused_outputs = run_moe_compute_once(
                sentinel_tensor,
                tilize_input_tensor=tt_input_b,
                expert_indices_tensor=tt_indices_b,
                expert_scores_tensor=tt_scores_b,
            )
            assert (
                reused_outputs[5].buffer_address() == sentinel_tensor.buffer_address()
            ), "slot 5 must be the caller's optional_output_tensor"
            assert_partial_output(
                reused_outputs[5],
                slot_owner_b,
                combine_goldens_b,
                "Caller tensor reused after a routing change",
                previous_bits,
            )
            deallocate_l1_moe_compute_outputs(reused_outputs)
            for tensor in (sentinel_tensor, tt_input_b, tt_indices_b, tt_scores_b):
                ttnn.deallocate(tensor)
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
            "l1_small_size": 16384,
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
            "l1_small_size": 16384,
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
@pytest.mark.parametrize("tokens_per_device", [1, 32], ids=["t1", "t32"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_single_card_flash_next(mesh_device, mesh_shape, compute_only, tokens_per_device):
    """Single-card MoE compute on a 1x1 mesh, Qwen3.8-Flash-Next-shaped workload (hidden=2560, N=640, SILU).

    512 routed experts over four devices = 128 experts per device. N=640 is 20 intermediate tiles, so on an
    8-bank Blackhole ring the cores own 3 or 2 gate/up columns (odd and even counts on one ring), and on a
    12-core Wormhole ring 2 or 1.
    """
    hidden_size = 2560
    ring_n = effective_matmul_ring_size(mesh_device)
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=128,
        tokens_per_device=tokens_per_device,
        selected_experts_k=8,
        N=640,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size, matmul_ring_size=ring_n),
        dtype=ttnn.bfloat16,
        activation_type=MoEActivationFunction.SILU,
        has_bias=False,
        compute_only=compute_only,
    )


# Other public MoE expert shapes, one per W0/W1 layout case on an 8-bank ring: compact (the busiest core owns fewer
# columns than the uniform even stride) and uniform stride (6/5 and 8/7 columns), plus bias rows on the two shapes
# whose cores own an odd column count (the half block-column path with the bias tile row; with bias every shape
# stores 14-tile transactions, so the half-width W2 iteration is not reachable). name: (hidden, expert intermediate,
# top-k, experts per device, activation, has_bias); 16-32 experts keep the BF4 weight preparation short.
_MOE_OTHER_SHAPES = {
    "qwen36_35b_a3b": (2048, 512, 8, 32, MoEActivationFunction.SILU, False),  # 2 columns per core: compact
    "gemma4_26b_a4b": (2816, 704, 8, 32, MoEActivationFunction.GELU, False),  # 3/2 columns: compact
    "gemma4_26b_a4b_bias": (2816, 704, 8, 32, MoEActivationFunction.GELU, True),  # 3/2 columns + bias tile row
    "flash_next_bias": (2560, 640, 10, 16, MoEActivationFunction.SILU, True),  # 3/2 columns + bias (14-tile)
    "glm45_air": (4096, 1408, 8, 32, MoEActivationFunction.SILU, False),  # 6/5 columns: uniform stride
    "nemotron3_nano": (2688, 1856, 6, 16, MoEActivationFunction.SILU, False),  # 8/7 columns: uniform stride
}


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 500000}],
    indirect=True,
)
@pytest.mark.parametrize("shape", sorted(_MOE_OTHER_SHAPES))
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_single_card_other_shapes(mesh_device, mesh_shape, shape):
    """Single-card MoE compute on a 1x1 mesh for other public expert shapes (compute_only)."""
    hidden_size, intermediate, k, experts, activation, has_bias = _MOE_OTHER_SHAPES[shape]
    ring_n = effective_matmul_ring_size(mesh_device)
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=experts,
        tokens_per_device=32,
        selected_experts_k=k,
        N=intermediate,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size, matmul_ring_size=ring_n),
        dtype=ttnn.bfloat16,
        activation_type=activation,
        has_bias=has_bias,
        compute_only=True,
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
    [{"l1_small_size": 16384, "dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 500000}],
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
    [{"l1_small_size": 16384, "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW, "trace_region_size": 500000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_single_card_full_local_b1(mesh_device, mesh_shape):
    """Regression for tt-metal#52371: B=1 dense token-map stride in SingleDevice mode."""
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
    [{"l1_small_size": 16384, "dispatch_core_axis": ttnn.DispatchCoreAxis.ROW, "trace_region_size": 500000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@pytest.mark.parametrize("tokens_per_device", [1, 33])
def test_moe_compute_single_card_full_local_shared_expert(mesh_device, mesh_shape, tokens_per_device):
    """Preserve the existing 1x1 shared-expert path against the independent routed-expert golden."""
    hidden_size = 2048
    ring_n = effective_matmul_ring_size(mesh_device)
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=16,
        tokens_per_device=tokens_per_device,
        selected_experts_k=8,
        N=512,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size, matmul_ring_size=ring_n),
        dtype=ttnn.bfloat16,
        activation_type=MoEActivationFunction.SILU,
        compute_only=False,
        num_shared_experts_per_device=1,
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


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.ROW, "trace_region_size": 500000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 4), (1, 4))], indirect=["mesh_device"])
def test_moe_compute_multi_device_local_axis_writes_owned_rows_only(mesh_device, mesh_shape, expect_error):
    """The local output on a 1x4 mesh is one partial per device that writes only the rows of the
    experts the device holds and leaves the other rows as the buffer held them, like the combine.
    Checked on an op-allocated output, on a caller tensor pre-filled with 1.0, and on that tensor
    reused after a routing change (its other rows then hold the previous routing's results). The
    mesh_device fixture skips this on machines with fewer than four devices."""
    if mesh_device.get_num_devices() < 2:
        pytest.skip("the owned-row check needs at least two devices")
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
        check_writes_owned_rows_only=True,
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


def _call_moe_compute_for_rejection(mesh_device, inputs=None, **overrides):
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
    return ttnn.experimental.moe_compute(
        *(_minimal_rejection_inputs(mesh_device) if inputs is None else inputs), **kwargs
    )


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
def test_moe_compute_local_output_rejects_four_rings(mesh_device, mesh_shape, expect_error):
    """prefill_rings admits the replay ring (1) and two or three rings (2, 3); more rings are not implemented and the
    op says so before any kernel launch."""
    with expect_error(RuntimeError, r"prefill_rings=4: one replay ring \(1\), two or three rings \(2, 3\)"):
        _call_moe_compute_for_rejection(mesh_device, cluster_axis=0, zero_fill_non_owned_rows=False, prefill_rings=4)


@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_local_output_rejects_zero_fill_with_two_rings(mesh_device, mesh_shape, expect_error):
    """With two rings the zero fill of the unowned rows would be written by both rings' cores and could land after
    the other ring's row writes; the op requires zero_fill_non_owned_rows=False (the slab's setting)."""
    with expect_error(RuntimeError, r"prefill_rings=2: the zero fill of the unowned rows"):
        _call_moe_compute_for_rejection(mesh_device, cluster_axis=0, zero_fill_non_owned_rows=True, prefill_rings=2)


@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@pytest.mark.parametrize("prefill_rings", [1, 2, 3])
def test_moe_compute_local_output_rejects_wormhole_replay(mesh_device, mesh_shape, prefill_rings, expect_error):
    if mesh_device.arch() != ttnn.device.Arch.WORMHOLE_B0:
        pytest.skip("Wormhole replay admission control")
    with expect_error(RuntimeError, "prefill_rings>0 is supported only on Blackhole"):
        _call_moe_compute_for_rejection(
            mesh_device, cluster_axis=0, zero_fill_non_owned_rows=False, prefill_rings=prefill_rings
        )


@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_pipeline_rejects_prefill_ring(mesh_device, mesh_shape, expect_error):
    with expect_error(RuntimeError, "enable_a2a_pipeline requires the streaming ring"):
        _call_moe_compute_for_rejection(mesh_device, cluster_axis=0, prefill_rings=1, enable_a2a_pipeline=True)


@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_local_axis_rejects_width_sharded_output(mesh_device, mesh_shape, expect_error):
    """The local output path writes one token row per TensorAccessor page. A WIDTH_SHARDED (or
    BLOCK / ND sharded) row-major [k, T, H] output has (1, shard width) pages, so a row would span
    several pages and a core's slice would need a column-page split that is not implemented
    (the combine's own validation applies the same rule); the op says so before any kernel launch.
    Shapes are those of _minimal_rejection_inputs (k=8, T=32, H=7168)."""
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))])
    width_sharded = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [8 * 32, 7168 // 16], ttnn.ShardOrientation.ROW_MAJOR),
    )
    with expect_error(RuntimeError, r"a row is not split across column pages"):
        _call_moe_compute_for_rejection(mesh_device, cluster_axis=0, output_memory_config=width_sharded)


@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 4), (1, 4))], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "sharded_input", ["expert indices", "expert scores", "expert mapping"], ids=["indices", "scores", "mapping"]
)
@pytest.mark.parametrize("local_combine", [False, True], ids=["local_output", "fused_local"])
def test_moe_compute_local_axis_rejects_sharded_routing_metadata(
    mesh_device, mesh_shape, expect_error, sharded_input, local_combine
):
    """On a multi-device mesh the local output path reads the token set and its routing metadata
    (expert indices, scores and mapping) as the same full set at every coordinate, so each of the
    four must be fully replicated, not only the activations: a dim-0-sharded copy of one of the
    three routing tensors with replicated activations is rejected before any kernel launch, naming
    the tensor. Shapes are those of _minimal_rejection_inputs (T=32, k=8, 8 experts); a 1x1 mesh has
    no topology to check, so the mesh_device fixture skips this on a single card."""
    if mesh_device.get_num_devices() < 2:
        pytest.skip("a 1x1 mesh has no sharded topology to reject")
    num_devices = mesh_device.get_num_devices()
    tokens_per_device, selected_experts_k, experts = 32, 8, 8
    tt_sparse, tt_indices, tt_scores, tt_mapping, tt_w0_w1, tt_w2 = _minimal_rejection_inputs(mesh_device)

    def dim0_sharded(torch_tensor, tt_dtype):
        return ttnn.from_torch(
            torch_tensor,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=tt_dtype,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )

    if sharded_input == "expert indices":
        tt_indices = dim0_sharded(
            torch.zeros(num_devices, tokens_per_device, selected_experts_k, dtype=torch.uint16), ttnn.uint16
        )
    elif sharded_input == "expert scores":
        tt_scores = dim0_sharded(
            torch.zeros(num_devices, tokens_per_device, selected_experts_k, dtype=torch.bfloat16), ttnn.bfloat16
        )
    else:
        tt_mapping = dim0_sharded(torch.zeros(num_devices, experts, dtype=torch.uint16), ttnn.uint16)
    with expect_error(RuntimeError, rf"fully replicated {sharded_input} topology"):
        _call_moe_compute_for_rejection(
            mesh_device,
            inputs=(tt_sparse, tt_indices, tt_scores, tt_mapping, tt_w0_w1, tt_w2),
            cluster_axis=0,
            local_combine=local_combine,
        )


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


# ---------------------------------------------------------------------------------------------------------------------
# The local output path at prefill token counts. One call over T tokens must write the same (token, expert) pages as
# the fused local combine writes over the same rows in 128-token calls (today's production slab form): the ring
# kernels never see the token count and every row's arithmetic is independent of its chunk-mates, so only the chunk
# sequence differs. The routing tensors name 512 experts of which this device holds 128 (the mapping row sends the
# other 384 to devices 1..3 that a 1x1 mesh does not have: exactly what one device of a 1x4 line sees).
# ---------------------------------------------------------------------------------------------------------------------

_PREFILL_HIDDEN = 2560
_PREFILL_N = 640
_PREFILL_EXPERTS_PER_DEVICE = 128
_PREFILL_EXPERTS = 512
_PREFILL_K = 10
_PREFILL_REFERENCE_TOKENS = 128
_PREFILL_TOKENS_PER_CHUNK = 32


def _prefill_routing(case, tokens, seed=20260925):
    """``(indices [tokens, K] int64 of global expert ids, scores [tokens, K] bf16)`` for one routing case. Local
    experts are 0..127; ids 128..511 belong to other devices and produce no local pair."""
    k = _PREFILL_K
    experts = _PREFILL_EXPERTS
    local = _PREFILL_EXPERTS_PER_DEVICE
    generator = torch.Generator().manual_seed(seed)
    slots = torch.arange(k)

    def distinct_remote(num_slots):
        # num_slots distinct ids >= local for every token (num_slots < 384, so the arithmetic progression is distinct).
        return local + (num_slots * torch.arange(tokens)[:, None] + torch.arange(num_slots)[None, :]) % (
            experts - local
        )

    if case == "zipf":
        # A hot head over a random permutation of the 512 experts: about a quarter of the pairs land here,
        # with one or a few experts taking hundreds of tokens (the captured natural-text shape).
        weights = 1.0 / torch.arange(1, experts + 1, dtype=torch.float32) ** 0.8
        weights = weights[torch.randperm(experts, generator=generator)]
        indices = torch.multinomial(weights.expand(tokens, -1), k, replacement=False, generator=generator)
    elif case == "one_expert":
        # Every token routes slot 0 to local expert 5: one segment of `tokens` entries, tokens / 32 chunks.
        indices = torch.empty(tokens, k, dtype=torch.int64)
        indices[:, 0] = 5
        indices[:, 1:] = distinct_remote(k - 1)
    elif case == "uniform_local":
        # Every slot local, every expert with tokens * K / 128 entries: the packed lists at their capacity.
        indices = (k * torch.arange(tokens)[:, None] + slots[None, :]) % local
    elif case == "random_local":
        # Every slot local with random experts: capacity again, skewed segment lengths.
        indices = torch.stack([torch.randperm(local, generator=generator)[:k] for _ in range(tokens)])
    elif case == "no_local":
        # No local pair at all: an empty call (zero chunks) that still writes its zero rows.
        indices = distinct_remote(k)
    else:
        raise ValueError(case)
    scores = torch.rand(tokens, k, generator=generator) + 0.05
    scores = (scores / scores.sum(dim=1, keepdim=True)).to(torch.bfloat16)
    return indices.to(torch.int64), scores


def _prefill_local_lists(indices):
    """Per local expert, the (token ids, k slots) of its pairs in ascending token order (the list order the tilize
    cores produce), and the per-expert counts."""
    lists = []
    counts = torch.zeros(_PREFILL_EXPERTS_PER_DEVICE, dtype=torch.int64)
    for e in range(_PREFILL_EXPERTS_PER_DEVICE):
        where = torch.nonzero(indices == e)  # row-major: ascending token
        lists.append((where[:, 0].tolist(), where[:, 1].tolist()))
        counts[e] = where.shape[0]
    return lists, counts


def _bf16_bits(tensor):
    return ttnn.to_torch(tensor).contiguous().view(torch.int16)


@torch.no_grad()
def _run_moe_compute_local_output_prefill_test(
    mesh_device,
    tokens,
    routing_case,
    zero_fill=True,
    sentinel_output=False,
    second_routing_case=None,
    prefill_rings=0,
):
    arch = mesh_device.arch()
    if arch not in (ttnn.device.Arch.WORMHOLE_B0, ttnn.device.Arch.BLACKHOLE):
        pytest.skip(f"MoE compute single-card test: arch {arch} is not supported (only WH and BH).")
    if prefill_rings > 0 and arch != ttnn.device.Arch.BLACKHOLE:
        pytest.skip("Replay rings are supported only on Blackhole; Wormhole streaming is covered separately")
    torch.manual_seed(2003)

    hidden_size, N, k = _PREFILL_HIDDEN, _PREFILL_N, _PREFILL_K
    experts_per_device, experts = _PREFILL_EXPERTS_PER_DEVICE, _PREFILL_EXPERTS
    ring_n = effective_matmul_ring_size(mesh_device)
    output_width_shard_dim = auto_output_width_shard_dim(hidden_size, matmul_ring_size=ring_n)
    output_height_shard_dim = 1  # the 128-token form on an 8-bank ring; the local output path ignores it
    num_layers = 1
    layer_id = 0

    indices, scores = _prefill_routing(routing_case, tokens)
    golden_lists, golden_counts = _prefill_local_lists(indices)
    golden_chunks = int(((golden_counts + _PREFILL_TOKENS_PER_CHUNK - 1) // _PREFILL_TOKENS_PER_CHUNK).sum())
    logger.info(
        f"Local output prefill: tokens {tokens}, routing {routing_case}: {int(golden_counts.sum())} local pairs, "
        f"{int((golden_counts > 0).sum())} active experts, max count {int(golden_counts.max())}, {golden_chunks} chunks"
    )

    # Every device of a 1x4 line sees this mapping row: experts 0..127 here, the rest elsewhere.
    expert_mapping = (torch.arange(experts) // experts_per_device).to(torch.uint16).reshape(1, experts)
    tt_expert_mapping = ttnn.from_torch(
        expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    hidden = torch.randn(tokens, hidden_size, dtype=torch.float32).to(torch.bfloat16)

    w0_w1_shard_map, w2_shard_map, dram_core_range_set = get_weight_core_shard_maps(mesh_device, hidden_size, N)
    torch_w0 = create_torch_w0(num_layers, experts_per_device, hidden_size, N)
    torch_w1 = create_torch_w1(num_layers, experts_per_device, hidden_size, N)
    torch_w2 = create_torch_w2(num_layers, experts_per_device, N, hidden_size)
    w0_w1_mem_config, w2_mem_config, _, _ = get_weight_mem_configs(
        num_layers, experts_per_device, hidden_size, N, w0_w1_shard_map, w2_shard_map, dram_core_range_set
    )
    tt_w0_w1, tt_w2 = _build_quantized_weight_tensors_cpu_prepare(
        mesh_device,
        torch_w0,
        torch_w1,
        torch_w2,
        None,
        None,
        None,
        num_layers,
        experts_per_device,
        hidden_size,
        N,
        False,
        w0_w1_shard_map,
        w2_shard_map,
        w0_w1_mem_config,
        w2_mem_config,
    )

    drain = ttnn.experimental.get_moe_tilize_drain_core(
        mesh_device, output_height_shard_dim, output_width_shard_dim, hidden_size
    )
    drain_core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(drain.x, drain.y), ttnn.CoreCoord(drain.x, drain.y))})

    def upload_call_inputs(rows, row_indices, row_scores, output_fill=0.0):
        num_rows = rows.shape[0]
        tt_rows = ttnn.from_torch(
            rows.reshape(1, num_rows, hidden_size),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_indices = ttnn.from_torch(
            row_indices.to(torch.int32).reshape(1, num_rows, k),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=create_sharded_memory_config(drain_core, [num_rows, k], ttnn.uint16),
        )
        tt_scores = ttnn.from_torch(
            row_scores.reshape(1, num_rows, k),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=create_sharded_memory_config(drain_core, [num_rows, k], ttnn.bfloat16),
        )
        tt_output = ttnn.from_torch(
            torch.full((k, num_rows, hidden_size), output_fill, dtype=torch.bfloat16),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return tt_rows, tt_indices, tt_scores, tt_output

    def run(tt_rows, tt_indices, tt_scores, tt_output, cluster_axis, rings=prefill_rings, pipeline=None):
        # zero_fill_non_owned_rows and prefill_rings apply to the local output path only; the fused reference
        # keeps the defaults.
        fill_kwargs = (
            {}
            if cluster_axis is None
            else {
                "zero_fill_non_owned_rows": zero_fill,
                "prefill_rings": rings,
                "enable_a2a_pipeline": rings == 0 if pipeline is None else pipeline,
            }
        )
        return ttnn.experimental.moe_compute(
            tt_rows,
            tt_indices,
            tt_scores,
            tt_expert_mapping,
            tt_w0_w1,
            tt_w2,
            layer_id=layer_id,
            output_height_shard_dim=output_height_shard_dim,
            intermediate_size=N,
            has_bias=False,
            cluster_axis=cluster_axis,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            optional_output_tensor=tt_output,
            activation_type=MoEActivationFunction.SILU,
            compute_only=False,
            **fill_kwargs,
        )

    def fused_reference_bits(indices, scores):
        # The fused local combine (cluster_axis=None on a 1x1 mesh, today's production slab form) over the same
        # rows, 128 tokens per call, pages re-indexed k * T + t; its buffer starts zero, so unowned rows read zero.
        reference_bits = []
        for start in range(0, tokens, _PREFILL_REFERENCE_TOKENS):
            stop = min(start + _PREFILL_REFERENCE_TOKENS, tokens)
            call_inputs = upload_call_inputs(hidden[start:stop], indices[start:stop], scores[start:stop])
            outputs = run(*call_inputs, cluster_axis=None)
            reference_bits.append(_bf16_bits(outputs[5]))
            for tensor in (outputs[0], outputs[1], outputs[2], outputs[4], *call_inputs):
                ttnn.deallocate(tensor)
        reference_bits = torch.cat(reference_bits, dim=1)
        assert reference_bits.shape == (k, tokens, hidden_size)
        return reference_bits

    reference_bits = fused_reference_bits(indices, scores)
    if sentinel_output:
        # The caller's buffer holds a sentinel: with the fill the unowned rows come back zero, without it they keep
        # the sentinel; the owned rows are the reference either way.
        sentinel_bits = torch.tensor(1.0, dtype=torch.bfloat16).view(torch.int16)
        unowned = (indices >= experts_per_device).t().unsqueeze(-1)  # [K, T, 1]
        assert bool(unowned.any()), "the sentinel check needs unowned rows"
        expected_bits = reference_bits if zero_fill else torch.where(unowned, sentinel_bits, reference_bits)
    else:
        expected_bits = reference_bits

    # One local output call over all tokens (cluster_axis=0, extent 1 on a 1x1 mesh).
    call_inputs = upload_call_inputs(hidden, indices, scores, output_fill=1.0 if sentinel_output else 0.0)
    outputs = run(*call_inputs, cluster_axis=0)
    assert len(outputs) == 6
    assert outputs[5].buffer_address() == call_inputs[3].buffer_address(), "slot 5 must be the caller's output"
    output_bits = _bf16_bits(outputs[5])

    counts = ttnn.to_torch(outputs[0]).to(torch.int64)[0, :experts_per_device]
    assert torch.equal(counts, golden_counts), f"per-expert counts differ from the routing histogram: {counts}"

    packed = ttnn.to_torch(outputs[2]).flatten().to(torch.int64) & 0xFFFFFFFF
    assert packed.numel() == token_list_page_words(tokens, k, experts_per_device), "packed page size"
    starts = packed[: experts_per_device + 1].tolist()
    assert starts == token_list_segment_starts(golden_counts.tolist()), "segment starts differ from the alignment rule"
    decoded = decode_packed_token_lists(packed, experts_per_device, golden_counts.tolist())
    for e, ((token_ids, k_slots), (golden_tokens, golden_k)) in enumerate(zip(decoded, golden_lists)):
        assert token_ids.tolist() == golden_tokens, f"expert {e}: token ids differ"
        assert k_slots.tolist() == golden_k, f"expert {e}: k slots differ"
    logger.info(
        f"Packed token lists: {starts[-1]} entries in {experts_per_device} segments, "
        f"header {token_list_header_words(experts_per_device)} words"
    )

    def assert_bitwise(output_bits, expected_bits, indices, what):
        mismatches = output_bits != expected_bits
        if bool(mismatches.any()):
            k_slot, t, _ = torch.nonzero(mismatches)[0].tolist()
            expert = int(indices[t, k_slot])
            rows_differing = int(mismatches.any(dim=-1).sum())
            raise AssertionError(
                f"{what} differs from the expected pages: {int(mismatches.sum())} of {mismatches.numel()} bf16 values "
                f"in {rows_differing} rows, first at (k={k_slot}, t={t}) routed to expert {expert} "
                f"({'local' if expert < experts_per_device else 'remote'})"
            )

    assert_bitwise(output_bits, expected_bits, indices, f"local output at {tokens} tokens")
    logger.info(
        f"Local output at {tokens} tokens (prefill_rings {prefill_rings}): {k * tokens} pages bitwise equal to the "
        f"128-token fused local combine ({golden_chunks} chunks in one call"
        f"{'; sentinel rows checked' if sentinel_output else ''})"
    )
    for tensor in (outputs[0], outputs[1], outputs[2], outputs[4], *call_inputs):
        ttnn.deallocate(tensor)

    if tokens == 128 and prefill_rings == 0:
        # The explicit pipeline flag must be part of the program-cache key.
        # The serial path keeps the legacy feed size and produces identical pages.
        previous_entries = mesh_device.num_program_cache_entries()
        for iteration in range(2):
            call_inputs = upload_call_inputs(hidden, indices, scores)
            outputs = run(*call_inputs, cluster_axis=0, pipeline=False)
            assert_bitwise(_bf16_bits(outputs[5]), output_bits, indices, "serial vs pipelined local output")
            entries = mesh_device.num_program_cache_entries()
            if iteration == 0:
                assert entries > previous_entries, "pipeline policy reused the wrong cached program"
                serial_entries = entries
            else:
                assert entries == serial_entries, "fresh addresses missed the serial program cache"
            for tensor in (outputs[0], outputs[1], outputs[2], outputs[4], *call_inputs):
                ttnn.deallocate(tensor)

    if prefill_rings > 0:
        # The same inputs through the un-replayed op (prefill_rings=0): "the same bits" asserted directly, not only
        # through the shared 128-token reference.
        call_inputs = upload_call_inputs(hidden, indices, scores, output_fill=1.0 if sentinel_output else 0.0)
        outputs = run(*call_inputs, cluster_axis=0, rings=0)
        assert_bitwise(_bf16_bits(outputs[5]), output_bits, indices, f"prefill_rings=0 vs {prefill_rings}")
        logger.info(f"Local output at {tokens} tokens: prefill_rings {prefill_rings} and 0 give the same pages")
        for tensor in (outputs[0], outputs[1], outputs[2], outputs[4], *call_inputs):
            ttnn.deallocate(tensor)

    if second_routing_case is not None:
        # The same program (a cache hit: same T, K, shapes) on another routing: the drain reuses its page, its
        # staging slots and the pair lists without clearing them, every read bounded by this call's counts.
        indices_b, scores_b = _prefill_routing(second_routing_case, tokens, seed=20260926)
        reference_b = fused_reference_bits(indices_b, scores_b)
        call_inputs = upload_call_inputs(hidden, indices_b, scores_b)
        outputs = run(*call_inputs, cluster_axis=0)
        assert_bitwise(_bf16_bits(outputs[5]), reference_b, indices_b, f"cached program on {second_routing_case}")
        logger.info(f"Local output at {tokens} tokens, cached program, routing {second_routing_case}: bitwise")
        for tensor in (outputs[0], outputs[1], outputs[2], outputs[4], *call_inputs):
            ttnn.deallocate(tensor)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 500000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "tokens, routing_case, zero_fill, sentinel_output, second_routing_case, prefill_rings",
    [
        (128, "zipf", True, False, None, 0),
        (512, "zipf", True, False, None, 0),
        (2048, "zipf", True, False, "random_local", 0),
        (2048, "zipf", True, True, None, 0),
        (2048, "zipf", False, True, None, 0),
        (99, "zipf", True, False, None, 0),
        (2050, "zipf", True, False, None, 0),
        (2048, "one_expert", True, False, None, 0),
        (2048, "uniform_local", True, False, None, 0),
        (2048, "random_local", True, False, None, 0),
        (2048, "no_local", True, False, None, 0),
        (2048, "no_local", False, True, None, 0),
        # the replay ring: the weight slice read once per expert and replayed per chunk
        (128, "zipf", True, False, None, 1),
        (2048, "zipf", False, True, "random_local", 1),
        (2048, "one_expert", True, False, None, 1),
        (2048, "random_local", True, False, None, 1),
        (2048, "no_local", True, False, None, 1),
        # two rings: the owner table splits the chunks over a second ring of cores, each ring reading the slices of
        # the experts it owns chunks of (fill off: one ring's fill would race the other's rows)
        (128, "zipf", False, False, None, 2),
        (2048, "zipf", False, True, "random_local", 2),
        (2048, "one_expert", False, False, None, 2),
        (2048, "random_local", False, False, None, 2),
        (2048, "no_local", False, False, None, 2),
        # three rings: the third ring's cores two cells inside the ring box; chunks split three ways
        (2048, "zipf", False, True, "random_local", 3),
        (2048, "one_expert", False, False, None, 3),
        (2048, "random_local", False, False, None, 3),
    ],
    ids=[
        "128-zipf",
        "512-zipf",
        "2048-zipf-then-random_local",
        "2048-zipf-fill-sentinel",
        "2048-zipf-no_fill-sentinel",
        "99-zipf",
        "2050-zipf",
        "2048-one_expert",
        "2048-uniform_local",
        "2048-random_local",
        "2048-no_local",
        "2048-no_local-no_fill-sentinel",
        "128-zipf-replay",
        "2048-zipf-no_fill-then-random_local-replay",
        "2048-one_expert-replay",
        "2048-random_local-replay",
        "2048-no_local-replay",
        "128-zipf-rings2",
        "2048-zipf-no_fill-then-random_local-rings2",
        "2048-one_expert-rings2",
        "2048-random_local-rings2",
        "2048-no_local-rings2",
        "2048-zipf-no_fill-then-random_local-rings3",
        "2048-one_expert-rings3",
        "2048-random_local-rings3",
    ],
)
def test_moe_compute_single_card_local_output_prefill(
    mesh_device, mesh_shape, tokens, routing_case, zero_fill, sentinel_output, second_routing_case, prefill_rings
):
    """One local output call over a prefill slab's tokens (Qwen3.8-Flash-Next shape: hidden 2560, N 640, 128 of 512
    experts local, K 10) writes the same pages as the fused local combine over the same rows in 128-token calls,
    bitwise; the per-expert counts are the routing histogram and the packed token lists decode to the golden lists.
    Cases: the natural-text shape at 128 / 512 / 2048 tokens and at counts the four tilize cores split unevenly (99,
    2050); one expert taking every token (a 64-chunk segment); every slot local, uniform (the lists at capacity) and
    random; and no local pair (an empty call). With a sentinel-filled caller buffer the rows nobody owns come back
    zero with the fill and keep the sentinel without it (zero_fill_non_owned_rows=False), the owned rows being the
    reference either way; one row re-runs the cached program on a second routing. The replay rows (prefill_rings=1)
    keep each expert's weight slice resident for all of its chunks (one expert with 64 chunks, the capacity routing,
    an empty call) and must give the same bits. The two- and three-ring rows (prefill_rings=2, 3) split the chunks
    over further rings of cores, each reading the slices of the experts it owns chunks of (Blackhole only), and must
    give the same bits as one ring, also compared directly against prefill_rings=0 on the same inputs."""
    _run_moe_compute_local_output_prefill_test(
        mesh_device,
        tokens,
        routing_case,
        zero_fill=zero_fill,
        sentinel_output=sentinel_output,
        second_routing_case=second_routing_case,
        prefill_rings=prefill_rings,
    )
