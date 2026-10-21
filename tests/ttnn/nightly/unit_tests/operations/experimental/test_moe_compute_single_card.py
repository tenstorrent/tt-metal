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
from contextlib import contextmanager
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
    prepare_output_tensor_from_combine_writer,
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


def _xfail_on_short_wormhole_grid(mesh_device, num_combine_cores):
    """moe_compute cannot place 12 matmul + 16 combine cores disjointly on the harvested wh_n300_civ2
    7x8 worker grid (COL dispatch); ROW dispatch (8x7) and 12 combine cores fit."""
    worker_grid = mesh_device.compute_with_storage_grid_size()
    if (
        mesh_device.arch() == ttnn.device.Arch.WORMHOLE_B0
        and worker_grid.x <= 7
        and worker_grid.y <= 8
        and num_combine_cores >= 16
    ):
        pytest.xfail(
            f"moe_compute cannot place 12 matmul + {num_combine_cores} combine cores on the "
            f"{worker_grid.x}x{worker_grid.y} Wormhole worker grid; https://github.com/tenstorrent/tt-metal/issues/52246"
        )


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
    activation_limit=None,
    weight_initializer=None,
    math_fidelity=ttnn.MathFidelity.LoFi,
    fp32_dest_acc_en=False,
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
    to pack the weights so host tensor layout matches the op.
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

    # Combine cores = output height shards x output width shards.
    _xfail_on_short_wormhole_grid(mesh_device, output_height_shard_dim * output_width_shard_dim)

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

    if weight_initializer is None:
        torch_w0 = create_torch_w0(num_layers, experts_per_device, hidden_size, N)
        torch_w1 = create_torch_w1(num_layers, experts_per_device, hidden_size, N)
        torch_w2 = create_torch_w2(num_layers, experts_per_device, N, hidden_size)
    else:
        assert not has_bias and not compute_only, (
            f"weight_initializer supports the fused path without bias only; got has_bias={has_bias}, "
            f"compute_only={compute_only}"
        )
        torch_w0, torch_w1, torch_w2 = weight_initializer(num_layers, experts_per_device, hidden_size, N)

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
        activation_limit=activation_limit,
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
            activation_limit=activation_limit,
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            **ccl_kwargs,
        )

    def per_device_bf16_bits(tensor):
        # The raw bf16 bit patterns of every device's rows: the bitwise comparison of two output
        # placements must not go through a float compare (NaN != NaN, -0.0 == 0.0).
        return [
            ttnn.to_torch(device_tensor, mesh_composer=None).contiguous().view(torch.int16)
            for device_tensor in ttnn.get_device_tensors(tensor)
        ]

    def check_structured_combine(tt_combine_output):
        # Exact BFP4 weights leave no quantization error: check the gain as well as the correlation.
        actual = ttnn.to_torch(ttnn.get_device_tensors(tt_combine_output)[0]).float()
        expected = combine_goldens[0][layer_id].float()
        assert tuple(actual.shape) == tuple(
            expected.shape
        ), f"output shape {tuple(actual.shape)}, golden {tuple(expected.shape)}"
        assert bool(torch.isfinite(actual).all()), f"{int((~torch.isfinite(actual)).sum())} non-finite output values"
        relative_l2 = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected).clamp_min(1e-8)
        logger.debug(f"Structured weights combine relative L2: {float(relative_l2):.8f}")
        assert relative_l2 < 0.03, f"relative L2 {float(relative_l2):.6f} >= 0.03"

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
        if weight_initializer is not None:
            check_structured_combine(combine_output_tensor)

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
        if weight_initializer is not None:
            check_structured_combine(cache_hit_combine_output_tensor)

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
    The width-shard dim must match the op's derivation, so it is auto-derived.
    """
    hidden_size = 7168
    N = 2048
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=16,
        tokens_per_device=32,
        selected_experts_k=8,
        N=N,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size),
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
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=16,
        tokens_per_device=32,
        selected_experts_k=4,
        N=hidden_size,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size),
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
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_single_card_flash_next(mesh_device, mesh_shape, compute_only):
    """Single-card MoE compute on a 1x1 mesh, Qwen3.8-Flash-Next-shaped workload (hidden=2560, N=640, SILU).

    512 routed experts over four devices = 128 experts per device. N=640 is 20 intermediate tiles, so on an
    8-bank Blackhole ring the cores own 3 or 2 gate/up columns (odd and even counts on one ring), and on a
    12-core Wormhole ring 2 or 1.
    """
    hidden_size = 2560
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=128,
        tokens_per_device=32,
        selected_experts_k=8,
        N=640,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size),
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
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=experts,
        tokens_per_device=32,
        selected_experts_k=k,
        N=intermediate,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size),
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
    ring_n = effective_matmul_ring_size(mesh_device)
    N = max(cfg["N"], 32 * ring_n)
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=cfg["experts_per_device"],
        tokens_per_device=tokens_per_device,
        selected_experts_k=cfg["selected_experts_k"],
        N=N,
        hidden_size=cfg["hidden_size"],
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(cfg["hidden_size"]),
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
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=16,
        tokens_per_device=1,
        selected_experts_k=8,
        N=512,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size),
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
@pytest.mark.parametrize("hidden_size", [2048, 224], ids=["h2048", "h224"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_single_card_explicit_axis0(mesh_device, mesh_shape, hidden_size):
    """cluster_axis=0 on a 1x1 mesh names an axis of extent 1: the op takes the SingleCluster path,
    dm1 writes the final output, and it must match the golden like the cluster_axis=None call. Hidden
    224 is 7 tiles, fewer than the ring cores of an 8-bank Blackhole ring or the 12-core Wormhole
    ring, so there some ring cores own no output columns."""
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=16,
        tokens_per_device=1,
        selected_experts_k=8,
        N=512,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size),
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
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size),
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
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=16,
        tokens_per_device=8,
        selected_experts_k=8,
        N=512,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size),
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
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=16,
        tokens_per_device=8,
        selected_experts_k=8,
        N=512,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size),
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
def test_moe_compute_local_axis_rejects_sharded_routing_metadata(mesh_device, mesh_shape, expect_error, sharded_input):
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
            mesh_device, inputs=(tt_sparse, tt_indices, tt_scores, tt_mapping, tt_w0_w1, tt_w2), cluster_axis=0
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


def _clamp_boundary_weights(layers, experts, hidden, intermediate):
    """Power-of-two sparse projections, exact in BFP4: every gate / up output is one scaled input and every
    down output one scaled activation. Inputs cover [-0.5, 0.5), so gate x32 and up x64 reach both clamp
    limits, and the negative gate branch stays unbounded."""
    assert intermediate <= hidden, f"structured weights need intermediate <= hidden; got {intermediate} > {hidden}"
    gate = torch.zeros((layers, experts, hidden, intermediate), dtype=torch.bfloat16)
    up = torch.zeros_like(gate)
    down = torch.zeros((layers, experts, intermediate, hidden), dtype=torch.bfloat16)
    columns = torch.arange(intermediate)
    gate[:, :, columns, columns] = 32.0
    up[:, :, (columns + 17) % hidden, columns] = 64.0
    down[:, :, columns, columns] = 1 / 32
    return gate, up, down


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 500000}], indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@pytest.mark.parametrize("tokens", [1, 4])
@pytest.mark.parametrize("activation_limit", [0.125, 10.0])
def test_moe_compute_single_card_clamped_silu(mesh_device, mesh_shape, tokens, activation_limit):
    """CLAMPED_SILU in SingleDevice with exact weights; the small limit clips both gate and up."""
    hidden_size = 4096
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=2,
        tokens_per_device=tokens,
        selected_experts_k=2,
        N=2048,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size),
        dtype=ttnn.bfloat16,
        activation_type=MoEActivationFunction.CLAMPED_SILU,
        activation_limit=activation_limit,
        compute_only=False,
        weight_initializer=_clamp_boundary_weights,
    )


@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "activation_limit, error",
    [
        (None, "activation_limit is required for CLAMPED_SILU"),
        (0.0, "activation_limit must be finite and positive"),
        (-0.0, "activation_limit must be finite and positive"),
        (-1.0, "activation_limit must be finite and positive"),
        (float("nan"), "activation_limit must be finite and positive"),
        (float("inf"), "activation_limit must be finite and positive"),
        (-float("inf"), "activation_limit must be finite and positive"),
    ],
    ids=["missing", "zero", "negative-zero", "negative", "nan", "positive-inf", "negative-inf"],
)
def test_moe_compute_clamped_silu_rejects_invalid_limit(mesh_device, mesh_shape, expect_error, activation_limit, error):
    before = mesh_device.num_program_cache_entries()
    with expect_error(RuntimeError, error):
        _call_moe_compute_for_rejection(
            mesh_device, activation_type=MoEActivationFunction.CLAMPED_SILU, activation_limit=activation_limit
        )
    after = mesh_device.num_program_cache_entries()
    assert after == before, f"a rejected call changed the program cache: {before} -> {after} entries"


@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "activation_type", [None, MoEActivationFunction.SILU, MoEActivationFunction.SWIGLU, MoEActivationFunction.GELU]
)
def test_moe_compute_other_activations_reject_limit(mesh_device, mesh_shape, expect_error, activation_type):
    before = mesh_device.num_program_cache_entries()
    with expect_error(RuntimeError, "must be omitted for other activations"):
        _call_moe_compute_for_rejection(mesh_device, activation_type=activation_type, activation_limit=10.0)
    after = mesh_device.num_program_cache_entries()
    assert after == before, f"a rejected call changed the program cache: {before} -> {after} entries"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 500000}], indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_clamped_silu_limit_is_program_cache_key(mesh_device, mesh_shape):
    """A second limit compiles the call's programs anew; returning to the first hits its cached programs."""
    mesh_device.disable_and_clear_program_cache()
    mesh_device.enable_program_cache()
    hidden_size = 512
    ring_n = effective_matmul_ring_size(mesh_device)
    cache_entries = []
    for activation_limit in (0.125, 10.0, 0.125):
        _run_moe_compute_single_card_test(
            mesh_device=mesh_device,
            mesh_shape=mesh_shape,
            experts_per_device=2,
            tokens_per_device=4,
            selected_experts_k=2,
            N=max(256, 32 * ring_n),
            hidden_size=hidden_size,
            output_height_shard_dim=4,
            output_width_shard_dim=auto_output_width_shard_dim(hidden_size),
            dtype=ttnn.bfloat16,
            activation_type=MoEActivationFunction.CLAMPED_SILU,
            activation_limit=activation_limit,
            compute_only=False,
            weight_initializer=_clamp_boundary_weights,
        )
        ttnn.synchronize_device(mesh_device)
        cache_entries.append(mesh_device.num_program_cache_entries())
    # one MoE call is one program on the ring path and two on the expert rows path (expert rows + combine side)
    assert cache_entries[1] - cache_entries[0] in (
        1,
        2,
    ), f"changing the clamp must compile the MoE call anew: cache entries {cache_entries}"
    assert (
        cache_entries[2] == cache_entries[1]
    ), f"returning to the original clamp must hit its cached program: cache entries {cache_entries}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 500000}], indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@pytest.mark.parametrize("tokens", [1, 4])
@pytest.mark.parametrize(
    "activation_type",
    [
        MoEActivationFunction.SILU,
        MoEActivationFunction.SWIGLU,
        MoEActivationFunction.GELU,
        MoEActivationFunction.CLAMPED_SILU,
    ],
)
def test_moe_compute_fp32_dest_activations(mesh_device, mesh_shape, tokens, activation_type):
    """HiFi4 with FP32 DEST: every activation's packer SFPU path on four FP32 DEST tiles, exact weights."""
    hidden_size = 768
    ring_n = effective_matmul_ring_size(mesh_device)
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=2,
        tokens_per_device=tokens,
        selected_experts_k=2,
        N=max(384, 32 * ring_n),
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size),
        dtype=ttnn.bfloat16,
        activation_type=activation_type,
        activation_limit=10.0 if activation_type == MoEActivationFunction.CLAMPED_SILU else None,
        compute_only=False,
        weight_initializer=_clamp_boundary_weights,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 500000}], indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_precision_is_program_cache_key(mesh_device, mesh_shape):
    """Fidelity and DEST format each select their own cached program."""
    mesh_device.disable_and_clear_program_cache()
    mesh_device.enable_program_cache()
    hidden_size = 768
    ring_n = effective_matmul_ring_size(mesh_device)
    modes = [
        (ttnn.MathFidelity.LoFi, False),
        (ttnn.MathFidelity.HiFi4, False),
        (ttnn.MathFidelity.HiFi4, True),
        (ttnn.MathFidelity.LoFi, True),
        (ttnn.MathFidelity.LoFi, False),
    ]
    cache_entries = []
    for fidelity, fp32_dest in modes:
        _run_moe_compute_single_card_test(
            mesh_device=mesh_device,
            mesh_shape=mesh_shape,
            experts_per_device=2,
            tokens_per_device=4,
            selected_experts_k=2,
            N=max(384, 32 * ring_n),
            hidden_size=hidden_size,
            output_height_shard_dim=4,
            output_width_shard_dim=auto_output_width_shard_dim(hidden_size),
            dtype=ttnn.bfloat16,
            activation_type=MoEActivationFunction.CLAMPED_SILU,
            activation_limit=10.0,
            compute_only=False,
            weight_initializer=_clamp_boundary_weights,
            math_fidelity=fidelity,
            fp32_dest_acc_en=fp32_dest,
        )
        ttnn.synchronize_device(mesh_device)
        cache_entries.append(mesh_device.num_program_cache_entries())
    # one MoE call is one program on the ring path and two on the expert rows path (expert rows + combine side)
    added = [b - a for a, b in zip(cache_entries[0:3], cache_entries[1:4])]
    assert added[0] in (1, 2) and added == [added[0]] * 3, cache_entries
    assert cache_entries[4] == cache_entries[3], "returning to default precision must hit its cached program"


# ---- expert rows path ----
# On Blackhole, moe_compute runs the expert-row program plus a second program: metadata and placement (ComputeOnly), or
# metadata, the double-buffer feed and selective_reduce_combine's own kernels (SingleDevice);
# TT_METAL_MOE_COMPUTE_KERNEL=ring|expert_rows forces a path and TT_METAL_MOE_COMPUTE_ROW_TILES=M the expert rows path's
# row tiles per job (M > 1: weight-stationary jobs). The tests below check, for both paths on the same inputs:
#   - outputs 0-2 (counts, activation rows, e_t) against the 6U goldens, and equal between the two paths;
#   - output 4: every row of the last two local experts (what moe_compute leaves in its double buffer) against an FP32
#     golden on the BFP4-rounded weights;
#   - SingleDevice output 5 (the combine output): bitwise equal between the paths, and against the FP32 golden;
#   - a repeated call and a traced call bitwise equal to the eager call, in every DEST mode and with the defaults;
#   - rows (output 4, and every row through the SingleDevice combine) of jobs of M row tiles bitwise equal to one row
#     tile per job, in every DEST mode.

# an e_t page: one u32 token per 16-byte entry (every _E_T_ENTRY_WORDS-th word), ended by _E_T_END
_E_T_ENTRY_WORDS = 4
_E_T_END = 0xFFFFFFFF
_TILE_WIDTH = 32  # values per tile row

_EXPERT_ROWS_DEVICE_PARAMS = [
    {"l1_small_size": 16384, "dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 8 << 20}
]
_KERNEL_ENV = "TT_METAL_MOE_COMPUTE_KERNEL"
_ROW_TILES_ENV = "TT_METAL_MOE_COMPUTE_ROW_TILES"


@contextmanager
def _env(key, value):
    old = os.environ.get(key)
    os.environ[key] = str(value)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def _kernel(name):
    return _env(_KERNEL_ENV, name)


def _skip_unless_supported_arch(mesh_device):
    """The expert rows path runs on Blackhole and Wormhole (on Wormhole only under
    TT_METAL_MOE_COMPUTE_KERNEL=expert_rows, which every test here sets for its expert rows calls)."""
    if mesh_device.arch() not in (ttnn.device.Arch.BLACKHOLE, ttnn.device.Arch.WORMHOLE_B0):
        pytest.skip("the expert rows path runs on Blackhole and Wormhole")


def _skip_row_tiles_on_wormhole(mesh_device):
    if mesh_device.arch() == ttnn.device.Arch.WORMHOLE_B0:
        pytest.skip("Wormhole runs one row tile per job: the M > 1 kernels exceed its kernel config buffer")


def _bfp4_round(w):
    return ttnn.to_torch(ttnn.from_torch(w.float(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()


def _mismatch(a, b):
    """How many values of two equal-shaped tensors differ, for assert messages."""
    return f"{int((a != b).sum())} of {a.numel()} values differ"


def _routing(kind, tokens, local_experts, top_k, seed=2003):
    """Token-major top-k lists: random (uniform, distinct ids), skewed (expert 0 in every token, the upper half never
    chosen), empty (only the even experts), dup (one id twice per token, at random k), all_one (every k of every token
    names expert 0: top_k * tokens entries for one expert, more than its e_t page holds), hot_last (the last two experts
    in every token: `tokens` rows each), cold_last (the last two experts in the first 20 tokens only, the others about
    tokens * (top_k - 2) / (local_experts - 2) rows each)."""
    rng = random.Random(seed)
    out = []
    for _ in range(tokens):
        if kind == "random":
            ids = rng.sample(range(local_experts), top_k)
        elif kind == "skewed":
            ids = [0] + rng.sample(range(1, max(2, local_experts // 2)), top_k - 1)
        elif kind == "empty":
            ids = rng.sample(range(0, local_experts, 2), top_k)
        elif kind == "dup":
            ids = rng.sample(range(local_experts), top_k - 1)
            ids.insert(rng.randrange(top_k), ids[rng.randrange(top_k - 1)])
        elif kind == "all_one":
            ids = [0] * top_k
        elif kind == "hot_last":
            ids = [local_experts - 1, local_experts - 2] + rng.sample(range(local_experts - 2), top_k - 2)
            rng.shuffle(ids)
        elif kind == "cold_last":
            ids = (
                ([local_experts - 1, local_experts - 2] + rng.sample(range(local_experts - 2), top_k - 2))
                if len(out) < 20
                else rng.sample(range(local_experts - 2), top_k)
            )
            rng.shuffle(ids)
        else:
            raise ValueError(kind)
        out.append(ids)
    return out


class _Case:
    """One single-card ComputeOnly case: inputs on device, goldens, prepared weights."""

    def __init__(
        self,
        mesh_device,
        local_experts,
        tokens,
        top_k,
        intermediate_size,
        hidden_size,
        act=MoEActivationFunction.SILU,
        limit=None,
        has_bias=False,
        kind="random",
        ntp=4,
        layers=1,
        goldens=True,
    ):
        torch.manual_seed(2003)
        random.seed(2003)
        self.mesh_device, self.local_experts, self.tokens, self.top_k, self.intermediate_size, self.hidden_size = (
            mesh_device,
            local_experts,
            tokens,
            top_k,
            intermediate_size,
            hidden_size,
        )
        self.act, self.limit, self.has_bias, self.ntp, self.layers = act, limit, has_bias, ntp, layers
        mesh_shape = (1, 1)
        self.dp = auto_output_width_shard_dim(hidden_size)
        _xfail_on_short_wormhole_grid(mesh_device, ntp * self.dp)
        drain = ttnn.experimental.get_moe_tilize_drain_core(mesh_device, ntp, self.dp, hidden_size)
        drain_set = ttnn.CoreRangeSet({ttnn.CoreRange(drain, drain)})
        mapping = gen_expert_mapping(1, 1, None, local_experts, local_experts, local_experts)
        if kind == "random":
            sparse, indices, scores, _ = gen_sparse_buffer_and_indices(
                tokens, hidden_size, local_experts, top_k, mesh_shape, None
            )
        else:
            sparse = (torch.rand(1, tokens, hidden_size) - 0.5).bfloat16()
            indices = (
                torch.tensor(_routing(kind, tokens, local_experts, top_k), dtype=torch.int32)
                .reshape(1, tokens, top_k)
                .to(torch.uint16)
            )
            raw = torch.rand(1, tokens, top_k) + 1e-3
            scores = (raw / raw.sum(-1, keepdim=True)).bfloat16()
        self.token_experts = indices.reshape(tokens, top_k).tolist()
        self.counts = self.golden = None
        if goldens:  # the 6U goldens hold at most `tokens` rows per expert
            tilize, self.counts = compute_selective_tilize_golden(sparse, indices, scores, mapping, mesh_shape, None)
            self.golden_act, _ = compute_expert_activation_golden(indices, scores, mapping, mesh_shape, None)
            self.golden_e_t, _ = compute_e_t_golden(indices, mapping, mesh_shape, None)

        w01_map, w2_map, dram_set = get_weight_core_shard_maps(mesh_device, hidden_size, intermediate_size)
        w0, w1, w2 = (
            create_torch_w0(layers, local_experts, hidden_size, intermediate_size),
            create_torch_w1(layers, local_experts, hidden_size, intermediate_size),
            create_torch_w2(layers, local_experts, intermediate_size, hidden_size),
        )
        b0 = b1 = b2 = None
        if has_bias:
            b0 = (torch.randn(layers, local_experts, intermediate_size) * 0.12).to(torch.bfloat16)
            b1 = (torch.randn(layers, local_experts, intermediate_size) * 0.12).to(torch.bfloat16)
            b2 = (torch.randn(layers, local_experts, hidden_size) * 0.12).to(torch.bfloat16)
        # FP32 golden on the BFP4-rounded weights (bias tiles are BFP4 too)
        rw = lambda w: torch.stack(
            [torch.stack([_bfp4_round(w[l, e]) for e in range(local_experts)]) for l in range(layers)]
        )  # noqa
        r0, r1, r2 = rw(w0), rw(w1), rw(w2)
        rb = lambda b: None if b is None else torch.stack([_bfp4_round(b[l].float()) for l in range(layers)])  # noqa
        self.golden = (
            None
            if not goldens
            else compute_matmul_golden(
                tilize.unsqueeze(0).repeat([layers] + [1] * tilize.dim()),
                r0,
                r1,
                r2,
                layers,
                local_experts,
                1,
                tokens,
                hidden_size,
                torch_b0=rb(b0),
                torch_b1=rb(b1),
                torch_b2=rb(b2),
                activation_type=act,
                activation_limit=limit,
            ).float()
        )
        m01, m2, _, _ = get_weight_mem_configs(
            layers, local_experts, hidden_size, intermediate_size, w01_map, w2_map, dram_set, has_bias=has_bias
        )
        if has_bias:
            p01 = prepare_w0_w1_tensor_with_bias(
                w0, w1, b0, b1, layers, local_experts, hidden_size, intermediate_size, w01_map
            )
            p2 = prepare_w2_tensor_with_bias(
                w2, b2, layers, local_experts, intermediate_size, hidden_size, w2_map, w01_map
            )
        else:
            p01 = prepare_w0_w1_tensor_for_moe_compute(
                w0, w1, layers, local_experts, hidden_size, intermediate_size, w01_map
            )
            p2 = prepare_w2_tensor_for_moe_compute(
                w2, layers, local_experts, intermediate_size, hidden_size, w2_map, w01_map
            )
        # host BFP4 rounding, the same the golden's weights get
        upload = lambda t, mc: ttnn.from_torch(  # noqa: E731
            t, dtype=ttnn.bfloat4_b, device=mesh_device, layout=ttnn.TILE_LAYOUT, memory_config=mc
        )
        self.w01, self.w2 = upload(p01, m01), upload(p2, m2)
        self.x = ttnn.from_torch(
            sparse,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self.idx = ttnn.from_torch(
            indices.reshape(1, tokens, top_k),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=create_sharded_memory_config(drain_set, [tokens, top_k], ttnn.uint16),
        )
        self.sc = ttnn.from_torch(
            scores.reshape(1, tokens, top_k),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=create_sharded_memory_config(drain_set, [tokens, top_k], ttnn.bfloat16),
        )
        self.map = ttnn.from_torch(
            mapping,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        grid = mesh_device.compute_with_storage_grid_size()
        self.all_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        self.out_cores = ttnn.experimental.get_moe_combine_cores(mesh_device, ntp, self.dp, hidden_size)
        self.bbox = ttnn.experimental.get_moe_worker_mcast_bounding_box(mesh_device, ntp, self.dp, hidden_size)

    def call(self, compute_only=True, out=None, layer_id=0, **precision):
        return ttnn.experimental.moe_compute(
            self.x,
            self.idx,
            self.sc,
            self.map,
            self.w01,
            self.w2,
            layer_id=layer_id,
            output_height_shard_dim=self.ntp,
            intermediate_size=self.intermediate_size,
            has_bias=self.has_bias,
            activation_type=self.act,
            activation_limit=self.limit,
            compute_only=compute_only,
            optional_output_tensor=out,
            **precision,
        )

    def new_combine_output(self):
        return ttnn.from_torch(
            torch.zeros([self.top_k, self.tokens, self.hidden_size], dtype=torch.bfloat16),
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )

    def combine_golden(self):
        """[top_k, tokens, hidden_size]: page (k, t) holds token t's row of its k-th expert; pages of no routed row stay
        zero."""
        out = torch.zeros(self.top_k, self.tokens, self.hidden_size)
        for e in range(self.local_experts):
            # one row per (token, k) naming e, in token order; a repeated id lands on its last k
            tokens = [t for t in range(self.tokens) for _ in range(self.token_experts[t].count(e))]
            for i, t in enumerate(tokens):
                k_last = len(self.token_experts[t]) - 1 - self.token_experts[t][::-1].index(e)
                out[k_last, t] = self.golden[0, 0, e, i]
        return out

    def combine_rel_l2(self, got):
        ref = self.combine_golden()
        return float((got.float() - ref).norm() / ref.norm())

    def read(self, outs, free=True, check=False):
        """Host copies: counts, activation, e_t (u32), and the last two experts' rows [2, rows, hidden_size] (bf16).
        check: the 6U validators on outputs 0-2."""
        assert len(outs) == 5, f"compute_only must return 5 tensors, got {len(outs)}"
        dram = [ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in (outs[0], outs[1], outs[2], outs[4])]
        if check:
            ok_counts = validate_per_expert_tokens(
                self.mesh_device, self.local_experts, 1, dram[0], self.counts, self.bbox
            )
            ok_act = validate_activation(self.mesh_device, self.local_experts, 1, dram[1], self.golden_act)
            ok_e_t = validate_e_t(self.mesh_device, self.tokens, self.local_experts, 1, dram[2], self.golden_e_t)
            assert ok_counts and ok_act and ok_e_t, f"metadata: counts {ok_counts} activation {ok_act} e_t {ok_e_t}"
        host = [ttnn.to_torch(t) for t in dram]
        rows = None
        if self.counts is not None:
            bufc = torch.zeros(2, dtype=self.counts.dtype)
            for e in range(max(0, self.local_experts - 2), self.local_experts):
                bufc[e % 2] = self.counts[0, e]
            rows = prepare_output_tensor_from_combine_writer(
                host[3], bufc, 0, self.all_cores, self.out_cores, self.ntp, self.dp, 2, self.hidden_size
            )
        for t in dram:
            ttnn.deallocate(t)
        if free:
            for t in (outs[0], outs[1], outs[2], outs[4]):
                ttnn.deallocate(t)
        # activation rows: [token, k per expert, score bits per expert]; the row's alignment words are not defined
        words = 2 * self.local_experts + 1
        act = host[1].reshape(self.tokens, -1)[:, :words]
        return dict(counts=host[0], act=act, e_t=host[2], rows=rows)

    def rows_rel_l2(self, got, layer=0):
        num = den = 0.0
        for e in range(max(0, self.local_experts - 2), self.local_experts):
            n = int(self.counts[0, e])
            ref = self.golden[layer, 0, e, :n].float()
            num += float((got["rows"][e % 2, :n].float() - ref).norm() ** 2)
            den += float(ref.norm() ** 2)
        return (num / den) ** 0.5 if den else 0.0

    def defined_equal(self, a, b):
        """Bitwise equality of everything the contract defines."""
        if not torch.equal(a["counts"][:, : self.local_experts], b["counts"][:, : self.local_experts]):
            return "counts"
        if not torch.equal(a["act"], b["act"]):
            return "activation"
        for e, lst in enumerate(self.golden_e_t[0].values()):
            n = len(lst) + 1
            if not torch.equal(
                a["e_t"][e, : _E_T_ENTRY_WORDS * n : _E_T_ENTRY_WORDS],
                b["e_t"][e, : _E_T_ENTRY_WORDS * n : _E_T_ENTRY_WORDS],
            ):
                return f"e_t {e}"
        for e in range(max(0, self.local_experts - 2), self.local_experts):
            n = int(self.counts[0, e])
            if not torch.equal(a["rows"][e % 2, :n].view(torch.int16), b["rows"][e % 2, :n].view(torch.int16)):
                return f"rows {e}"
        return None


def _assert_rows_bitwise(case, got_read, want_read, label):
    """The last two local experts' rows of two reads (output 4) are bitwise equal."""
    for e in range(max(0, case.local_experts - 2), case.local_experts):
        n = int(case.counts[0, e])
        got, want = got_read["rows"][e % 2, :n].view(torch.int16), want_read["rows"][e % 2, :n].view(torch.int16)
        assert torch.equal(got, want), f"expert {e}: {label}: {_mismatch(got, want)}"


def _traced(mesh_device, call, replays=1):
    """`call` captured as a trace and replayed `replays` times: its outputs, and the trace id to release after
    reading them."""
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        outs = call()
    finally:
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for _ in range(replays):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    return outs, tid


def _programs_added(mesh_device, fn):
    before = mesh_device.num_program_cache_entries()
    out = fn()
    return out, mesh_device.num_program_cache_entries() - before


def _expert_rows_and_ring(case, **precision):
    """One call on each path (fresh program cache): the expert rows path compiles its two programs, the ring path one;
    outputs 0-2 match the goldens on both; the last two experts' rows are bitwise equal between the paths (both
    accumulate every K row in the same order on the same weights)."""
    with _kernel("expert_rows"):
        outs, added = _programs_added(case.mesh_device, lambda: case.call(**precision))
        assert added == 2, f"expert rows call compiled {added} programs, expected 2 (expert rows + metadata)"
        expert_rows = case.read(outs, check=True)
    with _kernel("ring"):
        outs, added = _programs_added(case.mesh_device, lambda: case.call(**precision))
        assert added == 1, f"ring call compiled {added} programs, expected 1"
        ring = case.read(outs, check=True)
    got, want = expert_rows["counts"][:, : case.local_experts], ring["counts"][:, : case.local_experts]
    assert torch.equal(got, want), f"counts differ between the paths: {got[0].tolist()} vs {want[0].tolist()}"
    _assert_rows_bitwise(case, expert_rows, ring, "rows of the expert rows path differ from the ring path's")
    return expert_rows, ring


def _single_device(case, **precision):
    """One SingleDevice call on each path (fresh program cache): outputs 0-2 match the goldens on both; the combine
    output is bitwise equal between the paths (the combine copies rows that are bitwise equal). Returns the combine
    outputs."""
    res = {}
    for name, programs in (("expert_rows", 2), ("ring", 1)):
        with _kernel(name):
            out = case.new_combine_output()
            outs, added = _programs_added(case.mesh_device, lambda: case.call(compute_only=False, out=out, **precision))
            assert len(outs) == 6, f"SingleDevice must return 6 tensors, got {len(outs)}"
            assert added == programs, f"{name} SingleDevice call compiled {added} programs, expected {programs}"
            res[name] = ttnn.to_torch(outs[5]).reshape(case.top_k, case.tokens, case.hidden_size)
            res[name + "_outs"] = case.read(outs[:5], check=True)
            ttnn.deallocate(outs[5])
    got, want = res["expert_rows"].view(torch.int16), res["ring"].view(torch.int16)
    assert torch.equal(got, want), f"combine outputs differ: {_mismatch(got, want)}"
    expert_rows, ring = res["expert_rows_outs"], res["ring_outs"]
    _assert_rows_bitwise(case, expert_rows, ring, "double-buffer rows differ between the paths")
    return res


# moe_compute's single-card shapes and model shapes (local experts, tokens, top_k, intermediate size, hidden size,
# activation, activation limit)
_EXPERT_ROWS_SHAPES = {
    "deepseek_e16_t32": (16, 32, 8, 2048, 7168, MoEActivationFunction.SILU, None),
    "qwen36_e32_t32": (32, 32, 8, 512, 2048, MoEActivationFunction.SILU, None),
    "qwen36_e32_t128": (32, 128, 8, 512, 2048, MoEActivationFunction.SILU, None),
    "flash_next_e128_t1": (128, 1, 8, 640, 2560, MoEActivationFunction.SILU, None),
    "glm53_e18_t32_clamped": (18, 32, 8, 2048, 4096, MoEActivationFunction.CLAMPED_SILU, 10.0),
    "gpt_oss_e16_t32_swiglu": (16, 32, 4, 2880, 2880, MoEActivationFunction.SWIGLU, None),
    "gemma_e8_t32_gelu": (8, 32, 8, 704, 2816, MoEActivationFunction.GELU, None),
}


@pytest.mark.parametrize("device_params", _EXPERT_ROWS_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("shape", sorted(_EXPERT_ROWS_SHAPES))
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_shapes(mesh_device, mesh_shape, shape):
    _skip_unless_supported_arch(mesh_device)
    local_experts, tokens, top_k, intermediate_size, hidden_size, act, limit = _EXPERT_ROWS_SHAPES[shape]
    case = _Case(mesh_device, local_experts, tokens, top_k, intermediate_size, hidden_size, act=act, limit=limit)
    expert_rows, ring = _expert_rows_and_ring(case, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    rel_expert_rows, rel_ring = case.rows_rel_l2(expert_rows), case.rows_rel_l2(ring)
    logger.debug(f"{shape}: rows rel L2 expert rows {rel_expert_rows:.5f} ring {rel_ring:.5f} (HiFi4, FP32 DEST)")
    assert rel_expert_rows <= 0.01, f"expert rows rel L2 {rel_expert_rows}"


@pytest.mark.parametrize("device_params", _EXPERT_ROWS_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("act", [MoEActivationFunction.SILU, MoEActivationFunction.SWIGLU], ids=["silu", "swiglu"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_bias(mesh_device, mesh_shape, act):
    _skip_unless_supported_arch(mesh_device)
    hidden_size = 2880 if act == MoEActivationFunction.SWIGLU else 2048
    intermediate_size = 2880 if act == MoEActivationFunction.SWIGLU else 1024
    case = _Case(mesh_device, 8, 32, 4, intermediate_size, hidden_size, act=act, has_bias=True)
    expert_rows, ring = _expert_rows_and_ring(case, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    rel_expert_rows, rel_ring = case.rows_rel_l2(expert_rows), case.rows_rel_l2(ring)
    logger.debug(f"bias {act}: rows rel L2 expert rows {rel_expert_rows:.5f} ring {rel_ring:.5f}")
    assert rel_expert_rows <= 0.01, f"expert rows rel L2 {rel_expert_rows}"


@pytest.mark.parametrize("device_params", _EXPERT_ROWS_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_top_k_max(mesh_device, mesh_shape):
    """top_k 32, the expert rows reader's largest: both paths, outputs match the goldens."""
    _skip_unless_supported_arch(mesh_device)
    case = _Case(mesh_device, 64, 1, 32, 512, 2048)
    expert_rows, _ = _expert_rows_and_ring(case, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    rel = case.rows_rel_l2(expert_rows)
    assert rel <= 0.01, f"expert rows rel L2 {rel}"


@pytest.mark.parametrize("device_params", _EXPERT_ROWS_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_top_k_over_max(mesh_device, mesh_shape, expect_error):
    """top_k 33: a forced expert rows call is refused before any kernel compiles, and the automatic choice runs the
    ring (one program), whose outputs match the goldens."""
    _skip_unless_supported_arch(mesh_device)
    case = _Case(mesh_device, 64, 1, 33, 512, 2048)
    precision = dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    with _kernel("expert_rows"), expect_error(RuntimeError, "top_k 33 outside 1..32"):
        case.call(**precision)
    outs, added = _programs_added(mesh_device, lambda: case.call(**precision))
    assert added == 1, f"top_k 33 compiled {added} programs, expected the ring's 1"
    case.read(outs, check=True)


@pytest.mark.parametrize("device_params", _EXPERT_ROWS_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "act, limit, error",
    [
        (MoEActivationFunction.CLAMPED_SILU, None, "activation_limit is required for CLAMPED_SILU"),
        (MoEActivationFunction.CLAMPED_SILU, float("nan"), "activation_limit must be finite and positive"),
        (MoEActivationFunction.SILU, 10.0, "must be omitted for other activations"),
    ],
    ids=["missing", "nan", "silu-with-limit"],
)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_rejects_activation_limit(mesh_device, mesh_shape, expect_error, act, limit, error):
    """A forced expert rows call checks activation_limit before any kernel compiles."""
    _skip_unless_supported_arch(mesh_device)
    case = _Case(mesh_device, 8, 32, 4, 1024, 2048, act=act, limit=limit, goldens=False)
    before = mesh_device.num_program_cache_entries()
    with _kernel("expert_rows"), expect_error(RuntimeError, error):
        case.call()
    after = mesh_device.num_program_cache_entries()
    assert after == before, f"a rejected call changed the program cache: {before} -> {after} entries"


@pytest.mark.parametrize("device_params", _EXPERT_ROWS_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("tokens", [1, 3, 32, 63, 128])
# hidden size 448 = 14 tiles: output width shard dim 2, and a W2 output tile in every DRAM bank (Wormhole 12,
# Blackhole 8)
@pytest.mark.parametrize("cfg", [(4, 4, 256, 512), (8, 4, 256, 448)], ids=["c4_h512", "c2_h448"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_tokens(mesh_device, mesh_shape, cfg, tokens):
    _skip_unless_supported_arch(mesh_device)
    local_experts, top_k, intermediate_size, hidden_size = cfg
    intermediate_size = max(intermediate_size, _TILE_WIDTH * effective_matmul_ring_size(mesh_device))
    case = _Case(mesh_device, local_experts, tokens, top_k, intermediate_size, hidden_size)
    expert_rows, ring = _expert_rows_and_ring(case, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    rel = case.rows_rel_l2(expert_rows)
    assert rel <= 0.01, f"expert rows rel L2 {rel}"


@pytest.mark.parametrize("device_params", _EXPERT_ROWS_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("kind", ["skewed", "empty"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_routing(mesh_device, mesh_shape, kind):
    """Uneven experts (one expert in every token: jobs of 32 rows chained) and experts with no token."""
    _skip_unless_supported_arch(mesh_device)
    case = _Case(mesh_device, 16, 128, 8, 512, 2048, kind=kind)
    expert_rows, _ = _expert_rows_and_ring(case, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    rel = case.rows_rel_l2(expert_rows)
    assert rel <= 0.01, f"expert rows rel L2 {rel}"


@pytest.mark.parametrize("device_params", _EXPERT_ROWS_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_many_jobs_per_group(mesh_device, mesh_shape):
    """36 local experts, 128 tokens of top-8: more jobs than expert groups, every intermediate slot reused."""
    _skip_unless_supported_arch(mesh_device)
    case = _Case(mesh_device, 36, 128, 8, 2048, 4096)
    expert_rows, _ = _expert_rows_and_ring(case, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    rel = case.rows_rel_l2(expert_rows)
    assert rel <= 0.01, f"expert rows rel L2 {rel}"


_PRECISION_MODES = {
    "default": {},
    "lofi_bf16": dict(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False),
    "lofi_fp32": dict(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True),
    "hifi2_bf16": dict(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False),
    "hifi4_fp32": dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
}


@pytest.mark.parametrize("device_params", _EXPERT_ROWS_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mode", sorted(_PRECISION_MODES))
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_deterministic(mesh_device, mesh_shape, mode):
    """Eager, repeated eager and a trace replay give bitwise-equal outputs in every DEST mode and with the defaults."""
    _skip_unless_supported_arch(mesh_device)
    precision = _PRECISION_MODES[mode]
    case = _Case(mesh_device, 32, 128, 8, 512, 2048)
    with _kernel("expert_rows"):
        first = case.read(case.call(**precision), check=True)
        second = case.read(case.call(**precision))
        assert (diff := case.defined_equal(first, second)) is None, f"repeated eager call differs: {diff}"
        outs, tid = _traced(mesh_device, lambda: case.call(**precision))
        traced = case.read(outs, free=False)
        ttnn.release_trace(mesh_device, tid)
    assert (diff := case.defined_equal(first, traced)) is None, f"traced call differs: {diff}"


_SINGLE_DEVICE_SHAPES = [
    "deepseek_e16_t32",
    "flash_next_e128_t1",
    "gemma_e8_t32_gelu",
    "glm53_e18_t32_clamped",
    "gpt_oss_e16_t32_swiglu",
]


@pytest.mark.parametrize("device_params", _EXPERT_ROWS_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("shape", _SINGLE_DEVICE_SHAPES)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_full_local(mesh_device, mesh_shape, shape):
    """SingleDevice (compute_only=False on one card): the combine output of the expert rows path is bitwise equal to the
    ring path's and within 1 % rel L2 of the FP32 golden (HiFi4, FP32 DEST)."""
    _skip_unless_supported_arch(mesh_device)
    local_experts, tokens, top_k, intermediate_size, hidden_size, act, limit = _EXPERT_ROWS_SHAPES[shape]
    case = _Case(mesh_device, local_experts, tokens, top_k, intermediate_size, hidden_size, act=act, limit=limit)
    res = _single_device(case, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    rel = case.combine_rel_l2(res["expert_rows"])
    logger.debug(f"{shape}: SingleDevice combine rel L2 {rel:.5f} (HiFi4, FP32 DEST)")
    assert rel <= 0.01, f"combine output rel L2 {rel}"


@pytest.mark.parametrize("device_params", _EXPERT_ROWS_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("tokens", [1, 64])
@pytest.mark.parametrize("kind", ["random", "skewed", "empty"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_full_local_routing(mesh_device, mesh_shape, kind, tokens):
    """SingleDevice over token counts and uneven experts (experts with no token, one expert in every token)."""
    _skip_unless_supported_arch(mesh_device)
    case = _Case(mesh_device, 16, tokens, 8, 512, 2048, kind=kind)
    res = _single_device(case, math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    rel = case.combine_rel_l2(res["expert_rows"])
    assert rel <= 0.01, f"combine output rel L2 {rel}"


@pytest.mark.parametrize("device_params", _EXPERT_ROWS_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mode", sorted(_PRECISION_MODES))
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_full_local_deterministic(mesh_device, mesh_shape, mode):
    """SingleDevice: eager, repeated eager and a trace replay give bitwise-equal combine outputs in every DEST mode and
    with the defaults."""
    _skip_unless_supported_arch(mesh_device)
    precision = _PRECISION_MODES[mode]
    case = _Case(mesh_device, 32, 128, 8, 512, 2048)

    def run():
        out = case.new_combine_output()
        outs = case.call(compute_only=False, out=out, **precision)
        return outs, out

    def combine(outs):
        got = ttnn.to_torch(outs[5]).view(torch.int16)
        case.read(outs[:5])
        ttnn.deallocate(outs[5])
        return got

    with _kernel("expert_rows"):
        first = combine(run()[0])
        second = combine(run()[0])
        assert torch.equal(first, second), f"repeated eager SingleDevice call differs: {_mismatch(first, second)}"
        out = case.new_combine_output()
        outs, tid = _traced(mesh_device, lambda: case.call(compute_only=False, out=out, **precision))
        traced = ttnn.to_torch(outs[5]).view(torch.int16)
        ttnn.release_trace(mesh_device, tid)
    assert torch.equal(first, traced), f"traced SingleDevice call differs: {_mismatch(first, traced)}"


@pytest.mark.parametrize("device_params", _EXPERT_ROWS_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_layer_id(mesh_device, mesh_shape):
    """Three layers of different weights: each layer_id's rows match that layer's golden, and the two expert rows
    programs compiled for the first layer serve the others (layer_id is a runtime offset, not a cache key)."""
    _skip_unless_supported_arch(mesh_device)
    case = _Case(mesh_device, 8, 32, 8, 512, 2048, layers=3)
    precision = dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    entries = []
    with _kernel("expert_rows"):
        for layer in (0, 1, 2, 1):
            got = case.read(case.call(layer_id=layer, **precision), check=True)
            entries.append(mesh_device.num_program_cache_entries())
            rel = case.rows_rel_l2(got, layer)
            other = case.rows_rel_l2(got, (layer + 1) % 3)
            logger.debug(f"layer {layer}: rows rel L2 {rel:.5f} (against layer {(layer + 1) % 3}: {other:.3f})")
            assert rel <= 0.01, f"layer {layer}: rows rel L2 {rel}"
            assert other > 0.5, f"layer {layer}: the rows also match layer {(layer + 1) % 3}"
    assert entries[1:] == [entries[0]] * 3, f"program cache entries per layer: {entries}"


@pytest.mark.parametrize("device_params", _EXPERT_ROWS_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("compute_only", [True, False], ids=["compute_only", "full_local"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_duplicate_ids(mesh_device, mesh_shape, compute_only):
    """A token naming one expert at two k is listed once per k, as moe_compute lists it: counts and the e_t page hold
    every (token, k), the activation row holds the last k, each listing has its row, and the combine writes page
    (last k, t). Checked against the routing itself (the ring path's e_t pages put a repeated token out of order),
    counts against the ring path's."""
    _skip_unless_supported_arch(mesh_device)
    case = _Case(mesh_device, 16, 32, 8, 512, 2048, kind="dup")
    precision = dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    with _kernel("expert_rows"):
        out = None if compute_only else case.new_combine_output()
        outs = case.call(compute_only=compute_only, out=out, **precision)
        combine = None
        if not compute_only:
            combine = ttnn.to_torch(outs[5]).reshape(case.top_k, case.tokens, case.hidden_size)
            ttnn.deallocate(outs[5])
        expert_rows = case.read(outs[:5])
    with _kernel("ring"):
        ring = case.read(case.call(**precision))
    lists = [
        [t for t in range(case.tokens) for _ in range(case.token_experts[t].count(e))]
        for e in range(case.local_experts)
    ]
    counts = [len(lst) for lst in lists]
    got_counts = expert_rows["counts"][0, : case.local_experts].tolist()
    assert got_counts == counts, f"counts {got_counts}, expected one per (token, k): {counts}"
    got, want = expert_rows["counts"][:, : case.local_experts], ring["counts"][:, : case.local_experts]
    assert torch.equal(got, want), f"counts {got[0].tolist()} differ from the ring path's {want[0].tolist()}"
    for e in range(case.local_experts):
        n = counts[e]
        page = expert_rows["e_t"][e, : _E_T_ENTRY_WORDS * n : _E_T_ENTRY_WORDS].tolist()
        assert page == lists[e], f"e_t page {e}: {page}, expected {lists[e]}"
    act = expert_rows["act"].to(torch.int64)
    rows = [t for t in range(case.tokens) if any(e < case.local_experts for e in case.token_experts[t])]
    for r, t in enumerate(rows):
        assert int(act[r, 0]) == t, f"activation row {r} is token {int(act[r, 0])}, expected {t}"
        for e in range(case.local_experts):
            ks = [k for k, x in enumerate(case.token_experts[t]) if x == e]
            got = int(act[r, 1 + e])
            assert (got == ks[-1]) if ks else (got >= case.top_k), f"token {t} expert {e}: k {got}, listed at {ks}"
    rel = case.rows_rel_l2(expert_rows)
    assert rel <= 0.01, f"expert rows rel L2 {rel}"
    if combine is not None:
        rel = case.combine_rel_l2(combine)
        assert rel <= 0.01, f"combine output rel L2 {rel}"


@pytest.mark.parametrize("device_params", _EXPERT_ROWS_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("compute_only", [True, False], ids=["compute_only", "full_local"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_listing_over_tokens(mesh_device, mesh_shape, compute_only):
    """Every k of every token names expert 0 (top_k * tokens entries, more than its e_t page holds): the expert rows
    path keeps the first `tokens` entries in token order (tokens 0 .. tokens / top_k - 1, top_k times each), reports
    that count, and finishes."""
    _skip_unless_supported_arch(mesh_device)
    tokens, top_k = 32, 8
    case = _Case(mesh_device, 8, tokens, top_k, 512, 2048, kind="all_one", goldens=False)
    with _kernel("expert_rows"):
        out = None if compute_only else case.new_combine_output()
        outs = case.call(compute_only=compute_only, out=out)
        if not compute_only:
            ttnn.deallocate(outs[5])
        got = case.read(outs[:5])
    assert int(got["counts"][0, 0]) == tokens, f"expert 0 keeps {int(got['counts'][0, 0])} rows, expected {tokens}"
    others = [int(got["counts"][0, e]) for e in range(1, case.local_experts)]
    assert not any(others), f"experts 1..{case.local_experts - 1} keep rows: {others}"
    expected = [t for t in range(tokens // top_k) for _ in range(top_k)]
    page = got["e_t"][0, : _E_T_ENTRY_WORDS * tokens : _E_T_ENTRY_WORDS].tolist()
    assert page == expected, f"expert 0's page {page}, expected the first tokens entries in token order {expected}"
    terminator = int(got["e_t"][0, _E_T_ENTRY_WORDS * tokens]) & _E_T_END
    assert terminator == _E_T_END, f"page terminator {terminator:#x}"


# weight-stationary jobs: (local experts, tokens, top_k, intermediate size, hidden size, activation, limit, bias);
# hot_last routing gives the two experts output 4 keeps `tokens` rows (jobs of several row tiles), cold_last gives them
# 20 (one-row-tile jobs among jobs of several). On 8 DRAM banks 2560/640 (20-tile transactions, a last W2 output group
# two tiles wide) and 2816/704 have compact W0/W1 layouts: odd column counts (half block-columns) and slices that go on
# in the next bank; 2880/704 with bias stores an odd K, 2880 / 32 + 1 bias row = 91 tiles, in its half block-columns;
# 2048/1792 has 1792 / 32 / 8 = 7 columns per bank, so a core holds a column pair and a half block-column, and with 10
# experts of top 8 most jobs are of 3 row tiles: hot_last 96 rows for the last two experts and 96 * 6 / 8 = 72 for the
# others, cold_last (20 * 6 + 76 * 8) / 8 = 91 for the others; they leave 4- and 2-tile partials between x chunks.
_ROW_TILE_SHAPES = {
    "deepseek_e16_t128": (16, 128, 8, 2048, 7168, MoEActivationFunction.SILU, None, False),
    "glm53_e18_t96_clamped": (18, 96, 8, 2048, 4096, MoEActivationFunction.CLAMPED_SILU, 10.0, False),
    "qwen36_e32_t128": (32, 128, 8, 512, 2048, MoEActivationFunction.SILU, None, False),
    "gpt_oss_e8_t128_bias": (8, 128, 4, 2880, 2880, MoEActivationFunction.SWIGLU, None, True),
    "flash_next_e32_t128": (32, 128, 8, 640, 2560, MoEActivationFunction.SILU, None, False),
    "gemma_e16_t128_gelu": (16, 128, 8, 704, 2816, MoEActivationFunction.GELU, None, False),
    "h2880_n704_e8_t128_bias": (8, 128, 4, 704, 2880, MoEActivationFunction.SWIGLU, None, True),
    "h2048_n1792_e10_t96": (10, 96, 8, 1792, 2048, MoEActivationFunction.SILU, None, False),
}


def _rows_with(case, row_tiles, **precision):
    """One ComputeOnly expert rows call with jobs of `row_tiles` row tiles (fresh program cache), read back and
    checked."""
    case.mesh_device.disable_and_clear_program_cache()
    case.mesh_device.enable_program_cache()
    with _kernel("expert_rows"), _env(_ROW_TILES_ENV, row_tiles):
        outs, added = _programs_added(case.mesh_device, lambda: case.call(**precision))
        assert added == 2, f"row tiles {row_tiles}: compiled {added} programs, expected the 2 expert rows programs"
        return case.read(outs, check=True)


# every shape once, each (row tiles, routing) pair on two shapes
_ROW_TILE_CASES = [
    ("deepseek_e16_t128", 2, "hot_last"),
    ("flash_next_e32_t128", 4, "hot_last"),
    ("gemma_e16_t128_gelu", 2, "cold_last"),
    ("glm53_e18_t96_clamped", 4, "cold_last"),
    ("gpt_oss_e8_t128_bias", 2, "hot_last"),
    ("h2048_n1792_e10_t96", 4, "hot_last"),
    ("h2880_n704_e8_t128_bias", 2, "cold_last"),
    ("qwen36_e32_t128", 4, "cold_last"),
]


@pytest.mark.parametrize("device_params", _EXPERT_ROWS_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("shape, row_tiles, kind", _ROW_TILE_CASES)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_row_tiles(mesh_device, mesh_shape, shape, row_tiles, kind):
    """Jobs of several row tiles (each expert's weights read once for up to 32 M rows), and jobs of one row tile in the
    same program, give the rows of one row tile per job bitwise, and those equal the ring's."""
    _skip_unless_supported_arch(mesh_device)
    _skip_row_tiles_on_wormhole(mesh_device)
    local_experts, tokens, top_k, intermediate_size, hidden_size, act, limit, bias = _ROW_TILE_SHAPES[shape]
    case = _Case(
        mesh_device,
        local_experts,
        tokens,
        top_k,
        intermediate_size,
        hidden_size,
        act=act,
        limit=limit,
        has_bias=bias,
        kind=kind,
    )
    precision = dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    one = _rows_with(case, 1, **precision)
    many = _rows_with(case, row_tiles, **precision)
    assert (diff := case.defined_equal(one, many)) is None, f"{row_tiles} row tiles per job differ: {diff}"
    with _kernel("ring"):
        ring = case.read(case.call(**precision), check=True)
    # against the ring: counts and rows, as _expert_rows_and_ring (outputs 1 / 2 are checked against the goldens above)
    got, want = many["counts"][:, :local_experts], ring["counts"][:, :local_experts]
    assert torch.equal(got, want), f"counts {got[0].tolist()} differ from the ring's {want[0].tolist()}"
    _assert_rows_bitwise(case, many, ring, f"{row_tiles} row tiles per job differ from the ring's rows")
    rel = case.rows_rel_l2(many)
    logger.debug(f"{shape} {kind} M {row_tiles}: rows rel L2 {rel:.5f} (HiFi4, FP32 DEST)")
    assert rel <= 0.01, f"rows rel L2 {rel}"


@pytest.mark.parametrize("device_params", _EXPERT_ROWS_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "mode, shape, kind",
    [
        ("default", "deepseek_e16_t128", "hot_last"),
        ("hifi2_bf16", "flash_next_e32_t128", "cold_last"),
        ("hifi4_fp32", "gemma_e16_t128_gelu", "hot_last"),
        ("lofi_bf16", "deepseek_e16_t128", "cold_last"),
        ("lofi_fp32", "flash_next_e32_t128", "hot_last"),
    ],
)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_row_tiles_deterministic(mesh_device, mesh_shape, mode, shape, kind):
    """In every DEST mode: jobs of 4 row tiles (and of one row tile in the same program) give one row tile per job's
    rows bitwise, and those equal the ring's; eager, repeated eager and two trace replays are bitwise equal."""
    _skip_unless_supported_arch(mesh_device)
    _skip_row_tiles_on_wormhole(mesh_device)
    precision = _PRECISION_MODES[mode]
    local_experts, tokens, top_k, intermediate_size, hidden_size, act, limit, bias = _ROW_TILE_SHAPES[shape]
    case = _Case(
        mesh_device,
        local_experts,
        tokens,
        top_k,
        intermediate_size,
        hidden_size,
        act=act,
        limit=limit,
        has_bias=bias,
        kind=kind,
    )
    one = _rows_with(case, 1, **precision)
    first = _rows_with(case, 4, **precision)
    assert (diff := case.defined_equal(one, first)) is None, f"4 row tiles per job differ from 1: {diff}"
    with _kernel("ring"):
        ring = case.read(case.call(**precision))
    _assert_rows_bitwise(case, first, ring, "4 row tiles per job differ from the ring's rows")
    with _kernel("expert_rows"), _env(_ROW_TILES_ENV, 4):
        second = case.read(case.call(**precision))
        assert (diff := case.defined_equal(first, second)) is None, f"repeated eager call differs: {diff}"
        outs, tid = _traced(mesh_device, lambda: case.call(**precision), replays=2)
        traced = case.read(outs, free=False)
        ttnn.release_trace(mesh_device, tid)
    assert (diff := case.defined_equal(first, traced)) is None, f"traced call differs: {diff}"


@pytest.mark.parametrize("device_params", _EXPERT_ROWS_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("kind, mode", [("hot_last", "hifi4_fp32"), ("cold_last", "lofi_bf16")])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_row_tiles_full_local(mesh_device, mesh_shape, kind, mode):
    """SingleDevice: the combine output (every routed row of every expert) with jobs of 4 row tiles is bitwise equal to
    one row tile per job and to the ring path's."""
    _skip_unless_supported_arch(mesh_device)
    _skip_row_tiles_on_wormhole(mesh_device)
    precision = _PRECISION_MODES[mode]
    case = _Case(mesh_device, 16, 128, 8, 2048, 7168, kind=kind)
    res = _single_device(case, **precision)
    case.mesh_device.disable_and_clear_program_cache()
    case.mesh_device.enable_program_cache()
    with _kernel("expert_rows"), _env(_ROW_TILES_ENV, 4):
        out = case.new_combine_output()
        outs, added = _programs_added(case.mesh_device, lambda: case.call(compute_only=False, out=out, **precision))
        assert added == 2, f"SingleDevice call with 4 row tiles per job compiled {added} programs, expected 2"
        many = ttnn.to_torch(outs[5]).reshape(case.top_k, case.tokens, case.hidden_size)
        case.read(outs[:5], check=True)
        ttnn.deallocate(outs[5])
    got, want = many.view(torch.int16), res["expert_rows"].view(torch.int16)
    assert torch.equal(got, want), f"combine output differs from M = 1: {_mismatch(got, want)}"


@pytest.mark.parametrize("device_params", _EXPERT_ROWS_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_expert_rows_row_tiles_refused_on_wormhole(mesh_device, mesh_shape, expect_error):
    """Wormhole: a forced M > 1 is refused before any kernel compiles."""
    if mesh_device.arch() != ttnn.device.Arch.WORMHOLE_B0:
        pytest.skip("Wormhole runs one row tile per job; other architectures run M > 1")
    case = _Case(mesh_device, 16, 64, 8, 512, 2048, goldens=False)
    before = mesh_device.num_program_cache_entries()
    with _kernel("expert_rows"), _env(_ROW_TILES_ENV, 2):
        with expect_error(RuntimeError, "Wormhole runs one row tile per job"):
            case.call()
    after = mesh_device.num_program_cache_entries()
    assert after == before, f"a refused call changed the program cache: {before} -> {after} entries"
