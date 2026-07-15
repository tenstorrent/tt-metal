# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Single-card MoE compute test (1x1 mesh, cluster_axis=None). Runs on both WH
and BH; other arches are skipped at fixture time.

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
    skip_on_ci=False,
    compute_only=True,
):
    """
    Single-card MoE compute test body. cluster_axis is fixed to None
    (no dispatch axis on 1x1 mesh).

    The matmul ring size is auto-detected from the live DRAM-bank count (12 on WH, 7/8 on
    BH) — the same ``effective_matmul_ring_size(mesh_device)`` the public op uses — and is used
    to pack the weights so host tensor layout matches the op's ring-aware width-parallel
    auto-derivation.
    """
    # The MoE op uses tilize cores keyed off the per-arch layout table in the program
    # factory's `get_layout()` (see #41827) and matmul cores from
    # `get_optimal_dram_bank_to_logical_worker_assignment`. WH expects a 7x10 unharvested
    # logical worker grid (drain at (6,9)); BH expects an 11x10 grid (drain at (10,9)).
    # On a harvested WH (e.g. n150_L = 7x9 grid with COL-axis dispatch) the harvested-row
    # remapping shifts the matmul layout into the y=8 row, which then overlaps the
    # filtered tilize cores — the op TT_FATALs on `tilize and matmul bounding boxes
    # cannot overlap`. Untangling the matmul/tilize layouts under harvest is a follow-up
    # (arch-aware core placement); for now we skip on harvested grids.
    arch = mesh_device.arch()
    grid = mesh_device.compute_with_storage_grid_size()
    if arch not in (ttnn.device.Arch.WORMHOLE_B0, ttnn.device.Arch.BLACKHOLE):
        pytest.skip(f"MoE compute single-card test: arch {arch} is not supported (only WH and BH).")
    elif arch == ttnn.device.Arch.BLACKHOLE and skip_on_ci:
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
    assert num_devices == 1, "single-card test must be run on a 1x1 mesh"
    num_dispatch_devices = num_devices  # cluster_axis is None
    num_replicated_devices = num_devices // num_dispatch_devices
    total_tokens = tokens_per_device * num_dispatch_devices

    experts = experts_per_device * num_devices
    experts_per_cluster = experts // num_replicated_devices

    logger.info(f"Single-card MoE compute test:")
    logger.info(f"  mesh_shape: {mesh_shape}")
    logger.info(f"  cluster_axis: {cluster_axis}")
    logger.info(f"  compute_only: {compute_only}")
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
    drain_core_coord = ttnn.experimental.get_moe_tilize_drain_core(
        mesh_device,
        output_height_shard_dim,
        output_width_shard_dim,
        hidden_size,
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
    sparse_buffer, expert_indices, expert_scores, _ = gen_sparse_buffer_and_indices(
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

    # Sparse buffer / indices / scores tensors.
    tt_sparse_buffer = ttnn.from_torch(
        sparse_buffer,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dtype,
        memory_config=sparse_mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    expert_indices_flat = expert_indices.reshape(total_tokens, selected_experts_k)
    expert_indices_replicated = expert_indices_flat.unsqueeze(0).repeat(num_devices, 1, 1)
    tt_expert_indices = ttnn.from_torch(
        expert_indices_replicated,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=expert_indices_mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    expert_scores_flat = expert_scores.reshape(total_tokens, selected_experts_k)
    expert_scores_replicated = expert_scores_flat.unsqueeze(0).repeat(num_devices, 1, 1)
    tt_expert_scores = ttnn.from_torch(
        expert_scores_replicated,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=dtype,
        memory_config=expert_scores_mem_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
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

    def create_combine_output_tensor():
        torch_combine_output = torch.zeros([selected_experts_k, total_tokens, hidden_size], dtype=torch.bfloat16)
        return ttnn.from_torch(
            torch_combine_output,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )

    def run_moe_compute_once(optional_combine_output_tensor):
        return ttnn.experimental.moe_compute(
            tt_sparse_buffer,
            tt_expert_indices,
            tt_expert_scores,
            tt_expert_mapping,
            tt_w0_w1,
            tt_w2,
            layer_id=layer_id,
            output_height_shard_dim=output_height_shard_dim,
            intermediate_size=N,
            has_bias=has_bias,
            # cluster_axis=None: required for both compute_only and single-device fused (FullLocal).
            # topology/num_links/mux/semaphore must be None for both paths.
            cluster_axis=None,
            topology=None,
            num_links=None,
            mux_core_range_set=None,
            optional_output_tensor=optional_combine_output_tensor,
            optional_cross_device_semaphore=None,
            activation_type=activation_type,
            compute_only=compute_only,
        )

    def deallocate_l1_moe_compute_outputs(output_tensors):
        # Slots 3 and 4 share a backing buffer; deallocating slot 4 releases the shared L1 output.
        for tensor in (output_tensors[0], output_tensors[1], output_tensors[2], output_tensors[4]):
            ttnn.deallocate(tensor)

    # For local-full mode (compute_only=False on 1x1), pre-allocate the optional combine output tensor.
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
    output_shard_cores = ttnn.experimental.get_moe_combine_cores(
        mesh_device, output_height_shard_dim, output_width_shard_dim, hidden_size
    )
    worker_mcast_bbox = ttnn.experimental.get_moe_worker_mcast_bounding_box(
        mesh_device, output_height_shard_dim, output_width_shard_dim, hidden_size
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
        # validate_combine uses cluster_axis for the mesh composer; on 1x1, dim=1 is correct.
        combine_all_passed = validate_combine(
            layer_id,
            mesh_device,
            cluster_axis=1,  # single device: either axis works
            tt_combine_output=combine_output_tensor,
            combine_goldens=combine_goldens,
            pcc_threshold=base_pcc_threshold,
        )

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
    assert matmul_all_passed, "Matmul output tensor verification failed!"
    if not compute_only:
        assert combine_all_passed, "Combine output tensor verification failed!"

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

        cache_hit_combine_output_tensor = ttnn.to_memory_config(
            cache_hit_outputs[5], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        cache_hit_combine_all_passed = validate_combine(
            layer_id,
            mesh_device,
            cluster_axis=1,  # single device: either axis works
            tt_combine_output=cache_hit_combine_output_tensor,
            combine_goldens=combine_goldens,
            pcc_threshold=base_pcc_threshold,
        )
        logger.info(f"Combine Output Tensor Cache Hit: {'PASSED' if cache_hit_combine_all_passed else 'FAILED'}")
        assert cache_hit_combine_all_passed, "Combine output tensor cache-hit verification failed!"

        deallocate_l1_moe_compute_outputs(cache_hit_outputs)
        ttnn.deallocate(cache_hit_outputs[5])


# DeepSeek-shaped workload mirrored on a single WH card so kernels see the same dims.
#
# device_params: must match the 6U test environment.
#   - dispatch_core_axis=COL: reserves a *column* (not a row) for fast-dispatch cores.
#     The MoE op hardcodes tilize cores at logical (5,8)(5,9)(6,8)(6,9) and a combine
#     bounding box up to y=7. With ROW-axis dispatch, one row gets eaten from the top
#     of the compute grid (logical y range becomes 0..8 on unharvested WH, 0..7 on a
#     harvested n150_L), and (6,9)/(5,9) fall out of range. COL-axis matches the 6U
#     test setup and keeps all 10 functional rows (logical y=0..9) available.
#
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
@pytest.mark.parametrize("has_bias", [False, True], ids=["no_bias", "with_bias"])
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_single_card_deepseek(
    mesh_device, mesh_shape, has_bias, compute_only, is_ci_env, is_ci_v2_env
):
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
        experts_per_device=64,
        tokens_per_device=32,
        selected_experts_k=8,
        N=N,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size, matmul_ring_size=ring_n),
        dtype=ttnn.bfloat16,
        activation_type=MoEActivationFunction.SILU,
        has_bias=has_bias,
        skip_on_ci=is_ci_env or is_ci_v2_env,
        compute_only=compute_only,
    )


# GPT-OSS canonical config from the GPT-OSS entry in `_MODELS_1x8` (test_moe_compute_6U.py):
#   experts_per_device=4, has_bias=True, activation=SWIGLU.
# Ring-aware width dim is auto-derived using effective_matmul_ring_size (see output_width_shard_dim).
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
        experts_per_device=64,
        tokens_per_device=32,
        selected_experts_k=4,
        N=hidden_size,
        hidden_size=hidden_size,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(hidden_size, matmul_ring_size=ring_n),
        dtype=ttnn.bfloat16,
        activation_type=MoEActivationFunction.SWIGLU,
        has_bias=True,
        skip_on_ci=is_ci_env or is_ci_v2_env,
        compute_only=compute_only,
    )


# Minimal sanity check that compute_only=True with conflicting CCL kwargs is rejected.
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
def test_moe_compute_compute_only_rejects_cluster_axis(mesh_device, mesh_shape, expect_error):
    """compute_only=True with cluster_axis set must raise (loud rejection per spec)."""
    # Build minimal valid input shapes; we do NOT need the op to actually run --
    # validation must reject the bad arg combination before kernel launch.
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
    # Dummy weights -- shapes don't matter; the failure should fire at the public
    # API validation step before the device op runs.
    dummy_weight = torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16)
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

    with expect_error(RuntimeError, r"compute_only.*cluster_axis"):
        ttnn.experimental.moe_compute(
            tt_sparse,
            tt_indices,
            tt_scores,
            tt_mapping,
            tt_w0_w1,
            tt_w2,
            layer_id=0,
            output_height_shard_dim=4,
            intermediate_size=2048,
            has_bias=False,
            cluster_axis=1,  # <-- conflicting with compute_only=True
            topology=None,
            num_links=None,
            mux_core_range_set=None,
            optional_output_tensor=None,
            optional_cross_device_semaphore=None,
            activation_type=MoEActivationFunction.SILU,
            compute_only=True,
        )
