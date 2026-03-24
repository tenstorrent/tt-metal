# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import ttnn
from conftest import reset_fabric, set_fabric
from tests.scripts.common import get_updated_device_params


# Standalone entrypoint for the current minimal reproduced router-path hang.
# It should run with just:
#   python3 -m pytest -svv tests/ttnn/unit_tests/operations/data_movement/test_again_hang_router_hidden_micro_repro.py
LOOPS_ENV_VAR = "TT_METAL_RESHAPE_HANG_REPRO_LOOPS"


def _get_num_loops():
    return int(os.getenv(LOOPS_ENV_VAR, "1"))


def _deallocate_tensors(*tensors):
    for tensor in tensors:
        if tensor is None:
            continue
        try:
            ttnn.deallocate(tensor)
        except Exception:
            pass


def _prepare_mesh_repro(mesh_shape, device_params):
    if not ttnn.using_distributed_env() and mesh_shape[0] * mesh_shape[1] > ttnn.get_num_devices():
        pytest.skip("Requested more devices than available. Test not applicable for machine")

    updated_device_params = get_updated_device_params(device_params)
    fabric_config = updated_device_params.pop("fabric_config", None)
    fabric_tensix_config = updated_device_params.pop("fabric_tensix_config", None)
    reliability_mode = updated_device_params.pop("reliability_mode", None)
    fabric_manager = updated_device_params.pop("fabric_manager", None)
    fabric_router_config = updated_device_params.pop("fabric_router_config", None)
    fabric_state = (
        fabric_config,
        reliability_mode,
        fabric_tensix_config,
        fabric_manager,
        fabric_router_config,
    )
    _set_repro_fabric(fabric_state)
    return updated_device_params, fabric_state


def _set_repro_fabric(fabric_state):
    fabric_config, reliability_mode, fabric_tensix_config, fabric_manager, fabric_router_config = fabric_state
    set_fabric(fabric_config, reliability_mode, fabric_tensix_config, fabric_manager, fabric_router_config)


def _reset_repro_fabric(fabric_state):
    if fabric_state is None:
        return
    fabric_config, _, _, _, _ = fabric_state
    reset_fabric(fabric_config)


def _open_repro_mesh_device(mesh_shape, open_device_params):
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape), **open_device_params)


def _close_repro_mesh_device(mesh_device):
    if mesh_device is None:
        return

    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh_device)


def _make_hifi4_compute_config(mesh_device, packer_l1_acc=False):
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
    )


def _make_replicated_tiled_from_host(host_tensor, mesh_device, replicated, mem, output_dtype=ttnn.bfloat16):
    host_rm = None
    host_tile = None
    output_tensor = None
    try:
        host_rm = ttnn.from_torch(
            host_tensor.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=replicated,
            memory_config=mem,
        )
        host_tile = ttnn.to_layout(
            host_rm,
            ttnn.TILE_LAYOUT,
            memory_config=mem,
        )
        if output_dtype == ttnn.bfloat16:
            output_tensor = host_tile
            host_tile = None
        else:
            output_tensor = ttnn.typecast(host_tile, dtype=output_dtype)
        return output_tensor
    finally:
        _deallocate_tensors(host_rm, host_tile)


def _make_replicated_permuted_bfp8_weight(host_tensor, mesh_device, replicated, mem):
    host_rm = None
    host_tile = None
    host_permuted = None
    host_bfp8 = None
    try:
        host_rm = ttnn.from_torch(
            host_tensor.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=replicated,
            memory_config=mem,
        )
        host_tile = ttnn.to_layout(
            host_rm,
            ttnn.TILE_LAYOUT,
            memory_config=mem,
        )
        host_permuted = ttnn.permute(host_tile, (1, 0))
        host_bfp8 = ttnn.typecast(host_permuted, dtype=ttnn.bfloat8_b)
        return host_bfp8
    finally:
        _deallocate_tensors(host_rm, host_tile, host_permuted)


def _make_local_expert_ids_tile(mesh_device, num_experts, local_experts, mem):
    local_expert_ids_rm = None
    local_expert_ids_3d_rm = None
    local_expert_ids_tile = None
    try:
        local_expert_ids_rm = ttnn.from_torch(
            torch.arange(num_experts, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            memory_config=mem,
        )
        local_expert_ids_3d_rm = ttnn.reshape(
            local_expert_ids_rm,
            ttnn.Shape((1, 1, local_experts)),
            memory_config=mem,
        )
        local_expert_ids_tile = ttnn.to_layout(
            local_expert_ids_3d_rm,
            ttnn.TILE_LAYOUT,
            memory_config=mem,
        )
        return local_expert_ids_tile
    finally:
        _deallocate_tensors(local_expert_ids_rm, local_expert_ids_3d_rm)


def _trace_repro_step(test_label, loop_index, num_loops, layer_index, num_layers, step_name):
    print(
        f"[{test_label}] loop={loop_index + 1}/{num_loops} " f"layer={layer_index + 1}/{num_layers} step={step_name}",
        flush=True,
    )


def _run_again_hang_router_hidden_micro_repro_mesh(mesh_shape, device_params, test_label):
    num_loops = _get_num_loops()
    num_layers = 2

    batch = 32
    seq_len = 17
    tokens = batch * seq_len
    hidden = 2880
    attn_hidden = 512
    topk = 4
    num_experts = 32
    mem = ttnn.DRAM_MEMORY_CONFIG

    assert mesh_shape == (1, 8), "router-hidden micro repro is fixed to the 1x8 mesh shape"

    num_devices = mesh_shape[0] * mesh_shape[1]
    local_experts = num_experts // num_devices
    assert local_experts == topk, "Expected 4 local experts on a 1x8 mesh"

    mesh_open_device_params = None
    fabric_state = None
    mesh_device = None

    initial_state_base = None
    local_expert_ids_tile = None
    primer_attn_in_weight = None
    primer_attn_out_weight = None
    primer_attn_out_bias = None
    primer_moe_norm_weight = None
    primer_moe_gate_weight = None
    primer_moe_gate_bias = None
    primer_moe_expert_outputs = None
    trigger_gate_weight = None
    trigger_gate_bias = None
    manual_router_hidden_trigger_input = None
    state_3d = None

    try:
        torch.manual_seed(0)

        mesh_open_device_params, fabric_state = _prepare_mesh_repro(mesh_shape, device_params)
        mesh_device = _open_repro_mesh_device(mesh_shape, mesh_open_device_params)
        replicated = ttnn.ReplicateTensorToMesh(mesh_device)

        hifi4_compute_config = _make_hifi4_compute_config(mesh_device)
        rms_norm_compute_config = _make_hifi4_compute_config(mesh_device, packer_l1_acc=True)

        def _run_full_moe_layer(router_hidden_3d, residual_3d, gate_weight, gate_bias, expert_outputs):
            router_hidden_f32 = None
            router_hidden_2d = None
            router_logits_f32 = None
            router_logits = None
            sorted_scores = None
            sorted_indices_u16 = None
            topk_scores = None
            topk_weights_tile = None
            topk_weights_rm = None
            topk_weights_3d_rm = None
            topk_weights_3d_tile = None
            topk_indices_i32 = None
            topk_indices = None
            topk_indices_3d = None
            local_expert_mask = None
            masked_weights_3d_tile = None
            masked_weights_3d_rm = None
            masked_weights_2d_rm = None
            masked_weights_2d_tile = None
            masked_weights_permuted = None
            masked_weights_4d = None
            weighted_expert_outputs = None
            combined_expert_outputs = None
            reduce_scatter_input = None
            scattered_4d = None
            scattered_3d = None
            gathered = None
            post_collective = None

            try:
                router_hidden_f32 = ttnn.typecast(router_hidden_3d, dtype=ttnn.float32)
                router_hidden_2d = ttnn.reshape(
                    router_hidden_f32,
                    ttnn.Shape((tokens, hidden)),
                    memory_config=mem,
                )
                router_logits_f32 = ttnn.linear(
                    router_hidden_2d,
                    gate_weight,
                    bias=gate_bias,
                    memory_config=mem,
                    compute_kernel_config=hifi4_compute_config,
                )
                router_logits = ttnn.typecast(router_logits_f32, dtype=ttnn.bfloat16)
                sorted_scores, sorted_indices_u16 = ttnn.sort(router_logits, dim=1, descending=True)
                topk_scores = ttnn.slice(
                    sorted_scores,
                    [0, 0],
                    [tokens, topk],
                    memory_config=mem,
                )
                topk_weights_tile = ttnn.softmax(
                    topk_scores,
                    dim=1,
                    numeric_stable=True,
                    memory_config=mem,
                    compute_kernel_config=hifi4_compute_config,
                )

                topk_weights_rm = ttnn.to_layout(
                    topk_weights_tile,
                    ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=mem,
                )
                topk_weights_3d_rm = ttnn.reshape(
                    topk_weights_rm,
                    ttnn.Shape((tokens, 1, topk)),
                    memory_config=mem,
                )
                topk_weights_3d_tile = ttnn.to_layout(
                    topk_weights_3d_rm,
                    ttnn.TILE_LAYOUT,
                    memory_config=mem,
                )

                topk_indices_i32 = ttnn.typecast(sorted_indices_u16, dtype=ttnn.int32)
                topk_indices = ttnn.slice(
                    topk_indices_i32,
                    [0, 0],
                    [tokens, topk],
                    memory_config=mem,
                )
                topk_indices_3d = ttnn.reshape(
                    topk_indices,
                    ttnn.Shape((tokens, topk, 1)),
                    memory_config=mem,
                )
                local_expert_mask = ttnn.eq(
                    topk_indices_3d,
                    local_expert_ids_tile,
                    dtype=ttnn.bfloat16,
                )
                masked_weights_3d_tile = ttnn.matmul(
                    topk_weights_3d_tile,
                    local_expert_mask,
                    memory_config=mem,
                    compute_kernel_config=hifi4_compute_config,
                )

                masked_weights_3d_rm = ttnn.to_layout(
                    masked_weights_3d_tile,
                    ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=mem,
                )
                masked_weights_2d_rm = ttnn.reshape(
                    masked_weights_3d_rm,
                    ttnn.Shape((tokens, topk)),
                    memory_config=mem,
                )
                masked_weights_2d_tile = ttnn.to_layout(
                    masked_weights_2d_rm,
                    ttnn.TILE_LAYOUT,
                    memory_config=mem,
                )
                masked_weights_permuted = ttnn.permute(masked_weights_2d_tile, (1, 0))
                masked_weights_4d = ttnn.reshape(
                    masked_weights_permuted,
                    ttnn.Shape((local_experts, batch, seq_len, 1)),
                    memory_config=mem,
                )

                weighted_expert_outputs = ttnn.multiply(expert_outputs, masked_weights_4d)
                combined_expert_outputs = ttnn.sum(weighted_expert_outputs, dim=0)
                reduce_scatter_input = ttnn.reshape(
                    combined_expert_outputs,
                    ttnn.Shape((1, batch, seq_len, hidden)),
                    memory_config=mem,
                )
                scattered_4d = ttnn.reduce_scatter(
                    reduce_scatter_input,
                    dim=1,
                    cluster_axis=1,
                    num_links=1,
                    topology=ttnn.Topology.Ring,
                    memory_config=mem,
                )
                scattered_3d = ttnn.reshape(
                    scattered_4d,
                    ttnn.Shape((local_experts, seq_len, hidden)),
                    memory_config=mem,
                )
                gathered = ttnn.all_gather(
                    scattered_3d,
                    dim=0,
                    cluster_axis=1,
                    num_links=1,
                    topology=ttnn.Topology.Ring,
                    memory_config=mem,
                )
                post_collective = ttnn.add(residual_3d, gathered)
                return post_collective
            finally:
                _deallocate_tensors(
                    router_hidden_f32,
                    router_hidden_2d,
                    router_logits_f32,
                    router_logits,
                    sorted_scores,
                    sorted_indices_u16,
                    topk_scores,
                    topk_weights_tile,
                    topk_weights_rm,
                    topk_weights_3d_rm,
                    topk_weights_3d_tile,
                    topk_indices_i32,
                    topk_indices,
                    topk_indices_3d,
                    local_expert_mask,
                    masked_weights_3d_tile,
                    masked_weights_3d_rm,
                    masked_weights_2d_rm,
                    masked_weights_2d_tile,
                    masked_weights_permuted,
                    masked_weights_4d,
                    weighted_expert_outputs,
                    combined_expert_outputs,
                    reduce_scatter_input,
                    scattered_4d,
                    scattered_3d,
                    gathered,
                )

        def _run_router_hidden_trigger_layer(router_hidden_3d, gate_weight, gate_bias, loop_index, layer_index):
            router_hidden_f32 = None
            router_hidden_2d = None
            router_logits_f32 = None
            router_logits = None
            sorted_scores = None
            sorted_indices_u16 = None
            topk_scores = None
            topk_indices_i32 = None
            topk_indices = None
            topk_indices_3d = None

            try:
                router_hidden_f32 = ttnn.typecast(router_hidden_3d, dtype=ttnn.float32)
                router_hidden_2d = ttnn.reshape(
                    router_hidden_f32,
                    ttnn.Shape((tokens, hidden)),
                    memory_config=mem,
                )
                router_logits_f32 = ttnn.linear(
                    router_hidden_2d,
                    gate_weight,
                    bias=gate_bias,
                    memory_config=mem,
                    compute_kernel_config=hifi4_compute_config,
                )
                router_logits = ttnn.typecast(router_logits_f32, dtype=ttnn.bfloat16)
                sorted_scores, sorted_indices_u16 = ttnn.sort(router_logits, dim=1, descending=True)
                topk_scores = ttnn.slice(
                    sorted_scores,
                    [0, 0],
                    [tokens, topk],
                    memory_config=mem,
                )
                _trace_repro_step(test_label, loop_index, num_loops, layer_index, num_layers, "before_sort_sync")
                ttnn.synchronize_device(mesh_device)
                _trace_repro_step(test_label, loop_index, num_loops, layer_index, num_layers, "after_sort_sync")

                topk_indices_i32 = ttnn.typecast(sorted_indices_u16, dtype=ttnn.int32)
                topk_indices = ttnn.slice(
                    topk_indices_i32,
                    [0, 0],
                    [tokens, topk],
                    memory_config=mem,
                )
                topk_indices_3d = ttnn.reshape(
                    topk_indices,
                    ttnn.Shape((tokens, topk, 1)),
                    memory_config=mem,
                )
                _trace_repro_step(
                    test_label,
                    loop_index,
                    num_loops,
                    layer_index,
                    num_layers,
                    "before_indices_reshape_cut_sync",
                )
                ttnn.synchronize_device(mesh_device)
                _trace_repro_step(
                    test_label,
                    loop_index,
                    num_loops,
                    layer_index,
                    num_layers,
                    "after_indices_reshape_cut_sync",
                )
            finally:
                _deallocate_tensors(
                    router_hidden_f32,
                    router_hidden_2d,
                    router_logits_f32,
                    router_logits,
                    sorted_scores,
                    sorted_indices_u16,
                    topk_scores,
                    topk_indices_i32,
                    topk_indices,
                    topk_indices_3d,
                )

        initial_state_base = ttnn.from_torch(
            torch.randn(batch, seq_len, hidden, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=replicated,
            memory_config=mem,
        )
        local_expert_ids_tile = _make_local_expert_ids_tile(mesh_device, num_experts, local_experts, mem)

        primer_attn_in_weight = _make_replicated_permuted_bfp8_weight(
            torch.randn(attn_hidden, hidden, dtype=torch.bfloat16),
            mesh_device,
            replicated,
            mem,
        )
        primer_attn_out_weight = _make_replicated_permuted_bfp8_weight(
            torch.randn(hidden, attn_hidden, dtype=torch.bfloat16),
            mesh_device,
            replicated,
            mem,
        )
        primer_attn_out_bias = _make_replicated_tiled_from_host(
            torch.randn(1, hidden, dtype=torch.bfloat16),
            mesh_device,
            replicated,
            mem,
        )
        primer_moe_norm_weight = _make_replicated_tiled_from_host(
            torch.ones(hidden, dtype=torch.bfloat16),
            mesh_device,
            replicated,
            mem,
        )
        primer_moe_gate_weight = _make_replicated_permuted_bfp8_weight(
            torch.randn(num_experts, hidden, dtype=torch.bfloat16),
            mesh_device,
            replicated,
            mem,
        )
        primer_moe_gate_bias = _make_replicated_tiled_from_host(
            torch.randn(num_experts, dtype=torch.bfloat16),
            mesh_device,
            replicated,
            mem,
            output_dtype=ttnn.float32,
        )
        primer_moe_expert_outputs = ttnn.from_torch(
            torch.randn(local_experts, batch, seq_len, hidden, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=replicated,
            memory_config=mem,
        )

        trigger_gate_weight = _make_replicated_permuted_bfp8_weight(
            torch.randn(num_experts, hidden, dtype=torch.bfloat16),
            mesh_device,
            replicated,
            mem,
        )
        trigger_gate_bias = _make_replicated_tiled_from_host(
            torch.randn(num_experts, dtype=torch.bfloat16),
            mesh_device,
            replicated,
            mem,
            output_dtype=ttnn.float32,
        )
        manual_router_hidden_trigger_input = ttnn.from_torch(
            torch.randn(batch, seq_len, hidden, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=replicated,
            memory_config=mem,
        )

        print(
            "all tensors pre-staged on device for again_hang router-hidden micro repro "
            f"(layers={num_layers}, loops={num_loops})"
        )

        for loop in range(num_loops):
            print(f"running {test_label} {loop + 1}/{num_loops}")

            state_3d = initial_state_base
            state_is_new = False

            state_2d = None
            attn_proj_2d = None
            attn_out_2d = None
            attn_ccl_input = None
            attn_ccl_gathered = None
            attn_ccl_summed = None
            attn_ccl_biased = None
            attn_post_3d = None
            post_attn_residual = None
            moe_input_3d = None
            next_state_3d = None

            try:
                _trace_repro_step(test_label, loop, num_loops, 0, num_layers, "enter_layer")
                state_2d = ttnn.reshape(
                    state_3d,
                    ttnn.Shape((tokens, hidden)),
                    memory_config=mem,
                )
                attn_proj_2d = ttnn.linear(
                    state_2d,
                    primer_attn_in_weight,
                    memory_config=mem,
                    compute_kernel_config=hifi4_compute_config,
                )
                attn_out_2d = ttnn.matmul(
                    attn_proj_2d,
                    primer_attn_out_weight,
                    memory_config=mem,
                    compute_kernel_config=hifi4_compute_config,
                )
                attn_ccl_input = ttnn.reshape(
                    attn_out_2d,
                    ttnn.Shape((1, tokens, hidden)),
                    memory_config=mem,
                )
                attn_ccl_gathered = ttnn.all_gather(
                    attn_ccl_input,
                    dim=0,
                    cluster_axis=1,
                    num_links=1,
                    topology=ttnn.Topology.Ring,
                    memory_config=mem,
                )
                attn_ccl_summed = ttnn.sum(attn_ccl_gathered, dim=0)
                attn_ccl_biased = ttnn.add(
                    attn_ccl_summed,
                    primer_attn_out_bias,
                )
                attn_post_3d = ttnn.reshape(
                    attn_ccl_biased,
                    ttnn.Shape((batch, seq_len, hidden)),
                    memory_config=mem,
                )
                post_attn_residual = ttnn.add(state_3d, attn_post_3d)
                moe_input_3d = ttnn.rms_norm(
                    post_attn_residual,
                    weight=primer_moe_norm_weight,
                    epsilon=1e-5,
                    memory_config=mem,
                    compute_kernel_config=rms_norm_compute_config,
                )
                _trace_repro_step(test_label, loop, num_loops, 0, num_layers, "before_moe")
                next_state_3d = _run_full_moe_layer(
                    moe_input_3d,
                    post_attn_residual,
                    primer_moe_gate_weight,
                    primer_moe_gate_bias,
                    primer_moe_expert_outputs,
                )
                _trace_repro_step(test_label, loop, num_loops, 0, num_layers, "after_moe_return")
            finally:
                _deallocate_tensors(
                    state_2d,
                    attn_proj_2d,
                    attn_out_2d,
                    attn_ccl_input,
                    attn_ccl_gathered,
                    attn_ccl_summed,
                    attn_ccl_biased,
                    attn_post_3d,
                    moe_input_3d,
                    post_attn_residual,
                )

            if state_is_new:
                ttnn.deallocate(state_3d)
            state_3d = next_state_3d
            state_is_new = True
            _trace_repro_step(test_label, loop, num_loops, 0, num_layers, "after_layer_cleanup")

            _trace_repro_step(test_label, loop, num_loops, 1, num_layers, "enter_layer")
            _trace_repro_step(test_label, loop, num_loops, 1, num_layers, "using_prestaged_router_hidden_trigger")
            _trace_repro_step(test_label, loop, num_loops, 1, num_layers, "before_moe")
            _run_router_hidden_trigger_layer(
                manual_router_hidden_trigger_input,
                trigger_gate_weight,
                trigger_gate_bias,
                loop,
                1,
            )
            _trace_repro_step(test_label, loop, num_loops, 1, num_layers, "after_requested_moe_stop_stage")

            if state_is_new:
                ttnn.deallocate(state_3d)
            state_3d = None

        print(f"{test_label}: {num_loops} loops completed")
    finally:
        _deallocate_tensors(
            initial_state_base,
            local_expert_ids_tile,
            primer_attn_in_weight,
            primer_attn_out_weight,
            primer_attn_out_bias,
            primer_moe_norm_weight,
            primer_moe_gate_weight,
            primer_moe_gate_bias,
            primer_moe_expert_outputs,
            trigger_gate_weight,
            trigger_gate_bias,
            manual_router_hidden_trigger_input,
            state_3d,
        )
        _close_repro_mesh_device(mesh_device)
        _reset_repro_fabric(fabric_state)


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_shape", [(1, 8)], ids=["1x8"])
def test_again_hang_real_ccl_one_pass_layer2_prestaged_router_hidden_indices_trigger_micro_repro_mesh(
    mesh_shape, device_params
):
    """
    Current smallest reproduced failing chain:
      layer 1 runs the full path as the primer,
      layer 2 skips the attention prefix and starts from pre-staged
      router_hidden_3d, then runs router linear -> sort -> indices slice ->
      candidate #2 and stops immediately.

    Hang meaning:
      If the last marker is before_indices_reshape_cut_sync in layer 2, the
      current minimal failing chain is primer -> pre-staged router_hidden_3d ->
      router path -> #2 reshape.
    """
    _run_again_hang_router_hidden_micro_repro_mesh(
        mesh_shape,
        device_params,
        "again_hang_real_ccl_one_pass_layer2_prestaged_router_hidden_indices_trigger_micro_mesh",
    )
