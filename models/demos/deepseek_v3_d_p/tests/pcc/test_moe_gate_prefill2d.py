# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import random
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate as ReferenceMoEGate
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    create_fabric_router_config,
    get_gate_outputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import TtMoEGateConfig, TtMoEGatePrefill
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    ValidationResult,
    validate_per_device_exact,
    validate_per_device_pcc,
    validate_per_device_recall,
)
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_validation_results
from models.demos.deepseek_v3_d_p.utils.test_utils import adjust_shapes_for_testing, get_input_mem_config


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (2, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
            id="mesh-2x2",
        ),
        pytest.param(
            (4, 2),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
            id="mesh-4x2",
        ),
        pytest.param(
            (2, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="mesh-2x4",
        ),
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_forward_pass(
    mesh_device,
    num_links,
    topology,
):
    random.seed(42)
    torch.manual_seed(42)

    config = TtMoEGateConfig()
    config.ccl_config["NUM_LINKS"] = num_links
    adjust_shapes_for_testing(config, mesh_device)

    ref_config = SimpleNamespace(
        num_experts_per_tok=config.n_activated_experts,
        n_routed_experts=config.n_routed_experts,
        routed_scaling_factor=config.route_scale,
        scoring_func=config.score_func,
        topk_method="noaux_tc",
        n_group=config.n_expert_groups,
        topk_group=config.n_limited_groups,
        norm_topk_prob=True,
        hidden_size=config.dim,
    )
    reference_model = ReferenceMoEGate(ref_config, use_bitonic_sort=False)
    tt_model = TtMoEGatePrefill(config, mesh_device)

    torch_input = torch.randn(config.max_seq_len, config.dim, dtype=torch.bfloat16)

    torch_weight = torch.randn(reference_model.weight.data.shape, dtype=torch.bfloat16).T
    torch_bias = torch.randn(reference_model.e_score_correction_bias.data.shape, dtype=torch.bfloat16) * 1000
    reference_model.weight.data = torch_weight.T
    reference_model.e_score_correction_bias.data = torch_bias

    # Reference forward pass
    reference_model.eval()
    reference_model.to(torch.bfloat16)
    reference_topk_indices, reference_topk_weights = reference_model(torch_input.unsqueeze(0))
    reference_logits = torch_input @ torch_weight

    sharded_mem_config = get_input_mem_config(config, mesh_device.shape)

    tt_model.weight = ttnn.from_torch(
        torch_weight,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0),
            mesh_shape=mesh_device.shape,
        ),
    )

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=sharded_mem_config,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(0, 1),
            mesh_shape=mesh_device.shape,
        ),
    )

    tt_model.bias = ttnn.from_torch(
        # ttnn.experimental.deepseek_grouped_gate() requires bias to be broadcasted already
        torch_bias.repeat(config.sp_dim).view(config.sp_dim, -1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    n_sp_devices = mesh_device.shape[0]
    n_tp_devices = mesh_device.shape[1]
    n_routed_experts = config.n_routed_experts

    dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=n_routed_experts,
        dispatch_group_size=n_sp_devices,
        num_dispatch_groups=n_tp_devices,
    )
    tt_model.routing_setup.experts_in_dispatch_group = ttnn.from_torch(
        dispatch_table,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0),
            mesh_shape=mesh_device.shape,
        ),
    )

    # Reshape reference outputs to match device mesh structure upfront
    seq_len_per_device = reference_logits.shape[0] // mesh_device.shape[0]
    reference_logits_reshaped = reference_logits.view(mesh_device.shape[0], seq_len_per_device, -1)
    reference_topk_weights_reshaped = reference_topk_weights.view(mesh_device.shape[0], seq_len_per_device, -1)
    reference_topk_indices_reshaped = reference_topk_indices.view(mesh_device.shape[0], seq_len_per_device, -1)

    tt_topk_weights, tt_topk_indices, tt_logits, dispatch_offsets, total_counts_per_expert = tt_model(tt_input)
    per_device_topk_weight = ttnn.get_device_tensors(tt_topk_weights)
    per_device_topk_indices = ttnn.get_device_tensors(tt_topk_indices)
    per_device_topk_logits = ttnn.get_device_tensors(tt_logits)

    num_devices = mesh_device.shape[0] * mesh_device.shape[1]

    recall_result = validate_per_device_recall(
        get_actual=lambda i: ttnn.to_torch(per_device_topk_indices[i]),
        get_expected=lambda i: reference_topk_indices_reshaped[i // n_tp_devices],
        num_devices=num_devices,
        threshold=0.999,
        name="indices_recall",
        mesh_shape=mesh_device.shape,
    )
    logits_result = validate_per_device_pcc(
        get_actual=lambda i: ttnn.to_torch(per_device_topk_logits[i]),
        get_expected=lambda i: reference_logits_reshaped[i // n_tp_devices],
        num_devices=num_devices,
        threshold=0.99,
        name="topk_logits",
        mesh_shape=mesh_device.shape,
    )
    weights_result = validate_per_device_pcc(
        get_actual=lambda i: ttnn.to_torch(per_device_topk_weight[i]),
        get_expected=lambda i: reference_topk_weights_reshaped[i // n_tp_devices],
        num_devices=num_devices,
        threshold=0.98,
        name="topk_scores",
        mesh_shape=mesh_device.shape,
    )

    # Compute reference dispatch offsets from tt_topk_indices, masked by dispatch table
    indices_for_gate = torch.zeros(n_sp_devices, seq_len_per_device, config.n_activated_experts, dtype=torch.int32)
    for row in range(n_sp_devices):
        device_id = row * n_tp_devices
        indices_for_gate[row] = ttnn.to_torch(per_device_topk_indices[device_id]).int()

    experts_per_chip = n_routed_experts // (n_sp_devices * n_tp_devices)
    expert_offsets, expert_token_counts, _ = get_gate_outputs(
        indices=indices_for_gate,
        dispatch_group_size=n_sp_devices,
        num_routed_experts=n_routed_experts,
        experts_per_chip=experts_per_chip,
        seq_len_per_chip=seq_len_per_device,
        num_experts_per_tok=config.n_activated_experts,
        expert_dispatch_table=dispatch_table,
    )

    reference_totals = expert_token_counts[:, 0, :].long()

    per_device_dispatch_offsets = ttnn.get_device_tensors(dispatch_offsets)
    offsets_result = validate_per_device_exact(
        get_actual=lambda i: ttnn.to_torch(per_device_dispatch_offsets[i]).long(),
        get_expected=lambda i: expert_offsets[
            i % n_tp_devices, (i // n_tp_devices) : (i // n_tp_devices) + 1, :
        ].long(),
        num_devices=len(per_device_dispatch_offsets),
        name="dispatch_offsets",
        mesh_shape=mesh_device.shape,
    )

    per_device_totals = ttnn.get_device_tensors(total_counts_per_expert)
    totals_result = validate_per_device_exact(
        get_actual=lambda i: ttnn.to_torch(per_device_totals[i]).long(),
        get_expected=lambda i: reference_totals[(i % n_tp_devices) : (i % n_tp_devices) + 1, :].long(),
        num_devices=len(per_device_totals),
        name="total_counts",
        mesh_shape=mesh_device.shape,
    )

    all_results = [recall_result, logits_result, weights_result, offsets_result, totals_result]
    log_validation_results(
        results=all_results,
        num_dispatch_groups=n_tp_devices,
        dispatch_group_size=n_sp_devices,
        title="Gate Prefill2D Validation",
    )
    merged = ValidationResult.merge(all_results, name="gate_prefill2d")
    merged.assert_passed("Gate prefill2d validation failed")

    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_topk_weights)
    ttnn.deallocate(tt_topk_indices)
    ttnn.deallocate(tt_logits)
    ttnn.deallocate(dispatch_offsets)
    ttnn.deallocate(total_counts_per_expert)
