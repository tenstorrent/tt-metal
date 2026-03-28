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
    get_ep_mesh_composer,
    get_gate_outputs,
    get_sp_mesh_composer,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import TtMoEGateConfig, TtMoEGatePrefill
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    ValidationResult,
    compare_exact,
    compare_pcc,
    compare_recall,
    validate_composed,
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

    seq_len_per_device = reference_logits.shape[0] // mesh_device.shape[0]

    tt_topk_weights, tt_topk_indices, tt_logits, dispatch_offsets, total_counts_per_expert = tt_model(tt_input)

    sp_composer = get_sp_mesh_composer(mesh_device)

    # SP-replicated checks: compose into [1, n_sp_devices, ...] for validate_composed
    composed_indices = ttnn.to_torch(tt_topk_indices, mesh_composer=sp_composer)
    composed_indices_2d = composed_indices.view(1, n_sp_devices, seq_len_per_device, -1)
    ref_indices_2d = reference_topk_indices.view(1, n_sp_devices, seq_len_per_device, -1)

    recall_result = validate_composed(
        composed_indices_2d,
        ref_indices_2d,
        1,
        n_sp_devices,
        compare_recall(0.999),
        name="indices_recall",
        broadcast_groups=n_tp_devices,
    )

    composed_logits = ttnn.to_torch(tt_logits, mesh_composer=sp_composer)
    composed_logits_2d = composed_logits.view(1, n_sp_devices, seq_len_per_device, -1)
    ref_logits_2d = reference_logits.view(1, n_sp_devices, seq_len_per_device, -1)

    logits_result = validate_composed(
        composed_logits_2d,
        ref_logits_2d,
        1,
        n_sp_devices,
        compare_pcc(0.99),
        name="topk_logits",
        broadcast_groups=n_tp_devices,
    )

    composed_weights = ttnn.to_torch(tt_topk_weights, mesh_composer=sp_composer)
    composed_weights_2d = composed_weights.view(1, n_sp_devices, seq_len_per_device, -1)
    ref_weights_2d = reference_topk_weights.view(1, n_sp_devices, seq_len_per_device, -1)

    weights_result = validate_composed(
        composed_weights_2d,
        ref_weights_2d,
        1,
        n_sp_devices,
        compare_pcc(0.98),
        name="topk_scores",
        broadcast_groups=n_tp_devices,
    )

    # EP-sharded checks: offsets and totals
    indices_for_gate = composed_indices.view(n_sp_devices, seq_len_per_device, -1).int()

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

    ep_composer = get_ep_mesh_composer(mesh_device)

    dispatch_offsets_4d = ttnn.unsqueeze_to_4D(dispatch_offsets)
    composed_offsets = ttnn.to_torch(dispatch_offsets_4d, mesh_composer=ep_composer).squeeze(2).long()

    offsets_result = validate_composed(
        composed_offsets,
        expert_offsets.long(),
        n_tp_devices,
        n_sp_devices,
        compare_exact,
        name="dispatch_offsets",
    )

    total_counts_4d = ttnn.unsqueeze_to_4D(total_counts_per_expert)
    composed_totals = ttnn.to_torch(total_counts_4d, mesh_composer=ep_composer).squeeze(2).long()
    # Totals replicated across chips — broadcast row 0 reference to all chips
    reference_totals = expert_token_counts[:, 0, :].unsqueeze(1).expand(-1, n_sp_devices, -1).long()

    totals_result = validate_composed(
        composed_totals,
        reference_totals,
        n_tp_devices,
        n_sp_devices,
        compare_exact,
        name="total_counts",
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
