# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
    create_gate_weights,
    get_ep_mesh_composer,
    get_gate_outputs,
    get_sp_mesh_composer,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode, TtMoEGateConfig, TtMoEGatePrefill
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
    "gate_fallback_mode",
    [
        GateComputeMode.HOST_ALL,
    ],
)
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
    gate_fallback_mode,
):
    random.seed(42)
    torch.manual_seed(42)

    # Create reference gate
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
    reference_model = ReferenceMoEGate(ref_config, use_bitonic_sort=True)

    # Create dummy weights and bias
    gate_w = create_gate_weights(config.n_routed_experts, config.dim)
    reference_model.weight.data = gate_w["weight"]  # (n_routed_experts, dim) — HF convention
    reference_model.e_score_correction_bias.data = gate_w["e_score_correction_bias"]

    # Create input tensor
    n_sp_devices = mesh_device.shape[0]
    torch_input = torch.randn(config.sp_dim * n_sp_devices, config.dim, dtype=torch.bfloat16) + 1

    # Reference forward pass
    reference_model.eval()
    reference_model.to(torch.bfloat16)
    reference_topk_indices, reference_topk_scores = reference_model.grouped_forward(torch_input.unsqueeze(0))
    reference_logits = torch_input @ gate_w["weight"].T

    # Create TT input tensor
    sharded_mem_config = get_input_mem_config(config, mesh_device.shape)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=sharded_mem_config,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(0, -1),  # tensor parallel
            mesh_shape=mesh_device.shape,
        ),
    )

    # Create TT gate
    n_sp_devices = mesh_device.shape[0]
    n_tp_devices = mesh_device.shape[1]
    n_routed_experts = config.n_routed_experts

    dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=n_routed_experts,
        dispatch_group_size=n_sp_devices,
        num_dispatch_groups=n_tp_devices,
    )
    tt_model = TtMoEGatePrefill(
        config,
        mesh_device,
        dispatch_table=dispatch_table,
        weight=gate_w["weight"],
        bias=gate_w["e_score_correction_bias"],
        fallback_mode=gate_fallback_mode,
    )
    # TT forward pass
    tt_topk_scores, tt_topk_indices, tt_logits, tt_dispatch_offsets, tt_total_counts_per_expert = tt_model(tt_input)

    # Validation
    seq_len_per_device = reference_logits.shape[0] // mesh_device.shape[0]
    sp_composer = get_sp_mesh_composer(mesh_device)

    # SP-replicated checks: compose into [1, n_sp_devices, ...] for validate_composed
    host_tt_topk_indices = ttnn.to_torch(tt_topk_indices, mesh_composer=sp_composer)
    host_tt_topk_indices = host_tt_topk_indices.view(1, n_sp_devices, seq_len_per_device, -1).sort(dim=-1).values
    reference_topk_indices = reference_topk_indices.view(1, n_sp_devices, seq_len_per_device, -1).sort(dim=-1).values

    recall_topk_indices = validate_composed(
        host_tt_topk_indices,
        reference_topk_indices,
        1,
        n_sp_devices,
        compare_recall(0.997),
        name="recall_topk_indices",
        broadcast_groups=n_tp_devices,
    )

    host_tt_logits = ttnn.to_torch(tt_logits, mesh_composer=sp_composer)
    host_tt_logits = host_tt_logits.view(1, n_sp_devices, seq_len_per_device, -1)
    reference_logits = reference_logits.view(1, n_sp_devices, seq_len_per_device, -1)

    pcc_logits = validate_composed(
        host_tt_logits,
        reference_logits,
        1,
        n_sp_devices,
        compare_pcc(0.997),
        name="pcc_logits",
        broadcast_groups=n_tp_devices,
    )

    host_tt_topk_scores = ttnn.to_torch(tt_topk_scores, mesh_composer=sp_composer)
    host_tt_topk_scores = host_tt_topk_scores.view(1, n_sp_devices, seq_len_per_device, -1)
    reference_topk_scores = reference_topk_scores.view(1, n_sp_devices, seq_len_per_device, -1)

    pcc_scores = validate_composed(
        host_tt_topk_scores,
        reference_topk_scores,
        1,
        n_sp_devices,
        compare_pcc(0.99),
        name="pcc_scores",
        broadcast_groups=n_tp_devices,
    )

    # EP-sharded checks: offsets and totals
    experts_per_chip = n_routed_experts // (n_sp_devices * n_tp_devices)
    ref_dispatch_offsets, ref_expert_token_counts, _ = get_gate_outputs(
        indices=reference_topk_indices.view(n_sp_devices, seq_len_per_device, -1).int(),
        dispatch_group_size=n_sp_devices,
        num_routed_experts=n_routed_experts,
        experts_per_chip=experts_per_chip,
        seq_len_per_chip=seq_len_per_device,
        num_experts_per_tok=config.n_activated_experts,
        expert_dispatch_table=dispatch_table,
    )

    ep_composer = get_ep_mesh_composer(mesh_device)

    tt_dispatch_offsets = ttnn.unsqueeze_to_4D(tt_dispatch_offsets)
    host_dispatch_offsets = ttnn.to_torch(tt_dispatch_offsets, mesh_composer=ep_composer).squeeze(2).long()

    exact_dispatch_offsets = validate_composed(
        host_dispatch_offsets,
        ref_dispatch_offsets.long(),
        n_tp_devices,
        n_sp_devices,
        compare_exact,
        name="dispatch_offsets",
    )

    tt_total_counts_per_expert = ttnn.unsqueeze_to_4D(tt_total_counts_per_expert)
    host_tt_total_counts_per_expert = (
        ttnn.to_torch(tt_total_counts_per_expert, mesh_composer=ep_composer).squeeze(2).long()
    )
    # Totals replicated across chips — broadcast row 0 reference to all chips
    ref_expert_token_counts = ref_expert_token_counts[:, 0, :].unsqueeze(1).expand(-1, n_sp_devices, -1).long()

    expert_token_counts = validate_composed(
        host_tt_total_counts_per_expert,
        ref_expert_token_counts,
        n_tp_devices,
        n_sp_devices,
        compare_exact,
        name="expert_token_counts",
    )

    all_results = [recall_topk_indices, pcc_logits, pcc_scores, exact_dispatch_offsets, expert_token_counts]
    for res in all_results:
        res.log_mismatches()
    log_validation_results(
        results=all_results,
        num_dispatch_groups=n_tp_devices,
        dispatch_group_size=n_sp_devices,
        title="Gate Prefill2D Validation",
    )
    merged = ValidationResult.merge(all_results, name="gate_prefill2d")
    merged.assert_passed("Gate prefill2d validation failed")
