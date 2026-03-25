# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import random
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate as ReferenceMoEGate
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_expert_dispatch_table, create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.moe_gate_prefill2d import MoEGateConfig, MoEGatePrefill
from models.demos.deepseek_v3_d_p.utils.test_utils import (
    adjust_shapes_for_testing,
    calculate_average_recall,
    get_input_mem_config,
)
from tests.ttnn.utils_for_testing import comp_pcc


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

    config = MoEGateConfig()
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
    tt_model = MoEGatePrefill(config, mesh_device)

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

    dispatch_table = create_expert_dispatch_table(
        num_routed_experts=n_routed_experts,
        dispatch_group_size=n_sp_devices,
        num_dispatch_groups=n_tp_devices,
    )
    if n_tp_devices == 1:
        tt_model.expert_dispatch_table = ttnn.from_torch(
            dispatch_table[0],
            device=mesh_device,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
    else:
        tt_model.expert_dispatch_table = ttnn.from_torch(
            dispatch_table,
            device=mesh_device,
            dtype=ttnn.int32,
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

    all_passed = True
    assert_msgs = []
    for device_id in range(mesh_device.shape[0] * mesh_device.shape[1]):
        # Convert output back to torch
        tt_topk_weights_torch = ttnn.to_torch(per_device_topk_weight[device_id])
        tt_topk_indices_torch = ttnn.to_torch(per_device_topk_indices[device_id])
        tt_logits_torch = ttnn.to_torch(per_device_topk_logits[device_id])

        # Calculate device position in mesh
        row = device_id // mesh_device.shape[1]
        col = device_id % mesh_device.shape[1]

        # Get the corresponding reference slice (same across columns due to reduction)
        ref_logits = reference_logits_reshaped[row, :, :]
        ref_weights = reference_topk_weights_reshaped[row, :, :]
        ref_indices = reference_topk_indices_reshaped[row, :, :]

        # Test recall with individual logging
        recall = calculate_average_recall(tt_topk_indices_torch, ref_indices)
        recall_passed = recall > 0.999
        status_char = "✅" if recall_passed else "❌"
        logger.info(
            f"{status_char} Device {device_id} (row={row}, col={col}): Recall = {recall:.4f} (threshold: 0.999)"
        )
        if not recall_passed:
            all_passed = False
            assert_msgs.append(f"Device {device_id} (row={row}, col={col}): Recall is {recall:.4f}, expected > 0.999")

        # Test logits PCC with individual logging
        logits_passed, logits_pcc = comp_pcc(tt_logits_torch, ref_logits, 0.99)
        status_char = "✅" if logits_passed else "❌"
        logger.info(
            f"{status_char} Device {device_id} (row={row}, col={col}): Logits PCC = {logits_pcc:.4f} (threshold: 0.99)"
        )
        if not logits_passed:
            all_passed = False
            assert_msgs.append(
                f"Device {device_id} (row={row}, col={col}): Logits PCC is {logits_pcc:.4f}, expected > 0.99"
            )

        # Test weights PCC with individual logging
        weights_passed, weights_pcc = comp_pcc(tt_topk_weights_torch, ref_weights, 0.98)
        status_char = "✅" if weights_passed else "❌"
        logger.info(
            f"{status_char} Device {device_id} (row={row}, col={col}): Weights PCC = {weights_pcc:.4f} (threshold: 0.98)"
        )
        if not weights_passed:
            all_passed = False
            assert_msgs.append(
                f"Device {device_id} (row={row}, col={col}): Weights PCC is {weights_pcc:.4f}, expected > 0.99"
            )

    # Compute reference dispatch offsets from tt_topk_indices, masked by dispatch table
    indices_for_gate = torch.zeros(n_sp_devices, seq_len_per_device, config.n_activated_experts, dtype=torch.int32)
    for row in range(n_sp_devices):
        device_id = row * n_tp_devices
        indices_for_gate[row] = ttnn.to_torch(per_device_topk_indices[device_id]).int()

    # Unmasked expert counter: same for all dispatch groups (all groups see the same tokens)
    expert_counter = torch.zeros((n_sp_devices, n_routed_experts), dtype=torch.int32)
    for chip in range(n_sp_devices):
        for token in range(seq_len_per_device):
            for k in range(config.n_activated_experts):
                expert_id = indices_for_gate[chip, token, k].item()
                expert_counter[chip, expert_id] += 1

    # Per-dispatch-group masked offsets and totals
    per_group_offsets = {}
    per_group_totals = {}
    for group in range(n_tp_devices):
        group_mask = dispatch_table[group] >= 0
        masked_counter = expert_counter.clone()
        masked_counter[:, ~group_mask] = 0
        cum_sum = torch.cumsum(masked_counter, dim=0)
        offsets = torch.vstack([torch.zeros([1, n_routed_experts], dtype=torch.int32), cum_sum[:-1]])
        per_group_offsets[group] = offsets.long()
        per_group_totals[group] = cum_sum[-1:].long()

    per_device_dispatch_offsets = ttnn.get_device_tensors(dispatch_offsets)
    for device_id in range(len(per_device_dispatch_offsets)):
        tt_offsets_torch = ttnn.to_torch(per_device_dispatch_offsets[device_id]).long()
        row = device_id // n_tp_devices
        col = device_id % n_tp_devices
        reference_offsets = per_group_offsets[col]

        offsets_match = torch.equal(tt_offsets_torch, reference_offsets)
        status_char = "✅" if offsets_match else "❌"
        if offsets_match:
            logger.info(f"{status_char} Device {device_id} (row={row}, col={col}): Dispatch offsets match exactly")
        else:
            diff = (tt_offsets_torch - reference_offsets).abs()
            max_diff = diff.max().item()
            num_mismatches = (diff > 0).sum().item()
            total_elements = diff.numel()
            logger.info(
                f"{status_char} Device {device_id} (row={row}, col={col}): "
                f"Dispatch offsets MISMATCH - max_diff={max_diff}, "
                f"mismatches={num_mismatches}/{total_elements}"
            )
            all_passed = False
            assert_msgs.append(
                f"Device {device_id} (row={row}, col={col}): "
                f"Dispatch offsets mismatch (max_diff={max_diff}, mismatches={num_mismatches}/{total_elements})"
            )

    per_device_totals = ttnn.get_device_tensors(total_counts_per_expert)
    for device_id in range(len(per_device_totals)):
        tt_totals_torch = ttnn.to_torch(per_device_totals[device_id]).long()
        row = device_id // n_tp_devices
        col = device_id % n_tp_devices
        reference_totals = per_group_totals[col]

        totals_match = torch.equal(tt_totals_torch, reference_totals)
        status_char = "✅" if totals_match else "❌"
        if totals_match:
            logger.info(f"{status_char} Device {device_id} (row={row}, col={col}): Total counts match exactly")
        else:
            diff = (tt_totals_torch - reference_totals).abs()
            max_diff = diff.max().item()
            num_mismatches = (diff > 0).sum().item()
            total_elements = diff.numel()
            logger.info(
                f"{status_char} Device {device_id} (row={row}, col={col}): "
                f"Total counts MISMATCH - max_diff={max_diff}, "
                f"mismatches={num_mismatches}/{total_elements}"
            )
            all_passed = False
            assert_msgs.append(
                f"Device {device_id} (row={row}, col={col}): "
                f"Total counts mismatch (max_diff={max_diff}, mismatches={num_mismatches}/{total_elements})"
            )

    assert all_passed, "\n".join(assert_msgs)

    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_topk_weights)
    ttnn.deallocate(tt_topk_indices)
    ttnn.deallocate(tt_logits)
    ttnn.deallocate(dispatch_offsets)
    ttnn.deallocate(total_counts_per_expert)
