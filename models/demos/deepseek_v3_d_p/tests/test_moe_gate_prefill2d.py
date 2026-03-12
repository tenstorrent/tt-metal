# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import random
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3_d_p.reference.deepseek.model import Gate as ReferenceMoEGate
from models.demos.deepseek_v3_d_p.reference.deepseek.model import linear as referenceLinear
from models.demos.deepseek_v3_d_p.tt.moe_gate_prefill2d import MoEGatePrefill
from tests.ttnn.utils_for_testing import comp_pcc

random.seed(42)
torch.manual_seed(42)


@dataclass
class MoEGateConfig:
    # gate_params

    ccl_config = {}
    mm_configs = {}

    dim: int = 7168
    max_seq_len = 4096 * 32
    sp_dim = 4096
    n_routed_experts: int = 256
    n_shared_experts: int = 2
    n_activated_experts: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    route_scale: float = 1.0
    score_func: str = "sigmoid"
    summed_experts_per_group: int = 2
    topk_groups: int = 4

    # grid_config
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 9))})
    num_cores = 110

    mm_configs["DEFAULT_PROGRAM_CONFIG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
        in0_block_w=56,
        out_subblock_h=2,
        out_subblock_w=4,
        out_block_h=2,
        out_block_w=4,
        per_core_M=2,
        per_core_N=8,
        fuse_batch=True,
        mcast_in0=False,
    )
    mm_configs["DEFAULT_COMPUTE_CONFIG"] = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    ccl_config["DISPATCH_AXIS"] = 0
    ccl_config["TP_AXIS"] = 1
    ccl_config["NUM_LINKS"] = 2


def get_or_create_2d_mesh(device_params=None):
    fabric_cfg = (device_params or {}).get("fabric_config", ttnn.FabricConfig.DISABLED)
    cluster_type = ttnn.cluster.get_cluster_type()

    if cluster_type == ttnn.cluster.ClusterType.BLACKHOLE_GALAXY:
        if fabric_cfg != ttnn.FabricConfig.DISABLED:
            ttnn.set_fabric_config(fabric_cfg)
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
        return mesh

    elif cluster_type == ttnn.cluster.ClusterType.P150_X8:
        if fabric_cfg != ttnn.FabricConfig.DISABLED:
            ttnn.set_fabric_config(fabric_cfg)
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 2))
        return mesh

    elif cluster_type == ttnn.cluster.ClusterType.P150_X4:
        if fabric_cfg != ttnn.FabricConfig.DISABLED:
            ttnn.set_fabric_config(fabric_cfg)
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 2))
        return mesh

    else:
        raise ValueError(
            f"Unsupported cluster type, expected P150_X4, P150_X8 or BLACKHOLE_GALAXY, but got {cluster_type}"
        )


def calculate_average_recall(predicted_experts, reference_experts):
    recall = 0

    for i in range(predicted_experts.shape[0]):
        pred_row_set = set([e.item() for e in predicted_experts[i]])
        ref_row_set = set([e.item() for e in reference_experts[i]])
        recall += len(pred_row_set.intersection(ref_row_set)) / len(ref_row_set) if len(ref_row_set) > 0 else 0

    return recall / predicted_experts.shape[0]


def get_input_mem_config(config, mesh_shape):
    shard_height = (config.sp_dim + config.num_cores - 1) // config.num_cores
    shard_height = ((shard_height + 31) // 32) * 32
    shard_width = (config.dim + mesh_shape[1] - 1) // mesh_shape[1]
    sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(shard_height, shard_width),
        core_grid=config.core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return sharded_mem_config


def adjust_shapes_for_testing(config, mesh_device):
    """
    Adjust the input dimensions just in case the test is tun on a smaller grid to preserve per-device shapes
    """
    n_sp_devices, n_tp_devices = mesh_device.shape
    if n_sp_devices != 32:
        config.max_seq_len = config.max_seq_len // (32 // n_sp_devices)
    if n_tp_devices != 4:
        config.dim = config.dim // (4 // n_tp_devices)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_forward_pass(
    device_params,
):
    mesh_device = get_or_create_2d_mesh(device_params)
    config = MoEGateConfig()
    adjust_shapes_for_testing(config, mesh_device)

    reference_model = ReferenceMoEGate(config)
    tt_model = MoEGatePrefill(config, mesh_device)

    torch_input = torch.randn(config.max_seq_len, config.dim, dtype=torch.bfloat16)

    torch_weight = torch.randn(reference_model.weight.data.shape, dtype=torch.bfloat16).T
    torch_bias = torch.randn(reference_model.bias.data.shape, dtype=torch.bfloat16)
    reference_model.weight.data = torch_weight
    reference_model.bias.data = torch_bias

    # Reference forward pass
    reference_model.eval()
    reference_model.to(torch.bfloat16)
    reference_topk_weights, reference_topk_indices = reference_model(torch_input)
    reference_logits = referenceLinear(torch_input, reference_model.weight)

    sharded_mem_config = get_input_mem_config(config, mesh_device.shape)

    tt_model.weight = ttnn.from_torch(
        torch_weight,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
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
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Reshape reference outputs to match device mesh structure upfront
    seq_len_per_device = reference_logits.shape[0] // mesh_device.shape[0]
    reference_logits_reshaped = reference_logits.view(mesh_device.shape[0], seq_len_per_device, -1)
    reference_topk_weights_reshaped = reference_topk_weights.view(mesh_device.shape[0], seq_len_per_device, -1)
    reference_topk_indices_reshaped = reference_topk_indices.view(mesh_device.shape[0], seq_len_per_device, -1)

    tt_topk_weights, tt_topk_indices, tt_logits, dispatch_offsets = tt_model(tt_input)
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
        recall_passed = recall > 0.95
        status_char = "✅" if recall_passed else "❌"
        logger.info(f"{status_char} Device {device_id} (row={row}, col={col}): Recall = {recall:.4f} (threshold: 0.95)")
        if not recall_passed:
            all_passed = False
            assert_msgs.append(f"Device {device_id} (row={row}, col={col}): Recall is {recall:.4f}, expected > 0.95")

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
        weights_passed, weights_pcc = comp_pcc(tt_topk_weights_torch, ref_weights, 0.53)
        status_char = "✅" if weights_passed else "❌"
        logger.info(
            f"{status_char} Device {device_id} (row={row}, col={col}): Weights PCC = {weights_pcc:.4f} (threshold: 0.53)"
        )
        if not weights_passed:
            all_passed = False
            assert_msgs.append(
                f"Device {device_id} (row={row}, col={col}): Weights PCC is {weights_pcc:.4f}, expected > 0.53"
            )

    # Compute reference dispatch offsets from tt_topk_indices (isolates bincount + cumsum logic)
    n_sp_devices = mesh_device.shape[0]
    n_tp_devices = mesh_device.shape[1]
    n_routed_experts = config.n_routed_experts

    ref_histograms = torch.zeros(n_sp_devices, n_routed_experts, dtype=torch.long)
    for row in range(n_sp_devices):
        device_id = row * n_tp_devices  # pick first column device for this row
        tt_indices_torch = ttnn.to_torch(per_device_topk_indices[device_id]).flatten().long()
        ref_histograms[row] = torch.bincount(tt_indices_torch, minlength=n_routed_experts)

    ref_cumsum = torch.cumsum(ref_histograms, dim=0)
    reference_offsets = torch.zeros(n_sp_devices + 1, n_routed_experts, dtype=torch.long)
    reference_offsets[1:] = ref_cumsum

    per_device_dispatch_offsets = ttnn.get_device_tensors(dispatch_offsets)
    for device_id in range(len(per_device_dispatch_offsets)):
        tt_offsets_torch = ttnn.to_torch(per_device_dispatch_offsets[device_id]).long()
        row = device_id // n_tp_devices
        col = device_id % n_tp_devices

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

    assert all_passed, "\n".join(assert_msgs)

    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_topk_weights)
    ttnn.deallocate(tt_topk_indices)
    ttnn.deallocate(tt_logits)
    ttnn.deallocate(dispatch_offsets)

    # Proper teardown
    if mesh_device:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    pytest.main([__file__])
