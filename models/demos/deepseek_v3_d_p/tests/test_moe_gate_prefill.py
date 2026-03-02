# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import random
from dataclasses import dataclass

import pytest
import torch

import ttnn
from models.common.tensor_utils import get_padded_hidden_dim, pad_to_shape

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3_d_p.reference.deepseek.model import Gate as ReferenceMoEGate2
from models.demos.deepseek_v3_d_p.reference.deepseek.model import linear as referenceLinear
from models.demos.deepseek_v3_d_p.tt.moe_gate_prefill import MoEGatePrefill
from tests.ttnn.utils_for_testing import comp_pcc

random.seed(42)
torch.manual_seed(42)


@dataclass
class MoEGateConfig:
    # gate_params

    ccl_config = {}
    mm_configs = {}

    dim: int = 7168
    max_seq_len = 4096
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
    num_devices = 4

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

    ccl_config["TOPOLOGY"] = ttnn.Topology.Ring
    ccl_config["CLUSTER_AXIS"] = (1,)
    ccl_config["NUM_LINKS"] = (2,)


def get_or_create_1x4_mesh(device_params=None):
    fabric_cfg = (device_params or {}).get("fabric_config", ttnn.FabricConfig.DISABLED)
    cluster_type = ttnn.cluster.get_cluster_type()

    if cluster_type == ttnn.cluster.ClusterType.BLACKHOLE_GALAXY:
        # Set fabric config BEFORE opening the parent mesh
        if fabric_cfg != ttnn.FabricConfig.DISABLED:
            ttnn.set_fabric_config(fabric_cfg)

        # Open the full system mesh (e.g., 8x4) and carve a 1x4 column
        parent_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
        submesh = parent_mesh.create_submesh(
            mesh_shape=ttnn.MeshShape(1, 4), coordinate=ttnn.MeshCoordinate(0, 0)  # first column
        )
        return submesh, parent_mesh

    elif cluster_type == ttnn.cluster.ClusterType.P150_X4:
        # QuietBox: open 1x4 directly; still set fabric config if requested
        if fabric_cfg != ttnn.FabricConfig.DISABLED:
            ttnn.set_fabric_config(fabric_cfg)
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
        return mesh, None

    else:
        raise ValueError(f"Unsupported cluster type, expected P150_X4 or BLACKHOLE_GALAXY, but got {cluster_type}")


def calculate_average_recall(predicted_experts, reference_experts):
    recall = 0

    for i in range(predicted_experts.shape[0]):
        pred_row_set = set([e.item() for e in predicted_experts[i]])
        ref_row_set = set([e.item() for e in reference_experts[i]])
        recall += len(pred_row_set.intersection(ref_row_set)) / len(ref_row_set) if len(ref_row_set) > 0 else 0

    return recall / predicted_experts.shape[0]


def get_input_mem_config(config, padded_dim):
    shard_height = (config.max_seq_len + config.num_cores - 1) // config.num_cores
    shard_height = ((shard_height + 31) // 32) * 32
    shard_width = (padded_dim + config.num_devices - 1) // config.num_devices

    sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(shard_height, shard_width),
        core_grid=config.core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return sharded_mem_config


@pytest.mark.parametrize("seq_len", [4096])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
def test_forward_pass(
    seq_len,
    device_params,
):
    submesh, parent_mesh = get_or_create_1x4_mesh(device_params)

    config = MoEGateConfig()
    reference_model = ReferenceMoEGate2(config)
    tt_model = MoEGatePrefill(config, seq_len, submesh)

    torch_input = torch.randn(seq_len, config.dim, dtype=torch.bfloat16)

    torch_weight = torch.randn(reference_model.weight.data.shape, dtype=torch.bfloat16).T
    torch_bias = torch.randn(reference_model.bias.data.shape, dtype=torch.bfloat16)
    reference_model.weight.data = torch_weight
    reference_model.bias.data = torch_bias

    # Reference forward pass
    reference_model.eval()
    reference_model.to(torch.bfloat16)
    reference_topk_weights, reference_topk_indices = reference_model(torch_input)
    reference_logits = referenceLinear(torch_input, reference_model.weight)

    # Compute tile-aligned padded dim for the sharded axis
    num_devices = submesh.get_num_devices()
    padded_dim = get_padded_hidden_dim(config.dim, num_devices, tile_size=32)
    torch_input_padded = pad_to_shape(torch_input, (seq_len, padded_dim), pad_value=0.0)
    torch_weight_padded = pad_to_shape(torch_weight, (padded_dim, torch_weight.shape[1]), pad_value=0.0)

    sharded_mem_config = get_input_mem_config(config, padded_dim)

    tt_model.weight = ttnn.from_torch(
        torch_weight_padded,
        device=submesh,
        dtype=ttnn.bfloat4_b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=0),
    )
    tt_model.bias = ttnn.from_torch(
        torch_bias.repeat(seq_len).view(seq_len, -1),
        device=submesh,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_input = ttnn.from_torch(
        torch_input_padded,
        device=submesh,
        dtype=ttnn.bfloat16,
        memory_config=sharded_mem_config,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=-1),
    )

    tt_topk_weights, tt_topk_indices, tt_logits = tt_model(tt_input)
    per_device_topk_weight = ttnn.get_device_tensors(tt_topk_weights)
    per_device_topk_indices = ttnn.get_device_tensors(tt_topk_indices)
    per_device_topk_logits = ttnn.get_device_tensors(tt_logits)

    for d in range(num_devices):
        # Convert output back to torch
        tt_topk_weights_torch = ttnn.to_torch(per_device_topk_weight[d])
        tt_topk_indices_torch = ttnn.to_torch(per_device_topk_indices[d])
        tt_logits_torch = ttnn.to_torch(per_device_topk_logits[d])

        logits_passed, logits_pcc = comp_pcc(tt_logits_torch, reference_logits, 0.99)
        weights_passed, weights_pcc = comp_pcc(tt_topk_weights_torch, reference_topk_weights, 0.53)
        recall = calculate_average_recall(tt_topk_indices_torch, reference_topk_indices)

        assert recall > 0.91, f"Recall is {recall}, expected recal > 0.91"
        assert logits_passed, f"Logits PCC is {logits_pcc}, expected PCC > 0.99"
        assert weights_passed, f"Weights PCC is {weights_pcc}, expected PCC > 0.53"

    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_topk_weights)
    ttnn.deallocate(tt_topk_indices)
    ttnn.deallocate(tt_logits)

    # Proper teardown
    if parent_mesh:
        ttnn.close_mesh_device(submesh)  # close submesh first
        ttnn.close_mesh_device(parent_mesh)
    else:
        ttnn.close_mesh_device(submesh)


if __name__ == "__main__":
    pytest.main([__file__])
