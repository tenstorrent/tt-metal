# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.common.tensor_utils import get_padded_hidden_dim, pad_to_shape

# Import from local reference files instead of HuggingFace
# from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate as ReferenceMoEGate
from models.demos.deepseek_v3.reference.deepseek.model import Gate as ReferenceMoEGate2
from models.demos.deepseek_v3.reference.deepseek.model import linear as referenceLinear
from models.demos.deepseek_v3.tt.moe_gate_prefill import MoEGatePrefill

# from models.demos.deepseek_v3.utils.run_config import create_run_config
# from models.demos.deepseek_v3.utils.test_utils import get_model_config, get_test_weight_config, run_module_forward
from tests.ttnn.utils_for_testing import comp_pcc

# pytestmark = pytest.mark.use_module_device


@dataclass
class MoEGateConfig:
    # gate_params
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

    # ccl config
    topology = ttnn.Topology.Ring

    # grid_config
    um_cores = 110


def calculate_average_recall(predicted_experts, reference_experts):
    recall = 0

    for i in range(predicted_experts.shape[0]):
        pred_row_set = set([e.item() for e in predicted_experts[i]])
        ref_row_set = set([e.item() for e in reference_experts[i]])
        recall += len(pred_row_set.intersection(ref_row_set)) / len(ref_row_set) if len(ref_row_set) > 0 else 0

    return recall / predicted_experts.shape[0]


@pytest.mark.parametrize("seq_len", [4096])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)  # 4 devices (tp=4)
def test_forward_pass(
    seq_len,
    device_params,
    mesh_device,
    # device
):
    """Test forward pass against reference model."""

    # if ref_gate_version == 1:
    #     reference_model = ReferenceMoEGate(hf_config, True)
    # else:
    config = MoEGateConfig()
    reference_model = ReferenceMoEGate2(config)
    tt_model = MoEGatePrefill(config, mesh_device)

    # Create input tensor
    torch_input = torch.randn(seq_len, config.dim, dtype=torch.bfloat16)
    # generate weight and bias:
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
    num_devices = mesh_device.get_num_devices()
    padded_dim = get_padded_hidden_dim(config.dim, num_devices, tile_size=32)

    torch_input_padded = pad_to_shape(torch_input, (seq_len, padded_dim), pad_value=0.0)
    torch_weight_padded = pad_to_shape(torch_weight, (padded_dim, torch_weight.shape[1]), pad_value=0.0)

    # 1. Core grid on each device (example: full grid)
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 9))})
    num_cores = config.num_cores

    shard_height = (seq_len + num_cores - 1) // num_cores  # round up
    shard_height = ((shard_height + 31) // 32) * 32  # round up to multiple of 32
    shard_width = (padded_dim + num_devices - 1) // num_devices

    # 3. Create height-sharded memory config
    sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(shard_height, shard_width),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    tt_model.weight = ttnn.from_torch(
        torch_weight_padded,
        device=mesh_device,
        dtype=ttnn.bfloat4_b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    tt_model.bias = ttnn.from_torch(
        torch_bias.repeat(config.max_seq_len).view(config.max_seq_len, -1),
        device=mesh_device,
        # mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Convert input to TTNN
    tt_input = ttnn.from_torch(
        torch_input_padded,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=sharded_mem_config,  # core-level height sharding
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )

    # TTNN forward pass using utility function
    tt_topk_weights, tt_topk_indices, tt_logits = tt_model(tt_input)
    per_device_topk_weight = ttnn.get_device_tensors(tt_topk_weights)
    per_device_topk_indices = ttnn.get_device_tensors(tt_topk_indices)
    per_device_topk_logits = ttnn.get_device_tensors(tt_logits)

    for d in range(num_devices):
        # Convert output back to torch
        tt_topk_weights_torch = ttnn.to_torch(per_device_topk_weight[d])
        tt_topk_indices_torch = ttnn.to_torch(per_device_topk_indices[d])
        tt_logits_torch = ttnn.to_torch(per_device_topk_logits[d])

        comp_pcc(tt_logits_torch, reference_logits, 0.9)
        comp_pcc(tt_topk_weights_torch, reference_topk_weights, 0.85)
        recall = calculate_average_recall(tt_topk_indices_torch, reference_topk_indices)
        assert (recall > 0.99, f"Recall is {recall}, expected recal > 0.9")

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_topk_weights)
    ttnn.deallocate(tt_topk_indices)
    ttnn.deallocate(tt_logits)

    # Compare outputs
    logger.info(f"Seq len: {seq_len}")


if __name__ == "__main__":
    pytest.main([__file__])
