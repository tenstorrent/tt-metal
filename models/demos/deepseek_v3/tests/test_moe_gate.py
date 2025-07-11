import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate
from models.utility_functions import comp_pcc


@pytest.fixture
def hf_config():
    """Load DeepSeek config for testing."""
    path = os.getenv("HF_MODEL", "/proj_sw/user_dev/deepseek-ai")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return config


def test_moe_gate(hf_config, device, batch_size=32):
    torch.manual_seed(1000)
    reference_model = MoEGate(hf_config)
    torch_input = torch.randn(1, batch_size, 7168)
    # torch_input = torch.stack([torch.randperm(7168).float()*0.001 for _ in range(batch_size)], dim=0).unsqueeze(0)
    top8_experts_indices, top8_experts_weights = reference_model(torch_input)
    assert top8_experts_weights.shape == (batch_size, 8)

    state_dict = reference_model.state_dict()

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_linear_weights = ttnn.from_torch(
        state_dict["weight"].T.unsqueeze(0).unsqueeze(0),
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    tt_logits = ttnn.linear(
        tt_input, tt_linear_weights, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=compute_kernel_config
    )
    # tt_scores = ttnn.sigmoid(tt_logits, vector_mode=4, fast_and_approximate_mode=False)
    tt_scores = ttnn.softmax(tt_logits, dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_logits.deallocate()

    tt_bias_correction_weights = ttnn.from_torch(
        state_dict["e_score_correction_bias"].repeat(batch_size, 1).unsqueeze(0).unsqueeze(0),
        device=device,
        dtype=ttnn.float32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_scores_with_bias = ttnn.add(
        tt_scores, tt_bias_correction_weights, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
    )
    tt_scores_grouped = ttnn.reshape(tt_scores_with_bias, (1, batch_size, 8, 32))
    tt_scores_grouped_padded = ttnn.pad(tt_scores_grouped, [(0, 0), (0, 0), (0, 0), (0, 64 - 32)], value=-float("inf"))
    ttnn.deallocate(tt_scores_grouped)
    ttnn_top2_values, ttnn_top2_indices = ttnn.topk(tt_scores_grouped_padded, 2, dim=3, largest=True, sorted=True)
    ttnn.deallocate(tt_scores_grouped_padded)
    ttnn.deallocate(ttnn_top2_indices)
    ttnn_group_scores = ttnn.sum(ttnn_top2_values, dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(ttnn_top2_values)
    ttnn_group_scores = ttnn.reshape(ttnn_group_scores, (1, 1, batch_size, 8))
    ttnn_group_scores = ttnn.pad(ttnn_group_scores, [(0, 0), (0, 0), (0, 0), (0, 64 - 8)], value=-float("inf"))
    ttnn_group_top4_values, ttnn_group_top4_indices = ttnn.topk(
        ttnn_group_scores, k=4, dim=3, largest=True, sorted=True
    )
    ttnn.deallocate(ttnn_group_scores)
    ttnn.deallocate(ttnn_group_top4_values)

    torch_inf_mask = torch.full((1, 1, batch_size, 8), -float("inf"))
    torch_ones_tensor = torch.ones((1, 1, batch_size, 4))
    tt_group_mask = ttnn.from_torch(
        torch_inf_mask, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_ones_tensor = ttnn.from_torch(
        torch_ones_tensor, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    tt_group_mask = ttnn.experimental.scatter(tt_group_mask, 3, ttnn_group_top4_indices, tt_ones_tensor)
    ttnn.deallocate(tt_ones_tensor)
    ttnn.deallocate(ttnn_group_top4_indices)

    tt_group_mask = ttnn.reshape(tt_group_mask, (1, batch_size, 8, 1))
    tt_scores_mask = ttnn.repeat(tt_group_mask, ttnn.Shape((1, 1, 1, 32)))
    ttnn.deallocate(tt_group_mask)
    tt_scores_mask = ttnn.reshape(tt_scores_mask, (1, 1, batch_size, 256))
    tt_scores_with_bias = ttnn.mul(tt_scores_with_bias, tt_scores_mask, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(tt_scores_mask)
    topk_experts = reference_model.top_k
    tt_top8_temp_values, tt_top8_experts_indices = ttnn.topk(
        tt_scores_with_bias, k=topk_experts, dim=3, largest=True, sorted=True
    )
    ttnn.deallocate(tt_scores_with_bias)
    ttnn.deallocate(tt_top8_temp_values)
    tt_top8_experts_weights = ttnn.experimental.gather(tt_scores, dim=3, index=tt_top8_experts_indices)
    ttnn.deallocate(tt_scores)

    denominator = ttnn.sum(tt_top8_experts_weights, dim=3, memory_config=ttnn.L1_MEMORY_CONFIG, keepdim=True)
    tt_top8_experts_weights = ttnn.div(tt_top8_experts_weights, denominator)
    ttnn.deallocate(denominator)

    tt_norm_eps = ttnn.from_torch(
        torch.tensor([1e-20]).repeat(batch_size, topk_experts).unsqueeze(0).unsqueeze(0),
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_top8_experts_weights = ttnn.add(
        tt_top8_experts_weights, tt_norm_eps, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
    )
    ttnn.deallocate(tt_norm_eps)

    tt_expert_scale = ttnn.from_torch(
        torch.tensor([reference_model.routed_scaling_factor])
        .repeat(batch_size, topk_experts)
        .unsqueeze(0)
        .unsqueeze(0),
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_top8_experts_weights = ttnn.mul(
        tt_top8_experts_weights, tt_expert_scale, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
    )
    ttnn.deallocate(tt_expert_scale)

    # compare pcc of tt and torch top8_experts_weights
    tt_top8_experts_weights = ttnn.to_torch(tt_top8_experts_weights).squeeze(0).squeeze(0)
    passing, pcc_message = comp_pcc(tt_top8_experts_weights, top8_experts_weights, 0.99)
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"top8_experts_weights output does not meet PCC requirement 0.99: {pcc_message}"

    # compare pcc of tt and torch top8_experts_indices
    tt_top8_experts_indices = ttnn.to_torch(tt_top8_experts_indices).squeeze(0).squeeze(0)
    passing, pcc_message = comp_pcc(tt_top8_experts_indices, top8_experts_indices, 0.99)
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"top8_experts_indices output does not meet PCC requirement 0.99: {pcc_message}"
