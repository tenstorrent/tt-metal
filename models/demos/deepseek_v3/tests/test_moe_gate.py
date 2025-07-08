import os

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate


@pytest.fixture
def hf_config():
    """Load DeepSeek config for testing."""
    path = os.getenv("HF_MODEL", "/proj_sw/user_dev/deepseek-ai")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return config


def test_moe_gate(hf_config, device, batch_size=32):
    torch.manual_seed(0)
    reference_model = MoEGate(hf_config)
    torch_input = torch.randn(1, batch_size, 7168)
    moe_gate = reference_model(torch_input)
    assert moe_gate[0].shape == (batch_size, 8)

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

    tt_logits = ttnn.linear(tt_input, tt_linear_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_scores = ttnn.sigmoid(tt_logits)
    tt_logits.deallocate()

    tt_bias_correction_weights = ttnn.from_torch(
        state_dict["e_score_correction_bias"].repeat(batch_size, 1).unsqueeze(0).unsqueeze(0),
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_scores_with_bias = ttnn.add(tt_scores, tt_bias_correction_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(tt_scores)
    tt_scores_grouped = ttnn.reshape(tt_scores_with_bias, (1, batch_size, 8, 32))
    tt_scores_grouped_padded = ttnn.pad(tt_scores_grouped, [(0, 0), (0, 0), (0, 0), (0, 64 - 32)], value=-float("inf"))
    ttnn.deallocate(tt_scores_grouped)
    ttnn_top2_values, ttnn_top2_indices = ttnn.topk(tt_scores_grouped_padded, 2, dim=3, largest=True, sorted=False)
    ttnn.deallocate(tt_scores_grouped_padded)
    ttnn.deallocate(ttnn_top2_indices)
    ttnn_group_scores = ttnn.sum(ttnn_top2_values, dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(ttnn_top2_values)
    ttnn_group_scores = ttnn.reshape(ttnn_group_scores, (1, 1, batch_size, 8))
    ttnn_group_scores = ttnn.pad(ttnn_group_scores, [(0, 0), (0, 0), (0, 0), (0, 64 - 8)], value=-float("inf"))
    ttnn_group_top4_values, ttnn_group_top4_indices = ttnn.topk(
        ttnn_group_scores, k=4, dim=3, largest=True, sorted=False
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

    tt_top8_expert_values, tt_top8_expert_indices = ttnn.topk(
        tt_scores_with_bias, k=8, dim=3, largest=True, sorted=False
    )
    ttnn.deallocate(tt_scores_with_bias)
