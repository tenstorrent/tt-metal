# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests MoE modules with TTNN acceleration."""

import pytest
import torch

from models.experimental.tt_symbiote.modules.moe import (
    Glm4MoeConfig,
    Glm4MoeMoE,
    TTNNGlm4MoeMoE,
)
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.core.utils import compare_fn_outputs
import ttnn


@pytest.fixture
def default_moe_config():
    """Default MoE configuration for testing."""
    return Glm4MoeConfig(
        hidden_size=4096,
        intermediate_size=10944,
        moe_intermediate_size=1408,
        num_local_experts=128,
        num_experts_per_tok=4,
        n_shared_experts=1,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
    )


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 245760, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True
)
def test_glm4_moe_full(mesh_device, default_moe_config):
    """Test full Glm4MoeMoE module with TTNN acceleration."""
    model = Glm4MoeMoE(default_moe_config).to(dtype=torch.bfloat16)
    model.eval()
    torch.set_grad_enabled(False)

    batch_size, seq_len = 2, 4
    inputs = torch.randn((batch_size, seq_len, default_moe_config.hidden_size), dtype=torch.bfloat16)
    outputs_torch = model(inputs)
    ttnn_model = TTNNGlm4MoeMoE.from_torch(model)
    set_device(ttnn_model, mesh_device)
    outputs_ttnn = ttnn_model(inputs)
    compare_fn_outputs(outputs_torch, outputs_ttnn, "Glm4MoeMoE")
