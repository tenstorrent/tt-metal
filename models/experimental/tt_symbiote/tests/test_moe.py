# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests MoE modules with TTNN acceleration (GLM-4.7-Flash)."""

import os

import pytest
import torch
import ttnn

from models.experimental.tt_symbiote.modules.moe import (
    Glm4MoeConfig,
    Glm4MoeMoE,
    TTNNMoE,
)
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.core.utils import compare_fn_outputs


# GLM-4.7-Flash MoE layer for real-weights tests.
REAL_WEIGHTS_MODEL_PATH = "zai-org/GLM-4.7-Flash"
REAL_WEIGHTS_LAYER_INDEX = 1

# Device mesh shape. Must be set in env so TTNNMoE run_on_devices can resolve architecture (e.g. T3K).
_MESH_DEVICE_ENV = "MESH_DEVICE"
if _MESH_DEVICE_ENV not in os.environ:
    os.environ[_MESH_DEVICE_ENV] = "T3K"
MESH_DEVICE = os.environ.get(_MESH_DEVICE_ENV, "T3K")


@pytest.fixture
def default_moe_config():
    """Default MoE configuration for testing."""
    return Glm4MoeConfig(
        hidden_size=2048,
        intermediate_size=10240,
        moe_intermediate_size=1536,
        num_local_experts=64,
        num_experts_per_tok=4,
        n_shared_experts=1,
        routed_scaling_factor=1.8,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
    )


@pytest.mark.parametrize(
    "real_weights",
    [
        True,  # Use real weights
        # False,  # Use random weights
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
            "BHGLX": (8, 4),
        }.get(MESH_DEVICE, (1, 8))
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 245760, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True
)
def test_glm4_moe_full(mesh_device, default_glm_config, real_weights):
    """Test full Glm4MoeMoE module with TTNN acceleration (GLM-4.7-Flash)."""
    if real_weights:
        from transformers import AutoModelForCausalLM

        full_model = AutoModelForCausalLM.from_pretrained(
            REAL_WEIGHTS_MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        model = full_model.model.layers[REAL_WEIGHTS_LAYER_INDEX].mlp
        hidden_size = full_model.config.hidden_size
    else:
        model = Glm4MoeMoE(default_glm_config).to(dtype=torch.bfloat16)
        hidden_size = default_glm_config.hidden_size
    model.eval()
    torch.set_grad_enabled(False)
    batch_size, seq_len = 1, 115
    inputs = torch.randn((batch_size, seq_len, hidden_size), dtype=torch.bfloat16)
    outputs_torch = model(inputs)
    ttnn_model = TTNNMoE.from_torch(model)
    set_device(ttnn_model, mesh_device)
    outputs_ttnn = ttnn_model(inputs)
    compare_fn_outputs(outputs_torch, outputs_ttnn, "Glm4MoeMoE")
