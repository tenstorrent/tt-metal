# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests MoE modules with TTNN acceleration."""

import pytest
import torch

from models.experimental.tt_symbiote.modules.moe import (
    Glm4MoeMoE,
    TTNNMoE,
)
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.core.utils import compare_fn_outputs
import ttnn


@pytest.mark.parametrize(
    "real_weights",
    [
        True,  # Use real weights
        False,  # Use random weights
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 245760, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True
)
def test_glm4_moe_full(mesh_device, default_glm_config, real_weights):
    """Test full Glm4MoeMoE module with TTNN acceleration."""
    if real_weights:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-4.7-Flash").model.layers[1].mlp
    else:
        model = Glm4MoeMoE(default_glm_config).to(dtype=torch.bfloat16)
    model.eval()
    torch.set_grad_enabled(False)

    batch_size, seq_len = 1, 115
    inputs = torch.randn((batch_size, seq_len, default_glm_config.hidden_size), dtype=torch.bfloat16)
    outputs_torch = model(inputs)
    ttnn_model = TTNNMoE.from_torch(model)
    set_device(ttnn_model, mesh_device)
    outputs_ttnn = ttnn_model(inputs)
    compare_fn_outputs(outputs_torch, outputs_ttnn, "Glm4MoeMoE")
