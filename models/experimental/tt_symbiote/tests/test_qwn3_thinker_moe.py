# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Tests MoE modules with TTNN acceleration."""

import os

import pytest
import torch
import ttnn

from transformers import Qwen3OmniMoeForConditionalGeneration

from models.experimental.tt_symbiote.modules.moe import (
    Glm4MoeMoE,
    Glm4MoeNaiveMoe,
    TTNNMoE,
)
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.core.utils import compare_fn_outputs


# Model path and layer index for real-weights tests.
# Use Qwen3-Omni-MoE thinker blocks with TTNNMoE/TTNNQwen3MoE.
REAL_WEIGHTS_MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
REAL_WEIGHTS_LAYER_INDEX = 1

# Device mesh shape. Must be set in env so TTNNMoE run_on_devices can resolve architecture (e.g. T3K).
_MESH_DEVICE_ENV = "MESH_DEVICE"
if _MESH_DEVICE_ENV not in os.environ:
    os.environ[_MESH_DEVICE_ENV] = "T3K"
MESH_DEVICE = os.environ.get(_MESH_DEVICE_ENV, "T3K")


def test_glm4_moe_naive_matches_qwen3_omni_thinker_experts():
    """Unit test: Glm4MoeNaiveMoe should match Qwen3OmniMoeThinkerTextSparseMoeBlock experts."""
    batch_size, seq_len = 2, 16

    # Load a real Qwen3-Omni-MoE model and grab a thinker MoE block with real config.
    torch.manual_seed(0)
    full_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        REAL_WEIGHTS_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    full_model.eval()

    # Take the first thinker text MoE block: thinker.model.layers[0].mlp
    hf_block = full_model.thinker.model.layers[0].mlp
    hf_block.eval()

    # Infer MoE dimensions from expert weights
    num_experts, two_intermediate, hidden_size = hf_block.experts.gate_up_proj.shape
    moe_intermediate_size = two_intermediate // 2

    # Symbiote Glm4MoeNaiveMoe experts-only block with aligned shapes
    class DummyGlmConfig:
        def __init__(self):
            self.num_local_experts = num_experts
            self.hidden_size = hidden_size
            self.moe_intermediate_size = moe_intermediate_size
            # Use the same initializer range as the pretrained config
            self.initializer_range = full_model.config.thinker_config.initializer_range

    glm_config = DummyGlmConfig()
    torch.manual_seed(0)
    glm_moe = Glm4MoeNaiveMoe(glm_config).to(dtype=torch.bfloat16).eval()

    # Copy expert weights from HF block into Glm4MoeNaiveMoe so they compute the same function
    with torch.no_grad():
        glm_moe.gate_up_proj.copy_(hf_block.experts.gate_up_proj)
        glm_moe.down_proj.copy_(hf_block.experts.down_proj)

    # Random input
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    b, s, h = x.shape
    x_flat = x.view(-1, h)

    with torch.no_grad():
        # Use HF router to get routing indices/weights
        _, routing_weights, selected_experts = hf_block.gate(x_flat)

        # HF experts output
        hf_out = hf_block.experts(x_flat, selected_experts, routing_weights).view(b, s, h)

        # Glm4MoeNaiveMoe output with same routing
        glm_out = glm_moe(x_flat, selected_experts, routing_weights).view(b, s, h)

    compare_fn_outputs(
        hf_out,
        glm_out,
        "Glm4MoeNaiveMoe_vs_Qwen3OmniMoeThinkerTextSparseMoeBlock",
    )

    # Explicit PCC check for clarity in this unit test
    hf_flat = hf_out.flatten().to(torch.float32)
    glm_flat = glm_out.flatten().to(torch.float32)
    pcc = torch.corrcoef(torch.stack([hf_flat, glm_flat]))[0, 1]
    print(f"PCC Glm4MoeNaiveMoe_vs_Qwen3OmniMoeThinkerTextSparseMoeBlock: {pcc.item()}")
    assert not torch.isnan(pcc), "PCC is NaN between HF and Glm4MoeNaiveMoe outputs"
    assert pcc > 0.999, f"PCC too low between HF and Glm4MoeNaiveMoe outputs: {pcc.item()}"


@pytest.mark.parametrize(
    "real_weights",
    [
        True,  # Use real weights
        False,  # Use random weights
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
    """Test full Glm4MoeMoE module with TTNN acceleration."""
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
    # Use generic TTNNMoE adapter for all supported MoE modules
    ttnn_model = TTNNMoE.from_torch(model)
    set_device(ttnn_model, mesh_device)
    outputs_ttnn = ttnn_model(inputs)
    compare_fn_outputs(outputs_torch, outputs_ttnn, "Glm4MoeMoE")
