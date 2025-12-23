"""Test for GLM4.5 Air model with TTNN backend."""

import os

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.tt_symbiote.modules.activation import TTNNSilu
from models.tt_symbiote.modules.linear import TTNNLinearLLama
from models.tt_symbiote.utils.device_management import set_device
from models.tt_symbiote.utils.module_replacement import register_module_replacement_dict


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
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
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
def test_glm(mesh_device):
    """Test GLM model with TTNN acceleration."""
    nn_to_ttnn = {
        nn.Linear: TTNNLinearLLama,
        nn.SiLU: TTNNSilu,
    }

    tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-4.5-Air")
    model = AutoModelForCausalLM.from_pretrained("zai-org/GLM-4.5-Air")
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    set_device(model, mesh_device)
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    outputs = model.generate(**inputs, max_new_tokens=40, use_cache=True)
    print(f"GLM OUTPUT: {tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:])}")
