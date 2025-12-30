"""Test for GLM4.5 Air model with TTNN backend."""

import os

import pytest
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import ttnn
from models.tt_symbiote.modules.activation import TTNNSilu
from models.tt_symbiote.modules.linear import TTNNLinear, TTNNLinearLLama
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
        {"role": "user", "content": "Who are you? What is your name? What is your data based on? What can you do?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    exclude_list = set(
        [
            "lm_head",
            "model.layers.12.self_attn.o_proj",
            "model.layers.13.self_attn.o_proj",
            "model.layers.16.self_attn.o_proj",
            "model.layers.0.self_attn.q_proj",
            "model.layers.20.self_attn.o_proj",
            "model.layers.10.self_attn.o_proj",
            "model.layers.15.self_attn.o_proj",
            "model.layers.22.self_attn.o_proj",
            "model.layers.14.self_attn.o_proj",
            "model.layers.17.self_attn.o_proj",
            "model.layers.29.self_attn.o_proj",
            "model.layers.9.self_attn.o_proj",
            "model.layers.11.self_attn.o_proj",
            "model.layers.19.self_attn.o_proj",
            "model.layers.28.self_attn.o_proj",
            "model.layers.23.self_attn.o_proj",
            "model.layers.24.self_attn.o_proj",
            "model.layers.16.self_attn.q_proj",
            "model.layers.31.self_attn.o_proj",
            "model.layers.31.self_attn.q_proj",
            "model.layers.41.self_attn.o_proj",
            "model.layers.15.self_attn.q_proj",
            "model.layers.13.self_attn.q_proj",
            "model.layers.33.self_attn.o_proj",
            "model.layers.38.self_attn.o_proj",
            "model.layers.32.self_attn.o_proj",
            "model.layers.44.self_attn.o_proj",
            "model.layers.40.self_attn.o_proj",
            "model.layers.12.self_attn.q_proj",
            "model.layers.39.self_attn.o_proj",
            "model.layers.45.self_attn.o_proj",
            "model.layers.36.self_attn.o_proj",
            "model.layers.45.self_attn.q_proj",
            "model.layers.30.self_attn.o_proj",
            "model.layers.37.self_attn.o_proj",
            "model.layers.35.self_attn.o_proj",
            "model.layers.30.self_attn.q_proj",
            "model.layers.27.self_attn.o_proj",
            "model.layers.43.self_attn.q_proj",
            "model.layers.37.self_attn.q_proj",
            "model.layers.18.self_attn.o_proj",
            "model.layers.11.self_attn.q_proj",
            "model.layers.21.self_attn.o_proj",
            "model.layers.18.self_attn.q_proj",
            "model.layers.25.self_attn.o_proj",
            "model.layers.38.self_attn.q_proj",
            "model.layers.10.self_attn.q_proj",
            "model.layers.9.self_attn.q_proj",
            "model.layers.29.self_attn.q_proj",
            "model.layers.14.self_attn.q_proj",
            "model.layers.17.self_attn.q_proj",
            "model.layers.43.self_attn.o_proj",
            "model.layers.41.self_attn.q_proj",
            "model.layers.42.self_attn.q_proj",
            "model.layers.42.self_attn.o_proj",
            "model.layers.36.self_attn.q_proj",
            "model.layers.39.self_attn.q_proj",
            "model.layers.8.self_attn.o_proj",
            "model.layers.40.self_attn.q_proj",
            "model.layers.19.self_attn.q_proj",
            "model.layers.32.self_attn.q_proj",
            "model.layers.26.self_attn.o_proj",
            "model.layers.44.self_attn.q_proj",
            "model.layers.34.self_attn.q_proj",
            "model.layers.1.self_attn.q_proj",
            "model.layers.34.self_attn.o_proj",
            "model.layers.33.self_attn.q_proj",
            "model.layers.6.self_attn.o_proj",
            "model.layers.23.self_attn.q_proj",
            "model.layers.24.self_attn.q_proj",
            "model.layers.20.self_attn.q_proj",
            "model.layers.35.self_attn.q_proj",
            "model.layers.22.self_attn.q_proj",
            "model.layers.3.self_attn.o_proj",
            "model.layers.5.self_attn.o_proj",
            "model.layers.21.self_attn.q_proj",
            "model.layers.28.self_attn.q_proj",
            "model.layers.1.self_attn.o_proj",
            "model.layers.26.self_attn.q_proj",
            "model.layers.3.self_attn.q_proj",
            "model.layers.25.self_attn.q_proj",
            "model.layers.7.self_attn.o_proj",
            "model.layers.4.self_attn.o_proj",
            "model.layers.2.self_attn.q_proj",
            "model.layers.4.self_attn.q_proj",
            "model.layers.6.self_attn.q_proj",
            "model.layers.8.self_attn.q_proj",
            "model.layers.27.self_attn.q_proj",
            "model.layers.2.self_attn.o_proj",
            "model.layers.7.self_attn.q_proj",
            "model.layers.5.self_attn.q_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.down_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.1.mlp.shared_experts.gate_proj",
            "model.layers.1.mlp.shared_experts.up_proj",
            "model.layers.34.mlp.shared_experts.down_proj",
            "model.layers.36.mlp.shared_experts.gate_proj",
            "model.layers.22.mlp.shared_experts.gate_proj",
            "model.layers.38.mlp.shared_experts.down_proj",
            "model.layers.39.mlp.shared_experts.gate_proj",
            "model.layers.31.mlp.shared_experts.gate_proj",
            "model.layers.19.mlp.shared_experts.gate_proj",
            "model.layers.8.mlp.shared_experts.down_proj",
            "model.layers.17.mlp.shared_experts.gate_proj",
            "model.layers.32.mlp.shared_experts.gate_proj",
            "model.layers.20.mlp.shared_experts.gate_proj",
            "model.layers.5.mlp.shared_experts.down_proj",
            "model.layers.10.mlp.shared_experts.gate_proj",
            "model.layers.3.mlp.shared_experts.gate_proj",
            "model.layers.3.mlp.shared_experts.down_proj",
            "model.layers.24.mlp.shared_experts.gate_proj",
            "model.layers.45.mlp.shared_experts.down_proj",
            "model.layers.37.mlp.shared_experts.down_proj",
            "model.layers.41.mlp.shared_experts.down_proj",
            "model.layers.31.mlp.shared_experts.down_proj",
            "model.layers.42.mlp.shared_experts.down_proj",
            "model.layers.20.mlp.shared_experts.up_proj",
            "model.layers.4.mlp.shared_experts.down_proj",
            "model.layers.38.mlp.shared_experts.gate_proj",
            "model.layers.23.mlp.shared_experts.up_proj",
        ]
    )
    modules1 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None, exclude_replacement=exclude_list)
    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
    }
    modules2 = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    set_device(model, mesh_device)
    all_modules = {**modules1, **modules2}
    for k, v in tqdm(all_modules.items()):
        v.preprocess_weights()
        if k in exclude_list:
            v.move_weights_to_device()
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    outputs = model.generate(**inputs, max_new_tokens=100, use_cache=True)
    print(f"GLM OUTPUT: {tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:])}")
