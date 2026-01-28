# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for LLaMA model with TTNN backend."""

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.activation import TTNNSilu
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.normalization import TTNNLayerNorm, TTNNRMSNorm
from models.experimental.tt_symbiote.modules.attention import LlamaAttention
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict


class LlamaMLP(nn.Module):
    def __init__(self, old_layer):
        super().__init__()
        self.config = old_layer.config
        self.hidden_size = old_layer.hidden_size
        self.intermediate_size = old_layer.intermediate_size
        self.gate_proj = old_layer.gate_proj
        self.up_proj = old_layer.up_proj
        self.down_proj = old_layer.down_proj
        assert old_layer.config.hidden_act == "silu", "Only SiLU activation is supported in this test."
        self.act_fn = nn.SiLU()

    @classmethod
    def from_torch(cls, old_layer):
        return cls(old_layer)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def test_llama(device):
    """Test LLaMA model with TTNN acceleration."""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").to(dtype=torch.bfloat16)
    nn_to_nn = {
        model.model.layers[0].mlp.__class__: LlamaMLP,
    }
    nn_to_ttnn = {
        nn.Linear: TTNNLinear,
        nn.SiLU: TTNNSilu,
        nn.LayerNorm: TTNNLayerNorm,
        model.model.layers[0].input_layernorm.__class__: TTNNRMSNorm,
        model.model.layers[0].self_attn.__class__: LlamaAttention,
    }
    modules = register_module_replacement_dict(model, nn_to_nn, model_config=None)
    messages = [
        {
            "role": "user",
            "content": "What is your favorite condiment? There are so many condiments to choose from, each bringing its unique flavor and texture to enhance different dishes. Do you prefer the classic taste of ketchup, the creamy richness of mayonnaise, the spicy kick of mustard, or perhaps something more exotic like sriracha or hoisin sauce? Maybe you enjoy the tangy zest of salsa or the smooth and savory taste of aioli. Share what your favorite condiment is and why you love it. Does it remind you of a specific dish or meal?",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    set_device(model, device)
    for k, v in tqdm(modules.items()):
        v.preprocess_weights()
        v.move_weights_to_device()
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    DispatchManager.clear_timings()
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    DispatchManager.save_stats_to_file("llama_timing_stats.csv")
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))
