# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for GPT-OSS model with TTNN backend."""

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from models.experimental.tt_symbiote.modules.activation import TTNNSilu
from models.experimental.tt_symbiote.modules.linear import TTNNLinearLLamaBFloat16
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict
from models.experimental.tt_symbiote.core.run_config import DispatchManager


def test_gptoss(device):
    """Test GPT-OSS model with TTNN acceleration."""
    nn_to_ttnn = {
        nn.Linear: TTNNLinearLLamaBFloat16,
        nn.SiLU: TTNNSilu,
    }
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")
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
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    DispatchManager.clear_timings()
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    DispatchManager.save_stats_to_file("gptoss_timing_stats.csv")
    print(f"GPTOSS OUTPUT: {tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:])}")
