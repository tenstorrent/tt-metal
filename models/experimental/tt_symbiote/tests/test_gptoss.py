# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for GPT-OSS model with TTNN backend."""

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.experimental.tt_symbiote.modules.activation import TTNNSilu
from models.experimental.tt_symbiote.modules.linear import TTNNLinearLLamaBFloat16
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict


def test_gptoss(device):
    """Test GPT-OSS model with TTNN acceleration."""
    nn_to_ttnn = {
        nn.Linear: TTNNLinearLLamaBFloat16,
        nn.SiLU: TTNNSilu,
    }
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b")
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
    set_device(model, device)
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    outputs = model.generate(**inputs, max_new_tokens=40, use_cache=True)
    print(f"GPTOSS OUTPUT: {tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:])}")
