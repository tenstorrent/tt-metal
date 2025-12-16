"""Test for LLaMA model with TTNN backend."""

import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

from models.tt_symbiote.modules.activation import TTNNSilu
from models.tt_symbiote.modules.linear import TTNNLinearLLama
from models.tt_symbiote.utils.device_management import set_device
from models.tt_symbiote.utils.module_replacement import register_module_replacement_dict


def test_llama(device):
    """Test LLaMA model with TTNN acceleration."""
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    nn_to_ttnn = {
        nn.Linear: TTNNLinearLLama,
        nn.SiLU: TTNNSilu,
    }
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM(config).to(dtype=torch.bfloat16)
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    register_module_replacement_dict(model, nn_to_ttnn, model_config=None)
    set_device(model, device)
    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead
    outputs = model.generate(inputs, max_new_tokens=100, use_cache=True)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
