from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import numpy as np
from torch import nn
from libs import tt_lib as ttm
from loguru import logger

from transformers import T5Model
from utility_functions import print_diff_argmax
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from python_api_testing.models.t5.t5_utils import torch2tt_tensor, tt2torch_tensor, read_model_config, print_corr_coef
from python_api_testing.models.t5.t5_attention import TtT5Attention
from python_api_testing.models.t5.t5_layer_norm import TtT5LayerNorm



class TtT5LayerSelfAttention(nn.Module):
    def __init__(self, config, state_dict, base_address, device, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = TtT5Attention(config, state_dict, f"{base_address}.SelfAttention", device, has_relative_attention_bias)
        self.layer_norm = TtT5LayerNorm(config, state_dict, f"{base_address}.layer_norm", device)
        # self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        #hidden_states = hidden_states + self.dropout(attention_output[0])
        hidden_states = ttm.tensor.add(hidden_states, attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


def test_T5LayerSelfAttention_inference(device):
    hf_reference_model = T5Model.from_pretrained("t5-small")
    hf_reference_model.eval()

    model_json_config = "tests/python_api_testing/models/t5/t5-small.json"
    config = read_model_config(model_json_config)
    block = 2
    has_relative_attention_bias = block == 0

    if config["is_decoder"]:
        hf_reference_module = hf_reference_model.decoder.block[block].layer[0]
        base_address = f"decoder.block.{block}.layer.0"
    else:
        hf_reference_module = hf_reference_model.encoder.block[block].layer[0]
        base_address = f"encoder.block.{block}.layer.0"

    # Prepare input
    torch.manual_seed(0)
    test_input = (torch.rand(32, 128, 512) * 2) - 1

    # PyTorch output
    pt_out = hf_reference_module(test_input)[0].unsqueeze(0)

    test_input = test_input.unsqueeze(0)

    # T5-small config file: https://huggingface.co/t5-small/resolve/main/config.json
    tt_model = TtT5LayerSelfAttention(config, hf_reference_model.state_dict(), base_address, device, has_relative_attention_bias)
    tt_out = tt_model(torch2tt_tensor(test_input, device))[0]
    tt_out = tt2torch_tensor(tt_out)

    print(pt_out[0, 0, 1:10, 1:10])
    print(tt_out[0, 0, 1:10, 1:10])

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.98)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("test_T5LayerSelfAttention_inference Passed!")
    else:
        logger.warning("test_T5LayerSelfAttention_inference Failed!")


if __name__ == "__main__":
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    test_T5LayerSelfAttention_inference(device)
    ttm.device.CloseDevice(device)
