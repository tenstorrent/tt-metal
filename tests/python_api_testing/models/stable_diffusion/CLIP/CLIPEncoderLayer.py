import numpy as np
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")


from typing import Optional, Tuple

import torch.nn as nn
import torch.nn.functional as F
import torch

from libs.tt_lib.utils import tilize_to_list, pad_weight



from libs import tt_lib as ttl
from utility_functions import print_diff_argmax, torch_to_tt_tensor, tt_to_torch_tensor, print_corr_coef
from python_api_testing.fused_ops.linear import Linear as tt_linear
from python_api_testing.fused_ops.layernorm import Layernorm as tt_layernorm
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose_and_pcc
from CLIPAttention import TtCLIPAttention
from CLIPMLP import TtCLIPMLP
from transformers import CLIPModel, CLIPConfig


class TtCLIPEncoderLayer(nn.Module):
    def __init__(self, device, state_dict, config=None, hidden_size=None, base_address="text_model.encoder.layers.10"):
        super().__init__()
        self.device = device
        self.embed_dim = config.hidden_size if config else hidden_size
        self.self_attn = TtCLIPAttention(device=device, config=config, state_dict=state_dict)


        self.layer_norm1_weight = tilize_to_list(pad_weight(state_dict[f"{base_address}.layer_norm1.weight"]))
        self.layer_norm1_bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.layer_norm1.bias"]))
        print(state_dict[f"{base_address}.layer_norm1.weight"].shape)
        # H = self.embed_dim
        # W = -1
        self.layer_norm1 = tt_layernorm(gamma=self.layer_norm1_weight, beta=self.layer_norm1_bias, epsilon=config.layer_norm_eps, H=-1, W=self.embed_dim, device=device, num_dims=1)


        self.mlp = TtCLIPMLP(device=device, config=config, state_dict=state_dict)

        self.layer_norm2_weight = tilize_to_list(pad_weight(state_dict[f"{base_address}.layer_norm2.weight"]))
        self.layer_norm2_bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.layer_norm2.bias"]))

        self.layer_norm2 = tt_layernorm(gamma=self.layer_norm2_weight, beta=self.layer_norm2_bias, epsilon=config.layer_norm_eps, H=-1, W=self.embed_dim, device=device, num_dims=1)

    def forward(
        self,
        hidden_states,
        attention_mask,
        causal_attention_mask,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:

        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = ttl.tensor.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = ttl.tensor.add(residual, hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


def run_clip_encoder_layer_inference(device):


    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    state_dict = model.state_dict()
    config = model.config.text_config

    D = 96 # should be 77
    hidden_states_shape = [1, 1, D, 512]
    attention_mask = None
    causal_attention_mask_shape = [1, 1, D, D]
    output_attentions = False

    hidden_states = torch.randn(hidden_states_shape)
    causal_attention_mask = torch.randn(causal_attention_mask_shape)

    torch_encoder = model.text_model.encoder.layers[10]
    torch_out = torch_encoder(hidden_states=hidden_states.squeeze(0), attention_mask=attention_mask, causal_attention_mask= causal_attention_mask, output_attentions=output_attentions)

    tt_hidden_states = torch_to_tt_tensor(hidden_states, device)
    tt_causal_attention_mask = torch_to_tt_tensor(causal_attention_mask, device)

    tt_encoder = TtCLIPEncoderLayer(device=device, config=config, state_dict=state_dict)

    tt_out = tt_encoder(hidden_states=tt_hidden_states, causal_attention_mask=tt_causal_attention_mask, attention_mask=attention_mask, output_attentions=output_attentions)
    tt_out = tt_to_torch_tensor(tt_out, host)
    print_diff_argmax(tt_out, torch_out)
    print(comp_allclose_and_pcc(torch_out, tt_out))



if __name__ == "__main__":
    # Initialize the device
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    ttl.device.InitializeDevice(device)
    host = ttl.device.GetHost()
    run_clip_encoder_layer_inference(device)
    ttl.device.CloseDevice(device)
