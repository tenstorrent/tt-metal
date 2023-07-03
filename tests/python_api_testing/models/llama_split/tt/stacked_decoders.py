from loguru import logger
import torch
import numpy as np
from torch import nn, Tensor
import tt_lib

from python_api_testing.models.llama.llama_decoder import TtLlamaDecoderLayer
from python_api_testing.models.llama.llama_attention import TtLlamaAttention
from python_api_testing.models.llama.llama_layer_norm import TtLlamaRMSNorm
from python_api_testing.models.llama.llama_utils import *


class TtLlamaDecoderModelStacked(torch.nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        config,
        start,
        count,
    ):
        super().__init__()
        self.device = device
        self.state_dict = state_dict
        self.base_url = base_url
        self.max_position_embeddings = max_position_embeddings
        self.config = config

        self.decoder_list = torch.nn.Sequential(
            *[
                TtLlamaDecoderLayer(
                    self.device,
                    self.state_dict,
                    self.base_url,
                    decoder_idx,
                    self.max_position_embeddings,
                    self.config,
                )
                for decoder_idx in range(start, start + count)
            ]
        )

        # add final normalization layer
        self.layer_num = None
        self.layer_position = "norm"
        self.final_layernorm = TtLlamaRMSNorm(
            self.device,
            state_dict=self.state_dict,
            base_url=self.base_url,
            layer_num=self.layer_num,
            layer_position=self.layer_position,
            hidden_size=self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

        # if it is CausalLM Llama model
        self.weight = torch2tt_tensor(self.state_dict[f"lm_head.weight"], self.device)
        self.bias = None

    def forward(self, x, y, half=1, has_layer_norm=False, is_causal=False):
        result = x
        for idx, decoder_layer in enumerate(self.decoder_list):
            result = decoder_layer(hidden_states=result, position_ids=y)[0]

        if half == 2:
            # add norm layer
            if has_layer_norm:
                result = self.final_layernorm(result)
            # add linear
            if is_causal:
                result = linear(result, self.weight, self.bias)

        return result
