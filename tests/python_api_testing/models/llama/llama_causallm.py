import math
import torch
from torch import nn
import tt_lib
from python_api_testing.models.llama.llama_utils import (
    tt2torch_tensor,
    torch2tt_tensor,
    linear,
)
from typing import List, Optional, Tuple, Union
from python_api_testing.models.llama.llama_layer_norm import TtLlamaRMSNorm
from python_api_testing.models.llama.llama_decoder import TtLlamaDecoderLayer
from python_api_testing.models.llama.llama_model import TtLlamaShared


class TtLlamaForCausalLM(TtLlamaShared):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        max_position_embeddings,
        config,
        num_decoders,
    ):
        super().__init__(
            device, state_dict, base_url, max_position_embeddings, config, num_decoders
        )

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        self.state_dict = state_dict  # hugging_face_reference_model.state_dict()

        self.weight = torch2tt_tensor(self.state_dict["lm_head.weight"], self.device)
        self.bias = None

    def forward(self, x):
        encoder_output = super().forward(x)
        return linear(x, self.weight, self.bias)
