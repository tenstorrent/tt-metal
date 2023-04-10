import math
from abc import abstractmethod
import torch
from torch import nn
from libs import tt_lib as ttl
from python_api_testing.models.t5.t5_utils import tt2torch_tensor, torch2tt_tensor
from typing import List, Optional, Tuple, Union
from fused_ops.linear import Linear as TtLinear
from fused_ops.softmax import softmax as TTsoftmax
from python_api_testing.models.llama.llama_layer_norm import TtLlamaRMSNorm
from python_api_testing.models.llama.llama_decoder import TtLlamaDecoderLayer
from python_api_testing.models.llama.llama_utils import *
from utility_functions import pad_activation, pad_weight, tilize_to_list, untilize, nearest_32, print_diff_argmax, tt2torch, tt2torch_rm
from python_api_testing.models.llama.llama_model import TtLlamaShared


class TtLlamaForCausalLM(TtLlamaShared):
    def __init__(self, device, state_dict, base_url, max_position_embeddings, config, num_decoders):
        super().__init__(device, state_dict, base_url, max_position_embeddings, config, num_decoders)

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        self.state_dict = state_dict # hugging_face_reference_model.state_dict()

        num_classes, hidden_size = state_dict["lm_head.weight"].shape

        weight = tilize_to_list(pad_weight(state_dict["lm_head.weight"]))
        bias = None

        # CausalLM linear
        self.CausalLM_linear = TtLinear(hidden_size, config.vocab_size, weight, bias, device)

    def forward(self, x):
        encoder_output = super().forward(x)
        return self.CausalLM_linear(encoder_output)
