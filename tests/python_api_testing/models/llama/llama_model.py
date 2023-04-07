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


class LlamaShared(torch.nn.Module):
    @abstractmethod
    # device, state_dict, base_url, max_position_embeddings, config, num_decoders
    def __init__(self, device, state_dict, base_url, max_position_embeddings, config, num_decoders):
        super().__init__()

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        self.device = device
        self.state_dict = state_dict
        self.base_url = base_url
        self.max_position_embeddings = max_position_embeddings

        # So far on CPU until we add embeddings support on device
        # self.embeddings = PytorchEmbeddings(hugging_face_reference_model)
        self.embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.embeddings.weight = torch.nn.Parameter(state_dict[f"model.embed_tokens.weight"])

        # stack all decoders
        self.decoders = torch.nn.Sequential(*[TtLlamaDecoderLayer(self.device, self.state_dict, self.base_url, decoder_idx, self.max_position_embeddings, config) for decoder_idx in range(num_decoders)])

        # add final normalization layer
        self.layer_num = None
        self.layer_position = 'norm'
        self.final_layernorm = TtLlamaRMSNorm(
            device,
            state_dict=self.state_dict,
            base_url=self.base_url,
            layer_num=self.layer_num,
            layer_position=self.layer_position,
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps
        )

        self.device = device

    @abstractmethod
    def forward(self, x):
        embeddings = self.embeddings(x)
        # Convert to ll buda tensor
        pad_embeddings = pad_activation(embeddings)
        tt_embeddings = ttl.tensor.Tensor(pad_embeddings.reshape(-1).tolist(), (pad_embeddings.shape[0], 1, pad_embeddings.shape[-2], pad_embeddings.shape[-1]), ttl.tensor.DataType.BFLOAT16,  ttl.tensor.Layout.ROW_MAJOR).to(ttl.tensor.Layout.TILE)
        tt_embeddings = tt_embeddings.to(self.device)

        # apply decoders
        encoder_output = self.decoders(tt_embeddings)

        # apply final norm layer
        encoder_output = self.final_layernorm(encoder_output)
        return encoder_output


class TtLlamaModel(LlamaShared):
    def __init__(self, device, state_dict, base_url, max_position_embeddings, config, num_decoders):
        # config, num_decoders, state_dict, device
        super().__init__(device, state_dict, base_url, max_position_embeddings, config, num_decoders)

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        self.state_dict = state_dict

        num_classes, hidden_size = state_dict["lm_head.weight"].shape

        weight = tilize_to_list(pad_weight(state_dict["lm_head.weight"]))
        bias = None

    def forward(self, x):
        encoder_output = super().forward(x)
        return encoder_output
