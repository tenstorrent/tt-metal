import torch
import pytest
from torch import nn
from abc import abstractmethod

import tt_lib

from tests.python_api_testing.models.falcon.falcon_decoder import TtFalconDecoderLayer
from models.utility_functions import (
    torch2tt_tensor,
    pad_by_zero,
)


class TtFalconModelShared(torch.nn.Module):
    @abstractmethod
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        num_layers,
        config,
        max_position_embeddings,
    ):
        super().__init__()

        # NOTE: Once we make embeddings run on device, pass in state dict
        # instead of model itself
        self.device = device
        self.state_dict = state_dict
        self.base_url = base_url
        self.config = config
        self.max_position_embeddings = max_position_embeddings

        # So far on CPU until we add embeddings support on device
        self.embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.embeddings.weight = torch.nn.Parameter(
            state_dict[f"{base_url}.word_embeddings.weight"]
        )

        # stack all decoders
        self.layers = torch.nn.ModuleList(
            [
                TtFalconDecoderLayer(
                    device=device,
                    state_dict=state_dict,
                    base_url=f"{base_url}.h",
                    layer_num=layer_num,
                    config=config,
                    max_position_embeddings=max_position_embeddings,
                )
                for layer_num in range(num_layers)
            ]
        )

        self.layernorm_gamma = pad_by_zero(
            self.state_dict[f"{base_url}.ln_f.weight"], device
        )[0]
        self.layernorm_beta = pad_by_zero(
            self.state_dict[f"{base_url}.ln_f.bias"], device
        )[0]
        self.layernorm_eps = config.layer_norm_epsilon

    def model_preprocessing(self, input_ids):
        # input_ids: torch.Tensor with shape [batch, seq_len]
        assert input_ids.dim() == 2
        embeddings = self.embeddings(input_ids)
        tt_embeddings = torch2tt_tensor(embeddings, self.device)

        batch, seq_len = input_ids.shape
        q_len, kv_seq_len = seq_len, seq_len
        tt_attention_mask = (
            torch.ones(batch, self.config.n_head, q_len, kv_seq_len) * -100000
        ).triu(diagonal=1)
        tt_attention_mask = torch2tt_tensor(tt_attention_mask, self.device)

        return tt_embeddings, tt_attention_mask

    @abstractmethod
    def forward(
        self,
        input_embeddings: tt_lib.tensor.Tensor,
        attention_mask: tt_lib.tensor.Tensor = None,
    ) -> tt_lib.tensor.Tensor:
        layer_output = input_embeddings
        for idx, layer in enumerate(self.layers):
            layer_output = layer(
                hidden_states=layer_output, alibi=None, attention_mask=attention_mask
            )[0]

        # apply final norm layer
        layer_output = tt_lib.tensor.layernorm(
            layer_output,
            self.layernorm_eps,  # These don't fit: self.layernorm_gamma, self.layernorm_beta
        )
        layer_output = tt_lib.tensor.bcast(
            layer_output,
            self.layernorm_gamma,
            tt_lib.tensor.BcastOpMath.MUL,
            tt_lib.tensor.BcastOpDim.H,
        )
        layer_output = tt_lib.tensor.bcast(
            layer_output,
            self.layernorm_beta,
            tt_lib.tensor.BcastOpMath.ADD,
            tt_lib.tensor.BcastOpDim.H,
        )
        layer_output = layer_output

        return layer_output


class TtFalconModel(TtFalconModelShared):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        num_layers,
        config,
        max_position_embeddings,
    ):
        super().__init__(
            device=device,
            state_dict=state_dict,
            base_url=base_url,
            num_layers=num_layers,
            config=config,
            max_position_embeddings=max_position_embeddings,
        )

    def forward(
        self,
        input_embeddings: tt_lib.tensor.Tensor,
        attention_mask: tt_lib.tensor.Tensor = None,
    ) -> tt_lib.tensor.Tensor:
        hidden_states = super().forward(input_embeddings, attention_mask)
        return hidden_states
