# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
import torch.nn as nn

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

import ttnn
from dataclasses import dataclass

from models.experimental.distilbert.tt.distilbert_embedding import TtDistilBert_Embeddings
from models.experimental.distilbert.tt.distilbert_transformer import TtTransformer


@dataclass
class TtBaseModelOutput:
    last_hidden_state: ttnn.Tensor = None
    hidden_states: Optional[Tuple[ttnn.Tensor]] = None
    attentions: Optional[Tuple[ttnn.Tensor]] = None


class TtDistilBertModel(nn.Module):
    def __init__(self, config, state_dict=None, base_address="", device=None):
        super().__init__()
        self.config = config
        self.state_dict = state_dict
        self.base_address = base_address
        self.device = device
        self.embeddings = TtDistilBert_Embeddings(
            self.config,
            state_dict=self.state_dict,
            base_address=f"{base_address}.embeddings",
            device=self.device,
        )
        self.transformer = TtTransformer(
            self.config,
            state_dict=self.state_dict,
            base_address=f"{base_address}.transformer",
            device=self.device,
        )

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """Implemented in PyTorch for now"""
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def get_head_mask(
        self,
        head_mask: Optional[ttnn.Tensor],
        num_hidden_layers: int,
        is_attention_chunked: bool = False,
    ) -> ttnn.Tensor:
        if head_mask is not None:
            torch_head_mask = tt_to_torch_tensor(head_mask)
        else:
            torch_head_mask = None

        if torch_head_mask is not None:
            torch_head_mask = self._convert_head_mask_to_5d(torch_head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                torch_head_mask = torch_head_mask.unsqueeze(-1)

            head_mask = torch_to_tt_tensor_rm(torch_head_mask, self.device)
        else:
            head_mask = [
                None,
            ] * num_hidden_layers

        return head_mask

    def forward(
        self,
        input_ids: Optional[ttnn.Tensor] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
        head_mask: Optional[ttnn.Tensor] = None,
        inputs_embeds: Optional[ttnn.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[TtBaseModelOutput, Tuple[ttnn.Tensor, ...]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = list(input_ids.shape)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.get_legacy_shape()[:-1]

        if attention_mask is not None:
            input_shape[0:0] = [1, 1]
            attention_mask = ttnn.ones(input_shape)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        """
        Torch tensor is passed as input for embedding to address low pcc
        """
        embeddings = self.embeddings(input_ids, inputs_embeds)

        return self.transformer(
            input=embeddings,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
