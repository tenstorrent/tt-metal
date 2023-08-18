from typing import Optional, Tuple, Union
import torch.nn as nn

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

import tt_lib
from dataclasses import dataclass

from models.squeezebert.tt.squeezebert_embedding import TtSqueezeBert_Embeddings
from models.squeezebert.tt.squeezebert_encoder import TtSqueezeBert_Encoder
from models.squeezebert.tt.squeezebert_pooler import TtSqueezeBert_Pooler


@dataclass
class TtBaseModelOutputWithPooling:
    last_hidden_state: tt_lib.tensor.Tensor = None
    pooler_output: tt_lib.tensor.Tensor = None
    hidden_states: Optional[Tuple[tt_lib.tensor.Tensor]] = None
    attentions: Optional[Tuple[tt_lib.tensor.Tensor]] = None


class TtSqueezeBertModel(nn.Module):
    def __init__(self, config, base_address="", state_dict=None, device=None) -> None:
        super().__init__()

        self.config = config
        self.base_address = base_address
        self.state_dict = state_dict
        self.device = device

        self.embeddings = TtSqueezeBert_Embeddings(
            self.config,
            state_dict=self.state_dict,
            base_address=f"{self.base_address}.embeddings",
            device=self.device,
        )

        self.encoder = TtSqueezeBert_Encoder(
            self.config,
            state_dict=self.state_dict,
            base_address=f"{self.base_address}.encoder",
            device=self.device,
        )

        self.pooler = TtSqueezeBert_Pooler(
            self.config,
            state_dict=self.state_dict,
            base_address=f"{self.base_address}.pooler",
            device=self.device,
        )

    def _convert_head_mask_to_5d(
        self, head_mask: tt_lib.tensor.Tensor, num_hidden_layers: int
    ):
        """Implemented in PyTorch for now"""
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = (
                head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            )  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(
            dtype=self.dtype
        )  # switch to float if need + fp16 compatibility
        return head_mask

    def get_head_mask(
        self,
        head_mask: Optional[tt_lib.tensor.Tensor],
        num_hidden_layers: int,
        is_attention_chunked: bool = False,
    ) -> tt_lib.tensor.Tensor:
        if head_mask is not None:
            torch_head_mask = tt_to_torch_tensor(head_mask)
        else:
            torch_head_mask = None

        if torch_head_mask is not None:
            torch_head_mask = self._convert_head_mask_to_5d(
                torch_head_mask, num_hidden_layers
            )
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
        input_ids: Optional[tt_lib.tensor.Tensor] = None,
        attention_mask: Optional[tt_lib.tensor.Tensor] = None,
        token_type_ids: Optional[tt_lib.tensor.Tensor] = None,
        position_ids: Optional[tt_lib.tensor.Tensor] = None,
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        inputs_embeds: Optional[tt_lib.tensor.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, TtBaseModelOutputWithPooling]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )

        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = tt_lib.tensor.ones(input_shape)

        if token_type_ids is None:
            token_type_ids = tt_lib.tensor.zeros(input_shape)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return TtBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
