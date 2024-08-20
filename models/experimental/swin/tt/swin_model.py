# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn


from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

import ttnn
from tt_lib.fallback_ops import fallback_ops
from models.experimental.swin.tt.swin_encoder import TtSwinEncoder
from models.experimental.swin.tt.swin_embeddings import TtSwinEmbeddings
from dataclasses import dataclass


@dataclass
class TtSwinModelOutput:
    last_hidden_state: ttnn.Tensor = None
    pooler_output: Optional[ttnn.Tensor] = None
    hidden_states: Optional[Tuple[ttnn.Tensor]] = None
    attentions: Optional[Tuple[ttnn.Tensor]] = None
    reshaped_hidden_states: Optional[Tuple[ttnn.Tensor]] = None


class TtSwinModel(nn.Module):
    def __init__(
        self,
        config,
        state_dict,
        base_address,
        device,
        add_pooling_layer=True,
        use_mask_token=False,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))
        self.embeddings = TtSwinEmbeddings(
            self.config,
            state_dict,
            f"{base_address}" + "embeddings",
            self.device,
            use_mask_token=use_mask_token,
        )
        self.encoder = TtSwinEncoder(
            self.config,
            self.embeddings.patch_grid,
            state_dict,
            f"{base_address}" + "encoder",
            self.device,
        )

        gamma = torch_to_tt_tensor_rm(state_dict[f"{base_address}" + "layernorm.weight"], self.device)
        beta = torch_to_tt_tensor_rm(state_dict[f"{base_address}" + "layernorm.bias"], self.device)
        self.layernorm = fallback_ops.LayerNorm(
            gamma, beta, normalized_shape=self.num_features, eps=config.layer_norm_eps
        )

        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

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
        pixel_values: Optional[ttnn.Tensor] = None,
        bool_masked_pos: Optional[ttnn.Tensor] = None,
        head_mask: Optional[ttnn.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TtSwinModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))

        embedding_output, input_dimensions = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)

        pooled_output = None
        if self.pooler is not None:
            sequence_output_transpose = ttnn.transpose(sequence_output, -2, -1)
            sequence_output_transpose = tt_to_torch_tensor(sequence_output_transpose).squeeze(0)
            pooled_output = self.pooler(sequence_output_transpose)
            pooled_output = torch.flatten(pooled_output, 1)
            pooled_output = torch_to_tt_tensor_rm(pooled_output, self.device)

        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]

            return output

        return TtSwinModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )
