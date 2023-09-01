"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

import torch
import tt_lib
import torch.nn as nn
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)
from models.bloom.tt.bloom_model import TtBloomModel
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from models.helper_funcs import Linear as TtLinear


@dataclass
class TtQuestionAnsweringModelOutput:
    loss: Optional[tt_lib.tensor.Tensor] = None
    start_logits: tt_lib.tensor.Tensor = None
    end_logits: tt_lib.tensor.Tensor = None
    hidden_states: Optional[Tuple[tt_lib.tensor.Tensor]] = None
    attentions: Optional[Tuple[tt_lib.tensor.Tensor]] = None


class TtBloomForQuestionAnswering(nn.Module):
    def __init__(self, config, state_dict, device):
        super().__init__()
        self.config = config
        self.state_dict = state_dict
        self.device = device

        self.transformer = TtBloomModel(
            self.config, self.state_dict, "transformer", self.device
        )

        self.qa_outputs_weight = torch_to_tt_tensor_rm(
            state_dict["qa_outputs.weight"], device
        )
        self.qa_outputs_bias = torch_to_tt_tensor_rm(
            state_dict["qa_outputs.bias"], device
        )

        self.qa_outputs = TtLinear(
            self.qa_outputs_weight.shape()[-1],
            self.qa_outputs_weight.shape()[-2],
            self.qa_outputs_weight,
            self.qa_outputs_bias,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[tt_lib.tensor.Tensor] = None,
        position_ids: Optional[tt_lib.tensor.Tensor] = None,
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        inputs_embeds: Optional[tt_lib.tensor.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[tt_lib.tensor.Tensor, ...], TtQuestionAnsweringModelOutput]:
        self.use_return_dict = False

        return_dict = return_dict if return_dict is not None else self.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        logits = tt_to_torch_tensor(logits)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None

        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)

            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        end_logits = torch_to_tt_tensor_rm(end_logits, self.device)
        start_logits = torch_to_tt_tensor_rm(start_logits, self.device)

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return TtQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
