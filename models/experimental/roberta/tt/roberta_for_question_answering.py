# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from typing import Optional, Tuple, Union
from dataclasses import dataclass

import ttnn

from models.experimental.roberta.tt.roberta_model import TtRobertaModel
from models.utility_functions import (
    tt2torch_tensor,
)
from models.experimental.roberta.roberta_common import torch2tt_tensor


@dataclass
class TtQuestionAnsweringModelOutput:
    loss: ttnn.Tensor = None
    start_logits: torch.Tensor = None
    end_logits: torch.Tensor = None
    hidden_states: ttnn.Tensor = None
    attentions: ttnn.Tensor = None


class TtRobertaForQuestionAnswering(nn.Module):
    def __init__(
        self,
        config,
        state_dict,
        base_address,
        device,
        reference_model,
    ):
        super().__init__()
        self.mem_config = ttnn.L1_MEMORY_CONFIG
        self.config = config
        self.device = device
        self.num_labels = config.num_labels

        self.roberta = TtRobertaModel(
            config=config,
            state_dict=state_dict,
            base_address=f"{base_address}" + "roberta",
            device=device,
            reference_model=reference_model.roberta,
            add_pooling_layer=False,
        )

        self.qa_outputs_weight = torch2tt_tensor(state_dict[f"qa_outputs.weight"], self.device)

        self.qa_outputs_bias = torch2tt_tensor(state_dict[f"qa_outputs.bias"], self.device)

    def linear(self, x, weight, bias):
        weight = ttnn.transpose(weight, -2, -1)
        x = ttnn.matmul(x, weight, memory_config=self.mem_config)
        x = ttnn.add(
            x,
            bias,
            memory_config=self.mem_config,
        )
        return x

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[ttnn.Tensor] = None,
        inputs_embeds: Optional[ttnn.Tensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[ttnn.Tensor], TtQuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state

        logits = self.linear(sequence_output, self.qa_outputs_weight, self.qa_outputs_bias)

        torch_logits = tt2torch_tensor(logits).squeeze(0)
        start_logits, end_logits = torch_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # TODO: Training not supported for now.
            pass
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

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
