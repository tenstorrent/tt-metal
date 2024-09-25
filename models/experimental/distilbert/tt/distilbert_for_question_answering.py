# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
import torch.nn as nn
from dataclasses import dataclass

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

import ttnn
from dataclasses import dataclass
from models.experimental.distilbert.tt.distilbert_model import TtDistilBertModel
from models.helper_funcs import Linear as TtLinear


@dataclass
class TtQuestionAnsweringModelOutput:
    loss: Optional[ttnn.Tensor] = None
    start_logits: ttnn.Tensor = None
    end_logits: ttnn.Tensor = None
    hidden_states: Optional[Tuple[ttnn.Tensor]] = None
    attentions: Optional[Tuple[ttnn.Tensor]] = None


class TtDistilBertForQuestionAnswering(nn.Module):
    def __init__(self, config, state_dict=None, base_address="", device=None) -> None:
        super().__init__()
        self.config = config
        self.state_dict = state_dict
        self.device = device
        self.distilbert = TtDistilBertModel(
            self.config,
            state_dict=self.state_dict,
            base_address=f"distilbert",
            device=self.device,
        )
        self.qa_weight = torch_to_tt_tensor_rm(state_dict["qa_outputs.weight"], self.device)
        self.qa_bias = torch_to_tt_tensor_rm(state_dict["qa_outputs.bias"], self.device)
        self.qa_linear = TtLinear(
            self.qa_weight.shape.with_tile_padding()[-1],
            self.qa_weight.shape.with_tile_padding()[-2],
            self.qa_weight,
            self.qa_bias,
        )
        if self.config.num_labels != 2:
            raise ValueError(f"config.num_labels should be 2, but it is {config.num_labels}")

    def forward(
        self,
        input_ids: Optional[ttnn.Tensor] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
        head_mask: Optional[ttnn.Tensor] = None,
        inputs_embeds: Optional[ttnn.Tensor] = None,
        start_positions: Optional[ttnn.Tensor] = None,
        end_positions: Optional[ttnn.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[TtQuestionAnsweringModelOutput, Tuple[ttnn.Tensor, ...]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = distilbert_output[0]
        logits = self.qa_linear(hidden_states)
        total_loss = None
        logits = tt_to_torch_tensor(logits)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = torch_to_tt_tensor_rm(start_logits.squeeze(-1), self.device, put_on_device=False)
        end_logits = torch_to_tt_tensor_rm(end_logits.squeeze(-1), self.device, put_on_device=False)

        if not return_dict:
            output = (start_logits, end_logits) + distilbert_output[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return TtQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=hidden_states,
            attentions=None,
        )
