# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from typing import Optional, Tuple, Union
from dataclasses import dataclass

import ttnn

from models.experimental.roberta.tt.roberta_model import TtRobertaModel
from models.experimental.roberta.roberta_common import torch2tt_tensor


@dataclass
class TtTokenClassifierOutput:
    loss: ttnn.Tensor = None
    logits: ttnn.Tensor = None
    hidden_states: ttnn.Tensor = None
    attentions: ttnn.Tensor = None


class TtRobertaForTokenClassification(nn.Module):
    def __init__(self, config, state_dict, base_address, device, reference_model):
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

        # TODO: Add when implementing training
        # classifier_dropout = (
        #     config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # )
        # self.dropout = nn.Dropout(classifier_dropout)

        self.classifier_weight = torch2tt_tensor(state_dict[f"classifier.weight"], self.device)
        self.classifier_bias = torch2tt_tensor(state_dict[f"classifier.bias"], self.device)

    def linear(self, x, weight, bias):
        weight = ttnn.transpose(weight, -2, -1)
        x = ttnn.matmul(x, weight, memory_config=self.mem_config)
        if bias is not None:
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
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[ttnn.Tensor], TtTokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
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

        # TODO: Add when implementing training
        # sequence_output = self.dropout(sequence_output)
        logits = self.linear(sequence_output, self.classifier_weight, self.classifier_bias)

        loss = None
        if labels is not None:
            # TODO: Training not supported for now.
            pass

            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TtTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
