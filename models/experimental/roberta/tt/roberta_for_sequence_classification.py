# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from typing import Optional, Tuple, Union
from dataclasses import dataclass

import ttnn

from models.experimental.roberta.tt.roberta_model import TtRobertaModel
from models.experimental.roberta.tt.roberta_classification_head import TtRobertaClassificationHead


@dataclass
class TtSequenceClassifierOutput:
    loss: ttnn.Tensor = None
    logits: ttnn.Tensor = None
    hidden_states: ttnn.Tensor = None
    attentions: ttnn.Tensor = None


class TtRobertaForSequenceClassification(nn.Module):
    def __init__(self, config, state_dict, base_address, device, reference_model):
        super().__init__()
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
        self.classifier = TtRobertaClassificationHead(
            config=self.config,
            base_address=f"{base_address}" + "classifier",
            device=self.device,
            state_dict=state_dict,
        )

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
    ) -> Union[Tuple[ttnn.Tensor], TtSequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
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
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # TODO: Training not supported for now.
            pass

            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TtSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
