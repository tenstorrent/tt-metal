# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
from dataclasses import dataclass

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

import ttnn
from models.experimental.swin.swin_helper_funcs import linear as TtLinear
from models.experimental.swin.tt.swin_model import TtSwinModel


@dataclass
class TtSwinImageClassifierOutput:
    loss: Optional[ttnn.Tensor] = None
    logits: ttnn.Tensor = None
    hidden_states: Optional[Tuple[ttnn.Tensor]] = None
    attentions: Optional[Tuple[ttnn.Tensor]] = None
    reshaped_hidden_states: Optional[Tuple[ttnn.Tensor]] = None


class TtSwinForImageClassification(nn.Module):
    def __init__(self, config, state_dict, base_address, device) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.num_labels = self.config.num_labels
        self.swin = TtSwinModel(self.config, state_dict, base_address, self.device)

        self.weight = torch_to_tt_tensor_rm(state_dict["classifier.weight"], self.device)
        self.bias = torch_to_tt_tensor_rm(state_dict["classifier.bias"], self.device)

    # Classifier head
    def classifier(self, pooled_output: ttnn.Tensor):
        if self.config.num_labels > 0:
            return TtLinear(pooled_output, self.weight, self.bias)
        else:
            classifier = nn.Identity()
            pooled_output = tt_to_torch_tensor(pooled_output)
            pooled_output = classifier(pooled_output)
            pooled_output = torch_to_tt_tensor_rm(pooled_output, self.device)
            return pooled_output

    def forward(
        self,
        pixel_values: Optional[ttnn.Tensor] = None,
        head_mask: Optional[ttnn.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TtSwinImageClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.swin(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.pooler_output

        logits = self.classifier(pooled_output)
        logits = tt_to_torch_tensor(logits).squeeze(0).squeeze(0)
        loss = None
        if labels is not None:
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

        logits = torch_to_tt_tensor_rm(logits, self.device)
        if loss is not None:
            loss = torch_to_tt_tensor_rm(loss, self.device)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TtSwinImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
        )
