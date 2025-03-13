# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.demos.segformer.reference.segformer_model import (
    SegformerModelReference,
    SegformerPreTrainedModel,
)
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.segformer.modeling_segformer import SegFormerImageClassifierOutput


class SegformerForImageClassificationReference(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.segformer = SegformerModelReference(config)
        # Classifier head
        self.classifier = nn.Linear(config.hidden_sizes[-1], config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SegFormerImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        print("output shape", sequence_output.shape)
        # convert last hidden states to (batch_size, height*width, hidden_size)
        batch_size = sequence_output.shape[0]
        if self.config.reshape_last_stage:
            # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
            print("output shapeafter permute", sequence_output.shape)
            sequence_output = sequence_output.permute(0, 2, 3, 1)
        sequence_output = sequence_output.reshape(batch_size, -1, self.config.hidden_sizes[-1])
        print("output shape after reshape", sequence_output.shape)
        # global average pooling
        sequence_output = sequence_output.mean(dim=1)
        print("output shape after mean", sequence_output.shape)
        logits = self.classifier(sequence_output)
        print("output shape after linear", sequence_output.shape)
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
        if not return_dict:
            print("hi")
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        print("outputs", logits.shape, outputs.hidden_states, outputs.attentions)
        #         output shape torch.Size([1, 256, 16, 16])
        # output shapeafter permute torch.Size([1, 256, 16, 16])
        # output shape after reshape torch.Size([1, 256, 256])
        # output shape after mean torch.Size([1, 256])
        # output shape after linear torch.Size([1, 256])
        # outputs torch.Size([1, 1000]) None None
        return SegFormerImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
