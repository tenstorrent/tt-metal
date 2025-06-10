# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import ttnn
from models.demos.segformer.tt.ttnn_segformer_model import TtSegformerModel


@dataclass
class TtSegFormerImageClassifierOutput:
    loss: ttnn.bfloat16 = None
    logits: ttnn.bfloat16 = None
    hidden_states: ttnn.bfloat16 = None
    attentions: ttnn.bfloat16 = None


class TtSegformerForImageClassification:
    def __init__(self, config, parameters):
        self.config = config
        self.num_labels = config.num_labels
        self.segformer = TtSegformerModel(config, parameters=parameters.segformer)

    def __call__(
        self,
        pixel_values: ttnn.bfloat16 = None,
        labels: ttnn.bfloat16 = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        parameters=None,
        model=None,
    ) -> Union[Tuple, TtSegFormerImageClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device = pixel_values.device()
        outputs = self.segformer(
            device,
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            parameters=parameters.segformer,
        )
        sequence_output = outputs[0]
        batch_size = sequence_output.shape[0]
        sequence_output = ttnn.reshape(sequence_output, (batch_size, -1, self.config.hidden_sizes[-1]))
        sequence_output = ttnn.mean(sequence_output, dim=1, keepdim=True)
        sequence_output = ttnn.squeeze(sequence_output, dim=0)
        logits = ttnn.linear(
            sequence_output,
            parameters.classifier.weight,
            bias=parameters.classifier.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=12),
            dtype=ttnn.bfloat8_b,
        )
        loss = None
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return TtSegFormerImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
