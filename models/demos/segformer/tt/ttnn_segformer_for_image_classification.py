# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union, Tuple
import ttnn
from models.demos.segformer.tt.ttnn_segformer_model import TtSegformerModel
from dataclasses import dataclass


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
        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            parameters=parameters.segformer,
        )
        sequence_output = outputs[0]
        print(
            "ttn output",
            sequence_output.shape,
            sequence_output.layout,
            sequence_output.dtype,
            sequence_output.memory_config(),
        )
        # convert last hidden states to (batch_size, height*width, hidden_size)
        batch_size = sequence_output.shape[0]
        # if self.config.reshape_last_stage:
        # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
        # sequence_output = ttnn.permute(sequence_output, (0, 2, 3, 1))
        sequence_output = ttnn.reshape(sequence_output, (batch_size, -1, self.config.hidden_sizes[-1]))
        print(
            "ttn output after reshape",
            sequence_output.shape,
            sequence_output.layout,
            sequence_output.dtype,
            sequence_output.memory_config(),
        )
        # global average pooling
        sequence_output = ttnn.mean(sequence_output, dim=1)
        print(
            "ttn output after mean",
            sequence_output.shape,
            sequence_output.layout,
            sequence_output.dtype,
            sequence_output.memory_config(),
        )
        sequence_output = ttnn.to_layout(sequence_output, layout=ttnn.ROW_MAJOR_LAYOUT)
        print(
            "ttn output after meanlayotu,",
            sequence_output.shape,
            sequence_output.layout,
            sequence_output.dtype,
            sequence_output.memory_config(),
        )
        sequence_output = ttnn.reshape(sequence_output, (batch_size, sequence_output.shape[-1]))
        print(
            "ttn output after reshape",
            sequence_output.shape,
            sequence_output.layout,
            sequence_output.dtype,
            sequence_output.memory_config(),
        )
        sequence_output = ttnn.to_layout(sequence_output, layout=ttnn.TILE_LAYOUT)
        print(
            "ttn output after layout",
            sequence_output.shape,
            sequence_output.layout,
            sequence_output.dtype,
            sequence_output.memory_config(),
        )
        logits = ttnn.linear(
            sequence_output,
            parameters.classifier.weight,
            bias=parameters.classifier.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(y=8, x=12),
            dtype=ttnn.bfloat8_b,
        )
        print("ttn output after linear", logits.shape, logits.layout, logits.dtype, logits.memory_config())
        loss = None
        if not return_dict:
            print("hiii")
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return TtSegFormerImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
