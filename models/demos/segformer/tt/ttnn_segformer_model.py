# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import ttnn
from models.demos.segformer.tt.ttnn_segformer_encoder import TtSegformerEncoder


@dataclass
class TtBaseModelOutput:
    last_hidden_state: ttnn.bfloat16 = None
    hidden_states: ttnn.bfloat16 = None
    attentions: ttnn.bfloat16 = None

    def __getitem__(self, idx):
        if idx == 0:
            return self.last_hidden_state
        elif idx == 1:
            return self.hidden_states
        elif idx == 2:
            return self.attentions
        else:
            raise IndexError("Index out of range")


class TtSegformerModel:
    def __init__(self, config, parameters):
        super().__init__()
        self.config = config

        # hierarchical Transformer encoder
        self.encoder = TtSegformerEncoder(config, parameters.encoder)

    def __call__(
        self,
        device,
        pixel_values: ttnn.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        parameters=None,
        min_channels=8,
    ) -> Union[Tuple, TtBaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        print("input is", pixel_values.shape, pixel_values.dtype, pixel_values.layout)
        N, C, H, W = pixel_values.shape
        if C < min_channels:
            channel_padding_needed = min_channels - C
            nchw = ttnn.pad(pixel_values, ((0, 0), (0, channel_padding_needed), (0, 0), (0, 0)), value=0.0)
        else:
            nchw = pixel_values
        nhwc = ttnn.permute(nchw, (0, 2, 3, 1))
        ttnn.deallocate(nchw)
        ttnn.deallocate(pixel_values)
        nhwc = ttnn.reallocate(nhwc)
        print("input is", nhwc.shape, nhwc.dtype, nhwc.layout)
        encoder_outputs = self.encoder(
            device,
            nhwc,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            parameters=parameters.encoder,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return TtBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
