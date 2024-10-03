# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_segformer.tt.ttnn_segformer_encoder import TtSegformerEncoder
from typing import Optional, Tuple, Union
from dataclasses import dataclass


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
        pixel_values: ttnn.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        parameters=None,
    ) -> Union[Tuple, TtBaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            pixel_values,
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
