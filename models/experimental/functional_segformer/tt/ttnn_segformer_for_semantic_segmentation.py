# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_segformer.tt.ttnn_segformer_decode_head import TtSegformerDecodeHead
from models.experimental.functional_segformer.tt.ttnn_segformer_model import TtSegformerModel
from typing import Tuple, Union, Optional
from dataclasses import dataclass


@dataclass
class TtSemanticSegmenterOutput:
    loss: ttnn.bfloat16 = None
    logits: ttnn.bfloat16 = None
    hidden_states: ttnn.bfloat16 = None
    attentions: ttnn.bfloat16 = None


class TtSegformerForSemanticSegmentation:
    def __init__(self, config, parameters):
        super().__init__()
        self.segformer = TtSegformerModel(config, parameters=parameters.segformer)
        self.decode_head = TtSegformerDecodeHead(config, parameters=parameters.decode_head)
        self.config = config

    def __call__(
        self,
        pixel_values: ttnn.bfloat16,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict: Optional[bool] = None,
        parameters=None,
    ) -> Union[Tuple, TtSemanticSegmenterOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if labels is not None and self.config.num_labels < 1:
            raise ValueError(f"Number of labels should be >=0: {self.config.num_labels}")

        outputs = self.segformer(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
            parameters=parameters.segformer,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        logits = self.decode_head(encoder_hidden_states, parameters=parameters.decode_head)

        loss = None

        return TtSemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
