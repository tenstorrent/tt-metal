# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import ttnn
from models.experimental.functional_segformer.tt.ttnn_segformer_mlp import TtSegformerMLP
from torch import nn
import tt_lib


def torch_to_ttnn(input, device, layout=ttnn.TILE_LAYOUT):
    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_layout(input, layout)
    input = ttnn.to_device(input, device)
    return input


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


class TtSegformerDecodeHead:
    def __init__(self, config, parameters):
        super().__init__()
        # linear layers which will unify the channel dimension of each of the encoder blocks to the same config.decoder_hidden_size
        mlps = []
        for i in range(config.num_encoder_blocks):
            setattr(
                self,
                f"mlp_{i}",
                TtSegformerMLP(),
            )
            mlps.append(f"mlp_{i}")
        self.linear_c = mlps

        self.linear_fuse = parameters.linear_fuse

        self.classifier = parameters.classifier

        self.config = config

    def __call__(self, encoder_hidden_states: ttnn.bfloat16, parameters) -> ttnn.Tensor:
        device = encoder_hidden_states[-1].device()
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp_name in zip(encoder_hidden_states, self.linear_c):
            if self.config.reshape_last_stage is False and (encoder_hidden_state.shape) == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = ttnn.reshape(encoder_hidden_state, (batch_size, height, width, -1))
                encoder_hidden_state = ttnn.permute(encoder_hidden_state, (0, 3, 1, 2))

            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            mlp = getattr(self, mlp_name)
            encoder_hidden_state = mlp(encoder_hidden_state, parameters=parameters[mlp_name])
            encoder_hidden_state = ttnn.permute(encoder_hidden_state, (0, 2, 1))
            encoder_hidden_state = ttnn.from_device(encoder_hidden_state)
            encoder_hidden_state = ttnn.to_layout(encoder_hidden_state, layout=ttnn.ROW_MAJOR_LAYOUT)
            encoder_hidden_state = ttnn.reshape(encoder_hidden_state, (batch_size, -1, height, width))
            encoder_hidden_state = ttnn.to_device(encoder_hidden_state, device)
            encoder_hidden_state = ttnn.to_layout(encoder_hidden_state, layout=ttnn.TILE_LAYOUT)
            encoder_hidden_state = ttnn_to_torch(encoder_hidden_state)

            # interpolate is kept in torch as we don't have support
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state,
                size=ttnn_to_torch(encoder_hidden_states[0]).size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            encoder_hidden_state = torch_to_ttnn(encoder_hidden_state, device=device)

            all_hidden_states += (encoder_hidden_state,)

        concated_tensor = ttnn.concat(all_hidden_states[::-1], dim=1)

        concated_tensor = ttnn.permute(concated_tensor, (0, 2, 3, 1))
        concated_tensor = tt_lib.tensor.interleaved_to_sharded(
            concated_tensor, self.linear_fuse.conv.input_sharded_memory_config
        )
        hidden_states = self.linear_fuse(concated_tensor)

        logits = self.classifier(hidden_states)
        logits = self.classifier.copy_output_from_device(logits)
        logits = ttnn.to_device(logits, device)
        logits = ttnn.permute(logits, (0, 3, 1, 2))
        # logits are of shape (batch_size, num_labels, height/4, width/4)

        return logits
