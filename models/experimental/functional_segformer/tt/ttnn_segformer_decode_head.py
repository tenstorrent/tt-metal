# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import ttnn
from models.experimental.functional_segformer.tt.ttnn_segformer_mlp import TtSegformerMLP
from torch import nn
import tt_lib
from models.experimental.functional_segformer.tt.common import Conv
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


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
            mlp = TtSegformerMLP()
            mlps.append(mlp)
        self.linear_c = mlps

        self.linear_fuse = Conv(
            [1, 1, 0, 0],
            parameters=parameters["linear_fuse"],
            activation="relu",
        )

        self.classifier = Conv(
            [1, 1, 0, 0],
            parameters=parameters["classifier"],
        )

        self.config = config

    def __call__(self, encoder_hidden_states: ttnn.bfloat16, parameters) -> ttnn.Tensor:
        device = encoder_hidden_states[-1].device()
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        index = 0
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            # This condition was the original PyTorch graph with the redundant TMs to unfold/fold/unfold
            if 0:
                if self.config.reshape_last_stage is False and (encoder_hidden_state.shape) == 3:
                    height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                    encoder_hidden_state = ttnn.reshape(encoder_hidden_state, (batch_size, height, width, -1))
                    encoder_hidden_state = ttnn.permute(encoder_hidden_state, (0, 3, 1, 2))
                height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            else:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-2]))

            # print("iii", index)
            # print("ee0", encoder_hidden_state.shape)
            encoder_hidden_state = mlp(encoder_hidden_state, parameters=parameters["linear_c"][index])

            # print("ee1", encoder_hidden_state.shape)
            encoder_hidden_state = ttnn.to_layout(encoder_hidden_state, layout=ttnn.ROW_MAJOR_LAYOUT)
            encoder_hidden_state = ttnn.reshape(encoder_hidden_state, (batch_size, height, width, -1))

            ## Original TM but I replaced it with one reshape in the previous line.
            ## To revisit to investigate on the low PCC
            """
            if len(encoder_hidden_state.shape) == 3:
                encoder_hidden_state = ttnn.permute(encoder_hidden_state, (0, 2, 1))
            else:
                encoder_hidden_state = ttnn.permute(encoder_hidden_state, (0, 1, 3, 2))
            print("ee2", encoder_hidden_state.shape)
            encoder_hidden_state = ttnn.from_device(encoder_hidden_state)
            encoder_hidden_state = ttnn.to_layout(encoder_hidden_state, layout=ttnn.ROW_MAJOR_LAYOUT)
            encoder_hidden_state = ttnn.reshape(encoder_hidden_state, (batch_size, -1, height, width))
            encoder_hidden_state = ttnn.to_device(encoder_hidden_state, device)
            print("ee3", encoder_hidden_state.shape)
            encoder_hidden_state = ttnn.permute(encoder_hidden_state, (0, 2, 3, 1))
            print("ee4", encoder_hidden_state.shape)
            """

            if encoder_hidden_state.shape[-2] == 16:
                ncores = 16
            elif encoder_hidden_state.shape[-2] == 32:
                ncores = 32
            else:
                ncores = 64

            shard_grid = get_shard_grid_from_num_cores(ncores, device)
            shard_orientation = ttnn.ShardOrientation.ROW_MAJOR

            shard_height = math.ceil(
                encoder_hidden_state.shape[0] * encoder_hidden_state.shape[1] * encoder_hidden_state.shape[2] / ncores
            )
            shard_width = encoder_hidden_state.shape[3]
            shard_spec = ttnn.ShardSpec(shard_grid, (shard_height, shard_width), shard_orientation, False)
            input_memory_config = ttnn.MemoryConfig(
                ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
            )
            encoder_hidden_state = ttnn.to_memory_config(encoder_hidden_state, memory_config=input_memory_config)

            # print(shard_spec)
            # print("up", encoder_hidden_state.shape)
            encoder_hidden_state = ttnn.upsample(
                encoder_hidden_state,
                scale_factor=(128 // encoder_hidden_state.shape[2], 128 // encoder_hidden_state.shape[2], 1),
                mode="bilinear",
            )

            encoder_hidden_state = ttnn.to_memory_config(
                encoder_hidden_state, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
            )
            # print("up2", encoder_hidden_state.shape)
            # encoder_hidden_state = ttnn.permute(encoder_hidden_state, (0, 3, 1, 2))
            # print("up3", encoder_hidden_state.shape)

            # all_hidden_states += (encoder_hidden_state,)
            # encoder_hidden_state = ttnn.to_layout(encoder_hidden_state, ttnn.TILE_LAYOUT)
            if index == 0:
                concated_tensor = encoder_hidden_state
            else:
                concated_tensor = ttnn.concat(
                    [concated_tensor, encoder_hidden_state], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG
                )
                ttnn.deallocate(encoder_hidden_state)
                ttnn.reallocate(concated_tensor)
            # print("conc", index, concated_tensor.shape)
            index += 1

        # Replaced with the phased concat of each 2 inputs, to handle the L1 OOM
        ###concated_tensor = ttnn.concat(all_hidden_states[::-1], dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

        # ttnn.deallocate(encoder_hidden_state)
        # concated_tensor = ttnn.to_memory_config(concated_tensor, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
        # ttnn.reallocate(concated_tensor)

        # print("conc", concated_tensor.shape)

        # concated_tensor = ttnn.to_layout(concated_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        # concated_tensor = ttnn.permute(concated_tensor, (0, 3, 1, 2))
        # concated_tensor = ttnn.reshape(concated_tensor, (concated_tensor[0], concated_tensor[-1], concated_tensor[1], concated_tensor[2]) )

        ttnn.reallocate(concated_tensor)
        # print("conc_p", concated_tensor.shape)

        hidden_states, __, __ = self.linear_fuse(device, concated_tensor)
        # print("c1", hidden_states.shape)

        logits, __, __ = self.classifier(device, hidden_states)
        logits_shape = logits.shape
        logits = ttnn.to_device(logits, device=device)
        logits = ttnn.to_memory_config(logits, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        logits = ttnn.to_layout(logits, layout=ttnn.ROW_MAJOR_LAYOUT)
        # print("logits", logits.shape)
        h = w = int(math.sqrt(logits.shape[-2]))
        logits = ttnn.reshape(logits, (logits_shape[0], h, w, logits_shape[-1]))
        # print(logits.shape)
        logits = ttnn.permute(logits, (0, 3, 1, 2))
        logits = ttnn.to_layout(logits, layout=ttnn.ROW_MAJOR_LAYOUT)
        logits = logits[:, :150, :, :]  # returns out_channel 160 instead of 150
        logits = ttnn.to_layout(logits, layout=ttnn.TILE_LAYOUT)
        # logits are of shape (batch_size, num_labels, height/4, width/4)

        return logits
