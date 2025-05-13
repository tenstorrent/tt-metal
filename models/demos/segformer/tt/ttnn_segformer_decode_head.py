# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import ttnn
from models.demos.segformer.tt.common import Conv
from models.demos.segformer.tt.ttnn_segformer_mlp import TtSegformerMLP
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


def torch_to_ttnn(input, device, layout=ttnn.TILE_LAYOUT):
    input = ttnn.from_torch(input, ttnn.bfloat8_b)
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

        self.linear_fuse = Conv([1, 1, 0, 0], parameters=parameters["linear_fuse"], activation="relu", deallocate=False)

        self.classifier = Conv(
            [1, 1, 0, 0],
            parameters=parameters["classifier"],
        )

        self.config = config

    def __call__(self, device, encoder_hidden_states: ttnn.bfloat8_b, parameters) -> ttnn.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        concated_tensor = 0
        index = 0
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            height = width = int(math.sqrt(encoder_hidden_state.shape[-2]))
            encoder_hidden_state = mlp(device, encoder_hidden_state, parameters=parameters["linear_c"][index])
            encoder_hidden_state = ttnn.to_layout(encoder_hidden_state, layout=ttnn.ROW_MAJOR_LAYOUT)
            encoder_hidden_state = ttnn.reshape(encoder_hidden_state, (batch_size, height, width, -1))

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
            shard_spec = ttnn.ShardSpec(shard_grid, (shard_height, shard_width), shard_orientation)
            input_memory_config = ttnn.MemoryConfig(
                ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
            )
            encoder_hidden_state = ttnn.to_memory_config(encoder_hidden_state, memory_config=input_memory_config)

            encoder_hidden_state = ttnn.upsample(
                encoder_hidden_state,
                scale_factor=(128 // encoder_hidden_state.shape[2], 128 // encoder_hidden_state.shape[2]),
                mode="bilinear",
            )

            encoder_hidden_state_to_concat = ttnn.to_memory_config(
                encoder_hidden_state, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16
            )

            ttnn.deallocate(encoder_hidden_state)
            encoder_hidden_state_to_concat = ttnn.reallocate(encoder_hidden_state_to_concat)

            all_hidden_states += (encoder_hidden_state_to_concat,)

            index += 1

        concated_tensor = ttnn.concat(all_hidden_states[::-1], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(all_hidden_states[0])
        ttnn.deallocate(all_hidden_states[1])
        ttnn.deallocate(all_hidden_states[2])
        ttnn.deallocate(all_hidden_states[3])
        concated_tensor = ttnn.reallocate(concated_tensor)

        concated_tensor_tile = ttnn.to_layout(concated_tensor, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(concated_tensor)
        concated_tensor_tile = ttnn.reallocate(concated_tensor_tile)

        hidden_states, __, __ = self.linear_fuse(device, concated_tensor_tile)
        logits, __, __ = self.classifier(device, hidden_states)

        return logits
