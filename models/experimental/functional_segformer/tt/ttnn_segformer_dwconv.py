# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_segformer.tt.common import Conv


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


class TtSegformerDWConv:
    def __init__(self, parameters, dim):
        super().__init__()
        if dim == 1024:
            self.dwconv = Conv(
                [1, 1, 1, 1],
                parameters=parameters["dwconv"],
                groups=dim,
                height_sharding=False,
                act_block_h=32,
                dtype=ttnn.bfloat8_b,
            )
        elif dim == 640:
            self.dwconv = Conv(
                [1, 1, 1, 1],
                parameters=parameters["dwconv"],
                groups=dim,
                height_sharding=False,
                act_block_h=32,
                dtype=ttnn.bfloat8_b,
            )
        else:
            self.dwconv = Conv([1, 1, 1, 1], parameters=parameters["dwconv"], groups=dim, dtype=ttnn.bfloat8_b)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        height: int,
        width: int,
        device,
    ):
        batch_size, seq_len, num_channels = hidden_states.shape

        """hidden_states = ttnn.permute(hidden_states, (0, 2, 1))
        hidden_states = ttnn.from_device(hidden_states)
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(hidden_states, (batch_size, hidden_states.shape[0], hidden_states.shape[1]))
        hidden_states = ttnn.reshape(hidden_states, (batch_size, num_channels, height, width))
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        hidden_states = ttnn.to_device(hidden_states, device=device)

        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)

        hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))"""

        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(hidden_states, (batch_size, height, width, num_channels))

        hidden_states = self.dwconv(device, hidden_states)

        """hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)
        hidden_states = ttnn.to_device(hidden_states, device=device)

        hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2))

        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.from_device(hidden_states)

        hidden_states = ttnn.reshape(
            hidden_states,
            (hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2] * hidden_states.shape[3]),
        )
        hidden_states = ttnn.to_device(hidden_states, device)
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)

        hidden_states = ttnn.permute(hidden_states, (0, 2, 1))
        hidden_states = ttnn.reshape(hidden_states, (batch_size, hidden_states.shape[0], hidden_states.shape[1]))
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)"""

        hidden_states = ttnn.reshape(
            hidden_states,
            (hidden_states.shape[0], hidden_states.shape[1] * hidden_states.shape[2], hidden_states.shape[3]),
        )
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)
        hidden_states = ttnn.to_device(hidden_states, device=device)
        return hidden_states
