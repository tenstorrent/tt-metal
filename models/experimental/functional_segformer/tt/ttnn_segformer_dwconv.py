# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_segformer.tt.common import Conv


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
        # print("dw0", hidden_states.shape)

        if len(hidden_states.shape) == 3:
            batch_size, seq_len, num_channels = hidden_states.shape
        elif len(hidden_states.shape) == 4:
            batch_size, __, seq_len, num_channels = hidden_states.shape

        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(hidden_states, (batch_size, height, width, num_channels))

        # print("dw1", hidden_states.shape)
        hidden_states = self.dwconv(device, hidden_states)
        # print("dw2", hidden_states.shape)
        # hidden_states = ttnn.reshape(
        #     hidden_states,
        #     (hidden_states.shape[0], hidden_states.shape[1] * hidden_states.shape[2], hidden_states.shape[3]),
        # )
        # hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)
        # hidden_states = ttnn.to_device(hidden_states, device=device)
        # print("dw3", hidden_states.shape)

        return hidden_states
