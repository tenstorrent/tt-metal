# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.segformer.tt.common import Conv


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
        device,
        hidden_states: ttnn.Tensor,
        height: int,
        width: int,
    ):
        if len(hidden_states.shape) == 3:
            batch_size, seq_len, num_channels = hidden_states.shape
        elif len(hidden_states.shape) == 4:
            batch_size, __, seq_len, num_channels = hidden_states.shape

        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.reshape(hidden_states, (batch_size, height, width, num_channels))

        hidden_states = self.dwconv(device, hidden_states)

        return hidden_states
