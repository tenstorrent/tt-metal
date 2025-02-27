# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

# shape, shard_layout, shard end core, shard shape, attention head dim
CROSS_DOWN_BLOCKS_HIDDEN_STATES_INFO = (
    ([2, 320, 64, 64], ttnn.TensorMemoryLayout.BLOCK_SHARDED, (4, 7), (1024, 64)),
    ([2, 640, 32, 32], ttnn.TensorMemoryLayout.BLOCK_SHARDED, (4, 7), (256, 128)),
    ([2, 1280, 16, 16], ttnn.TensorMemoryLayout.BLOCK_SHARDED, (7, 7), (64, 160)),
)

DOWN_MID_UP_BLOCKS_HIDDEN_STATES_INFO = ([2, 1280, 8, 8], ttnn.TensorMemoryLayout.BLOCK_SHARDED, (7, 3), (32, 160))

CROSS_UP_BLOCKS_HIDDEN_STATES_INFO = (
    ([2, 1280, 16, 16], ttnn.TensorMemoryLayout.BLOCK_SHARDED, (7, 7), (64, 160)),
    ([2, 640, 32, 32], ttnn.TensorMemoryLayout.BLOCK_SHARDED, (4, 7), (256, 128)),
    ([2, 320, 64, 64], ttnn.TensorMemoryLayout.BLOCK_SHARDED, (4, 7), (1024, 64)),
)

# hidden states info, attention head dim, block (up/down/min), block index, attention index
TRANSFORMER_PARAMETERIZATIONS = (
    (CROSS_DOWN_BLOCKS_HIDDEN_STATES_INFO[0] + (40, "down", 0, 0)),
    (CROSS_DOWN_BLOCKS_HIDDEN_STATES_INFO[0] + (40, "down", 0, 1)),
    (CROSS_DOWN_BLOCKS_HIDDEN_STATES_INFO[1] + (80, "down", 1, 0)),
    (CROSS_DOWN_BLOCKS_HIDDEN_STATES_INFO[1] + (80, "down", 1, 1)),
    (CROSS_DOWN_BLOCKS_HIDDEN_STATES_INFO[2] + (160, "down", 2, 0)),
    (CROSS_DOWN_BLOCKS_HIDDEN_STATES_INFO[2] + (160, "down", 2, 1)),
    (DOWN_MID_UP_BLOCKS_HIDDEN_STATES_INFO + (160, "mid", 0, 0)),
    (CROSS_UP_BLOCKS_HIDDEN_STATES_INFO[0] + (160, "up", 1, 0)),
    (CROSS_UP_BLOCKS_HIDDEN_STATES_INFO[0] + (160, "up", 1, 1)),
    (CROSS_UP_BLOCKS_HIDDEN_STATES_INFO[0] + (160, "up", 1, 2)),
    (CROSS_UP_BLOCKS_HIDDEN_STATES_INFO[1] + (80, "up", 2, 0)),
    (CROSS_UP_BLOCKS_HIDDEN_STATES_INFO[1] + (80, "up", 2, 1)),
    (CROSS_UP_BLOCKS_HIDDEN_STATES_INFO[1] + (80, "up", 2, 2)),
    (CROSS_UP_BLOCKS_HIDDEN_STATES_INFO[2] + (40, "up", 3, 0)),
    (CROSS_UP_BLOCKS_HIDDEN_STATES_INFO[2] + (40, "up", 3, 1)),
    (CROSS_UP_BLOCKS_HIDDEN_STATES_INFO[2] + (40, "up", 3, 2)),
)
