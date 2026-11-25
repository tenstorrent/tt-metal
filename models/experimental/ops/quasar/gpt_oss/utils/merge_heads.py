# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn


def merge_heads_on_device(input_tensor, *, num_heads, memory_config=None, output_tensor=None, sub_core_grids=None):
    """Do what ttnn.experimental.nlp_concat_heads_decode does, with device ops that do not place one head per core.

    Takes [1, batch, heads, head_dim] in TILE layout and returns [1, 1, batch padded to a tile row,
    num_heads * head_dim] in TILE layout, interleaved in DRAM. In row-major layout the heads of each user are
    contiguous, so a reshape merges them into the hidden dimension. The heads dimension of the input must be
    num_heads: unlike nlp_concat_heads_decode, this function does not remove padding of the heads dimension.
    """
    if memory_config is not None or output_tensor is not None or sub_core_grids is not None:
        raise NotImplementedError(
            "merge_heads_on_device does not support memory_config, output_tensor or sub_core_grids"
        )
    quasar = ttnn.experimental.quasar
    _, batch, heads, head_dim = input_tensor.shape
    if heads != num_heads:
        raise NotImplementedError(f"merge_heads_on_device needs {num_heads} heads in the input, got {heads}")
    x = input_tensor
    if x.is_sharded():
        x = quasar.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
    x = quasar.untilize(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.reshape(x, (1, 1, batch, num_heads * head_dim))
    padded_batch = ((batch + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    if padded_batch != batch:
        # The mainline pad: the row-major pad of ttnn.experimental.quasar fails on Quasar (tenstorrent/tt-metal#55526).
        x = ttnn.pad(x, [(0, 0), (0, 0), (0, padded_batch - batch), (0, 0)], value=0.0)
    return quasar.tilize(x, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
