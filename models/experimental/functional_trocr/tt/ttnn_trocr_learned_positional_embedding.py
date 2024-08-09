# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn


def TtTrOCRPositionalEmbedding(
    input_ids: torch.Tensor,
    past_key_values_length: int = 0,
    parameters=None,
    device=None,
):
    offset = 2
    bsz, seq_len = input_ids.shape[:2]
    positions = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long).expand(bsz, -1)
    positions = positions + offset
    positions = ttnn.from_torch(positions, device=device, dtype=ttnn.ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.embedding(positions, weight=parameters.weight)
    output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
    output = ttnn.to_device(output, device)

    return output
