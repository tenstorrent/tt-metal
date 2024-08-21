# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc


parameters = {
    "W": [1, 8],
    "Z": [1, 8],
    "Y": [1, 31, 32, 33, 64],
    "X": [2, 30, 32, 34, 64, 66, 8192],
    "layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    "dtype_in": [ttnn.bfloat16, ttnn.bfloat8_b],
    "dtype_out": [ttnn.bfloat16, ttnn.bfloat8_b],
}


def skip(W, Z, Y, X, layout, dtype_in, dtype_out):
    if dtype_in != ttnn.bfloat16 or dtype_out != ttnn.bfloat16:
        return True, "(Un)tilizing works with the bfloat16 dtype only"

    if Y % 32 == 0 and X == 8192 and layout == ttnn.TILE_LAYOUT:
        return True, "Tilize without padding cannot handle large input rows"

    return False, None


def run(W, Z, Y, X, layout, dtype_in, dtype_out, *, device):
    torch_input_tensor = torch.randn((W, Z, Y, X), dtype=torch.float32)
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=dtype_in,
        device=device,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # can't specify dtype when converting to row major layout
    if layout == ttnn.TILE_LAYOUT:
        output = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    else:
        output = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype_out)
    output = ttnn.to_torch(output)

    return check_with_pcc(torch_input_tensor, output, 0.999)
