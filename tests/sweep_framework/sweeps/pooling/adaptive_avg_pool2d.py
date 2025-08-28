# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
import random
import ttnn
from tests.sweep_framework.sweep_utils.adaptive_pool2d_common import run_adaptive_avg_pool2d, invalidate_vector

random.seed(0)

parameters = {
    "adaptive_avg_pool2d_suite": {
        "input_shape": [
            [1, 64, 224, 224],
            [1, 128, 112, 112],
            [2, 64, 224, 224],
            [1, 256, 56, 56],
        ],
        "output_size": [
            [1, 1],
            [7, 7],
            [14, 14],
        ],
        "input_a_dtype": [ttnn.bfloat16],
        "input_a_layout": [ttnn.TILE_LAYOUT],
        "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
        "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    },
}


def run(
    input_shape,
    output_size,
    input_a_dtype,
    input_a_layout,
    input_a_memory_config,
    output_memory_config,
    *,
    device,
) -> list:
    [N, C, H, W] = input_shape
    [out_h, out_w] = output_size

    return run_adaptive_avg_pool2d(
        in_n=N,
        in_c=C,
        in_h=H,
        in_w=W,
        out_h=out_h,
        out_w=out_w,
        dtype=input_a_dtype,
        device=device,
        memory_config=output_memory_config,
        layout=input_a_layout,
    )
