# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc, start_measuring_time, stop_measuring_time
from models.utility_functions import torch_random
from tests.sweep_framework.sweep_utils.roofline_utils import get_run_return

TIMEOUT = 15

parameters = {
    "default": {
        "params": [
            ((1, 1, 1024), [1024], 1e-05),
            ((1, 1, 768), [768], 1e-05),
            ((1, 10, 768), [768], 1e-05),
            ((1, 1024, 160), [160], 1e-05),
            ((1, 1024), [1024], 1e-12),
            ((1, 12, 128), [128], 1e-12),
            ((1, 12, 768), [768], 1e-12),
            ((1, 1200, 320), [320], 1e-05),
            ((1, 1370, 1280), [1280], 1e-06),
            ((1, 14, 128), [128], 1e-12),
            ((1, 14, 14, 1024), [1024], 1e-05),
            ((1, 14, 14, 384), [384], 1e-05),
            ((1, 14, 14, 512), [512], 1e-05),
            ((1, 14, 14, 768), [768], 1e-05),
            ((1, 14, 768), [768], 1e-12),
            ((1, 1445, 192), [192], 1e-12),
            ((1, 1500, 768), [768], 1e-05),
            ((1, 16, 16, 384), [384], 1e-05),
            ((1, 16, 16, 512), [512], 1e-05),
            ((1, 16, 768), [768], 1e-12),
            ((1, 16384, 32), [32], 1e-05),
            ((1, 19, 1024), [1024], 1e-05),
            ((1, 19200, 64), [64], 1e-05),
            ((1, 196, 768), [768], 1e-06),
            ((1, 197, 1024), [1024], 1e-06),
            ((1, 197, 1024), [1024], 1e-12),
            ((1, 197, 768), [768], 1e-06),
            ((1, 197, 768), [768], 1e-12),
            ((1, 2048, 768), [768], 1e-05),
            ((1, 24, 768), [768], 1e-05),
            ((1, 25, 768), [768], 1e-12),
            ((1, 256, 1024), [1024], 1e-12),
            ((1, 256, 1280), [1280], 1e-05),
            ((1, 256, 160), [160], 1e-05),
            ((1, 256, 256), [256], 1e-05),
            ((1, 256, 32), [32], 1e-05),
            ((1, 256, 512), [512], 1e-05),
            ((1, 256, 64), [64], 1e-05),
            ((1, 28, 28, 192), [192], 1e-05),
            ((1, 28, 28, 256), [256], 1e-05),
            ((1, 28, 28, 384), [384], 1e-05),
            ((1, 28, 28, 512), [512], 1e-05),
            ((1, 300, 128), [128], 1e-05),
            ((1, 300, 320), [320], 1e-05),
            ((1, 300, 512), [512], 1e-05),
            ((1, 300, 64), [64], 1e-05),
            ((1, 32, 1536), [1536], 1e-05),
            ((1, 32, 32, 192), [192], 1e-05),
            ((1, 32, 32, 256), [256], 1e-05),
            ((1, 4, 768), [768], 1e-05),
            ((1, 4096, 320), [320], 1e-05),
            ((1, 4096, 64), [64], 1e-05),
            ((1, 45, 768), [768], 1e-05),
            ((1, 4800, 128), [128], 1e-05),
            ((1, 5, 1024), [1024], 1e-05),
            ((1, 50, 1024), [1024], 1e-06),
            ((1, 50, 768), [768], 1e-05),
            ((1, 50, 768), [768], 1e-06),
            ((1, 56, 56, 128), [128], 1e-05),
            ((1, 56, 56, 96), [96], 1e-05),
            ((1, 59, 1024), [1024], 1e-05),
            ((1, 64, 64, 128), [128], 1e-05),
            ((1, 64, 64, 96), [96], 1e-05),
            ((1, 7, 4544), [4544], 1e-05),
            ((1, 7, 7, 1024), [1024], 1e-05),
            ((1, 7, 7, 1536), [1536], 1e-05),
            ((1, 7, 7, 2048), [2048], 1e-05),
            ((1, 7, 7, 768), [768], 1e-05),
            ((1, 7, 768), [768], 1e-05),
            ((1, 768), [768], 1e-05),
            ((1, 768), [768], 1e-12),
            ((1, 8, 768), [768], 1e-12),
            ((1, 8, 8, 1024), [1024], 1e-05),
            ((1, 8, 8, 768), [768], 1e-05),
            ((1, 9, 1024), [1024], 1e-12),
            ((1, 9, 128), [128], 1e-12),
            ((1, 9, 2048), [2048], 1e-12),
            ((1, 9, 4096), [4096], 1e-12),
            ((1, 9, 768), [768], 1e-12),
            ((1, 100, 1280), [1280], 1e-05),
            ((1, 100, 640), [640], 1e-05),
            ((1, 500, 1280), [1280], 1e-05),
            ((1, 500, 320), [320], 1e-05),
            ((1, 500, 640), [640], 1e-05),
            ((100, 1, 256), [256], 1e-05),
            ((2, 7, 512), [512], 1e-05),
            ((920, 1, 256), [256], 1e-05),
        ],
    }
}


def run_layer_norm(device, params):
    [input_shape, normalized_shape, eps] = params
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    torch_weight_tensor = torch.rand(normalized_shape, dtype=torch.float32)
    torch_bias_tensor = torch.rand(normalized_shape, dtype=torch.float32)
    torch_output_tensor = torch.layer_norm(
        torch_input_tensor, normalized_shape, weight=torch_weight_tensor, bias=torch_bias_tensor, eps=eps
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    weight_tensor = ttnn.from_torch(torch_weight_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    bias_tensor = ttnn.from_torch(torch_bias_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    start_time = start_measuring_time()
    op_output_tensor = ttnn.layer_norm(input_tensor, weight=weight_tensor, bias=bias_tensor, epsilon=eps)
    output_tensor = ttnn.to_torch(op_output_tensor)
    e2e_perf = stop_measuring_time(start_time)
    expected_pcc = 0.999
    tensors = [input_tensor, weight_tensor, bias_tensor, op_output_tensor]
    return get_run_return(torch_output_tensor, output_tensor, expected_pcc, tensors, e2e_perf)


@pytest.mark.parametrize("params", parameters["default"]["params"])
def test_layer_norm(device, params):
    run_layer_norm(device, params)


def run(
    params,
    *,
    device,
) -> list:
    return run_layer_norm(device, params)
