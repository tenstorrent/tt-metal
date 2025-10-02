# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout
from models.common.utility_functions import is_grayskull, is_blackhole, torch_random

ttnn_to_torch_dtype = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}


@pytest.mark.parametrize("width", [32, 64])
@pytest.mark.parametrize("val", [0.00001])
@pytest.mark.parametrize(
    "from_to", [(ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT), (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT)]
)
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_to_layout(device, width, from_to, dtype, val):
    from_layout, to_layout = from_to
    torch_input = torch.tensor(val, dtype=ttnn_to_torch_dtype[dtype])

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        device=device,
        layout=from_layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    assert input_tensor.layout == from_layout

    output_tensor = ttnn.to_layout(input_tensor, to_layout)
    assert output_tensor.layout == to_layout

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_input, output_tensor)
    assert torch.allclose(torch_input, output_tensor)
