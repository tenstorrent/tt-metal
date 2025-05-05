# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import comp_pcc
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


from . import models as md


# List of models to test
model_dict = {
    1: [md.model_1.TestMatmulModel1TTNN, md.model_1.TestMatmulModel1, 2],
    2: [md.model_2.TestMatmulModel2TTNN, md.model_2.TestMatmulModel2, 2],
    3: [md.model_3.TestMatmulModel3TTNN, md.model_3.TestMatmulModel3, 4],
    4: [md.model_4.TestMatmulModel4TTNN, md.model_4.TestMatmulModel4, 4],
    5: [md.model_5.TestMatmulModel5TTNN, md.model_5.TestMatmulModel5, 13],
    6: [md.model_6.TestMatmulModel6TTNN, md.model_6.TestMatmulModel6, 6],
    7: [md.model_7.TestMatmulModel7TTNN, md.model_7.TestMatmulModel7, 4],
    8: [md.model_8.TestMatmulModel8TTNN, md.model_8.TestMatmulModel8, 6],
    9: [md.model_9.TestMatmulModel9TTNN, md.model_9.TestMatmulModel9, 10],
    10: [md.model_10.TestMatmulModel10TTNN, md.model_10.TestMatmulModel10, 6],
}

# List of data types to test
dtype_dict = {
    "float32": {"torch": torch.float32, "ttnn": ttnn.float32},
    # "bfloat16": ttnn.bfloat16
}

# List of shapes to test
shape_list = [
    # [2, 16, 256, 256],
    [2, 16, 32, 32],
    # [4, 32, 128, 128],
]


@pytest.mark.parametrize("dtype", dtype_dict.keys(), ids=dtype_dict.keys())
@pytest.mark.parametrize("shape", shape_list, ids=["x".join(map(str, shape)) for shape in shape_list])
@pytest.mark.parametrize("model", model_dict.keys(), ids=["model_" + str(item) for item in model_dict.keys()])
def test_run_matmul(
    dtype,
    shape,
    device,
    model,
):
    ttnn_model, torch_model, input_no = model_dict[model]
    # Create the input tensor and the model configuration
    ttnn_inputs = []
    torch_inputs = []
    for _ in range(input_no):
        data = torch.randn(shape, dtype=dtype_dict[dtype]["torch"])
        mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        ttnn_inputs.append(
            torch2tt_tensor(data, device, tt_memory_config=mem_config, tt_dtype=dtype_dict[dtype]["ttnn"])
        )
        torch_inputs.append(data)

    ttnn_out = ttnn_model()(*ttnn_inputs)
    torch_out = torch_model()(*torch_inputs)

    for i, item in enumerate(ttnn_out):
        out = tt2torch_tensor(item)
        passing, output = comp_pcc(out, torch_out[i], 0.9999)
        logger.info(output)
        assert passing
