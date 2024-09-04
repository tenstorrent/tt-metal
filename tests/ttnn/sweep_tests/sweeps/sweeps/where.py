# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from models.utility_functions import torch_random


parameters = {
    "batch_sizes": [(1,)],
    "height": [384, 1024],
    "width": [1024, 4096],
    "input_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "input_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    "value1": [5.5, 15.8],
    "value2": [3.7, 12.3],
    "type": ["TensorTensorTensor", "TensorTensorScalar", "TensorScalarTensor", "TensorScalarScalar"],
}


def skip(
    batch_sizes,
    height,
    width,
    input_dtype,
    input_memory_config,
    output_memory_config,
    layout,
    value1,
    value2,
    type,
) -> Tuple[bool, Optional[str]]:
    if input_dtype == ttnn.bfloat8_b or layout == ttnn.ROW_MAJOR_LAYOUT:
        return True, "Skipped as BFLOAT8_B or ROW_MAJOR_LAYOUT not supported"
    return False, None


def run(
    batch_sizes,
    height,
    width,
    input_dtype,
    input_memory_config,
    output_memory_config,
    layout,
    value1,
    value2,
    type,
    *,
    device,
) -> Tuple[bool, Optional[str]]:
    input_shape = (*batch_sizes, height, width)

    torch_predicate = torch.rand(input_shape, dtype=torch.bfloat16)
    predicate = ttnn.from_torch(
        torch_predicate, dtype=input_dtype, device=device, layout=layout, memory_config=input_memory_config
    )
    if type == "TensorTensorTensor":
        torch_true_value = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_false_value = torch.rand(input_shape, dtype=torch.bfloat16)
        true_value = ttnn.from_torch(
            torch_true_value, dtype=input_dtype, device=device, layout=layout, memory_config=input_memory_config
        )
        false_value = ttnn.from_torch(
            torch_false_value, dtype=input_dtype, device=device, layout=layout, memory_config=input_memory_config
        )
    elif type == "TensorTensorScalar":
        torch_true_value = torch.rand(input_shape, dtype=torch.bfloat16)
        torch_false_value = value2
        true_value = ttnn.from_torch(
            torch_true_value, dtype=input_dtype, device=device, layout=layout, memory_config=input_memory_config
        )
        false_value = torch_false_value
    elif type == "TensorScalarTensor":
        torch_true_value = value1
        torch_false_value = torch.rand(input_shape, dtype=torch.bfloat16)
        true_value = torch_true_value
        false_value = ttnn.from_torch(
            torch_false_value, dtype=input_dtype, device=device, layout=layout, memory_config=input_memory_config
        )
    elif type == "TensorScalarScalar":
        torch_true_value = value1
        torch_false_value = value2
        true_value = torch_true_value
        false_value = torch_false_value

    torch_output_tensor = torch.where(torch_predicate.to(torch.bool), torch_true_value, torch_false_value)

    output_tensor = ttnn.where(predicate, true_value, false_value, memory_config=output_memory_config)
    output_tensor = ttnn.to_torch(output_tensor)

    return check_with_pcc(torch_output_tensor, output_tensor)
