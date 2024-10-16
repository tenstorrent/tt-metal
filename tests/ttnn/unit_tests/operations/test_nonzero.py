# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import numpy as np
import ttnn

from tests.ttnn.utils_for_testing import assert_equal


tt_dtype_to_torch_dtype = {
    ttnn.uint16: torch.int16,
    ttnn.uint32: torch.int32,
    ttnn.float32: torch.float,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.float,
}


@pytest.mark.parametrize(
    "seed",
    [
        0,
        42,
        13,
        9,
    ],
)
@pytest.mark.parametrize(
    "B, b",
    [
        (16, 3),
        (32, 6),
    ],
)
@pytest.mark.parametrize(
    "tt_dtype",
    [
        ttnn.bfloat16,
        ttnn.uint32,
    ],
)
def test_indexed_slice(seed, B, b, tt_dtype, device):
    torch.manual_seed(seed)

    dtype = tt_dtype_to_torch_dtype[tt_dtype]
    zero_tensor = torch.zeros(1, 1, 1, b, dtype=dtype)
    if tt_dtype == ttnn.uint32:
        non_zero_tensor = torch.randint(1, 100, (1, 1, 1, B - b), dtype=dtype)
    else:
        non_zero_tensor = torch.rand((1, 1, 1, B - b), dtype=dtype)
        non_zero_tensor.add(1, alpha=1)
    torch_input_tensor = torch.concat((zero_tensor, non_zero_tensor), dim=3)
    torch_input_tensor = torch_input_tensor[torch.randperm(torch_input_tensor.size()[2])]

    input_tt = ttnn.from_torch(torch_input_tensor, dtype=tt_dtype, device=device)

    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    num_indices_tt, indices_tt = ttnn.nonzero(input_tt, memory_config=mem_config)
    torch_num_indices = num_indices_tt.cpu().to_torch()
    torch_indices = indices_tt.cpu().to_torch()

    num_non_zeros = torch_num_indices[0, 0, 0, 0].item()
    assert num_non_zeros == B - b

    a_pt = (
        ttnn.slice(indices_tt, (0, 0, 0, 0), (1, 1, 1, num_non_zeros), memory_config=mem_config)
        .cpu()
        .to(ttnn.ROW_MAJOR_LAYOUT)
        .to_torch()
    )

    golden_output = torch.arange(start=b, end=B, step=1, dtype=torch.int32)
    golden_output = torch.reshape(golden_output, (1, 1, 1, B - b))
    assert torch.allclose(golden_output, a_pt)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 1, 31])),
        (torch.Size([1, 1, 1, 63])),
    ),
)
def test_nonzero(input_shapes, device):
    torch.manual_seed(0)

    torch_input_tensor = torch.ones(input_shapes)
    torch_input_tensor[..., ::2] = 0

    torch_output_tensor = torch.nonzero(torch_input_tensor, as_tuple=True)
    torch_output_tensor = torch_output_tensor[3].unsqueeze(0).unsqueeze(0).unsqueeze(0)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_tensor = ttnn.nonzero(input_tensor)

    output_tensor1 = ttnn.to_layout(output_tensor[0], ttnn.ROW_MAJOR_LAYOUT)
    output_tensor1 = ttnn.from_device(output_tensor1)
    output_tensor1 = ttnn.to_torch(output_tensor1)
    no_of_non_zero_indices = output_tensor1[..., 0].item()

    output_tensor2 = ttnn.to_layout(output_tensor[1], ttnn.ROW_MAJOR_LAYOUT)
    output_tensor2 = ttnn.from_device(output_tensor2)
    output_tensor2 = ttnn.to_torch(output_tensor2)
    tt_output_tensor = output_tensor2[:, :, :, :no_of_non_zero_indices]

    assert_equal(torch_output_tensor, tt_output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 1, 32])),),
)
def test_nonzero_callback(input_shapes, device, use_program_cache):
    torch.manual_seed(0)
    num_program_cache_entries_list = []

    for i in range(2):
        torch_input_tensor = torch.ones(input_shapes)
        torch_input_tensor[..., ::2] = 0

        torch_output_tensor = torch.nonzero(torch_input_tensor, as_tuple=True)
        torch_output_tensor = torch_output_tensor[3].unsqueeze(0).unsqueeze(0).unsqueeze(0)

        input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        output_tensor = ttnn.nonzero(input_tensor)

        output_tensor1 = ttnn.to_layout(output_tensor[0], ttnn.ROW_MAJOR_LAYOUT)
        output_tensor1 = ttnn.from_device(output_tensor1)
        output_tensor1 = ttnn.to_torch(output_tensor1)
        no_of_non_zero_indices = output_tensor1[..., 0].item()

        output_tensor2 = ttnn.to_layout(output_tensor[1], ttnn.ROW_MAJOR_LAYOUT)
        output_tensor2 = ttnn.from_device(output_tensor2)
        output_tensor2 = ttnn.to_torch(output_tensor2)
        tt_output_tensor = output_tensor2[:, :, :, :no_of_non_zero_indices]

        assert_equal(torch_output_tensor, tt_output_tensor)
        torch_dummy = torch.randn([32, 32])
        ttnn_dummy = ttnn.from_torch(torch_dummy, device=device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]
