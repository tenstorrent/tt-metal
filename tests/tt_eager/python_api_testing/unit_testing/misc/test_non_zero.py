# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import ttnn
from tests.ttnn.utils_for_testing import tt_dtype_to_torch_dtype


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

    if tt_dtype == ttnn.uint32:
        dtype = torch.int32
    else:
        dtype = tt_dtype_to_torch_dtype[tt_dtype]

    zero_tensor = torch.zeros(1, 1, 1, b, dtype=dtype)
    if tt_dtype == ttnn.uint32:
        non_zero_tensor = torch.randint(1, 100, (1, 1, 1, B - b), dtype=dtype)
    else:
        non_zero_tensor = torch.rand((1, 1, 1, B - b), dtype=dtype)
        non_zero_tensor.add(1, alpha=1)
    torch_input_tensor = torch.concat((zero_tensor, non_zero_tensor), dim=3)

    input_tt = ttnn.Tensor(torch_input_tensor, tt_dtype).to(device)

    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    output = ttnn.nonzero(input_tt, memory_config=mem_config)

    count_tensor = ttnn.to_torch(ttnn.from_device(output[0]))
    num_non_zeros = int(count_tensor[0, 0, 0, 0].item())
    assert num_non_zeros == B - b

    indices_tensor = ttnn.to_torch(ttnn.from_device(output[1]))

    # output[1] has shape [1, 1, 1, N*4]: N packed (b,n,h,c) uint32 4-tuples.
    # Flatten across the page dims so the slice works for any page layout.
    flat_coords = indices_tensor.reshape(-1)
    tt_coords = flat_coords[: num_non_zeros * 4].reshape(num_non_zeros, 4).int()

    ref_coords = torch.nonzero(torch_input_tensor.float(), as_tuple=False).int()

    assert torch.equal(tt_coords, ref_coords)
