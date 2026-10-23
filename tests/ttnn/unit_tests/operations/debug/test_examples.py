# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize("height", [64])
@pytest.mark.parametrize("width", [128])
def test_composite_example(device, height, width):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.composite_example(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("height", [64])
@pytest.mark.parametrize("width", [128])
@pytest.mark.parametrize("return_outputs", [[False, True], [True, False], [True, True]])
def test_composite_example_multiple_return(device, height, width, return_outputs):
    torch.manual_seed(0)

    return_output1, return_output2 = return_outputs

    # run torch
    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor

    # run TT
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output1, output2 = ttnn.composite_example_multiple_return(
        input_tensor, return_output1=return_output1, return_output2=return_output2
    )

    if return_output1:
        output_tensor1 = ttnn.to_torch(output1)
        assert_equal(torch_output_tensor, output_tensor1)
    else:
        assert output1 == None

    if return_output2:
        output_tensor2 = ttnn.to_torch(output2)
        assert_equal(torch_output_tensor, output_tensor2)
    else:
        assert output2 == None


@pytest.mark.parametrize("return_outputs", [[False, True], [True, False], [True, True]])
def test_composite_example_multiple_return_program_cache(device, return_outputs):
    torch.manual_seed(0)
    return_output1, return_output2 = return_outputs

    spacers, input_addresses = [], set()
    num_entries_before = device.num_program_cache_entries()
    for i in range(3):
        # A growing live allocation moves every tensor below to a new address on each cache hit, and fresh
        # data per iteration makes a stale address show up as a mismatch.
        spacers.append(ttnn.from_torch(torch.zeros((32, 32 * (i + 1))), layout=ttnn.TILE_LAYOUT, device=device))
        torch_input_tensor = torch.rand((64, 128), dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
        input_addresses.add(input_tensor.buffer_address())
        output1, output2 = ttnn.composite_example_multiple_return(
            input_tensor, return_output1=return_output1, return_output2=return_output2
        )
        for requested, output in ((return_output1, output1), (return_output2, output2)):
            if requested:
                assert_equal(torch_input_tensor, ttnn.to_torch(output))
            else:
                assert output == None

    assert len(input_addresses) > 1
    assert device.num_program_cache_entries() - num_entries_before == 1
