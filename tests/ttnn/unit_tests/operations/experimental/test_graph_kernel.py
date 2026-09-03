# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize("num_inputs", [1, 2, 4])
@pytest.mark.parametrize("shape", [(32, 32), (2, 64, 128), (1, 1, 256, 64)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_graph_kernel_basis_copies_first_input(device, num_inputs, shape, layout):
    torch_inputs = [torch.randn(shape, dtype=torch.bfloat16) for _ in range(num_inputs)]
    inputs = [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device) for t in torch_inputs]

    output = ttnn.graph_kernel(inputs, "identity(in0)")

    assert output.shape == inputs[0].shape
    assert output.dtype == inputs[0].dtype
    assert output.layout == inputs[0].layout
    assert_equal(torch_inputs[0], ttnn.to_torch(output))


def test_graph_kernel_text_changes_program_hash(device):
    torch_input = torch.randn((32, 32), dtype=torch.bfloat16)
    x = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    before = device.num_program_cache_entries()
    ttnn.graph_kernel([x], "a")
    ttnn.graph_kernel([x], "a")
    ttnn.graph_kernel([x], "b")
    assert device.num_program_cache_entries() - before == 2


def test_graph_kernel_rejects_empty_inputs(device, expect_error):
    with expect_error(RuntimeError, "at least one input tensor is required"):
        ttnn.graph_kernel([], "empty")
