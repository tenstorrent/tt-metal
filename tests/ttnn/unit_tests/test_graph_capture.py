# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn


@pytest.mark.skip(reason="Graph capture is causing other tests to fail")
@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("size", [64])
@pytest.mark.parametrize("mode", [ttnn.graph.GraphRunMode.FAKE, ttnn.graph.GraphRunMode.REAL])
def test_graph_capture(device, scalar, size, mode):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((size,), dtype=torch.bfloat16)

    ttnn.graph.begin_graph_capture(mode)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = input_tensor + scalar
    output_tensor = ttnn.to_torch(output_tensor, torch_rank=1)
    captured_graph = ttnn.graph.end_graph_capture()

    assert captured_graph[0]["name"] == "capture_start"
    assert captured_graph[1]["name"] == "begin_function"
    assert captured_graph[1]["params"]["name"] == "tt::tt_metal::detail::convert_python_tensor_to_tt_tensor"
    assert captured_graph[-1]["name"] == "buffer_deallocate"
