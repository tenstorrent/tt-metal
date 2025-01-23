# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pathlib
import pytest

import torch

import ttnn


@pytest.mark.parametrize("size", [64])
@pytest.mark.parametrize("mode", [ttnn.graph.RunMode.NO_DISPATCH, ttnn.graph.RunMode.NORMAL])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float, torch.bfloat16])
def test_convert_python_tensor(device, size, mode, dtype):
    torch.manual_seed(0)

    ttnn.graph.begin_graph_capture(mode)
    torch_input_tensor = torch.rand((size,), (dtype))
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.to_torch(input_tensor, torch_rank=1)
    captured_graph = ttnn.graph.end_graph_capture()
    calltrace = ttnn.graph.extract_calltrace(captured_graph)

    assert output_tensor == input_tensor
    assert "tt::tt_metal::detail::convert_python_tensor_to_tt_tensor" in calltrace
    assert captured_graph[0]["node_type"] == "capture_start"
    assert captured_graph[1]["node_type"] == "function_start"
    assert captured_graph[1]["params"]["name"] == "tt::tt_metal::detail::convert_python_tensor_to_tt_tensor"
    assert captured_graph[-2]["node_type"] == "buffer_deallocate"
    assert captured_graph[-1]["node_type"] == "capture_end"


@pytest.mark.parametrize("size", [64])
@pytest.mark.parametrize("mode", [ttnn.graph.RunMode.NO_DISPATCH, ttnn.graph.RunMode.NORMAL])
@pytest.mark.parametrize("dtype", [ttnn.bfloat4_b, ttnn.bfloat8_b])
def test_convert_python_tensor_bfp_b(device, size, mode, dtype):
    torch.manual_seed(0)

    ttnn.graph.begin_graph_capture(mode)
    torch_input_tensor = torch.rand((size,), torch.float)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=(dtype))
    output_tensor = ttnn.to_torch(input_tensor, torch_rank=1)
    captured_graph = ttnn.graph.end_graph_capture()
    calltrace = ttnn.graph.extract_calltrace(captured_graph)

    assert output_tensor == input_tensor
    assert "tt::tt_metal::detail::convert_python_tensor_to_tt_tensor" in calltrace
    assert captured_graph[0]["node_type"] == "capture_start"
    assert captured_graph[1]["node_type"] == "function_start"
    assert captured_graph[1]["params"]["name"] == "tt::tt_metal::detail::convert_python_tensor_to_tt_tensor"
    assert captured_graph[-2]["node_type"] == "buffer_deallocate"
    assert captured_graph[-1]["node_type"] == "capture_end"
