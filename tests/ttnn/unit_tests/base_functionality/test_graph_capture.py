# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pathlib
import pytest
import torch
import ttnn
from models.common.utility_functions import is_watcher_enabled
from ttnn.graph_tracer_utils import GraphTracerUtils


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("size", [64])
@pytest.mark.parametrize("mode", [ttnn.graph.RunMode.NO_DISPATCH, ttnn.graph.RunMode.NORMAL])
def test_graph_capture(tmp_path, device, scalar, size, mode):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((size,), dtype=torch.bfloat16)

    ttnn.graph.begin_graph_capture(mode)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = input_tensor + scalar
    output_tensor = ttnn.to_torch(output_tensor, torch_rank=1)
    captured_graph = ttnn.graph.end_graph_capture()
    calltrace = ttnn.graph.extract_calltrace(captured_graph)

    assert "ttnn::convert_python_tensor_to_tt_tensor" in calltrace
    assert captured_graph[0]["node_type"] == "capture_start"
    assert captured_graph[1]["node_type"] == "function_start"
    assert captured_graph[1]["params"]["name"] == "ttnn::convert_python_tensor_to_tt_tensor"
    assert captured_graph[-2]["node_type"] == "buffer_deallocate"
    assert captured_graph[-1]["node_type"] == "capture_end"

    ttnn.graph.pretty_print(captured_graph)

    ttnn.graph.visualize(captured_graph, file_name=tmp_path / pathlib.Path("graph.svg"))


def test_graph_capture_with_all_parameters(device):
    if is_watcher_enabled():
        pytest.skip("Skipping due to failure with watcher enabled, github issue #37096")
    # Create input tensor
    torch_input = torch.rand((1, 1, 2048, 512), dtype=torch.bfloat16)

    # TT operations
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_input = tt_input.reshape(1, 2048, 4, 128)
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    ttnn.transpose(tt_input, 1, 2)
    captured_graph = ttnn.graph.end_graph_capture()

    # Note: High-level function tracing (ttnn::transpose) was removed from decorators.hpp
    # Now only device operations are captured. Find PermuteDeviceOperation
    permute_op = None
    for node in captured_graph:
        if node.get("node_type") == "function_start" and node.get("params", {}).get("name") == "PermuteDeviceOperation":
            permute_op = node
            break

    assert permute_op is not None, "PermuteDeviceOperation should be in the captured graph"

    # PermuteDeviceOperation arguments
    node_permute = permute_op["arguments"]
    assert (
        node_permute[0]
        == "[ unsupported type , std::reference_wrapper<ttnn::operations::data_movement::PermuteDeviceOperation::operation_attributes_t const>]"
    )
    assert (
        node_permute[1]
        == "[ unsupported type , std::reference_wrapper<std::vector<std::reference_wrapper<tt::tt_metal::Tensor const>, std::allocator<std::reference_wrapper<tt::tt_metal::Tensor const> > > >]"
    )

    # tt::tt_metal::create_device_tensor
    create_tensor_op = None
    for node in captured_graph:
        if (
            node.get("node_type") == "function_start"
            and node.get("params", {}).get("name") == "tt::tt_metal::create_device_tensor"
        ):
            # Find the one that creates the output tensor (should have the correct shape)
            args = node.get("arguments", [])
            if len(args) > 0 and "Shape([1, 4, 2048, 128])" in str(args[0]):
                create_tensor_op = node
                break

    assert create_tensor_op is not None, "create_device_tensor should be in the captured graph"
    node5 = create_tensor_op["arguments"]
    assert node5[0] == "Shape([1, 4, 2048, 128])"
    assert node5[1] == "DataType::BFLOAT16"
    assert node5[2] == "Layout::ROW_MAJOR"
    assert node5[3].isnumeric()
    assert node5[0] == "Shape([1, 4, 2048, 128])"
    assert node5[1] == "DataType::BFLOAT16"


def test_graph_capture_without_memory_config(device):
    # Create input tensor
    input_shape = (1, 1, 1, 32)
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

    # TT operations
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    tt_other = ttnn.from_torch(
        torch_input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    tt_out = ttnn.operations.moreh.dot(tt_input, tt_other, dtype=ttnn.bfloat16, output=None)
    captured_graph = ttnn.graph.end_graph_capture()

    # Note: High-level function tracing (ttnn::moreh_dot) was removed from decorators.hpp
    # Now only device operations are captured. Find MorehDotOperation
    moreh_dot_op = None
    for node in captured_graph:
        if node.get("node_type") == "function_start" and node.get("params", {}).get("name") == "MorehDotOperation":
            moreh_dot_op = node
            break

    assert moreh_dot_op is not None, "MorehDotOperation should be in the captured graph"

    # MorehDotOperation arguments
    node_moreh = moreh_dot_op["arguments"]
    assert (
        node_moreh[0]
        == "[ unsupported type , std::reference_wrapper<ttnn::operations::moreh::moreh_dot::MorehDotOperation::operation_attributes_t const>]"
    )
    assert (
        node_moreh[1]
        == "[ unsupported type , std::reference_wrapper<std::vector<std::reference_wrapper<tt::tt_metal::Tensor const>, std::allocator<std::reference_wrapper<tt::tt_metal::Tensor const> > > >]"
    )

    # tt::tt_metal::create_device_tensor
    create_tensor_op = None
    for node in captured_graph:
        if (
            node.get("node_type") == "function_start"
            and node.get("params", {}).get("name") == "tt::tt_metal::create_device_tensor"
        ):
            # Find the one that creates the output tensor (should have shape [1, 1, 1, 1])
            args = node.get("arguments", [])
            if len(args) > 0 and "Shape([1, 1, 1, 1])" in str(args[0]):
                create_tensor_op = node
                break

    assert create_tensor_op is not None, "create_device_tensor should be in the captured graph"
    node7 = create_tensor_op["arguments"]
    assert node7[0] == "Shape([1, 1, 1, 1])"
    assert node7[1] == "DataType::BFLOAT16"
    assert node7[2] == "Layout::TILE"
    assert node7[3].isnumeric()
    assert (
        node7[4]
        == "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)"
    )


def test_graph_capture_without_dtype(device):
    torch_input = torch.randint(0, 100, (32, 32), dtype=torch.int32)
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    tt_output = ttnn.moreh_full_like(tt_input, 3)
    captured_graph = ttnn.graph.end_graph_capture()

    # Note: High-level function tracing (ttnn::moreh_full_like) was removed from decorators.hpp
    # Now only device operations are captured. Find FullLikeOperation
    full_like_op = None
    for node in captured_graph:
        if node.get("node_type") == "function_start" and node.get("params", {}).get("name") == "FullLikeOperation":
            full_like_op = node
            break

    assert full_like_op is not None, "FullLikeOperation should be in the captured graph"

    # FullLikeOperation arguments
    node_full_like = full_like_op["arguments"]
    assert (
        node_full_like[0]
        == "[ unsupported type , std::reference_wrapper<ttnn::operations::full_like::FullLikeOperation::operation_attributes_t const>]"
    )
    assert (
        node_full_like[1]
        == "[ unsupported type , std::reference_wrapper<std::vector<std::reference_wrapper<tt::tt_metal::Tensor const>, std::allocator<std::reference_wrapper<tt::tt_metal::Tensor const> > > >]"
    )

    # tt::tt_metal::create_device_tensor
    create_tensor_op = None
    for node in captured_graph:
        if (
            node.get("node_type") == "function_start"
            and node.get("params", {}).get("name") == "tt::tt_metal::create_device_tensor"
        ):
            # Find the one that creates the output tensor (should have shape [32, 32])
            args = node.get("arguments", [])
            if len(args) > 0 and "Shape([32, 32])" in str(args[0]):
                create_tensor_op = node
                break

    assert create_tensor_op is not None, "create_device_tensor should be in the captured graph"
    node5 = create_tensor_op["arguments"]
    assert node5[0] == "Shape([32, 32])"
    assert node5[1] == "DataType::INT32"
    assert node5[2] == "Layout::TILE"
    assert node5[3].isnumeric()
    assert (
        node5[4]
        == "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)"
    )


def test_graph_capture_with_all_parameters_json_output(device):
    if is_watcher_enabled():
        pytest.skip("Skipping due to failure with watcher enabled, github issue #37096")
    # Create input tensor
    torch_input = torch.rand((1, 1, 2048, 512), dtype=torch.bfloat16)

    # TT operations
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    tt_input = tt_input.reshape(1, 2048, 4, 128)
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    ttnn.transpose(tt_input, 1, 2)
    captured_graph = ttnn.graph.end_graph_capture()

    data = GraphTracerUtils.serialize_graph(captured_graph)
    assert "content" in data
    assert isinstance(data["content"], list)
    # Note: High-level function tracing (ttnn::transpose) was removed from decorators.hpp
    # Now only device operations are captured: PermuteDeviceOperation, create_device_tensor
    assert len(data["content"]) == 2

    # Content item 0: PermuteDeviceOperation (ttnn::transpose is no longer captured)
    item0 = data["content"][0]
    assert item0["operation"] == "PermuteDeviceOperation"
    assert len(item0["arguments"]) == 2
    assert item0["arguments"][0]["arg0"] == {
        "unsupported type": "std::reference_wrapper<ttnn::operations::data_movement::PermuteDeviceOperation::operation_attributes_t const>"
    }
    assert item0["arguments"][1]["arg1"] == {
        "unsupported type": "std::reference_wrapper<std::vector<std::reference_wrapper<tt::tt_metal::Tensor const>, std::allocator<std::reference_wrapper<tt::tt_metal::Tensor const> > > >"
    }

    # Content item 1: create_device_tensor
    item1 = data["content"][1]
    assert item1["operation"] == "tt::tt_metal::create_device_tensor"
    arg0_item1 = item1["arguments"][0]["arg0"]
    assert arg0_item1["Shape"] == [1, 4, 2048, 128]
    assert item1["arguments"][1]["arg1"] == "DataType::BFLOAT16"
    assert item1["arguments"][2]["arg2"] == "Layout::ROW_MAJOR"
    assert item1["arguments"][3]["arg3"].isnumeric()

    arg4_item1 = item1["arguments"][4]["arg4"]
    mem_config_item1 = arg4_item1["MemoryConfig"]
    assert mem_config_item1["memory_layout"] == "TensorMemoryLayout::INTERLEAVED"
    assert mem_config_item1["buffer_type"] == "BufferType::L1"
    assert mem_config_item1["shard_spec"] == "std::nullopt"


def test_graph_capture_without_memory_config_json_output(device):
    # Create input tensor
    input_shape = (1, 1, 1, 32)
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

    # TT operations
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    tt_other = ttnn.from_torch(
        torch_input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    tt_out = ttnn.operations.moreh.dot(tt_input, tt_other, dtype=ttnn.bfloat16, output=None)
    captured_graph = ttnn.graph.end_graph_capture()
    data = GraphTracerUtils.serialize_graph(captured_graph)

    assert "content" in data
    assert isinstance(data["content"], list)
    # Note: High-level function tracing (ttnn::moreh_dot) was removed from decorators.hpp
    # Now only device operations are captured: MorehDotOperation, create_device_tensor
    assert len(data["content"]) == 2

    # --- Content item 0: MorehDotOperation (device operation) ---
    item0 = data["content"][0]
    assert item0["operation"] == "MorehDotOperation"
    assert len(item0["arguments"]) == 2
    assert item0["arguments"][0]["arg0"] == {
        "unsupported type": "std::reference_wrapper<ttnn::operations::moreh::moreh_dot::MorehDotOperation::operation_attributes_t const>"
    }
    assert item0["arguments"][1]["arg1"] == {
        "unsupported type": "std::reference_wrapper<std::vector<std::reference_wrapper<tt::tt_metal::Tensor const>, std::allocator<std::reference_wrapper<tt::tt_metal::Tensor const> > > >"
    }

    # --- Content item 1: create_device_tensor ---
    item1 = data["content"][1]
    assert item1["operation"] == "tt::tt_metal::create_device_tensor"
    assert len(item1["arguments"]) == 5

    # arg0
    arg0_item1 = item1["arguments"][0]["arg0"]
    assert arg0_item1["Shape"] == [1, 1, 1, 1]
    assert item1["arguments"][1]["arg1"] == "DataType::BFLOAT16"
    assert item1["arguments"][2]["arg2"] == "Layout::TILE"
    assert item1["arguments"][3]["arg3"].isnumeric()

    arg4_item1 = item1["arguments"][4]["arg4"]
    mem_config_item1 = arg4_item1["MemoryConfig"]
    assert mem_config_item1["memory_layout"] == "TensorMemoryLayout::INTERLEAVED"
    assert mem_config_item1["buffer_type"] == "BufferType::DRAM"
    assert mem_config_item1["shard_spec"] == "std::nullopt"


def test_graph_capture_without_dtype_json_output(device):
    torch_input = torch.randint(0, 100, (32, 32), dtype=torch.int32)
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    tt_output = ttnn.moreh_full_like(tt_input, 3)
    captured_graph = ttnn.graph.end_graph_capture()
    data = GraphTracerUtils.serialize_graph(captured_graph)

    assert "content" in data
    assert isinstance(data["content"], list)
    # Note: High-level function tracing (ttnn::moreh_full_like) was removed from decorators.hpp
    # Now only device operations are captured: FullLikeOperation, create_device_tensor
    assert len(data["content"]) == 2

    # --- Content item 0: FullLikeOperation (device operation) ---
    item0 = data["content"][0]
    assert item0["operation"] == "FullLikeOperation"
    assert len(item0["arguments"]) == 2
    assert item0["arguments"][0]["arg0"] == {
        "unsupported type": "std::reference_wrapper<ttnn::operations::full_like::FullLikeOperation::operation_attributes_t const>"
    }
    assert item0["arguments"][1]["arg1"] == {
        "unsupported type": "std::reference_wrapper<std::vector<std::reference_wrapper<tt::tt_metal::Tensor const>, std::allocator<std::reference_wrapper<tt::tt_metal::Tensor const> > > >"
    }

    # --- Content item 1: create_device_tensor ---
    item1 = data["content"][1]
    assert item1["operation"] == "tt::tt_metal::create_device_tensor"
    assert len(item1["arguments"]) == 5

    # arg0: Check the Shape
    arg0_item1 = item1["arguments"][0]["arg0"]
    assert arg0_item1["Shape"] == [32, 32]

    # arg1
    assert item1["arguments"][1]["arg1"] == "DataType::INT32"
    # arg2
    assert item1["arguments"][2]["arg2"] == "Layout::TILE"
    # arg3
    assert item1["arguments"][3]["arg3"].isnumeric()

    # arg4: Check the MemoryConfig
    arg4_item1 = item1["arguments"][4]["arg4"]
    mem_config_item1 = arg4_item1["MemoryConfig"]
    assert mem_config_item1["memory_layout"] == "TensorMemoryLayout::INTERLEAVED"
    assert mem_config_item1["buffer_type"] == "BufferType::DRAM"
    assert mem_config_item1["shard_spec"] == "std::nullopt"


def test_extract_levelized_graph(device):
    """Test extract_levelized_graph API"""
    torch.manual_seed(0)
    torch_input_tensor = torch.rand((64,), dtype=torch.bfloat16)

    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = input_tensor + 3
    output_tensor = ttnn.to_torch(output_tensor, torch_rank=1)
    captured_graph = ttnn.graph.end_graph_capture()

    # Test with default max_level (1)
    levelized_graph = ttnn.graph.extract_levelized_graph(captured_graph)
    assert isinstance(levelized_graph, list)
    assert len(levelized_graph) > 0

    # Verify structure of first vertex
    if len(levelized_graph) > 0:
        vertex = levelized_graph[0]
        assert "counter" in vertex
        assert "stacking_level" in vertex
        assert "name" in vertex
        assert "in_edges" in vertex
        assert "out_edges" in vertex
        assert "internals" in vertex
        assert "output_shape" in vertex
        assert isinstance(vertex["in_edges"], list)
        assert isinstance(vertex["out_edges"], list)
        assert isinstance(vertex["internals"], list)

    # Test with explicit max_level = 1
    levelized_graph_1 = ttnn.graph.extract_levelized_graph(captured_graph, max_level=1)
    assert isinstance(levelized_graph_1, list)
    assert len(levelized_graph_1) == len(levelized_graph)  # Should be same as default

    # Test with max_level = 2
    levelized_graph_2 = ttnn.graph.extract_levelized_graph(captured_graph, max_level=2)
    assert isinstance(levelized_graph_2, list)
    # Level 2 should have at least as many vertices as level 1 (possibly more)
    assert len(levelized_graph_2) >= len(levelized_graph_1)
