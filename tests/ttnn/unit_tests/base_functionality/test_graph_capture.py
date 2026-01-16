# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pathlib
import pytest
import torch
import ttnn
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

    assert "tt::tt_metal::detail::convert_python_tensor_to_tt_tensor" in calltrace
    assert captured_graph[0]["node_type"] == "capture_start"
    assert captured_graph[1]["node_type"] == "function_start"
    assert captured_graph[1]["params"]["name"] == "tt::tt_metal::detail::convert_python_tensor_to_tt_tensor"
    assert captured_graph[-2]["node_type"] == "buffer_deallocate"
    assert captured_graph[-1]["node_type"] == "capture_end"

    ttnn.graph.pretty_print(captured_graph)

    ttnn.graph.visualize(captured_graph, file_name=tmp_path / pathlib.Path("graph.svg"))


def test_graph_capture_with_all_parameters(device):
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

    # ttnn::transpose
    node1 = captured_graph[1]["arguments"]
    assert (
        node1[0]
        == "Tensor(storage=DeviceStorage(),tensor_spec=TensorSpec(logical_shape=Shape([1, 2048, 4, 128]),tensor_layout=TensorLayout(dtype=DataType::BFLOAT16,page_config=PageConfig(config=RowMajorPageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0),alignment=Alignment([1]))))"
    )
    assert node1[1] == "1"
    assert node1[2] == "2"
    assert node1[3] == "nullopt"
    assert node1[4] == "0"

    # ttnn::prim::permute
    node4 = captured_graph[4]["arguments"]
    assert (
        node4[0]
        == "[ unsupported type , std::reference_wrapper<ttnn::operations::data_movement::PermuteDeviceOperation::operation_attributes_t const>]"
    )
    assert (
        node4[1]
        == "[ unsupported type , std::reference_wrapper<std::vector<std::reference_wrapper<tt::tt_metal::Tensor const>, std::allocator<std::reference_wrapper<tt::tt_metal::Tensor const> > > >]"
    )

    # tt::tt_metal::create_device_tensor (shifted by 1 due to device operation tracking)
    node5 = captured_graph[5]["arguments"]
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

    # ttnn::moreh_dot
    node1 = captured_graph[1]["arguments"]
    assert (
        node1[0]
        == "Tensor(storage=DeviceStorage(),tensor_spec=TensorSpec(logical_shape=Shape([1, 1, 1, 32]),tensor_layout=TensorLayout(dtype=DataType::BFLOAT16,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0),alignment=Alignment([32, 32]))))"
    )
    assert (
        node1[1]
        == "Tensor(storage=DeviceStorage(),tensor_spec=TensorSpec(logical_shape=Shape([1, 1, 1, 32]),tensor_layout=TensorLayout(dtype=DataType::BFLOAT16,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0),alignment=Alignment([32, 32]))))"
    )
    assert node1[2] == "nullopt"
    assert node1[3] == "DataType::BFLOAT16"
    assert node1[4] == "nullopt"
    assert (
        node1[5]
        == "[ unsupported type , std::reference_wrapper<std::optional<std::variant<ttnn::GrayskullComputeKernelConfig, ttnn::WormholeComputeKernelConfig> > const>]"
    )

    # tt::tt_metal::create_device_tensor
    node7 = captured_graph[7]["arguments"]
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

    # ttnn::moreh_full_like
    node1 = captured_graph[1]["arguments"]
    assert (
        node1[0]
        == "Tensor(storage=DeviceStorage(),tensor_spec=TensorSpec(logical_shape=Shape([32, 32]),tensor_layout=TensorLayout(dtype=DataType::INT32,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0),alignment=Alignment([32, 32]))))"
    )
    assert node1[1] == "3"
    assert node1[2] == "nullopt"
    assert node1[3] == "nullopt"
    assert node1[4] == "nullopt"

    # FullLikeOperation (now tracked at level 2)
    assert captured_graph[4]["params"]["name"] == "FullLikeOperation"

    # tt::tt_metal::create_device_tensor
    node5 = captured_graph[5]["arguments"]
    assert node5[0] == "Shape([32, 32])"
    assert node5[1] == "DataType::INT32"
    assert node5[2] == "Layout::TILE"
    assert node5[3].isnumeric()
    assert (
        node5[4]
        == "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)"
    )


def test_graph_capture_with_all_parameters_json_output(device):
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
    assert len(data["content"]) == 3  # ttnn::transpose, PermuteDeviceOperation, create_device_tensor

    # Content item 0: ttnn::transpose
    item0 = data["content"][0]
    assert item0["operation"] == "ttnn::transpose"
    assert len(item0["arguments"]) == 5

    # arg0 is now the tensor argument
    tensor = item0["arguments"][0]["arg0"]["Tensor"]

    tspec = tensor["tensor_spec"]
    assert tspec["logical_shape"] == [1, 2048, 4, 128]
    tlayout = tspec["tensor_layout"]
    assert tlayout["dtype"] == "DataType::BFLOAT16"
    tile = tlayout["page_config"]["config"]["tile"]
    assert tile["tile_shape"] == "{32, 32}"
    assert tile["face_shape"] == "{16, 16}"
    assert tile["num_faces"] == 4
    mem_config_tensor = tlayout["memory_config"]
    assert mem_config_tensor["memory_layout"] == "TensorMemoryLayout::INTERLEAVED"
    assert mem_config_tensor["buffer_type"] == "BufferType::L1"
    assert mem_config_tensor["shard_spec"] == "std::nullopt"
    assert tlayout["alignment"] == [1]

    # arg1 to arg4 (previously arg2 to arg5)
    assert item0["arguments"][1]["arg1"] == "1"
    assert item0["arguments"][2]["arg2"] == "2"
    assert item0["arguments"][3]["arg3"] == "nullopt"
    assert item0["arguments"][4]["arg4"] == "0"

    item1 = data["content"][0]
    assert item1["operation"] == "ttnn::transpose"
    item1_arguments = item1["arguments"]
    assert item1_arguments[0]["arg0"]["Tensor"]["tensor_spec"]["logical_shape"] == [1, 2048, 4, 128]
    assert item1_arguments[1]["arg1"] == "1"
    assert item1_arguments[2]["arg2"] == "2"
    assert item1_arguments[3]["arg3"] == "nullopt"
    assert item1_arguments[4]["arg4"] == "0"

    # Content item 2
    item2 = data["content"][1]
    assert item2["operation"] == "PermuteDeviceOperation"
    assert len(item2["arguments"]) == 2
    assert item2["arguments"][0]["arg0"] == {
        "unsupported type": "std::reference_wrapper<ttnn::operations::data_movement::PermuteDeviceOperation::operation_attributes_t const>"
    }
    assert item2["arguments"][1]["arg1"] == {
        "unsupported type": "std::reference_wrapper<std::vector<std::reference_wrapper<tt::tt_metal::Tensor const>, std::allocator<std::reference_wrapper<tt::tt_metal::Tensor const> > > >"
    }

    arg0_item2 = item2["arguments"][0]["arg0"]
    assert arg0_item2 == {
        "unsupported type": "std::reference_wrapper<ttnn::operations::data_movement::PermuteDeviceOperation::operation_attributes_t const>"
    }

    item3 = data["content"][2]
    arg0_item3 = item3["arguments"][0]["arg0"]
    assert arg0_item3["Shape"] == [1, 4, 2048, 128]
    assert item3["arguments"][1]["arg1"] == "DataType::BFLOAT16"
    assert item3["arguments"][2]["arg2"] == "Layout::ROW_MAJOR"
    assert item3["arguments"][3]["arg3"].isnumeric()

    arg4_item3 = item3["arguments"][4]["arg4"]
    mem_config_item3 = arg4_item3["MemoryConfig"]
    assert mem_config_item3["memory_layout"] == "TensorMemoryLayout::INTERLEAVED"
    assert mem_config_item3["buffer_type"] == "BufferType::L1"
    assert mem_config_item3["shard_spec"] == "std::nullopt"


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
    assert len(data["content"]) == 3  # ttnn::moreh_dot, MorehDotOperation, create_device_tensor

    # --- Content item 0: ttnn::moreh_dot ---
    item0 = data["content"][0]
    assert item0["operation"] == "ttnn::moreh_dot"
    assert len(item0["arguments"]) == 6

    # arg0
    arg0_item0 = item0["arguments"][0]["arg0"]
    tensor0 = arg0_item0["Tensor"]

    tspec0 = tensor0["tensor_spec"]
    assert tspec0["logical_shape"] == [1, 1, 1, 32]
    tlayout0 = tspec0["tensor_layout"]
    assert tlayout0["dtype"] == "DataType::BFLOAT16"
    tile0 = tlayout0["page_config"]["config"]["tile"]
    assert tile0["tile_shape"] == "{32, 32}"
    assert tile0["face_shape"] == "{16, 16}"
    assert tile0["num_faces"] == 4
    mem_config_tensor0 = tlayout0["memory_config"]
    assert mem_config_tensor0["memory_layout"] == "TensorMemoryLayout::INTERLEAVED"
    assert mem_config_tensor0["buffer_type"] == "BufferType::DRAM"
    assert mem_config_tensor0["shard_spec"] == "std::nullopt"
    assert tlayout0["alignment"] == [32, 32]

    # arg1
    arg1_item0 = item0["arguments"][1]["arg1"]
    tensor1 = arg1_item0["Tensor"]

    tspec1 = tensor1["tensor_spec"]
    assert tspec1["logical_shape"] == [1, 1, 1, 32]
    tlayout1 = tspec1["tensor_layout"]
    assert tlayout1["dtype"] == "DataType::BFLOAT16"
    tile1 = tlayout1["page_config"]["config"]["tile"]
    assert tile1["tile_shape"] == "{32, 32}"
    assert tile1["face_shape"] == "{16, 16}"
    assert tile1["num_faces"] == 4
    mem_config_tensor1 = tlayout1["memory_config"]
    assert mem_config_tensor1["memory_layout"] == "TensorMemoryLayout::INTERLEAVED"
    assert mem_config_tensor1["buffer_type"] == "BufferType::DRAM"
    assert mem_config_tensor1["shard_spec"] == "std::nullopt"
    assert tlayout1["alignment"] == [32, 32]

    # arg2 to arg5
    assert item0["arguments"][2]["arg2"] == "nullopt"
    assert item0["arguments"][3]["arg3"] == "DataType::BFLOAT16"
    assert item0["arguments"][4]["arg4"] == "nullopt"
    assert item0["arguments"][5]["arg5"] == {
        "unsupported type": "std::reference_wrapper<std::optional<std::variant<ttnn::GrayskullComputeKernelConfig, ttnn::WormholeComputeKernelConfig> > const>"
    }

    # --- Content item 1: MorehDotOperation (device operation now tracked) ---
    item1 = data["content"][1]
    assert item1["operation"] == "MorehDotOperation"

    assert item1["arguments"][0]["arg0"] == {
        "unsupported type": "std::reference_wrapper<ttnn::operations::moreh::moreh_dot::MorehDotOperation::operation_attributes_t const>"
    }
    assert item1["arguments"][1]["arg1"] == {
        "unsupported type": "std::reference_wrapper<std::vector<std::reference_wrapper<tt::tt_metal::Tensor const>, std::allocator<std::reference_wrapper<tt::tt_metal::Tensor const> > > >"
    }

    # --- Content item 2 ---
    item2 = data["content"][2]
    assert item2["operation"] == "tt::tt_metal::create_device_tensor"
    assert len(item2["arguments"]) == 5

    # arg0
    arg0_item2 = item2["arguments"][0]["arg0"]
    assert arg0_item2["Shape"] == [1, 1, 1, 1]
    assert item2["arguments"][1]["arg1"] == "DataType::BFLOAT16"
    assert item2["arguments"][2]["arg2"] == "Layout::TILE"
    assert item2["arguments"][3]["arg3"].isnumeric()

    arg4_item2 = item2["arguments"][4]["arg4"]
    mem_config_item2 = arg4_item2["MemoryConfig"]
    assert mem_config_item2["memory_layout"] == "TensorMemoryLayout::INTERLEAVED"
    assert mem_config_item2["buffer_type"] == "BufferType::DRAM"
    assert mem_config_item2["shard_spec"] == "std::nullopt"


def test_graph_capture_without_dtype_json_output(device):
    torch_input = torch.randint(0, 100, (32, 32), dtype=torch.int32)
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    tt_output = ttnn.moreh_full_like(tt_input, 3)
    captured_graph = ttnn.graph.end_graph_capture()
    data = GraphTracerUtils.serialize_graph(captured_graph)

    assert "content" in data
    assert isinstance(data["content"], list)
    assert len(data["content"]) == 3  # ttnn::moreh_full_like, FullLikeOperation, create_device_tensor

    # --- Content item 0: ttnn::moreh_full_like ---
    item0 = data["content"][0]
    assert item0["operation"] == "ttnn::moreh_full_like"
    assert len(item0["arguments"]) == 5

    # arg0: Check the Tensor structure in arg0
    arg0_item0 = item0["arguments"][0]["arg0"]
    tensor0 = arg0_item0["Tensor"]

    # Check tensor_spec
    tspec0 = tensor0["tensor_spec"]
    assert tspec0["logical_shape"] == [32, 32]

    tlayout0 = tspec0["tensor_layout"]
    assert tlayout0["dtype"] == "DataType::INT32"
    tile0 = tlayout0["page_config"]["config"]["tile"]
    assert tile0["tile_shape"] == "{32, 32}"
    assert tile0["face_shape"] == "{16, 16}"
    assert tile0["num_faces"] == 4

    mem_config_tensor0 = tlayout0["memory_config"]
    assert mem_config_tensor0["memory_layout"] == "TensorMemoryLayout::INTERLEAVED"
    assert mem_config_tensor0["buffer_type"] == "BufferType::DRAM"
    assert mem_config_tensor0["shard_spec"] == "std::nullopt"
    assert tlayout0["alignment"] == [32, 32]

    # arg1 to arg4
    assert item0["arguments"][1]["arg1"] == "3"
    assert item0["arguments"][2]["arg2"] == "nullopt"
    assert item0["arguments"][3]["arg3"] == "nullopt"
    assert item0["arguments"][4]["arg4"] == "nullopt"

    # --- Content item 1: tt::tt_metal::create_device_tensor ---
    item1 = data["content"][1]
    assert item1["operation"] == "FullLikeOperation"
    assert len(item1["arguments"]) == 2
    assert item1["arguments"][0]["arg0"] == {
        "unsupported type": "std::reference_wrapper<ttnn::operations::full_like::FullLikeOperation::operation_attributes_t const>"
    }
    assert item1["arguments"][1]["arg1"] == {
        "unsupported type": "std::reference_wrapper<std::vector<std::reference_wrapper<tt::tt_metal::Tensor const>, std::allocator<std::reference_wrapper<tt::tt_metal::Tensor const> > > >"
    }

    # --- Content item 2 ---
    item2 = data["content"][2]
    assert item2["operation"] == "tt::tt_metal::create_device_tensor"
    assert len(item2["arguments"]) == 5

    # arg0: Check the Shape
    arg0_item2 = item2["arguments"][0]["arg0"]
    assert arg0_item2["Shape"] == [32, 32]

    # arg1
    assert item2["arguments"][1]["arg1"] == "DataType::INT32"
    # arg2
    assert item2["arguments"][2]["arg2"] == "Layout::TILE"
    # arg3
    assert item2["arguments"][3]["arg3"].isnumeric()

    # arg4: Check the MemoryConfig
    arg4_item2 = item2["arguments"][4]["arg4"]
    mem_config_item2 = arg4_item2["MemoryConfig"]
    assert mem_config_item2["memory_layout"] == "TensorMemoryLayout::INTERLEAVED"
    assert mem_config_item2["buffer_type"] == "BufferType::DRAM"
    assert mem_config_item2["shard_spec"] == "std::nullopt"


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
