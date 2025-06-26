# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pathlib
import pytest
import torch
import ttnn
from ttnn.graph_tracer_utils import GraphTracerUtils
import time


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

    node1 = captured_graph[1]["arguments"]
    # ttnn:transpose
    assert node1[0] == "\x00"
    assert (
        node1[1]
        == "Tensor(storage=DeviceStorage(),tensor_spec=TensorSpec(logical_shape=Shape([1, 2048, 4, 128]),tensor_layout=TensorLayout(dtype=DataType::BFLOAT16,page_config=PageConfig(config=RowMajorPageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0),alignment=Alignment([1]))))"
    )
    assert node1[2] == "1"
    assert node1[3] == "2"
    assert node1[4] == "nullopt"
    assert node1[5] == "0"

    # ttnn::prim::permute
    node4 = captured_graph[4]["arguments"]
    assert (
        node4[0]
        == "Tensor(storage=DeviceStorage(),tensor_spec=TensorSpec(logical_shape=Shape([1, 2048, 4, 128]),tensor_layout=TensorLayout(dtype=DataType::BFLOAT16,page_config=PageConfig(config=RowMajorPageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0),alignment=Alignment([1]))))"
    )
    assert node4[1] == "SmallVector([0, 2, 1, 3])"
    assert (
        node4[2]
        == "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)"
    )
    assert node4[3] == "[ unsupported type , std::reference_wrapper<std::nullopt_t const>]"
    assert node4[4] == "0"

    # PermuteDeviceOperation
    node6 = captured_graph[6]["arguments"]
    assert (
        node6[0]
        == "[ unsupported type , std::reference_wrapper<ttnn::operations::data_movement::PermuteDeviceOperation::operation_attributes_t const>]"
    )
    assert (
        node6[1]
        == "[ unsupported type , std::reference_wrapper<ttnn::operations::data_movement::PermuteDeviceOperation::tensor_args_t const>]"
    )

    # tt::tt_metal::create_device_tensor
    node7 = captured_graph[7]["arguments"]
    assert node7[0] == "Shape([1, 4, 2048, 128])"
    assert node7[1] == "DataType::BFLOAT16"
    assert node7[2] == "Layout::ROW_MAJOR"
    assert node7[3] == "[ unsupported type , std::reference_wrapper<tt::tt_metal::IDevice*>]"
    assert (
        node7[4]
        == "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0)"
    )


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
    assert node1[5] == "nullopt"

    # ttnn::prim::moreh_dot
    node6 = captured_graph[6]["arguments"]
    assert (
        node6[0]
        == "Tensor(storage=DeviceStorage(),tensor_spec=TensorSpec(logical_shape=Shape([1, 1, 1, 32]),tensor_layout=TensorLayout(dtype=DataType::BFLOAT16,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0),alignment=Alignment([32, 32]))))"
    )
    assert (
        node6[1]
        == "Tensor(storage=DeviceStorage(),tensor_spec=TensorSpec(logical_shape=Shape([1, 1, 1, 32]),tensor_layout=TensorLayout(dtype=DataType::BFLOAT16,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0),alignment=Alignment([32, 32]))))"
    )
    assert node6[2] == "nullopt"
    assert node6[3] == "DataType::BFLOAT16"
    assert node6[4] == "nullopt"
    assert node6[5] == "nullopt"

    # MorehDotOperation
    node9 = captured_graph[9]["arguments"]
    assert (
        node9[0]
        == "[ unsupported type , std::reference_wrapper<ttnn::operations::moreh::moreh_dot::MorehDotOperation::operation_attributes_t const>]"
    )
    assert (
        node9[1]
        == "[ unsupported type , std::reference_wrapper<ttnn::operations::moreh::moreh_dot::MorehDotOperation::tensor_args_t const>]"
    )

    # tt::tt_metal::create_device_tensor
    node10 = captured_graph[10]["arguments"]
    assert node10[0] == "Shape([1, 1, 1, 1])"
    assert node10[1] == "DataType::BFLOAT16"
    assert node10[2] == "Layout::TILE"
    assert node10[3] == "[ unsupported type , std::reference_wrapper<tt::tt_metal::IDevice*>]"
    assert (
        node10[4]
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

    # ttnn::prim::moreh_full_like
    node4 = captured_graph[4]["arguments"]
    assert (
        node4[0]
        == "Tensor(storage=DeviceStorage(),tensor_spec=TensorSpec(logical_shape=Shape([32, 32]),tensor_layout=TensorLayout(dtype=DataType::INT32,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0),alignment=Alignment([32, 32]))))"
    )
    assert node4[1] == "3"
    assert node4[2] == "nullopt"
    assert node4[3] == "nullopt"
    assert node4[4] == "nullopt"

    # FullLikeOperation
    node6 = captured_graph[6]["arguments"]
    assert (
        node6[0]
        == "[ unsupported type , std::reference_wrapper<ttnn::operations::full_like::FullLikeOperation::operation_attributes_t const>]"
    )
    assert (
        node6[1]
        == "[ unsupported type , std::reference_wrapper<ttnn::operations::full_like::FullLikeOperation::tensor_args_t const>]"
    )

    # tt::tt_metal::create_device_tensor
    node7 = captured_graph[7]["arguments"]
    assert node7[0] == "Shape([32, 32])"
    assert node7[1] == "DataType::INT32"
    assert node7[2] == "Layout::TILE"
    assert node7[3] == "[ unsupported type , std::reference_wrapper<tt::tt_metal::IDevice*>]"
    assert (
        node7[4]
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
    # Content item 0
    assert "content" in data
    assert isinstance(data["content"], list)
    assert len(data["content"]) == 4

    item0 = data["content"][0]
    assert item0["operation"] == "ttnn::transpose"
    assert len(item0["arguments"]) == 6

    # arg0
    assert item0["arguments"][0]["arg0"] == "\u0000"

    # arg1
    arg1 = item0["arguments"][1]["arg1"]
    tensor = arg1["Tensor"]

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

    # arg2 to arg5
    assert item0["arguments"][2]["arg2"] == "1"
    assert item0["arguments"][3]["arg3"] == "2"
    assert item0["arguments"][4]["arg4"] == "nullopt"
    assert item0["arguments"][5]["arg5"] == "0"

    # Content item 1
    item1 = data["content"][1]
    assert item1["operation"] == "ttnn::prim::permute"
    assert len(item1["arguments"]) == 5

    arg2_item1 = item1["arguments"][2]["arg2"]["MemoryConfig"]
    assert arg2_item1["memory_layout"] == "TensorMemoryLayout::INTERLEAVED"
    assert arg2_item1["buffer_type"] == "BufferType::L1"
    assert arg2_item1["shard_spec"] == "std::nullopt"
    assert item1["arguments"][3]["arg3"] == "[ unsupported type , std::reference_wrapper<std::nullopt_t const>]"
    assert item1["arguments"][4]["arg4"] == "0"

    # Content item 2
    item2 = data["content"][2]
    assert item2["operation"] == "PermuteDeviceOperation"
    assert len(item2["arguments"]) == 2
    assert (
        item2["arguments"][0]["arg0"]
        == "[ unsupported type , std::reference_wrapper<ttnn::operations::data_movement::PermuteDeviceOperation::operation_attributes_t const>]"
    )
    assert (
        item2["arguments"][1]["arg1"]
        == "[ unsupported type , std::reference_wrapper<ttnn::operations::data_movement::PermuteDeviceOperation::tensor_args_t const>]"
    )

    # Content item 3
    item3 = data["content"][3]
    assert item3["operation"] == "tt::tt_metal::create_device_tensor"
    assert len(item3["arguments"]) == 5

    arg0_item3 = item3["arguments"][0]["arg0"]
    assert arg0_item3["Shape"] == [1, 4, 2048, 128]
    assert item3["arguments"][1]["arg1"] == "DataType::BFLOAT16"
    assert item3["arguments"][2]["arg2"] == "Layout::ROW_MAJOR"
    assert item3["arguments"][3]["arg3"] == "[ unsupported type , std::reference_wrapper<tt::tt_metal::IDevice*>]"

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
    assert len(data["content"]) == 4

    # --- Content item 0 ---
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
    assert item0["arguments"][5]["arg5"] == "nullopt"

    # --- Content item 1 ---
    item1 = data["content"][1]
    assert item1["operation"] == "ttnn::prim::moreh_dot"
    assert len(item1["arguments"]) == 6

    # arg0
    arg0_item1 = item1["arguments"][0]["arg0"]
    tensor0_item1 = arg0_item1["Tensor"]

    tspec0_item1 = tensor0_item1["tensor_spec"]
    assert tspec0_item1["logical_shape"] == [1, 1, 1, 32]
    tlayout0_item1 = tspec0_item1["tensor_layout"]
    assert tlayout0_item1["dtype"] == "DataType::BFLOAT16"
    tile0_item1 = tlayout0_item1["page_config"]["config"]["tile"]
    assert tile0_item1["tile_shape"] == "{32, 32}"
    assert tile0_item1["face_shape"] == "{16, 16}"
    assert tile0_item1["num_faces"] == 4
    mem_config_tensor0_item1 = tlayout0_item1["memory_config"]
    assert mem_config_tensor0_item1["memory_layout"] == "TensorMemoryLayout::INTERLEAVED"
    assert mem_config_tensor0_item1["buffer_type"] == "BufferType::DRAM"
    assert mem_config_tensor0_item1["shard_spec"] == "std::nullopt"
    assert tlayout0_item1["alignment"] == [32, 32]

    # arg1
    arg1_item1 = item1["arguments"][1]["arg1"]
    tensor1_item1 = arg1_item1["Tensor"]

    tspec1_item1 = tensor1_item1["tensor_spec"]
    assert tspec1_item1["logical_shape"] == [1, 1, 1, 32]
    tlayout1_item1 = tspec1_item1["tensor_layout"]
    assert tlayout1_item1["dtype"] == "DataType::BFLOAT16"
    tile1_item1 = tlayout1_item1["page_config"]["config"]["tile"]
    assert tile1_item1["tile_shape"] == "{32, 32}"
    assert tile1_item1["face_shape"] == "{16, 16}"
    assert tile1_item1["num_faces"] == 4
    mem_config_tensor1_item1 = tlayout1_item1["memory_config"]
    assert mem_config_tensor1_item1["memory_layout"] == "TensorMemoryLayout::INTERLEAVED"
    assert mem_config_tensor1_item1["buffer_type"] == "BufferType::DRAM"
    assert mem_config_tensor1_item1["shard_spec"] == "std::nullopt"
    assert tlayout1_item1["alignment"] == [32, 32]

    # arg2 to arg5
    assert item1["arguments"][2]["arg2"] == "nullopt"
    assert item1["arguments"][3]["arg3"] == "DataType::BFLOAT16"
    assert item1["arguments"][4]["arg4"] == "nullopt"
    assert item1["arguments"][5]["arg5"] == "nullopt"

    # --- Content item 2 ---
    item2 = data["content"][2]
    assert item2["operation"] == "MorehDotOperation"
    assert len(item2["arguments"]) == 2
    assert (
        item2["arguments"][0]["arg0"]
        == "[ unsupported type , std::reference_wrapper<ttnn::operations::moreh::moreh_dot::MorehDotOperation::operation_attributes_t const>]"
    )
    assert (
        item2["arguments"][1]["arg1"]
        == "[ unsupported type , std::reference_wrapper<ttnn::operations::moreh::moreh_dot::MorehDotOperation::tensor_args_t const>]"
    )

    # --- Content item 3 ---
    item3 = data["content"][3]
    assert item3["operation"] == "tt::tt_metal::create_device_tensor"
    assert len(item3["arguments"]) == 5

    # arg0
    arg0_item3 = item3["arguments"][0]["arg0"]
    assert arg0_item3["Shape"] == [1, 1, 1, 1]
    assert item3["arguments"][1]["arg1"] == "DataType::BFLOAT16"
    assert item3["arguments"][2]["arg2"] == "Layout::TILE"
    assert item3["arguments"][3]["arg3"] == "[ unsupported type , std::reference_wrapper<tt::tt_metal::IDevice*>]"

    arg4_item3 = item3["arguments"][4]["arg4"]
    mem_config_item3 = arg4_item3["MemoryConfig"]
    assert mem_config_item3["memory_layout"] == "TensorMemoryLayout::INTERLEAVED"
    assert mem_config_item3["buffer_type"] == "BufferType::DRAM"
    assert mem_config_item3["shard_spec"] == "std::nullopt"


def test_graph_capture_without_dtype_json_output(device):
    torch_input = torch.randint(0, 100, (32, 32), dtype=torch.int32)
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    tt_output = ttnn.moreh_full_like(tt_input, 3)
    captured_graph = ttnn.graph.end_graph_capture()
    data = GraphTracerUtils.serialize_graph(captured_graph)

    # Check top-level structure
    assert "content" in data
    assert isinstance(data["content"], list)
    assert len(data["content"]) == 4

    # --- Content item 0 ---
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

    # --- Content item 1 ---
    item1 = data["content"][1]
    assert item1["operation"] == "ttnn::prim::moreh_full_like"
    assert len(item1["arguments"]) == 5

    # arg0: Check the Tensor structure in arg0 for item1
    arg0_item1 = item1["arguments"][0]["arg0"]
    tensor1 = arg0_item1["Tensor"]

    tspec1 = tensor1["tensor_spec"]
    assert tspec1["logical_shape"] == [32, 32]

    tlayout1 = tspec1["tensor_layout"]
    assert tlayout1["dtype"] == "DataType::INT32"
    tile1 = tlayout1["page_config"]["config"]["tile"]
    assert tile1["tile_shape"] == "{32, 32}"
    assert tile1["face_shape"] == "{16, 16}"
    assert tile1["num_faces"] == 4

    mem_config_tensor1 = tlayout1["memory_config"]
    assert mem_config_tensor1["memory_layout"] == "TensorMemoryLayout::INTERLEAVED"
    assert mem_config_tensor1["buffer_type"] == "BufferType::DRAM"
    assert mem_config_tensor1["shard_spec"] == "std::nullopt"
    assert tlayout1["alignment"] == [32, 32]

    # arg1 to arg4 in item1
    assert item1["arguments"][1]["arg1"] == "3"
    assert item1["arguments"][2]["arg2"] == "nullopt"
    assert item1["arguments"][3]["arg3"] == "nullopt"
    assert item1["arguments"][4]["arg4"] == "nullopt"

    # --- Content item 2 ---
    item2 = data["content"][2]
    assert item2["operation"] == "FullLikeOperation"
    assert len(item2["arguments"]) == 2
    assert (
        item2["arguments"][0]["arg0"]
        == "[ unsupported type , std::reference_wrapper<ttnn::operations::full_like::FullLikeOperation::operation_attributes_t const>]"
    )
    assert (
        item2["arguments"][1]["arg1"]
        == "[ unsupported type , std::reference_wrapper<ttnn::operations::full_like::FullLikeOperation::tensor_args_t const>]"
    )

    # --- Content item 3 ---
    item3 = data["content"][3]
    assert item3["operation"] == "tt::tt_metal::create_device_tensor"
    assert len(item3["arguments"]) == 5

    # arg0: Check the Shape
    arg0_item3 = item3["arguments"][0]["arg0"]
    assert arg0_item3["Shape"] == [32, 32]

    # arg1
    assert item3["arguments"][1]["arg1"] == "DataType::INT32"
    # arg2
    assert item3["arguments"][2]["arg2"] == "Layout::TILE"
    # arg3
    assert item3["arguments"][3]["arg3"] == "[ unsupported type , std::reference_wrapper<tt::tt_metal::IDevice*>]"

    # arg4: Check the MemoryConfig
    arg4_item3 = item3["arguments"][4]["arg4"]
    mem_config_item3 = arg4_item3["MemoryConfig"]
    assert mem_config_item3["memory_layout"] == "TensorMemoryLayout::INTERLEAVED"
    assert mem_config_item3["buffer_type"] == "BufferType::DRAM"
    assert mem_config_item3["shard_spec"] == "std::nullopt"


# this test is meaningless unless you compile with --ttnn-enable-operation-timeout
def test_graph_capture_with_hang_host_operation(device):
    # Create input tensor
    tt_input = ttnn.empty(
        shape=(1, 1, 2048, 512),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)

    failed = False
    try:
        output = ttnn.experimental.test.test_hang_host_operation(tt_input)
        failed = False
    except RuntimeError as e:
        captured_graph = ttnn.graph.end_graph_capture()
        failed = True
        assert "TIMEOUT" in str(e)
        assert "ttnn::experimental::test::test_hang_host_operation" in str(e)

    # this is the normal case for CI
    if not failed:
        assert output == tt_input
    else:
        # this is the case for --ttnn-enable-operation-timeout
        # the graph should have captured the arguments to the hang operation
        assert (
            captured_graph[1]["arguments"][0]
            == "Tensor(storage=DeviceStorage(),tensor_spec=TensorSpec(logical_shape=Shape([1, 1, 2048, 512]),tensor_layout=TensorLayout(dtype=DataType::BFLOAT16,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0),alignment=Alignment([32, 32]))))"
        )


# this test is meaningless unless you compile with --ttnn-enable-operation-timeout
def test_graph_capture_with_hang_device_operation(device):
    # Create input tensor
    tt_input = ttnn.empty(
        shape=(1, 1, 2048, 512),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)

    failed = False
    try:
        output = ttnn.prim.test_hang_device_operation(tt_input)
        ttnn._ttnn.device.synchronize_device(device)
        failed = False
    except Exception as e:
        print("Exception captured")
        ttnn._ttnn.device.close_device(device)
        captured_graph = ttnn.graph.end_graph_capture()
        failed = True
        assert "TIMEOUT" in str(e)
        assert "potential hang detected, please check the graph capture" in str(e)

    # this is the normal case for CI
    if not failed:
        assert output == tt_input
    else:
        print(captured_graph)
        # this is the case for --ttnn-enable-operation-timeout
        # the graph should have captured the arguments to the hang operation
        assert (
            captured_graph[1]["arguments"][0]
            == "Tensor(storage=DeviceStorage(),tensor_spec=TensorSpec(logical_shape=Shape([1, 1, 2048, 512]),tensor_layout=TensorLayout(dtype=DataType::BFLOAT16,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt,nd_shard_spec=std::nullopt,created_with_nd_shard_spec=0),alignment=Alignment([32, 32]))))"
        )
