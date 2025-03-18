# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pathlib
import pytest

import torch

import ttnn


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
        == "Tensor(storage=DeviceStorage(memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt)),tensor_spec=TensorSpec(logical_shape=Shape([1, 2048, 4, 128]),tensor_layout=TensorLayout(dtype=DataType::BFLOAT16,page_config=PageConfig(config=RowMajorPageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt),alignment=Alignment([1]))))"
    )
    assert node1[2] == "1"
    assert node1[3] == "2"
    assert node1[4] == "nullopt"
    assert node1[5] == "0"

    # ttnn::prim::permute
    node4 = captured_graph[4]["arguments"]
    assert (
        node4[0]
        == "Tensor(storage=DeviceStorage(memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt)),tensor_spec=TensorSpec(logical_shape=Shape([1, 2048, 4, 128]),tensor_layout=TensorLayout(dtype=DataType::BFLOAT16,page_config=PageConfig(config=RowMajorPageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt),alignment=Alignment([1]))))"
    )
    assert node4[1] == "SmallVector([0, 2, 1, 3])"
    assert (
        node4[2]
        == "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt)"
    )
    assert node4[3] == "[ unsupported type , std::__1::reference_wrapper<std::__1::nullopt_t const>]"
    assert node4[4] == "0"

    # PermuteDeviceOperation
    node6 = captured_graph[6]["arguments"]
    assert (
        node6[0]
        == "[ unsupported type , std::__1::reference_wrapper<ttnn::operations::data_movement::PermuteDeviceOperation::operation_attributes_t const>]"
    )
    assert (
        node6[1]
        == "[ unsupported type , std::__1::reference_wrapper<ttnn::operations::data_movement::PermuteDeviceOperation::tensor_args_t const>]"
    )

    # tt::tt_metal::create_device_tensor
    node7 = captured_graph[7]["arguments"]
    assert node7[0] == "Shape([1, 4, 2048, 128])"
    assert node7[1] == "DataType::BFLOAT16"
    assert node7[2] == "Row Major"
    assert node7[3] == "[ unsupported type , std::__1::reference_wrapper<tt::tt_metal::IDevice*>]"
    assert (
        node7[4]
        == "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt)"
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
        == "Tensor(storage=DeviceStorage(memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)),tensor_spec=TensorSpec(logical_shape=Shape([1, 1, 1, 32]),tensor_layout=TensorLayout(dtype=DataType::BFLOAT16,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt),alignment=Alignment([32, 32]))))"
    )
    assert (
        node1[1]
        == "Tensor(storage=DeviceStorage(memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)),tensor_spec=TensorSpec(logical_shape=Shape([1, 1, 1, 32]),tensor_layout=TensorLayout(dtype=DataType::BFLOAT16,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt),alignment=Alignment([32, 32]))))"
    )
    assert node1[2] == "nullopt"
    assert node1[3] == "DataType::BFLOAT16"
    assert node1[4] == "nullopt"
    assert (
        node1[5]
        == "[ unsupported type , std::__1::reference_wrapper<std::__1::optional<std::__1::variant<ttnn::GrayskullComputeKernelConfig, ttnn::WormholeComputeKernelConfig>> const>]"
    )

    # ttnn::prim::moreh_dot
    node6 = captured_graph[6]["arguments"]
    assert (
        node6[0]
        == "Tensor(storage=DeviceStorage(memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)),tensor_spec=TensorSpec(logical_shape=Shape([1, 1, 1, 32]),tensor_layout=TensorLayout(dtype=DataType::BFLOAT16,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt),alignment=Alignment([32, 32]))))"
    )
    assert (
        node6[1]
        == "Tensor(storage=DeviceStorage(memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)),tensor_spec=TensorSpec(logical_shape=Shape([1, 1, 1, 32]),tensor_layout=TensorLayout(dtype=DataType::BFLOAT16,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt),alignment=Alignment([32, 32]))))"
    )
    assert node6[2] == "nullopt"
    assert node6[3] == "DataType::BFLOAT16"
    assert node6[4] == "nullopt"
    assert (
        node6[5]
        == "[ unsupported type , std::__1::reference_wrapper<std::__1::optional<std::__1::variant<ttnn::GrayskullComputeKernelConfig, ttnn::WormholeComputeKernelConfig>> const>]"
    )

    # MorehDotOperation
    node9 = captured_graph[9]["arguments"]
    assert (
        node9[0]
        == "[ unsupported type , std::__1::reference_wrapper<ttnn::operations::moreh::moreh_dot::MorehDotOperation::operation_attributes_t const>]"
    )
    assert (
        node9[1]
        == "[ unsupported type , std::__1::reference_wrapper<ttnn::operations::moreh::moreh_dot::MorehDotOperation::tensor_args_t const>]"
    )

    # tt::tt_metal::create_device_tensor
    node10 = captured_graph[10]["arguments"]
    assert node10[0] == "Shape([1, 1, 1, 1])"
    assert node10[1] == "DataType::BFLOAT16"
    assert node10[2] == "Tile"
    assert node10[3] == "[ unsupported type , std::__1::reference_wrapper<tt::tt_metal::IDevice*>]"
    assert (
        node10[4]
        == "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)"
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
        == "Tensor(storage=DeviceStorage(memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)),tensor_spec=TensorSpec(logical_shape=Shape([32, 32]),tensor_layout=TensorLayout(dtype=DataType::INT32,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt),alignment=Alignment([32, 32]))))"
    )
    assert node1[1] == "[ unsupported type , std::__1::reference_wrapper<std::__1::variant<float, int>>]"
    assert node1[2] == "nullopt"
    assert node1[3] == "nullopt"
    assert node1[4] == "nullopt"

    # ttnn::prim::moreh_full_like
    node4 = captured_graph[4]["arguments"]
    assert (
        node4[0]
        == "Tensor(storage=DeviceStorage(memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)),tensor_spec=TensorSpec(logical_shape=Shape([32, 32]),tensor_layout=TensorLayout(dtype=DataType::INT32,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt),alignment=Alignment([32, 32]))))"
    )
    assert node4[1] == "[ unsupported type , std::__1::reference_wrapper<std::__1::variant<float, int> const>]"
    assert node4[2] == "nullopt"
    assert node4[3] == "nullopt"
    assert node4[4] == "nullopt"

    # FullLikeOperation
    node6 = captured_graph[6]["arguments"]
    assert (
        node6[0]
        == "[ unsupported type , std::__1::reference_wrapper<ttnn::operations::full_like::FullLikeOperation::operation_attributes_t const>]"
    )
    assert (
        node6[1]
        == "[ unsupported type , std::__1::reference_wrapper<ttnn::operations::full_like::FullLikeOperation::tensor_args_t const>]"
    )

    # tt::tt_metal::create_device_tensor
    node7 = captured_graph[7]["arguments"]
    assert node7[0] == "Shape([32, 32])"
    assert node7[1] == "DataType::INT32"
    assert node7[2] == "Tile"
    assert node7[3] == "[ unsupported type , std::__1::reference_wrapper<tt::tt_metal::IDevice*>]"
    assert (
        node7[4]
        == "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)"
    )
