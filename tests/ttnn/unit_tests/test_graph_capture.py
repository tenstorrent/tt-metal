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

    # ttnn:transpose
    assert (
        captured_graph[1]["arguments"]
        == "[ \x00, std::__1::reference_wrapper<tt::stl::StrongType<unsigned char, ttnn::QueueIdTag>>],[ Tensor(storage=DeviceStorage(memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt)),tensor_spec=TensorSpec(logical_shape=Shape([1, 2048, 4, 128]),tensor_layout=TensorLayout(dtype=BFLOAT16,page_config=PageConfig(config=RowMajorPageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt),alignment=Alignment([1])))), std::__1::reference_wrapper<tt::tt_metal::Tensor const>],[ 1, std::__1::reference_wrapper<long const>],[ 2, std::__1::reference_wrapper<long const>],[ nullopt, std::__1::reference_wrapper<std::__1::optional<tt::tt_metal::MemoryConfig> const>],[ 0, std::__1::reference_wrapper<std::__1::optional<float> const>]"
    )
    # ttnn::prim::permute
    assert (
        captured_graph[4]["arguments"]
        == "[ Tensor(storage=DeviceStorage(memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt)),tensor_spec=TensorSpec(logical_shape=Shape([1, 2048, 4, 128]),tensor_layout=TensorLayout(dtype=BFLOAT16,page_config=PageConfig(config=RowMajorPageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt),alignment=Alignment([1])))), std::__1::reference_wrapper<tt::tt_metal::Tensor const>],[ SmallVector([0, 2, 1, 3]), std::__1::reference_wrapper<tt::stl::SmallVector<unsigned int, 8ul>>],[ MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt), std::__1::reference_wrapper<tt::tt_metal::MemoryConfig const>],[ unsupported type , std::__1::reference_wrapper<std::__1::nullopt_t const>],[ 0, std::__1::reference_wrapper<std::__1::optional<float> const>]"
    )
    # PermuteDeviceOperation
    assert (
        captured_graph[6]["arguments"]
        == "[ unsupported type , std::__1::reference_wrapper<ttnn::operations::data_movement::PermuteDeviceOperation::operation_attributes_t const>],[ unsupported type , std::__1::reference_wrapper<ttnn::operations::data_movement::PermuteDeviceOperation::tensor_args_t const>]"
    )
    # tt::tt_metal::create_device_tensor
    assert (
        captured_graph[7]["arguments"]
        == "[ Shape([1, 4, 2048, 128]), std::__1::reference_wrapper<tt::tt_metal::Shape const>],[ BFLOAT16, std::__1::reference_wrapper<tt::tt_metal::DataType>],[ Row Major, std::__1::reference_wrapper<tt::tt_metal::Layout>],[ unsupported type , std::__1::reference_wrapper<tt::tt_metal::v0::IDevice*>],[ MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt), std::__1::reference_wrapper<tt::tt_metal::MemoryConfig const>]"
    )


def test_graph_capture_without_memory_config(device):
    # Create input tensor
    input_shape = (1, 1, 1, 32)
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_other = torch.rand(input_shape, dtype=torch.bfloat16)

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
    assert (
        captured_graph[1]["arguments"]
        == "[ Tensor(storage=DeviceStorage(memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)),tensor_spec=TensorSpec(logical_shape=Shape([1, 1, 1, 32]),tensor_layout=TensorLayout(dtype=BFLOAT16,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt),alignment=Alignment([32, 32])))), std::__1::reference_wrapper<tt::tt_metal::Tensor const>],[ Tensor(storage=DeviceStorage(memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)),tensor_spec=TensorSpec(logical_shape=Shape([1, 1, 1, 32]),tensor_layout=TensorLayout(dtype=BFLOAT16,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt),alignment=Alignment([32, 32])))), std::__1::reference_wrapper<tt::tt_metal::Tensor const>],[ nullopt, std::__1::reference_wrapper<std::__1::optional<tt::tt_metal::Tensor> const>],[ BFLOAT16, std::__1::reference_wrapper<std::__1::optional<tt::tt_metal::DataType> const>],[ nullopt, std::__1::reference_wrapper<std::__1::optional<tt::tt_metal::MemoryConfig> const>],[ unsupported type , std::__1::reference_wrapper<std::__1::optional<std::__1::variant<ttnn::GrayskullComputeKernelConfig, ttnn::WormholeComputeKernelConfig>> const>]"
    )
    # ttnn::prim::moreh_dot
    assert (
        captured_graph[6]["arguments"]
        == "[ Tensor(storage=DeviceStorage(memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)),tensor_spec=TensorSpec(logical_shape=Shape([1, 1, 1, 32]),tensor_layout=TensorLayout(dtype=BFLOAT16,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt),alignment=Alignment([32, 32])))), std::__1::reference_wrapper<tt::tt_metal::Tensor const>],[ Tensor(storage=DeviceStorage(memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)),tensor_spec=TensorSpec(logical_shape=Shape([1, 1, 1, 32]),tensor_layout=TensorLayout(dtype=BFLOAT16,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt),alignment=Alignment([32, 32])))), std::__1::reference_wrapper<tt::tt_metal::Tensor const>],[ nullopt, std::__1::reference_wrapper<std::__1::optional<tt::tt_metal::Tensor> const>],[ BFLOAT16, std::__1::reference_wrapper<std::__1::optional<tt::tt_metal::DataType> const>],[ nullopt, std::__1::reference_wrapper<std::__1::optional<tt::tt_metal::MemoryConfig> const>],[ unsupported type , std::__1::reference_wrapper<std::__1::optional<std::__1::variant<ttnn::GrayskullComputeKernelConfig, ttnn::WormholeComputeKernelConfig>> const>]"
    )
    # MorehDotOperation
    assert (
        captured_graph[9]["arguments"]
        == "[ unsupported type , std::__1::reference_wrapper<ttnn::operations::moreh::moreh_dot::MorehDotOperation::operation_attributes_t const>],[ unsupported type , std::__1::reference_wrapper<ttnn::operations::moreh::moreh_dot::MorehDotOperation::tensor_args_t const>]"
    )
    # tt::tt_metal::create_device_tensor
    assert (
        captured_graph[10]["arguments"]
        == "[ Shape([1, 1, 1, 1]), std::__1::reference_wrapper<tt::tt_metal::Shape const>],[ BFLOAT16, std::__1::reference_wrapper<tt::tt_metal::DataType>],[ Tile, std::__1::reference_wrapper<tt::tt_metal::Layout>],[ unsupported type , std::__1::reference_wrapper<tt::tt_metal::v0::IDevice*>],[ MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt), std::__1::reference_wrapper<tt::tt_metal::MemoryConfig const>]"
    )


def test_graph_capture_without_dtype(device):
    torch_input = torch.randint(0, 100, (32, 32), dtype=torch.int32)
    tt_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    tt_output = ttnn.moreh_full_like(tt_input, 3)
    captured_graph = ttnn.graph.end_graph_capture()

    # ttnn::moreh_full_like
    assert (
        captured_graph[1]["arguments"]
        == "[ Tensor(storage=DeviceStorage(memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)),tensor_spec=TensorSpec(logical_shape=Shape([32, 32]),tensor_layout=TensorLayout(dtype=INT32,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt),alignment=Alignment([32, 32])))), std::__1::reference_wrapper<tt::tt_metal::Tensor const>],[ unsupported type , std::__1::reference_wrapper<std::__1::variant<float, int>>],[ nullopt, std::__1::reference_wrapper<std::__1::optional<tt::tt_metal::DataType> const>],[ nullopt, std::__1::reference_wrapper<std::__1::optional<tt::tt_metal::Layout> const>],[ nullopt, std::__1::reference_wrapper<std::__1::optional<tt::tt_metal::MemoryConfig> const>]"
    )
    # ttnn::prim::moreh_full_like
    assert (
        captured_graph[4]["arguments"]
        == "[ Tensor(storage=DeviceStorage(memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)),tensor_spec=TensorSpec(logical_shape=Shape([32, 32]),tensor_layout=TensorLayout(dtype=INT32,page_config=PageConfig(config=TilePageConfig(tile=Tile(tile_shape={32, 32},face_shape={16, 16},num_faces=4))),memory_config=MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt),alignment=Alignment([32, 32])))), std::__1::reference_wrapper<tt::tt_metal::Tensor const>],[ unsupported type , std::__1::reference_wrapper<std::__1::variant<float, int> const>],[ nullopt, std::__1::reference_wrapper<std::__1::optional<tt::tt_metal::DataType> const>],[ nullopt, std::__1::reference_wrapper<std::__1::optional<tt::tt_metal::Layout> const>],[ nullopt, std::__1::reference_wrapper<std::__1::optional<tt::tt_metal::MemoryConfig> const>]"
    )
    # FullLikeOperation
    assert (
        captured_graph[6]["arguments"]
        == "[ unsupported type , std::__1::reference_wrapper<ttnn::operations::full_like::FullLikeOperation::operation_attributes_t const>],[ unsupported type , std::__1::reference_wrapper<ttnn::operations::full_like::FullLikeOperation::tensor_args_t const>]"
    )
    # tt::tt_metal::create_device_tensor
    assert (
        captured_graph[7]["arguments"]
        == "[ Shape([32, 32]), std::__1::reference_wrapper<tt::tt_metal::Shape const>],[ INT32, std::__1::reference_wrapper<tt::tt_metal::DataType>],[ Tile, std::__1::reference_wrapper<tt::tt_metal::Layout>],[ unsupported type , std::__1::reference_wrapper<tt::tt_metal::v0::IDevice*>],[ MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt), std::__1::reference_wrapper<tt::tt_metal::MemoryConfig const>]"
    )
