# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pathlib
import pytest
import torch
import ttnn
from ttnn.graph_tracer_utils import GraphTracerUtils
from ttnn.operations.conv2d import Conv2dConfig


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


def test_program_cache_invalidation_across_dispatch_modes(device):
    def test_conv(device):
        weights_shape = (32, 3, 3, 3)
        bias_shape = (1, 1, 1, 32)

        conv_params = {
            "in_channels": 3,
            "out_channels": 32,
            "batch_size": 1,
            "input_height": 320,
            "input_width": 320,
            "kernel_size": (3, 3),
            "stride": (1, 1),
            "padding": (1, 1),
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": Conv2dConfig(
                weights_dtype=ttnn.bfloat8_b,
                activation=None,
                deallocate_activation=True,
                reallocate_halo_output=True,
                config_tensors_in_dram=True,
                act_block_h_override=128,
                act_block_w_div=1,
                reshard_if_not_optimal=False,
                override_sharding_config=False,
                shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                core_grid=None,
                transpose_shards=False,
                output_layout=ttnn.TILE_LAYOUT,
                enable_act_double_buffer=True,
                enable_weights_double_buffer=False,
                full_inner_dim=False,
                enable_kernel_stride_folding=False,
                enable_activation_reuse=False,
            ),
        }

        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=False,
        )

        torch_input = torch.randn([1, 1, 102400, 16]).bfloat16()
        ttnn_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                    [1600, 16],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )

        weight = torch.randn(weights_shape).bfloat16()
        bias = torch.randn(bias_shape).bfloat16()
        ttnn_weights = ttnn.from_torch(weight, dtype=ttnn.float32)
        ttnn_bias = ttnn.from_torch(bias, dtype=ttnn.float32)

        [x, [output_height, output_width], _] = ttnn.conv2d(
            input_tensor=ttnn_input,
            weight_tensor=ttnn_weights,
            bias_tensor=ttnn_bias,
            **conv_params,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat8_b,
        )

    try:
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
        test_conv(device)
        ttnn.graph.end_graph_capture()
        test_conv(device)
    except Exception as e:
        print(f"Error during test_conv: {e}")
        assert False

    assert True


# Tests for new features: stack traces, buffer pages, full tensor info


def test_stack_trace_control():
    """Test stack trace enable/disable API"""
    # Default should be disabled
    assert not ttnn.graph.is_stack_trace_enabled()

    # Enable
    ttnn.graph.enable_stack_traces()
    assert ttnn.graph.is_stack_trace_enabled()

    # Disable
    ttnn.graph.disable_stack_traces()
    assert not ttnn.graph.is_stack_trace_enabled()


def test_buffer_pages_control():
    """Test buffer pages enable/disable API"""
    # Default should be disabled
    assert not ttnn.graph.is_buffer_pages_enabled()

    # Enable
    ttnn.graph.enable_buffer_pages()
    assert ttnn.graph.is_buffer_pages_enabled()

    # Disable
    ttnn.graph.disable_buffer_pages()
    assert not ttnn.graph.is_buffer_pages_enabled()


@pytest.mark.parametrize("mode", [ttnn.graph.RunMode.NO_DISPATCH, ttnn.graph.RunMode.NORMAL])
def test_stack_traces_captured_when_enabled(device, mode):
    """Test that stack traces are captured when enabled"""
    ttnn.graph.enable_stack_traces()

    try:
        ttnn.graph.begin_graph_capture(mode)
        input_tensor = ttnn.from_torch(torch.rand((32,), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.relu(input_tensor)
        captured_graph = ttnn.graph.end_graph_capture()
    finally:
        ttnn.graph.disable_stack_traces()

    # Find function_start nodes and check for stack traces
    found_stack_trace = False
    for node in captured_graph:
        if node["node_type"] == "function_start":
            if "stack_trace" in node and len(node["stack_trace"]) > 0:
                found_stack_trace = True
                # Stack trace should be a list of strings
                assert isinstance(node["stack_trace"], list)
                assert all(isinstance(entry, str) for entry in node["stack_trace"])
                break

    assert found_stack_trace, "Expected stack traces to be captured when enabled"


@pytest.mark.parametrize("mode", [ttnn.graph.RunMode.NO_DISPATCH, ttnn.graph.RunMode.NORMAL])
def test_stack_traces_not_captured_when_disabled(device, mode):
    """Test that stack traces are NOT captured when disabled"""
    ttnn.graph.disable_stack_traces()

    ttnn.graph.begin_graph_capture(mode)
    input_tensor = ttnn.from_torch(torch.rand((32,), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    captured_graph = ttnn.graph.end_graph_capture()

    # No function_start node should have non-empty stack traces
    for node in captured_graph:
        if node["node_type"] == "function_start":
            stack_trace = node.get("stack_trace", [])
            assert len(stack_trace) == 0, "Stack trace should be empty when disabled"


@pytest.mark.parametrize("mode", [ttnn.graph.RunMode.NO_DISPATCH, ttnn.graph.RunMode.NORMAL])
def test_full_tensor_info_captured(device, mode):
    """Test that full tensor info (dtype, layout, memory_config, etc.) is captured"""
    ttnn.graph.begin_graph_capture(mode)
    input_tensor = ttnn.from_torch(
        torch.rand((1, 1, 32, 32), dtype=torch.bfloat16),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    captured_graph = ttnn.graph.end_graph_capture()

    # Find tensor nodes and verify they have full info
    found_tensor_with_full_info = False
    for node in captured_graph:
        if node["node_type"] == "tensor":
            params = node["params"]
            # Check for required fields
            assert "tensor_id" in params
            assert "shape" in params

            # Check for extended tensor info (dtype, layout)
            if "dtype" in params:
                found_tensor_with_full_info = True
                assert isinstance(params["dtype"], str)
                assert "layout" in params
                assert isinstance(params["layout"], str)

                # For device tensors, check device-specific fields
                if "device_id" in params:
                    assert isinstance(params["device_id"], str)
                    assert "address" in params
                    assert isinstance(params["address"], str)

    assert found_tensor_with_full_info, "Expected at least one tensor with full info"


@pytest.mark.parametrize("mode", [ttnn.graph.RunMode.NO_DISPATCH, ttnn.graph.RunMode.NORMAL])
def test_duration_captured(device, mode):
    """Test that durations are captured for function_end and capture_end nodes"""
    ttnn.graph.begin_graph_capture(mode)
    input_tensor = ttnn.from_torch(torch.rand((32,), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.relu(input_tensor)
    captured_graph = ttnn.graph.end_graph_capture()

    # Check function_end nodes have duration
    found_function_end_with_duration = False
    for node in captured_graph:
        if node["node_type"] == "function_end":
            if "duration_ns" in node:
                found_function_end_with_duration = True
                assert isinstance(node["duration_ns"], int)
                assert node["duration_ns"] >= 0

    # Check capture_end node has duration
    found_capture_end_with_duration = False
    for node in captured_graph:
        if node["node_type"] == "capture_end":
            if "duration_ns" in node:
                found_capture_end_with_duration = True
                assert isinstance(node["duration_ns"], int)
                assert node["duration_ns"] >= 0

    assert found_function_end_with_duration, "Expected function_end nodes to have duration"
    assert found_capture_end_with_duration, "Expected capture_end node to have duration"


def test_get_current_report(device):
    """Test get_current_report API during active capture"""
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)

    input_tensor = ttnn.from_torch(
        torch.rand((1, 1, 32, 32), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device
    )

    # Get report while capture is active
    report = ttnn.graph.get_current_report()

    ttnn.graph.end_graph_capture()

    # Verify report structure
    assert "version" in report
    assert report["version"] == ttnn.graph.REPORT_VERSION
    assert "graph" in report
    assert "devices" in report
    assert "metadata" in report

    # Graph should have some nodes
    assert len(report["graph"]) > 0

    # Metadata should have timestamp
    assert "capture_timestamp_ns" in report["metadata"]
    assert report["metadata"]["capture_timestamp_ns"] > 0
