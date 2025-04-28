# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn as nn
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, divup


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],
        [5, 96, 64],
    ],
)
def test_zeros_like(device, input_shape):
    torch_input_tensor = torch.rand((input_shape), dtype=torch.bfloat16)
    torch_output_tensor = torch.zeros_like(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device)
    output_tensor = ttnn.zeros_like(input_tensor)
    assert ttnn.is_tensor_storage_on_device(output_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],
        [5, 96, 64],
    ],
)
def test_ones_like(device, input_shape):
    torch_input_tensor = torch.rand((input_shape), dtype=torch.bfloat16)
    torch_output_tensor = torch.ones_like(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.ones_like(input_tensor)
    assert ttnn.is_tensor_storage_on_device(output_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "input_shape",
    [[32, 32], [5, 96, 64], [1, 2, 64, 64], [1, 2, 4, 64, 64]],
)
@pytest.mark.parametrize(
    "fill_value",
    [-5.25, 0, 1.0],
)
def test_full_like(device, input_shape, fill_value):
    torch_input_tensor = torch.rand((input_shape), dtype=torch.bfloat16)
    torch_output_tensor = torch.full_like(torch_input_tensor, fill_value)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.full_like(input_tensor, fill_value=fill_value)
    assert ttnn.is_tensor_storage_on_device(output_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "input_shape",
    [[32, 32], [5, 96, 64], [1, 2, 64, 64], [1, 2, 4, 64, 64]],
)
@pytest.mark.parametrize(
    "fill_value",
    [-5.25, 0, 1.0],
)
def test_full_like_bf8b(device, input_shape, fill_value):
    torch_input_tensor = torch.rand((input_shape), dtype=torch.bfloat16)
    torch_output_tensor = torch.full_like(torch_input_tensor, fill_value)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.full_like(input_tensor, fill_value=fill_value)
    assert ttnn.is_tensor_storage_on_device(output_tensor)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor).to(torch.bfloat16)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],
        [5, 96, 64],
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [-5.25, 0, 1.0],
)
@pytest.mark.parametrize(
    "layout",
    [ttnn.Layout.ROW_MAJOR, ttnn.Layout.TILE],
)
def test_full_like_opt_tensor(device, input_shape, fill_value, layout):
    torch_input_tensor = torch.rand((input_shape), dtype=torch.bfloat16)
    torch_output_tensor = torch.full_like(torch_input_tensor, fill_value)

    opt_tensor = torch.ones(input_shape, dtype=torch.bfloat16)
    opt_tensor = ttnn.from_torch(
        opt_tensor, ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout)
    input_tensor = ttnn.to_device(input_tensor, device)

    cq_id = 0
    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.full_like(input_tensor, fill_value=fill_value, optional_tensor=opt_tensor, queue_id=cq_id)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    assert ttnn.is_tensor_storage_on_device(opt_tensor)
    opt_tensor = ttnn.from_device(opt_tensor)
    opt_tensor = ttnn.to_torch(opt_tensor)

    assert_with_pcc(torch_output_tensor, opt_tensor, 0.9999)
    assert torch.allclose(torch_output_tensor, opt_tensor)


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],
        [5, 96, 64],
    ],
)
def test_ones(device, input_shape):
    torch_tensor = torch.ones(input_shape, dtype=torch.bfloat16)

    tensor = ttnn.ones(input_shape, device=device)
    assert ttnn.is_tensor_storage_on_device(tensor)
    tensor = ttnn.to_torch(tensor)

    assert_with_pcc(torch_tensor, tensor, 0.9999)
    assert torch.allclose(torch_tensor, tensor)


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],
        [5, 96, 64],
    ],
)
def test_zeros(device, input_shape):
    torch_tensor = torch.zeros(input_shape, dtype=torch.bfloat16)

    tensor = ttnn.zeros(input_shape, device=device)
    assert ttnn.is_tensor_storage_on_device(tensor)
    tensor = ttnn.to_torch(tensor)

    assert_with_pcc(torch_tensor, tensor, 0.9999)
    assert torch.allclose(torch_tensor, tensor)


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],
        [5, 96, 64],
        [1, 50257],
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [-5.25, 0, 1.0],
)
@pytest.mark.parametrize(
    "layout",
    [ttnn.Layout.ROW_MAJOR, ttnn.Layout.TILE],
)
def test_full(device, input_shape, fill_value, layout):
    torch_tensor = torch.full(input_shape, dtype=torch.bfloat16, fill_value=fill_value)

    tensor = ttnn.full(input_shape, device=device, fill_value=fill_value, layout=layout)
    assert ttnn.is_tensor_storage_on_device(tensor)
    tensor = ttnn.to_torch(tensor)

    assert_with_pcc(torch_tensor, tensor, 0.9999)
    assert torch.allclose(torch_tensor, tensor)


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],
        [5, 96, 64],
        [1, 50257],
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [-5.25, 0, 2.5, 9],
)
@pytest.mark.parametrize(
    "layout",
    [ttnn.Layout.ROW_MAJOR, ttnn.Layout.TILE],
)
def test_full_with_opt_tensor(device, input_shape, layout, fill_value):
    torch_tensor = torch.full(input_shape, dtype=torch.bfloat16, fill_value=fill_value)
    opt_tensor = torch.ones(input_shape, dtype=torch.bfloat16)
    opt_tensor = ttnn.from_torch(
        opt_tensor, ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    cq_id = 0
    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.full(input_shape, device=device, fill_value=fill_value, optional_tensor=opt_tensor, queue_id=cq_id)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))
    assert ttnn.is_tensor_storage_on_device(opt_tensor)
    opt_tensor = ttnn.to_torch(opt_tensor)

    assert_with_pcc(torch_tensor, opt_tensor, 0.9999)
    assert torch.allclose(torch_tensor, opt_tensor)


@pytest.mark.parametrize(
    "input_shape",
    [
        [32, 32],
        [5, 96, 64],
        [1, 50257],
    ],
)
@pytest.mark.parametrize(
    "fill_value",
    [-5.25, 0, 1.0],
)
@pytest.mark.parametrize(
    "layout",
    [ttnn.Layout.ROW_MAJOR, ttnn.Layout.TILE],
)
def test_full_multi_device(mesh_device, input_shape, fill_value, layout):
    torch_tensor = torch.full(input_shape, dtype=torch.bfloat16, fill_value=fill_value)

    tensor = ttnn.full(input_shape, device=mesh_device, fill_value=fill_value, layout=layout)
    assert ttnn.is_tensor_storage_on_device(tensor)
    output_tensors = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor.cpu())]
    for output_tensor in output_tensors:
        assert_with_pcc(torch_tensor, output_tensor, 0.9999)
        assert torch.allclose(torch_tensor, output_tensor)


@pytest.mark.parametrize(
    "start",
    [4, 8, 16, 0, 201, 135, 98],
)
@pytest.mark.parametrize(
    "end",
    [100, 103, 226, 300, 3, 1, 0],
)
@pytest.mark.parametrize(
    "step",
    [1, 2, 3, 5, 0, -1, -3, -4],
)
def test_arange(device, start, end, step):
    if (start > end and step > 0) or (start < end and step < 0) or (step == 0):
        pytest.skip(f"Skipping invalid case: start={start}, end={end}, step={step}")

    torch_output_tensor = torch.arange(start, end, step)

    output_tensor = ttnn.arange(start, end, step, ttnn.bfloat16, device)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize(
    "start",
    [4, 8, 16, 0, 201, 135, 98],
)
@pytest.mark.parametrize(
    "end",
    [100, 103, 226, 300, 3, 1, 0],
)
@pytest.mark.parametrize(
    "step",
    [1, 2, 3, 5, 0, -1, -3, -4],
)
def test_arange_multi_device(mesh_device, start, end, step):
    if (start > end and step > 0) or (start < end and step < 0) or (step == 0):
        pytest.skip(f"Skipping invalid case: start={start}, end={end}, step={step}")
    torch_output_tensor = torch.arange(start, end, step)

    output_tensor = ttnn.arange(
        start,
        end,
        step,
        ttnn.bfloat16,
        mesh_device,
    )
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensors = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(output_tensor.cpu())]
    for output_tensor in output_tensors:
        assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize(
    "input_shapes",
    [
        [1, 1, 32, 32],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
        [1, 3, 180, 64],
        [2, 640, 64, 64],
        [2, 1280, 64, 64],
    ],
)
def test_empty(device, input_shapes):
    torch_output_tensor = torch.empty((input_shapes), dtype=torch.bfloat16)

    output_tensor = ttnn.empty(input_shapes, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert list(torch_output_tensor.shape) == list(output_tensor.shape)


@pytest.mark.parametrize(
    "input_shapes",
    [
        [1, 1, 32, 32],
        [1, 1, 320, 384],
        [1, 3, 320, 384],
        [1, 3, 180, 64],
        [2, 640, 64, 64],
        [2, 1280, 64, 64],
    ],
)
def test_empty_multi_device(mesh_device, input_shapes):
    torch_output_tensor = torch.empty((input_shapes), dtype=torch.bfloat16)

    output_tensor = ttnn.empty(input_shapes, ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensors = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(output_tensor.cpu())]
    for output_tensor in output_tensors:
        assert list(torch_output_tensor.shape) == list(output_tensor.shape)


@pytest.mark.parametrize(
    "input_shapes",
    [
        [2, 1, 4, 4],  # 256x256
        [2, 1280, 8, 8],
        [2, 640, 16, 16],
        [2, 1280, 8, 8],  # 512x512
        [2, 1280, 16, 16],
        [2, 1280, 16, 16],
    ],
)
def test_empty_like(device, input_shapes):
    torch_input_tensor = torch.ones((input_shapes), dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.empty_like(input_tensor, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert list(torch_input_tensor.shape) == list(output_tensor.shape)


@pytest.mark.parametrize(
    "input_shapes",
    [
        [2, 1, 4, 4],  # 256x256
        [2, 1280, 8, 8],
        [2, 640, 16, 16],
        [2, 1280, 8, 8],  # 512x512
        [2, 1280, 16, 16],
        [2, 1280, 16, 16],
    ],
)
def test_empty_like_multi_device(mesh_device, input_shapes):
    torch_input_tensor = torch.empty((input_shapes), dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, mesh_device)
    output_tensor = ttnn.empty_like(input_tensor, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensors = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(output_tensor.cpu())]
    for output_tensor in output_tensors:
        assert list(torch_input_tensor.shape) == list(output_tensor.shape)


@pytest.mark.parametrize("input_shape, dtype", [([32, 32], ttnn.bfloat8_b), ((5, 96, 64), ttnn.bfloat8_b)])
def test_zeros_bfp8(device, input_shape, dtype):
    tensor = ttnn.zeros(input_shape, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    assert tensor.dtype == ttnn.bfloat8_b, f"Expected dtype {dtype}, but got {tensor.dtype}"
    assert tensor.storage_type() == ttnn.StorageType.DEVICE


@pytest.mark.parametrize("input_shape, dtype", [([32, 32], ttnn.bfloat4_b), ((5, 96, 64), ttnn.bfloat4_b)])
def test_zeros_bfp4(device, input_shape, dtype):
    tensor = ttnn.zeros(input_shape, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    assert tensor.dtype == ttnn.bfloat4_b, f"Expected dtype {dtype}, but got {tensor.dtype}"
    assert tensor.storage_type() == ttnn.StorageType.DEVICE
