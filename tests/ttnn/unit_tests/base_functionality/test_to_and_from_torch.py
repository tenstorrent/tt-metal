# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device


def test_from_torch_none():
    assert ttnn.from_torch(None) is None


@pytest.mark.parametrize(
    "shape",
    [
        (2,),
        (2, 3),
        (2, 3, 4),
        (2, 3, 4, 5),
        (2, 3, 4, 5, 6),
        (2, 3, 4, 5, 6, 7),
        (2, 3, 4, 5, 6, 7, 8),
        (2, 3, 4, 5, 6, 7, 8, 9),
    ],
)
def test_to_and_from(shape):
    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_input_tensor)
    torch_output_tensor = ttnn.to_torch(tensor)
    assert torch.allclose(torch_input_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (2, 3, 4),
        (2, 3, 4, 5),
        (2, 3, 4, 5, 6),
        (2, 3, 4, 5, 6, 7),
        (2, 3, 4, 5, 6, 7, 8),
        (2, 3, 4, 5, 6, 7, 8, 9),
    ],
)
def test_to_and_from_using_tile_layout(shape):
    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
    tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT)
    torch_output_tensor = ttnn.to_torch(tensor)
    assert torch.allclose(torch_input_tensor, torch_output_tensor)


@pytest.mark.parametrize("height", [7])
@pytest.mark.parametrize("width", [3])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.int32, ttnn.uint8])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_to_and_from_2D(height, width, dtype, layout):
    if (dtype == ttnn.bfloat8_b or dtype == ttnn.bfloat4_b) and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("ROW_MAJOR_LAYOUT not supported for bfloat8_b and bfloat4_b")

    if dtype == ttnn.uint8:
        torch_input_tensor = torch.randint(0, 255, (height, width)).to(torch.uint8)
    elif dtype == ttnn.int32:
        torch_input_tensor = torch.randint(-100, 100, (height, width)).to(torch.int32)
    else:
        torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)

    tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=layout)
    torch_output_tensor = ttnn.to_torch(tensor).to(torch_input_tensor.dtype)

    allclose_kwargs = {}
    if dtype == ttnn.bfloat4_b:
        allclose_kwargs["atol"] = 0.01
        allclose_kwargs["rtol"] = 0.3
        assert_with_pcc(torch_input_tensor, torch_output_tensor, pcc=0.9)
    else:
        if dtype == ttnn.bfloat8_b:
            allclose_kwargs["atol"] = 1e-2
        assert torch.allclose(torch_input_tensor, torch_output_tensor, **allclose_kwargs)


@pytest.mark.skip(reason="GH Issue #15719")
def test_from_torch_large(device):
    torch_x = torch.rand((2048, 1024, 32, 32), dtype=torch.bfloat16)
    x_tensor = ttnn.from_torch(torch_x, layout=ttnn.TILE_LAYOUT)
    x_tensor = ttnn.to_torch(x_tensor)
    assert torch.allclose(torch_x, x_tensor)


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (1),
        (2),
        (127),
        (0),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pad_value", [None, 1])
def test_to_for_01_rank(shape, layout, dtype, pad_value):
    if pad_value != None and layout != ttnn.TILE_LAYOUT:
        pytest.skip("Pad value is only supported for tile layout")
    torch_input_tensor = torch.rand(shape, dtype=dtype)
    tensor = ttnn.from_torch(torch_input_tensor, layout=layout, pad_value=pad_value)
    torch_output_tensor = ttnn.to_torch(tensor)
    assert torch_input_tensor.shape == torch_output_tensor.shape
    assert torch.allclose(torch_input_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (1),
        (2),
        (127),
        (0),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pad_value", [None, 1])
def test_to_for_01_rank_on_device(device, shape, layout, dtype, pad_value):
    if pad_value != None and layout != ttnn.TILE_LAYOUT:
        pytest.skip("Pad value is only supported for tile layout")
    torch_input_tensor = torch.rand(shape, dtype=dtype)
    tensor = ttnn.from_torch(torch_input_tensor, layout=layout, pad_value=pad_value, device=device)
    torch_output_tensor = ttnn.to_torch(tensor)
    assert torch_input_tensor.shape == torch_output_tensor.shape
    assert torch.allclose(torch_input_tensor, torch_output_tensor)


# Regression test for issue #31136: to_torch with mesh_composer=None on device-sharded tensor
# Issue: https://github.com/tenstorrent/tt-metal/issues/31136
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 3, 3),
        (1, 1, 32, 32),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_to_torch_with_mesh_composer_none(device, shape, layout):
    """Regression test for issue #31136: to_torch with mesh_composer=None on device-sharded tensor"""
    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)

    ttnn_tensor = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        dtype=ttnn.bfloat16,
        layout=layout,
    )

    torch_output_tensor = ttnn_tensor.to_torch(mesh_composer=None)

    assert torch.allclose(torch_input_tensor, torch_output_tensor)
