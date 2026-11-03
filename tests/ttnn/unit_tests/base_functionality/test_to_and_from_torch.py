# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
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


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
@pytest.mark.parametrize("K, N", [(64, 128), (128, 64)])
def test_col_tilize_to_device_and_back(device, dtype, K, N):
    """col_tilize=True should produce the same on-device tensor as manually transposing first."""
    torch.manual_seed(0)
    torch_W = torch.randn(1, 1, K, N, dtype=torch.bfloat16)

    # col_tilize path: transpose happens inside from_torch, result shape is (1,1,N,K)
    tt_col = ttnn.from_torch(torch_W, dtype=dtype, col_tilize=True, device=device)
    result_col = ttnn.to_torch(tt_col)

    # manual path: transpose in torch first, then send to device normally
    tt_manual = ttnn.from_torch(torch_W.transpose(-1, -2).contiguous(), dtype=dtype, device=device)
    result_manual = ttnn.to_torch(tt_manual)

    assert result_col.shape == result_manual.shape
    pcc_threshold = 0.99 if dtype == ttnn.bfloat8_b else 0.98
    assert_with_pcc(result_manual, result_col, pcc=pcc_threshold)


# #63423: private file mappings above 32 MiB upload correctly; a read-only pin of them used to hang the process.
@pytest.mark.skipif("TT_METAL_SIMULATOR" in os.environ, reason="its thread timeout would end a slow simulator session")
@pytest.mark.timeout(120, method="thread")
@pytest.mark.parametrize("source", ["safetensors", "torch_load_mmap", "numpy_memmap_copy_on_write"])
def test_from_torch_private_file_mapping_above_pin_threshold(device, tmp_path, source):
    torch.manual_seed(0)
    torch_input = torch.randn((7282, 2304), dtype=torch.bfloat16)  # 33,555,456 bytes, just above 32 MiB
    if source == "safetensors":
        safetensors = pytest.importorskip("safetensors")
        from safetensors.torch import save_file

        save_file({"w": torch_input}, str(tmp_path / "w.safetensors"))
        mapped = safetensors.safe_open(str(tmp_path / "w.safetensors"), "pt").get_tensor("w")
    elif source == "torch_load_mmap":
        torch.save(torch_input, tmp_path / "w.pt")
        mapped = torch.load(tmp_path / "w.pt", mmap=True, weights_only=True)
    else:
        torch_input.view(torch.int16).numpy().tofile(tmp_path / "w.bin")
        memmap = np.memmap(tmp_path / "w.bin", mode="c", dtype=np.int16, shape=tuple(torch_input.shape))
        mapped = torch.from_numpy(memmap).view(torch.bfloat16)

    tensor = ttnn.from_torch(mapped, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    assert torch.equal(ttnn.to_torch(tensor), torch_input)
