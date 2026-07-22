# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

import ttnn

from ...utils import tensor
from ...utils.tracing import Tracer


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_full(mesh_device: ttnn.MeshDevice) -> None:
    shape = (32, 32)
    dtype = ttnn.bfloat16
    value = 10

    full = Tracer(
        lambda: tensor.full(shape, value, dtype=dtype, device=mesh_device),
        device=mesh_device,
    )

    result = full()
    ref = torch.full(shape, value, dtype=torch.bfloat16)

    assert result.dtype == dtype
    assert tuple(result.shape) == tuple(ref.shape)

    result_torch = tensor.to_torch(result)
    assert torch.allclose(result_torch, ref, atol=0, rtol=0)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_arange(mesh_device: ttnn.MeshDevice) -> None:
    dtype = ttnn.bfloat16
    start, end, step = 10, 20, 2

    arange = Tracer(
        lambda: tensor.arange(start, end, step, dtype=dtype, device=mesh_device),
        device=mesh_device,
    )

    result = arange()
    ref = torch.arange(start, end, step, dtype=torch.bfloat16)

    assert result.dtype == dtype
    assert tuple(result.shape) == tuple(ref.shape)

    result_torch = tensor.to_torch(result)
    assert torch.allclose(result_torch, ref, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("shape", "dim", "front", "back"),
    [
        # On device tile padding does not support front padding.
        # rank <= 4: direct ttnn.pad path
        ((2, 32, 64), 0, 0, 2),
        ((2, 32, 64), 1, 0, 3),
        # rank > 4, dim >= rank - 3: direct ttnn.pad path (last 3 dims)
        ((2, 2, 2, 32, 64), 3, 0, 2),
        ((2, 2, 2, 32, 64), -1, 0, 4),
        # rank > 4, dim < rank - 3: reshape workaround path (early dims)
        ((2, 2, 2, 32, 64), 0, 0, 2),
        ((2, 2, 2, 32, 64), 1, 0, 3),
    ],
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_pad_single(mesh_device: ttnn.MeshDevice, shape: tuple, dim: int, front: int, back: int) -> None:
    dtype = ttnn.bfloat16

    x_torch = torch.randn(shape, dtype=torch.bfloat16)
    x_tt = tensor.from_torch(x_torch, device=mesh_device, dtype=dtype)

    pad = Tracer(
        lambda: tensor.pad_single(x_tt, dim=dim, front=front, back=back),
        device=mesh_device,
    )

    result = pad()

    pad_per_dim = [(front, back) if i == (dim % len(shape)) else (0, 0) for i in reversed(range(len(shape)))]
    ref = torch.nn.functional.pad(x_torch, [v for pair in pad_per_dim for v in pair])

    assert result.dtype == dtype
    assert tuple(result.shape) == tuple(ref.shape)

    result_torch = tensor.to_torch(result)
    assert torch.allclose(result_torch, ref, atol=0, rtol=0)


@pytest.mark.parametrize("diagonal", [0, 1, -1])
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_tril(mesh_device: ttnn.MeshDevice, diagonal: int) -> None:
    shape = (1, 1, 64, 64)
    dtype = ttnn.bfloat16

    x_torch = torch.ones(shape, dtype=torch.bfloat16)
    x_tt = tensor.from_torch(x_torch, device=mesh_device, dtype=dtype)

    tril = Tracer(
        lambda: tensor.tril(x_tt, diagonal=diagonal),
        device=mesh_device,
    )

    result = tril()
    ref = torch.tril(x_torch, diagonal=diagonal)

    assert result.dtype == dtype
    assert tuple(result.shape) == tuple(ref.shape)

    result_torch = tensor.to_torch(result)
    assert torch.allclose(result_torch, ref, atol=0, rtol=0)


@pytest.mark.parametrize("diagonal", [0, 1, -1])
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_triu(mesh_device: ttnn.MeshDevice, diagonal: int) -> None:
    shape = (1, 1, 64, 64)
    dtype = ttnn.bfloat16

    x_torch = torch.ones(shape, dtype=torch.bfloat16)
    x_tt = tensor.from_torch(x_torch, device=mesh_device, dtype=dtype)

    triu = Tracer(
        lambda: tensor.triu(x_tt, diagonal=diagonal),
        device=mesh_device,
    )

    result = triu()
    ref = torch.triu(x_torch, diagonal=diagonal)

    assert result.dtype == dtype
    assert tuple(result.shape) == tuple(ref.shape)

    result_torch = tensor.to_torch(result)
    assert torch.allclose(result_torch, ref, atol=0, rtol=0)
