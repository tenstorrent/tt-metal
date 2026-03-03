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

    t = full()

    assert t.shape == shape
    assert t.dtype == dtype
    assert tensor.to_torch(t).eq(value).all()


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_arange(mesh_device: ttnn.MeshDevice) -> None:
    dtype = ttnn.bfloat16
    start, end, step = 10, 20, 2

    arange = Tracer(
        lambda: tensor.arange(start, end, step, dtype=dtype, device=mesh_device),
        device=mesh_device,
    )

    t = arange()
    t_ref = torch.arange(start, end, step)

    assert t.dtype == dtype
    assert tensor.to_torch(t).eq(t_ref).all()
