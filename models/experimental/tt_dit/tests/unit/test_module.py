# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from tempfile import TemporaryDirectory

import pytest
import torch
import ttnn

from ...layers.module import Module, Parameter


class TinyLinear(Module):
    def __init__(self, in_dim: int, out_dim: int, *, bias: bool, device: ttnn.MeshDevice) -> None:
        super().__init__()

        self.weight = Parameter(total_shape=[in_dim, out_dim], device=device)
        self.bias = Parameter(total_shape=[out_dim], device=device) if bias else None

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(x, self.weight.data, bias=self.bias.data if self.bias is not None else None)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"].transpose_(0, 1)


class TinyFeedForward(Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, device: ttnn.MeshDevice) -> None:
        super().__init__()

        self.linear1 = TinyLinear(in_dim, hidden_dim, device=device, bias=True)
        self.linear2 = TinyLinear(hidden_dim, out_dim, device=device, bias=True)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.linear1.forward(x)
        x = ttnn.silu(x)
        return self.linear2.forward(x)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_load_torch_state(mesh_device: ttnn.MeshDevice) -> None:
    model = TinyFeedForward(10, 20, 30, device=mesh_device)

    state_dict = {
        "linear1.weight": torch.randn([20, 10]),
        "linear1.bias": torch.randn([20]),
        "linear2.weight": torch.randn([30, 20]),
        "something.unexpected": torch.randn([1]),
    }

    missing, unexpected = model.load_torch_state_dict(state_dict, strict=False)

    assert missing == ["linear2.bias"]
    assert unexpected == ["something.unexpected"]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_save_and_load(mesh_device: ttnn.MeshDevice) -> None:
    model1 = TinyFeedForward(10, 20, 30, device=mesh_device)
    model2 = TinyFeedForward(10, 20, 30, device=mesh_device)

    state_dict = {
        "linear1.weight": torch.randn([20, 10]),
        "linear1.bias": torch.randn([20]),
        "linear2.weight": torch.randn([30, 20]),
        "linear2.bias": torch.randn([30]),
    }

    model1.load_torch_state_dict(state_dict)

    with TemporaryDirectory() as dirname:
        model1.save(dirname)
        model2.load(dirname)

    x = ttnn.from_torch(torch.randn([10]), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    y1 = ttnn.to_torch(model1.forward(x))
    y2 = ttnn.to_torch(model2.forward(x))

    assert torch.equal(y1, y2)
