from __future__ import annotations

import pytest
import torch
import ttnn

from ...layers.module import Module, Parameter


class TinyLinear(Module):
    def __init__(self, in_dim: int, out_dim: int, *, device: ttnn.MeshDevice, init: bool = False) -> None:
        super().__init__()

        self.weight = Parameter(shape=[in_dim, out_dim], device=device, init=init)
        self.bias = Parameter(shape=[in_dim, out_dim], device=device, init=init)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return x @ self.weight.data + self.bias.data

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"].transpose_(0, 1)


class TinyFeedForward(Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, *, device: ttnn.MeshDevice, init: bool = False
    ) -> None:
        super().__init__()

        self.linear1 = TinyLinear(in_dim, hidden_dim, device=device, init=init)
        self.linear2 = TinyLinear(hidden_dim, out_dim, device=device, init=init)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.linear1.forward(x)
        x = ttnn.silu(x)
        return self.linear2.forward(x)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_module(mesh_device: ttnn.MeshDevice) -> None:
    model = TinyFeedForward(10, 20, 30, device=mesh_device)

    state_dict = {
        "something.unexpected": torch.randn([1]),
    }

    missing, unexpected = model.load_torch_state_dict(state_dict, strict=False)

    assert missing == ["linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"]
    assert unexpected == ["something.unexpected"]
