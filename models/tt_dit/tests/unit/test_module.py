# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from tempfile import TemporaryDirectory

import pytest
import torch

import ttnn

from ...layers.module import Module, Parameter
from ...parallel.manager import CCLManager
from ...utils.tensor import to_torch
from ...utils.test import mesh_device_config_to_string


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


class ShardableModule(Module):
    def __init__(self, *, device: ttnn.MeshDevice) -> None:
        super().__init__()

        self.replicated = Parameter(total_shape=[512, 256], device=device)
        self.tensor_parallel = Parameter(total_shape=[512, 256], mesh_axes=[None, 0], device=device)
        self.small = Parameter(total_shape=[32, 32], device=device)

    def forward(self) -> None:
        pass


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        (shape, {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True})
        for shape in [(2, 4), (4, 8)]
    ],
    ids=mesh_device_config_to_string,
    indirect=True,
)
def test_shard_and_gather(mesh_device: ttnn.MeshDevice, expect_error) -> None:  # noqa: ANN001
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    model = ShardableModule(device=mesh_device)
    state = {name: torch.randn([512, 256], dtype=torch.bfloat16) for name in ("replicated", "tensor_parallel")}
    state["small"] = torch.randn([32, 32], dtype=torch.bfloat16)
    model.load_torch_state_dict(state)
    params = dict(model.named_parameters())

    def assert_layout_matches_data() -> None:
        for name, p in params.items():
            assert tuple(p._data.shape) == p.local_shape, name  # noqa: SLF001
            assert torch.equal(to_torch(p._data, mesh_axes=p.mesh_axes), state[name]), name  # noqa: SLF001

    model.shard(1)
    assert not model.is_loaded() and model.is_sharded()
    assert params["replicated"].mesh_axes == (1, None)
    assert params["tensor_parallel"].mesh_axes == (1, 0)
    assert params["tensor_parallel"].durable_mesh_axes == (None, 0)
    assert not params["small"].is_sharded()  # no dim splits into whole tiles
    assert_layout_matches_data()
    with expect_error(RuntimeError, "gather it first"):
        params["replicated"].data  # noqa: B018

    # Axis 0 is in `tensor_parallel`'s durable layout, axis 1 already transient: both refuse, and change nothing.
    with expect_error(ValueError, "durable layout"):
        model.shard(0)
    with expect_error(ValueError, "already has a transient split"):
        model.shard(1)
    assert params["replicated"].mesh_axes == (1, None)

    model.gather(ccl_manager)
    assert model.is_loaded() and not model.is_sharded()
    for name, p in params.items():
        assert p.mesh_axes == p.durable_mesh_axes, name
    for name in ("replicated", "small"):
        for device_tensor in ttnn.get_device_tensors(params[name].data):
            assert torch.equal(ttnn.to_torch(device_tensor), state[name]), name
    assert_layout_matches_data()

    # A parameter with no durable split can be split transiently over both axes and gathered back in one call.
    replicated = params["replicated"]
    replicated.shard(1)
    replicated.shard(0)
    assert replicated.mesh_axes == (1, 0)
    assert_layout_matches_data()
    replicated.gather(ccl_manager)
    assert replicated.mesh_axes == (None, None)
    assert_layout_matches_data()

    replicated.shard(1)
    replicated.deallocate()
    assert replicated.mesh_axes == (None, None) and not replicated.is_sharded()
