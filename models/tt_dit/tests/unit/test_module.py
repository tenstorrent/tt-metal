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

    # Axis 0 is in `tensor_parallel`'s durable layout: the call refuses and changes nothing.
    with expect_error(ValueError, "durable layout"):
        model.shard(0)
    assert model.is_loaded() and not model.is_transiently_sharded()

    model.shard(1)
    assert not model.is_loaded() and model.is_transiently_sharded()
    assert params["replicated"].mesh_axes == (1, None)
    assert params["tensor_parallel"].mesh_axes == (1, 0)
    assert params["tensor_parallel"].durable_mesh_axes == (None, 0)
    assert not params["small"].is_transiently_sharded()  # no dim splits into whole tiles
    assert_layout_matches_data()
    with expect_error(RuntimeError, "gather it first"):
        params["replicated"].data  # noqa: B018
    with expect_error(RuntimeError, "ShardableModule has weights with a transient split"):
        model()

    # While sharded, sharding again refuses over any axis and changes nothing.
    for axis in (0, 1):
        with expect_error(ValueError, "already has a transient split"):
            model.shard(axis)
    assert params["replicated"].mesh_axes == (1, None)

    model.gather(ccl_manager)
    assert model.is_loaded() and not model.is_transiently_sharded()
    model()
    for name, p in params.items():
        assert p.mesh_axes == p.durable_mesh_axes, name
    for name in ("replicated", "small"):
        for device_tensor in ttnn.get_device_tensors(params[name].data):
            assert torch.equal(ttnn.to_torch(device_tensor), state[name]), name
    assert_layout_matches_data()

    # A parameter is sharded over one mesh axis at a time, even one with no durable split.
    replicated = params["replicated"]
    defer_free: list[ttnn.Tensor] = []
    replicated.shard(1, defer_free=defer_free)
    with expect_error(ValueError, "already has a transient split"):
        replicated.shard(0, defer_free=defer_free)
    assert replicated.mesh_axes == (1, None) and len(defer_free) == 1
    assert_layout_matches_data()
    with expect_error(RuntimeError, "ShardableModule has weights with a transient split"):
        model()  # Parameter.shard alone, without Module.shard, is enough for the call to refuse
    replicated.gather(ccl_manager, defer_free=defer_free)
    assert replicated.mesh_axes == (None, None) and len(defer_free) == 2
    for t in defer_free:
        ttnn.deallocate(t)
    assert_layout_matches_data()

    replicated.shard(1, defer_free=[])
    replicated.deallocate()
    assert replicated.mesh_axes == (None, None) and not replicated.is_transiently_sharded()


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_shard_without_a_split_keeps_the_module_loaded(mesh_device: ttnn.MeshDevice) -> None:
    model = ShardableModule(device=mesh_device)
    model.load_torch_state_dict(
        {
            "replicated": torch.randn([512, 256], dtype=torch.bfloat16),
            "tensor_parallel": torch.randn([512, 256], dtype=torch.bfloat16),
            "small": torch.randn([32, 32], dtype=torch.bfloat16),
        }
    )
    evictions = []
    model.register_eviction_hook(lambda: evictions.append(True))

    model.shard(1)  # one device on the axis: nothing splits

    assert model.is_loaded() and not model.is_transiently_sharded() and not evictions
    model()


class FreesExtraState(ShardableModule):
    def deallocate_weights(self) -> None:
        super().deallocate_weights()


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_shard_refuses_a_deallocate_weights_override(
    mesh_device: ttnn.MeshDevice, expect_error
) -> None:  # noqa: ANN001
    model = FreesExtraState(device=mesh_device)
    with expect_error(TypeError, "FreesExtraState overrides deallocate_weights"):
        model.shard(1)
