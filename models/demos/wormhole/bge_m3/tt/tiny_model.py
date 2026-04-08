# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight


@dataclass
class TinyLinearConfig:
    weight: LazyWeight
    bias: LazyWeight | None = None
    mesh_device: ttnn.MeshDevice | None = None
    dtype: ttnn.DataType | None = None
    memory_config: ttnn.MemoryConfig | None = None
    compute_kernel_config: object | None = None
    activation: object | None = None


class TinyLinear(LightweightModule):
    def __init__(
        self,
        weight: LazyWeight,
        bias: LazyWeight | None = None,
    ):
        super().__init__()
        self.config = _resolve_tiny_linear_config(TinyLinearConfig(weight=weight, bias=bias))
        self._device_weights_loaded = False

    @classmethod
    def from_config(cls, config: TinyLinearConfig) -> "TinyLinear":
        instance = object.__new__(cls)
        super(TinyLinear, instance).__init__()
        instance.config = _resolve_tiny_linear_config(config)
        instance._device_weights_loaded = False
        return instance

    def load_device_weights(self) -> None:
        if self._device_weights_loaded:
            return

        self.weight = self.config.weight.get_device_weight()
        self.bias = self.config.bias.get_device_weight() if self.config.bias is not None else None
        self._device_weights_loaded = True

    def forward(self, hidden_states: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        self.load_device_weights()
        hidden_states = _load_input_device_tensor(hidden_states, self.config)
        return ttnn.linear(
            hidden_states,
            self.weight,
            bias=self.bias,
            memory_config=self.config.memory_config,
            dtype=self.config.dtype,
            compute_kernel_config=self.config.compute_kernel_config,
            activation=self.config.activation,
        )


class ColBERTLinear(TinyLinear):
    pass


class SparseLinear(TinyLinear):
    @classmethod
    def from_config(cls, config: TinyLinearConfig) -> "SparseLinear":
        instance = object.__new__(cls)
        super(TinyLinear, instance).__init__()
        instance.config = _resolve_tiny_linear_config(
            replace(config, activation="relu" if config.activation is None else config.activation)
        )
        instance._device_weights_loaded = False
        return instance


def _default_tiny_linear_compute_kernel_config() -> ttnn.WormholeComputeKernelConfig:
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _resolve_tiny_linear_config(config: TinyLinearConfig) -> TinyLinearConfig:
    to_set: dict[str, object] = {}

    if config.dtype is None:
        to_set["dtype"] = ttnn.bfloat16
    if config.memory_config is None:
        to_set["memory_config"] = ttnn.DRAM_MEMORY_CONFIG
    if config.compute_kernel_config is None:
        to_set["compute_kernel_config"] = _default_tiny_linear_compute_kernel_config()

    param_devices = [
        param.device for param in (config.weight, config.bias) if param is not None and param.device is not None
    ]
    if param_devices and any(device != param_devices[0] for device in param_devices):
        raise ValueError("All TinyLinear parameters must target the same device")
    if config.mesh_device is not None and param_devices and param_devices[0] != config.mesh_device:
        raise ValueError("All TinyLinear parameters must target the configured mesh_device")

    mesh_device = (
        config.mesh_device
        if config.mesh_device is not None
        else (param_devices[0] if param_devices else ttnn.GetDefaultDevice())
    )
    if mesh_device is None:
        raise ValueError("Unable to resolve target device for TinyLinear")

    dtype = to_set.get("dtype", config.dtype)
    memory_config = to_set.get("memory_config", config.memory_config)

    to_set["weight"] = resolve_lazy_weight(
        config.weight,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        mesh_mapper_config=None,
    )
    if config.bias is not None:
        to_set["bias"] = resolve_lazy_weight(
            config.bias,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            mesh_mapper_config=None,
        )

    return replace(config, **to_set)


def _load_input_device_tensor(x: ttnn.Tensor | LazyWeight, config: TinyLinearConfig) -> ttnn.Tensor:
    mem_cfg = config.memory_config
    assert mem_cfg is not None, "memory_config must be resolved before loading input tensor"

    if isinstance(x, LazyWeight):
        resolved_x = resolve_lazy_weight(
            x,
            device=config.weight.device,
            memory_config=mem_cfg,
            mesh_mapper_config=None,
            layout=ttnn.TILE_LAYOUT,
        )
        return resolved_x.get_device_weight()

    assert isinstance(x, ttnn.Tensor), "x must be a ttnn tensor at this point!"
    if x.memory_config() != mem_cfg:
        raise ValueError("Input tensor memory config does not match the config!")

    return x
