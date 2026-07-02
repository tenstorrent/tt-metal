# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight


@dataclass
class BgeM3MLPConfig:
    # Required weights
    wi_weight: LazyWeight
    wo_weight: LazyWeight

    # Model dimensions
    hidden_size: int
    intermediate_size: int

    # Optional biases
    wi_bias: LazyWeight | None = None
    wo_bias: LazyWeight | None = None
    mesh_device: ttnn.MeshDevice | None = None

    # Activation
    activation: str = "gelu"

    # Runtime config fields (resolved by _resolve_mlp_config or Optimizations)
    wi_dtype: ttnn.DataType | None = None
    wo_dtype: ttnn.DataType | None = None
    activation_dtype: ttnn.DataType | None = None
    wi_memcfg: ttnn.MemoryConfig | None = None
    wo_memcfg: ttnn.MemoryConfig | None = None
    activation_memcfg: ttnn.MemoryConfig | None = None
    wi_prg_config: object | None = None
    wo_prg_config: object | None = None
    wi_compute_kernel_cfg: object | None = None
    wo_compute_kernel_cfg: object | None = None
    wi_minimal_config: object | None = None
    wo_minimal_config: object | None = None
    core_grid: ttnn.CoreGrid | None = None
    max_seq_len: int | None = None
    max_batch_size: int | None = None


class BgeM3MLP(LightweightModule):
    """
    BGE-M3 encoder-mode MLP block.

    Forward path: wi -> gelu -> wo.
    """

    def __init__(
        self,
        wi_weight: LazyWeight,
        wo_weight: LazyWeight,
        hidden_size: int,
        intermediate_size: int,
        wi_bias: LazyWeight | None = None,
        wo_bias: LazyWeight | None = None,
        activation: str = "gelu",
    ):
        super().__init__()
        self.config = _resolve_mlp_config(
            BgeM3MLPConfig(
                wi_weight=wi_weight,
                wo_weight=wo_weight,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                wi_bias=wi_bias,
                wo_bias=wo_bias,
                activation=activation,
            )
        )
        self._device_weights_loaded = False

    @classmethod
    def from_config(cls, config: BgeM3MLPConfig) -> "BgeM3MLP":
        instance = object.__new__(cls)
        super(BgeM3MLP, instance).__init__()
        instance.config = _resolve_mlp_config(config)
        instance._device_weights_loaded = False
        return instance

    def load_device_weights(self) -> None:
        if self._device_weights_loaded:
            return
        self.wi_weight = self.config.wi_weight.get_device_weight()
        self.wo_weight = self.config.wo_weight.get_device_weight()
        self.wi_bias = self.config.wi_bias.get_device_weight() if self.config.wi_bias is not None else None
        self.wo_bias = self.config.wo_bias.get_device_weight() if self.config.wo_bias is not None else None
        self._device_weights_loaded = True

    def forward(self, hidden_states: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        self.load_device_weights()
        hidden_states = _load_input_device_tensor(hidden_states, self.config)

        wi_core_grid = None if self.config.wi_prg_config is not None else self.config.core_grid
        wi_activation = None if self.config.wi_prg_config is not None else "gelu"

        if self.config.wi_minimal_config is not None and self.config.wi_prg_config is None:
            activated = ttnn.experimental.minimal_matmul(
                input_tensor=hidden_states,
                weight_tensor=self.wi_weight,
                bias_tensor=self.wi_bias,
                fused_activation=(ttnn.UnaryOpType.GELU, True),
                config=self.config.wi_minimal_config,
                memory_config=self.config.wi_memcfg,
                dtype=self.config.wi_dtype,
                compute_kernel_config=self.config.wi_compute_kernel_cfg,
            )
        else:
            activated = ttnn.linear(
                hidden_states,
                self.wi_weight,
                memory_config=self.config.wi_memcfg,
                dtype=self.config.wi_dtype,
                bias=self.wi_bias,
                program_config=self.config.wi_prg_config,
                compute_kernel_config=self.config.wi_compute_kernel_cfg,
                activation=wi_activation,
                core_grid=wi_core_grid,
            )

        wo_core_grid = None if self.config.wo_prg_config is not None else self.config.core_grid
        if self.config.wo_minimal_config is not None and self.config.wo_prg_config is None:
            output = ttnn.experimental.minimal_matmul(
                input_tensor=activated,
                weight_tensor=self.wo_weight,
                bias_tensor=self.wo_bias,
                fused_activation=None,
                config=self.config.wo_minimal_config,
                memory_config=self.config.wo_memcfg,
                dtype=self.config.wo_dtype,
                compute_kernel_config=self.config.wo_compute_kernel_cfg,
            )
        else:
            output = ttnn.linear(
                activated,
                self.wo_weight,
                memory_config=self.config.wo_memcfg,
                dtype=self.config.wo_dtype,
                bias=self.wo_bias,
                program_config=self.config.wo_prg_config,
                compute_kernel_config=self.config.wo_compute_kernel_cfg,
                core_grid=wo_core_grid,
            )
        ttnn.deallocate(activated)
        return output


def _resolve_mlp_config(config: BgeM3MLPConfig) -> BgeM3MLPConfig:
    if config.activation != "gelu":
        raise ValueError(f"Unsupported activation '{config.activation}'. Only 'gelu' is supported.")

    to_set: dict[str, object] = {}

    if config.wi_dtype is None:
        to_set["wi_dtype"] = ttnn.bfloat16
    if config.wo_dtype is None:
        to_set["wo_dtype"] = ttnn.bfloat16
    if config.activation_dtype is None:
        to_set["activation_dtype"] = ttnn.bfloat16

    # Resolve device
    param_devices = [
        p.device
        for p in (config.wi_weight, config.wi_bias, config.wo_weight, config.wo_bias)
        if p is not None and p.device is not None
    ]
    if param_devices and any(d != param_devices[0] for d in param_devices):
        raise ValueError("All MLP parameters must target the same device")

    mesh_device = config.mesh_device or (param_devices[0] if param_devices else ttnn.GetDefaultDevice())
    if mesh_device is None:
        raise ValueError("Unable to resolve target device for BgeM3MLP")
    if config.mesh_device is None:
        to_set["mesh_device"] = mesh_device

    # Defaults: DRAM for everything, basic compute kernel
    if config.activation_memcfg is None:
        to_set["activation_memcfg"] = ttnn.DRAM_MEMORY_CONFIG
    if config.wi_memcfg is None:
        to_set["wi_memcfg"] = ttnn.DRAM_MEMORY_CONFIG
    if config.wo_memcfg is None:
        to_set["wo_memcfg"] = ttnn.DRAM_MEMORY_CONFIG
    if config.wi_compute_kernel_cfg is None:
        to_set["wi_compute_kernel_cfg"] = _default_compute_kernel(mesh_device)
    if config.wo_compute_kernel_cfg is None:
        to_set["wo_compute_kernel_cfg"] = _default_compute_kernel(mesh_device)
    if config.core_grid is None:
        to_set["core_grid"] = _default_core_grid(mesh_device)

    # Resolve weights
    wi_dtype = to_set.get("wi_dtype", config.wi_dtype)
    wo_dtype = to_set.get("wo_dtype", config.wo_dtype)
    weight_mem = ttnn.DRAM_MEMORY_CONFIG

    to_set["wi_weight"] = resolve_lazy_weight(
        config.wi_weight,
        device=mesh_device,
        dtype=wi_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=weight_mem,
        mesh_mapper_config=None,
    )
    to_set["wo_weight"] = resolve_lazy_weight(
        config.wo_weight,
        device=mesh_device,
        dtype=wo_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=weight_mem,
        mesh_mapper_config=None,
    )
    if config.wi_bias is not None:
        to_set["wi_bias"] = resolve_lazy_weight(
            config.wi_bias,
            device=mesh_device,
            dtype=wi_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_mem,
            mesh_mapper_config=None,
        )
    if config.wo_bias is not None:
        to_set["wo_bias"] = resolve_lazy_weight(
            config.wo_bias,
            device=mesh_device,
            dtype=wo_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_mem,
            mesh_mapper_config=None,
        )

    return replace(config, **to_set)


def _default_compute_kernel(mesh_device):
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _default_core_grid(mesh_device):
    try:
        g = mesh_device.compute_with_storage_grid_size()
        return ttnn.CoreGrid(y=int(g.y), x=int(g.x))
    except Exception:
        return ttnn.CoreGrid(y=8, x=8)


def _load_input_device_tensor(x: ttnn.Tensor | LazyWeight, config: BgeM3MLPConfig) -> ttnn.Tensor:
    mem_cfg = config.activation_memcfg
    assert mem_cfg is not None, "activation_memcfg must be resolved before loading input tensor"

    if isinstance(x, LazyWeight):
        resolved_x = resolve_lazy_weight(
            x,
            device=config.wi_weight.device,
            memory_config=mem_cfg,
            mesh_mapper_config=None,
            layout=ttnn.TILE_LAYOUT,
        )
        return resolved_x.get_device_weight()

    assert isinstance(x, ttnn.Tensor), "x must be a ttnn tensor at this point!"
    return x
