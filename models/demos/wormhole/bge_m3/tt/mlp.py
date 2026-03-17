# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

    # Activation
    activation: str = "gelu"

    # Optional runtime config fields (resolved later)
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
        """
        Materialize LazyWeights on device.
        """
        if self._device_weights_loaded:
            return

        self.wi_weight = self.config.wi_weight.get_device_weight()
        self.wo_weight = self.config.wo_weight.get_device_weight()
        self.wi_bias = self.config.wi_bias.get_device_weight() if self.config.wi_bias is not None else None
        self.wo_bias = self.config.wo_bias.get_device_weight() if self.config.wo_bias is not None else None
        self._device_weights_loaded = True

    def forward(self, hidden_states: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """
        Encoder-style forward path.
        """
        self.load_device_weights()
        hidden_states = _load_input_device_tensor(hidden_states, self.config)

        intermediate = ttnn.linear(
            hidden_states,
            self.wi_weight,
            memory_config=self.config.wi_memcfg,
            dtype=self.config.wi_dtype,
            bias=self.wi_bias,
            program_config=self.config.wi_prg_config,
            compute_kernel_config=self.config.wi_compute_kernel_cfg,
        )
        activated = ttnn.gelu(
            intermediate,
            memory_config=self.config.activation_memcfg,
            fast_and_approximate_mode=False,
        )
        ttnn.deallocate(intermediate)

        output = ttnn.linear(
            activated,
            self.wo_weight,
            memory_config=self.config.wo_memcfg,
            dtype=self.config.wo_dtype,
            bias=self.wo_bias,
            program_config=self.config.wo_prg_config,
            compute_kernel_config=self.config.wo_compute_kernel_cfg,
        )
        ttnn.deallocate(activated)

        return output


def _default_mlp_compute_kernel_config() -> ttnn.WormholeComputeKernelConfig:
    """
    Default matmul kernel for BF16-oriented inference bring-up.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _resolve_mlp_config(config: BgeM3MLPConfig) -> BgeM3MLPConfig:
    """
    Resolve MLP config defaults and materialize LazyWeight metadata.
    """
    if config.activation != "gelu":
        raise ValueError(f"Unsupported activation '{config.activation}'. Only 'gelu' is currently supported.")

    to_set: dict[str, object] = {}

    # Default numerics and memory for bring-up.
    if config.wi_dtype is None:
        to_set["wi_dtype"] = ttnn.bfloat16
    if config.wo_dtype is None:
        to_set["wo_dtype"] = ttnn.bfloat16
    if config.activation_dtype is None:
        to_set["activation_dtype"] = ttnn.bfloat16
    if config.wi_memcfg is None:
        to_set["wi_memcfg"] = ttnn.DRAM_MEMORY_CONFIG
    if config.wo_memcfg is None:
        to_set["wo_memcfg"] = ttnn.DRAM_MEMORY_CONFIG
    if config.activation_memcfg is None:
        to_set["activation_memcfg"] = ttnn.DRAM_MEMORY_CONFIG

    if config.wi_compute_kernel_cfg is None:
        to_set["wi_compute_kernel_cfg"] = _default_mlp_compute_kernel_config()
    if config.wo_compute_kernel_cfg is None:
        to_set["wo_compute_kernel_cfg"] = _default_mlp_compute_kernel_config()

    # All parameters must target a single device.
    param_devices = [
        param.device
        for param in (config.wi_weight, config.wi_bias, config.wo_weight, config.wo_bias)
        if param is not None and param.device is not None
    ]
    if param_devices and any(device != param_devices[0] for device in param_devices):
        raise ValueError("All MLP parameters must target the same device")
    mesh_device = param_devices[0] if param_devices else ttnn.GetDefaultDevice()
    if mesh_device is None:
        raise ValueError("Unable to resolve target device for BgeM3MLP")

    wi_dtype = to_set.get("wi_dtype", config.wi_dtype)
    wo_dtype = to_set.get("wo_dtype", config.wo_dtype)
    wi_memcfg = to_set.get("wi_memcfg", config.wi_memcfg)
    wo_memcfg = to_set.get("wo_memcfg", config.wo_memcfg)

    to_set["wi_weight"] = resolve_lazy_weight(
        config.wi_weight,
        device=mesh_device,
        dtype=wi_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=wi_memcfg,
        mesh_mapper_config=None,
    )
    to_set["wo_weight"] = resolve_lazy_weight(
        config.wo_weight,
        device=mesh_device,
        dtype=wo_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=wo_memcfg,
        mesh_mapper_config=None,
    )
    if config.wi_bias is not None:
        to_set["wi_bias"] = resolve_lazy_weight(
            config.wi_bias,
            device=mesh_device,
            dtype=wi_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=wi_memcfg,
            mesh_mapper_config=None,
        )
    if config.wo_bias is not None:
        to_set["wo_bias"] = resolve_lazy_weight(
            config.wo_bias,
            device=mesh_device,
            dtype=wo_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=wo_memcfg,
            mesh_mapper_config=None,
        )

    return replace(config, **to_set)


def _load_input_device_tensor(x: ttnn.Tensor | LazyWeight, config: BgeM3MLPConfig) -> ttnn.Tensor:
    """
    Resolve input to device tensor if x is LazyWeight; otherwise sanity-check x.
    """
    mem_cfg = config.wi_memcfg
    assert mem_cfg is not None, "wi_memcfg must be resolved before loading input tensor"

    if isinstance(x, LazyWeight):
        resolved_x = resolve_lazy_weight(
            x,
            device=config.wi_weight.device,
            memory_config=mem_cfg,
            mesh_mapper_config=None,  # replicated
            layout=ttnn.TILE_LAYOUT,
        )
        return resolved_x.get_device_weight()

    assert isinstance(x, ttnn.Tensor), "x must be a ttnn tensor at this point!"
    if x.memory_config() != mem_cfg:
        raise ValueError("Input tensor memory config does not match the config!")

    return x
