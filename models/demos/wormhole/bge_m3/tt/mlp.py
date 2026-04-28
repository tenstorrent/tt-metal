# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from models.demos.wormhole.bge_m3.tt.device_kernels import (
    bge_m3_linear_activation_memory_config,
    bge_m3_matmul_core_grid,
    bge_m3_mlp_wi_compute_kernel_config,
    bge_m3_mlp_wi_output_memory_config,
    bge_m3_mlp_wo_compute_kernel_config,
    bge_m3_weight_dram_memory_config,
)


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
        batch_size, _, seq_len, _ = hidden_states.shape
        runtime_batch = int(batch_size)
        runtime_seq = int(seq_len)
        core_grid = bge_m3_matmul_core_grid(self.config.mesh_device, runtime_seq, runtime_batch)
        wi_prg_config = self.config.wi_prg_config or _runtime_mlp_wi_program_config(
            self.config.mesh_device,
            runtime_seq,
            runtime_batch,
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
        )
        wi_core_grid = None if wi_prg_config is not None else core_grid
        wi_activation = None if wi_prg_config is not None else "gelu"
        wi_compute_kernel_cfg = bge_m3_mlp_wi_compute_kernel_config(
            self.config.mesh_device,
            max_seq_len=runtime_seq,
            max_batch_size=runtime_batch,
        )
        wo_compute_kernel_cfg = bge_m3_mlp_wo_compute_kernel_config(
            self.config.mesh_device,
            max_seq_len=runtime_seq,
            max_batch_size=runtime_batch,
        )
        # GELU stays fused into Wi, either through the default activation arg or an explicit S512 program config.
        activated = ttnn.linear(
            hidden_states,
            self.wi_weight,
            memory_config=self.config.wi_memcfg,
            dtype=self.config.wi_dtype,
            bias=self.wi_bias,
            program_config=wi_prg_config,
            compute_kernel_config=wi_compute_kernel_cfg,
            activation=wi_activation,
            core_grid=wi_core_grid,
        )

        output = ttnn.linear(
            activated,
            self.wo_weight,
            memory_config=self.config.wo_memcfg,
            dtype=self.config.wo_dtype,
            bias=self.wo_bias,
            program_config=self.config.wo_prg_config,
            compute_kernel_config=wo_compute_kernel_cfg,
            core_grid=core_grid,
        )
        ttnn.deallocate(activated)

        return output


def _resolve_mlp_config(config: BgeM3MLPConfig) -> BgeM3MLPConfig:
    """
    Resolve MLP config defaults and materialize LazyWeight metadata.
    """
    if config.activation != "gelu":
        raise ValueError(f"Unsupported activation '{config.activation}'. Only 'gelu' is currently supported.")

    to_set: dict[str, object] = {}

    if config.wi_dtype is None:
        to_set["wi_dtype"] = ttnn.bfloat16
    if config.wo_dtype is None:
        to_set["wo_dtype"] = ttnn.bfloat16
    if config.activation_dtype is None:
        to_set["activation_dtype"] = ttnn.bfloat16
    max_seq = config.max_seq_len
    max_batch = config.max_batch_size if config.max_batch_size is not None else 1

    # All parameters must target a single device.
    param_devices = [
        param.device
        for param in (config.wi_weight, config.wi_bias, config.wo_weight, config.wo_bias)
        if param is not None and param.device is not None
    ]
    if param_devices and any(device != param_devices[0] for device in param_devices):
        raise ValueError("All MLP parameters must target the same device")
    if config.mesh_device is not None and param_devices and param_devices[0] != config.mesh_device:
        raise ValueError("All MLP parameters must target the configured mesh_device")

    mesh_device = (
        config.mesh_device
        if config.mesh_device is not None
        else (param_devices[0] if param_devices else ttnn.GetDefaultDevice())
    )
    if mesh_device is None:
        raise ValueError("Unable to resolve target device for BgeM3MLP")

    if config.mesh_device is None:
        to_set["mesh_device"] = mesh_device

    act_mem = bge_m3_linear_activation_memory_config(max_seq, max_batch)
    wi_out_mem = bge_m3_mlp_wi_output_memory_config(max_seq, max_batch, mesh_device)
    if config.activation_memcfg is None:
        to_set["activation_memcfg"] = act_mem
    if config.wi_memcfg is None:
        to_set["wi_memcfg"] = wi_out_mem
    if config.wo_memcfg is None:
        to_set["wo_memcfg"] = act_mem

    if config.wi_compute_kernel_cfg is None:
        to_set["wi_compute_kernel_cfg"] = bge_m3_mlp_wi_compute_kernel_config(
            mesh_device, max_seq_len=max_seq, max_batch_size=max_batch
        )
    if config.wo_compute_kernel_cfg is None:
        to_set["wo_compute_kernel_cfg"] = bge_m3_mlp_wo_compute_kernel_config(
            mesh_device, max_seq_len=max_seq, max_batch_size=max_batch
        )
    wi_dtype = to_set.get("wi_dtype", config.wi_dtype)
    wo_dtype = to_set.get("wo_dtype", config.wo_dtype)
    weight_dram = bge_m3_weight_dram_memory_config()

    to_set["wi_weight"] = resolve_lazy_weight(
        config.wi_weight,
        device=mesh_device,
        dtype=wi_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=weight_dram,
        mesh_mapper_config=None,
    )
    to_set["wo_weight"] = resolve_lazy_weight(
        config.wo_weight,
        device=mesh_device,
        dtype=wo_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=weight_dram,
        mesh_mapper_config=None,
    )
    if config.wi_bias is not None:
        to_set["wi_bias"] = resolve_lazy_weight(
            config.wi_bias,
            device=mesh_device,
            dtype=wi_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_dram,
            mesh_mapper_config=None,
        )
    if config.wo_bias is not None:
        to_set["wo_bias"] = resolve_lazy_weight(
            config.wo_bias,
            device=mesh_device,
            dtype=wo_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_dram,
            mesh_mapper_config=None,
        )

    return replace(config, **to_set)


def _runtime_mlp_wi_program_config(
    mesh_device,
    max_seq_len: int | None,
    max_batch_size: int | None,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> object | None:
    max_batch = 1 if max_batch_size is None else max(1, int(max_batch_size))
    if max_seq_len != 512:
        return None

    if max_batch == 32:
        return _b32s512_mlp_wi_program_config(
            mesh_device,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
    if max_batch != 1:
        return None

    return _b1s512_mlp_wi_program_config(
        mesh_device,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )


def _b1s512_mlp_wi_program_config(
    mesh_device,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> object:
    max_seq_len = 512
    max_batch = 1

    core_grid = bge_m3_matmul_core_grid(mesh_device, max_seq_len, max_batch)
    hidden_tiles = hidden_size // 32
    intermediate_tiles = intermediate_size // 32
    m_tiles = max_seq_len // 32
    per_core_m = (m_tiles + core_grid.y - 1) // core_grid.y
    per_core_n = (intermediate_tiles + core_grid.x - 1) // core_grid.x

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(core_grid.x, core_grid.y),
        in0_block_w=min(4, hidden_tiles),
        out_subblock_h=1,
        out_subblock_w=2,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=(ttnn.UnaryOpType.GELU, True),
    )


def _b32s512_mlp_wi_program_config(
    mesh_device,
    *,
    hidden_size: int,
    intermediate_size: int,
) -> object | None:
    return _b32s512_sequence_mlp_program_config(
        mesh_device,
        input_size=hidden_size,
        output_size=intermediate_size,
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=3,
        fused_activation=(ttnn.UnaryOpType.GELU, True),
    )


def _b32s512_sequence_mlp_program_config(
    mesh_device,
    *,
    input_size: int,
    output_size: int,
    in0_block_w: int,
    out_subblock_h: int,
    out_subblock_w: int,
    fused_activation,
) -> object | None:
    max_seq_len = 512
    tile_size = 32
    grid_x = 11
    grid_y = 10
    device_grid = mesh_device.compute_with_storage_grid_size()
    if device_grid.x < grid_x or device_grid.y < grid_y:
        return None

    input_tiles = input_size // tile_size
    output_tiles = output_size // tile_size
    seq_tiles = max_seq_len // tile_size
    per_core_m = (seq_tiles + grid_y - 1) // grid_y
    per_core_n = (output_tiles + grid_x - 1) // grid_x

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        in0_block_w=min(in0_block_w, input_tiles),
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=False,
    )


def _load_input_device_tensor(x: ttnn.Tensor | LazyWeight, config: BgeM3MLPConfig) -> ttnn.Tensor:
    """
    Resolve input to device tensor if x is LazyWeight; otherwise sanity-check x.
    """
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
    if x.memory_config() != mem_cfg:
        raise ValueError("Input tensor memory config does not match the config!")

    return x
