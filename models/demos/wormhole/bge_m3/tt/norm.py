# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from models.common.tensor_utils import TILE_SIZE

SHARD_HEIGHT = TILE_SIZE


@dataclass
class LayerNorm1DConfig:
    # Required affine parameters
    weight: LazyWeight
    bias: LazyWeight

    # Device
    mesh_device: ttnn.MeshDevice | None = None

    # Numerics
    eps: float = 1e-5

    # Optional input memcfg override
    input_memcfg: ttnn.MemoryConfig | None = None

    max_seq_len: int | None = None
    max_batch_size: int | None = None
    output_memcfg: ttnn.MemoryConfig | None = None

    # Compute kernel config
    compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None
    program_config: object | None = None
    sharded_memcfg: ttnn.MemoryConfig | None = None

    def is_resolved(self) -> bool:
        return self.mesh_device is not None and self.compute_kernel_config is not None


class LayerNorm1D(LightweightModule):
    def __init__(self, weight: LazyWeight, bias: LazyWeight, eps: float = 1e-5):
        super().__init__()
        self.config = _resolve_1d_config(LayerNorm1DConfig(weight=weight, bias=bias, eps=eps))
        self.eps = self.config.eps
        self._device_weights_loaded = False

    @classmethod
    def from_config(cls, config: LayerNorm1DConfig) -> "LayerNorm1D":
        instance = object.__new__(cls)
        super(LayerNorm1D, instance).__init__()
        instance.config = _resolve_1d_config(config)
        instance.eps = instance.config.eps
        instance._device_weights_loaded = False
        return instance

    def load_device_weights(self) -> None:
        if self._device_weights_loaded:
            return
        assert self.config.is_resolved(), "config must be resolved before loading device weights!"
        self.weight = self.config.weight.get_device_weight()
        self.bias = self.config.bias.get_device_weight()
        self._device_weights_loaded = True

    def forward(
        self,
        x: ttnn.Tensor | LazyWeight,
        residual_input_tensor: ttnn.Tensor | None = None,
        return_sharded: bool = False,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """
        If return_sharded=True and a sharded_memcfg is configured, returns a
        tuple (interleaved_out, sharded_out). The caller can pass `sharded_out`
        as `residual_input_tensor` to the NEXT LayerNorm to skip its residual
        I->S reshard (-1.1 us/call). The sharded_out tensor is OWNED by the
        caller, who must deallocate it after the next LN consumes it.
        """
        self.load_device_weights()
        x = _load_input_device_tensor(x, self.config)
        assert self.weight is not None and self.bias is not None, "weights must be loaded before forward"

        original_dtype = x.dtype
        out_mem = self.config.output_memcfg or ttnn.DRAM_MEMORY_CONFIG
        program_config = self.config.program_config
        memory_config = out_mem

        if self.config.sharded_memcfg is not None:
            # bfloat8_b + sharded LN: ttnn.layer_norm accumulates quantization
            # error with bf8b inputs across 24 encoder layers (PCC ~0.50).
            # Fix: fused typecast+reshard via interleaved_to_sharded(output_dtype=bf16).
            # Single op, bit-identical to separate typecast + reshard.
            # Non-sharded paths (e.g. B32) are unaffected.
            if x.dtype == ttnn.bfloat8_b:
                x = ttnn.interleaved_to_sharded(x, self.config.sharded_memcfg, output_dtype=ttnn.bfloat16)
            else:
                x = ttnn.to_memory_config(x, memory_config=self.config.sharded_memcfg)
            if residual_input_tensor is not None:
                if residual_input_tensor.dtype == ttnn.bfloat8_b:
                    residual_input_tensor = ttnn.interleaved_to_sharded(
                        residual_input_tensor, self.config.sharded_memcfg, output_dtype=ttnn.bfloat16
                    )
                else:
                    # to_memory_config is a no-op if already in sharded_memcfg
                    residual_input_tensor = ttnn.to_memory_config(
                        residual_input_tensor, memory_config=self.config.sharded_memcfg
                    )
            memory_config = self.config.sharded_memcfg

        sharded_output = ttnn.layer_norm(
            x,
            epsilon=self.config.eps,
            weight=self.weight,
            bias=self.bias,
            residual_input_tensor=residual_input_tensor,
            program_config=program_config,
            memory_config=memory_config,
            compute_kernel_config=self.config.compute_kernel_config,
        )

        if self.config.sharded_memcfg is not None and out_mem != memory_config:
            # Fused sharded→interleaved with bf16→bf8b typecast.
            # The next LN re-typecasts bf8b→bf16 via fused I->S anyway, so
            # writing the interleaved tensor as bf8b here halves NoC bytes
            # without changing what the LN kernel actually consumes.
            # Math-equivalent vs the bf16 interleaved path.
            if original_dtype == ttnn.bfloat8_b:
                interleaved_output = ttnn.sharded_to_interleaved(
                    sharded_output, memory_config=out_mem, output_dtype=ttnn.bfloat8_b
                )
            else:
                interleaved_output = ttnn.to_memory_config(sharded_output, memory_config=out_mem)
            if return_sharded:
                return interleaved_output, sharded_output
            ttnn.deallocate(sharded_output)
            return interleaved_output
        # No sharded handoff possible.
        if return_sharded:
            return sharded_output, None
        return sharded_output


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _load_input_device_tensor(x, config):
    if isinstance(x, LazyWeight):
        mem_cfg = config.input_memcfg or ttnn.DRAM_MEMORY_CONFIG
        assert config.mesh_device is not None, "mesh_device must be resolved"
        resolved_x = resolve_lazy_weight(
            x,
            device=config.mesh_device,
            memory_config=mem_cfg,
            mesh_mapper_config=None,
            layout=ttnn.TILE_LAYOUT,
        )
        return resolved_x.get_device_weight()
    assert isinstance(x, ttnn.Tensor), f"x must be ttnn.Tensor or LazyWeight, got {type(x)}"
    return x


def _resolve_1d_config(config: LayerNorm1DConfig) -> LayerNorm1DConfig:
    to_set: dict[str, object] = {}

    # Resolve device
    weight_device = config.weight.device
    bias_device = config.bias.device
    if weight_device is not None and bias_device is not None and weight_device != bias_device:
        raise ValueError("LayerNorm weight and bias must target the same device")

    mesh_device = config.mesh_device or weight_device or bias_device or ttnn.GetDefaultDevice()
    if mesh_device is None:
        raise ValueError("Unable to resolve mesh_device")
    if config.mesh_device is None:
        to_set["mesh_device"] = mesh_device

    # Validate affine sizes
    dim = config.weight.source.numel()
    if config.bias.source.numel() != dim:
        raise ValueError(f"LayerNorm weight/bias numel mismatch: {dim} vs {config.bias.source.numel()}")
    if dim % SHARD_HEIGHT != 0:
        raise ValueError(f"LayerNorm hidden dim must be divisible by {SHARD_HEIGHT}, got {dim}")

    # Defaults: DRAM, HiFi4
    if config.compute_kernel_config is None:
        to_set["compute_kernel_config"] = _default_compute_kernel(mesh_device)
    if config.output_memcfg is None:
        to_set["output_memcfg"] = ttnn.DRAM_MEMORY_CONFIG

    weight_mem = ttnn.DRAM_MEMORY_CONFIG

    # Resolve weights with correct shape
    expected_shape = (1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT)

    weight_source = config.weight.source
    if tuple(weight_source.shape) != expected_shape:
        weight_source = weight_source.reshape(*expected_shape)
    bias_source = config.bias.source
    if tuple(bias_source.shape) != expected_shape:
        bias_source = bias_source.reshape(*expected_shape)

    to_set["weight"] = resolve_lazy_weight(
        replace(config.weight, source=weight_source),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=weight_mem,
        mesh_mapper_config=None,
    )
    to_set["bias"] = resolve_lazy_weight(
        replace(config.bias, source=bias_source),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
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
