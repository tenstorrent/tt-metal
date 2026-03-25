# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

    # Compute kernel config
    compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None

    def is_resolved(self) -> bool:
        return self.mesh_device is not None and self.compute_kernel_config is not None


class LayerNorm1D(LightweightModule):
    """
    BGE-M3 LayerNorm public API scaffold.

    Step 3 class skeleton:
      - simple API via __init__(weight, bias, eps)
      - power API via from_config(config)
      - config is resolved at construction time
    """

    def __init__(self, weight: LazyWeight, bias: LazyWeight, eps: float = 1e-5):
        super().__init__()
        self.config = _resolve_1d_config(LayerNorm1DConfig(weight=weight, bias=bias, eps=eps))
        self.eps = self.config.eps
        self._device_weights_loaded = False

    @classmethod
    def from_config(cls, config: LayerNorm1DConfig) -> "LayerNorm1D":
        """
        Power API entrypoint.

        Step 2 resolves config defaults and program/memory configs.
        """
        instance = object.__new__(cls)
        super(LayerNorm1D, instance).__init__()
        instance.config = _resolve_1d_config(config)
        instance.eps = instance.config.eps
        instance._device_weights_loaded = False
        return instance

    def load_device_weights(self) -> None:
        """Load affine weights once for all forward paths."""
        if self._device_weights_loaded:
            return

        cfg = self.config
        assert cfg.is_resolved(), "config must be resolved before loading device weights!"

        # Single-device path: weight and bias are materialized locally.
        self.weight = cfg.weight.get_device_weight()
        self.bias = cfg.bias.get_device_weight()
        self._device_weights_loaded = True

    def forward(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """Single-forward interleaved LayerNorm entrypoint."""
        self.load_device_weights()
        cfg = self.config
        x = _load_input_device_tensor(x, cfg)
        assert self.weight is not None and self.bias is not None, "weights must be loaded before forward"

        return ttnn.layer_norm(
            x,
            epsilon=cfg.eps,
            weight=self.weight,
            bias=self.bias,
            program_config=None,
            memory_config=None,
            compute_kernel_config=cfg.compute_kernel_config,
        )


################################################################################
# Helpers
################################################################################


def _load_input_device_tensor(x: ttnn.Tensor | LazyWeight, config: LayerNorm1DConfig) -> ttnn.Tensor:
    """
    Resolve input to a device tensor when provided as LazyWeight.

    This LayerNorm module supports single-device execution and replicated mesh execution.
    """
    if isinstance(x, LazyWeight):
        mem_cfg = config.input_memcfg or ttnn.DRAM_MEMORY_CONFIG
        mesh_device = config.mesh_device
        assert mesh_device is not None, "mesh_device must be resolved before loading input tensors"

        resolved_x = resolve_lazy_weight(
            x,
            device=mesh_device,
            memory_config=mem_cfg,
            mesh_mapper_config=None,
            layout=ttnn.TILE_LAYOUT,
        )
        return resolved_x.get_device_weight()

    assert isinstance(x, ttnn.Tensor), f"x must be ttnn.Tensor or LazyWeight, got {type(x)}"
    return x


def _resolve_1d_config(config: LayerNorm1DConfig) -> LayerNorm1DConfig:
    """
    Resolve LayerNorm1D config defaults and materializable LazyWeights.

    Mirrors the staged resolver flow used by RMSNorm1D while adapting for
    LayerNorm affine parameters (weight + bias).
    """
    to_set: dict[str, object] = {}

    # --- Phase 1: derive mesh_device ---
    weight_device = config.weight.device
    bias_device = config.bias.device
    if weight_device is not None and bias_device is not None and weight_device != bias_device:
        raise ValueError("LayerNorm weight and bias must target the same device")

    mesh_device = config.mesh_device
    if mesh_device is None:
        mesh_device = weight_device or bias_device
    if mesh_device is None:
        mesh_device = ttnn.GetDefaultDevice()
    if mesh_device is None:
        raise ValueError("Unable to resolve mesh_device")

    if weight_device is not None and weight_device != mesh_device:
        raise ValueError("LayerNorm weight must target the resolved mesh_device")
    if bias_device is not None and bias_device != mesh_device:
        raise ValueError("LayerNorm bias must target the resolved mesh_device")

    if config.mesh_device is None:
        to_set["mesh_device"] = mesh_device

    # --- Phase 2: determine hidden dim D and validate affine sizes ---
    dim = config.weight.source.numel()
    if config.bias.source.numel() != dim:
        raise ValueError(f"LayerNorm weight/bias numel mismatch: weight={dim}, bias={config.bias.source.numel()}")
    if dim % SHARD_HEIGHT != 0:
        raise ValueError(f"LayerNorm hidden dim must be divisible by {SHARD_HEIGHT}, got {dim}")

    # --- Phase 3: compute kernel config default ---
    if config.compute_kernel_config is None:
        to_set["compute_kernel_config"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # --- Phase 4: resolve local LazyWeights for replicated execution ---
    expected_shape = (1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT)

    weight_source = config.weight.source
    if tuple(weight_source.shape) != expected_shape:
        weight_source = weight_source.reshape(*expected_shape)
    bias_source = config.bias.source
    if tuple(bias_source.shape) != expected_shape:
        bias_source = bias_source.reshape(*expected_shape)

    transformed_weight = replace(config.weight, source=weight_source)
    transformed_bias = replace(config.bias, source=bias_source)

    to_set["weight"] = resolve_lazy_weight(
        transformed_weight,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper_config=None,
    )
    to_set["bias"] = resolve_lazy_weight(
        transformed_bias,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper_config=None,
    )

    return replace(config, **to_set)
