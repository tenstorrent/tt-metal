# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MLPConfig: Pre-computed configuration for MLP module.

This separates config computation from forward pass logic, making the
MLP module cleaner and more testable.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Protocol

import ttnn


class HardwareTopology(Enum):
    """Hardware topology determines CCL strategy"""

    SINGLE_CHIP = auto()  # N150 - no CCL needed
    LINEAR_1D = auto()  # N300, T3K - reduce_scatter
    GALAXY_2D = auto()  # TG (32 devices) - 2D mesh CCL


@dataclass
class MLPLayerConfig:
    """Per-layer configuration (can vary by decoder layer for mixed precision)"""

    # Weight dtypes
    ff1_ff3_dtype: ttnn.DataType = ttnn.bfloat8_b
    ff2_dtype: ttnn.DataType = ttnn.bfloat8_b
    activation_dtype: Optional[ttnn.DataType] = None  # None = use input dtype

    # Compute kernel configs
    ff1_ff3_compute_config: Any = None
    ff2_compute_config: Any = None


@dataclass
class MLPProgramConfigs:
    """Program configs for matmul operations"""

    # Decode mode configs (static)
    decode_w1_w3: Any = None
    decode_w2: Any = None

    # Prefill mode configs (may be callables taking seq_len)
    prefill_w1_w3: Callable[[int], Any] = None
    prefill_w2: Callable[[int], Any] = None

    # TG-specific configs (only used when topology == GALAXY_2D)
    tg_w1_w3: Any = None
    tg_w2: Any = None


@dataclass
class MLPMemoryConfigs:
    """Memory configurations for MLP"""

    # Decode mode
    decode_output: Any = None  # L1_WIDTH_SHARDED or similar
    decode_residual: Any = None
    sharded_mlp2_input: Any = None
    sharded_attn_input: Any = None

    # CCL memory configs
    ff1_out_reduce_scatter: Any = None
    ff1_out_gathered: Any = None
    ff2_out_reduce_scatter: Any = None

    # Prefill mode
    prefill_output: Any = ttnn.DRAM_MEMORY_CONFIG


@dataclass
class MLPConfig:
    """
    Complete MLP configuration, pre-computed at model init time.

    This encapsulates all the config decisions so the forward pass
    can be clean and branch-free (within a given mode).
    """

    # Model dimensions
    dim: int
    hidden_dim: int

    # Hardware topology
    topology: HardwareTopology = HardwareTopology.SINGLE_CHIP
    cluster_shape: tuple = (1, 1)
    num_devices: int = 1

    # Prefill settings
    prefill_len_cutoff: int = 1024  # 512 for BH, 1024 for WH

    # Activation type
    activation_type: Any = None  # ttnn.UnaryOpType.SILU

    # Nested configs
    program_configs: MLPProgramConfigs = field(default_factory=MLPProgramConfigs)
    memory_configs: MLPMemoryConfigs = field(default_factory=MLPMemoryConfigs)

    # Per-layer configs (indexed by layer_num)
    # If None, use default_layer_config for all layers
    layer_configs: Optional[dict] = None
    default_layer_config: MLPLayerConfig = field(default_factory=MLPLayerConfig)

    def get_layer_config(self, layer_num: int) -> MLPLayerConfig:
        """Get config for a specific layer (for mixed precision support)"""
        if self.layer_configs and layer_num in self.layer_configs:
            return self.layer_configs[layer_num]
        return self.default_layer_config

    def get_program_config(self, mode: str, seq_len: int = 32) -> tuple:
        """Get (pc_w1_w3, pc_w2) for the given mode"""
        if mode == "decode":
            if self.topology == HardwareTopology.GALAXY_2D and self.dim >= 4096:
                return self.program_configs.tg_w1_w3, self.program_configs.tg_w2
            return self.program_configs.decode_w1_w3, self.program_configs.decode_w2
        else:  # prefill
            pc1 = self.program_configs.prefill_w1_w3
            pc2 = self.program_configs.prefill_w2
            # Prefill configs may be callables
            if callable(pc1):
                pc1 = pc1(seq_len)
            if callable(pc2):
                pc2 = pc2(seq_len)
            return pc1, pc2

    def get_memory_config(self, mode: str) -> Any:
        """Get output memory config for the given mode"""
        if mode == "decode":
            return self.memory_configs.decode_output
        return self.memory_configs.prefill_output

    @property
    def is_galaxy(self) -> bool:
        return self.topology == HardwareTopology.GALAXY_2D

    @property
    def needs_input_reshape(self) -> bool:
        """Whether prefill needs input reshaping"""
        return True  # Will check seq_len >= prefill_len_cutoff at runtime


class CCLStrategy(Protocol):
    """Protocol for CCL (Collective Communication) strategies"""

    def reduce_after_ff1_ff3(
        self,
        w1_out: ttnn.Tensor,
        w3_out: ttnn.Tensor,
        mode: str,
        config: MLPConfig,
    ) -> tuple:
        """Reduce w1 and w3 outputs across devices. Returns (w1_reduced, w3_reduced)"""
        ...

    def all_gather_before_ff2(
        self,
        tensor: ttnn.Tensor,
        mode: str,
        config: MLPConfig,
    ) -> ttnn.Tensor:
        """All-gather before FF2 (TG only). Returns gathered tensor."""
        ...

    def reduce_after_ff2(
        self,
        w2_out: ttnn.Tensor,
        mode: str,
        config: MLPConfig,
    ) -> ttnn.Tensor:
        """Reduce FF2 output across devices. Returns reduced tensor."""
        ...


def create_mlp_config_from_model_args(args, model_config: dict, layer_num: int) -> MLPConfig:
    """
    Factory function to create MLPConfig from ModelArgs and model_config dict.

    This bridges the gap between the current config system and the new modular one.
    """
    from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

    # Determine topology
    if args.num_devices == 1:
        topology = HardwareTopology.SINGLE_CHIP
    elif args.is_galaxy:
        topology = HardwareTopology.GALAXY_2D
    else:
        topology = HardwareTopology.LINEAR_1D

    # Build program configs
    prog_configs = MLPProgramConfigs(
        decode_w1_w3=model_config.get("DECODE_MLP_W1_W3_PRG_CONFIG"),
        decode_w2=model_config.get("DECODE_MLP_W2_PRG_CONFIG"),
        prefill_w1_w3=model_config.get("PREFILL_MLP_W1_W3_PRG_CONFIG"),
        prefill_w2=model_config.get("PREFILL_MLP_W2_PRG_CONFIG"),
        tg_w1_w3=model_config.get("FF1_3_TG_PROGCFG"),
        tg_w2=model_config.get("FF2_TG_PROGCFG"),
    )

    # Build memory configs
    mem_configs = MLPMemoryConfigs(
        decode_output=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        decode_residual=model_config.get("DECODE_RESIDUAL_MEMCFG"),
        sharded_mlp2_input=model_config.get("SHARDED_MLP2_INPUT_MEMCFG"),
        sharded_attn_input=model_config.get("SHARDED_ATTN_INPUT_MEMCFG"),
        ff1_out_reduce_scatter=model_config.get("FF1_OUT_REDUCE_SCATTER_MEMCFG"),
        ff1_out_gathered=model_config.get("FF1_OUT_GATHERED_MEMCFG"),
        ff2_out_reduce_scatter=model_config.get("FF2_OUT_REDUCE_SCATTER_MEMCFG"),
        prefill_output=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Build layer config from DECODERS_OPTIMIZATIONS
    decoders_opt = model_config.get("DECODERS_OPTIMIZATIONS")
    layer_config = MLPLayerConfig()

    if decoders_opt:
        layer_config.ff1_ff3_dtype = decoders_opt.get_tensor_dtype(layer_num, TensorGroup.FF1_FF3) or ttnn.bfloat8_b
        layer_config.ff2_dtype = decoders_opt.get_tensor_dtype(layer_num, TensorGroup.FF2) or ttnn.bfloat8_b
        layer_config.activation_dtype = decoders_opt.get_tensor_dtype(layer_num, TensorGroup.ACTIVATION)
        layer_config.ff1_ff3_compute_config = decoders_opt.get_math_fidelity(layer_num, OpGroup.LI_FF1_FF3, args)
        layer_config.ff2_compute_config = decoders_opt.get_math_fidelity(layer_num, OpGroup.LI_FF2, args)

    # Activation type
    activation_type = getattr(args, "mlp_activation_type", ttnn.UnaryOpType.SILU)

    return MLPConfig(
        dim=args.dim,
        hidden_dim=args.hidden_dim,
        topology=topology,
        cluster_shape=tuple(args.cluster_shape) if args.cluster_shape else (1, 1),
        num_devices=args.num_devices,
        prefill_len_cutoff=args.prefill_len_cutoff,
        activation_type=activation_type,
        program_configs=prog_configs,
        memory_configs=mem_configs,
        default_layer_config=layer_config,
    )
