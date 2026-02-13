# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2-style LM Head module for 1D-topology devices: N150 (1x1), N300 (1x2), T3K (1x8).

Computes logits over the vocabulary by splitting the output projection into
weight chunks that fit in L1, running linear ops per chunk, and concatenating.

Each device holds the correct logits for its vocab shard (column-parallel linear,
no partial sums). The caller handles all_gather for multi-device argmax.

Execution path:
  for each (weight, pc): linear(x, weight) → sharded_to_interleaved → append
  → concat
"""

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from models.common.tensor_utils import TILE_SIZE

# =============================================================================
# Config dataclass
# =============================================================================


@dataclass
class LMHead1DConfig:
    """
    Configuration for LMHead1D.

    Simple usage (pre-split weights):
        config = LMHead1DConfig(output_weights=[w1, w2, w3])

    The program_configs and other fields are auto-computed from weights if None.
    """

    # Required: output projection weights (already split for L1 fit)
    output_weights: List[LazyWeight]

    # Optional: device
    mesh_device: ttnn.MeshDevice | None = None

    # Optional: derived from weights if None
    dim: int | None = None

    # Optional: batch/tile config
    max_batch_size: int = 32

    # Optional: power-user overrides
    program_configs: List | None = None
    compute_kernel_config: ttnn.WormholeComputeKernelConfig | None = None
    lm_head_dtype: ttnn.DataType = ttnn.bfloat8_b
    output_memcfg: ttnn.MemoryConfig | None = None

    # Input memory config (None = DRAM interleaved for simple API, width-sharded for from_model_args)
    input_memcfg: ttnn.MemoryConfig | None = None

    # Weight memory configs (None = auto-compute)
    weights_memcfgs: List[ttnn.MemoryConfig] | None = None

    def is_resolved(self) -> bool:
        return all(getattr(self, f) is not None for f in self.__dataclass_fields__)


# =============================================================================
# LMHead1D
# =============================================================================


class LMHead1D(LightweightModule):
    """
    LM Head for non-TG (1D) devices.

    Splits vocabulary projection into L1-sized chunks, runs linear per chunk,
    and concatenates. Each device holds correct logits for its vocab shard;
    caller handles all_gather for multi-device argmax.

    Simple API:
        lm_head = LMHead1D(output_weights=[w1, w2])

    Power API:
        config = LMHead1DConfig(output_weights=[w1, w2], lm_head_dtype=ttnn.bfloat16)
        lm_head = LMHead1D.from_config(config)

    Execution path:
      for each (w, pc): linear(x, w) → sharded_to_interleaved → concat
    """

    def __init__(self, output_weights: List[LazyWeight]):
        super().__init__()
        self.config = _resolve_lm_head_1d_config(LMHead1DConfig(output_weights=output_weights))
        self._device_weights_loaded = False

    @classmethod
    def from_config(cls, config: LMHead1DConfig):
        instance = object.__new__(cls)
        super(LMHead1D, instance).__init__()
        instance.config = _resolve_lm_head_1d_config(config)
        instance._device_weights_loaded = False
        return instance

    def load_device_weights(self):
        if self._device_weights_loaded:
            return
        self.output_weights = [w.get_device_weight() for w in self.config.output_weights]
        self._device_weights_loaded = True

    def forward(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """
        Compute logits over vocabulary.

        Args:
            x: Input hidden states, shape [1, 1, batch_rows, dim].

        Returns:
            Logits tensor, shape [1, 1, batch_rows, padded_vocab / num_devices] per device.
        """
        self.load_device_weights()
        x = _load_input_device_tensor(x, self.config)
        cfg = self.config

        outputs = []
        for weight, pc in zip(self.output_weights, cfg.program_configs):
            if pc is not None:
                # DRAM-sharded path (from_model_args): width-sharded output → interleaved
                output = ttnn.linear(
                    x,
                    weight,
                    compute_kernel_config=cfg.compute_kernel_config,
                    program_config=pc,
                    memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                    dtype=cfg.lm_head_dtype,
                )
                outputs.append(ttnn.sharded_to_interleaved(output, memory_config=cfg.output_memcfg))
            else:
                # Auto path (simple API): interleaved output directly
                output = ttnn.linear(
                    x,
                    weight,
                    compute_kernel_config=cfg.compute_kernel_config,
                    memory_config=cfg.output_memcfg,
                    dtype=cfg.lm_head_dtype,
                )
                outputs.append(output)

        # Concatenate splits
        output = ttnn.concat(outputs, dim=-1, memory_config=cfg.output_memcfg)

        return output

    # [INFO] this is the entry point for TTTv1 model_config.py and will retire with TTTv1
    @classmethod
    def from_model_args(
        cls,
        mesh_device,
        args,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        max_columns_per_device,
        dtype=None,
        model_config=None,
        tt_ccl=None,
    ):
        """Factory method for backward compatibility with ModelArgs.

        Note: tt_ccl is accepted for signature compatibility with TTTv1 LMHead
        but is not used -- 1D LMHead does not need CCL (column-parallel, no partial sums).
        """
        if args.is_galaxy:
            raise ValueError("LMHead1D cannot be used for Galaxy devices.")

        import torch

        vocab_size = args.vocab_size
        num_devices = mesh_device.get_num_devices()
        dim = args.dim
        padded_vocab_size = math.ceil(vocab_size / 32) * 32
        size_per_device = padded_vocab_size // num_devices
        num_splits = math.ceil(size_per_device / max_columns_per_device)
        split_sizes = [min(size_per_device, max_columns_per_device)] * (num_splits - 1)
        split_sizes.append(size_per_device - sum(split_sizes))

        # Build output weights
        torch_output_weights = state_dict[f"{state_dict_prefix}output.weight"].permute(1, 0)
        if vocab_size < padded_vocab_size:
            padding_size = padded_vocab_size - vocab_size
            torch_output_weights = torch.cat(
                [
                    torch_output_weights,
                    torch.zeros(torch_output_weights.shape[0], padding_size, dtype=torch_output_weights.dtype),
                ],
                dim=-1,
            )

        # DRAM grid for weight memory configs
        dram_size = mesh_device.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_size.x - 1, dram_size.y - 1))}
        )

        cache_dir = None if args.dummy_weights else Path(weight_cache_path) / "lm_head"

        output_weights = []
        weights_memcfgs = []
        for i, split_size in enumerate(split_sizes):
            device_splits = []
            for device_idx in range(num_devices):
                start = device_idx * size_per_device + sum(split_sizes[:i])
                end = start + split_size
                device_splits.append(torch_output_weights[:, start:end])
            combined_split = torch.cat(device_splits, dim=-1)

            mem_cfg = _create_dram_sharded_mem_config(
                k=dim,
                n=math.ceil(combined_split.shape[-1] / num_devices),
                dram_grid=dram_grid,
                tile_size=TILE_SIZE,
                dram_cores=dram_size.x,
            )
            weights_memcfgs.append(mem_cfg)

            w_dtype = dtype if dtype is not None else ttnn.bfloat8_b
            output_weights.append(
                LazyWeight(
                    source=combined_split,
                    dtype=w_dtype,
                    device=mesh_device,
                    mesh_mapper_config=ttnn.MeshMapperConfig(
                        placements=[ttnn.PlacementShard(-1)],
                        mesh_shape_override=ttnn.MeshShape([num_devices]),
                    ),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=mem_cfg,
                    cache_dir_weight_name=(cache_dir, f"output_split_{i}_{combined_split.shape[-1]}")
                    if cache_dir
                    else None,
                )
            )

        # Program configs - use the args.lm_head_core_grid which TTTv1 carefully computes
        tile_padded_batch_rows = TILE_SIZE * math.ceil(args.max_batch_size / TILE_SIZE)
        lm_head_core_grid = args.lm_head_core_grid
        program_configs = [
            args.dram_matmul_config(
                tile_padded_batch_rows,
                dim,
                ss,
                lm_head_core_grid.num_cores,
            )
            for ss in split_sizes
        ]

        # Get input memory config: width-sharded across lm_head_core_grid
        # (DRAM-sharded matmul requires width-sharded input in L1)
        input_memcfg = ttnn.create_sharded_memory_config(
            (
                tile_padded_batch_rows,
                _nearest_32(dim // lm_head_core_grid.num_cores),
            ),
            lm_head_core_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        config = LMHead1DConfig(
            output_weights=output_weights,
            mesh_device=mesh_device,
            dim=dim,
            max_batch_size=args.max_batch_size,
            program_configs=program_configs,
            compute_kernel_config=_compute_kernel_config_hifi2(),
            lm_head_dtype=getattr(args, "lm_head_dtype", ttnn.bfloat8_b),
            output_memcfg=ttnn.L1_MEMORY_CONFIG,
            input_memcfg=input_memcfg,
            weights_memcfgs=weights_memcfgs,
        )
        return cls.from_config(config)


# =============================================================================
# Config resolution
# =============================================================================


def _resolve_lm_head_1d_config(config: LMHead1DConfig) -> LMHead1DConfig:
    """Resolve defaults for LMHead1DConfig."""
    to_set = {}

    # Mesh device
    mesh_device = config.mesh_device
    if mesh_device is None:
        mesh_device = config.output_weights[0].device
    if mesh_device is None:
        mesh_device = ttnn.GetDefaultDevice()
    if config.mesh_device is None:
        to_set["mesh_device"] = mesh_device

    assert mesh_device is not None

    # Dim
    dim = config.dim
    if dim is None:
        dim = config.output_weights[0].source.shape[-2]
        to_set["dim"] = dim

    # Compute kernel config
    if config.compute_kernel_config is None:
        to_set["compute_kernel_config"] = _compute_kernel_config_hifi2()

    # Output memcfg
    if config.output_memcfg is None:
        to_set["output_memcfg"] = ttnn.L1_MEMORY_CONFIG

    # Input memcfg
    if config.input_memcfg is None:
        to_set["input_memcfg"] = ttnn.DRAM_MEMORY_CONFIG

    # Program configs
    num_devices = mesh_device.get_num_devices()

    if config.program_configs is None:
        # Use None program configs for auto-resolve (let ttnn.linear auto-select).
        # DRAM-sharded program configs require matching DRAM-sharded weight memory configs,
        # which are only set up correctly via from_model_args.
        pcs = [None for _ in config.output_weights]
        to_set["program_configs"] = pcs

    # Weight memory configs + resolve LazyWeights
    if config.weights_memcfgs is None:
        # Use regular DRAM for auto-resolve (DRAM sharded requires DRAM-core-aligned padding
        # which is handled by from_model_args when it explicitly provides weights_memcfgs)
        memcfgs = [ttnn.DRAM_MEMORY_CONFIG for _ in config.output_weights]
        to_set["weights_memcfgs"] = memcfgs

    weights_memcfgs = (
        config.weights_memcfgs if config.weights_memcfgs is not None else to_set.get("weights_memcfgs", [])
    )

    resolved_weights = []
    for i, w in enumerate(config.output_weights):
        mem_cfg = weights_memcfgs[i] if i < len(weights_memcfgs) else ttnn.DRAM_MEMORY_CONFIG
        resolved_weights.append(
            resolve_lazy_weight(
                w,
                device=mesh_device,
                memory_config=mem_cfg,
                mesh_mapper_config=ttnn.MeshMapperConfig(
                    placements=[ttnn.PlacementShard(-1)],
                    mesh_shape_override=ttnn.MeshShape([num_devices]),
                ),
                layout=ttnn.TILE_LAYOUT,
                dtype=config.lm_head_dtype,
            )
        )
    to_set["output_weights"] = resolved_weights

    resolved = replace(config, **to_set)
    return resolved


def _load_input_device_tensor(x: ttnn.Tensor | LazyWeight, config: LMHead1DConfig) -> ttnn.Tensor:
    """Resolve input tensor using config's input_memcfg."""
    if isinstance(x, LazyWeight):
        resolved_x = resolve_lazy_weight(
            x,
            device=config.mesh_device,
            memory_config=config.input_memcfg,
            mesh_mapper_config=None,
            layout=ttnn.TILE_LAYOUT,
        )
        return resolved_x.get_device_weight()
    assert isinstance(x, ttnn.Tensor)
    return x


# =============================================================================
# Config helper functions (adapted from TTTv1 model_config.py)
# =============================================================================


def _nearest_32(x: int) -> int:
    return math.ceil(x / 32) * 32


def _compute_kernel_config_hifi2() -> ttnn.WormholeComputeKernelConfig:
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _create_dram_sharded_mem_config(
    k: int, n: int, dram_grid: ttnn.CoreRangeSet, tile_size: int = TILE_SIZE, dram_cores: int = 12
) -> ttnn.MemoryConfig:
    padded_size = math.ceil(n / (tile_size * dram_cores)) * (tile_size * dram_cores)
    shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)
