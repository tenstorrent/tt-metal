# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
from models.common.tensor_utils import nearest_32 as _nearest_32

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
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None
    # Logical output width for each split on one device. A physical weight shard
    # may be tile-padded beyond this width; forward trims it before concat.
    output_split_sizes: List[int] | None = None
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
        self.config = resolve_lm_head_1d_arch_config(LMHead1DConfig(output_weights=output_weights))
        self._device_weights_loaded = False

    @classmethod
    def from_config(cls, config: LMHead1DConfig):
        instance = object.__new__(cls)
        super(LMHead1D, instance).__init__()
        instance.config = resolve_lm_head_1d_arch_config(config)
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
        for weight, pc, logical_width in zip(self.output_weights, cfg.program_configs, cfg.output_split_sizes):
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
                output = ttnn.sharded_to_interleaved(output, memory_config=cfg.output_memcfg)
            else:
                # Auto path (simple API): interleaved output directly
                output = ttnn.linear(
                    x,
                    weight,
                    compute_kernel_config=cfg.compute_kernel_config,
                    memory_config=cfg.output_memcfg,
                    dtype=cfg.lm_head_dtype,
                )
            if output.shape[-1] != logical_width:
                output = ttnn.slice(
                    output,
                    (0, 0, 0, 0),
                    (output.shape[0], output.shape[1], output.shape[2], logical_width),
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
        padded_vocab_size = math.ceil(vocab_size / (TILE_SIZE * num_devices)) * (TILE_SIZE * num_devices)
        size_per_device = padded_vocab_size // num_devices
        num_splits = math.ceil(size_per_device / max_columns_per_device)
        split_sizes = [min(size_per_device, max_columns_per_device)] * (num_splits - 1)
        split_sizes.append(size_per_device - sum(split_sizes))

        # Build output weights
        source_weight = state_dict[f"{state_dict_prefix}output.weight"]
        if tuple(source_weight.shape) != (vocab_size, dim):
            raise ValueError(
                f"LMHead1D output weight must have shape {(vocab_size, dim)}, got {tuple(source_weight.shape)}"
            )
        torch_output_weights = source_weight.permute(1, 0)
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
            physical_split_size = math.ceil(split_size / TILE_SIZE) * TILE_SIZE
            for device_idx in range(num_devices):
                start = device_idx * size_per_device + sum(split_sizes[:i])
                end = start + split_size
                device_split = torch_output_weights[:, start:end]
                if split_size < physical_split_size:
                    device_split = torch.cat(
                        [
                            device_split,
                            torch.zeros(dim, physical_split_size - split_size, dtype=device_split.dtype),
                        ],
                        dim=-1,
                    )
                device_splits.append(device_split)
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
                    cache_dir_weight_name=(
                        (cache_dir, f"output_split_{i}_logical_{split_size}_physical_{combined_split.shape[-1]}")
                        if cache_dir
                        else None
                    ),
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
            output_split_sizes=split_sizes,
            lm_head_dtype=getattr(args, "lm_head_dtype", ttnn.bfloat8_b),
            output_memcfg=ttnn.L1_MEMORY_CONFIG,
            input_memcfg=input_memcfg,
            weights_memcfgs=weights_memcfgs,
        )
        return cls.from_config(config)


# =============================================================================
# Config resolution
# =============================================================================


def _compute_kernel_config_hifi2(arch) -> ttnn.DeviceComputeKernelConfig:
    """Construct the TTTv1 LM-head recipe for the selected architecture."""
    return ttnn.init_device_compute_kernel_config(
        arch,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _validate_lm_head_compute_kernel_config(compute_kernel_config) -> None:
    if compute_kernel_config is None:
        raise ValueError("LMHead1D architecture config requires compute_kernel_config")
    if not isinstance(compute_kernel_config, ttnn.WormholeComputeKernelConfig):
        raise TypeError(
            "LMHead1D compute_kernel_config must be a concrete TTNN compute kernel config, "
            f"got {type(compute_kernel_config).__name__}"
        )


def _derive_lm_head_mesh_device(config: LMHead1DConfig) -> ttnn.MeshDevice:
    if not config.output_weights:
        raise ValueError("LMHead1D requires at least one output weight")
    mesh_device = config.mesh_device
    if mesh_device is None:
        mesh_device = config.output_weights[0].device
    if mesh_device is None:
        mesh_device = ttnn.GetDefaultDevice()
    if mesh_device is None:
        raise ValueError("LMHead1D mesh_device must be available")
    for split_index, weight in enumerate(config.output_weights):
        if weight.device is not None and weight.device is not mesh_device:
            raise ValueError(f"LMHead1D mesh_device must match output weight {split_index} device")
    return mesh_device


def _validate_lm_head_program_configs(config: LMHead1DConfig) -> None:
    num_splits = len(config.output_weights)
    if num_splits == 0:
        raise ValueError("LMHead1D requires at least one output weight")
    if config.program_configs is None or len(config.program_configs) != num_splits:
        raise ValueError("LMHead1D requires exactly one program config per output-weight split")
    if config.weights_memcfgs is None or len(config.weights_memcfgs) != num_splits:
        raise ValueError("LMHead1D requires exactly one weight memory config per output-weight split")
    if config.output_split_sizes is None or len(config.output_split_sizes) != num_splits:
        raise ValueError("LMHead1D requires exactly one logical output width per output-weight split")

    num_devices = config.mesh_device.get_num_devices()
    physical_widths = []
    for split_index, (weight, logical_width) in enumerate(zip(config.output_weights, config.output_split_sizes)):
        source_shape = weight.source.shape
        if len(source_shape) < 2 or source_shape[-2] != config.dim:
            raise ValueError(
                f"LMHead1D split {split_index} weight K dimension must equal dim={config.dim}, got {source_shape}"
            )
        combined_width = source_shape[-1]
        if combined_width % num_devices:
            raise ValueError(
                f"LMHead1D split {split_index} combined physical width {combined_width} "
                f"must be divisible by {num_devices} devices"
            )
        physical_width = combined_width // num_devices
        if physical_width % TILE_SIZE:
            raise ValueError(
                f"LMHead1D split {split_index} physical per-device width {physical_width} must be tile aligned"
            )
        if not isinstance(logical_width, int) or logical_width <= 0 or logical_width > physical_width:
            raise ValueError(
                f"LMHead1D split {split_index} logical width must satisfy 0 < logical <= {physical_width}, "
                f"got {logical_width}"
            )
        physical_widths.append(physical_width)

    has_explicit_program = any(program_config is not None for program_config in config.program_configs)
    if has_explicit_program:
        if config.input_memcfg is None or not config.input_memcfg.is_sharded():
            raise ValueError("LMHead1D DRAM-sharded programs require a sharded input_memcfg")
        if config.input_memcfg.buffer_type != ttnn.BufferType.L1:
            raise ValueError("LMHead1D DRAM-sharded programs require input_memcfg in L1")
        if config.input_memcfg.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            raise ValueError("LMHead1D DRAM-sharded programs require width-sharded input_memcfg")
        num_compute_cores = config.input_memcfg.shard_spec.grid.num_cores()

    for split_index, program_config in enumerate(config.program_configs):
        if program_config is None:
            continue
        if not isinstance(program_config, ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig):
            raise TypeError(
                "LMHead1D explicit program configs must use the DRAM-sharded matmul path; "
                f"split {split_index} has {type(program_config).__name__}"
            )
        if program_config.in0_block_w <= 0 or program_config.per_core_M <= 0 or program_config.per_core_N <= 0:
            raise ValueError(f"LMHead1D split {split_index} program dimensions must be positive")
        dim_tiles = config.dim // TILE_SIZE
        if dim_tiles % program_config.in0_block_w != 0:
            raise ValueError(f"LMHead1D split {split_index} dim tiles must be divisible by in0_block_w")
        expected_per_core_m = math.ceil(config.max_batch_size / TILE_SIZE)
        if program_config.per_core_M != expected_per_core_m:
            raise ValueError(
                f"LMHead1D split {split_index} per_core_M={program_config.per_core_M} "
                f"does not match padded batch tiles {expected_per_core_m}"
            )

        weight_memcfg = config.weights_memcfgs[split_index]
        if not weight_memcfg.is_sharded():
            raise ValueError(f"LMHead1D split {split_index} DRAM-sharded program requires sharded weight memory")
        if weight_memcfg.buffer_type != ttnn.BufferType.DRAM:
            raise ValueError(f"LMHead1D split {split_index} weight memory must reside in DRAM")
        if weight_memcfg.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            raise ValueError(f"LMHead1D split {split_index} weight memory must be width sharded")
        covered_width = program_config.per_core_N * TILE_SIZE * num_compute_cores
        if covered_width < physical_widths[split_index]:
            raise ValueError(
                f"LMHead1D split {split_index} program covers {covered_width} columns, "
                f"below physical width {physical_widths[split_index]}"
            )


def resolve_lm_head_1d_arch_config(config: LMHead1DConfig) -> LMHead1DConfig:
    """Return an independent, fully resolved config after one architecture query."""
    mesh_device = _derive_lm_head_mesh_device(config)
    arch = mesh_device.arch()

    if arch not in (ttnn.device.Arch.WORMHOLE_B0, ttnn.device.Arch.BLACKHOLE):
        raise ValueError(f"Unsupported LMHead1D architecture: {arch}")
    requested_compute = config.compute_kernel_config or _compute_kernel_config_hifi2(arch)
    try:
        compute_kernel_config = ttnn.init_device_compute_kernel_config(arch, requested_compute)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid LMHead1D compute recipe for {arch}: {error}") from error

    _validate_lm_head_compute_kernel_config(compute_kernel_config)
    resolved_common = _resolve_lm_head_1d_config(config)
    resolved_common = replace(resolved_common, compute_kernel_config=compute_kernel_config)
    _validate_lm_head_program_configs(resolved_common)
    return resolved_common


def _resolve_lm_head_1d_config(config: LMHead1DConfig) -> LMHead1DConfig:
    """Resolve defaults for LMHead1DConfig."""
    if not config.output_weights:
        raise ValueError("LMHead1D requires at least one output weight")
    to_set = {}

    # Mesh device
    mesh_device = _derive_lm_head_mesh_device(config)
    if config.mesh_device is None:
        to_set["mesh_device"] = mesh_device

    # Dim
    dim = config.dim
    if dim is None:
        dim = config.output_weights[0].source.shape[-2]
        to_set["dim"] = dim

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

    if config.output_split_sizes is None:
        split_sizes = []
        for split_index, weight in enumerate(config.output_weights):
            combined_width = weight.source.shape[-1]
            if combined_width % num_devices:
                raise ValueError(
                    f"LMHead1D split {split_index} combined physical width {combined_width} "
                    f"must be divisible by {num_devices} devices"
                )
            split_sizes.append(combined_width // num_devices)
        to_set["output_split_sizes"] = split_sizes
    elif len(config.output_split_sizes) != len(config.output_weights):
        raise ValueError("LMHead1D requires exactly one logical output split size per output weight")

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


def _create_dram_sharded_mem_config(
    k: int, n: int, dram_grid: ttnn.CoreRangeSet, tile_size: int = TILE_SIZE, dram_cores: int = 12
) -> ttnn.MemoryConfig:
    padded_size = math.ceil(n / (tile_size * dram_cores)) * (tile_size * dram_cores)
    shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)
