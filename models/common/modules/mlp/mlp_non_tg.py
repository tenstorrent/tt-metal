# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2-style MLP module for non-TG (non-Galaxy) devices: N150, N300, T3K.
-- DP is important! regardless of Galaxy or not
-- everything that V1 supports -- including rudimentary Galaxy support! --> meaning that all the existing tests should still work.
-- Let's revisit Galaxy work if it turns out to be not worth

Single unified MLPNonTG class with separate forward methods:
  - decode_forward(): For decode mode (seq_len <= 32)
  - prefill_forward(): For prefill mode (seq_len > 32)
  - forward(x, mode): Dispatcher that calls the appropriate method

Execution paths:
  Decode:  linear(w1) → linear(w3) → mul+silu → reshard → linear(w2) → all_reduce(sharded) → reshard
  Prefill: [reshape] → linear(w1) → linear(w3) → mul+silu → linear(w2) → all_reduce → reshape

Config classes use a mixin pattern: subclass and override methods to customize behavior.
"""

# to stop Python from raising NameError: name 'MLPNonTGConfig' is not defined.
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

# =============================================================================
# Utility functions
# =============================================================================


def pad_dim_to_size(x: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """Pads the specified dimension of the input tensor with zeros."""
    if dim < 0:
        dim = x.dim() + dim
    current_size = x.size(dim)
    pad_size = size - current_size

    if pad_size < 0:
        raise ValueError(f"Target size {size} is smaller than current size {current_size} on dim {dim}")

    if pad_size == 0:
        return x

    pad = [0] * (2 * x.dim())
    pad_index = 2 * (x.dim() - dim - 1)
    pad[pad_index + 1] = pad_size

    return torch.nn.functional.pad(x, pad, mode="constant", value=0)


def ccl_topology_non_tg(num_devices: int):
    """CCL topology for non-TG devices."""
    if num_devices == 8 and ttnn.cluster.get_cluster_type() in [
        ttnn.cluster.ClusterType.T3K,
        ttnn.cluster.ClusterType.GALAXY,
    ]:
        return ttnn.Topology.Ring
    elif num_devices > 1:
        return ttnn.Topology.Linear
    return None


# =============================================================================
# Config helper functions (adapted from model_config.py)
# =============================================================================


def _find_largest_divisor(n: int, max_divisor: int = 8) -> int:
    """Find largest divisor of n up to max_divisor."""
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def _find_grid(n_tiles: int, max_rows: int = 8, max_cols: int = 8) -> tuple[int, int]:
    """Find grid dimensions (rows, cols) that evenly divide n_tiles."""
    max_cores = max_rows * max_cols
    target = 32
    possible_cores = [k for k in range(1, max_cores + 1) if n_tiles % k == 0]
    possible_cores.sort(key=lambda x: abs(x - target))

    for cores in possible_cores:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols

    raise AssertionError(f"Cannot find grid for {n_tiles} tiles within {max_rows}x{max_cols}")


def _find_grid_k_n(k_tiles: int, n_tiles: int, max_rows: int = 8, max_cols: int = 8) -> tuple[int, int]:
    """Find grid that evenly divides both K and N tile counts."""
    max_cores = max_rows * max_cols
    possible_cores = [c for c in range(1, max_cores + 1) if k_tiles % c == 0 and n_tiles % c == 0]
    possible_cores.sort(reverse=True)

    for cores in possible_cores:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_cols:
                    return rows, cols

    raise AssertionError(f"Cannot find grid for K={k_tiles}, N={n_tiles} tiles")


def _find_prefill_grid(row_tiles: int, col_tiles: int, max_rows: int = 8, max_cols: int = 8) -> tuple[int, int]:
    """Find grid where row_tiles divides rows and col_tiles divides cols."""
    cols = next((i for i in range(max_cols, 0, -1) if col_tiles % i == 0), None)
    rows = next((i for i in range(max_rows, 0, -1) if row_tiles % i == 0), None)
    assert cols is not None and rows is not None
    return rows, cols


def _get_out_subblock_w(per_core_n: int, out_subblock_h: int = 1) -> int:
    """Get output subblock width that divides per_core_n and satisfies constraints."""
    for w in range(min(8, per_core_n), 0, -1):
        if per_core_n % w == 0 and w * out_subblock_h <= 8:
            return w
    return 1


def _dram_shard_core_grid(k: int, tile_size: int = 32) -> ttnn.CoreGrid:
    """Get core grid for DRAM sharding based on K dimension."""
    rows, cols = _find_grid(k // tile_size)
    return ttnn.CoreGrid(x=cols, y=rows)


def _dram_shard_core_grid_k_n(k: int, n: int, tile_size: int = 32) -> ttnn.CoreGrid:
    """Get core grid for DRAM sharding based on K and N dimensions."""
    rows, cols = _find_grid_k_n(k // tile_size, n // tile_size)
    return ttnn.CoreGrid(x=cols, y=rows)


def _dram_matmul_config(
    m: int, k: int, n: int, num_cores: int, tile_size: int = 32, fused_activation=None
) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
    """Create DRAM-sharded matmul program config."""
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_find_largest_divisor(k // (tile_size * num_cores)),
        per_core_M=math.ceil(m / tile_size),
        per_core_N=math.ceil(n / (tile_size * num_cores)),
        fused_activation=fused_activation,
    )


def _matmul_config(
    m: int,
    k: int,
    n: int,
    grid_size: tuple[int, int],
    tile_size: int = 32,
    in0_block_w: int = None,
    fuse_batch: bool = False,
    fused_activation=None,
    per_core_m: int = None,
    per_core_n: int = None,
) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
    """Create multicast matmul program config."""
    if per_core_m is None:
        per_core_m = math.ceil(m / (tile_size * grid_size[1]))
    if per_core_n is None:
        per_core_n = math.ceil(n / (tile_size * grid_size[0]))

    out_subblock_h = 1
    out_subblock_w = _get_out_subblock_w(per_core_n, out_subblock_h)

    if in0_block_w is None:
        in0_block_w = _find_largest_divisor(k // (tile_size * grid_size[1]))

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=fuse_batch,
    )


def _compute_kernel_config_hifi2_fp16() -> ttnn.WormholeComputeKernelConfig:
    """Default compute kernel config for MLP (HiFi2 with FP16 accumulation)."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _create_dram_sharded_mem_config(
    k: int, n: int, dram_grid: ttnn.CoreRangeSet, tile_size: int = 32, dram_cores: int = 12
) -> ttnn.MemoryConfig:
    """Create DRAM-sharded memory config for weight tensors."""
    padded_size = math.ceil(n / (tile_size * dram_cores)) * (tile_size * dram_cores)
    shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


# =============================================================================
# Config classes with overridable methods (mixin pattern)
# =============================================================================


class MLPNonTGDecodeConfigs:
    """
    Decode config methods. Subclass and override to customize.

    All methods access terminal params via self.cfg (the parent MLPNonTGConfig).
    """

    def __init__(self, cfg: MLPNonTGConfig):
        self.cfg = cfg

    def _mlp_core_grid(self) -> ttnn.CoreGrid:
        """Core grid for w1/w3 matmuls. Override to customize."""
        return _dram_shard_core_grid_k_n(self.cfg.dim, self.cfg.hidden_dim // self.cfg.num_devices)

    def _mlp2_core_grid(self) -> ttnn.CoreGrid:
        """Core grid for w2 matmul. Override to customize."""
        return _dram_shard_core_grid_k_n(self.cfg.hidden_dim // self.cfg.num_devices, self.cfg.dim)

    def w1_w3_prg_config(self):
        """Program config for w1/w3 decode matmuls. Override to customize."""
        return _dram_matmul_config(
            m=self.cfg.tile_padded_batch_rows,
            k=self.cfg.dim,
            n=self.cfg.hidden_dim // self.cfg.num_devices,
            num_cores=self._mlp_core_grid().num_cores,
        )

    def w2_prg_config(self):
        """Program config for w2 decode matmul. Override to customize."""
        return _dram_matmul_config(
            m=self.cfg.tile_padded_batch_rows,
            k=self.cfg.hidden_dim // self.cfg.num_devices,
            n=self.cfg.dim,
            num_cores=self._mlp2_core_grid().num_cores,
        )

    def sharded_mlp2_input_memcfg(self):
        """Memory config for resharding before w2. Override to customize."""
        mlp2_grid = self._mlp2_core_grid()
        return ttnn.create_sharded_memory_config(
            (
                self.cfg.tile_padded_batch_rows,
                self.cfg.hidden_dim // self.cfg.num_devices // mlp2_grid.num_cores,
            ),
            mlp2_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def decode_residual_memcfg(self):
        """Memory config for final output. Override to customize."""
        residual_grid = _dram_shard_core_grid(self.cfg.dim // self.cfg.num_devices)
        return ttnn.create_sharded_memory_config(
            (
                self.cfg.tile_padded_batch_rows,
                self.cfg.dim // residual_grid.num_cores // self.cfg.num_devices,
            ),
            residual_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )


class MLPNonTGPrefillConfigs:
    """
    Prefill config methods. Subclass and override to customize.

    Methods take seq_len as argument since prefill configs are sequence-length dependent.
    """

    def __init__(self, cfg: MLPNonTGConfig):
        self.cfg = cfg

    def _mlp_grid(self, seq_len: int) -> tuple[int, int]:
        """Grid for prefill matmuls. Override to customize."""
        prefill_rows = 8
        return _find_prefill_grid(prefill_rows, self.cfg.dim // self.cfg.tile_size)

    def _mlp2_grid(self, seq_len: int) -> tuple[int, int]:
        """Grid for w2 prefill matmul. Override to customize."""
        prefill_rows = 8
        return _find_prefill_grid(prefill_rows, self.cfg.hidden_dim // self.cfg.tile_size)

    def w1_w3_prg_config(self, seq_len: int):
        """Program config for w1/w3 prefill matmuls. Override to customize."""
        n_w1_w3 = self.cfg.hidden_dim // self.cfg.num_devices
        dram_shard_grid_width = 8  # WH default
        return _matmul_config(
            m=min(seq_len, self.cfg.prefill_len_cutoff),
            k=self.cfg.dim,
            n=n_w1_w3,
            grid_size=self._mlp_grid(seq_len),
            per_core_n=math.ceil(n_w1_w3 / (self.cfg.tile_size * dram_shard_grid_width)),
        )

    def w2_prg_config(self, seq_len: int):
        """Program config for w2 prefill matmul. Override to customize."""
        n_w2 = self.cfg.dim
        dram_shard_grid_width = 8
        return _matmul_config(
            m=min(seq_len, self.cfg.prefill_len_cutoff),
            k=self.cfg.hidden_dim,
            n=n_w2,
            grid_size=self._mlp2_grid(seq_len),
            per_core_n=math.ceil(n_w2 / (self.cfg.tile_size * dram_shard_grid_width)),
        )


class MLPNonTGOptimizationConfig:
    """
    Optimization settings (dtypes, compute kernels). Subclass and override to customize.

    Default is 'performance' preset (BFP8 weights, HiFi2 FP16 accumulation).
    """

    def __init__(self, cfg: MLPNonTGConfig):
        self.cfg = cfg

    def ff1_3_dtype(self):
        """Dtype for w1/w3 weights. Override to customize."""
        return ttnn.bfloat8_b

    def ff2_dtype(self):
        """Dtype for w2 weights. Override to customize."""
        return ttnn.bfloat8_b

    def activation_dtype(self):
        """Dtype for activations. None means use default (bfloat16 for linear, bfloat8_b for mul)."""
        return None

    def li_ff1_3_compute_kernel_cfg(self):
        """Compute kernel config for w1/w3 matmuls. Override to customize."""
        return _compute_kernel_config_hifi2_fp16()

    def li_ff2_compute_kernel_cfg(self):
        """Compute kernel config for w2 matmul. Override to customize."""
        return _compute_kernel_config_hifi2_fp16()


# =============================================================================
# Top-level config dataclass
# =============================================================================


@dataclass
class MLPNonTGConfig:
    """
    Top-level configuration for non-TG MLP.

    Pass custom subclasses via _decode_cls, _prefill_cls, _optimization_cls to override behavior.
    """

    # Required terminal params
    dim: int
    hidden_dim: int
    # todo)){ derive these from the mesh_device? maybe mesh_device should be part of the config?
    num_devices: int
    cluster_shape: list  # [rows, cols] # this is just mesh_device.shape in model_config.py
    # }todo))

    prefill_len_cutoff: int  # 512 (BH) or 1024 (WH)

    # Optional params with sensible defaults
    # todo)){ should be from a static lookup table based on the device name?
    tile_size: int = 32
    max_batch_size: int = 32
    # }todo))
    dummy_weights: bool = False
    # todo)) should be part of the tt_ccl -- mesh_device?
    num_reduce_scatter_links: int = 1

    mlp_activation_type: Any = field(default_factory=lambda: ttnn.UnaryOpType.SILU)

    # DRAM grid for weight memory configs (set in __post_init__ if None)
    dram_grid: ttnn.CoreRangeSet = None
    dram_cores: int = 12  # WH default

    # Subclass hooks - pass custom classes to override config behavior
    _decode_cls: type = field(default=MLPNonTGDecodeConfigs, repr=False)
    _prefill_cls: type = field(default=MLPNonTGPrefillConfigs, repr=False)
    _optimization_cls: type = field(default=MLPNonTGOptimizationConfig, repr=False)

    # Computed fields (set in __post_init__)
    tile_padded_batch_rows: int = field(init=False)
    decode: MLPNonTGDecodeConfigs = field(init=False, repr=False)
    prefill: MLPNonTGPrefillConfigs = field(init=False, repr=False)
    optimization: MLPNonTGOptimizationConfig = field(init=False, repr=False)

    def __post_init__(self):
        # MLPNonTG uses 1D column-parallel sharding - 2D meshes not supported
        assert self.cluster_shape[0] == 1, (
            f"MLPNonTG only supports 1D meshes (cluster_shape[0] must be 1). "
            f"Got cluster_shape={self.cluster_shape}. For 2D meshes, use MLPTG instead."
        )

        self.tile_padded_batch_rows = self.tile_size * math.ceil(self.max_batch_size / self.tile_size)

        # Create default DRAM grid if not provided
        if self.dram_grid is None:
            self.dram_grid = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.dram_cores - 1, 0))}
            )

        # Instantiate sub-configs with self as parent
        self.decode = self._decode_cls(self)
        self.prefill = self._prefill_cls(self)
        self.optimization = self._optimization_cls(self)

    def w1_w3_mem_config(self):
        """Memory config for w1/w3 weights. 1D sharded: (dim, hidden_dim // num_devices)."""
        return _create_dram_sharded_mem_config(
            k=self.dim,
            n=self.hidden_dim // self.num_devices,
            dram_grid=self.dram_grid,
            tile_size=self.tile_size,
            dram_cores=self.dram_cores,
        )

    def w2_mem_config(self):
        """Memory config for w2 weights. 1D sharded: (hidden_dim // num_devices, dim)."""
        return _create_dram_sharded_mem_config(
            k=self.hidden_dim // self.num_devices,
            n=self.dim,
            dram_grid=self.dram_grid,
            tile_size=self.tile_size,
            dram_cores=self.dram_cores,
        )


# =============================================================================
# MLPNonTG - Unified MLP for non-TG devices with decode and prefill modes
# =============================================================================


# todo)) make the weights with LazyWeight
class MLPNonTG(LightweightModule):
    """
    MLP for non-TG devices supporting both decode and prefill modes.

    Execution paths:
      Decode:  linear(w1) → linear(w3) → mul+silu → reshard → linear(w2) → all_reduce(sharded) → reshard
      Prefill: [reshape] → linear(w1) → linear(w3) → mul+silu → linear(w2) → all_reduce → reshape
    """

    def __init__(
        self,
        # todo)) maybe we could group mesh_device and tt_ccl into a single object? OR implement singleton pattern in ccl.py --> then we can remove tt_ccl from the interface here and instead do a lookup of the singleton to use within the constructor!
        mesh_device,
        tt_ccl,
        config: MLPNonTGConfig,
        # todo)){ use LazyWeights and spell out query, key, and value separately?
        state_dict,
        weight_cache_path,
        layer_num: int,  # this is only used for cache naming!
        state_dict_prefix: Optional[str] = None,
        # }todo))
        # todo)) this could be grouped into tt_ccl?
        ccl_topology: Callable[[int], Any] | Any = ccl_topology_non_tg,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.config = config
        self.layer_num = layer_num
        self.ccl_topology = ccl_topology(config.num_devices) if callable(ccl_topology) else ccl_topology

        # Get optimization settings by calling methods
        opt = config.optimization
        self.activation_dtype = opt.activation_dtype()
        self.li_ff1_3_compute_kernel_cfg = opt.li_ff1_3_compute_kernel_cfg()
        self.li_ff2_compute_kernel_cfg = opt.li_ff2_compute_kernel_cfg()
        self.linear_dtype = self.activation_dtype or ttnn.bfloat16
        self.mul_dtype = self.activation_dtype or ttnn.bfloat8_b

        # Pre-compute all_reduce settings
        self._is_single_device = list(mesh_device.shape) == [1, 1]

        # Activation type
        self.activation_type = config.mlp_activation_type

        # Get decode configs by calling methods (computed once at init)
        self.decode_pc_w1_w3 = config.decode.w1_w3_prg_config()
        self.decode_pc_w2 = config.decode.w2_prg_config()
        self.sharded_mlp2_input_memcfg = config.decode.sharded_mlp2_input_memcfg()
        self.decode_residual_memcfg = config.decode.decode_residual_memcfg()

        # Store prefill config object for runtime method calls
        self._prefill_cfg = config.prefill
        self.prefill_len_cutoff = config.prefill_len_cutoff

        # Weight loading
        if state_dict_prefix is None:
            state_dict_prefix = f"layers.{layer_num}.feed_forward"

        torch_weight = lambda name: torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
        pad_hidden_dim = lambda tensor, dim_idx: pad_dim_to_size(tensor, dim=dim_idx, size=config.hidden_dim)

        if config.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}"

        # Get memory configs by calling methods
        w1_w3_mem_cfg = config.w1_w3_mem_config()
        w2_mem_cfg = config.w2_mem_config()

        def load_weight(name: str, w_dtype, shard_dim: int, mem_cfg) -> ttnn.Tensor:
            """Load weight with 1D sharding across all devices on shard_dim."""
            return ttnn.as_tensor(
                pad_hidden_dim(torch_weight(name[:2]), shard_dim),
                dtype=w_dtype,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=shard_dim),
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem_cfg,
                cache_file_name=cache_name(name),
            )

        # w1/w3: (dim, hidden_dim) sharded on dim=-1 (hidden_dim) -> (dim, hidden_dim // num_devices)
        # w2: (hidden_dim, dim) sharded on dim=-2 (hidden_dim) -> (hidden_dim // num_devices, dim)
        self.w1 = load_weight("w1_sharded", opt.ff1_3_dtype(), shard_dim=-1, mem_cfg=w1_w3_mem_cfg)
        self.w2 = load_weight("w2_sharded", opt.ff2_dtype(), shard_dim=-2, mem_cfg=w2_mem_cfg)
        self.w3 = load_weight("w3_sharded", opt.ff1_3_dtype(), shard_dim=-1, mem_cfg=w1_w3_mem_cfg)

    @classmethod
    def from_model_args(
        cls,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        weight_cache_path,
        layer_num: int,
        state_dict_prefix: Optional[str] = None,
        decode_cls: type = None,
        prefill_cls: type = None,
        optimization_cls: type = None,
    ):
        """
        Factory method for backward compatibility with ModelArgs.

        Pass custom config subclasses to override default behavior.
        """
        if args.is_galaxy:
            raise ValueError("MLPNonTG cannot be used for Galaxy devices.")

        # Get model_config once for all subclasses
        model_config = args.get_model_config()

        # Create subclass that uses model_config for decode settings (captures via closure)
        if decode_cls is None:

            class _ArgsDecodeConfigs(MLPNonTGDecodeConfigs):
                """Decode config using pre-computed values from model_config."""

                def w1_w3_prg_config(self):
                    return model_config.get("DECODE_MLP_W1_W3_PRG_CONFIG")

                def w2_prg_config(self):
                    return model_config.get("DECODE_MLP_W2_PRG_CONFIG")

                def sharded_mlp2_input_memcfg(self):
                    return model_config.get("SHARDED_MLP2_INPUT_MEMCFG")

                def decode_residual_memcfg(self):
                    return model_config.get("DECODE_RESIDUAL_MEMCFG")

            decode_cls = _ArgsDecodeConfigs

        # Create subclass that uses model_config for prefill settings (captures via closure)
        if prefill_cls is None:

            class _ArgsPrefillConfigs(MLPNonTGPrefillConfigs):
                """Prefill config using pre-computed lambdas from model_config."""

                def w1_w3_prg_config(self, seq_len: int):
                    return model_config.get("PREFILL_MLP_W1_W3_PRG_CONFIG")(seq_len)

                def w2_prg_config(self, seq_len: int):
                    return model_config.get("PREFILL_MLP_W2_PRG_CONFIG")(seq_len)

            prefill_cls = _ArgsPrefillConfigs

        # Create subclass that uses args for optimization settings (captures via closure)
        if optimization_cls is None:
            from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

            decoders_opt = model_config.get("DECODERS_OPTIMIZATIONS")
            effective_layer_num = max(layer_num, 0)

            class _ArgsOptimizationConfig(MLPNonTGOptimizationConfig):
                """Optimization config using DecodersPrecision from args."""

                def ff1_3_dtype(self):
                    return decoders_opt.get_tensor_dtype(decoder_id=effective_layer_num, tensor=TensorGroup.FF1_FF3)

                def ff2_dtype(self):
                    return decoders_opt.get_tensor_dtype(decoder_id=effective_layer_num, tensor=TensorGroup.FF2)

                def activation_dtype(self):
                    return decoders_opt.get_tensor_dtype(decoder_id=effective_layer_num, tensor=TensorGroup.ACTIVATION)

                def li_ff1_3_compute_kernel_cfg(self):
                    return decoders_opt.get_math_fidelity(
                        decoder_id=effective_layer_num, op=OpGroup.LI_FF1_FF3, configuration=args
                    )

                def li_ff2_compute_kernel_cfg(self):
                    return decoders_opt.get_math_fidelity(
                        decoder_id=effective_layer_num, op=OpGroup.LI_FF2, configuration=args
                    )

            optimization_cls = _ArgsOptimizationConfig

        config = MLPNonTGConfig(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            num_devices=args.num_devices,
            cluster_shape=args.cluster_shape,
            prefill_len_cutoff=args.prefill_len_cutoff,
            tile_size=args.tile_size,
            max_batch_size=args.max_batch_size,
            dummy_weights=args.dummy_weights,
            num_reduce_scatter_links=args.num_reduce_scatter_links,
            mlp_activation_type=getattr(args, "mlp_activation_type", ttnn.UnaryOpType.SILU),
            dram_grid=args.dram_weight_grid,
            dram_cores=args.dram_grid_size.x,
            _decode_cls=decode_cls,
            _prefill_cls=prefill_cls,
            _optimization_cls=optimization_cls,
        )

        if state_dict_prefix is None:
            state_dict_prefix = args.get_state_dict_prefix("MLP", layer_num)

        return cls(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            config=config,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            state_dict_prefix=state_dict_prefix,
            ccl_topology=args.ccl_topology(),
        )

    def _all_reduce_decode(self, w2_out: ttnn.Tensor) -> ttnn.Tensor:
        """All-reduce for decode mode (sharded input)."""
        if self._is_single_device:
            return w2_out

        original_shape = w2_out.shape
        if original_shape[0] != 1 or original_shape[1] != 1:
            w2_out = ttnn.reshape(
                w2_out, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
            )

        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            w2_out,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.config.num_reduce_scatter_links,
            memory_config=w2_out.memory_config(),
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.ccl_topology,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        w2_out.deallocate(True)
        return reduced

    def _all_reduce_prefill(self, w2_out: ttnn.Tensor) -> ttnn.Tensor:
        """All-reduce for prefill mode (interleaved input)."""
        if self._is_single_device:
            return w2_out

        original_shape = w2_out.shape
        if original_shape[0] != 1 or original_shape[1] != 1:
            w2_out = ttnn.reshape(
                w2_out, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
            )

        if w2_out.is_sharded():
            w2_out_sharded = w2_out
            w2_out = ttnn.sharded_to_interleaved(w2_out_sharded, ttnn.L1_MEMORY_CONFIG)
            w2_out_sharded.deallocate(True)

        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            w2_out,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.config.num_reduce_scatter_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.ccl_topology,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        w2_out.deallocate(True)
        return reduced

    def decode_forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Decode forward - NO if-else, fully flattened.

        Execution path:
          linear(w1) → linear(w3) → mul+silu → reshard → linear(w2) → all_reduce(sharded) → reshard
        """
        # --- STAGE 1: W1/W3 Linear (L1 sharded) ---
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=self.linear_dtype,
            core_grid=None,
            compute_kernel_config=self.li_ff1_3_compute_kernel_cfg,
            program_config=self.decode_pc_w1_w3,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=self.linear_dtype,
            core_grid=None,
            compute_kernel_config=self.li_ff1_3_compute_kernel_cfg,
            program_config=self.decode_pc_w1_w3,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        # --- STAGE 2: No CCL for non-TG ---

        # --- STAGE 3: Activation + Multiply ---
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[self.activation_type],
            dtype=self.mul_dtype,
            memory_config=w1_out.memory_config(),
        )

        # --- STAGE 3.5: Reshard for w2 ---
        w2_in = ttnn.to_memory_config(w2_in, self.sharded_mlp2_input_memcfg)

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        # --- STAGE 4: No all_gather for non-TG ---

        # --- STAGE 5: W2 Linear ---
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=self.li_ff2_compute_kernel_cfg,
            dtype=self.linear_dtype,
            program_config=self.decode_pc_w2,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            core_grid=None,
        )
        ttnn.deallocate(w2_in)

        # --- STAGE 6: Final All-Reduce (decode: sharded=True, no runtime branching) ---
        w2_out_reduced = self._all_reduce_decode(w2_out)

        # --- STAGE 7: Reshape + Final memory config ---
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )
        w2_out_reduced = ttnn.to_memory_config(w2_out_reduced, self.decode_residual_memcfg)

        return w2_out_reduced

    def prefill_forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Prefill forward - minimal runtime logic for seq_len-dependent configs.

        Execution path:
          [reshape if seq_len >= cutoff] → linear(w1) → linear(w3) → mul+silu → linear(w2) → all_reduce → reshape
        """
        seq_len = x.shape[-2]

        # Seq_len-dependent: reshape for long sequences
        if seq_len >= self.prefill_len_cutoff:
            x = ttnn.reshape(x, [1, seq_len // self.prefill_len_cutoff, self.prefill_len_cutoff, -1])

        # Seq_len-dependent: get program configs by calling methods
        pc_w1_w3 = self._prefill_cfg.w1_w3_prg_config(seq_len)
        pc_w2 = self._prefill_cfg.w2_prg_config(seq_len)

        # --- STAGE 1: W1/W3 Linear (DRAM) ---
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=self.linear_dtype,
            core_grid=None,
            compute_kernel_config=self.li_ff1_3_compute_kernel_cfg,
            program_config=pc_w1_w3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=self.linear_dtype,
            core_grid=None,
            compute_kernel_config=self.li_ff1_3_compute_kernel_cfg,
            program_config=pc_w1_w3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        # --- STAGE 2: No CCL for non-TG ---

        # --- STAGE 3: Activation + Multiply ---
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[self.activation_type],
            dtype=self.mul_dtype,
            memory_config=w1_out.memory_config(),
        )

        # --- STAGE 3.5: No reshard for prefill ---

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        # --- STAGE 4: No all_gather for non-TG ---

        # --- STAGE 5: W2 Linear ---
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=self.li_ff2_compute_kernel_cfg,
            dtype=self.linear_dtype,
            program_config=pc_w2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=None,
        )
        ttnn.deallocate(w2_in)

        # --- STAGE 6: Final All-Reduce (prefill: sharded=False) ---
        w2_out_reduced = self._all_reduce_prefill(w2_out)

        # --- STAGE 7: Reshape (no final memory config change for prefill) ---
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

        return w2_out_reduced

    def forward(self, x: ttnn.Tensor, mode: str) -> ttnn.Tensor:
        """Dispatch to the appropriate forward method based on mode."""
        if mode == "decode":
            return self.decode_forward(x)
        else:
            return self.prefill_forward(x)
