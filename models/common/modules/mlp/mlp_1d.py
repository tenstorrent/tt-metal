# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2-style MLP module for 1D-topology devices: N150 (1x1), N300 (1x2), T3K (1x8), Galaxy (1x32).

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
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.tensor_utils import pad_dim_to_size
from models.common.utility_functions import is_blackhole

# =============================================================================
# MeshContext - Hardware/runtime context
# =============================================================================


@dataclass
class MeshContext:
    """
    Hardware/runtime context for mesh devices and CCL.

    Encapsulates mesh_device and tt_ccl with derived properties and overridable methods.
    Follows the same mixin pattern as config classes - subclass and override methods to customize.
    """

    mesh_device: ttnn.MeshDevice
    tt_ccl: "TT_CCL"  # todo)) modeled after tt_transformers.tt.ccl.TT_CCL but we may want to make a ABC/Protocol for this!

    # Overridable methods
    def num_devices(self) -> int:
        return self.mesh_device.get_num_devices()

    def cluster_shape(self) -> list:
        return list(self.mesh_device.shape)

    def dram_grid_size(self) -> ttnn.CoreCoord:
        """Returns DRAM grid size CoreCoord from mesh_device."""
        return self.mesh_device.dram_grid_size()

    def topology(self) -> Any:
        """CCL topology. Override for custom behavior."""
        if self.num_devices() == 8 and ttnn.cluster.get_cluster_type() in [
            ttnn.cluster.ClusterType.T3K,
            ttnn.cluster.ClusterType.GALAXY,
        ]:
            return ttnn.Topology.Ring
        elif self.num_devices() > 1:
            return ttnn.Topology.Linear
        return None

    def num_reduce_scatter_links(self) -> int:
        """Number of reduce scatter links. Override for custom behavior."""
        return 1


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


class MLP1DDecodeConfigs:
    """
    Decode config methods. Subclass and override to customize.

    All methods access terminal params via self.cfg (the parent MLPNonTGConfig).
    """

    def __init__(self, cfg: MLP1DConfig):
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


class MLP1DPrefillConfigs:
    """
    Prefill config methods. Subclass and override to customize.

    Methods take seq_len as argument since prefill configs are sequence-length dependent.
    """

    def __init__(self, cfg: MLP1DConfig):
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


class MLP1DOptimizationConfig:
    """
    Optimization settings (dtypes, compute kernels). Subclass and override to customize.

    Default is 'performance' preset (BFP8 weights, HiFi2 FP16 accumulation).
    """

    def __init__(self, cfg: MLP1DConfig):
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
class MLP1DConfig:
    """
    Top-level configuration for non-TG MLP.

    Subclass and override decode/prefill/optimization properties to customize behavior.
    Weight sources are callables that return torch tensors (transposed, padded as needed).
    """

    # Required params
    dim: int
    hidden_dim: int
    mesh_ctx: MeshContext  # Hardware/runtime context

    # Optional params
    max_batch_size: int = 32
    mlp_activation_type: Any = field(default_factory=lambda: ttnn.UnaryOpType.SILU)

    def __post_init__(self):
        # MLPNonTG uses 1D column-parallel sharding - 2D meshes not supported
        assert self.cluster_shape[0] == 1, (
            f"MLPNonTG only supports 1D meshes (cluster_shape[0] must be 1). "
            f"Got cluster_shape={self.cluster_shape}. For 2D meshes, use MLPTG instead."
        )

    # Sub-configs - override these factory methods in subclasses
    @cached_property
    def decode_config(self) -> MLP1DDecodeConfigs:
        return MLP1DDecodeConfigs(self)

    @cached_property
    def prefill_config(self) -> MLP1DPrefillConfigs:
        return MLP1DPrefillConfigs(self)

    @cached_property
    def optimization_config(self) -> MLP1DOptimizationConfig:
        return MLP1DOptimizationConfig(self)

    # Cached properties - shorthand for mesh_ctx access
    @cached_property
    def num_devices(self) -> int:
        return self.mesh_ctx.num_devices()

    @cached_property
    def cluster_shape(self) -> list:
        return self.mesh_ctx.cluster_shape()

    @cached_property
    def dram_grid_size(self) -> ttnn.CoreCoord:
        return self.mesh_ctx.dram_grid_size()

    @cached_property
    def num_reduce_scatter_links(self) -> int:
        return self.mesh_ctx.num_reduce_scatter_links()

    # Computed properties (cached for efficiency)
    @cached_property
    def tile_size(self) -> int:
        return 32

    @cached_property
    def tile_padded_batch_rows(self) -> int:
        return self.tile_size * math.ceil(self.max_batch_size / self.tile_size)

    @cached_property
    def prefill_len_cutoff(self) -> int:
        return 512 if is_blackhole() else 1024

    @cached_property
    def dram_grid(self) -> ttnn.CoreRangeSet:
        dram_size = self.dram_grid_size
        return ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(dram_size.x - 1, dram_size.y - 1),
                )
            }
        )

    def w1_w3_mem_config(self):
        """Memory config for w1/w3 weights. 1D sharded: (dim, hidden_dim // num_devices)."""
        return _create_dram_sharded_mem_config(
            k=self.dim,
            n=self.hidden_dim // self.num_devices,
            dram_grid=self.dram_grid,
            tile_size=self.tile_size,
            dram_cores=self.dram_grid_size.x,
        )

    def w2_mem_config(self):
        """Memory config for w2 weights. 1D sharded: (hidden_dim // num_devices, dim)."""
        return _create_dram_sharded_mem_config(
            k=self.hidden_dim // self.num_devices,
            n=self.dim,
            dram_grid=self.dram_grid,
            tile_size=self.tile_size,
            dram_cores=self.dram_grid_size.x,
        )

    # Lazy weight descriptors - override these in subclass to provide weights
    @cached_property
    def lazy_w1(self) -> LazyWeight:
        raise NotImplementedError("Override lazy_w1 in subclass to provide w1 weight")

    @cached_property
    def lazy_w2(self) -> LazyWeight:
        raise NotImplementedError("Override lazy_w2 in subclass to provide w2 weight")

    @cached_property
    def lazy_w3(self) -> LazyWeight:
        raise NotImplementedError("Override lazy_w3 in subclass to provide w3 weight")

    # Materialized weights (lazy-loaded on first access via LazyWeight)
    @cached_property
    def w1(self) -> ttnn.Tensor:
        """w1 weight: (dim, hidden_dim) sharded on dim=-1."""
        return self.lazy_w1.get_weight()

    @cached_property
    def w2(self) -> ttnn.Tensor:
        """w2 weight: (hidden_dim, dim) sharded on dim=-2."""
        return self.lazy_w2.get_weight()

    @cached_property
    def w3(self) -> ttnn.Tensor:
        """w3 weight: (dim, hidden_dim) sharded on dim=-1."""
        return self.lazy_w3.get_weight()


# =============================================================================
# MLP1D - Unified MLP for 1D-topology devices (Linear or Ring) with decode and prefill modes
# =============================================================================


class MLP1D(LightweightModule):
    """
    MLP for non-TG devices supporting both decode and prefill modes.

    Execution paths:
      Decode:  linear(w1) → linear(w3) → mul+silu → reshard → linear(w2) → all_reduce(sharded) → reshard
      Prefill: [reshape] → linear(w1) → linear(w3) → mul+silu → linear(w2) → all_reduce → reshape
    """

    def __init__(self, config: MLP1DConfig):
        super().__init__()

        # Get hardware context from config
        mesh_ctx = config.mesh_ctx
        self.mesh_device = mesh_ctx.mesh_device
        self.tt_ccl = mesh_ctx.tt_ccl
        self.config = config
        self.ccl_topology = mesh_ctx.topology()

        # Get optimization settings
        opt = config.optimization_config
        self.activation_dtype = opt.activation_dtype()
        self.li_ff1_3_compute_kernel_cfg = opt.li_ff1_3_compute_kernel_cfg()
        self.li_ff2_compute_kernel_cfg = opt.li_ff2_compute_kernel_cfg()
        self.linear_dtype = self.activation_dtype or ttnn.bfloat16
        self.mul_dtype = self.activation_dtype or ttnn.bfloat8_b

        # Pre-compute all_reduce settings
        self._is_single_device = config.cluster_shape == [1, 1]

        # Activation type
        self.activation_type = config.mlp_activation_type

        # Get decode configs (computed once at init)
        self.decode_pc_w1_w3 = config.decode_config.w1_w3_prg_config()
        self.decode_pc_w2 = config.decode_config.w2_prg_config()
        self.sharded_mlp2_input_memcfg = config.decode_config.sharded_mlp2_input_memcfg()
        self.decode_residual_memcfg = config.decode_config.decode_residual_memcfg()

        # Store prefill config object for runtime method calls
        self._prefill_cfg = config.prefill_config
        self.prefill_len_cutoff = config.prefill_len_cutoff

        # Weights from config (lazy-loaded on first access)
        self.w1 = config.w1
        self.w2 = config.w2
        self.w3 = config.w3

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
    ):
        """Factory method for backward compatibility with ModelArgs."""
        if args.is_galaxy:
            raise ValueError("MLPNonTG cannot be used for Galaxy devices.")

        # Get model_config for sub-config closures
        model_config = args.get_model_config()
        decoders_opt = model_config.get("DECODERS_OPTIMIZATIONS")
        effective_layer_num = max(layer_num, 0)

        import torch

        from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

        # Create MLPNonTGConfig subclass with all overrides (captures args/model_config via closure)
        class _ArgsMLPNonTGConfig(MLP1DConfig):
            @cached_property
            def decode(self) -> MLP1DDecodeConfigs:
                class _Decode(MLP1DDecodeConfigs):
                    def w1_w3_prg_config(inner_self):
                        return model_config.get("DECODE_MLP_W1_W3_PRG_CONFIG")

                    def w2_prg_config(inner_self):
                        return model_config.get("DECODE_MLP_W2_PRG_CONFIG")

                    def sharded_mlp2_input_memcfg(inner_self):
                        return model_config.get("SHARDED_MLP2_INPUT_MEMCFG")

                    def decode_residual_memcfg(inner_self):
                        return model_config.get("DECODE_RESIDUAL_MEMCFG")

                return _Decode(self)

            @cached_property
            def prefill(self) -> MLP1DPrefillConfigs:
                class _Prefill(MLP1DPrefillConfigs):
                    def w1_w3_prg_config(inner_self, seq_len: int):
                        return model_config.get("PREFILL_MLP_W1_W3_PRG_CONFIG")(seq_len)

                    def w2_prg_config(inner_self, seq_len: int):
                        return model_config.get("PREFILL_MLP_W2_PRG_CONFIG")(seq_len)

                return _Prefill(self)

            @cached_property
            def optimization(self) -> MLP1DOptimizationConfig:
                class _Opt(MLP1DOptimizationConfig):
                    def ff1_3_dtype(inner_self):
                        return decoders_opt.get_tensor_dtype(decoder_id=effective_layer_num, tensor=TensorGroup.FF1_FF3)

                    def ff2_dtype(inner_self):
                        return decoders_opt.get_tensor_dtype(decoder_id=effective_layer_num, tensor=TensorGroup.FF2)

                    def activation_dtype(inner_self):
                        return decoders_opt.get_tensor_dtype(
                            decoder_id=effective_layer_num, tensor=TensorGroup.ACTIVATION
                        )

                    def li_ff1_3_compute_kernel_cfg(inner_self):
                        return decoders_opt.get_math_fidelity(
                            decoder_id=effective_layer_num, op=OpGroup.LI_FF1_FF3, configuration=args
                        )

                    def li_ff2_compute_kernel_cfg(inner_self):
                        return decoders_opt.get_math_fidelity(
                            decoder_id=effective_layer_num, op=OpGroup.LI_FF2, configuration=args
                        )

                return _Opt(self)

        # Create MeshContext subclass with args overrides
        class _ArgsMeshContext(MeshContext):
            def dram_grid_size(inner_self) -> ttnn.CoreCoord:
                return args.dram_grid_size

            def topology(inner_self):
                return args.ccl_topology()

            def num_reduce_scatter_links(inner_self) -> int:
                return args.num_reduce_scatter_links

        mesh_ctx = _ArgsMeshContext(mesh_device=mesh_device, tt_ccl=tt_ccl)

        if state_dict_prefix is None:
            state_dict_prefix = args.get_state_dict_prefix("MLP", layer_num)

        # Weight source factories (capture state_dict, prefix, hidden_dim via closure)
        def make_weight_source(name: str, shard_dim: int):
            def source():
                tensor = torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
                return pad_dim_to_size(tensor, dim=shard_dim, size=args.hidden_dim)

            return source

        cache_dir = None if args.dummy_weights else Path(weight_cache_path) / state_dict_prefix

        # Create LazyWeight instances (captures config, sources via closure)
        def make_lazy_weight(name: str, shard_dim: int, dtype_fn, mem_cfg_fn) -> LazyWeight:
            return LazyWeight(
                source=make_weight_source(name, shard_dim),
                dtype=dtype_fn(),
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=shard_dim),
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem_cfg_fn(),
                cache_dir=cache_dir,
                weight_name=f"{name}_sharded",
            )

        # We need config to exist first to get mem_config, so override lazy_* as cached_properties
        class _FinalConfig(_ArgsMLPNonTGConfig):
            @cached_property
            def lazy_w1(inner_self) -> LazyWeight:
                return make_lazy_weight("w1", -1, inner_self.optimization.ff1_3_dtype, inner_self.w1_w3_mem_config)

            @cached_property
            def lazy_w2(inner_self) -> LazyWeight:
                return make_lazy_weight("w2", -2, inner_self.optimization.ff2_dtype, inner_self.w2_mem_config)

            @cached_property
            def lazy_w3(inner_self) -> LazyWeight:
                return make_lazy_weight("w3", -1, inner_self.optimization.ff1_3_dtype, inner_self.w1_w3_mem_config)

        config = _FinalConfig(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            mesh_ctx=mesh_ctx,
            max_batch_size=args.max_batch_size,
            mlp_activation_type=getattr(args, "mlp_activation_type", ttnn.UnaryOpType.SILU),
        )

        return cls(config)

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
