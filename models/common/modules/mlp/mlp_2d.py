# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2-style MLP module for TG (Galaxy) devices with 2D mesh topology.

Single unified MLP2D class with separate forward methods:
  - decode_forward(): For decode mode (seq_len <= 32)
  - prefill_forward(): For prefill mode (seq_len > 32)
  - forward(x, mode): Dispatcher that calls the appropriate method

Execution paths (still has dim-based branching):
  - dim < 8192 decode:  linear → linear → all_reduce(×2) → mul+silu → linear → all_reduce
  - dim >= 8192 OR prefill: linear → linear → reduce_scatter(×2) → mul+silu → all_gather → linear → all_reduce

Config classes use a mixin pattern: subclass and override methods to customize behavior.
"""

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
# MeshContext2D - Hardware/runtime context for 2D mesh
# =============================================================================


@dataclass
class MeshContext2D:
    """
    Hardware/runtime context for 2D mesh devices and CCL.

    Encapsulates mesh_device and tt_ccl with derived properties and overridable methods.
    Follows the same mixin pattern as config classes - subclass and override methods to customize.
    """

    mesh_device: ttnn.MeshDevice
    tt_ccl: "TT_CCL"

    # NOTE: we have TTNN APIs and do we really need extra boilerplates here? --> we should distinguish between users of TTTv2 and maintainers of TTTv2 --> the API tightening is useful for only the maintainers --> the users do not care about!
    # it is OK to group mesh_device and tt_ccl together in the config classes but we should not do that in the MLP2D class!
    # Overridable methods
    def num_devices(self) -> int:
        return self.mesh_device.get_num_devices()

    def cluster_shape(self) -> list:
        return list(self.mesh_device.shape)

    def dram_grid_size(self) -> ttnn.CoreCoord:
        """Returns DRAM grid size CoreCoord from mesh_device."""
        return self.mesh_device.dram_grid_size()

    # todo)) maybe this should be a TTNN API? Turn this into a helper function that has a tt-metal issue created for it to TTNN ops team
    def topology(self) -> Any:
        """CCL topology. Override for custom behavior."""
        if self.num_devices() == 8 and ttnn.cluster.get_cluster_type() in [
            ttnn.cluster.ClusterType.T3K,
            ttnn.cluster.ClusterType.GALAXY,
        ]:
            # NOTE: we always want to do ring if it is available
            return ttnn.Topology.Ring
        elif self.num_devices() > 1:
            # NOTE: this should be a fallback when the ring is not available
            return ttnn.Topology.Linear
        return None

    # todo)) maybe this should be a TTNN API? Turn this into a helper function that has a tt-metal issue created for it to TTNN ops team
    def num_reduce_scatter_links(self) -> int:
        """Number of reduce scatter links. Override for custom behavior."""
        return 1

    # todo)) maybe this should be a TTNN API? Turn this into a helper function that has a tt-metal issue created for it to TTNN ops team
    def num_all_gather_links(self) -> int:
        """Number of all gather links. Override for custom behavior."""
        return 2

    def ccl_dtype(self) -> Any:
        """CCL dtype. Override for custom behavior."""
        return ttnn.bfloat8_b


# =============================================================================
# Config helper functions (adapted from model_config.py for TG)
# =============================================================================


def _find_largest_divisor(n: int, max_divisor: int = 8) -> int:
    """Find largest divisor of n up to max_divisor."""
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def _find_prefill_grid(row_tiles: int, col_tiles: int, max_rows: int = 8, max_cols: int = 8) -> tuple[int, int]:
    """Find grid where row_tiles divides rows and col_tiles divides cols."""
    cols = next((i for i in range(max_cols, 0, -1) if col_tiles % i == 0), None)
    rows = next((i for i in range(max_rows, 0, -1) if row_tiles % i == 0), None)
    assert cols is not None and rows is not None
    # NOTE: TTNN matmul program configs expect (cores_x, cores_y).
    # Here, `cols` corresponds to X and `rows` corresponds to Y.
    return cols, rows


def _get_out_subblock_w(per_core_n: int, out_subblock_h: int = 1) -> int:
    """Get output subblock width that divides per_core_n and satisfies constraints."""
    for w in range(min(8, per_core_n), 0, -1):
        if per_core_n % w == 0 and w * out_subblock_h <= 8:
            return w
    return 1


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
        # For 2D matmul, K is partitioned along the grid X dimension (cores_x).
        # `grid_size` is (cores_x, cores_y).
        in0_block_w = _find_largest_divisor(k // (tile_size * grid_size[0]))

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


# =============================================================================
# Config classes with overridable methods (mixin pattern)
# =============================================================================


class MLP2DDecodeConfigs:
    """
    Decode config methods for TG. Subclass and override to customize.

    All methods access terminal params via self.cfg (the parent MLP2DConfig).
    """

    def __init__(self, cfg: MLP2DConfig):
        self.cfg = cfg

    def ff1_3_prg_config(self):
        """Program config for w1/w3 decode matmuls (TG). Override to customize."""
        # TG uses simpler program config or None for small dims
        if self.cfg.dim >= 4096:
            # For large models, use a specific TG config
            # This is typically provided via model_config in practice
            return None  # Override in subclass or from_model_args
        return None

    def ff2_prg_config(self):
        """Program config for w2 decode matmul (TG). Override to customize."""
        if self.cfg.dim >= 4096:
            return None  # Override in subclass or from_model_args
        return None

    def ff1_out_reduce_scatter_memcfg(self):
        """Memory config for reduce_scatter output after ff1. Override to customize."""
        return None  # Override in subclass or from_model_args

    def ff1_out_gathered_memcfg(self):
        """Memory config for all_reduce output after ff1. Override to customize."""
        return None  # Override in subclass or from_model_args

    def ff2_out_reduce_scatter_memcfg(self):
        """Memory config for reduce_scatter output after ff2. Override to customize."""
        return None  # Override in subclass or from_model_args

    def sharded_attn_input_memcfg(self):
        """Memory config for final output (sharded for attention input). Override to customize."""
        return None  # Override in subclass or from_model_args


class MLP2DPrefillConfigs:
    """
    Prefill config methods for TG. Subclass and override to customize.

    Methods take seq_len as argument since prefill configs are sequence-length dependent.
    """

    def __init__(self, cfg: MLP2DConfig):
        self.cfg = cfg

    def _mlp_grid(self, seq_len: int) -> tuple[int, int]:
        """Grid for prefill matmuls. Override to customize."""
        # Pick a grid that evenly divides both M (rows) and local N (cols) in tiles
        # to avoid padding-sensitive matmul configs.
        mesh_dim0, _mesh_dim1 = self.cfg.cluster_shape
        m = min(seq_len, self.cfg.prefill_len_cutoff)
        m_tiles = m // self.cfg.tile_size
        n_local = self.cfg.hidden_dim // mesh_dim0
        n_tiles = n_local // self.cfg.tile_size
        return _find_prefill_grid(m_tiles, n_tiles)

    def _mlp2_grid(self, seq_len: int) -> tuple[int, int]:
        """Grid for w2 prefill matmul. Override to customize."""
        # Pick a grid that evenly divides both M (rows) and local N (cols) in tiles.
        _mesh_dim0, mesh_dim1 = self.cfg.cluster_shape
        m = min(seq_len, self.cfg.prefill_len_cutoff)
        m_tiles = m // self.cfg.tile_size
        n_local = self.cfg.dim // mesh_dim1
        n_tiles = n_local // self.cfg.tile_size
        return _find_prefill_grid(m_tiles, n_tiles)

    def w1_w3_prg_config(self, seq_len: int):
        """Program config for w1/w3 prefill matmuls. Override to customize."""
        # TG uses 2D weight sharding:
        # - w1/w3 shard dims = (-1, -2) => N sharded across mesh_dim0 (cluster_shape[0]),
        #                               K sharded across mesh_dim1 (cluster_shape[1])
        # Therefore, per-device matmul shape is:
        #   (M x K_local) @ (K_local x N_local)
        # where:
        #   K_local = dim / cluster_shape[1]
        #   N_local = hidden_dim / cluster_shape[0]
        mesh_dim0, mesh_dim1 = self.cfg.cluster_shape
        k_local = self.cfg.dim // mesh_dim1
        n_local = self.cfg.hidden_dim // mesh_dim0
        return _matmul_config(
            m=min(seq_len, self.cfg.prefill_len_cutoff),
            k=k_local,
            n=n_local,
            grid_size=self._mlp_grid(seq_len),
        )

    def w2_prg_config(self, seq_len: int):
        """Program config for w2 prefill matmul. Override to customize."""
        # TG uses 2D weight sharding:
        # - w2 shard dims = (-2, -1) => K sharded across mesh_dim0 (cluster_shape[0]),
        #                              N sharded across mesh_dim1 (cluster_shape[1])
        # Therefore, per-device matmul shape is:
        #   (M x K_local) @ (K_local x N_local)
        # where:
        #   K_local = hidden_dim / cluster_shape[0]
        #   N_local = dim / cluster_shape[1]
        mesh_dim0, mesh_dim1 = self.cfg.cluster_shape
        k_local = self.cfg.hidden_dim // mesh_dim0
        n_local = self.cfg.dim // mesh_dim1
        return _matmul_config(
            m=min(seq_len, self.cfg.prefill_len_cutoff),
            k=k_local,
            n=n_local,
            grid_size=self._mlp2_grid(seq_len),
        )


class MLP2DOptimizationConfig:
    """
    Optimization settings for TG (dtypes, compute kernels). Subclass and override to customize.

    Default is 'performance' preset (BFP8 weights, HiFi2 FP16 accumulation).
    """

    def __init__(self, cfg: MLP2DConfig):
        self.cfg = cfg

    def ff1_3_dtype(self):
        """Dtype for w1/w3 weights. Override to customize."""
        return ttnn.bfloat8_b

    def ff2_dtype(self):
        """Dtype for w2 weights. Override to customize."""
        return ttnn.bfloat8_b

    def activation_dtype(self):
        """Dtype for activations. None means use default (bfloat8_b for TG)."""
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
class MLP2DConfig:
    """
    Top-level configuration for TG (Galaxy) MLP with 2D mesh.

    Subclass and override decode/prefill/optimization properties to customize behavior.
    Weight sources are callables that return torch tensors (transposed, padded as needed).
    """

    # Required params
    dim: int
    hidden_dim: int
    mesh_ctx: MeshContext2D  # Hardware/runtime context

    # Optional params
    max_batch_size: int = 32
    mlp_activation_type: Any = field(default_factory=lambda: ttnn.UnaryOpType.SILU)

    def __post_init__(self):
        # MLP2D is designed for 2D mesh topologies (cluster_shape[0] > 1 and cluster_shape[1] > 1)
        # Note: from_model_args() enforces Galaxy (4x8 or 8x4) because it uses model_config.py
        # which has Galaxy-specific hardcoded values. Direct MLP2DConfig usage is more flexible.
        assert self.cluster_shape[0] > 1 and self.cluster_shape[1] > 1, (
            f"MLP2D requires 2D mesh (both cluster_shape dimensions > 1). "
            f"Got cluster_shape={self.cluster_shape}. For 1D meshes, use MLP1D instead."
        )

    # todo)) make the configs flatter --> one click away! --> this is what killed deepseek code
    # Sub-configs - override these factory methods in subclasses
    @cached_property
    def decode_config(self) -> MLP2DDecodeConfigs:
        return MLP2DDecodeConfigs(self)

    @cached_property
    def prefill_config(self) -> MLP2DPrefillConfigs:
        return MLP2DPrefillConfigs(self)

    @cached_property
    def optimization_config(self) -> MLP2DOptimizationConfig:
        return MLP2DOptimizationConfig(self)

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

    @cached_property
    def num_all_gather_links(self) -> int:
        return self.mesh_ctx.num_all_gather_links()

    @cached_property
    def ccl_dtype(self) -> Any:
        return self.mesh_ctx.ccl_dtype()

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

    # TG uses DRAM_MEMORY_CONFIG for weights (not DRAM-sharded like 1D)
    def w1_w3_mem_config(self):
        """Memory config for w1/w3 weights. TG uses DRAM interleaved."""
        return ttnn.DRAM_MEMORY_CONFIG

    def w2_mem_config(self):
        """Memory config for w2 weights. TG uses DRAM interleaved."""
        return ttnn.DRAM_MEMORY_CONFIG

    # 2D sharding dims for weights
    @cached_property
    def w1_shard_dims(self) -> tuple[int, int]:
        """Shard dims for w1/w3: (rows, cols) for ShardTensor2dMesh."""
        return (-1, -2)  # TG: shard on last two dims

    @cached_property
    def w2_shard_dims(self) -> tuple[int, int]:
        """Shard dims for w2: (rows, cols) for ShardTensor2dMesh."""
        return (-2, -1)  # TG: shard on last two dims (transposed)

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
        """w1 weight: (dim, hidden_dim) 2D-sharded."""
        return self.lazy_w1.get_device_weight()

    @cached_property
    def w2(self) -> ttnn.Tensor:
        """w2 weight: (hidden_dim, dim) 2D-sharded."""
        return self.lazy_w2.get_device_weight()

    @cached_property
    def w3(self) -> ttnn.Tensor:
        """w3 weight: (dim, hidden_dim) 2D-sharded."""
        return self.lazy_w3.get_device_weight()


# =============================================================================
# MLP2D - MLP for 2D-topology (Galaxy) devices with decode and prefill modes
# =============================================================================


class MLP2D(LightweightModule):
    """
    MLP for TG (Galaxy) devices supporting both decode and prefill modes.

    Execution paths:
      Decode (dim < 8192):  linear → linear → all_reduce(×2) → mul+silu → linear → all_reduce
      Decode (dim >= 8192): linear → linear → reduce_scatter(×2) → mul+silu → all_gather → linear → all_reduce
      Prefill:              [reshape] → linear → linear → reduce_scatter(×2) → mul+silu → all_gather → linear → all_reduce → reshape
    """

    def __init__(self, config: MLP2DConfig):
        super().__init__()

        # Get hardware context from config
        mesh_ctx = config.mesh_ctx
        self.mesh_device = mesh_ctx.mesh_device
        self.tt_ccl = mesh_ctx.tt_ccl
        self.config = config
        self.ccl_topology = mesh_ctx.topology()

        # Model dimension (for dim-based branching)
        self.dim = config.dim

        # Get optimization settings
        opt = config.optimization_config
        self.activation_dtype = opt.activation_dtype()
        self.li_ff1_3_compute_kernel_cfg = opt.li_ff1_3_compute_kernel_cfg()
        self.li_ff2_compute_kernel_cfg = opt.li_ff2_compute_kernel_cfg()
        self.mul_dtype = self.activation_dtype or ttnn.bfloat8_b

        # Activation type
        self.activation_type = config.mlp_activation_type

        # Get decode configs (computed once at init)
        self.decode_pc_ff1_3 = config.decode_config.ff1_3_prg_config()
        self.decode_pc_ff2 = config.decode_config.ff2_prg_config()
        self.ff1_out_reduce_scatter_memcfg = config.decode_config.ff1_out_reduce_scatter_memcfg()
        self.ff1_out_gathered_memcfg = config.decode_config.ff1_out_gathered_memcfg()
        self.ff2_out_reduce_scatter_memcfg = config.decode_config.ff2_out_reduce_scatter_memcfg()
        self.sharded_attn_input_memcfg = config.decode_config.sharded_attn_input_memcfg()

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
        # MLP2D requires Galaxy topology (4x8 or 8x4) due to Galaxy-specific CCL operations
        valid_shapes = [(4, 8), (8, 4)]
        shape_tuple = tuple(args.cluster_shape)
        if shape_tuple not in valid_shapes:
            # IMPORTANT: do this validation before touching mesh_device/tt_ccl/model_config
            # so negative tests don't need to open a mesh device or initialize fabric.
            raise ValueError(
                f"MLP2D requires Galaxy topology (8x4). Got cluster_shape={args.cluster_shape}. "
                "For non-Galaxy devices, use MLP1D instead."
            )

        # Get model_config for sub-config closures
        model_config = args.get_model_config()
        decoders_opt = model_config.get("DECODERS_OPTIMIZATIONS")
        effective_layer_num = max(layer_num, 0)

        import torch

        from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

        # Create MLP2DConfig subclass with all overrides (captures args/model_config via closure)
        class _ArgsMLP2DConfig(MLP2DConfig):
            @cached_property
            def decode_config(self) -> MLP2DDecodeConfigs:
                class _Decode(MLP2DDecodeConfigs):
                    def ff1_3_prg_config(inner_self):
                        return model_config.get("FF1_3_TG_PROGCFG")

                    def ff2_prg_config(inner_self):
                        # TT-Transformers TG FF2 config assumes a specific intermediate sharding that
                        # doesn't match this MLP2D implementation for dim<8192. Let TTNN pick defaults.
                        if inner_self.cfg.dim < 8192:
                            return None
                        return model_config.get("FF2_TG_PROGCFG")

                    def ff1_out_reduce_scatter_memcfg(inner_self):
                        return model_config.get("FF1_OUT_REDUCE_SCATTER_MEMCFG")

                    def ff1_out_gathered_memcfg(inner_self):
                        # TT-Transformers config uses shard height 32*4 here; MLP2D tensors are height 32.
                        # Passing this into to_memory_config can TT_FATAL on shard-height mismatch.
                        return None

                    def ff2_out_reduce_scatter_memcfg(inner_self):
                        # Some TT-Transformers configs size this as shard_height=32*cluster_rows (e.g. 256 on 8x4),
                        # but MLP2D decode tensors here are height=32. Use the attention-input sharding instead.
                        if inner_self.cfg.dim < 8192:
                            return model_config.get("SHARDED_ATTN_INPUT_MEMCFG")
                        return model_config.get("FF2_OUT_REDUCE_SCATTER_MEMCFG")

                    def sharded_attn_input_memcfg(inner_self):
                        return model_config.get("SHARDED_ATTN_INPUT_MEMCFG")

                return _Decode(self)

            @cached_property
            def prefill_config(self) -> MLP2DPrefillConfigs:
                class _Prefill(MLP2DPrefillConfigs):
                    def w1_w3_prg_config(inner_self, seq_len: int):
                        return model_config.get("PREFILL_MLP_W1_W3_PRG_CONFIG")(seq_len)

                    def w2_prg_config(inner_self, seq_len: int):
                        return model_config.get("PREFILL_MLP_W2_PRG_CONFIG")(seq_len)

                return _Prefill(self)

            @cached_property
            def optimization_config(self) -> MLP2DOptimizationConfig:
                class _Opt(MLP2DOptimizationConfig):
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

        # Create MeshContext2D subclass with args overrides
        class _ArgsMeshContext2D(MeshContext2D):
            def dram_grid_size(inner_self) -> ttnn.CoreCoord:
                return args.dram_grid_size

            def topology(inner_self):
                return args.ccl_topology()

            def num_reduce_scatter_links(inner_self) -> int:
                return args.num_reduce_scatter_links

            def num_all_gather_links(inner_self) -> int:
                return args.num_all_gather_links

            def ccl_dtype(inner_self):
                return args.ccl_dtype

        mesh_ctx = _ArgsMeshContext2D(mesh_device=mesh_device, tt_ccl=tt_ccl)

        if state_dict_prefix is None:
            state_dict_prefix = args.get_state_dict_prefix("MLP", layer_num)

        hidden_dim_string = f".hidden_dim_{args.hidden_dim}" if args.hidden_dim != args.unpadded_hidden_dim else ""

        # Weight source factories (capture state_dict, prefix, hidden_dim via closure)
        # TG pads on the opposite dim compared to 1D
        def make_weight_source(name: str, pad_dim: int):
            def source():
                tensor = torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
                return pad_dim_to_size(tensor, dim=pad_dim, size=args.hidden_dim)

            return source

        cache_dir = None if args.dummy_weights else Path(weight_cache_path) / state_dict_prefix

        # Create LazyWeight instances with 2D mesh mapper
        def make_lazy_weight(name: str, pad_dim: int, shard_dims: tuple[int, int], dtype_fn, mem_cfg_fn) -> LazyWeight:
            return LazyWeight(
                source=make_weight_source(name, pad_dim),
                dtype=dtype_fn(),
                device=mesh_device,
                mesh_mapper_config=ttnn.MeshMapperConfig(
                    placements=list(shard_dims), mesh_shape_override=ttnn.MeshShape(cluster_shape)
                ),
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem_cfg_fn(),
                cache_dir=cache_dir,
                weight_name=f"{name}_sharded{hidden_dim_string}",
            )

        # We need config to exist first to get mem_config, so override lazy_* as cached_properties
        class _FinalConfig(_ArgsMLP2DConfig):
            @cached_property
            def lazy_w1(inner_self) -> LazyWeight:
                # w1: pad on -1 (hidden_dim), shard dims (-1, -2)
                return make_lazy_weight(
                    "w1",
                    -1,
                    inner_self.w1_shard_dims,
                    inner_self.optimization_config.ff1_3_dtype,
                    inner_self.w1_w3_mem_config,
                )

            @cached_property
            def lazy_w2(inner_self) -> LazyWeight:
                # w2: pad on -2 (hidden_dim on input side), shard dims (-2, -1)
                return make_lazy_weight(
                    "w2",
                    -2,
                    inner_self.w2_shard_dims,
                    inner_self.optimization_config.ff2_dtype,
                    inner_self.w2_mem_config,
                )

            @cached_property
            def lazy_w3(inner_self) -> LazyWeight:
                # w3: pad on -1 (hidden_dim), shard dims (-1, -2)
                return make_lazy_weight(
                    "w3",
                    -1,
                    inner_self.w1_shard_dims,
                    inner_self.optimization_config.ff1_3_dtype,
                    inner_self.w1_w3_mem_config,
                )

        config = _FinalConfig(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            mesh_ctx=mesh_ctx,
            max_batch_size=args.max_batch_size,
            mlp_activation_type=getattr(args, "mlp_activation_type", ttnn.UnaryOpType.SILU),
        )

        return cls(config)

    def _all_reduce_tg(
        self,
        input_tensor: ttnn.Tensor,
        cluster_axis: int,
        dim: int,
        sharded: bool,
        memory_config: Any,
        reduce_scatter_memory_config: Any = None,
        use_composite: bool = False,
    ) -> ttnn.Tensor:
        """
        All-reduce for TG (Galaxy) devices along specified cluster axis.
        """
        # Ensure dim 0 and 1 are 1
        original_shape = input_tensor.shape
        if original_shape[0] != 1 or original_shape[1] != 1:
            input_tensor = ttnn.reshape(
                input_tensor, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
            )

        # Cast to CCL dtype
        if input_tensor.dtype != self.config.ccl_dtype:
            input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG, self.config.ccl_dtype)
            if sharded and memory_config is not None:
                input_tensor = ttnn.to_memory_config(input_tensor, memory_config, self.config.ccl_dtype)

        if not sharded:
            input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

        input_mem_cfg = input_tensor.memory_config()
        # In composite all-reduce (RS + AG), the RS output memcfg can be different from the final desired memcfg.
        # If not provided, fall back to the input tensor's memory config (this guarantees shard height matches).
        rs_mem_cfg = ttnn.DRAM_MEMORY_CONFIG if not sharded else (reduce_scatter_memory_config or input_mem_cfg)

        reduced_tensor = ttnn.experimental.reduce_scatter_minimal_async(
            input_tensor,
            persistent_output_buffers=None,
            dim=dim,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            num_links=self.config.num_reduce_scatter_links,
            cluster_axis=cluster_axis,
            memory_config=rs_mem_cfg,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.ccl_topology,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        reduced_tensor = ttnn.experimental.all_gather_async(
            reduced_tensor,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=self.config.num_all_gather_links,
            cluster_axis=cluster_axis,
            topology=self.ccl_topology,
            memory_config=input_mem_cfg,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        reduced_tensor = ttnn.reshape(reduced_tensor, original_shape)
        # Preserve requested sharding on the final output (when provided).
        if sharded and memory_config is not None:
            reduced_tensor = ttnn.to_memory_config(reduced_tensor, memory_config)
        return reduced_tensor

    def _reduce_scatter_axis1(self, tensor: ttnn.Tensor, memory_config: Any) -> ttnn.Tensor:
        """Reduce scatter along cluster axis 1."""
        cluster_axis = 1
        return ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            num_links=self.config.num_reduce_scatter_links,
            cluster_axis=cluster_axis,
            memory_config=memory_config,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

    def _all_gather_axis1(self, tensor: ttnn.Tensor, memory_config: Any) -> ttnn.Tensor:
        """All gather along cluster axis 1."""
        cluster_axis = 1
        return ttnn.experimental.all_gather_async(
            tensor,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=2,
            cluster_axis=cluster_axis,
            topology=ttnn.Topology.Linear,
            memory_config=memory_config,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

    def decode_forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Decode forward for TG.

        Still has dim-based branching:
          - dim < 8192: all_reduce path
          - dim >= 8192: reduce_scatter + all_gather path
        """
        # --- STAGE 1: W1/W3 Linear (L1 sharded) ---
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat8_b,
            core_grid=None,
            compute_kernel_config=self.li_ff1_3_compute_kernel_cfg,
            program_config=self.decode_pc_ff1_3,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=ttnn.bfloat8_b,
            core_grid=None,
            compute_kernel_config=self.li_ff1_3_compute_kernel_cfg,
            program_config=self.decode_pc_ff1_3,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        # --- STAGE 2: CCL after W1/W3 (dim-dependent path) ---
        input_mem_cfg = w1_out.memory_config()

        w1_out = self._reduce_scatter_axis1(w1_out, self.ff1_out_reduce_scatter_memcfg)
        w3_out = self._reduce_scatter_axis1(w3_out, self.ff1_out_reduce_scatter_memcfg)

        # --- STAGE 3: Activation + Multiply ---
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[self.activation_type],
            dtype=self.mul_dtype,
            memory_config=w1_out.memory_config(),
        )

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        # --- STAGE 4: All-gather before W2 (if we used reduce_scatter) ---
        w2_in = self._all_gather_axis1(w2_in, input_mem_cfg)
        w2_in = ttnn.to_memory_config(w2_in, ttnn.L1_MEMORY_CONFIG)

        # --- STAGE 5: W2 Linear ---
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=self.li_ff2_compute_kernel_cfg,
            dtype=self.config.ccl_dtype,
            program_config=self.decode_pc_ff2,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            core_grid=None,
        )
        ttnn.deallocate(w2_in)

        # --- STAGE 6: Final All-Reduce ---
        w2_out_reduced = self._all_reduce_tg(
            w2_out,
            cluster_axis=0,
            # Use dim=3 so the composite reduce_scatter/all_gather path can legally shard and gather.
            dim=3,
            sharded=True,
            memory_config=self.ff2_out_reduce_scatter_memcfg,
            # Provide RS output memcfg explicitly (from `from_model_args` when applicable) so composite
            # all-reduce doesn't have to guess/fall back to input_mem_cfg.
            reduce_scatter_memory_config=self.ff2_out_reduce_scatter_memcfg,
            # Always use composite here; it produces correct replication semantics on the reduced axis.
            use_composite=True,
        )

        # --- STAGE 7: Reshape + Final memory config ---
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )
        # NOTE: For direct-API usage (e.g. unit tests) decode configs may leave this unset.
        if self.sharded_attn_input_memcfg is not None:
            w2_out_reduced = ttnn.to_memory_config(w2_out_reduced, self.sharded_attn_input_memcfg)

        return w2_out_reduced

    def prefill_forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Prefill forward for TG - uses reduce_scatter + all_gather path.

        Execution path:
          [reshape] → linear → linear → reduce_scatter(×2) → mul+silu → all_gather → linear → all_reduce → reshape
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
            dtype=ttnn.bfloat8_b,
            core_grid=None,
            compute_kernel_config=self.li_ff1_3_compute_kernel_cfg,
            program_config=pc_w1_w3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=ttnn.bfloat8_b,
            core_grid=None,
            compute_kernel_config=self.li_ff1_3_compute_kernel_cfg,
            program_config=pc_w1_w3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        # --- STAGE 2: CCL after W1/W3 (reduce_scatter for prefill) ---
        input_mem_cfg = w1_out.memory_config()

        w1_out = self._reduce_scatter_axis1(w1_out, None)  # None mem_config for prefill
        w3_out = self._reduce_scatter_axis1(w3_out, None)

        # --- STAGE 3: Activation + Multiply ---
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[self.activation_type],
            dtype=self.mul_dtype,
            memory_config=w1_out.memory_config(),
        )

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        # --- STAGE 4: All-gather before W2 ---
        w2_in = self._all_gather_axis1(w2_in, input_mem_cfg)
        # No L1 conversion for prefill

        # --- STAGE 5: W2 Linear ---
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=self.li_ff2_compute_kernel_cfg,
            dtype=self.config.ccl_dtype,
            program_config=pc_w2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=None,
        )
        ttnn.deallocate(w2_in)

        # --- STAGE 6: Final All-Reduce ---
        w2_out_reduced = self._all_reduce_tg(
            w2_out,
            cluster_axis=0,
            dim=3,  # Prefill always uses dim=3
            sharded=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # Use composite RS+AG for correctness; avoids dim==3 non-composite reshape/reduce pitfalls.
            use_composite=True,
        )

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
