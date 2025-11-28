# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2-style MLP modules for non-TG (non-Galaxy) devices: N150, N300, T3K.

Two specialized modules with NO if-else in their forward paths:
  - MLPNonTGDecode: For decode mode (seq_len <= 32)
  - MLPNonTGPrefill: For prefill mode (seq_len > 32)

Execution path:
  linear(w1) → linear(w3) → mul+activation → [reshard] → linear(w2) → all_reduce → [reshard]
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

# =============================================================================
# Utility functions
# =============================================================================


def pad_to_size(x: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    """Pads the specified dimension of the input tensor with zeros."""
    if dim < 0:
        dim = x.dim() + dim
    current_size = x.size(dim)
    pad_size = size - current_size

    if pad_size == 0:
        return x

    pad = [0] * (2 * x.dim())
    pad_index = 2 * (x.dim() - dim - 1)
    pad[pad_index + 1] = pad_size

    return torch.nn.functional.pad(x, pad, mode="constant", value=0)


def tt_all_reduce(
    input_tensor,
    mesh_device,
    tt_ccl,
    cluster_axis=0,
    dim=0,
    num_reduce_scatter_links=1,
    num_all_gather_links=2,
    topology=ttnn.Topology.Linear,
    memory_config=None,
    sharded=False,
    dtype=ttnn.bfloat16,
    use_composite=False,
):
    """All-reduce for non-TG devices (simplified - no TG branches)."""
    # N150: single device, no-op
    if list(mesh_device.shape) == [1, 1] or (cluster_axis == 1 and 1 in list(mesh_device.shape)):
        return input_tensor

    # Ensure dim 0 and 1 are 1
    original_shape = input_tensor.shape
    if original_shape[0] != 1 or original_shape[1] != 1:
        input_tensor = ttnn.reshape(
            input_tensor, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

    # N300 and T3K: reduce_scatter path
    if 1 in list(mesh_device.shape):
        if input_tensor.is_sharded() and not sharded:
            input_tensor_sharded = input_tensor
            input_tensor = ttnn.sharded_to_interleaved(input_tensor_sharded, ttnn.L1_MEMORY_CONFIG)
            input_tensor_sharded.deallocate(True)

        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            input_tensor,
            persistent_output_buffers=None,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=num_reduce_scatter_links,
            memory_config=memory_config,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topology,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        input_tensor.deallocate(True)
        return reduced

    # Fallback (shouldn't happen for non-TG)
    return input_tensor


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
# Tightened Config dataclasses (non-TG only)
# =============================================================================


@dataclass
class MLPNonTGConfig:
    """Configuration for non-TG MLP. Excludes all TG-specific fields."""

    # Model dimensions
    dim: int
    hidden_dim: int

    # Device topology
    cluster_shape: list  # [rows, cols], e.g., [1, 1] for N150, [1, 2] for N300, [1, 8] for T3K
    num_devices: int  # 1, 2, or 8

    # Prefill config
    prefill_len_cutoff: int  # 512 (BH) or 1024 (WH)

    # Weight loading
    dummy_weights: bool = False
    unpadded_hidden_dim: Optional[int] = None

    # CCL parameters
    ccl_dtype: Any = ttnn.bfloat8_b
    num_reduce_scatter_links: int = 1
    num_all_gather_links: int = 2

    # Activation type
    mlp_activation_type: Any = ttnn.UnaryOpType.SILU

    # Weight memory configs (required for non-TG)
    w1_w3_mem_config: Any = None
    w2_mem_config: Any = None

    def __post_init__(self):
        if self.unpadded_hidden_dim is None:
            self.unpadded_hidden_dim = self.hidden_dim


@dataclass
class MLPNonTGDecodeConfigs:
    """Program/memory configs for decode mode only."""

    # Program configs
    w1_w3_prg_config: Any = None
    w2_prg_config: Any = None

    # Memory configs
    sharded_mlp2_input_memcfg: Any = None  # reshard before w2
    decode_residual_memcfg: Any = None  # final output memory config


@dataclass
class MLPNonTGPrefillConfigs:
    """Program/memory configs for prefill mode only."""

    # Program configs (callables that take seq_len)
    w1_w3_prg_config: Callable[[int], Any] = None
    w2_prg_config: Callable[[int], Any] = None


@dataclass
class MLPNonTGOptimizationConfig:
    """Per-layer optimization settings."""

    # Weight dtypes
    ff1_3_dtype: Any = ttnn.bfloat8_b
    ff2_dtype: Any = ttnn.bfloat8_b

    # Activation dtype (None means use default)
    activation_dtype: Any = None

    # Compute kernel configs for math fidelity
    li_ff1_3_compute_kernel_cfg: Any = None
    li_ff2_compute_kernel_cfg: Any = None


# =============================================================================
# Base class with shared weight loading
# =============================================================================


class _MLPNonTGBase(LightweightModule):
    """Base class for non-TG MLP with shared weight loading logic."""

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        config: MLPNonTGConfig,
        optimization_config: MLPNonTGOptimizationConfig,
        state_dict,
        weight_cache_path,
        layer_num: int,
        state_dict_prefix: Optional[str] = None,
        ccl_topology: Callable[[int], Any] | Any = ccl_topology_non_tg,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.config = config
        self.layer_num = layer_num
        self.ccl_topology = ccl_topology(config.num_devices) if callable(ccl_topology) else ccl_topology

        # Pre-compute optimization settings
        self.activation_dtype = optimization_config.activation_dtype
        self.li_ff1_3_compute_kernel_cfg = optimization_config.li_ff1_3_compute_kernel_cfg
        self.li_ff2_compute_kernel_cfg = optimization_config.li_ff2_compute_kernel_cfg
        self.linear_dtype = self.activation_dtype or ttnn.bfloat16
        self.mul_dtype = self.activation_dtype or ttnn.bfloat8_b

        # Pre-compute all_reduce settings
        self.all_reduce_use_composite = config.dim == 8192

        # Activation type
        self.activation_type = config.mlp_activation_type

        # Weight loading
        if state_dict_prefix is None:
            state_dict_prefix = f"layers.{layer_num}.feed_forward"

        torch_weight = lambda name: torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
        pad_hidden_dim = lambda tensor, dim_idx: pad_to_size(tensor, dim=dim_idx, size=config.hidden_dim)

        hidden_dim_string = (
            f".hidden_dim_{config.hidden_dim}" if config.hidden_dim != config.unpadded_hidden_dim else ""
        )

        if config.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{state_dict_prefix}.{name}{hidden_dim_string}"

        def load_weight(name: str, w_dtype, dims: tuple) -> ttnn.Tensor:
            return ttnn.as_tensor(
                pad_hidden_dim(torch_weight(name[:2]), dims[-1]),
                dtype=w_dtype,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=config.cluster_shape),
                layout=ttnn.TILE_LAYOUT,
                memory_config=config.w2_mem_config if "w2" in name else config.w1_w3_mem_config,
                cache_file_name=cache_name(name),
            )

        self.w1 = load_weight("w1_sharded", optimization_config.ff1_3_dtype, dims=(-2, -1))
        self.w2 = load_weight("w2_sharded", optimization_config.ff2_dtype, dims=(-1, -2))
        self.w3 = load_weight("w3_sharded", optimization_config.ff1_3_dtype, dims=(-2, -1))


# =============================================================================
# MLPNonTGDecode - Decode mode with NO if-else in forward
# =============================================================================


class MLPNonTGDecode(_MLPNonTGBase):
    """
    MLP for non-TG devices in DECODE mode. NO if-else in forward().

    Execution path (fully flattened):
      linear(w1) → linear(w3) → mul+silu → reshard → linear(w2) → all_reduce(sharded) → reshard
    """

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        config: MLPNonTGConfig,
        decode_configs: MLPNonTGDecodeConfigs,
        optimization_config: MLPNonTGOptimizationConfig,
        state_dict,
        weight_cache_path,
        layer_num: int,
        state_dict_prefix: Optional[str] = None,
        ccl_topology: Callable[[int], Any] | Any = ccl_topology_non_tg,
    ):
        super().__init__(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            config=config,
            optimization_config=optimization_config,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            state_dict_prefix=state_dict_prefix,
            ccl_topology=ccl_topology,
        )

        # Decode-specific configs (all static, no runtime branching needed)
        self.pc_1 = decode_configs.w1_w3_prg_config
        self.pc_2 = decode_configs.w2_prg_config
        self.pc_3 = decode_configs.w1_w3_prg_config
        self.sharded_mlp2_input_memcfg = decode_configs.sharded_mlp2_input_memcfg
        self.decode_residual_memcfg = decode_configs.decode_residual_memcfg

    @classmethod
    def from_model_args(
        cls,
        mesh_device,
        tt_ccl,
        args,  # ModelArgs
        model_config: dict,
        state_dict,
        weight_cache_path,
        layer_num: int,
        dtype,
        state_dict_prefix: Optional[str] = None,
    ):
        """Factory method for backward compatibility with ModelArgs interface."""
        if args.is_galaxy:
            raise ValueError("MLPNonTGDecode cannot be used for Galaxy devices.")

        config = MLPNonTGConfig(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            cluster_shape=args.cluster_shape,
            num_devices=args.num_devices,
            prefill_len_cutoff=args.prefill_len_cutoff,
            dummy_weights=args.dummy_weights,
            unpadded_hidden_dim=args.unpadded_hidden_dim,
            ccl_dtype=args.ccl_dtype,
            num_reduce_scatter_links=args.num_reduce_scatter_links,
            num_all_gather_links=args.num_all_gather_links,
            mlp_activation_type=getattr(args, "mlp_activation_type", ttnn.UnaryOpType.SILU),
            w1_w3_mem_config=args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices),
            w2_mem_config=args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim),
        )

        decode_configs = MLPNonTGDecodeConfigs(
            w1_w3_prg_config=model_config.get("DECODE_MLP_W1_W3_PRG_CONFIG"),
            w2_prg_config=model_config.get("DECODE_MLP_W2_PRG_CONFIG"),
            sharded_mlp2_input_memcfg=model_config.get("SHARDED_MLP2_INPUT_MEMCFG"),
            decode_residual_memcfg=model_config.get("DECODE_RESIDUAL_MEMCFG"),
        )

        decoders_opt = model_config.get("DECODERS_OPTIMIZATIONS")
        effective_layer_num = max(layer_num, 0)

        from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

        optimization_config = MLPNonTGOptimizationConfig(
            ff1_3_dtype=decoders_opt.get_tensor_dtype(decoder_id=effective_layer_num, tensor=TensorGroup.FF1_FF3),
            ff2_dtype=decoders_opt.get_tensor_dtype(decoder_id=effective_layer_num, tensor=TensorGroup.FF2),
            activation_dtype=decoders_opt.get_tensor_dtype(
                decoder_id=effective_layer_num, tensor=TensorGroup.ACTIVATION
            ),
            li_ff1_3_compute_kernel_cfg=decoders_opt.get_math_fidelity(
                decoder_id=effective_layer_num, op=OpGroup.LI_FF1_FF3, configuration=args
            ),
            li_ff2_compute_kernel_cfg=decoders_opt.get_math_fidelity(
                decoder_id=effective_layer_num, op=OpGroup.LI_FF2, configuration=args
            ),
        )

        if state_dict_prefix is None:
            state_dict_prefix = args.get_state_dict_prefix("MLP", layer_num)

        return cls(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            config=config,
            decode_configs=decode_configs,
            optimization_config=optimization_config,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            state_dict_prefix=state_dict_prefix,
            ccl_topology=args.ccl_topology(),
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
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
            program_config=self.pc_1,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=self.linear_dtype,
            core_grid=None,
            compute_kernel_config=self.li_ff1_3_compute_kernel_cfg,
            program_config=self.pc_3,
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
            program_config=self.pc_2,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            core_grid=None,
        )
        ttnn.deallocate(w2_in)

        # --- STAGE 6: Final All-Reduce (sharded=True for decode) ---
        w2_out_reduced = tt_all_reduce(
            w2_out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            num_reduce_scatter_links=self.config.num_reduce_scatter_links,
            num_all_gather_links=self.config.num_all_gather_links,
            sharded=True,
            memory_config=w2_out.memory_config(),
            dtype=self.config.ccl_dtype,
            use_composite=self.all_reduce_use_composite,
            topology=self.ccl_topology,
        )

        # --- STAGE 7: Reshape + Final memory config ---
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )
        w2_out_reduced = ttnn.to_memory_config(w2_out_reduced, self.decode_residual_memcfg)

        return w2_out_reduced


# =============================================================================
# MLPNonTGPrefill - Prefill mode with minimal runtime logic
# =============================================================================


class MLPNonTGPrefill(_MLPNonTGBase):
    """
    MLP for non-TG devices in PREFILL mode.

    Execution path:
      [reshape] → linear(w1) → linear(w3) → mul+silu → linear(w2) → all_reduce → reshape

    Note: seq_len-dependent logic remains (program config callables, optional reshape).
    """

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        config: MLPNonTGConfig,
        prefill_configs: MLPNonTGPrefillConfigs,
        optimization_config: MLPNonTGOptimizationConfig,
        state_dict,
        weight_cache_path,
        layer_num: int,
        state_dict_prefix: Optional[str] = None,
        ccl_topology: Callable[[int], Any] | Any = ccl_topology_non_tg,
    ):
        super().__init__(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            config=config,
            optimization_config=optimization_config,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            state_dict_prefix=state_dict_prefix,
            ccl_topology=ccl_topology,
        )

        # Prefill-specific configs (callables for seq_len-dependent program configs)
        self._get_pc_w1_w3 = prefill_configs.w1_w3_prg_config
        self._get_pc_w2 = prefill_configs.w2_prg_config
        self.prefill_len_cutoff = config.prefill_len_cutoff

    @classmethod
    def from_model_args(
        cls,
        mesh_device,
        tt_ccl,
        args,  # ModelArgs
        model_config: dict,
        state_dict,
        weight_cache_path,
        layer_num: int,
        dtype,
        state_dict_prefix: Optional[str] = None,
    ):
        """Factory method for backward compatibility with ModelArgs interface."""
        if args.is_galaxy:
            raise ValueError("MLPNonTGPrefill cannot be used for Galaxy devices.")

        config = MLPNonTGConfig(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            cluster_shape=args.cluster_shape,
            num_devices=args.num_devices,
            prefill_len_cutoff=args.prefill_len_cutoff,
            dummy_weights=args.dummy_weights,
            unpadded_hidden_dim=args.unpadded_hidden_dim,
            ccl_dtype=args.ccl_dtype,
            num_reduce_scatter_links=args.num_reduce_scatter_links,
            num_all_gather_links=args.num_all_gather_links,
            mlp_activation_type=getattr(args, "mlp_activation_type", ttnn.UnaryOpType.SILU),
            w1_w3_mem_config=args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices),
            w2_mem_config=args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim),
        )

        prefill_configs = MLPNonTGPrefillConfigs(
            w1_w3_prg_config=model_config.get("PREFILL_MLP_W1_W3_PRG_CONFIG"),
            w2_prg_config=model_config.get("PREFILL_MLP_W2_PRG_CONFIG"),
        )

        decoders_opt = model_config.get("DECODERS_OPTIMIZATIONS")
        effective_layer_num = max(layer_num, 0)

        from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

        optimization_config = MLPNonTGOptimizationConfig(
            ff1_3_dtype=decoders_opt.get_tensor_dtype(decoder_id=effective_layer_num, tensor=TensorGroup.FF1_FF3),
            ff2_dtype=decoders_opt.get_tensor_dtype(decoder_id=effective_layer_num, tensor=TensorGroup.FF2),
            activation_dtype=decoders_opt.get_tensor_dtype(
                decoder_id=effective_layer_num, tensor=TensorGroup.ACTIVATION
            ),
            li_ff1_3_compute_kernel_cfg=decoders_opt.get_math_fidelity(
                decoder_id=effective_layer_num, op=OpGroup.LI_FF1_FF3, configuration=args
            ),
            li_ff2_compute_kernel_cfg=decoders_opt.get_math_fidelity(
                decoder_id=effective_layer_num, op=OpGroup.LI_FF2, configuration=args
            ),
        )

        if state_dict_prefix is None:
            state_dict_prefix = args.get_state_dict_prefix("MLP", layer_num)

        return cls(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            config=config,
            prefill_configs=prefill_configs,
            optimization_config=optimization_config,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            state_dict_prefix=state_dict_prefix,
            ccl_topology=args.ccl_topology(),
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Prefill forward - minimal runtime logic for seq_len-dependent configs.

        Execution path:
          [reshape if seq_len >= cutoff] → linear(w1) → linear(w3) → mul+silu → linear(w2) → all_reduce → reshape
        """
        seq_len = x.shape[-2]

        # Seq_len-dependent: reshape for long sequences
        if seq_len >= self.prefill_len_cutoff:
            x = ttnn.reshape(x, [1, seq_len // self.prefill_len_cutoff, self.prefill_len_cutoff, -1])

        # Seq_len-dependent: program configs
        pc_1 = self._get_pc_w1_w3(seq_len)
        pc_2 = self._get_pc_w2(seq_len)
        pc_3 = self._get_pc_w1_w3(seq_len)

        # --- STAGE 1: W1/W3 Linear (DRAM) ---
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=self.linear_dtype,
            core_grid=None,
            compute_kernel_config=self.li_ff1_3_compute_kernel_cfg,
            program_config=pc_1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=self.linear_dtype,
            core_grid=None,
            compute_kernel_config=self.li_ff1_3_compute_kernel_cfg,
            program_config=pc_3,
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
            program_config=pc_2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=None,
        )
        ttnn.deallocate(w2_in)

        # --- STAGE 6: Final All-Reduce (sharded=False for prefill) ---
        w2_out_reduced = tt_all_reduce(
            w2_out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            num_reduce_scatter_links=self.config.num_reduce_scatter_links,
            num_all_gather_links=self.config.num_all_gather_links,
            sharded=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.config.ccl_dtype,
            use_composite=self.all_reduce_use_composite,
            topology=self.ccl_topology,
        )

        # --- STAGE 7: Reshape (no final memory config change for prefill) ---
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

        return w2_out_reduced


# =============================================================================
# Convenience: MLPNonTG that dispatches to Decode or Prefill
# =============================================================================


class MLPNonTG(LightweightModule):
    """
    Convenience wrapper that holds both decode and prefill modules.

    Use this if you need to switch between modes at runtime.
    For best performance, use MLPNonTGDecode or MLPNonTGPrefill directly.
    """

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        config: MLPNonTGConfig,
        decode_configs: MLPNonTGDecodeConfigs,
        prefill_configs: MLPNonTGPrefillConfigs,
        optimization_config: MLPNonTGOptimizationConfig,
        state_dict,
        weight_cache_path,
        layer_num: int,
        state_dict_prefix: Optional[str] = None,
        ccl_topology: Callable[[int], Any] | Any = ccl_topology_non_tg,
    ):
        super().__init__()

        # Create both decode and prefill modules (they share weights via the same state_dict)
        self._decode = MLPNonTGDecode(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            config=config,
            decode_configs=decode_configs,
            optimization_config=optimization_config,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            state_dict_prefix=state_dict_prefix,
            ccl_topology=ccl_topology,
        )
        self._prefill = MLPNonTGPrefill(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            config=config,
            prefill_configs=prefill_configs,
            optimization_config=optimization_config,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            state_dict_prefix=state_dict_prefix,
            ccl_topology=ccl_topology,
        )

    @classmethod
    def from_model_args(
        cls,
        mesh_device,
        tt_ccl,
        args,
        model_config: dict,
        state_dict,
        weight_cache_path,
        layer_num: int,
        state_dict_prefix: Optional[str] = None,
    ):
        """Factory method for backward compatibility."""
        if args.is_galaxy:
            raise ValueError("MLPNonTG cannot be used for Galaxy devices.")

        config = MLPNonTGConfig(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            cluster_shape=args.cluster_shape,
            num_devices=args.num_devices,
            prefill_len_cutoff=args.prefill_len_cutoff,
            dummy_weights=args.dummy_weights,
            unpadded_hidden_dim=args.unpadded_hidden_dim,
            ccl_dtype=args.ccl_dtype,
            num_reduce_scatter_links=args.num_reduce_scatter_links,
            num_all_gather_links=args.num_all_gather_links,
            mlp_activation_type=getattr(args, "mlp_activation_type", ttnn.UnaryOpType.SILU),
            w1_w3_mem_config=args.create_dram_sharded_mem_config(args.dim, args.hidden_dim // args.num_devices),
            w2_mem_config=args.create_dram_sharded_mem_config(args.hidden_dim // args.num_devices, args.dim),
        )

        decode_configs = MLPNonTGDecodeConfigs(
            w1_w3_prg_config=model_config.get("DECODE_MLP_W1_W3_PRG_CONFIG"),
            w2_prg_config=model_config.get("DECODE_MLP_W2_PRG_CONFIG"),
            sharded_mlp2_input_memcfg=model_config.get("SHARDED_MLP2_INPUT_MEMCFG"),
            decode_residual_memcfg=model_config.get("DECODE_RESIDUAL_MEMCFG"),
        )

        prefill_configs = MLPNonTGPrefillConfigs(
            w1_w3_prg_config=model_config.get("PREFILL_MLP_W1_W3_PRG_CONFIG"),
            w2_prg_config=model_config.get("PREFILL_MLP_W2_PRG_CONFIG"),
        )

        decoders_opt = model_config.get("DECODERS_OPTIMIZATIONS")
        effective_layer_num = max(layer_num, 0)

        from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

        optimization_config = MLPNonTGOptimizationConfig(
            ff1_3_dtype=decoders_opt.get_tensor_dtype(decoder_id=effective_layer_num, tensor=TensorGroup.FF1_FF3),
            ff2_dtype=decoders_opt.get_tensor_dtype(decoder_id=effective_layer_num, tensor=TensorGroup.FF2),
            activation_dtype=decoders_opt.get_tensor_dtype(
                decoder_id=effective_layer_num, tensor=TensorGroup.ACTIVATION
            ),
            li_ff1_3_compute_kernel_cfg=decoders_opt.get_math_fidelity(
                decoder_id=effective_layer_num, op=OpGroup.LI_FF1_FF3, configuration=args
            ),
            li_ff2_compute_kernel_cfg=decoders_opt.get_math_fidelity(
                decoder_id=effective_layer_num, op=OpGroup.LI_FF2, configuration=args
            ),
        )

        if state_dict_prefix is None:
            state_dict_prefix = args.get_state_dict_prefix("MLP", layer_num)

        return cls(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            config=config,
            decode_configs=decode_configs,
            prefill_configs=prefill_configs,
            optimization_config=optimization_config,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            state_dict_prefix=state_dict_prefix,
            ccl_topology=args.ccl_topology(),
        )

    def forward(self, x: ttnn.Tensor, mode: str) -> ttnn.Tensor:
        """Dispatch to the appropriate specialized module."""
        if mode == "decode":
            return self._decode.forward(x)
        else:
            return self._prefill.forward(x)
