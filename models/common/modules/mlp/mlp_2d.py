# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2-style MLP module for TG (Galaxy) devices with 2D mesh topology.

Single unified MLP2D class with separate forward methods:
  - decode_forward(): For decode mode (seq_len <= 32)
  - prefill_forward(): For prefill mode (seq_len > 32)
  - forward(x, mode): Dispatcher that calls the appropriate method

Execution paths:
  - Unified: linear → linear → reduce_scatter(×2) → mul+silu → all_gather → linear → all_reduce

"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from models.common.modules.tt_ccl import TT_CCL, get_tt_ccl
from models.common.tensor_utils import pad_dim_to_size
from models.common.utility_functions import is_blackhole

# =============================================================================
# Top-level config dataclass
# =============================================================================


@dataclass
class MLP2DConfig:
    """
    Central configuration for MLP2D - the single source of truth for all settings.

    After __post_init__, all None fields are populated with derived defaults.

    Simple usage (all defaults):
        config = MLP2DConfig(w1, w2, w3)

    Override any field:
        config = MLP2DConfig(w1, w2, w3, max_batch_size=64)

    Full customization:
        config = MLP2DConfig(
            w1, w2, w3,
            mesh_device=custom_device,
            decode_w1_w3_prg_config=my_program_config,
            ...
        )
    """

    # Required: weights (LazyWeight)
    w1: LazyWeight
    w2: LazyWeight
    w3: LazyWeight

    # Optional: device and collectives
    mesh_device: ttnn.MeshDevice | None = None
    tt_ccl: TT_CCL | None = None
    topology: Optional[ttnn.Topology] = None  # None = auto-detect
    num_reduce_scatter_links: int = 1
    num_all_gather_links: int = 2

    # Optional: derived from weights if None
    dim: int | None = None
    hidden_dim: int | None = None

    # Optional: sensible defaults
    max_batch_size: int = 32
    mlp_activation_type: ttnn.UnaryOpType = ttnn.UnaryOpType.SILU

    # Optional: power-user overrides (None = compute defaults)
    w1_w3_memcfg: ttnn.MemoryConfig | None = None
    w2_memcfg: ttnn.MemoryConfig | None = None

    # Decode settings
    decode_input_memcfg: ttnn.MemoryConfig | None = None
    decode_w1_w3_prg_config: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig | None = None
    decode_w2_prg_config: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig | None = None
    ff1_out_reduce_scatter_memcfg: ttnn.MemoryConfig | None = None
    ff2_out_reduce_scatter_memcfg: ttnn.MemoryConfig | None = None
    sharded_attn_input_memcfg: ttnn.MemoryConfig | None = None

    # Prefill settings
    prefill_input_memcfg: ttnn.MemoryConfig | None = None
    prefill_w1_w3_prg_config: Callable[[int], ttnn.MatmulMultiCoreReuseMultiCastProgramConfig] | None = None
    prefill_w2_prg_config: Callable[[int], ttnn.MatmulMultiCoreReuseMultiCastProgramConfig] | None = None

    # Dtypes & Kernels
    w1_w3_dtype: ttnn.DataType | None = None
    w2_dtype: ttnn.DataType | None = None
    activation_dtype: ttnn.DataType | None = None
    ccl_dtype: ttnn.DataType | None = None
    mul_dtype: ttnn.DataType | None = None

    ff1_3_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    ff2_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None

    prefill_len_cutoff: int | None = None

    def is_resolved(self) -> bool:
        """Check if all fields except optional ones are resolved."""
        # These fields are optional overrides; they can stay None to let TTNN use defaults.
        optional = {
            "activation_dtype",
            "decode_w1_w3_prg_config",
            "decode_w2_prg_config",
            "ff1_out_reduce_scatter_memcfg",
            "ff2_out_reduce_scatter_memcfg",
            "sharded_attn_input_memcfg",
            "prefill_w1_w3_prg_config",
            "prefill_w2_prg_config",
        }
        # topology: None for single_device (CCL not needed)
        if self.mesh_device and self.mesh_device.get_num_devices() == 1:
            optional.add("topology")

        return all(getattr(self, f) is not None for f in self.__dataclass_fields__ if f not in optional)


# =============================================================================
# MLP2D - Unified MLP for 2D-topology devices (Galaxy)
# =============================================================================


class MLP2D(LightweightModule):
    """
    MLP for TG (Galaxy) devices supporting both decode and prefill modes.

    Execution paths:
      Unified: linear → linear → reduce_scatter(×2) → mul+silu → all_gather → linear → all_reduce
    """

    def __init__(self, w1: LazyWeight, w2: LazyWeight, w3: LazyWeight):
        """
        Simple API for 90% of users - derives all config from weights.

        Args:
            w1: Gate projection weight (dim, hidden_dim)
            w2: Down projection weight (hidden_dim, dim)
            w3: Up projection weight (dim, hidden_dim)
        """
        super().__init__()
        self.config = _resolve_mlp2d_config(MLP2DConfig(w1=w1, w2=w2, w3=w3))
        self._device_weights_loaded = False

    @classmethod
    def from_config(cls, config: MLP2DConfig):
        """
        Power API for 10% of users - any level of customization via config.
        """
        # bypass the __init__ method of the base class for power users who want to customize the config
        instance = object.__new__(cls)
        super(MLP2D, instance).__init__()
        instance.config = _resolve_mlp2d_config(config)
        instance._device_weights_loaded = False
        return instance

    def load_device_weights(self):
        if self._device_weights_loaded:
            return

        assert self.config.is_resolved(), "config must be resolved before loading device weights!"

        self.w1 = self.config.w1.get_device_weight()
        self.w2 = self.config.w2.get_device_weight()
        self.w3 = self.config.w3.get_device_weight()

        self._device_weights_loaded = True

    def _all_reduce_tg(
        self,
        input_tensor: ttnn.Tensor,
        cluster_axis: int,
        dim: int,
        sharded: bool,
        memory_config: Any,
        reduce_scatter_memory_config: Any = None,
    ) -> ttnn.Tensor:
        """
        All-reduce for TG (Galaxy) devices along specified cluster axis.
        """
        cfg = self.config
        # Ensure dim 0 and 1 are 1
        original_shape = input_tensor.shape
        if original_shape[0] != 1 or original_shape[1] != 1:
            input_tensor = ttnn.reshape(
                input_tensor, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
            )

        # Cast to CCL dtype
        if input_tensor.dtype != cfg.ccl_dtype:
            input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG, cfg.ccl_dtype)
            if sharded and memory_config is not None:
                input_tensor = ttnn.to_memory_config(input_tensor, memory_config, cfg.ccl_dtype)

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
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            num_links=cfg.num_reduce_scatter_links,
            cluster_axis=cluster_axis,
            memory_config=rs_mem_cfg,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=cfg.topology,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        reduced_tensor = ttnn.experimental.all_gather_async(
            reduced_tensor,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=cfg.num_all_gather_links,
            cluster_axis=cluster_axis,
            topology=cfg.topology,
            memory_config=input_mem_cfg,
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
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
        cfg = self.config
        cluster_axis = 1
        return ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            num_links=cfg.num_reduce_scatter_links,
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
        cfg = self.config
        cluster_axis = 1
        return ttnn.experimental.all_gather_async(
            tensor,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=2,
            cluster_axis=cluster_axis,
            topology=ttnn.Topology.Linear,
            memory_config=memory_config,
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

    def decode_forward(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """
        Decode forward for TG.

        Unified Path: linear → linear → reduce_scatter(×2) → mul+silu → all_gather → linear → all_reduce
        """
        self.load_device_weights()
        x = _load_input_device_tensor(x, self.config, mode="decode")
        cfg = self.config

        # --- STAGE 1: W1/W3 Linear (L1 sharded) ---
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat8_b,
            core_grid=None,
            compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
            program_config=cfg.decode_w1_w3_prg_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=ttnn.bfloat8_b,
            core_grid=None,
            compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
            program_config=cfg.decode_w1_w3_prg_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        # --- STAGE 2: CCL after W1/W3 (reduce_scatter) ---
        input_mem_cfg = w1_out.memory_config()

        w1_out = self._reduce_scatter_axis1(w1_out, cfg.ff1_out_reduce_scatter_memcfg)
        w3_out = self._reduce_scatter_axis1(w3_out, cfg.ff1_out_reduce_scatter_memcfg)

        # --- STAGE 3: Activation + Multiply ---
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[cfg.mlp_activation_type],
            dtype=cfg.mul_dtype,
            memory_config=w1_out.memory_config(),
        )

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        # --- STAGE 4: All-gather before W2 ---
        w2_in = self._all_gather_axis1(w2_in, input_mem_cfg)
        w2_in = ttnn.to_memory_config(w2_in, ttnn.L1_MEMORY_CONFIG)

        # --- STAGE 5: W2 Linear ---
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=cfg.ff2_compute_kernel_cfg,
            dtype=cfg.ccl_dtype,
            program_config=cfg.decode_w2_prg_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            core_grid=None,
        )
        ttnn.deallocate(w2_in)

        # --- STAGE 6: Final All-Reduce ---
        w2_out_reduced = self._all_reduce_tg(
            w2_out,
            cluster_axis=0,
            dim=3,
            sharded=True,
            memory_config=cfg.ff2_out_reduce_scatter_memcfg,
            reduce_scatter_memory_config=cfg.ff2_out_reduce_scatter_memcfg,
        )

        # --- STAGE 7: Reshape + Final memory config ---
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )
        # NOTE: For direct-API usage (e.g. unit tests) decode configs may leave this unset.
        if cfg.sharded_attn_input_memcfg is not None:
            w2_out_reduced = ttnn.to_memory_config(w2_out_reduced, cfg.sharded_attn_input_memcfg)

        return w2_out_reduced

    def prefill_forward(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """
        Prefill forward for TG.

        Unified Path: [reshape] → linear → linear → reduce_scatter(×2) → mul+silu → all_gather → linear → all_reduce → reshape
        """
        self.load_device_weights()
        x = _load_input_device_tensor(x, self.config, mode="prefill")
        cfg = self.config
        seq_len = x.shape[-2]

        # Seq_len-dependent: reshape for long sequences
        if seq_len >= cfg.prefill_len_cutoff:
            assert (
                seq_len % cfg.prefill_len_cutoff == 0
            ), f"seq_len ({seq_len}) must be divisible by prefill_len_cutoff ({cfg.prefill_len_cutoff})"
            x = ttnn.reshape(x, [1, seq_len // cfg.prefill_len_cutoff, cfg.prefill_len_cutoff, -1])

        # Seq_len-dependent: get program configs (None = let TTNN pick defaults)
        pc_w1_w3 = cfg.prefill_w1_w3_prg_config(seq_len) if cfg.prefill_w1_w3_prg_config else None
        pc_w2 = cfg.prefill_w2_prg_config(seq_len) if cfg.prefill_w2_prg_config else None

        # --- STAGE 1: W1/W3 Linear (DRAM) ---
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=ttnn.bfloat8_b,
            core_grid=None,
            compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
            program_config=pc_w1_w3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=ttnn.bfloat8_b,
            core_grid=None,
            compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
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
            input_tensor_a_activations=[cfg.mlp_activation_type],
            dtype=cfg.mul_dtype,
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
            compute_kernel_config=cfg.ff2_compute_kernel_cfg,
            dtype=cfg.ccl_dtype,
            program_config=pc_w2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=None,
        )
        ttnn.deallocate(w2_in)

        # --- STAGE 6: Final All-Reduce ---
        w2_out_reduced = self._all_reduce_tg(
            w2_out,
            cluster_axis=0,
            dim=3,
            sharded=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # --- STAGE 7: Reshape (no final memory config change for prefill) ---
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

        return w2_out_reduced

    def forward(self, x: ttnn.Tensor | LazyWeight, mode: str) -> ttnn.Tensor:
        """Dispatch to the appropriate forward method based on mode."""
        if mode == "decode":
            return self.decode_forward(x)
        else:
            return self.prefill_forward(x)

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

        import torch

        from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

        # Get model_config for overrides
        model_config = args.get_model_config()
        decoders_opt = model_config.get("DECODERS_OPTIMIZATIONS")
        effective_layer_num = max(layer_num, 0)

        # Extract settings
        ccl_topology = args.ccl_topology()

        if state_dict_prefix is None:
            state_dict_prefix = args.get_state_dict_prefix("MLP", layer_num)

        # Get dtypes
        ff1_3_dtype = decoders_opt.get_tensor_dtype(decoder_id=effective_layer_num, tensor=TensorGroup.FF1_FF3)
        ff2_dtype = decoders_opt.get_tensor_dtype(decoder_id=effective_layer_num, tensor=TensorGroup.FF2)
        activation_dtype = decoders_opt.get_tensor_dtype(decoder_id=effective_layer_num, tensor=TensorGroup.ACTIVATION)

        # Get compute kernel configs
        ff1_3_compute_kernel_cfg = decoders_opt.get_math_fidelity(
            decoder_id=effective_layer_num, op=OpGroup.LI_FF1_FF3, configuration=args
        )
        ff2_compute_kernel_cfg = decoders_opt.get_math_fidelity(
            decoder_id=effective_layer_num, op=OpGroup.LI_FF2, configuration=args
        )

        # Get decode program configs from model_config
        # Note: Handling legacy small dim behavior by setting configs to None if implicit check failed
        decode_w1_w3_prg_config = model_config.get("FF1_3_TG_PROGCFG")
        if args.dim < 8192:
            # TT-Transformers TG FF2 config assumes a specific intermediate sharding that
            # doesn't match this MLP2D implementation for dim<8192. Let TTNN pick defaults.
            decode_w1_w3_prg_config = None

        decode_w2_prg_config = model_config.get("FF2_TG_PROGCFG")
        if args.dim < 8192:
            decode_w2_prg_config = None

        # Memory configs
        ff1_out_reduce_scatter_memcfg = model_config.get("FF1_OUT_REDUCE_SCATTER_MEMCFG")

        # TT-Transformers config uses shard height 32*4 here; MLP2D tensors are height 32.
        # Passing this into to_memory_config can TT_FATAL on shard-height mismatch.
        ff2_out_reduce_scatter_memcfg = model_config.get("FF2_OUT_REDUCE_SCATTER_MEMCFG")
        if args.dim < 8192:
            # Some TT-Transformers configs size this as shard_height=32*cluster_rows (e.g. 256 on 8x4),
            # but MLP2D decode tensors here are height=32. Use the attention-input sharding instead.
            ff2_out_reduce_scatter_memcfg = model_config.get("SHARDED_ATTN_INPUT_MEMCFG")

        sharded_attn_input_memcfg = model_config.get("SHARDED_ATTN_INPUT_MEMCFG")

        # Prefill configs
        prefill_w1_w3_prg_config_factory = model_config.get("PREFILL_MLP_W1_W3_PRG_CONFIG")
        prefill_w2_prg_config_factory = model_config.get("PREFILL_MLP_W2_PRG_CONFIG")

        cache_dir = None if args.dummy_weights else Path(weight_cache_path) / state_dict_prefix
        hidden_dim_string = f".hidden_dim_{args.hidden_dim}" if args.hidden_dim != args.unpadded_hidden_dim else ""

        def make_weight_source(name: str, pad_dim: int):
            tensor = torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
            return pad_dim_to_size(tensor, dim=pad_dim, size=args.hidden_dim)

        # 2D sharding dims for weights
        w1_shard_dims = (-1, -2)
        w2_shard_dims = (-2, -1)

        w1 = LazyWeight(
            source=make_weight_source("w1", -1),
            dtype=ff1_3_dtype,
            device=mesh_device,
            mesh_mapper_config=ttnn.MeshMapperConfig(
                placements=[ttnn.PlacementShard(w1_shard_dims[0]), ttnn.PlacementShard(w1_shard_dims[1])],
                mesh_shape_override=ttnn.MeshShape(args.cluster_shape),
            ),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,  # TG uses DRAM for weights
            cache_dir_weight_name=(cache_dir, f"w1_sharded{hidden_dim_string}") if cache_dir else None,
        )
        w2 = LazyWeight(
            source=make_weight_source("w2", -2),
            dtype=ff2_dtype,
            device=mesh_device,
            mesh_mapper_config=ttnn.MeshMapperConfig(
                placements=[ttnn.PlacementShard(w2_shard_dims[0]), ttnn.PlacementShard(w2_shard_dims[1])],
                mesh_shape_override=ttnn.MeshShape(args.cluster_shape),
            ),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_dir_weight_name=(cache_dir, f"w2_sharded{hidden_dim_string}") if cache_dir else None,
        )
        w3 = LazyWeight(
            source=make_weight_source("w3", -1),
            dtype=ff1_3_dtype,
            device=mesh_device,
            mesh_mapper_config=ttnn.MeshMapperConfig(
                placements=[ttnn.PlacementShard(w1_shard_dims[0]), ttnn.PlacementShard(w1_shard_dims[1])],
                mesh_shape_override=ttnn.MeshShape(args.cluster_shape),
            ),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_dir_weight_name=(cache_dir, f"w3_sharded{hidden_dim_string}") if cache_dir else None,
        )

        config = MLP2DConfig(
            w1=w1,
            w2=w2,
            w3=w3,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            max_batch_size=args.max_batch_size,
            mlp_activation_type=getattr(args, "mlp_activation_type", ttnn.UnaryOpType.SILU),
            topology=ccl_topology,
            num_reduce_scatter_links=args.num_reduce_scatter_links,
            num_all_gather_links=args.num_all_gather_links,
            decode_w1_w3_prg_config=decode_w1_w3_prg_config,
            decode_w2_prg_config=decode_w2_prg_config,
            ff1_out_reduce_scatter_memcfg=ff1_out_reduce_scatter_memcfg,
            ff2_out_reduce_scatter_memcfg=ff2_out_reduce_scatter_memcfg,
            sharded_attn_input_memcfg=sharded_attn_input_memcfg,
            prefill_w1_w3_prg_config=prefill_w1_w3_prg_config_factory,
            prefill_w2_prg_config=prefill_w2_prg_config_factory,
            w1_w3_dtype=ff1_3_dtype,
            w2_dtype=ff2_dtype,
            activation_dtype=activation_dtype,
            ccl_dtype=args.ccl_dtype,
            ff1_3_compute_kernel_cfg=ff1_3_compute_kernel_cfg,
            ff2_compute_kernel_cfg=ff2_compute_kernel_cfg,
        )

        return cls.from_config(config)


# =============================================================================
# Helper functions
# =============================================================================


# todo)) work with the CCL team to find opportunity to simplify this --> e.g., build into TTNN APIs?
def _default_topology(mesh_device: ttnn.MeshDevice) -> Optional[ttnn.Topology]:
    """Auto-detect CCL topology based on cluster type and device count."""
    num_devices = mesh_device.get_num_devices()
    if num_devices == 8 and ttnn.cluster.get_cluster_type() in [
        ttnn.cluster.ClusterType.T3K,
        ttnn.cluster.ClusterType.GALAXY,
    ]:
        # NOTE: we always want to do ring if it is available
        return ttnn.Topology.Ring
    elif num_devices > 1:
        # NOTE: this should be a fallback when the ring is not available
        return ttnn.Topology.Linear
    return None


def _compute_kernel_config_hifi2_fp16() -> ttnn.WormholeComputeKernelConfig:
    """Default compute kernel config for MLP (HiFi2 with FP16 accumulation)."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _resolve_mlp2d_config(config: MLP2DConfig) -> MLP2DConfig:
    """Materialize the config to known good defaults using replace pattern."""
    to_set = {}

    # --- Phase 1: Foundational fields ---

    # Derive dimensions
    dim = config.dim
    if config.dim is None:
        dim = config.w1.source.shape[-2]
        to_set["dim"] = dim

    hidden_dim = config.hidden_dim
    if config.hidden_dim is None:
        hidden_dim = config.w1.source.shape[-1]
        to_set["hidden_dim"] = hidden_dim

    # Derive mesh_device
    mesh_device = config.mesh_device
    if mesh_device is None:
        mesh_device = config.w1.device
    if mesh_device is None:
        mesh_device = ttnn.GetDefaultDevice()
    if config.mesh_device is None:
        to_set["mesh_device"] = mesh_device

    assert mesh_device is not None
    cluster_shape = list(mesh_device.shape)
    # MLP2D is designed for 2D mesh topologies (cluster_shape[0] > 1 and cluster_shape[1] > 1)
    # Note: from_model_args() enforces Galaxy (4x8 or 8x4) because it uses model_config.py
    # which has Galaxy-specific hardcoded values. Direct MLP2DConfig usage is more flexible.
    assert cluster_shape[0] > 1 and cluster_shape[1] > 1, (
        f"MLP2D requires 2D mesh (both cluster_shape dimensions > 1). "
        f"Got cluster_shape={cluster_shape}. For 1D meshes, use MLP1D instead."
    )

    # Derive tt_ccl
    tt_ccl = config.tt_ccl
    if config.tt_ccl is None:
        tt_ccl = get_tt_ccl(mesh_device)
        to_set["tt_ccl"] = tt_ccl

    # Auto-detect topology
    topology = config.topology
    if config.topology is None:
        topology = _default_topology(mesh_device)
        to_set["topology"] = topology

    # --- Phase 2: Dtypes and Tunings ---

    w1_w3_dtype = config.w1_w3_dtype or ttnn.bfloat8_b
    to_set["w1_w3_dtype"] = w1_w3_dtype
    w2_dtype = config.w2_dtype or ttnn.bfloat8_b
    to_set["w2_dtype"] = w2_dtype

    if config.ccl_dtype is None:
        to_set["ccl_dtype"] = ttnn.bfloat8_b
    if config.mul_dtype is None:
        to_set["mul_dtype"] = config.activation_dtype or ttnn.bfloat8_b

    if config.prefill_len_cutoff is None:
        to_set["prefill_len_cutoff"] = 512 if is_blackhole() else 1024

    # Compute kernel configs
    if config.ff1_3_compute_kernel_cfg is None:
        to_set["ff1_3_compute_kernel_cfg"] = _compute_kernel_config_hifi2_fp16()
    if config.ff2_compute_kernel_cfg is None:
        to_set["ff2_compute_kernel_cfg"] = _compute_kernel_config_hifi2_fp16()

    # --- Phase 2.5: Input Memory Configs ---

    if config.decode_input_memcfg is None:
        to_set["decode_input_memcfg"] = ttnn.L1_MEMORY_CONFIG

    if config.prefill_input_memcfg is None:
        to_set["prefill_input_memcfg"] = ttnn.DRAM_MEMORY_CONFIG

    # --- Phase 3: Prefill Program Configs ---
    # NOTE: prefill_w1_w3_prg_config and prefill_w2_prg_config are optional.
    # When None, TTNN picks defaults. Only from_model_args (Power API) provides these.
    # This keeps the Simple API working without complex 2D-aware program config generation.

    # --- Phase 4: Resolve Weights (always 2D sharded for MLP2D) ---

    # TG weights use DRAM interleaved (no specific shard memory config on weights themselves)
    w1_w3_memcfg = config.w1_w3_memcfg or ttnn.DRAM_MEMORY_CONFIG
    to_set["w1_w3_memcfg"] = w1_w3_memcfg
    w2_memcfg = config.w2_memcfg or ttnn.DRAM_MEMORY_CONFIG
    to_set["w2_memcfg"] = w2_memcfg

    # MLP2D ALWAYS uses 2D sharding - this is fundamental to how 2D mesh MLP works.
    # w1/w3: shard dims (-1, -2) = N sharded on mesh axis 0, K sharded on mesh axis 1
    # w2: shard dims (-2, -1) = K sharded on mesh axis 0, N sharded on mesh axis 1
    w1_w3_shard_dims = (-1, -2)
    w2_shard_dims = (-2, -1)

    def get_weight_mesh_mapper(lazy_weight: LazyWeight, shard_dims: tuple[int, int]):
        """Return existing mesh_mapper_config if set, else create 2D shard mapper."""
        existing = getattr(lazy_weight, "mesh_mapper_config", None)
        if existing is not None:
            return existing
        # Default: apply 2D sharding
        return ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(shard_dims[0]), ttnn.PlacementShard(shard_dims[1])],
            mesh_shape_override=ttnn.MeshShape(cluster_shape),
        )

    to_set["w1"] = resolve_lazy_weight(
        config.w1,
        device=mesh_device,
        memory_config=w1_w3_memcfg,
        mesh_mapper_config=get_weight_mesh_mapper(config.w1, w1_w3_shard_dims),
        layout=ttnn.TILE_LAYOUT,
        dtype=w1_w3_dtype,
    )

    to_set["w2"] = resolve_lazy_weight(
        config.w2,
        device=mesh_device,
        memory_config=w2_memcfg,
        mesh_mapper_config=get_weight_mesh_mapper(config.w2, w2_shard_dims),
        layout=ttnn.TILE_LAYOUT,
        dtype=w2_dtype,
    )

    to_set["w3"] = resolve_lazy_weight(
        config.w3,
        device=mesh_device,
        memory_config=w1_w3_memcfg,
        mesh_mapper_config=get_weight_mesh_mapper(config.w3, w1_w3_shard_dims),
        layout=ttnn.TILE_LAYOUT,
        dtype=w1_w3_dtype,
    )

    resolved_config = replace(config, **to_set)
    assert resolved_config.is_resolved(), "Config must be resolved!"
    return resolved_config


def _load_input_device_tensor(x: ttnn.Tensor | LazyWeight, config: MLP2DConfig, mode: str) -> ttnn.Tensor:
    """Resolve the input tensor to ttnn tensor if x is a LazyWeight, otherwise return as is."""
    assert mode in ["decode", "prefill"], "mode must be one of decode or prefill!"
    mem_cfg = config.decode_input_memcfg if mode == "decode" else config.prefill_input_memcfg
    if isinstance(x, LazyWeight):
        # For MLP2D, input must be sharded to match weight sharding:
        # - w1/w3 shard dims = (-1, -2): K sharded on mesh axis 1
        # - Input [batch, 1, seq, K]: shard K (dim -1) on mesh axis 1, replicate on axis 0
        cluster_shape = list(config.mesh_device.shape)
        input_mesh_mapper = ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)],
            mesh_shape_override=ttnn.MeshShape(cluster_shape),
        )
        resolved_x = resolve_lazy_weight(
            x,
            device=config.mesh_device,
            memory_config=mem_cfg,
            mesh_mapper_config=input_mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        return resolved_x.get_device_weight()

    assert isinstance(x, ttnn.Tensor), "x must be a ttnn tensor at this point!"
    return x
