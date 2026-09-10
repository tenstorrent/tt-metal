# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2-style MLP module for 1D-topology devices: N150 (1x1), N300 (1x2), T3K (1x8).

Single unified MLP1D class with separate forward methods:
  - decode_forward(): For decode mode
  - prefill_forward(): For prefill mode
  - forward(x, mode): Dispatcher that calls the appropriate method

Execution paths:
  Decode:  linear(w1) → linear(w3) → mul+silu → reshard → linear(w2) → all_reduce(sharded) → reshard
  Prefill: [reshape] → linear(w1) → linear(w3) → mul+silu → linear(w2) → all_reduce → reshape

"""

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from models.common.modules.tt_ccl import (
    CCL_CHUNKS_PER_SYNC,
    CCL_NUM_BUFFERS_PER_CHANNEL,
    CCL_NUM_WORKERS_PER_LINK,
    TT_CCL,
    default_topology,
    get_tt_ccl,
)
from models.common.tensor_utils import TILE_SIZE, get_out_subblock_w, get_padded_hidden_dim, pad_dim_to_size
from models.tt_transformers.tt.common import Mode

# =============================================================================
# Top-level config dataclass
# =============================================================================


@dataclass
class MLP1DConfig:
    """
    Central configuration for MLP1D - the single source of truth for all settings.

    Simple usage (all defaults):
        config = MLP1DConfig(w1, w2, w3)

    Override any field:
        config = MLP1DConfig(w1, w2, w3, max_batch_size=64, topology=ttnn.Topology.Ring)

    Full customization:
        config = MLP1DConfig(
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
    decode_rs_memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG
    decode_rs_chunks_per_sync: int = 1
    decode_rs_num_workers_per_link: int = 1

    # Optional: derived from weights if None
    dim: int | None = None
    hidden_dim: int | None = None

    # Optional: sensible defaults
    max_batch_size: int = 32
    mlp_activation_type: ttnn.UnaryOpType = ttnn.UnaryOpType.SILU

    # Optional: power-user overrides (None = compute defaults)
    w1_w3_memcfg: ttnn.MemoryConfig | None = None
    w2_memcfg: ttnn.MemoryConfig | None = None

    decode_input_memcfg: ttnn.MemoryConfig | None = None
    decode_w1_w3_prg_config: ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig | None = None
    decode_w2_prg_config: ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig | None = None
    decode_mlp2_input_memcfg: ttnn.MemoryConfig | None = None
    decode_residual_memcfg: ttnn.MemoryConfig | None = None

    prefill_input_memcfg: ttnn.MemoryConfig | None = None
    prefill_w1_w3_prg_config: Callable[[int], ttnn.MatmulMultiCoreReuseMultiCastProgramConfig] | None = None
    prefill_w2_prg_config: Callable[[int], ttnn.MatmulMultiCoreReuseMultiCastProgramConfig] | None = None

    # Optional: use ttnn.experimental.minimal_matmul (instead of ttnn.linear) for the W2 down-proj
    # prefill matmul above seq_len > 128 — matches TTTv1 (mlp.py L275-281), ~2.5x faster on the large
    # folded-batch down-proj. Default OFF so untouched models / decode stay byte-identical; the caller
    # opts in. FF1/FF3 stay on ttnn.linear (TTTv1 does too). The factory yields a ttnn.MinimalMatmulConfig
    # keyed on the folded seq_len (sibling to prefill_w2_prg_config).
    prefill_w2_minimal_matmul: bool = False
    prefill_w2_minimal_matmul_config: Callable[[int], "ttnn.MinimalMatmulConfig"] | None = None

    w1_w3_dtype: ttnn.DataType | None = None
    w2_dtype: ttnn.DataType | None = None
    activation_dtype: ttnn.DataType | None = None
    # If True, move W1 decode output to DRAM before W3 linear so W3 CB validation does not overlap W1 L1.
    decode_spill_w1_to_dram_before_w3: bool = False

    linear_dtype: ttnn.DataType | None = None
    mul_dtype: ttnn.DataType | None = None

    # Architecture-sensitive requests. ``None`` selects the concrete mesh/SKU
    # default during construction; explicit compatible values are copied and
    # preserved in the internal resolved state.
    ff1_3_compute_kernel_cfg: ttnn.DeviceComputeKernelConfig | None = None
    ff2_compute_kernel_cfg: ttnn.DeviceComputeKernelConfig | None = None
    decode_ff1_3_compute_kernel_cfg: ttnn.DeviceComputeKernelConfig | None = None
    decode_ff2_compute_kernel_cfg: ttnn.DeviceComputeKernelConfig | None = None
    prefill_len_cutoff: int | None = None
    prefill_dram_shard_grid_width: int | None = None
    prefill_ff1_ff3_grid: tuple[int, int] | None = None
    prefill_ff2_grid: tuple[int, int] | None = None

    def use_minimal_w2_matmul(self, seq_len: int) -> bool:
        """Whether the W2 down-proj prefill matmul should use minimal_matmul.

        Mirrors TTTv1 (mlp.py:275) — minimal_matmul only above ``seq_len > 128`` and only when the
        per-model opt-in is set. ``seq_len`` is the folded ``B*S`` length.
        """
        return bool(self.prefill_w2_minimal_matmul) and seq_len > 128

    def is_resolved(self) -> bool:
        """Check if all fields except optional ones are resolved."""
        # activation_dtype is optional override for linear_dtype and mul_dtype.
        # prefill_w2_minimal_matmul_config is only materialized when the minimal-matmul opt-in is set.
        optional = {
            "activation_dtype",
            "prefill_w2_minimal_matmul_config",
        }
        # topology: None for single_device (CCL not needed)
        if self.mesh_device and self.mesh_device.get_num_devices() == 1:
            optional.add("topology")

        return all(getattr(self, f) is not None for f in self.__dataclass_fields__ if f not in optional)


# =============================================================================
# MLP1D - Unified MLP for 1D-topology devices (Linear or Ring) with decode and prefill modes
# =============================================================================


class MLP1D(LightweightModule):
    """
    MLP for non-TG devices supporting both decode and prefill modes.

    Simple API (90% of users):
        mlp = MLP1D(w1, w2, w3)

    Power API (10% of users) - any level of customization via config:
        config = MLP1DConfig(w1, w2, w3, max_batch_size=64, topology=ttnn.Topology.Ring)
        mlp = MLP1D.from_config(config)

    Execution paths:
      Decode:  linear(w1) → linear(w3) → mul+silu → reshard → linear(w2) → all_reduce(sharded) → reshard
      Prefill: [reshape] → linear(w1) → linear(w3) → mul+silu → linear(w2) → all_reduce → reshape
    """

    def __init__(self, w1: LazyWeight, w2: LazyWeight, w3: LazyWeight):
        """
        Simple API for 90% of users - derives all config from weights.

        Args:
            w1: Gate projection weight (dim, hidden_dim), sharded on dim=-1
            w2: Down projection weight (hidden_dim, dim), sharded on dim=-2
            w3: Up projection weight (dim, hidden_dim), sharded on dim=-1

        The mesh_device is derived from w1.device(). tt_ccl is created/cached automatically.
        All other settings use sensible defaults.
        """
        super().__init__()
        self.config = resolve_mlp1d_arch_config(MLP1DConfig(w1=w1, w2=w2, w3=w3))
        self._device_weights_loaded = False

    @classmethod
    def from_config(cls, config: MLP1DConfig):
        """
        Power API for 10% of users - any level of customization via config.

        Override any subset of fields in MLP1DConfig:
            config = MLP1DConfig(w1, w2, w3, max_batch_size=64)
            mlp = MLP1D.from_config(config)

        Full customization:
            config = MLP1DConfig(
                w1, w2, w3,
                mesh_device=custom_device,
                decode_w1_w3_prg_config=my_program_config,
                ...
            )
            mlp = MLP1D.from_config(config)
        """
        if not isinstance(config, MLP1DConfig):
            raise TypeError("MLP1D.from_config expects MLP1DConfig")
        instance = object.__new__(cls)
        super(MLP1D, instance).__init__()
        instance.config = resolve_mlp1d_arch_config(config)
        instance._device_weights_loaded = False
        return instance

    def load_device_weights(self):
        """Materialize LazyWeights onto device. Called automatically on first forward; idempotent."""
        if self._device_weights_loaded:
            return

        assert self.config.is_resolved(), "config must be resolved before loading device weights!"

        self.w1 = self.config.w1.get_device_weight()
        self.w2 = self.config.w2.get_device_weight()
        self.w3 = self.config.w3.get_device_weight()

        self._device_weights_loaded = True

    def decode_forward(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """
        Decode forward - NO if-else, fully flattened.

        Execution path:
          linear(w1) → linear(w3) → mul+silu → reshard → linear(w2) → all_reduce(sharded) → reshard
        """
        self.load_device_weights()
        x = _load_input_device_tensor(x, self.config, mode="decode")
        cfg = self.config

        # --- STAGE 1: W1/W3 Linear (L1 sharded) ---
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=cfg.linear_dtype,
            core_grid=None,
            compute_kernel_config=cfg.decode_ff1_3_compute_kernel_cfg,
            program_config=cfg.decode_w1_w3_prg_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        if cfg.decode_spill_w1_to_dram_before_w3:
            w1_dram = ttnn.to_memory_config(w1_out, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(w1_out)
            w1_out = w1_dram

        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=cfg.linear_dtype,
            core_grid=None,
            compute_kernel_config=cfg.decode_ff1_3_compute_kernel_cfg,
            program_config=cfg.decode_w1_w3_prg_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        # --- STAGE 2: No CCL for non-TG ---

        # --- STAGE 3: Activation + Multiply ---
        mul_out_memcfg = ttnn.DRAM_MEMORY_CONFIG if cfg.decode_spill_w1_to_dram_before_w3 else w1_out.memory_config()
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[cfg.mlp_activation_type],
            dtype=cfg.mul_dtype,
            memory_config=mul_out_memcfg,
        )

        # --- STAGE 3.5: Reshard for w2 ---
        w2_in = ttnn.to_memory_config(w2_in, cfg.decode_mlp2_input_memcfg)

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        # --- STAGE 4: No all_gather for non-TG ---

        # --- STAGE 5: W2 Linear ---
        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=cfg.decode_ff2_compute_kernel_cfg,
            dtype=cfg.linear_dtype,
            program_config=cfg.decode_w2_prg_config,
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
        w2_out_reduced = ttnn.to_memory_config(w2_out_reduced, cfg.decode_residual_memcfg)

        return w2_out_reduced

    def prefill_forward(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """
        Prefill forward - minimal runtime logic for seq_len-dependent configs.

        Execution path:
          [reshape if seq_len >= cutoff] → linear(w1) → linear(w3) → mul+silu → linear(w2) → all_reduce → reshape
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

        # Seq_len-dependent: get program configs by calling methods on config
        pc_w1_w3 = cfg.prefill_w1_w3_prg_config(seq_len)
        pc_w2 = cfg.prefill_w2_prg_config(seq_len)

        # --- STAGE 1: W1/W3 Linear (DRAM) ---
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=cfg.linear_dtype,
            core_grid=None,
            compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
            program_config=pc_w1_w3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=cfg.linear_dtype,
            core_grid=None,
            compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
            program_config=pc_w1_w3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        # --- STAGE 2: No CCL for non-TG ---

        # --- STAGE 3: Activation + Multiply ---
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[cfg.mlp_activation_type],
            dtype=cfg.mul_dtype,
            memory_config=w1_out.memory_config(),
        )

        # --- STAGE 3.5: No reshard for prefill ---

        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)

        # --- STAGE 4: No all_gather for non-TG ---

        # --- STAGE 5: W2 Linear ---
        # Above seq_len > 128 the folded-batch down-proj is large; minimal_matmul is ~2.5x faster than
        # ttnn.linear there (TTTv1 parity). Opt-in via prefill_w2_minimal_matmul; output shape is
        # identical to the ttnn.linear path. FF1/FF3 stay on ttnn.linear (matches TTTv1).
        if cfg.use_minimal_w2_matmul(seq_len):
            w2_out = ttnn.experimental.minimal_matmul(
                w2_in,
                self.w2,
                compute_kernel_config=cfg.ff2_compute_kernel_cfg,
                config=cfg.prefill_w2_minimal_matmul_config(seq_len),
            )
        else:
            w2_out = ttnn.linear(
                w2_in,
                self.w2,
                compute_kernel_config=cfg.ff2_compute_kernel_cfg,
                dtype=cfg.linear_dtype,
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

    def forward(self, x: ttnn.Tensor | LazyWeight, mode: str | Mode) -> ttnn.Tensor:
        """Dispatch to the appropriate forward method based on mode."""
        if isinstance(mode, Mode):
            mode = mode.value
        if mode == "decode":
            return self.decode_forward(x)
        else:
            return self.prefill_forward(x)

    def _all_reduce_decode(self, w2_out: ttnn.Tensor) -> ttnn.Tensor:
        """All-reduce for decode mode (sharded input)."""
        cfg = self.config
        if cfg.mesh_device.get_num_devices() == 1:
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
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=cfg.num_reduce_scatter_links,
            memory_config=w2_out.memory_config(),
            intermediate_memory_config=cfg.decode_rs_memory_config,
            topology=cfg.topology,
            chunks_per_sync=cfg.decode_rs_chunks_per_sync,
            num_workers_per_link=cfg.decode_rs_num_workers_per_link,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
        )
        w2_out.deallocate(True)
        return reduced

    def _all_reduce_prefill(self, w2_out: ttnn.Tensor) -> ttnn.Tensor:
        """All-reduce for prefill mode (interleaved input)."""
        cfg = self.config
        if cfg.mesh_device.get_num_devices() == 1:
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
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=cfg.num_reduce_scatter_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=cfg.topology,
            chunks_per_sync=CCL_CHUNKS_PER_SYNC,
            num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
        )
        w2_out.deallocate(True)
        return reduced

    # [INFO] this is the entry point for TTTv1 model_config.py and will retire with TTTv1
    @classmethod
    def from_model_args(
        cls,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        weight_cache_path,
        layer_num: int,
        dtype=None,
        model_config=None,
        state_dict_prefix: Optional[str] = None,
        prefetcher=None,
    ):
        """Factory method for backward compatibility with ModelArgs.

        Args:
            mesh_device: The mesh device to use.
            tt_ccl: The TT CCL instance.
            args: Model arguments (ModelArgs instance).
            state_dict: The state dictionary containing weights.
            weight_cache_path: Path for weight caching.
            layer_num: The layer number.
            dtype: Optional data type for weights (for signature compatibility with TTTv1 MLP).
            model_config: Optional model config dict. If None, calls args.get_model_config().
            state_dict_prefix: Optional prefix for state dict keys.

        Note:
            The `dtype` parameter is accepted for signature compatibility with TTTv1 MLP
            but is not used directly - dtype is determined by the DecodersPrecision config.
        """
        if args.is_galaxy:
            raise ValueError("MLP1D cannot be used for Galaxy devices.")

        import torch

        from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

        # Get model_config for overrides - use passed model_config if provided
        if model_config is None:
            model_config = args.get_model_config()
        decoders_opt = model_config.get("DECODERS_OPTIMIZATIONS")
        effective_layer_num = max(layer_num, 0)

        # Extract settings from args/model_config
        ccl_topology = args.ccl_topology()

        if state_dict_prefix is None:
            state_dict_prefix = args.get_state_dict_prefix("MLP", layer_num)

        # Get dtypes from optimizer config
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
        decode_w1_w3_prg_config = args.get_mlp_ff1_3_prg_config(Mode.DECODE, None, None)
        decode_w2_prg_config = args.get_mlp_ff2_prg_config(Mode.DECODE, None, None)
        decode_mlp2_input_memcfg = args.get_mlp_binary_mult_mem_config(Mode.DECODE)
        decode_residual_memcfg = args.get_mlp_output_mem_config(Mode.DECODE, None)
        mlp_rs_cfg = model_config.get("MLP_RS_CONFIG", {})

        # Compute memory configs for weights
        num_devices = mesh_device.get_num_devices()
        tile_size = TILE_SIZE
        dram_size = mesh_device.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(dram_size.x - 1, dram_size.y - 1),
                )
            }
        )

        w1_w3_mem_config = _create_dram_sharded_mem_config(
            k=args.dim,
            n=args.hidden_dim // num_devices,
            dram_grid=dram_grid,
            tile_size=tile_size,
            dram_cores=dram_size.x,
        )
        w2_mem_config = _create_dram_sharded_mem_config(
            k=args.hidden_dim // num_devices,
            n=args.dim,
            dram_grid=dram_grid,
            tile_size=tile_size,
            dram_cores=dram_size.x,
        )

        cache_dir = None if args.dummy_weights else Path(weight_cache_path) / state_dict_prefix

        # Create LazyWeights
        def make_weight_source(name: str, shard_dim: int):
            tensor = torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
            return pad_dim_to_size(tensor, dim=shard_dim, size=args.hidden_dim)

        w1 = LazyWeight(
            source=make_weight_source("w1", -1),
            dtype=ff1_3_dtype,
            device=mesh_device,
            mesh_mapper_config=ttnn.MeshMapperConfig(
                placements=[ttnn.PlacementShard(-1)],
                mesh_shape_override=ttnn.MeshShape([mesh_device.get_num_devices()]),
            ),
            layout=ttnn.TILE_LAYOUT,
            memory_config=w1_w3_mem_config,
            cache_dir_weight_name=(cache_dir, "w1_sharded") if cache_dir else None,
        )
        w2 = LazyWeight(
            source=make_weight_source("w2", -2),
            dtype=ff2_dtype,
            device=mesh_device,
            mesh_mapper_config=ttnn.MeshMapperConfig(
                placements=[ttnn.PlacementShard(-2)],
                mesh_shape_override=ttnn.MeshShape([mesh_device.get_num_devices()]),
            ),
            layout=ttnn.TILE_LAYOUT,
            memory_config=w2_mem_config,
            cache_dir_weight_name=(cache_dir, "w2_sharded") if cache_dir else None,
        )
        w3 = LazyWeight(
            source=make_weight_source("w3", -1),
            dtype=ff1_3_dtype,
            device=mesh_device,
            mesh_mapper_config=ttnn.MeshMapperConfig(
                placements=[ttnn.PlacementShard(-1)],
                mesh_shape_override=ttnn.MeshShape([mesh_device.get_num_devices()]),
            ),
            layout=ttnn.TILE_LAYOUT,
            memory_config=w1_w3_mem_config,
            cache_dir_weight_name=(cache_dir, "w3_sharded") if cache_dir else None,
        )

        # Create config with all the overrides and use from_config
        config = MLP1DConfig(
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
            decode_rs_memory_config=mlp_rs_cfg.get("rs_memory_config", ttnn.L1_MEMORY_CONFIG),
            decode_rs_chunks_per_sync=mlp_rs_cfg.get("chunks_per_sync", 1),
            decode_rs_num_workers_per_link=mlp_rs_cfg.get("num_workers_per_link", 1),
            decode_w1_w3_prg_config=decode_w1_w3_prg_config,
            decode_w2_prg_config=decode_w2_prg_config,
            decode_mlp2_input_memcfg=decode_mlp2_input_memcfg,
            decode_residual_memcfg=decode_residual_memcfg,
            w1_w3_dtype=ff1_3_dtype,
            w2_dtype=ff2_dtype,
            activation_dtype=activation_dtype,
            decode_spill_w1_to_dram_before_w3=False,
            ff1_3_compute_kernel_cfg=ff1_3_compute_kernel_cfg,
            ff2_compute_kernel_cfg=ff2_compute_kernel_cfg,
            decode_ff1_3_compute_kernel_cfg=ff1_3_compute_kernel_cfg,
            decode_ff2_compute_kernel_cfg=ff2_compute_kernel_cfg,
            prefill_len_cutoff=args.prefill_len_cutoff,
        )
        return cls.from_config(config)


# =============================================================================
# Config helper functions (adapted from TTTv1 model_config.py)
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
    target = max_cores // 2  # prefer half the grid for balanced utilization
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


def _dram_shard_core_grid(k: int, tile_size: int = TILE_SIZE) -> ttnn.CoreGrid:
    """Get core grid for DRAM sharding based on K dimension."""
    rows, cols = _find_grid(k // tile_size)
    return ttnn.CoreGrid(x=cols, y=rows)


def _dram_shard_core_grid_k_n(k: int, n: int, tile_size: int = TILE_SIZE) -> ttnn.CoreGrid:
    """Get core grid for DRAM sharding based on K and N dimensions."""
    rows, cols = _find_grid_k_n(k // tile_size, n // tile_size)
    return ttnn.CoreGrid(x=cols, y=rows)


def _dram_matmul_config(
    m: int, k: int, n: int, num_cores: int, tile_size: int = TILE_SIZE, fused_activation=None
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
    tile_size: int = TILE_SIZE,
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
    out_subblock_w = get_out_subblock_w(per_core_n, out_subblock_h)

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


def _copy_compute_kernel_config(arch, config=None) -> ttnn.DeviceComputeKernelConfig:
    """Create an independent concrete compute config for ``arch``."""
    if config is None:
        values = {
            "math_fidelity": ttnn.MathFidelity.HiFi2,
            "math_approx_mode": False,
            "fp32_dest_acc_en": False,
            "packer_l1_acc": True,
            "dst_full_sync_en": False,
        }
    else:
        required = (
            "math_fidelity",
            "math_approx_mode",
            "fp32_dest_acc_en",
            "packer_l1_acc",
            "dst_full_sync_en",
        )
        missing = [field for field in required if not hasattr(config, field)]
        if missing:
            raise ValueError(f"MLP1D compute config is missing fields: {', '.join(missing)}")
        values = {field: getattr(config, field) for field in required}
        if hasattr(config, "throttle_level"):
            values["throttle_level"] = config.throttle_level

    candidate = ttnn.WormholeComputeKernelConfig(**values)
    return ttnn.init_device_compute_kernel_config(arch, candidate)


def _compute_kernel_config_hifi2_fp16(arch) -> ttnn.DeviceComputeKernelConfig:
    """Default compute kernel config for MLP (HiFi2 with FP16 accumulation)."""
    return _copy_compute_kernel_config(arch)


def _resolve_mlp1d_mesh(config: MLP1DConfig):
    mesh_device = config.mesh_device or getattr(config.w1, "device", None) or ttnn.GetDefaultDevice()
    if mesh_device is None:
        raise ValueError("MLP1D requires a mesh_device or a weight associated with a mesh device")
    for name in ("w1", "w2", "w3"):
        weight_device = getattr(getattr(config, name), "device", None)
        if weight_device is not None and weight_device != mesh_device:
            raise ValueError(f"MLP1D {name} device must match mesh_device")
    return mesh_device


def _validate_mlp1d_arch_config(config: MLP1DConfig, arch) -> None:
    for field in (
        "ff1_3_compute_kernel_cfg",
        "ff2_compute_kernel_cfg",
        "decode_ff1_3_compute_kernel_cfg",
        "decode_ff2_compute_kernel_cfg",
    ):
        if getattr(config, field) is None:
            raise ValueError(f"MLP1D resolved config requires {field}")

    if config.prefill_len_cutoff <= 0 or config.prefill_len_cutoff % TILE_SIZE:
        raise ValueError("MLP1D prefill_len_cutoff must be a positive multiple of the tile size")
    if config.prefill_dram_shard_grid_width <= 0:
        raise ValueError("MLP1D prefill DRAM shard grid width must be positive")

    mesh_device = _resolve_mlp1d_mesh(config)
    dram_width = mesh_device.dram_grid_size().x
    expected_dram_width = 8 if arch == ttnn.device.Arch.WORMHOLE_B0 else dram_width
    if config.prefill_dram_shard_grid_width != expected_dram_width:
        raise ValueError(
            "MLP1D prefill DRAM shard grid width does not match the resolved architecture/SKU "
            f"({config.prefill_dram_shard_grid_width} != {expected_dram_width})"
        )

    compute_grid = mesh_device.compute_with_storage_grid_size()
    dim = config.dim if config.dim is not None else config.w1.source.shape[-2]
    hidden_dim = config.hidden_dim if config.hidden_dim is not None else config.w1.source.shape[-1]
    grid_tile_shapes = {
        "prefill_ff1_ff3_grid": (8, dim // TILE_SIZE),
        "prefill_ff2_grid": (
            8,
            get_padded_hidden_dim(hidden_dim, mesh_device.get_num_devices(), TILE_SIZE) // TILE_SIZE,
        ),
    }
    for field, tile_shape in grid_tile_shapes.items():
        grid = getattr(config, field)
        if len(grid) != 2 or min(grid) <= 0:
            raise ValueError(f"MLP1D {field} must contain two positive dimensions")
        if grid[0] > compute_grid.x or grid[1] > compute_grid.y:
            raise ValueError(f"MLP1D {field} {grid} exceeds mesh compute grid ({compute_grid.x}, {compute_grid.y})")
        if tile_shape[0] % grid[0] or tile_shape[1] % grid[1]:
            raise ValueError(f"MLP1D {field} {grid} must evenly divide tile shape {tile_shape}")


def resolve_mlp1d_arch_config(config: MLP1DConfig) -> MLP1DConfig:
    """Return a fully resolved config after one concrete-mesh architecture query."""
    if not isinstance(config, MLP1DConfig):
        raise TypeError("resolve_mlp1d_arch_config expects MLP1DConfig")
    mesh_device = _resolve_mlp1d_mesh(config)
    arch = mesh_device.arch()
    if arch not in (ttnn.device.Arch.WORMHOLE_B0, ttnn.device.Arch.BLACKHOLE):
        raise ValueError(f"Unsupported MLP1D architecture: {arch}")

    dim = config.dim if config.dim is not None else config.w1.source.shape[-2]
    hidden_dim = config.hidden_dim if config.hidden_dim is not None else config.w1.source.shape[-1]
    if dim <= 0 or dim % TILE_SIZE:
        raise ValueError("MLP1D dim must be a positive multiple of the tile size")
    if hidden_dim <= 0:
        raise ValueError("MLP1D hidden_dim must be positive")
    num_devices = mesh_device.get_num_devices()
    padded_hidden_dim = get_padded_hidden_dim(hidden_dim, num_devices, TILE_SIZE)
    # Simple construction retains the legacy TTTv1 cutoff (1024 on Wormhole,
    # 512 on Blackhole). Blackhole uses the concrete SKU's DRAM width (7 on
    # P100, 8 on P150), while Wormhole's accepted recipe intentionally uses 8.
    default_prefill_len_cutoff = 1024 if arch == ttnn.device.Arch.WORMHOLE_B0 else 512
    default_prefill_dram_shard_grid_width = (
        8 if arch == ttnn.device.Arch.WORMHOLE_B0 else mesh_device.dram_grid_size().x
    )
    default_prefill_ff1_ff3_grid = _find_prefill_grid(8, dim // TILE_SIZE)
    default_prefill_ff2_grid = _find_prefill_grid(8, padded_hidden_dim // TILE_SIZE)
    resolved = replace(
        config,
        ff1_3_compute_kernel_cfg=_copy_compute_kernel_config(arch, config.ff1_3_compute_kernel_cfg),
        ff2_compute_kernel_cfg=_copy_compute_kernel_config(arch, config.ff2_compute_kernel_cfg),
        decode_ff1_3_compute_kernel_cfg=_copy_compute_kernel_config(arch, config.decode_ff1_3_compute_kernel_cfg),
        decode_ff2_compute_kernel_cfg=_copy_compute_kernel_config(arch, config.decode_ff2_compute_kernel_cfg),
        prefill_len_cutoff=(
            default_prefill_len_cutoff if config.prefill_len_cutoff is None else config.prefill_len_cutoff
        ),
        prefill_dram_shard_grid_width=(
            default_prefill_dram_shard_grid_width
            if config.prefill_dram_shard_grid_width is None
            else config.prefill_dram_shard_grid_width
        ),
        prefill_ff1_ff3_grid=(
            default_prefill_ff1_ff3_grid if config.prefill_ff1_ff3_grid is None else config.prefill_ff1_ff3_grid
        ),
        prefill_ff2_grid=(default_prefill_ff2_grid if config.prefill_ff2_grid is None else config.prefill_ff2_grid),
    )
    _validate_mlp1d_arch_config(resolved, arch)
    return _resolve_mlp1d_config(resolved)


def _create_dram_sharded_mem_config(
    k: int, n: int, dram_grid: ttnn.CoreRangeSet, tile_size: int = TILE_SIZE, dram_cores: int = 12
) -> ttnn.MemoryConfig:
    """Create DRAM-sharded memory config for weight tensors."""
    padded_size = math.ceil(n / (tile_size * dram_cores)) * (tile_size * dram_cores)
    shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def _resolve_mlp1d_config(config: MLP1DConfig) -> MLP1DConfig:
    """Materialize non-architecture fields after architecture requests resolve."""

    to_set = {}

    # --- Phase 1: Foundational fields (order matters due to dependencies) ---

    # Derive dimensions from weights
    # w1 is expected in TTNN layout: (dim, hidden_dim) - caller must transpose if needed
    dim = config.dim
    if config.dim is None:
        dim = config.w1.source.shape[-2]
        to_set["dim"] = dim

    hidden_dim = config.hidden_dim
    if config.hidden_dim is None:
        hidden_dim = config.w1.source.shape[-1]
        to_set["hidden_dim"] = hidden_dim

    # Derive and validate the concrete mesh without another architecture query.
    mesh_device = _resolve_mlp1d_mesh(config)
    if config.mesh_device is None:
        to_set["mesh_device"] = mesh_device

    # Derive tt_ccl
    tt_ccl = config.tt_ccl
    if config.tt_ccl is None:
        tt_ccl = get_tt_ccl(mesh_device)
        to_set["tt_ccl"] = tt_ccl

    assert tt_ccl is not None, "tt_ccl must be available at this point!"
    assert tt_ccl.mesh_device == mesh_device, "tt_ccl must match the device of mesh_device!"

    # Auto-detect topology
    topology = config.topology
    if config.topology is None:
        topology = default_topology(mesh_device)
        to_set["topology"] = topology

    # --- Phase 2: Derived fields ---

    num_devices = mesh_device.get_num_devices()
    tile_size = TILE_SIZE
    tile_padded_batch_rows = tile_size * math.ceil(config.max_batch_size / tile_size)

    # Compute padded hidden_dim for memory configs (must match auto-padding in LazyWeight)
    padded_hidden_dim = get_padded_hidden_dim(hidden_dim, num_devices, tile_size)

    # Always computed (not user-overridable None fields)
    w1_w3_dtype = config.w1_w3_dtype
    if config.w1_w3_dtype is None:
        w1_w3_dtype = ttnn.bfloat8_b
        to_set["w1_w3_dtype"] = w1_w3_dtype
    w2_dtype = config.w2_dtype
    if config.w2_dtype is None:
        w2_dtype = ttnn.bfloat8_b
        to_set["w2_dtype"] = w2_dtype

    if config.linear_dtype is None:
        to_set["linear_dtype"] = config.activation_dtype or ttnn.bfloat16
    if config.mul_dtype is None:
        to_set["mul_dtype"] = config.activation_dtype or ttnn.bfloat8_b
    prefill_len_cutoff = config.prefill_len_cutoff

    # --- Phase 3: Decode program configs ---
    # Note: Use padded_hidden_dim to match auto-padding in LazyWeight

    mlp_core_grid = _dram_shard_core_grid_k_n(dim, padded_hidden_dim // num_devices)

    if config.decode_input_memcfg is None:
        to_set["decode_input_memcfg"] = ttnn.create_sharded_memory_config(
            (tile_padded_batch_rows, dim // mlp_core_grid.num_cores),  # Shard shape: 1 shard per core
            mlp_core_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    if config.decode_w1_w3_prg_config is None:
        to_set["decode_w1_w3_prg_config"] = _dram_matmul_config(
            m=tile_padded_batch_rows,
            k=dim,
            n=padded_hidden_dim // num_devices,
            num_cores=mlp_core_grid.num_cores,
        )

    mlp2_core_grid = _dram_shard_core_grid_k_n(padded_hidden_dim // num_devices, dim)

    if config.decode_w2_prg_config is None:
        to_set["decode_w2_prg_config"] = _dram_matmul_config(
            m=tile_padded_batch_rows,
            k=padded_hidden_dim // num_devices,
            n=dim,
            num_cores=mlp2_core_grid.num_cores,
        )

    if config.decode_mlp2_input_memcfg is None:
        to_set["decode_mlp2_input_memcfg"] = ttnn.create_sharded_memory_config(
            (tile_padded_batch_rows, padded_hidden_dim // num_devices // mlp2_core_grid.num_cores),
            mlp2_core_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    if config.decode_residual_memcfg is None:
        residual_grid = _dram_shard_core_grid(dim // num_devices)
        to_set["decode_residual_memcfg"] = ttnn.create_sharded_memory_config(
            (tile_padded_batch_rows, dim // residual_grid.num_cores // num_devices),
            residual_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    # --- Phase 4: Prefill program configs ---

    # Matching per_core_N to the resolved SKU width avoids silent PCC issues
    # on P100 while preserving the accepted Wormhole width of 8.
    dram_shard_grid_width = config.prefill_dram_shard_grid_width

    if config.prefill_input_memcfg is None:
        to_set["prefill_input_memcfg"] = ttnn.DRAM_MEMORY_CONFIG

    if config.prefill_w1_w3_prg_config is None:
        prefill_mlp_grid_size = config.prefill_ff1_ff3_grid
        n_w1_w3 = padded_hidden_dim // num_devices

        @lru_cache
        def w1_w3_prg_config(seq_len: int):
            return _matmul_config(
                m=min(seq_len, prefill_len_cutoff),
                k=dim,
                n=n_w1_w3,
                grid_size=prefill_mlp_grid_size,
                per_core_n=math.ceil(n_w1_w3 / (tile_size * dram_shard_grid_width)),
            )

        to_set["prefill_w1_w3_prg_config"] = lambda seq_len: w1_w3_prg_config(seq_len)

    if config.prefill_w2_prg_config is None:
        n_w2 = dim
        grid_size = config.prefill_ff2_grid

        @lru_cache
        def w2_prg_config(seq_len: int):
            return _matmul_config(
                m=min(seq_len, prefill_len_cutoff),
                k=padded_hidden_dim,
                n=n_w2,
                grid_size=grid_size,
                per_core_n=math.ceil(n_w2 / (tile_size * dram_shard_grid_width)),
            )

        to_set["prefill_w2_prg_config"] = lambda seq_len: w2_prg_config(seq_len)

    # minimal_matmul config for W2 (only materialized when the opt-in is set). Block sizes mirror TTTv1
    # (mlp.py via model_config.py:1329-1334: 8/8/8 blocks); the compute grid is the W2 prefill grid
    # (find_prefill_grid over padded_hidden_dim), equivalent to TTTv1's mlp2_grid(seq_len).
    if config.prefill_w2_minimal_matmul and config.prefill_w2_minimal_matmul_config is None:
        minimal_w2_grid = config.prefill_ff2_grid

        @lru_cache
        def w2_minimal_matmul_config(seq_len: int):
            return ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                compute_with_storage_grid_size=ttnn.CoreCoord(minimal_w2_grid[0], minimal_w2_grid[1]),
            )

        to_set["prefill_w2_minimal_matmul_config"] = lambda seq_len: w2_minimal_matmul_config(seq_len)

    # --- Phase 5: Weight memory configs ---

    dram_grid_size = mesh_device.dram_grid_size()
    dram_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
            )
        }
    )

    w1_w3_memcfg = config.w1_w3_memcfg
    if w1_w3_memcfg is None:
        w1_w3_memcfg = _create_dram_sharded_mem_config(
            k=dim,
            n=padded_hidden_dim // num_devices,
            dram_grid=dram_grid,
            tile_size=tile_size,
            dram_cores=dram_grid_size.x,
        )
        to_set["w1_w3_memcfg"] = w1_w3_memcfg

    w2_memcfg = config.w2_memcfg
    if w2_memcfg is None:
        w2_memcfg = _create_dram_sharded_mem_config(
            k=padded_hidden_dim // num_devices,
            n=dim,
            dram_grid=dram_grid,
            tile_size=tile_size,
            dram_cores=dram_grid_size.x,
        )
        to_set["w2_memcfg"] = w2_memcfg

    # --- Phase 6: Resolve LazyWeights ---

    w1_w3_mesh_mapper_config = ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(-1)],
        mesh_shape_override=ttnn.MeshShape([num_devices]),
    )
    to_set["w1"] = resolve_lazy_weight(
        config.w1,
        device=mesh_device,
        memory_config=w1_w3_memcfg,
        mesh_mapper_config=w1_w3_mesh_mapper_config,
        layout=ttnn.TILE_LAYOUT,
        dtype=w1_w3_dtype,
    )

    to_set["w2"] = resolve_lazy_weight(
        config.w2,
        device=mesh_device,
        memory_config=w2_memcfg,
        mesh_mapper_config=ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(-2)],
            mesh_shape_override=ttnn.MeshShape([num_devices]),
        ),
        layout=ttnn.TILE_LAYOUT,
        dtype=w2_dtype,
    )

    to_set["w3"] = resolve_lazy_weight(
        config.w3,
        device=mesh_device,
        memory_config=w1_w3_memcfg,
        mesh_mapper_config=w1_w3_mesh_mapper_config,
        layout=ttnn.TILE_LAYOUT,
        dtype=w1_w3_dtype,
    )

    # --- Final: Create new config with all resolved fields ---
    # todo)) using the current if <field> is None else to_set[<field>] does not seem to be worth the saving in space. Maybe cleaner to build all the fields in to_set and then replace the config with them with:
    #     to_override_set = {k: v for k, v in kwargs.items() if getattr(config, k, None) is None}
    #     return replace(config, **to_override_set)
    resolved_config = replace(config, **to_set)

    assert all(
        [resolved_config.w1.is_resolved(), resolved_config.w2.is_resolved(), resolved_config.w3.is_resolved()]
    ), "All weights must be resolved!"
    assert resolved_config.is_resolved(), "Config must be resolved!"
    # check that the padded shapes match the config
    assert resolved_config.w1.padded_shape == (dim, padded_hidden_dim), "w1 padded_shape does not match the config!"
    assert resolved_config.w2.padded_shape == (padded_hidden_dim, dim), "w2 padded_shape does not match the config!"
    assert resolved_config.w3.padded_shape == (dim, padded_hidden_dim), "w3 padded_shape does not match the config!"

    return resolved_config


def _load_input_device_tensor(x: ttnn.Tensor | LazyWeight, config: MLP1DConfig, mode: str) -> ttnn.Tensor:
    """Resolve the input tensor to ttnn tensor if x is a LazyWeight, otherwise sanity check x and then return as is"""
    assert mode in ["decode", "prefill"], "mode must be one of decode or prefill!"
    mem_cfg = config.decode_input_memcfg if mode == "decode" else config.prefill_input_memcfg
    if isinstance(x, LazyWeight):
        # resolve in place
        resolved_x = resolve_lazy_weight(
            x,
            device=config.mesh_device,
            memory_config=mem_cfg,
            mesh_mapper_config=None,  # replicated
            layout=ttnn.TILE_LAYOUT,
        )
        return resolved_x.get_device_weight()

    assert isinstance(x, ttnn.Tensor), "x must be a ttnn tensor at this point!"
    if x.memory_config() != mem_cfg:
        raise ValueError("Input tensor memory config does not match the config!")

    return x
