# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2-style Attention module for TG (Galaxy) devices with 2D mesh topology (4x8 or 8x4).

Single unified Attention2D class with separate forward methods:
  - decode_forward(): For decode mode (single token per user)
  - prefill_forward(): For prefill mode (multiple tokens)
  - forward(x, mode, **kwargs): Dispatcher that calls the appropriate method

Execution paths:
  Decode:
    1. QKV matmul → L1_WIDTH_SHARDED, 1DProgramConfig, ccl_dtype
    2. tt_all_reduce(cluster_axis=1) → qkv_out_gathered_memcfg
    3. slice_mat matmul → batch 32→8, bfloat16, create_head_input_memcfg
    4. nlp_create_qkv_heads_decode(batch_size_per_device_group)
    5. Q/K normalization (if present)
    6. Rotary embedding
    7. KV cache update
    8. SDPA
    9. tt_all_gather(cluster_axis=1, dim=2) → gather_users_memcfg
    10. user_selection_matrix matmul → CoreGrid(y=4, x=8), bfloat16
    11. WO matmul → CoreGrid(y=4, x=8), bfloat8_b (NOT DRAM-sharded)
    12. tt_all_reduce(cluster_axis=0, dim=0 or 3) → self_out_reduce_scatter_memcfg

  Prefill:
    1. Reshape for long sequences (same as 1D)
    2. QKV matmul → DRAM, ccl_dtype
    3. tt_all_reduce(cluster_axis=1) → DRAM
    4. Reshape back
    5. nlp_create_qkv_heads
    6. Q/K normalization
    7. Rotary embedding
    8. prefill_prepare_tensor_for_kv_cache() → TG-SPECIFIC
    9. KV cache fill
    10. SDPA
    11. Reshape and concat heads
    12. WO matmul
    13. tt_all_reduce(cluster_axis=0, dim=0) → DRAM

Key design decisions:
  - TG (Galaxy) specific: requires 2D mesh topology (4x8 or 8x4)
  - Uses cluster_axis for CCL operations (axis=0 and axis=1)
  - Uses batch_size_per_device_group for KV cache (batch split across device groups)
  - Uses slice_mat and user_selection_matrix for TG-specific batch handling
  - Uses 1D program config (not DRAM-sharded) for QKV matmul
"""

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig
from models.common.modules.tt_ccl import TT_CCL, get_tt_ccl
from models.common.tensor_utils import TILE_SIZE
from models.common.utility_functions import is_blackhole, nearest_32

# =============================================================================
# Constants
# =============================================================================

MAX_QKV_MM_SEQ_LEN = 2048  # Maximum sequence length for single QKV matmul


# =============================================================================
# Attention2DConfig dataclass
# =============================================================================


@dataclass
class Attention2DConfig:
    """
    Central configuration for Attention2D - the single source of truth for all settings.

    Simple usage (all defaults):
        config = Attention2DConfig(wqkv, wo)

    Override any field:
        config = Attention2DConfig(wqkv, wo, n_heads=32, head_dim=128)

    Full customization:
        config = Attention2DConfig(
            wqkv, wo,
            mesh_device=custom_device,
            decode_xqkv_prg_config=my_program_config,
            ...
        )
    """

    # Required: weights (LazyWeight)
    wqkv: LazyWeight  # Combined QKV projection weight
    wo: LazyWeight  # Output projection weight

    # Optional: Q/K normalization configs (e.g., for Qwen models)
    # Composed sub-module pattern: RMSNorm1DConfig instead of raw weights
    q_norm_config: RMSNorm1DConfig | None = None
    k_norm_config: RMSNorm1DConfig | None = None

    # Optional: QKV bias (e.g., for Qwen models)
    wqkv_bias: "torch.Tensor | None" = None  # type: ignore

    # Device and collectives
    mesh_device: ttnn.MeshDevice | None = None
    tt_ccl: TT_CCL | None = None
    topology: Optional[ttnn.Topology] = None  # None = auto-detect
    num_reduce_scatter_links: int = 1
    num_all_gather_links: int = 2

    # Model dimensions (derived from weights if None)
    dim: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None
    head_dim: int | None = None
    qkv_size: int | None = None  # head_dim * (2 * n_kv_heads + n_heads)

    # Batch and sequence config
    max_batch_size: int = 32
    max_seq_len: int = 128 * 1024

    # Attention config
    scale: float | None = None  # Default: head_dim ** -0.5
    sliding_window: int | None = None  # For sliding window attention
    use_qk_fused: bool = False  # Fused Q/K rotary embedding

    # KV cache config
    use_paged_kv_cache: bool = False
    paged_attention_config: "PagedAttentionConfig | None" = None  # type: ignore
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    min_kv_prefill_shard_seqlen: int | None = None

    # Weight dtypes
    wqkv_dtype: ttnn.DataType | None = None
    wo_dtype: ttnn.DataType | None = None
    activation_dtype: ttnn.DataType | None = None

    # Weight memory configs
    wqkv_memcfg: ttnn.MemoryConfig | None = None
    wo_memcfg: ttnn.MemoryConfig | None = None

    # TG-SPECIFIC dimensions
    num_device_groups: int | None = None  # num_devices // n_kv_heads
    batch_size_per_device_group: int | None = None  # max_batch_size // num_device_groups

    # TG-SPECIFIC dtype
    ccl_dtype: ttnn.DataType | None = None  # Explicit CCL dtype for TG

    # TG-SPECIFIC memory configs
    create_head_input_memcfg: ttnn.MemoryConfig | None = None
    qkv_out_gathered_memcfg: Callable[[int], ttnn.MemoryConfig] | None = None
    gather_users_memcfg: Callable[[int], ttnn.MemoryConfig] | None = None
    self_out_gathered_memcfg: Callable[[int], ttnn.MemoryConfig] | None = None
    self_out_reduce_scatter_memcfg: ttnn.MemoryConfig | None = None

    # TG-SPECIFIC program configs (1D config, not DRAM-sharded)
    decode_xqkv_prg_config: "ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None" = None
    decode_sdpa_prg_config: ttnn.SDPAProgramConfig | None = None
    decode_create_qkv_head_memcfg: ttnn.MemoryConfig | None = None
    decode_scores_memcfg: Callable[[int], ttnn.MemoryConfig] | None = None
    decode_input_memcfg: ttnn.MemoryConfig | None = None
    decode_residual_memcfg: ttnn.MemoryConfig | None = None

    # Prefill program configs
    prefill_input_memcfg: ttnn.MemoryConfig | None = None
    prefill_xqkv_prg_config: Callable[[int], ttnn.MatmulMultiCoreReuseMultiCastProgramConfig] | None = None
    prefill_sdpa_prg_config: Callable[[int, int | None], ttnn.SDPAProgramConfig] | None = None
    prefill_wo_prg_config: Callable[[int], ttnn.MatmulMultiCoreReuseMultiCastProgramConfig] | None = None

    # Compute kernel configs
    li_qkv_decode_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    sdpa_decode_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    li_o_decode_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    li_qkv_prefill_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    sdpa_prefill_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    li_o_prefill_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None

    # Internal TG tensors (created during resolution)
    _slice_mat: ttnn.Tensor | None = field(default=None, repr=False)
    _user_selection_matrix: ttnn.Tensor | None = field(default=None, repr=False)

    # Internal: pre-computed bias tensors
    _wqkv_bias_decode: list[ttnn.Tensor] | None = field(default=None, repr=False)
    _wqkv_bias_prefill: ttnn.Tensor | None = field(default=None, repr=False)

    def is_resolved(self) -> bool:
        """Check if all required fields are resolved."""
        required = [
            "wqkv",
            "wo",
            "mesh_device",
            "dim",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "qkv_size",
            "scale",
            "tt_ccl",
            "topology",
            "ccl_dtype",
            "num_device_groups",
            "batch_size_per_device_group",
            "_slice_mat",
            "_user_selection_matrix",
            "decode_xqkv_prg_config",
            "decode_sdpa_prg_config",
            "li_qkv_decode_compute_kernel_cfg",
            "sdpa_decode_compute_kernel_cfg",
            "li_o_decode_compute_kernel_cfg",
        ]

        return all(getattr(self, f) is not None for f in required)


# =============================================================================
# Attention2D Class
# =============================================================================


class Attention2D(LightweightModule):
    """
    Attention for TG (Galaxy) devices with 2D mesh topology.

    Simple API (90% of users):
        attn = Attention2D(wqkv, wo)

    Power API (10% of users) - any level of customization via config:
        config = Attention2DConfig(wqkv, wo, n_heads=32, head_dim=128)
        attn = Attention2D.from_config(config)

    Execution paths:
      Decode:  QKV matmul → all_reduce(axis=1) → slice_mat → create_qkv_heads → rotary → KV cache → SDPA → all_gather(axis=1) → user_selection → WO matmul → all_reduce(axis=0)
      Prefill: [reshape] → QKV matmul → all_reduce(axis=1) → create_qkv_heads → rotary → prepare_for_kv_cache → KV cache → SDPA → concat_heads → WO matmul → all_reduce(axis=0)
    """

    def __init__(self, wqkv: LazyWeight, wo: LazyWeight):
        """
        Simple API for 90% of users - derives all config from weights.

        Args:
            wqkv: Combined QKV projection weight, sharded on dim=-1
            wo: Output projection weight, sharded on dim=-2

        Note: Use from_config() to set n_heads, n_kv_heads, head_dim explicitly
        since they cannot be reliably inferred from weight shapes alone.
        """
        super().__init__()
        self.config = _resolve_attention2d_config(Attention2DConfig(wqkv=wqkv, wo=wo))
        self._device_weights_loaded = False
        self.layer_past = None  # KV cache for non-paged mode

    @classmethod
    def from_config(cls, config: Attention2DConfig):
        """
        Power API for 10% of users - any level of customization via config.

        Override any subset of fields in Attention2DConfig:
            config = Attention2DConfig(wqkv, wo, n_heads=32, head_dim=128)
            attn = Attention2D.from_config(config)
        """
        instance = object.__new__(cls)
        super(Attention2D, instance).__init__()
        instance.config = _resolve_attention2d_config(config)
        instance._device_weights_loaded = False
        instance.layer_past = None
        return instance

    def load_device_weights(self):
        """Load weights to device lazily."""
        if self._device_weights_loaded:
            return

        assert self.config.is_resolved(), "config must be resolved before loading device weights!"

        cfg = self.config
        self.wqkv = cfg.wqkv.get_device_weight()
        self.wo = cfg.wo.get_device_weight()

        # TG-specific tensors
        self.slice_mat = cfg._slice_mat
        self.user_selection_matrix = cfg._user_selection_matrix

        # Initialize Q/K norm RMSNorm1D instances if configs present
        if cfg.q_norm_config is not None:
            self.q_norm = RMSNorm1D.from_config(cfg.q_norm_config)
            self.q_norm.load_device_weights()
        else:
            self.q_norm = None

        if cfg.k_norm_config is not None:
            self.k_norm = RMSNorm1D.from_config(cfg.k_norm_config)
            self.k_norm.load_device_weights()
        else:
            self.k_norm = None

        # Pre-computed bias tensors
        self.wqkv_bias_decode = cfg._wqkv_bias_decode
        self.wqkv_bias_prefill = cfg._wqkv_bias_prefill

        self._device_weights_loaded = True

    def init_kv_cache(self):
        """
        Initialize KV cache for non-paged mode.
        Called automatically if use_paged_kv_cache=False.
        """
        cfg = self.config

        # TG uses batch_size_per_device_group for KV cache
        batch_size = cfg.batch_size_per_device_group

        if cfg.paged_attention_config:
            # Paged attention - external cache
            cache_k = _zeros_like_paged_cache(
                cfg.paged_attention_config,
                1,  # n_local_kv_heads = 1 for TG
                cfg.head_dim,
            )
            cache_v = _zeros_like_paged_cache(
                cfg.paged_attention_config,
                1,
                cfg.head_dim,
            )
        else:
            # Standard cache - TG uses 1 local KV head per device
            cache_k = _zeros_like_kv_cache(
                batch_size,
                1,  # n_local_kv_heads = 1 for TG
                cfg.max_seq_len,
                cfg.head_dim,
            )
            cache_v = _zeros_like_kv_cache(
                batch_size,
                1,
                cfg.max_seq_len,
                cfg.head_dim,
            )

        self.layer_past = [
            ttnn.from_torch(
                k_or_v,
                dtype=cfg.kv_cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=cfg.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(cfg.mesh_device),
            )
            for k_or_v in [cache_k, cache_v]
        ]

    def decode_forward(
        self,
        x: ttnn.Tensor | LazyWeight,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        transformation_mat: ttnn.Tensor,
        page_table: ttnn.Tensor | None = None,
        kv_cache: list[ttnn.Tensor] | None = None,
    ) -> ttnn.Tensor:
        """
        Decode forward for TG - single token per user.

        Args:
            x: Input tensor (seq_len, 1, batch, dim)
            current_pos: Current position tensor (batch_size,)
            rot_mats: Tuple of (cos, sin) rotation matrices for rotary embedding
            transformation_mat: Transformation matrix for rotary embedding
            page_table: Page table for paged attention (optional)
            kv_cache: External KV cache [keys, values] (optional, uses self.layer_past if None)

        Returns:
            Output tensor with same shape as input
        """
        self.load_device_weights()
        x = _load_input_device_tensor(x, self.config, mode="decode")
        cfg = self.config

        n_local_heads = cfg.n_heads // cfg.n_kv_heads  # For TG: n_heads // n_kv_heads
        n_local_kv_heads = 1  # TG: 1 local KV head per device

        # --- STAGE 1: QKV Matmul ---
        # TG uses 1D program config (not DRAM-sharded)
        xqkv_fused_sharded = ttnn.linear(
            x,
            self.wqkv,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=cfg.decode_xqkv_prg_config,
            compute_kernel_config=cfg.li_qkv_decode_compute_kernel_cfg,
            dtype=cfg.ccl_dtype,
        )

        # Add bias if present
        if self.wqkv_bias_decode:
            num_tiles = int(math.ceil(xqkv_fused_sharded.shape[-2] / TILE_SIZE))
            xqkv_fused_sharded = xqkv_fused_sharded + self.wqkv_bias_decode[num_tiles - 1]

        ttnn.deallocate(x)

        # --- STAGE 2: All-reduce QKV along cluster axis 1 ---
        # TTTv1 uses dim=0 by default for decode all-reduce
        xqkv_fused = self._all_reduce_tg(
            xqkv_fused_sharded,
            cluster_axis=1,
            dim=0,  # TTTv1 default
            sharded=True,
            memory_config=cfg.qkv_out_gathered_memcfg(list(cfg.mesh_device.shape)[1]),
        )
        ttnn.deallocate(xqkv_fused_sharded)

        # --- STAGE 3: TG-specific slice_mat to reduce batch 32→8 ---
        xqkv_fused = ttnn.matmul(
            self.slice_mat,
            xqkv_fused,
            dtype=ttnn.bfloat16,
            memory_config=cfg.create_head_input_memcfg,
        )

        # Reshape such that true unpadded batch is tracked in shape
        fqkv_shape = xqkv_fused.shape
        xqkv_fused = ttnn.reshape(
            xqkv_fused, (1, 1, cfg.batch_size_per_device_group, fqkv_shape[3]), (1, 1, 32, fqkv_shape[3])
        )

        # --- STAGE 4: Create QKV Heads ---
        q_heads_pre_rot, k_heads_pre_rot, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=n_local_heads,
            num_kv_heads=n_local_kv_heads,
            memory_config=cfg.decode_create_qkv_head_memcfg,
        )
        ttnn.deallocate(xqkv_fused)

        # --- STAGE 5: Q/K Normalization (optional) ---
        # Workaround: RMSNorm doesn't support HEIGHT_SHARDED inputs
        if self.q_norm is not None:
            q_mem_cfg = q_heads_pre_rot.memory_config()
            q_heads_pre_rot = ttnn.to_memory_config(q_heads_pre_rot, ttnn.L1_MEMORY_CONFIG, dtype=q_heads_pre_rot.dtype)
            q_heads_pre_rot = self.q_norm.decode_forward(q_heads_pre_rot)
            q_heads_pre_rot = ttnn.to_memory_config(q_heads_pre_rot, q_mem_cfg, dtype=q_heads_pre_rot.dtype)
        if self.k_norm is not None:
            k_mem_cfg = k_heads_pre_rot.memory_config()
            k_heads_pre_rot = ttnn.to_memory_config(k_heads_pre_rot, ttnn.L1_MEMORY_CONFIG, dtype=k_heads_pre_rot.dtype)
            k_heads_pre_rot = self.k_norm.decode_forward(k_heads_pre_rot)
            k_heads_pre_rot = ttnn.to_memory_config(k_heads_pre_rot, k_mem_cfg, dtype=k_heads_pre_rot.dtype)

        # --- STAGE 6: Rotary Embedding ---
        if cfg.use_qk_fused:
            q_heads_pre_rot, k_heads_pre_rot = self._to_qk_fused_memory_config(q_heads_pre_rot, k_heads_pre_rot)
            q_heads, k_heads = ttnn.experimental.rotary_embedding_llama_fused_qk(
                q_heads_pre_rot, k_heads_pre_rot, rot_mats[0], rot_mats[1], transformation_mat
            )
        else:
            q_heads = ttnn.experimental.rotary_embedding_llama(
                q_heads_pre_rot, rot_mats[0], rot_mats[1], transformation_mat, is_decode_mode=True
            )
            k_heads = ttnn.experimental.rotary_embedding_llama(
                k_heads_pre_rot, rot_mats[0], rot_mats[1], transformation_mat, is_decode_mode=True
            )

        ttnn.deallocate(q_heads_pre_rot)
        ttnn.deallocate(k_heads_pre_rot)

        # --- STAGE 7: KV Cache Update ---
        if kv_cache:
            keys, values = kv_cache[0], kv_cache[1]
        else:
            keys, values = self.layer_past[0], self.layer_past[1]

        if cfg.use_qk_fused:
            ttnn.experimental.paged_fused_update_cache(
                keys, k_heads, values, v_heads, update_idxs_tensor=current_pos, page_table=page_table
            )
        else:
            ttnn.experimental.paged_update_cache(keys, k_heads, update_idxs_tensor=current_pos, page_table=page_table)
            ttnn.experimental.paged_update_cache(values, v_heads, update_idxs_tensor=current_pos, page_table=page_table)

        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        # --- STAGE 8: SDPA ---
        if page_table is not None:
            attn_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_heads,
                keys,
                values,
                page_table_tensor=page_table,
                cur_pos_tensor=current_pos,
                scale=cfg.scale,
                sliding_window_size=cfg.sliding_window,
                program_config=cfg.decode_sdpa_prg_config,
                compute_kernel_config=cfg.sdpa_decode_compute_kernel_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
                q_heads,
                keys,
                values,
                cur_pos_tensor=current_pos,
                scale=cfg.scale,
                sliding_window_size=cfg.sliding_window,
                program_config=cfg.decode_sdpa_prg_config,
                compute_kernel_config=cfg.sdpa_decode_compute_kernel_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        ttnn.deallocate(q_heads)

        # --- STAGE 9: Reshape for concat heads ---
        attn_output_sharded = ttnn.to_memory_config(
            attn_output, memory_config=cfg.decode_scores_memcfg(cfg.batch_size_per_device_group)
        )

        attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(attn_output_sharded, num_heads=n_local_heads)
        ttnn.deallocate(attn_output_sharded)
        ttnn.deallocate(attn_output)

        # --- STAGE 10: All-gather along cluster axis 1 ---
        # NOTE: SDPA may collapse tensor to one-per-row (8 devices for [8,4] mesh).
        # After all_reduce on axis 1, all 4 columns have identical data, so SDPA
        # may "optimize" by outputting only one tensor per row. This causes issues
        # with subsequent all_gather operations. This is a known issue that needs
        # investigation into SDPA behavior on 2D meshes.
        attn_output_gathered = self._all_gather_axis1(
            attn_output_cat,
            dim=2,
            memory_config=cfg.gather_users_memcfg(list(cfg.mesh_device.shape)[1]),
        )
        ttnn.deallocate(attn_output_cat)

        # --- STAGE 11: TG-specific user_selection_matrix ---
        attn_output_gathered = ttnn.to_memory_config(attn_output_gathered, ttnn.L1_MEMORY_CONFIG)
        attn_output = ttnn.matmul(
            self.user_selection_matrix,
            attn_output_gathered,
            core_grid=ttnn.CoreGrid(y=4, x=8),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output_gathered)

        # --- STAGE 12: WO Matmul ---
        # TG uses CoreGrid(y=4, x=8), NOT DRAM-sharded
        dense_out = ttnn.matmul(
            attn_output,
            self.wo,
            core_grid=ttnn.CoreGrid(y=4, x=8),
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=cfg.li_o_decode_compute_kernel_cfg,
        )
        ttnn.deallocate(attn_output)

        # --- STAGE 13: Final All-Reduce along cluster axis 0 ---
        # TG uses dim=0 for small hidden_size, dim=3 otherwise
        all_reduce_dim = 0 if cfg.dim < 8192 else 3
        # For dim=8192, TTTv1 uses composite all-reduce (reduce-scatter + all-gather)
        use_composite = cfg.dim == 8192

        if cfg.dim == 8192:
            memory_config = cfg.self_out_reduce_scatter_memcfg
        else:
            memory_config = cfg.self_out_gathered_memcfg(list(cfg.mesh_device.shape)[0])

        dense_out_reduced = self._all_reduce_tg(
            dense_out,
            cluster_axis=0,
            dim=all_reduce_dim,
            sharded=True,
            memory_config=memory_config,
            use_composite=use_composite,
        )

        return dense_out_reduced

    def prefill_forward(
        self,
        x: ttnn.Tensor | LazyWeight,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        transformation_mat: ttnn.Tensor,
        user_id: int = 0,
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        kv_cache: list[ttnn.Tensor] | None = None,
    ) -> ttnn.Tensor:
        """
        Prefill forward for TG - multiple tokens.

        Args:
            x: Input tensor (1, 1, seq_len, dim)
            rot_mats: Tuple of (cos, sin) rotation matrices
            transformation_mat: Transformation matrix for rotary embedding
            user_id: User ID for KV cache fill
            page_table: Page table for paged attention
            chunk_page_table: Page table for chunked prefill
            chunk_start_idx: Start index for chunked prefill
            kv_cache: External KV cache [keys, values]

        Returns:
            Output tensor (1, 1, seq_len, dim)
        """
        self.load_device_weights()
        x = _load_input_device_tensor(x, self.config, mode="prefill")
        cfg = self.config

        seq_len = x.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "seq_len must be divisible by 128"

        n_local_heads = cfg.n_heads // cfg.n_kv_heads  # For TG
        n_local_kv_heads = 1  # TG: 1 local KV head per device

        # --- STAGE 1: Reshape for long sequences ---
        if seq_len > MAX_QKV_MM_SEQ_LEN:
            if seq_len % MAX_QKV_MM_SEQ_LEN != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {MAX_QKV_MM_SEQ_LEN}")
            x = ttnn.reshape(x, [1, seq_len // MAX_QKV_MM_SEQ_LEN, MAX_QKV_MM_SEQ_LEN, -1])

        # --- STAGE 2: QKV Matmul ---
        xqkv_fused = ttnn.linear(
            x,
            self.wqkv,
            dtype=cfg.ccl_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=cfg.li_qkv_prefill_compute_kernel_cfg,
            program_config=cfg.prefill_xqkv_prg_config(seq_len),
        )

        # Add bias if present
        if self.wqkv_bias_prefill is not None:
            xqkv_fused = xqkv_fused + self.wqkv_bias_prefill

        # --- STAGE 3: All-Reduce QKV along cluster axis 1 ---
        # TTTv1 uses dim=0 by default for all-reduce
        xqkv_fused = self._all_reduce_tg(
            xqkv_fused,
            cluster_axis=1,
            dim=0,  # TTTv1 default
            sharded=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Reshape back
        if seq_len > MAX_QKV_MM_SEQ_LEN:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        ttnn.deallocate(x)

        # --- STAGE 4: Create QKV Heads ---
        q_heads_pre_rot, k_heads_pre_rot, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=n_local_heads,
            num_kv_heads=n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv_fused)

        # --- STAGE 5: Q/K Normalization (optional) ---
        if self.q_norm is not None:
            q_heads_pre_rot = self.q_norm.prefill_forward(q_heads_pre_rot)
        if self.k_norm is not None:
            k_heads_pre_rot = self.k_norm.prefill_forward(k_heads_pre_rot)

        # --- STAGE 6: Rotary Embedding ---
        if q_heads_pre_rot.dtype != ttnn.bfloat16:
            q_heads_pre_rot = ttnn.typecast(q_heads_pre_rot, dtype=ttnn.bfloat16)

        q_heads = ttnn.experimental.rotary_embedding_llama(
            q_heads_pre_rot, rot_mats[0], rot_mats[1], transformation_mat, is_decode_mode=False
        )
        ttnn.deallocate(q_heads_pre_rot)

        if k_heads_pre_rot.dtype != ttnn.bfloat16:
            k_heads_pre_rot = ttnn.typecast(k_heads_pre_rot, dtype=ttnn.bfloat16)

        k_heads = ttnn.experimental.rotary_embedding_llama(
            k_heads_pre_rot, rot_mats[0], rot_mats[1], transformation_mat, is_decode_mode=False
        )
        ttnn.deallocate(k_heads_pre_rot)

        # --- STAGE 7: Typecast to cache dtype ---
        if kv_cache:
            keys, values = kv_cache[0], kv_cache[1]
        else:
            keys, values = self.layer_past[0], self.layer_past[1]

        k_heads_cache_dtype = ttnn.typecast(k_heads, dtype=keys.dtype)
        ttnn.deallocate(k_heads)

        v_heads_cache_dtype = ttnn.typecast(v_heads, dtype=values.dtype)
        ttnn.deallocate(v_heads)

        # --- STAGE 8: TG-specific prepare tensor for KV cache ---
        k_fill = self.prefill_prepare_tensor_for_kv_cache(k_heads_cache_dtype, user_id)
        v_fill = self.prefill_prepare_tensor_for_kv_cache(v_heads_cache_dtype, user_id)

        # --- STAGE 9: KV Cache Fill ---
        if page_table is not None:
            block_size = keys.shape[2]
            fill_page_table = chunk_page_table if chunk_page_table is not None else page_table
            page_len = fill_page_table.shape[1] * block_size

            k_fill_sliced = k_fill[:, :, :page_len, :] if page_len < k_fill.shape[2] else k_fill
            v_fill_sliced = v_fill[:, :, :page_len, :] if page_len < v_fill.shape[2] else v_fill

            ttnn.experimental.paged_fill_cache(keys, k_fill_sliced, fill_page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(values, v_fill_sliced, fill_page_table, batch_idx=user_id)
        else:
            ttnn.fill_cache(keys, k_fill, user_id % cfg.batch_size_per_device_group)
            ttnn.fill_cache(values, v_fill, user_id % cfg.batch_size_per_device_group)

        # --- STAGE 10: SDPA ---
        q_heads_sdpa = ttnn.typecast(q_heads, dtype=cfg.activation_dtype or ttnn.bfloat8_b)
        ttnn.deallocate(q_heads)

        if chunk_start_idx is not None:
            attn_output = ttnn.transformer.chunked_scaled_dot_product_attention(
                input_tensor_q=q_heads_sdpa,
                input_tensor_k=keys,
                input_tensor_v=values,
                page_table_tensor=page_table,
                chunk_start_idx=chunk_start_idx,
                compute_kernel_config=cfg.sdpa_prefill_compute_kernel_cfg,
                program_config=cfg.prefill_sdpa_prg_config(seq_len, chunk_start_idx),
            )
        else:
            attn_output = ttnn.transformer.scaled_dot_product_attention(
                q_heads_sdpa,
                k_heads_cache_dtype,
                v_heads_cache_dtype,
                is_causal=True,
                sliding_window_size=cfg.sliding_window,
                scale=cfg.scale,
                compute_kernel_config=cfg.sdpa_prefill_compute_kernel_cfg,
                program_config=cfg.prefill_sdpa_prg_config(seq_len, None),
            )

        ttnn.deallocate(q_heads_sdpa)
        ttnn.deallocate(k_heads_cache_dtype)
        ttnn.deallocate(v_heads_cache_dtype)

        # --- STAGE 11: Reshape and Concat Heads ---
        attn_output = ttnn.reshape(attn_output, [1, n_local_heads, -1, cfg.head_dim])

        attn_output_concat = ttnn.experimental.nlp_concat_heads(attn_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_output)

        # --- STAGE 12: Reshape for long sequences ---
        if seq_len > 1024:
            attn_output_concat = ttnn.reshape(attn_output_concat, [1, seq_len // 1024, 1024, -1])

        # --- STAGE 13: WO Matmul ---
        output = ttnn.linear(
            attn_output_concat,
            self.wo,
            compute_kernel_config=cfg.li_o_prefill_compute_kernel_cfg,
            dtype=cfg.activation_dtype or ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=cfg.prefill_wo_prg_config(seq_len),
        )

        # --- STAGE 14: Reshape back ---
        if seq_len > 1024:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])

        ttnn.deallocate(attn_output_concat)

        # --- STAGE 15: All-Reduce output along cluster axis 0 ---
        output = self._all_reduce_tg(
            output,
            cluster_axis=0,
            dim=0,  # TG prefill uses dim=0
            sharded=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return output

    def forward(
        self,
        x: ttnn.Tensor | LazyWeight,
        current_pos: ttnn.Tensor | None,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        transformation_mat: ttnn.Tensor,
        user_id: int = 0,
        mode: str = "decode",
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        kv_cache: list[ttnn.Tensor] | None = None,
    ) -> ttnn.Tensor:
        """Dispatch to the appropriate forward method based on mode."""
        if mode == "prefill":
            return self.prefill_forward(
                x,
                rot_mats,
                transformation_mat,
                user_id=user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
            )
        else:
            return self.decode_forward(
                x,
                current_pos,
                rot_mats,
                transformation_mat,
                page_table=page_table,
                kv_cache=kv_cache,
            )

    # =========================================================================
    # TG-specific CCL methods
    # =========================================================================

    def _all_reduce_tg(
        self,
        input_tensor: ttnn.Tensor,
        cluster_axis: int,
        dim: int,
        sharded: bool,
        memory_config: Any,
        use_composite: bool = False,
    ) -> ttnn.Tensor:
        """
        All-reduce for TG (Galaxy) devices along specified cluster axis.

        Uses all_gather + fast_reduce_nc by default (like TTTv1).
        Set use_composite=True for reduce-scatter + all-gather approach.
        """
        cfg = self.config

        # Ensure dim 0 and 1 are 1
        original_shape = input_tensor.shape
        if original_shape[0] != 1 or original_shape[1] != 1:
            input_tensor = ttnn.reshape(
                input_tensor, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
            )

        # Cast to CCL dtype if needed
        # NOTE: DO NOT apply memory_config to input here - memory_config is designed for the
        # GATHERED output shape (e.g., 128 height after gathering from 4 devices), not the
        # input shape (32 height before gathering).
        if input_tensor.dtype != cfg.ccl_dtype:
            input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG, cfg.ccl_dtype)

        if not sharded:
            input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

        if not use_composite:
            # Non-composite: all_gather followed by fast_reduce_nc (TTTv1 default)
            # TTTv1: outputs to memory_config for sharded case (designed for gathered tensor shape)
            gathered_tensor = ttnn.experimental.all_gather_async(
                input_tensor,
                persistent_output_buffer=None,
                dim=dim,
                multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                num_links=cfg.num_all_gather_links,
                cluster_axis=cluster_axis,
                topology=cfg.topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG if not sharded else memory_config,
                barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

            if sharded:
                # Convert to L1 interleaved for fast_reduce_nc (TTTv1 pattern)
                gathered_tensor = ttnn.to_memory_config(gathered_tensor, ttnn.L1_MEMORY_CONFIG)

            reduced_tensor = ttnn.experimental.fast_reduce_nc(
                gathered_tensor,
                dims=[dim],
                output=None,
                compute_kernel_config=None,
                memory_config=ttnn.L1_MEMORY_CONFIG if sharded else ttnn.DRAM_MEMORY_CONFIG,
            )

            gathered_tensor.deallocate(True)
        else:
            # Composite: reduce-scatter followed by all-gather
            input_mem_cfg = input_tensor.memory_config()

            reduced_tensor = ttnn.experimental.reduce_scatter_minimal_async(
                input_tensor,
                persistent_output_buffers=None,
                dim=dim,
                multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
                barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                num_links=cfg.num_reduce_scatter_links,
                cluster_axis=cluster_axis,
                memory_config=ttnn.DRAM_MEMORY_CONFIG if not sharded else memory_config,
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

        # Reshape to original shape
        reduced_tensor = ttnn.reshape(reduced_tensor, original_shape)

        # NOTE: TTTv1 does NOT apply memory_config to the reduced tensor.
        # The memory_config is designed for the GATHERED tensor (e.g., height 128 after gathering
        # 4 devices worth of data), not the REDUCED tensor (height 32 after reduction).
        # The reduced tensor stays in L1_MEMORY_CONFIG (interleaved) as returned by fast_reduce_nc.

        return reduced_tensor

    def _all_gather_axis1(self, tensor: ttnn.Tensor, dim: int, memory_config: Any) -> ttnn.Tensor:
        """
        All gather along cluster axis 1.

        Args:
            tensor: Input tensor to gather
            dim: Dimension to gather along
            memory_config: Output memory config for the gathered tensor
        """
        cfg = self.config
        cluster_axis = 1

        # Skip all_gather if only 1 device on this axis (matches TTTv1 tt_all_gather check)
        mesh_shape = list(cfg.mesh_device.shape)
        if mesh_shape == [1, 1] or (cluster_axis == 1 and 1 in mesh_shape):
            return tensor

        # Ensure tensor is in DRAM for all_gather (TTTv1 pattern)
        if tensor.memory_config().is_sharded():
            tensor = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)

        gathered = ttnn.experimental.all_gather_async(
            tensor,
            persistent_output_buffer=None,
            dim=dim,
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

        tensor.deallocate(True)
        return gathered

    # =========================================================================
    # TG-specific helper methods
    # =========================================================================

    def prefill_prepare_tensor_for_kv_cache(self, key_or_value_layer: ttnn.Tensor, user_id: int) -> ttnn.Tensor:
        """
        TG-specific: Prepare tensor for KV cache by selecting the correct column chips.

        For TG topology, we need to select every 4th tensor starting from user_id // batch_size_per_device_group.
        """
        cfg = self.config
        tensor_copy = ttnn.clone(key_or_value_layer)
        # Get all tensors from multi-device tensor
        tensors = ttnn.get_device_tensors(tensor_copy)
        # Get only tensors from specific column chips
        # Get every 4th tensor starting from user_id // batch_size_per_device_group
        single_column_tensors = tensors[user_id // cfg.batch_size_per_device_group :: 4]
        # Create multi-device tensor
        multi_device_tensor = ttnn.combine_device_tensors(tensors=single_column_tensors)

        return multi_device_tensor

    def _to_qk_fused_memory_config(
        self, q_tensor: ttnn.Tensor, k_tensor: ttnn.Tensor
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Convert Q and K tensors to height-sharded memory layouts for fused QK ops."""
        n_q_heads = q_tensor.shape[2]
        n_kv_heads = k_tensor.shape[2]
        q_batch = q_tensor.shape[1]

        row_size = 8
        k_start_core = ttnn.CoreCoord(q_batch % row_size, q_batch // row_size)

        q_core_grid = ttnn.CoreRangeSet({_num_to_corerange(q_batch)})
        k_core_grid = ttnn.CoreRangeSet({_num_to_corerange(q_batch, start_core=k_start_core)})

        q_mem_config = ttnn.create_sharded_memory_config(
            shape=(nearest_32(n_q_heads), self.config.head_dim),
            core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        k_mem_config = ttnn.create_sharded_memory_config(
            shape=(nearest_32(n_kv_heads), self.config.head_dim),
            core_grid=k_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        q_tensor = ttnn.to_memory_config(q_tensor, q_mem_config)
        k_tensor = ttnn.to_memory_config(k_tensor, k_mem_config)
        return q_tensor, k_tensor

    # =========================================================================
    # Factory method for backward compatibility
    # =========================================================================

    @classmethod
    def from_model_args(
        cls,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        weight_cache_path,
        layer_num: int,
        transformation_mats: dict[str, ttnn.Tensor],
        paged_attention_config=None,
        use_paged_kv_cache: bool = False,
    ):
        """Factory method for backward compatibility with ModelArgs."""
        from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

        # Attention2D requires Galaxy topology (4x8 or 8x4) due to Galaxy-specific CCL operations
        valid_shapes = [(4, 8), (8, 4)]
        shape_tuple = tuple(args.cluster_shape)
        if shape_tuple not in valid_shapes:
            raise ValueError(
                f"Attention2D requires Galaxy topology (4x8 or 8x4). Got cluster_shape={args.cluster_shape}. "
                "For non-Galaxy devices, use Attention1D instead."
            )

        configuration = args
        model_config = configuration.get_model_config()
        decoders_opt = model_config.get("DECODERS_OPTIMIZATIONS")

        layer_name = configuration.get_state_dict_prefix("Attention", layer_num)
        if configuration.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{layer_name}.{name}"

        wq_str = f"{layer_name}.wq"
        wk_str = f"{layer_name}.wk"
        wv_str = f"{layer_name}.wv"
        wo_str = f"{layer_name}.wo"
        q_norm_str = f"{layer_name}.q_norm"
        k_norm_str = f"{layer_name}.k_norm"

        # Get dtypes
        wqkv_dtype = decoders_opt.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.WQKV)
        wo_dtype = decoders_opt.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.WO)
        kv_cache_dtype = decoders_opt.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.KV_CACHE)
        activation_dtype = decoders_opt.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.ACTIVATION)

        # Get compute kernel configs
        li_qkv_decode_cfg = decoders_opt.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_QKV_DECODE, configuration=configuration
        )
        sdpa_decode_cfg = decoders_opt.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.SDPA_DECODE, configuration=configuration
        )
        li_o_decode_cfg = decoders_opt.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_O_DECODE, configuration=configuration
        )
        li_qkv_prefill_cfg = decoders_opt.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_QKV_PREFILL, configuration=configuration
        )
        sdpa_prefill_cfg = decoders_opt.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.SDPA_PREFILL, configuration=configuration
        )
        li_o_prefill_cfg = decoders_opt.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_O_PREFILL, configuration=configuration
        )

        # Build combined QKV weight
        num_devices_per_group = configuration.n_kv_heads  # For TG
        qkv_list = []
        for i in range(num_devices_per_group):
            wq_selected = torch.chunk(state_dict[f"{wq_str}.weight"], num_devices_per_group, dim=0)[i]
            wk_selected = torch.chunk(state_dict[f"{wk_str}.weight"], num_devices_per_group, dim=0)[i]
            wv_selected = torch.chunk(state_dict[f"{wv_str}.weight"], num_devices_per_group, dim=0)[i]

            wq = torch.transpose(wq_selected, -2, -1)
            wk = torch.transpose(wk_selected, -2, -1)
            wv = torch.transpose(wv_selected, -2, -1)

            qkv = torch.cat([wq, wk, wv], dim=-1)
            qkv_list.append(qkv)

        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

        # Create LazyWeights with 2D sharding for TG
        wqkv = LazyWeight(
            source=qkv_cat,
            dtype=wqkv_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper_config=ttnn.MeshMapperConfig(
                placements=[ttnn.PlacementShard(3), ttnn.PlacementShard(2)],  # TG: dims=(3, 2)
                mesh_shape_override=ttnn.MeshShape(configuration.cluster_shape),
            ),
            cache_dir_weight_name=(Path(weight_cache_path) / layer_name, "wqkv_sharded_2d")
            if weight_cache_path
            else None,
        )

        pt_wo = state_dict[f"{wo_str}.weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

        wo = LazyWeight(
            source=pt_wo,
            dtype=wo_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper_config=ttnn.MeshMapperConfig(
                placements=[ttnn.PlacementShard(2), ttnn.PlacementShard(3)],  # TG: dims=(2, 3)
                mesh_shape_override=ttnn.MeshShape(configuration.cluster_shape),
            ),
            cache_dir_weight_name=(Path(weight_cache_path) / layer_name, "wo_width_sharded_2d")
            if weight_cache_path
            else None,
        )

        # Q/K norm configs (optional)
        q_norm_config = None
        k_norm_config = None

        qk_norm_compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        if f"{q_norm_str}.weight" in state_dict:
            q_norm_torch = state_dict[f"{q_norm_str}.weight"]
            q_norm_torch = q_norm_torch.reshape(1, 1, -1, TILE_SIZE)

            q_norm_lazy = LazyWeight(
                source=q_norm_torch,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_dir_weight_name=(Path(weight_cache_path) / layer_name, "q_norm") if weight_cache_path else None,
            )

            q_norm_config = RMSNorm1DConfig(
                weight=q_norm_lazy,
                mesh_device=mesh_device,
                eps=configuration.norm_eps,
                add_unit_offset=configuration.rms_norm_add_unit_offset,
                decode_in_sharded=False,
                decode_out_sharded=False,
                prefill_distributed=False,
                compute_kernel_config=qk_norm_compute_kernel,
            )

        if f"{k_norm_str}.weight" in state_dict:
            k_norm_torch = state_dict[f"{k_norm_str}.weight"]
            k_norm_torch = k_norm_torch.reshape(1, 1, -1, TILE_SIZE)

            k_norm_lazy = LazyWeight(
                source=k_norm_torch,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_dir_weight_name=(Path(weight_cache_path) / layer_name, "k_norm") if weight_cache_path else None,
            )

            k_norm_config = RMSNorm1DConfig(
                weight=k_norm_lazy,
                mesh_device=mesh_device,
                eps=configuration.norm_eps,
                add_unit_offset=configuration.rms_norm_add_unit_offset,
                decode_in_sharded=False,
                decode_out_sharded=False,
                prefill_distributed=False,
                compute_kernel_config=qk_norm_compute_kernel,
            )

        # Handle QKV bias
        wqkv_bias = None
        if f"{wq_str}.bias" in state_dict:
            qkv_bias = torch.concat(
                [
                    torch.concat(
                        [
                            torch.chunk(state_dict[f"{wq_str}.bias"], configuration.num_devices)[i],
                            torch.chunk(state_dict[f"{wk_str}.bias"], configuration.num_devices)[i],
                            torch.chunk(state_dict[f"{wv_str}.bias"], configuration.num_devices)[i],
                        ],
                        dim=-1,
                    )
                    for i in range(configuration.num_devices)
                ],
                dim=-1,
            )
            wqkv_bias = qkv_bias

        # Determine scale
        if configuration.query_pre_attn_scalar is not None:
            scale = configuration.query_pre_attn_scalar**-0.5
        else:
            scale = configuration.head_dim**-0.5

        # Build config
        config = Attention2DConfig(
            wqkv=wqkv,
            wo=wo,
            q_norm_config=q_norm_config,
            k_norm_config=k_norm_config,
            wqkv_bias=wqkv_bias,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            topology=configuration.ccl_topology(),
            num_reduce_scatter_links=configuration.num_reduce_scatter_links,
            num_all_gather_links=configuration.num_all_gather_links,
            dim=configuration.dim,
            n_heads=configuration.n_heads,
            n_kv_heads=configuration.n_kv_heads,
            head_dim=configuration.head_dim,
            qkv_size=configuration.qkv_size,
            max_batch_size=configuration.max_batch_size,
            max_seq_len=configuration.max_seq_len,
            scale=scale,
            sliding_window=configuration.sliding_window if hasattr(configuration, "sliding_window") else None,
            use_qk_fused=getattr(configuration, "use_qk_fused", False),
            use_paged_kv_cache=use_paged_kv_cache,
            paged_attention_config=paged_attention_config,
            kv_cache_dtype=kv_cache_dtype,
            min_kv_prefill_shard_seqlen=configuration.min_kv_prefill_shard_seqlen,
            wqkv_dtype=wqkv_dtype,
            wo_dtype=wo_dtype,
            activation_dtype=activation_dtype,
            ccl_dtype=configuration.ccl_dtype,
            # TG-specific memory configs from model_config
            create_head_input_memcfg=model_config.get("CREATE_HEAD_INPUT_MEMCFG"),
            qkv_out_gathered_memcfg=model_config.get("QKV_OUT_GATHERED_MEMCFG"),
            gather_users_memcfg=model_config.get("GATHER_USERS_MEMCFG"),
            self_out_gathered_memcfg=model_config.get("SELF_OUT_GATHERED_MEMCFG"),
            self_out_reduce_scatter_memcfg=model_config.get("SELF_OUT_REDUCE_SCATTER_MEMCFG"),
            # Program configs
            decode_xqkv_prg_config=model_config.get("XQKV_DECODE_PROGCFG"),
            decode_sdpa_prg_config=model_config.get("SDPA_DECODE_PROGCFG"),
            decode_residual_memcfg=model_config.get("DECODE_RESIDUAL_MEMCFG"),
            decode_create_qkv_head_memcfg=model_config.get("CREATE_QKV_DECODE_SHARD"),
            decode_scores_memcfg=model_config.get("SCORES_BATCHED_MM_OUTPUT_MEMCFG"),
            prefill_xqkv_prg_config=model_config.get("XQKV_PREFILL_PROGCFG"),
            prefill_sdpa_prg_config=model_config.get("SDPA_PROGCFG"),
            prefill_wo_prg_config=model_config.get("WO_PREFILL_PROGCFG"),
            li_qkv_decode_compute_kernel_cfg=li_qkv_decode_cfg,
            sdpa_decode_compute_kernel_cfg=sdpa_decode_cfg,
            li_o_decode_compute_kernel_cfg=li_o_decode_cfg,
            li_qkv_prefill_compute_kernel_cfg=li_qkv_prefill_cfg,
            sdpa_prefill_compute_kernel_cfg=sdpa_prefill_cfg,
            li_o_prefill_compute_kernel_cfg=li_o_prefill_cfg,
        )

        instance = cls.from_config(config)

        # Initialize KV cache if not using paged attention
        if not use_paged_kv_cache:
            instance.init_kv_cache()

        return instance


# =============================================================================
# Config resolution
# =============================================================================


def _resolve_attention2d_config(config: Attention2DConfig) -> Attention2DConfig:
    """Materialize the config with sensible defaults for TG (Galaxy) topology."""
    to_set = {}

    # --- Phase 1: Validate 2D mesh ---

    mesh_device = config.mesh_device
    if mesh_device is None:
        mesh_device = config.wqkv.device
    if mesh_device is None:
        mesh_device = ttnn.GetDefaultDevice()
    if config.mesh_device is None:
        to_set["mesh_device"] = mesh_device

    assert mesh_device is not None, "mesh_device must be available!"

    cluster_shape = list(mesh_device.shape)
    # Attention2D requires 2D mesh topology
    assert cluster_shape[0] > 1 and cluster_shape[1] > 1, (
        f"Attention2D requires 2D mesh (both cluster_shape dimensions > 1). "
        f"Got cluster_shape={cluster_shape}. For 1D meshes, use Attention1D instead."
    )

    num_devices = mesh_device.get_num_devices()

    # --- Phase 2: Derive tt_ccl and topology ---

    if config.tt_ccl is None:
        to_set["tt_ccl"] = get_tt_ccl(mesh_device)

    if config.topology is None:
        to_set["topology"] = _default_topology(mesh_device)

    topology = to_set.get("topology", config.topology)

    # --- Phase 3: Model dimensions ---

    dim = config.dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    qkv_size = config.qkv_size

    # Try to infer from weight shapes
    if dim is None:
        wqkv_shape = config.wqkv.source.shape
        dim = wqkv_shape[-2] if len(wqkv_shape) == 4 else wqkv_shape[0]
        to_set["dim"] = dim

    if head_dim is None:
        head_dim = 128
        to_set["head_dim"] = head_dim

    if n_heads is None:
        n_heads = dim // head_dim
        to_set["n_heads"] = n_heads

    if n_kv_heads is None:
        n_kv_heads = n_heads
        to_set["n_kv_heads"] = n_kv_heads

    if qkv_size is None:
        qkv_size = head_dim * (2 * n_kv_heads + n_heads)
        to_set["qkv_size"] = qkv_size

    if config.scale is None:
        to_set["scale"] = head_dim**-0.5

    if config.min_kv_prefill_shard_seqlen is None:
        # TG uses different calculation
        to_set["min_kv_prefill_shard_seqlen"] = (TILE_SIZE * 8 * 8) // 1  # n_local_kv_heads = 1

    # --- Phase 4: TG-specific dimensions ---

    num_device_groups = num_devices // n_kv_heads
    to_set["num_device_groups"] = num_device_groups

    batch_size_per_device_group = max(config.max_batch_size // num_device_groups, 1)
    to_set["batch_size_per_device_group"] = batch_size_per_device_group

    n_local_heads = n_heads // n_kv_heads
    n_local_kv_heads = 1

    # --- Phase 5: Dtypes ---

    if config.wqkv_dtype is None:
        to_set["wqkv_dtype"] = ttnn.bfloat8_b
    if config.wo_dtype is None:
        to_set["wo_dtype"] = ttnn.bfloat8_b
    if config.activation_dtype is None:
        to_set["activation_dtype"] = ttnn.bfloat16
    if config.ccl_dtype is None:
        to_set["ccl_dtype"] = ttnn.bfloat8_b

    # --- Phase 6: Compute kernel configs ---

    compute_kernel_hifi2 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    compute_kernel_hifi4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    if config.li_qkv_decode_compute_kernel_cfg is None:
        to_set["li_qkv_decode_compute_kernel_cfg"] = compute_kernel_hifi2
    if config.sdpa_decode_compute_kernel_cfg is None:
        to_set["sdpa_decode_compute_kernel_cfg"] = compute_kernel_hifi2
    if config.li_o_decode_compute_kernel_cfg is None:
        to_set["li_o_decode_compute_kernel_cfg"] = compute_kernel_hifi2
    if config.li_qkv_prefill_compute_kernel_cfg is None:
        to_set["li_qkv_prefill_compute_kernel_cfg"] = compute_kernel_hifi2
    if config.sdpa_prefill_compute_kernel_cfg is None:
        to_set["sdpa_prefill_compute_kernel_cfg"] = compute_kernel_hifi4
    if config.li_o_prefill_compute_kernel_cfg is None:
        to_set["li_o_prefill_compute_kernel_cfg"] = compute_kernel_hifi2

    # --- Phase 7: Program configs ---

    tile_size = TILE_SIZE
    tile_padded_batch_rows = tile_size * math.ceil(config.max_batch_size / tile_size)

    # TG decode uses 1D program config (not DRAM-sharded)
    if config.decode_xqkv_prg_config is None:
        # Calculate lm_head_num_rows like TTTv1 does for TG
        # For 32 devices (TG), find largest lm_head_num_rows such that dim % (32 * 32 * lm_head_num_rows) == 0
        lm_head_num_rows = 4
        while dim % (32 * 32 * lm_head_num_rows) != 0 and lm_head_num_rows > 1:
            lm_head_num_rows -= 1

        # TTTv1 uses fixed values: in0_block_w=1, per_core_M=1, per_core_N=1 for non-70b/90b
        to_set["decode_xqkv_prg_config"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, lm_head_num_rows),
            in0_block_w=1,  # Fixed for non-70b/90b models
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,  # Fixed for TG decode
            per_core_N=1,  # Fixed for TG decode
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    if config.decode_sdpa_prg_config is None:
        to_set["decode_sdpa_prg_config"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )

    if config.decode_create_qkv_head_memcfg is None:
        to_set["decode_create_qkv_head_memcfg"] = (
            ttnn.create_sharded_memory_config(
                shape=(TILE_SIZE, head_dim),
                core_grid=ttnn.CoreGrid(y=4, x=8),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            if is_blackhole()
            else ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )

    if config.decode_scores_memcfg is None:

        def scores_memcfg(batch_size):
            return ttnn.create_sharded_memory_config(
                shape=(math.ceil(n_local_heads / 32) * 32, head_dim),
                core_grid=ttnn.CoreRangeSet({_num_to_corerange(batch_size)}),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        to_set["decode_scores_memcfg"] = scores_memcfg

    # Prefill configs
    if config.prefill_xqkv_prg_config is None:
        # For TG, WQKV weight is sharded across cluster_shape[0] devices on qkv_size dimension
        # (matches TTTv1's dims=(3, 2) sharding)
        per_device_qkv_size = qkv_size // cluster_shape[0]

        @lru_cache
        def xqkv_prefill_prg_config(seq_len: int):
            return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 10) if is_blackhole() else (8, 8),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=max(1, 8 if seq_len >= MAX_QKV_MM_SEQ_LEN else math.ceil(seq_len / tile_size / 8)),
                per_core_N=max(1, math.ceil(per_device_qkv_size / tile_size / 8)),
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=seq_len <= MAX_QKV_MM_SEQ_LEN,
            )

        to_set["prefill_xqkv_prg_config"] = xqkv_prefill_prg_config

    if config.prefill_sdpa_prg_config is None:

        @lru_cache
        def sdpa_prg_config(seq_len: int, chunk_start_idx: int | None):
            q_chunk = 256 if seq_len >= 2048 else 64
            k_chunk = 256 if seq_len >= 2048 else 64

            if chunk_start_idx is not None and chunk_start_idx != 0:
                q_chunk = min(q_chunk, chunk_start_idx & -chunk_start_idx)
                k_chunk = min(k_chunk, chunk_start_idx & -chunk_start_idx)

            return ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                exp_approx_mode=False,
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
            )

        to_set["prefill_sdpa_prg_config"] = sdpa_prg_config

    if config.prefill_wo_prg_config is None:
        # For TG, WO weight is sharded across cluster_shape[1] devices on output dim
        per_device_output_dim = dim // cluster_shape[1]

        @lru_cache
        def wo_prefill_prg_config(seq_len: int):
            num_rows = min(seq_len, 1024)
            return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=max(1, num_rows // tile_size // 8),
                per_core_N=max(1, math.ceil(per_device_output_dim / tile_size / 8)),
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=seq_len <= 1024,
            )

        to_set["prefill_wo_prg_config"] = wo_prefill_prg_config

    # --- Phase 8: Memory configs ---

    dim = to_set.get("dim", config.dim)

    # Calculate lm_head_num_rows for TG memory configs (same logic as program config)
    lm_head_num_rows = 4
    while dim % (32 * 32 * lm_head_num_rows) != 0 and lm_head_num_rows > 1:
        lm_head_num_rows -= 1

    if config.decode_input_memcfg is None:
        # TTTv1 SHARDED_ATTN_INPUT_MEMCFG for TG
        # shape = (32, nearest_32(dim // (8 * lm_head_num_rows) // 4))
        shard_width = nearest_32(dim // (8 * lm_head_num_rows) // 4) if lm_head_num_rows > 0 else 32
        to_set["decode_input_memcfg"] = ttnn.create_sharded_memory_config(
            shape=(32, shard_width),
            core_grid=ttnn.CoreGrid(y=lm_head_num_rows, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    if config.prefill_input_memcfg is None:
        to_set["prefill_input_memcfg"] = ttnn.DRAM_MEMORY_CONFIG

    # TG-specific memory configs (provide defaults if not set by from_model_args)
    if config.create_head_input_memcfg is None:
        to_set["create_head_input_memcfg"] = ttnn.L1_MEMORY_CONFIG

    if config.qkv_out_gathered_memcfg is None:
        # TTTv1 uses: shape=(32 * mesh_cols, 32), num_cores based on dim
        # dim=8192 -> 40 cores, dim=4096 -> 24 cores, dim=3072 -> 20 cores, else 12
        num_cores = 40 if dim == 8192 else (24 if dim == 4096 else (20 if dim == 3072 else 12))

        def qkv_out_gathered_memcfg(num_devices_axis1, _num_cores=num_cores):
            return ttnn.create_sharded_memory_config(
                shape=(32 * num_devices_axis1, 32),
                core_grid=_num_to_coregrid(_num_cores),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        to_set["qkv_out_gathered_memcfg"] = qkv_out_gathered_memcfg

    if config.gather_users_memcfg is None:
        # TTTv1: shape=(32 * mesh_cols, 64 * n_local_heads), num_cores=32
        n_local_heads = to_set.get("n_local_heads", n_heads // n_kv_heads)
        head_dim = to_set.get("head_dim", config.head_dim)

        def gather_users_memcfg(num_devices_axis1, _n_local_heads=n_local_heads, _head_dim=head_dim):
            return ttnn.create_sharded_memory_config(
                shape=(32 * num_devices_axis1, _head_dim * _n_local_heads),
                core_grid=_num_to_coregrid(32),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        to_set["gather_users_memcfg"] = gather_users_memcfg

    if config.self_out_gathered_memcfg is None:
        # TTTv1: shape=(32 * mesh_rows, dim // 4 // min(32, dim // 4 // 32)), num_cores=min(32, dim // 4 // 32)
        num_cores_self_out = min(32, dim // 4 // 32) if dim >= 128 else 1

        def self_out_gathered_memcfg(num_devices_axis0, _dim=dim, _num_cores=num_cores_self_out):
            return ttnn.create_sharded_memory_config(
                shape=(32 * num_devices_axis0, _dim // 4 // _num_cores),
                core_grid=_num_to_coregrid(_num_cores),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        to_set["self_out_gathered_memcfg"] = self_out_gathered_memcfg

    if config.self_out_reduce_scatter_memcfg is None:
        # Used for dim=8192 composite all-reduce case
        to_set["self_out_reduce_scatter_memcfg"] = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

    if config.decode_residual_memcfg is None:
        to_set["decode_residual_memcfg"] = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

    # --- Phase 9: Resolve LazyWeights with 2D sharding ---

    wqkv_dtype = to_set.get("wqkv_dtype", config.wqkv_dtype)
    wo_dtype = to_set.get("wo_dtype", config.wo_dtype)

    # TG uses DRAM interleaved for weights
    wqkv_mem_config = config.wqkv_memcfg or ttnn.DRAM_MEMORY_CONFIG
    wo_mem_config = config.wo_memcfg or ttnn.DRAM_MEMORY_CONFIG

    # For TG (e.g., 4x8 mesh), matching TTTv1's dims=(3, 2):
    # - mesh dim 0 (4 devices): shard tensor dim 3 (qkv_size 6144) → 1536 each
    # - mesh dim 1 (8 devices): shard tensor dim 2 (input dim 4096) → 512 each
    to_set["wqkv"] = resolve_lazy_weight(
        config.wqkv,
        device=mesh_device,
        memory_config=wqkv_mem_config,
        mesh_mapper_config=ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(3), ttnn.PlacementShard(2)],  # TG: dims=(3, 2) like TTTv1
            mesh_shape_override=ttnn.MeshShape(cluster_shape),
        ),
        layout=ttnn.TILE_LAYOUT,
        dtype=wqkv_dtype,
    )

    # For WO, matching TTTv1's dims=(2, 3):
    # - mesh dim 0 (4 devices): shard tensor dim 2 (input dim from attention)
    # - mesh dim 1 (8 devices): shard tensor dim 3 (output dim)
    to_set["wo"] = resolve_lazy_weight(
        config.wo,
        device=mesh_device,
        memory_config=wo_mem_config,
        mesh_mapper_config=ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(2), ttnn.PlacementShard(3)],  # TG: dims=(2, 3) like TTTv1
            mesh_shape_override=ttnn.MeshShape(cluster_shape),
        ),
        layout=ttnn.TILE_LAYOUT,
        dtype=wo_dtype,
    )

    # Resolve Q/K norm configs if present
    if config.q_norm_config is not None:
        q_norm_cfg = config.q_norm_config
        if q_norm_cfg.mesh_device is None:
            q_norm_cfg = replace(q_norm_cfg, mesh_device=mesh_device)
        to_set["q_norm_config"] = q_norm_cfg

    if config.k_norm_config is not None:
        k_norm_cfg = config.k_norm_config
        if k_norm_cfg.mesh_device is None:
            k_norm_cfg = replace(k_norm_cfg, mesh_device=mesh_device)
        to_set["k_norm_config"] = k_norm_cfg

    # --- Phase 10: Create TG-specific tensors ---

    # slice_mat: reduces batch from 32 to 8 for each device group
    weight = torch.zeros(1, 32, 8, 32)
    for i in range(32):
        col = i % 4  # This determines which group of 8 to select
        weight[:, i, :, col * 8 : (col + 1) * 8] = torch.eye(8)

    slice_mat = ttnn.from_torch(
        weight,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=1),
    )
    to_set["_slice_mat"] = slice_mat

    # user_selection_matrix: selects users for output
    user_selection_matrix = torch.eye(8, 8)
    user_selection_matrix = torch.nn.functional.pad(user_selection_matrix, (0, 24), "constant", 0)  # (8, 32)
    user_selection_matrix = [user_selection_matrix] * 4
    user_selection_matrix = torch.block_diag(*user_selection_matrix)  # (32, 128)
    user_selection_tensor = ttnn.from_torch(
        user_selection_matrix,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    to_set["_user_selection_matrix"] = user_selection_tensor

    # --- Phase 11: Handle QKV bias ---

    if config.wqkv_bias is not None:
        qkv_bias = config.wqkv_bias

        # Prefill bias
        wqkv_bias_prefill = ttnn.from_torch(
            qkv_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=-1),
        )
        wqkv_bias_prefill = ttnn.reshape(
            wqkv_bias_prefill,
            (1, 1, 1, wqkv_bias_prefill.shape[-1]),
            (1, 1, wqkv_bias_prefill.shape[-2], wqkv_bias_prefill.shape[-1]),
        )
        to_set["_wqkv_bias_prefill"] = wqkv_bias_prefill

        # Decode bias - one per batch size multiple
        wqkv_bias_decode = []
        for batch_size in range(tile_size, tile_padded_batch_rows + tile_size, tile_size):
            qkv_bias_decode = qkv_bias.unsqueeze(0).expand(batch_size, -1)
            bias_tensor = ttnn.from_torch(
                qkv_bias_decode,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=-1),
            )
            wqkv_bias_decode.append(bias_tensor)
        to_set["_wqkv_bias_decode"] = wqkv_bias_decode

    return replace(config, **to_set)


# =============================================================================
# Helper functions
# =============================================================================


def _default_topology(mesh_device: ttnn.MeshDevice) -> Optional[ttnn.Topology]:
    """Auto-detect CCL topology based on cluster type and device count."""
    num_devices = mesh_device.get_num_devices()
    if num_devices == 8 and ttnn.cluster.get_cluster_type() in [
        ttnn.cluster.ClusterType.T3K,
        ttnn.cluster.ClusterType.GALAXY,
    ]:
        return ttnn.Topology.Ring
    elif num_devices > 1:
        return ttnn.Topology.Linear
    return None


def _num_to_corerange(
    num_cores: int,
    start_core: ttnn.CoreCoord = None,
    grid_x: int = 8,
    grid_y: int = 8,
) -> ttnn.CoreRange:
    """
    Construct a rectangular CoreRange of exactly `num_cores` cores starting at `start_core`.

    The CoreRange is allocated in row-major order semantics but must form
    a single contiguous rectangle representable by `ttnn.CoreRange`.
    """
    if start_core is None:
        start_core = ttnn.CoreCoord(0, 0)

    if num_cores == 1:
        return ttnn.CoreRange(start_core, start_core)

    sx, sy = start_core.x, start_core.y

    # --- rectangular availability ---
    remaining_x = grid_x - sx
    remaining_y = grid_y - sy

    # --- choose rectangle dimensions ---
    num_x = min(num_cores, remaining_x)
    num_y = num_cores // num_x

    end_x = sx + num_x - 1
    end_y = sy + num_y - 1

    return ttnn.CoreRange(start_core, ttnn.CoreCoord(end_x, end_y))


def _num_to_coregrid(num_cores: int) -> ttnn.CoreGrid:
    """
    Convert number of cores to a CoreGrid.

    Maps to TTTv1's num_to_coregrid function.
    """
    if num_cores % 8 == 0:
        return ttnn.CoreGrid(y=num_cores // 8, x=8)
    if num_cores == 12:
        return ttnn.CoreGrid(y=2, x=6)
    if num_cores == 20:
        return ttnn.CoreGrid(y=4, x=5)
    # Fallback for other values
    if num_cores <= 8:
        return ttnn.CoreGrid(y=1, x=num_cores)
    # Try to find a rectangular grid
    for y in range(8, 0, -1):
        if num_cores % y == 0 and num_cores // y <= 8:
            return ttnn.CoreGrid(y=y, x=num_cores // y)
    # Last resort
    return ttnn.CoreGrid(y=1, x=min(num_cores, 8))


def _zeros_like_kv_cache(batch_size: int, n_kv_heads: int, max_seq_len: int, head_dim: int) -> torch.Tensor:
    """Create zeros tensor for standard KV cache."""
    return torch.zeros((batch_size, n_kv_heads, max_seq_len, head_dim))


def _zeros_like_paged_cache(paged_config, n_kv_heads: int, head_dim: int) -> torch.Tensor:
    """Create zeros tensor for paged KV cache."""
    return torch.zeros((paged_config.max_num_blocks, n_kv_heads, paged_config.block_size, head_dim))


def _load_input_device_tensor(x: ttnn.Tensor | LazyWeight, config: Attention2DConfig, mode: str) -> ttnn.Tensor:
    """Resolve input tensor to ttnn.Tensor if LazyWeight, otherwise return as-is."""
    assert mode in ("decode", "prefill"), f"mode must be 'decode' or 'prefill', got {mode}"

    if isinstance(x, LazyWeight):
        mem_cfg = config.decode_input_memcfg if mode == "decode" else config.prefill_input_memcfg
        cluster_shape = list(config.mesh_device.shape)
        # For Attention2D, input is sharded on dim 3 on axis 1, replicated on axis 0
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

    assert isinstance(x, ttnn.Tensor), f"x must be ttnn.Tensor or LazyWeight, got {type(x)}"
    return x
