# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2-style Attention module for 1D-topology devices: N150 (1x1), N300 (1x2), T3K (1x8).

Single unified Attention1D class with separate forward methods:
  - decode_forward(): For decode mode (single token per user)
  - prefill_forward(): For prefill mode (multiple tokens)
  - forward(x, mode, **kwargs): Dispatcher that calls the appropriate method

Execution paths:
  Decode:  QKV matmul → all_reduce → create_qkv_heads → rotary → KV cache update → SDPA → concat_heads → WO matmul → all_reduce
  Prefill: [reshape] → QKV matmul → all_reduce → create_qkv_heads → rotary → KV cache fill → SDPA → concat_heads → [reshape] → all_gather → WO matmul

Key design decisions:
  - No static branching on topology in forward() - TG excluded at module level
  - Paged vs non-paged KV cache selected at runtime based on page_table presence
  - Fused all-gather matmul used for Ring topology (T3K 1x8) when dimensions allow
"""

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

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
# Attention1DConfig dataclass
# =============================================================================


@dataclass
class Attention1DConfig:
    """
    Central configuration for Attention1D - the single source of truth for all settings.

    Simple usage (all defaults):
        config = Attention1DConfig(wqkv, wo)

    Override any field:
        config = Attention1DConfig(wqkv, wo, n_heads=32, head_dim=128)

    Full customization:
        config = Attention1DConfig(
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

    # Decode program configs
    decode_input_memcfg: ttnn.MemoryConfig | None = None
    decode_xqkv_prg_config: "ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig | None" = None
    decode_sdpa_prg_config: ttnn.SDPAProgramConfig | None = None
    decode_attn_output_prg_config: "ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig | None" = None
    decode_residual_memcfg: ttnn.MemoryConfig | None = None
    decode_create_qkv_head_memcfg: ttnn.MemoryConfig | None = None
    decode_scores_memcfg: Callable[[int], ttnn.MemoryConfig] | None = None

    # Prefill program configs
    prefill_input_memcfg: ttnn.MemoryConfig | None = None
    prefill_xqkv_prg_config: Callable[[int], ttnn.MatmulMultiCoreReuseMultiCastProgramConfig] | None = None
    prefill_sdpa_prg_config: Callable[[int, int | None], ttnn.SDPAProgramConfig] | None = None
    prefill_wo_prg_config: Callable[[int], ttnn.MatmulMultiCoreReuseMultiCastProgramConfig] | None = None
    prefill_kv_memcfg: Callable[[int], ttnn.MemoryConfig] | None = None

    # Fused all-gather matmul (Ring topology)
    use_fused_all_gather_matmul: bool | None = None  # None = auto-detect
    decode_all_gather_matmul_prg_config: "ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None" = None
    decode_all_gather_matmul_memcfg: ttnn.MemoryConfig | None = None

    # Compute kernel configs
    li_qkv_decode_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    sdpa_decode_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    li_o_decode_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    li_qkv_prefill_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    sdpa_prefill_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    li_o_prefill_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None

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
            "decode_xqkv_prg_config",  # Required for all 1D (DRAM-sharded matmul)
            "decode_attn_output_prg_config",  # Required for all 1D (DRAM-sharded matmul)
            "decode_sdpa_prg_config",
            "decode_residual_memcfg",
            "prefill_xqkv_prg_config",
            "prefill_sdpa_prg_config",
            "prefill_wo_prg_config",
            "li_qkv_decode_compute_kernel_cfg",
            "sdpa_decode_compute_kernel_cfg",
            "li_o_decode_compute_kernel_cfg",
        ]

        # Multi-device needs CCL and topology
        if self.mesh_device and self.mesh_device.get_num_devices() > 1:
            required.extend(["tt_ccl", "topology"])

        return all(getattr(self, f) is not None for f in required)


# =============================================================================
# Attention1D Class
# =============================================================================


class Attention1D(LightweightModule):
    """
    Attention for 1D mesh topologies (N150, N300, T3K).

    Simple API (90% of users):
        attn = Attention1D(wqkv, wo)

    Power API (10% of users) - any level of customization via config:
        config = Attention1DConfig(wqkv, wo, n_heads=32, head_dim=128)
        attn = Attention1D.from_config(config)

    Execution paths:
      Decode:  QKV matmul → all_reduce → create_qkv_heads → rotary → KV cache → SDPA → concat_heads → WO matmul → all_reduce
      Prefill: [reshape] → QKV matmul → all_reduce → create_qkv_heads → rotary → KV cache → SDPA → concat_heads → WO matmul
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
        # todo)) this may not be needed anymore; maybe provide a simple API that is a little bigger surface than just the weights?
        super().__init__()
        self.config = _resolve_attention1d_config(Attention1DConfig(wqkv=wqkv, wo=wo))
        self._device_weights_loaded = False
        self.layer_past = None  # KV cache for non-paged mode

    @classmethod
    def from_config(cls, config: Attention1DConfig):
        """
        Power API for 10% of users - any level of customization via config.

        Override any subset of fields in Attention1DConfig:
            config = Attention1DConfig(wqkv, wo, n_heads=32, head_dim=128)
            attn = Attention1D.from_config(config)
        """
        instance = object.__new__(cls)
        super(Attention1D, instance).__init__()
        instance.config = _resolve_attention1d_config(config)
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

        if cfg.paged_attention_config:
            # Paged attention - external cache
            cache_k = _zeros_like_paged_cache(
                cfg.paged_attention_config,
                cfg.n_kv_heads // cfg.mesh_device.get_num_devices(),
                cfg.head_dim,
            )
            cache_v = _zeros_like_paged_cache(
                cfg.paged_attention_config,
                cfg.n_kv_heads // cfg.mesh_device.get_num_devices(),
                cfg.head_dim,
            )
        else:
            # Standard cache
            n_local_kv_heads = cfg.n_kv_heads // cfg.mesh_device.get_num_devices()
            cache_k = _zeros_like_kv_cache(
                cfg.max_batch_size,
                n_local_kv_heads,
                cfg.max_seq_len,
                cfg.head_dim,
            )
            cache_v = _zeros_like_kv_cache(
                cfg.max_batch_size,
                n_local_kv_heads,
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
        Decode forward - single token per user.

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

        num_devices = cfg.mesh_device.get_num_devices()
        n_local_heads = cfg.n_heads // num_devices
        n_local_kv_heads = cfg.n_kv_heads // num_devices

        # --- STAGE 1: QKV Matmul ---
        # All 1D topologies use DRAM-sharded matmul with L1_WIDTH_SHARDED output (matches TTTv1)
        xqkv_fused_sharded = ttnn.linear(
            x,
            self.wqkv,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=cfg.decode_xqkv_prg_config,
            compute_kernel_config=cfg.li_qkv_decode_compute_kernel_cfg,
            dtype=cfg.activation_dtype or ttnn.bfloat16,
        )

        # Add bias if present
        if self.wqkv_bias_decode:
            num_tiles = int(math.ceil(xqkv_fused_sharded.shape[-2] / TILE_SIZE))
            xqkv_fused_sharded = xqkv_fused_sharded + self.wqkv_bias_decode[num_tiles - 1]

        ttnn.deallocate(x)

        # --- STAGE 2: Convert QKV from sharded to interleaved (matches TTTv1) ---
        xqkv_fused = self._all_reduce_qkv_decode(xqkv_fused_sharded)
        ttnn.deallocate(xqkv_fused_sharded)

        # Reshape for create_qkv_heads
        fqkv_shape = xqkv_fused.shape
        xqkv_fused = ttnn.reshape(xqkv_fused, (1, 1, cfg.max_batch_size, fqkv_shape[3]), (1, 1, 32, fqkv_shape[3]))

        # --- STAGE 3: Create QKV Heads ---
        q_heads_pre_rot, k_heads_pre_rot, v_heads = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=n_local_heads,
            num_kv_heads=n_local_kv_heads,
            memory_config=cfg.decode_create_qkv_head_memcfg,
        )
        ttnn.deallocate(xqkv_fused)

        # --- STAGE 4: Q/K Normalization (optional) ---
        # Workaround: RMSNorm doesn't support HEIGHT_SHARDED inputs (TTTv1 norm_reshard pattern)
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

        # --- STAGE 5: Rotary Embedding ---
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

        # --- STAGE 6: KV Cache Update ---
        if kv_cache:
            keys, values = kv_cache[0], kv_cache[1]
        else:
            keys, values = self.layer_past[0], self.layer_past[1]

        # todo)) always use fused qk if it is better! --> does this work for all our models?
        # if it does not cover all models, we can make a separate module for fused --> maybe experimental?
        if cfg.use_qk_fused:
            ttnn.experimental.paged_fused_update_cache(
                keys, k_heads, values, v_heads, update_idxs_tensor=current_pos, page_table=page_table
            )
        else:
            ttnn.experimental.paged_update_cache(keys, k_heads, update_idxs_tensor=current_pos, page_table=page_table)
            ttnn.experimental.paged_update_cache(values, v_heads, update_idxs_tensor=current_pos, page_table=page_table)

        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        # --- STAGE 7: SDPA ---
        # todo)) prefer paged but do want to test both! --> PagedAttention1D and Attention1D
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

        # --- STAGE 8: Reshape for concat heads ---
        attn_output_sharded = ttnn.to_memory_config(
            attn_output, memory_config=cfg.decode_scores_memcfg(cfg.max_batch_size)
        )

        # --- STAGE 9: Concat Heads ---
        attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(attn_output_sharded, num_heads=n_local_heads)
        ttnn.deallocate(attn_output_sharded)
        ttnn.deallocate(attn_output)

        # --- STAGE 10: All-Gather + WO Matmul ---
        if cfg.use_fused_all_gather_matmul and cfg.topology == ttnn.Topology.Ring:
            dense_out = self._fused_all_gather_wo_decode(attn_output_cat)
        else:
            dense_out = self._separate_all_gather_wo_decode(attn_output_cat)

        ttnn.deallocate(attn_output_cat)

        # --- STAGE 11: Final All-Reduce ---
        dense_out_reduced = self._all_reduce_output_decode(dense_out)

        # --- STAGE 12: Final Memory Config ---
        dense_out_reduced = ttnn.to_memory_config(dense_out_reduced, cfg.decode_residual_memcfg)

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
        Prefill forward - multiple tokens.

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

        num_devices = cfg.mesh_device.get_num_devices()
        n_local_heads = cfg.n_heads // num_devices
        n_local_kv_heads = cfg.n_kv_heads // num_devices

        # todo)) maybe we can make each stage into a function and then allow concrete classes to override them?
        # q/k_norm could be in base
        # page should its own thing
        # llama optimized attentions --> fused qk and fused all-gather matmul
        # --- STAGE 1: Reshape for long sequences ---
        if seq_len > MAX_QKV_MM_SEQ_LEN:
            if seq_len % MAX_QKV_MM_SEQ_LEN != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {MAX_QKV_MM_SEQ_LEN}")
            x = ttnn.reshape(x, [1, seq_len // MAX_QKV_MM_SEQ_LEN, MAX_QKV_MM_SEQ_LEN, -1])

        # --- STAGE 2: QKV Matmul ---
        xqkv_fused = ttnn.linear(
            x,
            self.wqkv,
            dtype=cfg.activation_dtype or ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=cfg.li_qkv_prefill_compute_kernel_cfg,
            program_config=cfg.prefill_xqkv_prg_config(seq_len),
        )

        # Add bias if present
        if self.wqkv_bias_prefill is not None:
            xqkv_fused = xqkv_fused + self.wqkv_bias_prefill

        # --- STAGE 3: All-Reduce QKV ---
        xqkv_fused = self._all_reduce_qkv_prefill(xqkv_fused)

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
        # todo)) maybe typecast is already a no-op when source == target types
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

        # --- STAGE 8: Shard KV for long sequences ---
        if seq_len >= cfg.min_kv_prefill_shard_seqlen and page_table is None:
            k_fill = ttnn.interleaved_to_sharded(k_heads_cache_dtype, cfg.prefill_kv_memcfg(seq_len))
            v_fill = ttnn.interleaved_to_sharded(v_heads_cache_dtype, cfg.prefill_kv_memcfg(seq_len))
        else:
            k_fill = k_heads_cache_dtype
            v_fill = v_heads_cache_dtype

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
            ttnn.fill_cache(keys, k_fill, user_id % cfg.max_batch_size)
            ttnn.fill_cache(values, v_fill, user_id % cfg.max_batch_size)

        if seq_len >= cfg.min_kv_prefill_shard_seqlen and page_table is None:
            ttnn.deallocate(k_fill)
            ttnn.deallocate(v_fill)

        # --- STAGE 10: SDPA ---
        q_heads_sdpa = ttnn.typecast(q_heads, dtype=cfg.activation_dtype or ttnn.bfloat8_b)
        ttnn.deallocate(q_heads)

        # todo)) similar to paged versus non-paged; there may be some limitations around chunked.
        # we probably do not want all the combindations of paged, chunked, non-paged, and non-chunked;  --> do a deep dive on this!
        # for example, chunked and paged is normal combo! in fact, chunked requires paged! but paged does not require chunked!
        # --> can we just use chunked with a single chunk to run the else case here?
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

        # --- STAGE 13: All-Gather for Ring topology ---
        if cfg.use_fused_all_gather_matmul:
            attn_output_concat = ttnn.experimental.all_gather_async(
                attn_output_concat,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=1,
                topology=cfg.topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

        # --- STAGE 14: WO Matmul ---
        output = ttnn.linear(
            attn_output_concat,
            self.wo,
            compute_kernel_config=cfg.li_o_prefill_compute_kernel_cfg,
            dtype=cfg.activation_dtype or ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=cfg.prefill_wo_prg_config(seq_len),
        )

        # --- STAGE 15: Reshape back ---
        if seq_len > 1024:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])

        ttnn.deallocate(attn_output_concat)

        # --- STAGE 16: All-Reduce output (when not using fused all-gather) ---
        if not cfg.use_fused_all_gather_matmul:
            output = self._all_reduce_output_prefill(output)

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
    # Internal CCL methods
    # =========================================================================

    def _all_reduce_qkv_decode(self, xqkv: ttnn.Tensor) -> ttnn.Tensor:
        """
        All-reduce QKV for decode mode.

        For 1D topologies, tt_all_reduce with cluster_axis=1 returns input as-is (no reduction).
        We just need to convert from sharded to interleaved (matches TTTv1 non-TG path).
        """
        # All 1D topologies: convert L1_WIDTH_SHARDED to L1 interleaved (matches TTTv1)
        # bfloat16 is required by nlp_create_qkv_heads_decode
        return ttnn.sharded_to_interleaved(xqkv, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)

    def _all_reduce_qkv_prefill(self, xqkv: ttnn.Tensor) -> ttnn.Tensor:
        """All-reduce QKV for prefill mode (no-op for 1D topology)."""
        # For 1D non-TG topology, no cluster_axis=1 all-reduce needed
        return xqkv

    def _all_reduce_output_decode(self, output: ttnn.Tensor) -> ttnn.Tensor:
        """Final all-reduce for decode output."""
        cfg = self.config
        if cfg.mesh_device.get_num_devices() == 1:
            return output

        # Reshape to (1, 1, batch * heads, dim)
        original_shape = output.shape
        if original_shape[0] != 1 or original_shape[1] != 1:
            output = ttnn.reshape(
                output, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
            )

        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            output,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=cfg.num_reduce_scatter_links,
            memory_config=cfg.decode_residual_memcfg,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=cfg.topology,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        output.deallocate(True)
        return reduced

    def _all_reduce_output_prefill(self, output: ttnn.Tensor) -> ttnn.Tensor:
        """Final all-reduce for prefill output."""
        cfg = self.config
        if cfg.mesh_device.get_num_devices() == 1:
            return output

        original_shape = output.shape
        if original_shape[0] != 1 or original_shape[1] != 1:
            output = ttnn.reshape(
                output, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
            )

        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            output,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=cfg.num_reduce_scatter_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=cfg.topology,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        output.deallocate(True)
        return reduced

    def _fused_all_gather_wo_decode(self, attn_output_cat: ttnn.Tensor) -> ttnn.Tensor:
        """Fused all-gather matmul for Ring topology."""
        cfg = self.config

        attn_output_cat = ttnn.to_memory_config(attn_output_cat, cfg.decode_all_gather_matmul_memcfg)

        _, dense_out = ttnn.experimental.all_gather_matmul_async(
            attn_output_cat,
            self.wo,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            all_gather_core_grid_offset=(0, 4),
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=1,
            memory_config_ag=cfg.decode_all_gather_matmul_memcfg,
            memory_config_mm=cfg.decode_residual_memcfg,
            program_config=cfg.decode_all_gather_matmul_prg_config,
            compute_kernel_config=cfg.li_o_decode_compute_kernel_cfg,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        return ttnn.to_memory_config(dense_out, cfg.decode_residual_memcfg)

    def _separate_all_gather_wo_decode(self, attn_output_cat: ttnn.Tensor) -> ttnn.Tensor:
        """Separate all-gather then WO matmul (non-Ring or when fused not available)."""
        cfg = self.config

        # WO matmul
        dense_out = ttnn.linear(
            attn_output_cat,
            self.wo,
            program_config=cfg.decode_attn_output_prg_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=cfg.li_o_decode_compute_kernel_cfg,
        )

        return dense_out

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
        import torch

        from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

        if args.is_galaxy:
            raise ValueError("Attention1D cannot be used for Galaxy devices. Use Attention2D instead.")

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
        num_devices = mesh_device.get_num_devices()
        qkv_list = []
        for i in range(num_devices):
            wq_selected = torch.chunk(state_dict[f"{wq_str}.weight"], num_devices, dim=0)[i]
            wk_selected = torch.chunk(state_dict[f"{wk_str}.weight"], num_devices, dim=0)[i]
            wv_selected = torch.chunk(state_dict[f"{wv_str}.weight"], num_devices, dim=0)[i]

            wq = torch.transpose(wq_selected, -2, -1)
            wk = torch.transpose(wk_selected, -2, -1)
            wv = torch.transpose(wv_selected, -2, -1)

            qkv = torch.cat([wq, wk, wv], dim=-1)
            qkv_list.append(qkv)

        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

        # Create LazyWeights
        wqkv_mem_config = configuration.create_dram_sharded_mem_config(
            configuration.dim, configuration.qkv_size // num_devices
        )

        wqkv = LazyWeight(
            source=qkv_cat,
            dtype=wqkv_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=wqkv_mem_config,
            mesh_mapper_config=ttnn.MeshMapperConfig(
                placements=[ttnn.PlacementShard(-1)],
                mesh_shape_override=ttnn.MeshShape([num_devices]),
            ),
            cache_dir_weight_name=(Path(weight_cache_path) / layer_name, "wqkv_sharded") if weight_cache_path else None,
        )

        pt_wo = state_dict[f"{wo_str}.weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)
        wo_mem_config = configuration.create_dram_sharded_mem_config(
            (configuration.n_heads * configuration.head_dim) // num_devices, configuration.dim
        )

        use_fused_all_gather_matmul = model_config.get("USE_FUSED_ALL_GATHER_MATMUL", False)

        wo = LazyWeight(
            source=pt_wo,
            dtype=wo_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if use_fused_all_gather_matmul else wo_mem_config,
            mesh_mapper_config=ttnn.MeshMapperConfig(
                # For fused: shard on dim -1 (width) - matches TTTv1 dims=(2, 3)
                # For non-fused: shard on dim -2 (height) - matches TTTv1 dims=(3, 2)
                placements=[ttnn.PlacementShard(-1 if use_fused_all_gather_matmul else -2)],
                mesh_shape_override=ttnn.MeshShape([num_devices]),
            ),
            cache_dir_weight_name=(
                Path(weight_cache_path) / layer_name,
                "wo_width_sharded" if use_fused_all_gather_matmul else "wo",
            )
            if weight_cache_path
            else None,
        )

        # Q/K norm configs (optional) - using RMSNorm1DConfig composition pattern
        q_norm_config = None
        k_norm_config = None

        # Get compute kernel config for Q/K normalization
        # Q/K norm uses interleaved path (not sharded like regular decode RMSNorm)
        qk_norm_compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        if f"{q_norm_str}.weight" in state_dict:
            q_norm_torch = state_dict[f"{q_norm_str}.weight"]
            # Note: add_unit_offset is handled by RMSNorm1DConfig resolution
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
                decode_in_sharded=False,  # Q/K heads are interleaved after create_qkv_heads
                decode_out_sharded=False,
                prefill_distributed=False,  # Q/K norm doesn't need distributed prefill
                compute_kernel_config=qk_norm_compute_kernel,
            )

        if f"{k_norm_str}.weight" in state_dict:
            k_norm_torch = state_dict[f"{k_norm_str}.weight"]
            # Note: add_unit_offset is handled by RMSNorm1DConfig resolution
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
                decode_in_sharded=False,  # Q/K heads are interleaved after create_qkv_heads
                decode_out_sharded=False,
                prefill_distributed=False,  # Q/K norm doesn't need distributed prefill
                compute_kernel_config=qk_norm_compute_kernel,
            )

        # Handle QKV bias
        wqkv_bias = None
        if f"{wq_str}.bias" in state_dict:
            qkv_bias = torch.concat(
                [
                    torch.concat(
                        [
                            torch.chunk(state_dict[f"{wq_str}.bias"], num_devices)[i],
                            torch.chunk(state_dict[f"{wk_str}.bias"], num_devices)[i],
                            torch.chunk(state_dict[f"{wv_str}.bias"], num_devices)[i],
                        ],
                        dim=-1,
                    )
                    for i in range(num_devices)
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
        config = Attention1DConfig(
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
            decode_xqkv_prg_config=model_config.get("XQKV_DECODE_PROGCFG"),
            decode_sdpa_prg_config=model_config.get("SDPA_DECODE_PROGCFG"),
            decode_attn_output_prg_config=model_config.get("ATTN_OUTPUT_PROGCFG"),
            decode_residual_memcfg=model_config.get("DECODE_RESIDUAL_MEMCFG"),
            decode_create_qkv_head_memcfg=model_config.get("CREATE_QKV_DECODE_SHARD"),
            decode_scores_memcfg=model_config.get("SCORES_BATCHED_MM_OUTPUT_MEMCFG"),
            prefill_xqkv_prg_config=model_config.get("XQKV_PREFILL_PROGCFG"),
            prefill_sdpa_prg_config=model_config.get("SDPA_PROGCFG"),
            prefill_wo_prg_config=model_config.get("WO_PREFILL_PROGCFG"),
            prefill_kv_memcfg=model_config.get("KV_PREFILL_MEM_CFG"),
            use_fused_all_gather_matmul=use_fused_all_gather_matmul,
            decode_all_gather_matmul_prg_config=model_config.get("ATTN_ALL_GATHER_MATMUL_PROGCFG"),
            decode_all_gather_matmul_memcfg=model_config.get("ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"),
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


def _resolve_attention1d_config(config: Attention1DConfig) -> Attention1DConfig:
    """Materialize the config with sensible defaults."""
    to_set = {}

    # --- Phase 1: Foundational fields ---

    # Derive mesh_device
    mesh_device = config.mesh_device
    if mesh_device is None:
        mesh_device = config.wqkv.device
    if mesh_device is None:
        mesh_device = ttnn.GetDefaultDevice()
    if config.mesh_device is None:
        to_set["mesh_device"] = mesh_device

    assert mesh_device is not None, "mesh_device must be available!"

    num_devices = mesh_device.get_num_devices()

    # Derive tt_ccl
    if config.tt_ccl is None and num_devices > 1:
        to_set["tt_ccl"] = get_tt_ccl(mesh_device)

    # Auto-detect topology
    if config.topology is None:
        to_set["topology"] = _default_topology(mesh_device)

    topology = to_set.get("topology", config.topology)

    # --- Phase 2: Model dimensions ---

    # These must be provided or inferred
    dim = config.dim
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    qkv_size = config.qkv_size

    # Try to infer from weight shapes
    if dim is None:
        # wqkv shape is (1, 1, dim, qkv_size_per_device * num_devices) or (dim, qkv_size)
        wqkv_shape = config.wqkv.source.shape
        dim = wqkv_shape[-2] if len(wqkv_shape) == 4 else wqkv_shape[0]
        to_set["dim"] = dim

    if head_dim is None:
        # Default head_dim for common models
        head_dim = 128
        to_set["head_dim"] = head_dim

    if n_heads is None:
        # Can't reliably infer n_heads, use common default
        n_heads = dim // head_dim
        to_set["n_heads"] = n_heads

    if n_kv_heads is None:
        # Assume GQA with same kv_heads as heads
        n_kv_heads = n_heads
        to_set["n_kv_heads"] = n_kv_heads

    if qkv_size is None:
        qkv_size = head_dim * (2 * n_kv_heads + n_heads)
        to_set["qkv_size"] = qkv_size

    if config.scale is None:
        to_set["scale"] = head_dim**-0.5

    if config.min_kv_prefill_shard_seqlen is None:
        to_set["min_kv_prefill_shard_seqlen"] = (TILE_SIZE * 8 * 8) // (n_kv_heads // num_devices)

    # --- Phase 3: Dtypes ---

    if config.wqkv_dtype is None:
        to_set["wqkv_dtype"] = ttnn.bfloat8_b
    if config.wo_dtype is None:
        to_set["wo_dtype"] = ttnn.bfloat8_b
    if config.activation_dtype is None:
        to_set["activation_dtype"] = ttnn.bfloat16

    # --- Phase 4: Compute kernel configs ---

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

    # --- Phase 5: Program configs ---

    tile_size = TILE_SIZE
    tile_padded_batch_rows = tile_size * math.ceil(config.max_batch_size / tile_size)
    n_local_heads = n_heads // num_devices

    # Decode configs - use DRAM-sharded matmul for all 1D topologies (matches TTTv1)
    if config.decode_xqkv_prg_config is None:
        attn_input_grid = _dram_shard_core_grid(dim)
        to_set["decode_xqkv_prg_config"] = _dram_matmul_config(
            m=tile_padded_batch_rows,
            k=dim,
            n=qkv_size // num_devices,
            num_cores=attn_input_grid.num_cores,
        )

    if config.decode_sdpa_prg_config is None:
        to_set["decode_sdpa_prg_config"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )

    if config.decode_attn_output_prg_config is None:
        # Use DRAM-sharded matmul for all 1D topologies (matches TTTv1)
        to_set["decode_attn_output_prg_config"] = _dram_matmul_config(
            m=tile_padded_batch_rows,
            k=(n_heads * head_dim) // num_devices,
            n=dim,
            num_cores=n_local_heads,
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
        dram_shard_grid_width = 8

        @lru_cache
        def xqkv_prefill_prg_config(seq_len: int):
            return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 10) if is_blackhole() else (8, 8),
                in0_block_w=1,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=max(1, 8 if seq_len >= MAX_QKV_MM_SEQ_LEN else math.ceil(seq_len / tile_size / 8)),
                per_core_N=math.ceil(qkv_size / num_devices / 32 / dram_shard_grid_width),
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
        use_fused = to_set.get("use_fused_all_gather_matmul", config.use_fused_all_gather_matmul)
        if use_fused is None:
            use_fused = (
                num_devices == 8
                and (dim // tile_size // num_devices) % num_devices == 0
                and num_devices > 1
                and topology == ttnn.Topology.Ring
            )
            to_set["use_fused_all_gather_matmul"] = use_fused

        k_dim = (n_heads * head_dim) // num_devices
        n_dim = 1024 if use_fused and 1024 % (dim // num_devices) == 0 else dim
        dram_shard_grid_width = 8
        prefill_rows = 8

        @lru_cache
        def wo_prefill_prg_config(seq_len: int):
            num_rows = min(seq_len, 1024)
            grid_size = _find_prefill_grid(prefill_rows, k_dim // tile_size)
            return _matmul_config(
                m=num_rows,
                k=k_dim,
                n=n_dim,
                grid_size=grid_size,
                in0_block_w=1,
                fuse_batch=seq_len <= 1024,
                per_core_n=math.ceil(n_dim / (tile_size * dram_shard_grid_width)) if not use_fused else None,
            )

        to_set["prefill_wo_prg_config"] = wo_prefill_prg_config

    if config.prefill_kv_memcfg is None:
        n_local_kv_heads = n_kv_heads // num_devices

        @lru_cache
        def kv_prefill_memcfg(seq_len: int):
            # Shard the combined (n_kv_heads * seq_len) dimension across 64 cores
            # This ensures tile-aligned shard shapes (multiples of 32)
            num_cores = 8 * 8  # 64 cores
            shard_height = (n_local_kv_heads * seq_len) // num_cores
            return ttnn.create_sharded_memory_config(
                shape=(shard_height, head_dim),
                core_grid=ttnn.CoreGrid(y=8, x=8),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        to_set["prefill_kv_memcfg"] = kv_prefill_memcfg

    # Fused all-gather matmul configs
    if config.use_fused_all_gather_matmul is None:
        use_fused = (
            num_devices == 8
            and (dim // tile_size // num_devices) % num_devices == 0
            and num_devices > 1
            and topology == ttnn.Topology.Ring
        )
        to_set["use_fused_all_gather_matmul"] = use_fused

    use_fused = to_set.get("use_fused_all_gather_matmul", config.use_fused_all_gather_matmul)

    if use_fused and config.decode_all_gather_matmul_prg_config is None:
        do_core_grid_size = (8, 1)
        do_per_core_N = dim // num_devices // tile_size // (do_core_grid_size[0] * do_core_grid_size[1])

        to_set["decode_all_gather_matmul_prg_config"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=do_core_grid_size,
            in0_block_w=dim // tile_size // (do_core_grid_size[0] * do_core_grid_size[1]),
            out_subblock_h=1,
            out_subblock_w=_get_out_subblock_w(do_per_core_N, out_subblock_h=1),
            per_core_M=tile_padded_batch_rows // tile_size,
            per_core_N=do_per_core_N,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    if use_fused and config.decode_all_gather_matmul_memcfg is None:
        to_set["decode_all_gather_matmul_memcfg"] = ttnn.create_sharded_memory_config(
            (tile_padded_batch_rows, dim // 8),
            ttnn.CoreGrid(x=8, y=1),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    # --- Phase 6: Input/output memory configs ---

    if config.decode_input_memcfg is None:
        # All 1D topologies use WIDTH_SHARDED for DRAM-sharded matmul (matches TTTv1)
        attn_input_grid = _dram_shard_core_grid(dim)
        to_set["decode_input_memcfg"] = ttnn.create_sharded_memory_config(
            (tile_padded_batch_rows, dim // attn_input_grid.num_cores),
            attn_input_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    if config.prefill_input_memcfg is None:
        to_set["prefill_input_memcfg"] = ttnn.DRAM_MEMORY_CONFIG

    # --- Phase 7: Resolve LazyWeights ---

    wqkv_dtype = to_set.get("wqkv_dtype", config.wqkv_dtype)
    wo_dtype = to_set.get("wo_dtype", config.wo_dtype)

    wqkv_mem_config = config.wqkv_memcfg
    if wqkv_mem_config is None:
        # Use DRAM-sharded config for all 1D topologies (matches TTTv1 behavior)
        # Note: For multi-device, the per-device portion uses qkv_size/num_devices
        dram_grid_size = mesh_device.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        )
        # For multi-device, each device has qkv_size/num_devices columns
        per_device_qkv_size = qkv_size // num_devices
        wqkv_mem_config = _create_dram_sharded_mem_config(
            k=dim, n=per_device_qkv_size, dram_grid=dram_grid, dram_cores=dram_grid_size.x
        )

    to_set["wqkv"] = resolve_lazy_weight(
        config.wqkv,
        device=mesh_device,
        memory_config=wqkv_mem_config,
        mesh_mapper_config=ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(-1)],
            mesh_shape_override=ttnn.MeshShape([num_devices]),
        ),
        layout=ttnn.TILE_LAYOUT,
        dtype=wqkv_dtype,
    )

    wo_mem_config = config.wo_memcfg
    if wo_mem_config is None:
        # Fused all-gather matmul needs INTERLEAVED (TTTv1: dims=(2,3) for fused)
        # Non-fused uses DRAM-sharded for all 1D topologies (TTTv1 behavior)
        if use_fused:
            wo_mem_config = ttnn.DRAM_MEMORY_CONFIG
        else:
            dram_grid_size = mesh_device.dram_grid_size()
            dram_grid = ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
            )
            # For multi-device, each device has (n_heads * head_dim) / num_devices input dim
            per_device_hidden = (n_heads * head_dim) // num_devices
            wo_mem_config = _create_dram_sharded_mem_config(
                k=per_device_hidden, n=dim, dram_grid=dram_grid, dram_cores=dram_grid_size.x
            )

    to_set["wo"] = resolve_lazy_weight(
        config.wo,
        device=mesh_device,
        memory_config=wo_mem_config,
        mesh_mapper_config=ttnn.MeshMapperConfig(
            # For fused: shard on dim -1 (width) - matches TTTv1 dims=(2, 3)
            # For non-fused: shard on dim -2 (height) - matches TTTv1 dims=(3, 2)
            placements=[ttnn.PlacementShard(-1 if use_fused else -2)],
            mesh_shape_override=ttnn.MeshShape([num_devices]),
        ),
        layout=ttnn.TILE_LAYOUT,
        dtype=wo_dtype,
    )

    # Resolve Q/K norm configs if present
    # RMSNorm1DConfig needs mesh_device propagated from attention config
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

    # --- Phase 8: Handle QKV bias ---

    if config.wqkv_bias is not None:
        pass

        qkv_bias = config.wqkv_bias

        # Prefill bias
        wqkv_bias_prefill = ttnn.from_torch(
            qkv_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
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
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
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


def _find_prefill_grid(row_tiles: int, col_tiles: int, max_rows: int = 8, max_cols: int = 8) -> tuple[int, int]:
    """Find grid where row_tiles divides rows and col_tiles divides cols."""
    cols = next((i for i in range(max_cols, 0, -1) if col_tiles % i == 0), None)
    rows = next((i for i in range(max_rows, 0, -1) if row_tiles % i == 0), None)
    assert cols is not None and rows is not None
    return rows, cols


def _get_out_subblock_w(per_core_n: int, out_subblock_h: int = 1) -> int:
    """Get output subblock width that divides per_core_n and satisfies constraints."""
    out_subblock_w = 4
    while out_subblock_w > 1:
        if out_subblock_w * out_subblock_h <= 4 and per_core_n % out_subblock_w == 0:
            break
        out_subblock_w -= 1
    return out_subblock_w


def _dram_shard_core_grid(k: int, tile_size: int = TILE_SIZE) -> ttnn.CoreGrid:
    """Get core grid for DRAM sharding based on K dimension."""
    rows, cols = _find_grid(k // tile_size)
    return ttnn.CoreGrid(x=cols, y=rows)


def _dram_matmul_config(
    m: int, k: int, n: int, num_cores: int, tile_size: int = TILE_SIZE
) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
    """Create DRAM-sharded matmul program config."""
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_find_largest_divisor(k // (tile_size * num_cores)),
        per_core_M=math.ceil(m / tile_size),
        per_core_N=math.ceil(n / (tile_size * num_cores)),
        fused_activation=None,
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


def _create_dram_sharded_mem_config(
    k: int, n: int, dram_grid: ttnn.CoreRangeSet, tile_size: int = TILE_SIZE, dram_cores: int = 12
) -> ttnn.MemoryConfig:
    """Create DRAM-sharded memory config for weight tensors."""
    padded_size = math.ceil(n / (tile_size * dram_cores)) * (tile_size * dram_cores)
    shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def _num_to_corerange(num_cores: int, start_core: ttnn.CoreCoord = None) -> ttnn.CoreRange:
    """Convert number of cores to CoreRange."""
    if start_core is None:
        start_core = ttnn.CoreCoord(0, 0)

    if num_cores == 1:
        return ttnn.CoreRange(start_core, start_core)

    # Arrange cores in rows of 8
    row_size = 8
    start_x, start_y = start_core.x, start_core.y

    # Calculate end coordinates
    total_cores_with_start = start_x + num_cores
    end_y = start_y + (total_cores_with_start - 1) // row_size
    end_x = (total_cores_with_start - 1) % row_size

    return ttnn.CoreRange(start_core, ttnn.CoreCoord(end_x, end_y))


def _zeros_like_kv_cache(batch_size: int, n_kv_heads: int, max_seq_len: int, head_dim: int) -> "torch.Tensor":
    """Create zeros tensor for standard KV cache."""
    import torch

    return torch.zeros((batch_size, n_kv_heads, max_seq_len, head_dim))


def _zeros_like_paged_cache(paged_config, n_kv_heads: int, head_dim: int) -> "torch.Tensor":
    """Create zeros tensor for paged KV cache."""
    import torch

    return torch.zeros((paged_config.max_num_blocks, n_kv_heads, paged_config.block_size, head_dim))


def _load_input_device_tensor(x: ttnn.Tensor | LazyWeight, config: Attention1DConfig, mode: str) -> ttnn.Tensor:
    """Resolve input tensor to ttnn.Tensor if LazyWeight, otherwise return as-is."""
    assert mode in ("decode", "prefill"), f"mode must be 'decode' or 'prefill', got {mode}"

    if isinstance(x, LazyWeight):
        mem_cfg = config.decode_input_memcfg if mode == "decode" else config.prefill_input_memcfg
        resolved_x = resolve_lazy_weight(
            x,
            device=config.mesh_device,
            memory_config=mem_cfg,
            mesh_mapper_config=None,  # replicated
            layout=ttnn.TILE_LAYOUT,
        )
        return resolved_x.get_device_weight()

    assert isinstance(x, ttnn.Tensor), f"x must be ttnn.Tensor or LazyWeight, got {type(x)}"
    return x
