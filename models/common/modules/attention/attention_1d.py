# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2-style Attention module for 1D-topology devices: N150 (1x1), N300 (1x2), T3K (1x8).

Single unified Attention1D class with separate forward methods:
  - decode_forward(): For decode mode (single token per user)
  - prefill_forward(): For prefill mode (multiple tokens)
  - forward(x, mode, **kwargs): Dispatcher that calls the appropriate method

Execution paths:
  Decode:  QKV matmul → all_reduce → create_qkv_heads → rotary → KV cache update → SDPA → concat_heads → WO matmul → all_reduce
  Prefill: [reshape] → QKV matmul → all_reduce → create_qkv_heads → rotary → KV cache fill → SDPA → concat_heads → [reshape] → all_gather → WO matmul → reduce_scatter

Key design decisions:
  - No static branching on topology in forward() - TG excluded at module level
  - Paged vs non-paged KV cache selected at runtime based on page_table presence
  - Fused all-gather matmul used for Ring topology (T3K 1x8) when dimensions allow

Weight format requirements:
  Q and K weights MUST be in Meta format, not HuggingFace format. HuggingFace stores
  Q/K weights with a different head layout that is incompatible with TTNN's RoPE
  implementation. Use `reverse_permute` from `models.tt_transformers.tt.load_checkpoints`
  to convert HF weights to Meta format before passing to Attention1D:

    from models.tt_transformers.tt.load_checkpoints import reverse_permute
    wq_meta = reverse_permute(wq_hf, n_heads, n_heads * head_dim, dim)
    wk_meta = reverse_permute(wk_hf, n_kv_heads, n_kv_heads * head_dim, dim)

  The `from_model_args` factory handles this automatically via `convert_hf_to_meta`.
  When using `from_config` with weights extracted directly from HuggingFace models,
  you must apply this transformation manually.
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
from models.common.modules.tt_ccl import (
    CCL_CHUNKS_PER_SYNC,
    CCL_NUM_BUFFERS_PER_CHANNEL,
    CCL_NUM_WORKERS_PER_LINK,
    TT_CCL,
    default_topology,
    get_tt_ccl,
)
from models.common.tensor_utils import (
    TILE_SIZE,
    get_rot_transformation_mat,
    zeros_like_kv_cache,
    zeros_like_paged_cache,
)
from models.common.utility_functions import is_blackhole, nearest_32

# =============================================================================
# Constants
# =============================================================================

MAX_QKV_MM_SEQ_LEN = 2048  # Maximum sequence length for single QKV matmul

# Maximum sequence length for WO matmul operation on Wormhole/Blackhole.
# Sequences longer than this are reshaped to [1, seq_len // MAX_MM_SEQ_LEN, MAX_MM_SEQ_LEN, dim]
# to fit on device and parallelize computation. After matmul, reshaped back.
# Source: TTTv1 model_config.py "MAX_MM_SEQ_LEN": 1024
MAX_MM_SEQ_LEN = 1024

# Total tokens in KV cache must fit in device DRAM. The 128K limit is a hardware
# constraint for Wormhole devices (by 12GB DRAM per chip) - exceeding it causes OOM based on previous tests.
MAX_TOTAL_TOKENS = 128 * 1024  # 131072 tokens

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
    # IMPORTANT: Q and K weights must be in Meta format (not HuggingFace format).
    # HF weights require `reverse_permute` transformation for RoPE compatibility.
    # See module docstring for details.
    wqkv: LazyWeight  # Combined QKV projection weight (Q/K in Meta format)
    wo: LazyWeight  # Output projection weight

    # Optional: Q/K normalization configs (e.g., for Qwen models)
    # Composed sub-module pattern: RMSNorm1DConfig instead of raw weights
    q_norm_config: RMSNorm1DConfig | None = None
    k_norm_config: RMSNorm1DConfig | None = None

    # Optional: QKV bias (e.g., for Qwen models)
    wqkv_bias: LazyWeight | None = None

    # Device and collectives
    mesh_device: ttnn.MeshDevice | None = None
    tt_ccl: TT_CCL | None = None
    topology: Optional[ttnn.Topology] = None  # None = auto-detect
    num_reduce_scatter_links: int | None = None
    num_all_gather_links: int | None = None

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
    #
    # The KV cache is a static configuration, allocated once and reused for all forward calls.
    # This reflects how inference engines like vLLM manage KV caches:
    # - Cache shape is determined by static model config (num_kv_heads, head_dim, block_size)
    # - The cache is allocated once during model initialization (allocate_vllm_kv_cache)
    # - The same cache tensors are passed to every forward call
    #
    # kv_cache options:
    # - None: Set externally before forward (e.g., via from_model_args or direct assignment)
    # - tuple[LazyWeight, LazyWeight]: (keys, values) backed by cache files, resolved lazily
    # - tuple[ttnn.Tensor, ttnn.Tensor]: Pre-allocated (keys, values) tensors (e.g., from vLLM)
    use_vllm_paged_kv_cache: bool = False
    kv_cache: "tuple[LazyWeight, LazyWeight] | tuple[ttnn.Tensor, ttnn.Tensor] | None" = None
    paged_attention_config: "PagedAttentionConfig | None" = None  # type: ignore
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    # Threshold for sharding KV cache during prefill to handle update_cache memory limitations.
    min_kv_prefill_shard_seqlen: int | None = None

    # Weight dtypes (None = auto-resolved based on device arch / model config)
    wqkv_dtype: ttnn.DataType | None = None  # QKV projection weight dtype
    wo_dtype: ttnn.DataType | None = None  # Output projection weight dtype
    activation_dtype: ttnn.DataType | None = None  # Intermediate activation dtype (post-RoPE, etc.)

    # Weight memory configs (None = DRAM interleaved by default)
    wqkv_memcfg: ttnn.MemoryConfig | None = None  # QKV weight placement
    wo_memcfg: ttnn.MemoryConfig | None = None  # Output weight placement

    # Decode program configs (None = auto-derived in _resolve_attention1d_config)
    decode_input_memcfg: ttnn.MemoryConfig | None = None  # DRAM-sharded input for decode
    decode_xqkv_prg_config: "ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig | None" = None  # QKV matmul
    decode_sdpa_prg_config: ttnn.SDPAProgramConfig | None = None  # Scaled dot-product attention
    decode_attn_output_prg_config: "ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig | None" = (
        None  # WO matmul
    )
    decode_residual_memcfg: ttnn.MemoryConfig | None = None  # Residual add output placement
    decode_create_qkv_head_memcfg: ttnn.MemoryConfig | None = None  # QKV head split output
    decode_scores_memcfg: Callable[[int], ttnn.MemoryConfig] | None = None  # SDPA scores; f(batch_size)

    # Prefill program configs (Callable factories: seq_len → config)
    prefill_input_memcfg: ttnn.MemoryConfig | None = None  # DRAM interleaved input for prefill
    prefill_xqkv_prg_config: Callable[
        [int], ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
    ] | None = None  # f(seq_len)
    prefill_sdpa_prg_config: Callable[[int, int | None], ttnn.SDPAProgramConfig] | None = None  # f(seq_len, chunk_size)
    prefill_wo_prg_config: Callable[[int], ttnn.MatmulMultiCoreReuseMultiCastProgramConfig] | None = None  # f(seq_len)
    prefill_kv_memcfg: Callable[[int], ttnn.MemoryConfig] | None = None  # f(seq_len) for KV cache write

    # Fused all-gather matmul (Ring topology only, decode path)
    use_fused_all_gather_matmul: bool | None = None  # None = auto-detect based on topology + dim
    decode_all_gather_matmul_prg_config: "ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None" = (
        None  # Fused AG+WO
    )
    decode_all_gather_matmul_memcfg: ttnn.MemoryConfig | None = None  # Output placement for fused AG+WO

    # Separate all-gather (non-Ring topology or non-fused decode path)
    decode_gather_users_memcfg: ttnn.MemoryConfig | None = None  # Output placement for standalone all_gather

    # Compute kernel configs (None = auto-derived; fp32 dest acc, math fidelity, etc.)
    li_qkv_decode_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None  # Decode QKV matmul kernel
    sdpa_decode_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None  # Decode SDPA kernel
    li_o_decode_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None  # Decode WO matmul kernel
    li_qkv_prefill_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None  # Prefill QKV matmul kernel
    sdpa_prefill_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None  # Prefill SDPA kernel
    li_o_prefill_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None  # Prefill WO matmul kernel

    # Transformation matrices for rotary embedding (static, computed once).
    # These are NOT LazyWeights because they are:
    # - Small fixed-size tensors (32x32), not large model weights
    # - Computed programmatically (get_rot_transformation_mat), not loaded from checkpoints
    transformation_mat_decode: ttnn.Tensor | None = None
    transformation_mat_prefill: ttnn.Tensor | None = None

    # Internal: pre-computed bias LazyWeights (materialized in __init__)
    _wqkv_bias_decode: list[LazyWeight] | None = field(default=None, repr=False)
    _wqkv_bias_prefill: LazyWeight | None = field(default=None, repr=False)

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

    Simple API (90% of users) - requires weights, model dimensions, and memory bounds:
        attn = Attention1D(wqkv, wo, n_heads=32, n_kv_heads=8, head_dim=128,
                           max_batch_size=1, max_seq_len=2048)

    Power API (10% of users) - full customization via config:
        config = Attention1DConfig(wqkv, wo, n_heads=32, n_kv_heads=8, head_dim=128,
                                   sliding_window=4096, q_norm_config=..., ...)
        attn = Attention1D.from_config(config)

    Execution paths:
      Decode:  QKV matmul → all_reduce → create_qkv_heads → rotary → KV cache → SDPA → concat_heads → WO matmul → all_reduce
      Prefill: [reshape] → QKV matmul → all_reduce → create_qkv_heads → rotary → KV cache → SDPA → concat_heads → WO matmul

    KV Cache Management:
        The KV cache is a static configuration stored in Attention1DConfig.kv_cache,
        allocated once and reused for all forward calls. This design reflects how
        inference engines like vLLM manage KV caches: the cache shape is determined
        by static model config and the same tensors are reused for all forward calls.

        Options:
        - from_model_args(): Automatically allocates KV cache (TTTv1 compatibility)
        - from_config() with kv_cache: Pre-allocated (keys, values) tensors (vLLM integration)
        - from_config() with kv_cache as LazyWeight tuple: Cache backed by files
    """

    def __init__(
        self,
        wqkv: LazyWeight,
        wo: LazyWeight,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        max_batch_size: int,
        max_seq_len: int,
    ):
        """
        Simple API for 90% of users - requires weights, model dimensions, and memory bounds.

        Args:
            wqkv: Combined QKV projection weight, sharded on dim=-1
            wo: Output projection weight, sharded on dim=-2
            n_heads: Number of attention heads (from model config)
            n_kv_heads: Number of key/value heads for GQA (from model config)
            head_dim: Dimension per head (from model config)
            max_batch_size: Maximum batch size for KV cache allocation
            max_seq_len: Maximum sequence length for KV cache allocation

        Other settings (mesh_device, topology, program configs, etc.) are derived
        automatically. Use from_config() for full customization.

        Example:
            attn = Attention1D(wqkv, wo, n_heads=32, n_kv_heads=8, head_dim=128,
                               max_batch_size=1, max_seq_len=2048)
        """
        super().__init__()
        self.config = _resolve_attention1d_config(
            Attention1DConfig(
                wqkv=wqkv,
                wo=wo,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
            )
        )
        self._device_weights_loaded = False
        self._bind_forward_methods()

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
        instance._bind_forward_methods()
        return instance

    def prefill_forward(
        self,
        x: ttnn.Tensor | LazyWeight,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int = 0,
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
    ) -> ttnn.Tensor:
        """
        Prefill forward - multiple tokens.

        Args:
            x: Input tensor, shape (1, 1, seq_len, dim), TILE_LAYOUT.
                Memory config: DRAM interleaved (default ``prefill_input_memcfg``).
                If ``LazyWeight``, it is automatically placed with the correct memory config.
            rot_mats: Tuple of (cos, sin) rotation matrices for rotary embedding,
                each shape (1, 1, head_dim, head_dim), TILE_LAYOUT, DRAM interleaved.
            user_id: User ID for KV cache fill (selects which user's cache to write).
            page_table: Page table for paged attention (optional).
            chunk_page_table: Page table for chunked prefill (optional).
            chunk_start_idx: Start index for chunked prefill (optional).

        Returns:
            Output tensor (1, 1, seq_len, dim), DRAM interleaved.

        Note:
            seq_len must be divisible by 128 and > 0.
        """
        self.load_device_weights()
        x = _load_input_device_tensor(x, self.config, mode="prefill")
        cfg = self.config

        seq_len = x.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "seq_len must be divisible by 128"

        num_devices = cfg.mesh_device.get_num_devices()
        n_local_heads = cfg.n_heads // num_devices
        n_local_kv_heads = cfg.n_kv_heads // num_devices

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

        # --- STAGE 3: No all-reduce for 1D topologies ---
        # For 1D meshes (1xN), QKV weights are sharded only on axis 1 (columns).
        # After matmul, each device has complete rows - no partial sums to reduce.
        # TTTv1's tt_all_reduce with cluster_axis=1 is a no-op for 1D meshes.

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
        # Explicit dtype check is required - ttnn.typecast is NOT a no-op when source == target types.
        # Without this check, typecast would allocate a new tensor and run an identity copy kernel.
        if q_heads_pre_rot.dtype != ttnn.bfloat16:
            q_heads_pre_rot = ttnn.typecast(q_heads_pre_rot, dtype=ttnn.bfloat16)

        q_heads = ttnn.experimental.rotary_embedding_llama(
            q_heads_pre_rot, rot_mats[0], rot_mats[1], cfg.transformation_mat_prefill, is_decode_mode=False
        )
        ttnn.deallocate(q_heads_pre_rot)

        if k_heads_pre_rot.dtype != ttnn.bfloat16:
            k_heads_pre_rot = ttnn.typecast(k_heads_pre_rot, dtype=ttnn.bfloat16)

        k_heads = ttnn.experimental.rotary_embedding_llama(
            k_heads_pre_rot, rot_mats[0], rot_mats[1], cfg.transformation_mat_prefill, is_decode_mode=False
        )
        ttnn.deallocate(k_heads_pre_rot)

        # --- STAGE 7: Typecast to cache dtype ---
        keys, values = self.kv_cache

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
        # Method bound at construction based on paged_attention_config (see _bind_forward_methods)
        self._kv_fill_prefill(keys, values, k_fill, v_fill, user_id, page_table, chunk_page_table)

        # Deallocate sharded k_fill/v_fill only in non-paged mode (paged mode uses sliced views)
        is_paged = cfg.paged_attention_config is not None
        if seq_len >= cfg.min_kv_prefill_shard_seqlen and not is_paged:
            ttnn.deallocate(k_fill)
            ttnn.deallocate(v_fill)

        # --- STAGE 10: SDPA ---
        q_heads_sdpa = ttnn.typecast(q_heads, dtype=cfg.activation_dtype or ttnn.bfloat8_b)
        ttnn.deallocate(q_heads)

        # Chunked vs non-chunked is a RUNTIME decision (based on seq_len vs max_prefill_chunk_size),
        # so this branching is valid under TTTv2 principles. Valid combinations:
        # - Paged + Chunked: production vLLM serving (long prompts)
        # - Paged + Non-chunked: short prompts in vLLM
        # - Non-paged + Non-chunked: simple testing (uses else branch with contiguous KV)
        # Invalid combinations (rejected at config time in _resolve_attention1d_config):
        # - Non-paged + Chunked: chunked_sdpa requires page_table
        # - sliding_window + Chunked: chunked_sdpa does not implement window masking
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

        # --- STAGE 12: Reshape for long sequences (to fit WO matmul on device) ---
        if seq_len > MAX_MM_SEQ_LEN:
            attn_output_concat = ttnn.reshape(attn_output_concat, [1, seq_len // MAX_MM_SEQ_LEN, MAX_MM_SEQ_LEN, -1])

        # --- STAGE 13: All-Gather for Ring topology ---
        # Method bound at construction based on use_fused_all_gather_matmul (see _bind_forward_methods)
        attn_output_concat = self._all_gather_before_wo_prefill(attn_output_concat)

        # --- STAGE 14: WO Matmul ---
        output = ttnn.linear(
            attn_output_concat,
            self.wo,
            compute_kernel_config=cfg.li_o_prefill_compute_kernel_cfg,
            dtype=cfg.activation_dtype or ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=cfg.prefill_wo_prg_config(seq_len),
        )

        # --- STAGE 15: Reshape back (undo long sequence reshape) ---
        if seq_len > MAX_MM_SEQ_LEN:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])

        ttnn.deallocate(attn_output_concat)

        # --- STAGE 16: All-Reduce output ---
        # Method bound at construction based on use_fused_all_gather_matmul (see _bind_forward_methods)
        output = self._reduce_after_wo_prefill(output)

        return output

    def decode_forward(
        self,
        x: ttnn.Tensor | LazyWeight,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        Decode forward - single token per user.

        Args:
            x: Input tensor, shape (seq_len, 1, batch, dim), TILE_LAYOUT.
                Memory config: DRAM WIDTH_SHARDED (default ``decode_input_memcfg``).
                If ``LazyWeight``, it is automatically placed with the correct memory config.
            current_pos: Current position tensor, shape (batch_size,) on host.
            rot_mats: Tuple of (cos, sin) rotation matrices for rotary embedding,
                each shape (1, batch, head_dim, head_dim), TILE_LAYOUT, L1 interleaved.
            page_table: Page table for paged attention (optional).

        Returns:
            Output tensor (seq_len, 1, batch, dim), DRAM WIDTH_SHARDED
            (``decode_residual_memcfg``).
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
            overlap_qk_coregrid=self._decode_overlap_qk_coregrid,
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
        # Method bound at construction based on use_qk_fused (see _bind_forward_methods)
        q_heads, k_heads = self._rotary_embed_decode(q_heads_pre_rot, k_heads_pre_rot, rot_mats)
        ttnn.deallocate(q_heads_pre_rot)
        ttnn.deallocate(k_heads_pre_rot)

        # --- STAGE 6: KV Cache Update ---
        # Method bound at construction based on use_qk_fused (see _bind_forward_methods)
        keys, values = self.kv_cache
        self._kv_update_decode(keys, values, k_heads, v_heads, current_pos, page_table)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        # --- STAGE 7: SDPA ---
        # Method bound at construction based on paged_attention_config (see _bind_forward_methods)
        attn_output = self._sdpa_decode(q_heads, keys, values, current_pos, page_table)

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
        # Method bound at construction based on use_fused_all_gather_matmul (see _bind_forward_methods)
        dense_out = self._all_gather_wo_decode(attn_output_cat)

        ttnn.deallocate(attn_output_cat)

        # --- STAGE 11: Finalize output ---
        # Method bound at construction based on use_fused_all_gather_matmul (see _bind_forward_methods)
        # Fused path: returns dense_out as-is
        # Non-fused path: reduce-scatter + final memory config
        return self._finalize_decode_output(dense_out)

    def forward(
        self,
        x: ttnn.Tensor | LazyWeight,
        current_pos: ttnn.Tensor | None,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int = 0,
        mode: str = "decode",
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
    ) -> ttnn.Tensor:
        """Dispatch to the appropriate forward method based on mode."""
        if mode == "prefill":
            return self.prefill_forward(
                x,
                rot_mats,
                user_id=user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
            )
        else:
            return self.decode_forward(
                x,
                current_pos,
                rot_mats,
                page_table=page_table,
            )

    def _bind_forward_methods(self):
        """
        Bind forward method variants based on static config.

        This eliminates runtime if-statements in the hot path by pre-binding:
        - _sdpa_decode: paged or non-paged SDPA for decode
        - _kv_fill_prefill: paged or non-paged KV cache fill for prefill
        - _all_gather_before_wo_prefill: fused all-gather or no-op
        - _reduce_after_wo_prefill: reduce-scatter or no-op
        - _all_gather_wo_decode: fused or separate all-gather + WO matmul
        - _finalize_decode_output: fused (direct return) or non-fused (reduce + memcfg)
        - _rotary_embed_decode: fused QK or separate rotary embedding
        - _kv_update_decode: fused or separate KV cache update

        Note: page_table is still passed as a forward argument for vLLM compatibility
        (vLLM dynamically manages page tables), but the method binding is static.
        """
        cfg = self.config
        is_paged = cfg.paged_attention_config is not None

        # Bind paged vs non-paged methods
        if is_paged:
            self._sdpa_decode = self._sdpa_decode_paged
            self._kv_fill_prefill = self._kv_fill_prefill_paged
        else:
            self._sdpa_decode = self._sdpa_decode_non_paged
            self._kv_fill_prefill = self._kv_fill_prefill_non_paged

        # Bind fused all-gather methods (based on use_fused_all_gather_matmul + topology)
        use_fused = cfg.use_fused_all_gather_matmul and cfg.topology == ttnn.Topology.Ring
        if use_fused:
            self._all_gather_before_wo_prefill = self._all_gather_before_wo_prefill_fused
            self._reduce_after_wo_prefill = self._reduce_after_wo_prefill_fused
            self._all_gather_wo_decode = self._fused_all_gather_wo_decode
            self._finalize_decode_output = self._finalize_decode_output_fused
        else:
            self._all_gather_before_wo_prefill = self._all_gather_before_wo_prefill_noop
            self._reduce_after_wo_prefill = self._reduce_after_wo_prefill_non_fused
            self._all_gather_wo_decode = self._separate_all_gather_wo_decode
            self._finalize_decode_output = self._finalize_decode_output_non_fused

        # Bind fused QK methods (rotary embedding and KV cache update)
        self._decode_overlap_qk_coregrid = not cfg.use_qk_fused
        if cfg.use_qk_fused:
            self._rotary_embed_decode = self._rotary_embed_decode_fused
            self._kv_update_decode = self._kv_update_decode_fused
        else:
            self._rotary_embed_decode = self._rotary_embed_decode_nonfused
            self._kv_update_decode = self._kv_update_decode_nonfused

    # =========================================================================
    # Bound SDPA Decode Methods (paged vs non-paged)
    # =========================================================================

    def _sdpa_decode_paged(self, q_heads, keys, values, current_pos, page_table) -> ttnn.Tensor:
        """Paged SDPA decode - uses page_table for KV cache lookup."""
        cfg = self.config
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
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

    def _sdpa_decode_non_paged(self, q_heads, keys, values, current_pos, page_table) -> ttnn.Tensor:
        """Non-paged SDPA decode - contiguous KV cache (page_table ignored)."""
        cfg = self.config
        return ttnn.transformer.scaled_dot_product_attention_decode(
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

    # =========================================================================
    # Bound KV Fill Prefill Methods (paged vs non-paged)
    # =========================================================================

    def _kv_fill_prefill_paged(self, keys, values, k_fill, v_fill, user_id, page_table, chunk_page_table) -> None:
        """Paged KV cache fill for prefill - uses page_table for block allocation."""
        block_size = keys.shape[2]
        fill_page_table = chunk_page_table if chunk_page_table is not None else page_table
        page_len = fill_page_table.shape[1] * block_size

        k_fill_sliced = k_fill[:, :, :page_len, :] if page_len < k_fill.shape[2] else k_fill
        v_fill_sliced = v_fill[:, :, :page_len, :] if page_len < v_fill.shape[2] else v_fill

        ttnn.experimental.paged_fill_cache(keys, k_fill_sliced, fill_page_table, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(values, v_fill_sliced, fill_page_table, batch_idx=user_id)

    def _kv_fill_prefill_non_paged(self, keys, values, k_fill, v_fill, user_id, page_table, chunk_page_table) -> None:
        """Non-paged KV cache fill for prefill - contiguous cache (page_table ignored)."""
        cfg = self.config
        ttnn.fill_cache(keys, k_fill, user_id % cfg.max_batch_size)
        ttnn.fill_cache(values, v_fill, user_id % cfg.max_batch_size)

    # =========================================================================
    # Bound All-Gather/Reduce Methods for Prefill (fused vs non-fused)
    # =========================================================================

    def _all_gather_before_wo_prefill_fused(self, attn_output_concat: ttnn.Tensor) -> ttnn.Tensor:
        """Fused path: all-gather before WO matmul (Ring topology)."""
        cfg = self.config
        return ttnn.experimental.all_gather_async(
            attn_output_concat,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=1,
            topology=cfg.topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=CCL_CHUNKS_PER_SYNC,
            num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
        )

    def _all_gather_before_wo_prefill_noop(self, attn_output_concat: ttnn.Tensor) -> ttnn.Tensor:
        """Non-fused path: no all-gather before WO matmul."""
        return attn_output_concat

    def _reduce_after_wo_prefill_fused(self, output: ttnn.Tensor) -> ttnn.Tensor:
        """Fused path: no reduce after WO matmul (already complete from all-gather)."""
        return output

    def _reduce_after_wo_prefill_non_fused(self, output: ttnn.Tensor) -> ttnn.Tensor:
        """Non-fused path: reduce-scatter after WO matmul."""
        return self._all_reduce_output_prefill(output)

    # =========================================================================
    # Bound Finalize Methods for Decode (fused vs non-fused)
    # =========================================================================

    def _finalize_decode_output_fused(self, dense_out: ttnn.Tensor) -> ttnn.Tensor:
        """Fused path: output is already complete, return as-is."""
        return dense_out

    def _finalize_decode_output_non_fused(self, dense_out: ttnn.Tensor) -> ttnn.Tensor:
        """Non-fused path: reduce-scatter and apply final memory config."""
        cfg = self.config

        dense_out_reduced = self._all_reduce_output_decode(dense_out)

        # Only deallocate if a new tensor was created (multi-device case).
        # For single device, _all_reduce_output_decode returns input unchanged.
        if cfg.mesh_device.get_num_devices() > 1:
            ttnn.deallocate(dense_out)

        return ttnn.to_memory_config(dense_out_reduced, cfg.decode_residual_memcfg)

    # =========================================================================
    # Bound Rotary Embedding Methods for Decode (fused QK vs separate)
    # =========================================================================

    def _rotary_embed_decode_fused(
        self, q_heads_pre_rot: ttnn.Tensor, k_heads_pre_rot: ttnn.Tensor, rot_mats: tuple[ttnn.Tensor, ttnn.Tensor]
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Fused QK rotary embedding - single kernel for both Q and K.

        The fused kernel requires Q and K on non-overlapping core grids.
        After create_qkv_heads_decode with interleaved input, Q is already on the
        correct cores (first batch cores), so only K needs resharding to a
        non-overlapping grid — saving 1 dispatch vs the original 2-reshard approach.
        """
        cfg = self.config
        k_heads_pre_rot = self._reshard_k_for_fused(k_heads_pre_rot, q_heads_pre_rot.shape[1])
        return ttnn.experimental.rotary_embedding_llama_fused_qk(
            q_heads_pre_rot, k_heads_pre_rot, rot_mats[0], rot_mats[1], cfg.transformation_mat_decode
        )

    def _rotary_embed_decode_nonfused(
        self, q_heads_pre_rot: ttnn.Tensor, k_heads_pre_rot: ttnn.Tensor, rot_mats: tuple[ttnn.Tensor, ttnn.Tensor]
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Separate rotary embedding - independent kernels for Q and K."""
        cfg = self.config
        q_heads = ttnn.experimental.rotary_embedding_llama(
            q_heads_pre_rot, rot_mats[0], rot_mats[1], cfg.transformation_mat_decode, is_decode_mode=True
        )
        k_heads = ttnn.experimental.rotary_embedding_llama(
            k_heads_pre_rot, rot_mats[0], rot_mats[1], cfg.transformation_mat_decode, is_decode_mode=True
        )
        return q_heads, k_heads

    # =========================================================================
    # Bound KV Cache Update Methods for Decode (fused vs separate)
    # =========================================================================

    def _kv_update_decode_fused(
        self,
        keys: ttnn.Tensor,
        values: ttnn.Tensor,
        k_heads: ttnn.Tensor,
        v_heads: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor | None,
    ) -> None:
        """Fused KV cache update - single kernel for both K and V."""
        ttnn.experimental.paged_fused_update_cache(
            keys, k_heads, values, v_heads, update_idxs_tensor=current_pos, page_table=page_table
        )

    def _kv_update_decode_nonfused(
        self,
        keys: ttnn.Tensor,
        values: ttnn.Tensor,
        k_heads: ttnn.Tensor,
        v_heads: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor | None,
    ) -> None:
        """Separate KV cache update - independent kernels for K and V."""
        ttnn.experimental.paged_update_cache(keys, k_heads, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(values, v_heads, update_idxs_tensor=current_pos, page_table=page_table)

    def load_device_weights(self):
        """Materialize LazyWeights onto device. Called automatically on first forward; idempotent."""
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

        # Materialize bias LazyWeights
        if cfg._wqkv_bias_decode is not None:
            self.wqkv_bias_decode = [bias.get_device_weight() for bias in cfg._wqkv_bias_decode]
        else:
            self.wqkv_bias_decode = None
        if cfg._wqkv_bias_prefill is not None:
            self.wqkv_bias_prefill = cfg._wqkv_bias_prefill.get_device_weight()
        else:
            self.wqkv_bias_prefill = None

        # Resolve kv_cache from config (may be LazyWeight or ttnn.Tensor)
        if cfg.kv_cache is not None:
            keys, values = cfg.kv_cache
            if isinstance(keys, LazyWeight):
                keys = keys.get_device_weight()
            if isinstance(values, LazyWeight):
                values = values.get_device_weight()
            self.kv_cache = (keys, values)
        else:
            self.kv_cache = None

        self._device_weights_loaded = True

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

    def _all_reduce_output_decode(self, output: ttnn.Tensor) -> ttnn.Tensor:
        """
        Final all-reduce for decode output.

        For 1D topologies (1xN), this is reduce_scatter only.
        """
        cfg = self.config

        # Single device: no all-reduce needed
        if cfg.mesh_device.get_num_devices() == 1:
            return output

        # For 1D topologies: reduce_scatter across devices on axis 1
        # Convert sharded to interleaved first if needed
        if output.is_sharded():
            output_interleaved = ttnn.sharded_to_interleaved(output, ttnn.L1_MEMORY_CONFIG)
            output.deallocate(True)
        else:
            output_interleaved = output

        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            output_interleaved,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=cfg.num_reduce_scatter_links,
            memory_config=cfg.decode_residual_memcfg,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=cfg.topology,
            chunks_per_sync=CCL_CHUNKS_PER_SYNC,
            num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
        )
        output_interleaved.deallocate(True)

        return reduced

    def _all_reduce_output_prefill(self, output: ttnn.Tensor) -> ttnn.Tensor:
        """
        Final all-reduce for prefill output.

        For 1D topologies (1xN), this is reduce_scatter only.
        """
        cfg = self.config

        # Single device: no all-reduce needed
        if cfg.mesh_device.get_num_devices() == 1:
            return output

        # For 1D topologies: reduce_scatter across devices
        # Prefill uses interleaved memory, ensure we're in DRAM
        output = ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)

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
            chunks_per_sync=CCL_CHUNKS_PER_SYNC,
            num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
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
            chunks_per_sync=CCL_CHUNKS_PER_SYNC,
            num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
            num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
        )

        return ttnn.to_memory_config(dense_out, cfg.decode_residual_memcfg)

    def _separate_all_gather_wo_decode(self, attn_output_cat: ttnn.Tensor) -> ttnn.Tensor:
        """
        Separate all-gather then WO matmul (non-Ring or when fused not available).

        For 1D topologies with cluster_axis=1, all-gather is a no-op since there's only
        1 device on axis 0. The attention heads are already local to each device.
        """
        cfg = self.config

        # For 1D topologies (1xN mesh), all-gather with cluster_axis=1 is a no-op
        # because axis 0 has only 1 device. Skip the CCL call entirely.
        # This matches TTTv1 behavior where tt_all_gather returns input unchanged.

        # WO matmul
        dense_out = ttnn.linear(
            attn_output_cat,
            self.wo,
            program_config=cfg.decode_attn_output_prg_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=cfg.li_o_decode_compute_kernel_cfg,
        )

        return dense_out

    def _reshard_k_for_fused(self, k_tensor: ttnn.Tensor, q_batch: int) -> ttnn.Tensor:
        """Move K tensor to non-overlapping core grid for fused QK rotary embedding.

        After create_qkv_heads_decode with interleaved input, Q and K share the
        same core grid.  The fused rotary kernel requires non-overlapping grids.
        Q's grid (first ``q_batch`` cores) is already correct, so we only move K
        to the next ``q_batch`` cores — one dispatch instead of two.
        """
        n_kv_heads = k_tensor.shape[2]
        row_size = 8
        k_start_core = ttnn.CoreCoord(q_batch % row_size, q_batch // row_size)
        k_core_grid = ttnn.CoreRangeSet({_num_to_corerange(q_batch, start_core=k_start_core)})
        k_mem_config = ttnn.create_sharded_memory_config(
            shape=(nearest_32(n_kv_heads), self.config.head_dim),
            core_grid=k_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return ttnn.to_memory_config(k_tensor, k_mem_config)

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
                (
                    Path(weight_cache_path) / layer_name,
                    "wo_width_sharded" if use_fused_all_gather_matmul else "wo",
                )
                if weight_cache_path
                else None
            ),
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
            wqkv_bias = LazyWeight(source=qkv_bias)

        # Determine scale
        if configuration.query_pre_attn_scalar is not None:
            scale = configuration.query_pre_attn_scalar**-0.5
        else:
            scale = configuration.head_dim**-0.5

        # Build config
        # Note: kv_cache is created by default in _resolve_attention1d_config if not provided
        # and use_paged_kv_cache=False. For paged cache (vLLM), set use_paged_kv_cache=True.
        config = Attention1DConfig(
            wqkv=wqkv,
            wo=wo,
            q_norm_config=q_norm_config,
            k_norm_config=k_norm_config,
            wqkv_bias=wqkv_bias,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            topology=configuration.ccl_topology(),
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
            use_vllm_paged_kv_cache=use_paged_kv_cache,
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
            # Use provided transformation matrices if available; otherwise auto-created in resolve
            transformation_mat_decode=transformation_mats.get("decode") if transformation_mats else None,
            transformation_mat_prefill=transformation_mats.get("prefill") if transformation_mats else None,
        )

        return cls.from_config(config)


# =============================================================================
# Config resolution
# =============================================================================


def _resolve_attention1d_config(config: Attention1DConfig) -> Attention1DConfig:
    """Materialize the config with sensible defaults."""
    to_set = {}

    # --- Phase 1: Model dimensions (fail-fast validation, no device needed) ---
    # n_heads, n_kv_heads, head_dim MUST be provided - they cannot be reliably inferred
    # from weight shapes alone (different models have different configurations).

    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim

    if n_heads is None:
        raise ValueError(
            "n_heads must be provided. It cannot be reliably inferred from weights. "
            "Get this value from your model's config.json (num_attention_heads)."
        )
    if n_kv_heads is None:
        raise ValueError(
            "n_kv_heads must be provided. It cannot be reliably inferred from weights. "
            "Get this value from your model's config.json (num_key_value_heads)."
        )
    if head_dim is None:
        raise ValueError(
            "head_dim must be provided. It cannot be reliably inferred from weights. "
            "Typically: head_dim = hidden_size // num_attention_heads."
        )

    # --- Phase 1b: Token budget validation (fail-fast for memory) ---
    total_tokens = config.max_batch_size * config.max_seq_len
    if total_tokens > MAX_TOTAL_TOKENS:
        raise ValueError(
            f"Total token budget exceeded: max_batch_size ({config.max_batch_size}) × "
            f"max_seq_len ({config.max_seq_len}) = {total_tokens:,} tokens, "
            f"but maximum is {MAX_TOTAL_TOKENS:,} tokens (128K). "
            f"Reduce max_batch_size or max_seq_len to fit in device DRAM."
        )

    # Reject sliding_window + paged attention (chunked prefill doesn't support window masking)
    if config.sliding_window is not None and config.paged_attention_config is not None:
        raise ValueError(
            "Chunked prefill with sliding_window attention is not supported. "
            "chunked_scaled_dot_product_attention does not implement sliding window masking. "
            "Set sliding_window=None or disable paged attention / chunked prefill."
        )

    # --- Phase 2: Device and foundational fields ---

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
        to_set["topology"] = default_topology(mesh_device)

    topology = to_set.get("topology", config.topology)

    # Auto-detect num_links (same approach as TTTv1 ccl.py tt_all_reduce)
    tt_ccl = to_set.get("tt_ccl", config.tt_ccl)
    if config.num_reduce_scatter_links is None and num_devices > 1:
        to_set["num_reduce_scatter_links"] = tt_ccl.get_num_links()
    if config.num_all_gather_links is None and num_devices > 1:
        to_set["num_all_gather_links"] = tt_ccl.get_num_links()

    # --- Phase 3: Derived dimensions ---

    # dim CAN be reliably inferred from weight shapes
    dim = config.dim
    if dim is None:
        # wqkv shape is (1, 1, dim, qkv_size_per_device * num_devices) or (dim, qkv_size)
        wqkv_shape = config.wqkv.source.shape
        dim = wqkv_shape[-2] if len(wqkv_shape) == 4 else wqkv_shape[0]
        to_set["dim"] = dim

    # qkv_size is derived from the required dimensions
    qkv_size = config.qkv_size
    if qkv_size is None:
        qkv_size = head_dim * (2 * n_kv_heads + n_heads)
        to_set["qkv_size"] = qkv_size

    if config.scale is None:
        to_set["scale"] = head_dim**-0.5

    if config.min_kv_prefill_shard_seqlen is None:
        to_set["min_kv_prefill_shard_seqlen"] = (TILE_SIZE * 8 * 8) // (n_kv_heads // num_devices)

    # --- Phase 4: Dtypes ---

    if config.wqkv_dtype is None:
        to_set["wqkv_dtype"] = ttnn.bfloat8_b
    if config.wo_dtype is None:
        to_set["wo_dtype"] = ttnn.bfloat8_b
    if config.activation_dtype is None:
        to_set["activation_dtype"] = ttnn.bfloat16

    # --- Phase 5: Compute kernel configs ---

    compute_kernel_hifi2_fp16 = ttnn.WormholeComputeKernelConfig(
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
        to_set["li_qkv_decode_compute_kernel_cfg"] = compute_kernel_hifi2_fp16
    if config.sdpa_decode_compute_kernel_cfg is None:
        to_set["sdpa_decode_compute_kernel_cfg"] = compute_kernel_hifi2_fp16
    if config.li_o_decode_compute_kernel_cfg is None:
        to_set["li_o_decode_compute_kernel_cfg"] = compute_kernel_hifi2_fp16
    if config.li_qkv_prefill_compute_kernel_cfg is None:
        to_set["li_qkv_prefill_compute_kernel_cfg"] = compute_kernel_hifi2_fp16
    if config.sdpa_prefill_compute_kernel_cfg is None:
        to_set["sdpa_prefill_compute_kernel_cfg"] = compute_kernel_hifi4
    if config.li_o_prefill_compute_kernel_cfg is None:
        to_set["li_o_prefill_compute_kernel_cfg"] = compute_kernel_hifi2_fp16

    # --- Phase 6: Program configs ---

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
    # DRAM shard grid width: on Wormhole always 8 (despite 12 physical DRAM cores);
    # on Blackhole use actual DRAM grid width (7 for P100, 8 for P150).
    # Matching per_core_N to this width avoids silent PCC issues on P100.
    dram_shard_grid_width = 8 if not is_blackhole() else mesh_device.dram_grid_size().x

    if config.prefill_xqkv_prg_config is None:

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
        n_dim = MAX_MM_SEQ_LEN if use_fused and MAX_MM_SEQ_LEN % (dim // num_devices) == 0 else dim
        prefill_rows = 8

        @lru_cache
        def wo_prefill_prg_config(seq_len: int):
            num_rows = min(seq_len, MAX_MM_SEQ_LEN)
            grid_size = _find_prefill_grid(prefill_rows, k_dim // tile_size)
            return _matmul_config(
                m=num_rows,
                k=k_dim,
                n=n_dim,
                grid_size=grid_size,
                in0_block_w=1,
                fuse_batch=seq_len <= MAX_MM_SEQ_LEN,
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

    # Separate all-gather memory config (non-Ring or non-fused path)
    # Matches TTTv1 GATHER_USERS_MEMCFG: shape=(tile_size * mesh_cols, dim // 8 // users_core_grid.num_cores)
    if not use_fused and config.decode_gather_users_memcfg is None and num_devices > 1:
        mesh_cols = list(mesh_device.shape)[1]
        users_core_grid = _dram_shard_core_grid(dim // 8)
        to_set["decode_gather_users_memcfg"] = ttnn.create_sharded_memory_config(
            (tile_size * mesh_cols, dim // 8 // users_core_grid.num_cores),
            users_core_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    # --- Phase 7: Input/output memory configs ---

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

    # --- Phase 8: Resolve LazyWeights ---

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

    # --- Phase 9: Handle QKV bias ---

    if config.wqkv_bias is not None:
        # Extract source tensor from LazyWeight for custom transformation
        qkv_bias = config.wqkv_bias.source

        # Mesh mapper config for sharding bias on last dimension
        # For 1D mesh: [Replicate on dim 0, Shard on dim -1]
        bias_mesh_mapper_config = (
            ttnn.MeshMapperConfig(
                placements=[ttnn.PlacementReplicate(), ttnn.PlacementShard(-1)],
                mesh_shape_override=ttnn.MeshShape(1, num_devices),
            )
            if num_devices > 1
            else None  # Single device: replicate (no sharding)
        )

        # Prefill bias: transform to 4D shape (1, 1, 1, qkv_size)
        # Use replace() to create new LazyWeight with transformed source, then resolve_lazy_weight
        prefill_bias_lazy = resolve_lazy_weight(
            replace(config.wqkv_bias, source=qkv_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper_config=bias_mesh_mapper_config,
        )
        to_set["_wqkv_bias_prefill"] = prefill_bias_lazy

        # Decode bias - one per batch size multiple
        # Match TTTv1 pattern: 2D tensor (batch_size, qkv_size)
        wqkv_bias_decode = []
        for batch_size in range(tile_size, tile_padded_batch_rows + tile_size, tile_size):
            decode_bias_lazy = resolve_lazy_weight(
                replace(config.wqkv_bias, source=qkv_bias.unsqueeze(0).expand(batch_size, -1)),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper_config=bias_mesh_mapper_config,
            )
            wqkv_bias_decode.append(decode_bias_lazy)
        to_set["_wqkv_bias_decode"] = wqkv_bias_decode

    # --- Phase 10: Create transformation matrices for rotary embedding ---

    if config.transformation_mat_decode is None:
        use_qk_fused = config.use_qk_fused
        max_batch_size = config.max_batch_size
        doubled_batch_size = max_batch_size * 2 if use_qk_fused else max_batch_size

        # Get core grid for batch distribution
        core_grid = ttnn.CoreCoord(8, 8) if is_blackhole() else mesh_device.compute_with_storage_grid_size()
        batch_grid = ttnn.num_cores_to_corerangeset(doubled_batch_size, core_grid, row_wise=True)

        # Create transformation matrix repeated across batch cores
        trans_mat = get_rot_transformation_mat().repeat(1, 1, doubled_batch_size, 1)
        trans_mat_mem_config = ttnn.create_sharded_memory_config(
            shape=(TILE_SIZE, TILE_SIZE),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        transformation_mat_decode = ttnn.from_torch(
            trans_mat,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=trans_mat_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        to_set["transformation_mat_decode"] = transformation_mat_decode

    if config.transformation_mat_prefill is None:
        # Prefill uses simpler DRAM config (replicated across devices)
        prefill_trans_mat = get_rot_transformation_mat()
        transformation_mat_prefill = ttnn.from_torch(
            prefill_trans_mat,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        to_set["transformation_mat_prefill"] = transformation_mat_prefill

    # --- Phase 11: Resolve KV cache ---
    # KV cache is a static configuration, allocated once and reused for all forward calls.
    # If use_paged_kv_cache=True, the cache is managed externally (e.g., by vLLM) and not created here.
    if not config.use_vllm_paged_kv_cache:
        n_local_kv_heads = n_kv_heads // num_devices
        kv_cache_dtype = config.kv_cache_dtype
        kv_cache_defaults = dict(
            device=mesh_device,
            dtype=kv_cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper_config=ttnn.MeshMapperConfig(
                placements=[ttnn.PlacementReplicate()],
                mesh_shape_override=ttnn.MeshShape([num_devices]),
            ),
        )

        # Validate paged attention has enough blocks for the specified token budget
        if config.paged_attention_config is not None:
            paged_cfg = config.paged_attention_config
            block_size = paged_cfg.block_size
            max_num_blocks = paged_cfg.max_num_blocks
            # Each user needs ceil(max_seq_len / block_size) blocks
            blocks_per_user = (config.max_seq_len + block_size - 1) // block_size
            required_blocks = blocks_per_user * config.max_batch_size
            paged_cache_max_seq_len = (block_size * max_num_blocks) // config.max_batch_size

            if required_blocks > max_num_blocks:
                raise ValueError(
                    f"Paged attention block budget exceeded: "
                    f"max_batch_size ({config.max_batch_size}) × "
                    f"ceil(max_seq_len ({config.max_seq_len}) / block_size ({block_size})) = "
                    f"{required_blocks} blocks required, but max_num_blocks is only {max_num_blocks}. "
                    f"With current config, max supported seq_len is {paged_cache_max_seq_len}. "
                    f"Either increase max_num_blocks or reduce max_seq_len/max_batch_size."
                )

        if config.kv_cache is None:
            # Create default kv_cache LazyWeights
            if config.paged_attention_config:
                cache_k = zeros_like_paged_cache(config.paged_attention_config, n_local_kv_heads, head_dim)
                cache_v = zeros_like_paged_cache(config.paged_attention_config, n_local_kv_heads, head_dim)
            else:
                cache_k = zeros_like_kv_cache(config.max_batch_size, n_local_kv_heads, config.max_seq_len, head_dim)
                cache_v = zeros_like_kv_cache(config.max_batch_size, n_local_kv_heads, config.max_seq_len, head_dim)
            kv_cache = (LazyWeight(source=cache_k), LazyWeight(source=cache_v))
        else:
            kv_cache = config.kv_cache

        # Resolve defaults for LazyWeights; pass through pre-allocated tensors as-is
        resolved = []
        for i, entry in enumerate(kv_cache):
            if isinstance(entry, LazyWeight):
                resolved.append(resolve_lazy_weight(entry, **kv_cache_defaults))
            elif isinstance(entry, ttnn.Tensor):
                resolved.append(entry)
            else:
                raise TypeError(f"kv_cache[{i}] must be LazyWeight or ttnn.Tensor, got {type(entry).__name__}")
        to_set["kv_cache"] = tuple(resolved)

    return replace(config, **to_set)


# =============================================================================
# Helper functions
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
