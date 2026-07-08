# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Llama 3.2-3B-Instruct — native stack (no ``models/tt_transformers`` imports).

Architecture: standard Llama 1D transformer, identical topology to Llama 3.1-8B.
  hidden=3072, layers=28, n_heads=24, n_kv_heads=8, head_dim=128,
  intermediate=8192, vocab=128256, rope_theta=500000, RoPE llama3-scaled.

Mesh compatibility: N150 (1×1), N300 (1×2), and T3K (1×8).

TTTv1 source for precision recipes:
  ``models/tt_transformers/tt/model_config.py :: DecodersPrecision``
  (Llama-3 group: ``accuracy()`` lines 130-159, ``performance()`` lines 208-218)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.models.executor import EagerLLMExecutor, TracedLLMExecutor
from models.common.models.llama32_3b import weight_utils
from models.common.modules.attention.attention_1d import (
    Attention1D,
    Attention1DConfig,
    _dram_matmul_config,
    _dram_shard_core_grid,
)
from models.common.modules.embedding.embedding_1d import Embedding1D, Embedding1DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig, _nearest_32
from models.common.modules.mlp.mlp_1d import MLP1D, MLP1DConfig, _dram_shard_core_grid_k_n
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig, _create_sharded_norm_program_config
from models.common.modules.rope.rope_1d import Rope1DConfig, RotarySetup1D, prepare_rot_idxs
from models.common.modules.sampling.sampling_1d import Sampling1D
from models.common.modules.tt_ccl import default_topology, get_tt_ccl
from models.common.tensor_utils import TILE_SIZE, get_padded_hidden_dim

# =============================================================================
# Helpers
# =============================================================================


def _lazy(
    tensor: torch.Tensor,
    *,
    dtype: ttnn.DataType,
    cache: tuple[Path, str] | None,
) -> LazyWeight:
    return LazyWeight(source=tensor, dtype=dtype, cache_dir_weight_name=cache)


# =============================================================================
# TransformerBlock1D — single decoder layer
# =============================================================================


@dataclass
class TransformerBlock1DConfig:
    attention_norm_config: RMSNorm1DConfig
    attention_config: Attention1DConfig
    ff_norm_config: RMSNorm1DConfig
    mlp_config: MLP1DConfig

    decode_residual_memcfg: ttnn.MemoryConfig | None = None
    prefill_residual_memcfg: ttnn.MemoryConfig | None = None
    activation_dtype: ttnn.DataType | None = None


class TransformerBlock1D(LightweightModule):
    def __init__(
        self,
        attention_norm: RMSNorm1D,
        attention: Attention1D,
        ff_norm: RMSNorm1D,
        feed_forward: MLP1D,
        decode_residual_memcfg: ttnn.MemoryConfig | None = None,
        prefill_residual_memcfg: ttnn.MemoryConfig | None = None,
        activation_dtype: ttnn.DataType | None = None,
    ):
        super().__init__()
        self.attention_norm = attention_norm
        self.attention = attention
        self.ff_norm = ff_norm
        self.feed_forward = feed_forward
        self.decode_residual_memcfg = decode_residual_memcfg
        self.prefill_residual_memcfg = prefill_residual_memcfg or ttnn.DRAM_MEMORY_CONFIG
        self.activation_dtype = activation_dtype

    @classmethod
    def from_config(cls, config: TransformerBlock1DConfig) -> TransformerBlock1D:
        return cls(
            attention_norm=RMSNorm1D.from_config(config.attention_norm_config),
            attention=Attention1D.from_config(config.attention_config),
            ff_norm=RMSNorm1D.from_config(config.ff_norm_config),
            feed_forward=MLP1D.from_config(config.mlp_config),
            decode_residual_memcfg=config.decode_residual_memcfg,
            prefill_residual_memcfg=config.prefill_residual_memcfg,
            activation_dtype=config.activation_dtype,
        )

    def decode_forward(self, x: ttnn.Tensor, current_pos, rot_mats, page_table) -> ttnn.Tensor:
        residual = x

        x = _all_gather_rmsnorm_tensor(
            self.attention_norm, x, memory_config=self.attention_norm.config.decode_memory_config
        )
        attn_in = self.attention_norm.decode_forward(x)
        attn_out = self.attention.decode_forward(attn_in, current_pos, rot_mats, page_table=page_table)
        attn_out = ttnn.to_memory_config(attn_out, self.decode_residual_memcfg)

        hidden_states = ttnn.add(residual, attn_out, memory_config=self.decode_residual_memcfg)
        residual = hidden_states

        hidden_states = _all_gather_rmsnorm_tensor(
            self.ff_norm, hidden_states, memory_config=self.ff_norm.config.decode_memory_config
        )
        hidden_states = self.ff_norm.decode_forward(hidden_states)
        ttnn.deallocate(attn_out)
        hidden_states = self.feed_forward.decode_forward(hidden_states)

        out = ttnn.add(
            residual,
            hidden_states,
            memory_config=self.decode_residual_memcfg,
            dtype=self.activation_dtype or ttnn.bfloat16,
        )
        return out

    def prefill_forward(
        self, x: ttnn.Tensor, rot_mats, user_id, page_table, chunk_page_table, chunk_start_idx, batch_size: int = 1
    ) -> ttnn.Tensor:
        # For batched prefill (batch_size > 1) x is the folded [1,1,B*S,dim] hidden state; norm,
        # residual add and MLP are row-independent so they treat B*S as one long sequence unchanged.
        # Only attention unfolds the batch axis internally (see Attention1D.prefill_forward).
        residual = x

        attn_in = self.attention_norm.prefill_forward(x)
        attn_in = _all_gather_rmsnorm_tensor(self.attention_norm, attn_in)
        attn_out = self.attention.prefill_forward(
            attn_in,
            rot_mats,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            batch_size=batch_size,
        )
        attn_out = ttnn.to_memory_config(attn_out, self.prefill_residual_memcfg)

        hidden_states = ttnn.add(residual, attn_out, memory_config=self.prefill_residual_memcfg)
        residual = hidden_states
        x.deallocate(True)

        hidden_states = self.ff_norm.prefill_forward(hidden_states)
        hidden_states = _all_gather_rmsnorm_tensor(self.ff_norm, hidden_states)
        ttnn.deallocate(attn_out)
        hidden_states = self.feed_forward.prefill_forward(hidden_states)

        out = ttnn.add(
            residual,
            hidden_states,
            memory_config=self.prefill_residual_memcfg,
            dtype=self.activation_dtype or ttnn.bfloat16,
        )
        return out

    def forward(
        self,
        x,
        current_pos=None,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        batch_size: int = 1,
    ):
        if mode == "prefill":
            return self.prefill_forward(
                x, rot_mats, user_id, page_table, chunk_page_table, chunk_start_idx, batch_size=batch_size
            )
        return self.decode_forward(x, current_pos, rot_mats, page_table)


# =============================================================================
# RMSNorm gather helper
# =============================================================================


def _all_gather_rmsnorm_tensor(
    norm: RMSNorm1D, x: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig | None = None
) -> ttnn.Tensor:
    cfg = norm.config
    if cfg.mesh_device.get_num_devices() == 1 or x.shape[-1] == cfg.weight.source.numel():
        return x

    if memory_config is None:
        memory_config = x.memory_config()

    tt_ccl = cfg.tt_ccl or get_tt_ccl(cfg.mesh_device)
    # CCL pipelining matched to the proven mistral_7b / qwen25_7b N300 recipe
    # (num_links=1, chunks_per_sync=24, num_workers_per_link=4) for parity with the
    # reference ports. Measured neutral vs the prior (num_links=2, chunks_per_sync=10,
    # num_workers_per_link=2) config on N300 batch-1 decode (perf-tuning.md §Axis-3: the
    # per-layer gather is invisible in the decode-step budget on N150/N300; it only bites
    # at T3K+ scale), but kept to avoid oversubscribing N300's single inter-chip link.
    return ttnn.experimental.all_gather_async(
        x,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
        num_links=1,
        topology=default_topology(cfg.mesh_device),
        memory_config=memory_config,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        chunks_per_sync=24,
        num_workers_per_link=4,
        num_buffers_per_channel=2,
    )


# =============================================================================
# PrecisionConfig — TTTv1-matched recipes for Llama 3.2 3B Instruct
# =============================================================================

_LOFI_COMPUTE_KERNEL_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


@dataclass(frozen=True)
class Llama32_3BPrecisionConfig:
    """Per-layer precision + math-fidelity recipe for Llama 3.2 3B Instruct.

    Two module-level recipes: :data:`LLAMA32_3B_ACCURACY` and :data:`LLAMA32_3B_PERFORMANCE`.
    Pass one to :meth:`Llama32_3BTransformer1D.from_pretrained` via ``precision=``.

    Attention compute-kernel configs are absent: TTTv1 uses HIFI2 QKV/O decode,
    HIFI4 SDPA prefill, HIFI2 SDPA decode — matching ``Attention1D``'s TTTv2 defaults.
    """

    wqkv_dtype: ttnn.DataType = ttnn.bfloat8_b
    wo_dtype: ttnn.DataType = ttnn.bfloat8_b
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat8_b

    mlp_w1_w3_dtype: ttnn.DataType = ttnn.bfloat8_b
    mlp_w2_dtype: ttnn.DataType = ttnn.bfloat8_b
    # None → MLP1D default HIFI2_FP16 (matches TTTv1 LI_FF1_FF3 / LI_FF2 accuracy)
    mlp_ff1_3_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    mlp_ff2_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None

    lm_head_dtype: ttnn.DataType = ttnn.bfloat8_b


# TTTv1 DecodersPrecision.accuracy("Llama-3.2-3B-Instruct") (model_config.py Llama-3 group):
#   wqkv=BFP8, wo=BFP8, kv_cache=BFP8, mlp_w1_w3=BFP8, mlp_w2=BFP8,
#   LI_FF1_FF3=HIFI2_FP16, LI_FF2=HIFI2_FP16, SDPA_prefill=HIFI4 (Attention1D default)
LLAMA32_3B_ACCURACY = Llama32_3BPrecisionConfig()

# TTTv1 DecodersPrecision.performance("Llama-3.2-3B-Instruct"):
#   FF1_FF3 → BFP4, LI_FF1_FF3 → LOFI; all other fields same as accuracy.
LLAMA32_3B_PERFORMANCE = Llama32_3BPrecisionConfig(
    mlp_w1_w3_dtype=ttnn.bfloat4_b,
    mlp_ff1_3_compute_kernel_cfg=_LOFI_COMPUTE_KERNEL_CFG,
)


# =============================================================================
# Runtime configs
# =============================================================================


@dataclass
class Llama32_3BExecutorRuntimeConfig:
    """Engine-facing runtime knobs. Exposed as ``model.model_args`` for ``EagerLLMExecutor``."""

    n_layers: int
    n_kv_heads: int
    head_dim: int
    max_batch_size: int
    max_seq_len: int
    cluster_shape: list[int]
    max_prefill_chunk_size: int = 2048
    model_cache_path: Path | None = None
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    optimizations: Any = None
    # Batched prefill (parity caveat #12): fuse equal-length users into batched passes to close the
    # batch-32 TTFT gap. ``supports_batched_prefill`` is the per-model opt-in (the shared engine only
    # batches models whose prefill_forward threads ``batch_size``). ``max_prefill_batch_size`` caps the
    # per-group batch (8 = partial batching, design rec); ``disable_batched_prefill`` is the escape
    # hatch back to the sequential loop.
    supports_batched_prefill: bool = True
    max_prefill_batch_size: int = 8
    disable_batched_prefill: bool = False
    # When True (default), batched prefill runs norm+lm_head ONCE per group over the gathered last-token
    # rows (TTTv1 parity); False falls back to the bit-identical per-slot path (one lm_head per user).
    batched_prefill_batched_extract: bool = True

    def can_enable_trace(self, prefill_seq_len: int, num_cached_tokens: int = 0) -> bool:
        # Mirror TTTv1's prefill-trace gate (model_config.get_trace_prefill_supported_seq_lens):
        # only trace the seq lens TTTv1 lists — bigger seq lens already have small op2op gaps, so
        # tracing buys nothing. TTTv1 gives Llama-3.2-3B a model-specific override of N150 -> []
        # (no traced prefill on N150) and falls back to the family default for N300/T3K
        # ([128, 1024]). So: N150 -> none, N300/T3K -> {128, 1024}. Decode trace remains enabled at
        # the engine layer regardless.
        num_devices = int(self.cluster_shape[0]) * int(self.cluster_shape[1])
        allowed = {1: (), 2: (128, 1024), 8: (128, 1024)}.get(num_devices, ())
        return (
            prefill_seq_len in allowed
            and prefill_seq_len <= self.max_prefill_chunk_size
            and prefill_seq_len <= self.max_seq_len
        )


@dataclass
class Llama32_3BPagedAttentionConfig:
    """Paged KV layout for ``Attention1D`` (duck-typed; matches Attention1D's expected interface)."""

    block_size: int
    max_num_blocks: int


@dataclass
class Llama32_3BConfig:
    """Resolved hyper-parameters for a loaded HF Llama-3.2-3B-Instruct checkpoint."""

    hf_model_id: str
    dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    num_hidden_layers: int
    max_batch_size: int
    max_seq_len: int
    rope_table_len: int


# =============================================================================
# Wormhole tuning
# =============================================================================


@dataclass
class _Llama32_3BWHTuning:
    mlp_prefill_len_cutoff: int | None = None
    mlp_decode_spill_w1_to_dram: bool = False
    # Use ttnn.experimental.minimal_matmul for QKV + W2 prefill matmuls above seq_len > 128 (TTTv1
    # parity, PLAN_01). A/B escape hatch: set DISABLE_MINIMAL_MATMUL=1 to force ttnn.linear. Kept on
    # for consistency with the 1B/family ports + long-prompt prefill; ~parity on short prompts.
    prefill_minimal_matmul: bool = True


def _resolve_llama32_3b_wh_tuning(*, num_dev: int, max_batch_size: int) -> _Llama32_3BWHTuning:
    """Pick WH tuning knobs: prefill_len_cutoff=512 on N150, 1024 on N300 (matches TTTv1)."""
    t = _Llama32_3BWHTuning()
    t.mlp_prefill_len_cutoff = 512 if num_dev == 1 else 1024
    t.mlp_decode_spill_w1_to_dram = False
    t.prefill_minimal_matmul = not os.environ.get("DISABLE_MINIMAL_MATMUL")
    logger.info(
        f"MLP tuning for Llama-3.2-3B on {num_dev} device(s): "
        f"prefill_len_cutoff={t.mlp_prefill_len_cutoff}, "
        f"decode_spill_w1_to_dram={t.mlp_decode_spill_w1_to_dram}"
    )
    return t


# =============================================================================
# Layer + head builders
# =============================================================================


def _post_attn_norm_decode_configs(
    mlp: MLP1D,
    *,
    dim: int,
    hidden_dim: int,
    num_devices: int,
    max_batch_size: int,
) -> tuple[Any, ttnn.MemoryConfig]:
    """Resolve post-attention RMSNorm decode sharding to match MLP1D W1/W3 input."""
    padded_hidden = get_padded_hidden_dim(hidden_dim, num_devices, TILE_SIZE)
    grid = _dram_shard_core_grid_k_n(dim, padded_hidden // num_devices)
    tile_padded_batch_rows = TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE)
    program_config = _create_sharded_norm_program_config(dim, grid, tile_padded_batch_rows, TILE_SIZE)
    return program_config, mlp.config.decode_input_memcfg


def _build_decoder_layer(
    *,
    idx: int,
    hf_layer: Any,
    mcfg: Llama32_3BConfig,
    mesh_device: ttnn.MeshDevice,
    tt_ccl: Any,
    topology: Any,
    num_dev: int,
    torch_dtype: torch.dtype,
    precision: Llama32_3BPrecisionConfig,
    executor_mode: bool,
    paged_block_size: int | None,
    paged_max_blocks: int | None,
    cache_path: Path | None,
    wh: _Llama32_3BWHTuning,
    decode_residual_memcfg: ttnn.MemoryConfig,
) -> TransformerBlock1D:
    prefix = f"layer{idx}"

    wqkv, wo = weight_utils.attention_wqkv_wo_from_hf_layer(hf_layer.self_attn, num_dev)
    lazy_wqkv = _lazy(
        wqkv, dtype=precision.wqkv_dtype, cache=(cache_path / "attn", f"{prefix}_wqkv") if cache_path else None
    )
    lazy_wo = _lazy(wo, dtype=precision.wo_dtype, cache=(cache_path / "attn", f"{prefix}_wo") if cache_path else None)

    paged_cfg = None
    if executor_mode and paged_block_size is not None and paged_max_blocks is not None:
        paged_cfg = Llama32_3BPagedAttentionConfig(block_size=paged_block_size, max_num_blocks=paged_max_blocks)

    attn = Attention1D.from_config(
        Attention1DConfig(
            wqkv=lazy_wqkv,
            wo=lazy_wo,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            topology=topology,
            n_heads=mcfg.n_heads,
            n_kv_heads=mcfg.n_kv_heads,
            head_dim=mcfg.head_dim,
            max_batch_size=mcfg.max_batch_size,
            max_seq_len=mcfg.max_seq_len,
            use_vllm_paged_kv_cache=executor_mode,
            paged_attention_config=paged_cfg,
            kv_cache=None,
            kv_cache_dtype=precision.kv_cache_dtype,
            # TTTv1 parity: Llama-3 family decode SDPA runs HIFI2 with exp_approx_mode=True
            # (model_config.py `_default_settings` → SDPA_DECODE=HIFI2, used in BOTH accuracy
            # and performance). Attention1D's generic default builds this prog config with
            # exp_approx_mode=False, leaving decode SDPA slower than TTTv1. Flip it to match.
            decode_sdpa_prg_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                exp_approx_mode=True,
                q_chunk_size=0,
                k_chunk_size=0,
            ),
            prefill_qkv_minimal_matmul=wh.prefill_minimal_matmul,
        )
    )

    w1, w2, w3 = weight_utils.mlp_weights_from_hf_layer(hf_layer.mlp)
    mlp = MLP1D.from_config(
        MLP1DConfig(
            w1=_lazy(
                w1,
                dtype=precision.mlp_w1_w3_dtype,
                cache=(cache_path / "mlp", f"{prefix}_w1") if cache_path else None,
            ),
            w2=_lazy(
                w2,
                dtype=precision.mlp_w2_dtype,
                cache=(cache_path / "mlp", f"{prefix}_w2") if cache_path else None,
            ),
            w3=_lazy(
                w3,
                dtype=precision.mlp_w1_w3_dtype,
                cache=(cache_path / "mlp", f"{prefix}_w3") if cache_path else None,
            ),
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            topology=topology,
            max_batch_size=mcfg.max_batch_size,
            prefill_len_cutoff=wh.mlp_prefill_len_cutoff,
            decode_spill_w1_to_dram_before_w3=wh.mlp_decode_spill_w1_to_dram,
            w1_w3_dtype=precision.mlp_w1_w3_dtype,
            w2_dtype=precision.mlp_w2_dtype,
            ff1_3_compute_kernel_cfg=precision.mlp_ff1_3_compute_kernel_cfg,
            decode_ff1_3_compute_kernel_cfg=precision.mlp_ff1_3_compute_kernel_cfg,
            ff2_compute_kernel_cfg=precision.mlp_ff2_compute_kernel_cfg,
            decode_ff2_compute_kernel_cfg=precision.mlp_ff2_compute_kernel_cfg,
            prefill_w2_minimal_matmul=wh.prefill_minimal_matmul,
        )
    )

    post_attn_decode_program_config, post_attn_decode_memory_config = _post_attn_norm_decode_configs(
        mlp,
        dim=mcfg.dim,
        hidden_dim=mcfg.hidden_dim,
        num_devices=num_dev,
        max_batch_size=mcfg.max_batch_size,
    )

    def _build_norm(hf_norm: Any, name: str, **extra: Any) -> RMSNorm1D:
        lw = _lazy(
            weight_utils.rms_weight_torch(hf_norm).to(torch_dtype),
            dtype=ttnn.bfloat16,
            cache=(cache_path / "norm", f"{prefix}_{name}") if cache_path else None,
        )
        return RMSNorm1D.from_config(
            RMSNorm1DConfig(
                weight=lw,
                mesh_device=mesh_device,
                eps=mcfg.rms_norm_eps,
                max_batch_size=mcfg.max_batch_size,
                tt_ccl=tt_ccl,
                **extra,
            )
        )

    attn_norm = _build_norm(hf_layer.input_layernorm, "pre_attn")
    ff_norm = _build_norm(
        hf_layer.post_attention_layernorm,
        "post_attn",
        decode_program_config=post_attn_decode_program_config,
        decode_memory_config=post_attn_decode_memory_config,
    )

    return TransformerBlock1D(
        attention_norm=attn_norm,
        attention=attn,
        ff_norm=ff_norm,
        feed_forward=mlp,
        decode_residual_memcfg=decode_residual_memcfg,
        prefill_residual_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        activation_dtype=None,
    )


def _build_lm_head(
    *,
    mesh_device: ttnn.MeshDevice,
    hf_lm_head: torch.nn.Module,
    mcfg: Llama32_3BConfig,
    lm_head_dtype: ttnn.DataType,
    cache_path: Path | None,
) -> LMHead1D:
    lm_w = hf_lm_head.weight.detach().to(torch.bfloat16).clone()
    lm_splits, lm_split_sizes, lm_weights_memcfgs = weight_utils.build_lm_head_lazy_weights(
        mesh_device,
        lm_w,
        dim=mcfg.dim,
        vocab_size=mcfg.vocab_size,
        dtype=lm_head_dtype,
        cache_dir=cache_path / "lm_head" if cache_path else None,
    )
    lm_head_core_grid = _dram_shard_core_grid(mcfg.dim)
    tile = ttnn.TILE_SIZE
    tile_padded_batch_rows = tile * math.ceil(mcfg.max_batch_size / tile)
    lm_prog_configs = [
        _dram_matmul_config(tile_padded_batch_rows, mcfg.dim, ss, lm_head_core_grid.num_cores) for ss in lm_split_sizes
    ]
    lm_input_memcfg = ttnn.create_sharded_memory_config(
        (tile_padded_batch_rows, _nearest_32(mcfg.dim // lm_head_core_grid.num_cores)),
        lm_head_core_grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return LMHead1D.from_config(
        LMHead1DConfig(
            output_weights=lm_splits,
            mesh_device=mesh_device,
            dim=mcfg.dim,
            max_batch_size=mcfg.max_batch_size,
            lm_head_dtype=lm_head_dtype,
            program_configs=lm_prog_configs,
            compute_kernel_config=None,
            input_memcfg=lm_input_memcfg,
            weights_memcfgs=lm_weights_memcfgs,
        )
    )


# =============================================================================
# Llama32_3BTransformer1D
# =============================================================================


class Llama32_3BTransformer1D(LightweightModule):
    """TTTv2 Llama 3.2-3B-Instruct transformer.

    Construct via :meth:`from_pretrained`. Sub-modules (``embedding``, ``rope_setup``,
    ``layers``, ``norm``, ``lm_head``) are pre-built from HF weights with ``LazyWeight``.

    Bind KV cache with :meth:`set_kv_cache` before first executor forward.
    ``model_args`` is set to a :class:`Llama32_3BExecutorRuntimeConfig` when
    ``executor_mode=True``.
    """

    decode_residual_memcfg = ttnn.DRAM_MEMORY_CONFIG

    def __init__(
        self,
        cfg: Llama32_3BConfig,
        embedding: Embedding1D,
        rope_setup: RotarySetup1D,
        layers: List[TransformerBlock1D],
        norm: RMSNorm1D,
        lm_head: LMHead1D,
        mesh_device: ttnn.MeshDevice,
    ):
        super().__init__()
        self.cfg = cfg
        self.embedding = embedding
        self.rope_setup = rope_setup
        self.layers = layers
        self.norm = norm
        self.lm_head = lm_head
        self.mesh_device = mesh_device
        self.model_args: Llama32_3BExecutorRuntimeConfig | None = None

        self.vocab_size = cfg.vocab_size
        self.n_layers = cfg.num_hidden_layers
        self.num_devices = mesh_device.get_num_devices()
        self.tt_ccl = get_tt_ccl(mesh_device) if self.num_devices > 1 else None
        self.activation_dtypes = [None] * cfg.num_hidden_layers

        # EXPERIMENTAL (N150/N300 on-device-sampling evidence; see the on-device sampling handoff notes):
        # the prior `num_devices >= 8` gate assumed sub-8-device sampling could not be trace-captured.
        # test_sampling1d_trace_capture disproves that — argmax + top-k both capture/replay on 1x1 and
        # 1x2 — so the gate is relaxed here to all 1D meshes. The model owns its sampler; callers pick
        # behavior per request via sampling_params. Buffers are lazy (nothing materializes until the
        # first on-device sampled decode), so this is harmless when sampling_params is None.
        self.supports_on_device_sampling = self.num_devices >= 1
        self.sampling = (
            Sampling1D(
                vocab_size=self.vocab_size,
                mesh_device=mesh_device,
                tt_ccl=self.tt_ccl,
                max_batch_size=_nearest_32(cfg.max_batch_size),
                # Clone TTTv1's decision: default_sampling_force_argmax.allow_force_argmax=False for
                # all non-Galaxy meshes (only Llama-3.1-8B on TG flips it True). The perf recipe
                # (temp=0, top_p=0.08, top_k=32) routes through the cheap top-k op path — per-device
                # ttnn.topk -> all-gather of the [*,32] tuples -> ttnn.sampling — never the full-vocab
                # argmax all-gather. See models/tt_transformers/tt/model_config.py:1007.
                allow_force_argmax=False,
                pad_to_power_of_2=True,
            )
            if self.supports_on_device_sampling
            else None
        )

    @classmethod
    def from_pretrained(
        cls,
        mesh_device: ttnn.MeshDevice,
        hf_model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
        *,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        num_layers: int | None = None,
        cache_dir: Path | str | None = None,
        precision: Llama32_3BPrecisionConfig = LLAMA32_3B_ACCURACY,
        block_size: int = 32,
        executor_mode: bool = False,
    ) -> Llama32_3BTransformer1D:
        """Load HF weights and build TTNN modules (weights materialize on first forward).

        Args:
            mesh_device: Open mesh device (N150 ``(1,1)``, N300 ``(1,2)``, T3K ``(1,8)``).
            hf_model_id: Hugging Face hub id or local path.
            max_batch_size: Decode batch / KV allocation.
            max_seq_len: KV cache sequence budget per layer.
            num_layers: If set, truncate stack for smoke tests (``LLAMA32_3B_DEMO_NUM_LAYERS``).
            cache_dir: Directory for ``LazyWeight`` tensor caches.
            precision: Per-layer precision recipe. Defaults to :data:`LLAMA32_3B_ACCURACY`.
            block_size: Paged attention block size.
            executor_mode: If True, use external paged KV (``set_kv_cache`` + shared executor).
        """
        ttnn.SetDefaultDevice(mesh_device)
        cache_path = Path(cache_dir) if cache_dir else None
        num_dev = mesh_device.get_num_devices()
        tt_ccl = get_tt_ccl(mesh_device) if num_dev > 1 else None
        topology = default_topology(mesh_device)

        hf_cfg = AutoConfig.from_pretrained(hf_model_id)
        n_heads_hf = hf_cfg.num_attention_heads
        n_kv_hf = hf_cfg.num_key_value_heads
        if num_dev > 1 and (n_heads_hf % num_dev != 0 or n_kv_hf % num_dev != 0):
            raise ValueError(
                f"Checkpoint requires n_heads ({n_heads_hf}) and n_kv_heads ({n_kv_hf}) "
                f"each divisible by device count ({num_dev})."
            )

        torch_dtype = torch.bfloat16
        logger.info(f"Loading HF weights: {hf_model_id}")
        hf = AutoModelForCausalLM.from_pretrained(hf_model_id, torch_dtype=torch_dtype)
        hf.eval()
        base = hf.model

        n_layers = num_layers if num_layers is not None else hf_cfg.num_hidden_layers
        dim = hf_cfg.hidden_size
        n_heads = hf_cfg.num_attention_heads
        n_kv = hf_cfg.num_key_value_heads
        head_dim = dim // n_heads
        inter = hf_cfg.intermediate_size
        vocab = hf_cfg.vocab_size
        rope_len = max(max_seq_len * 2, 8192)
        rope_len = (rope_len + 127) // 128 * 128

        blocks_per_user = (max_seq_len + block_size - 1) // block_size
        max_num_blocks = blocks_per_user * max_batch_size

        mcfg = Llama32_3BConfig(
            hf_model_id=hf_model_id,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv,
            head_dim=head_dim,
            hidden_dim=inter,
            vocab_size=vocab,
            rms_norm_eps=hf_cfg.rms_norm_eps,
            rope_theta=getattr(hf_cfg, "rope_theta", 500_000.0),
            num_hidden_layers=n_layers,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            rope_table_len=rope_len,
        )

        emb_src = weight_utils.embed_tokens_torch(base.embed_tokens)
        embedding = Embedding1D.from_config(
            Embedding1DConfig(
                weights=_lazy(
                    emb_src,
                    dtype=ttnn.bfloat16,
                    cache=(cache_path / "embedding", "tok_embeddings") if cache_path else None,
                ),
                mesh_device=mesh_device,
                embed_scale=1.0,
            )
        )

        cos_t, sin_t = weight_utils.build_rope_cos_sin_torch(base.rotary_emb, rope_len, head_dim, torch_dtype)
        cos_lw = _lazy(cos_t, dtype=ttnn.bfloat16, cache=(cache_path / "rope", "cos") if cache_path else None)
        sin_lw = _lazy(sin_t, dtype=ttnn.bfloat16, cache=(cache_path / "rope", "sin") if cache_path else None)
        rope_setup = RotarySetup1D.from_config(
            Rope1DConfig(
                cos_matrix=cos_lw,
                sin_matrix=sin_lw,
                max_batch_size=max_batch_size,
                head_dim=head_dim,
                device=mesh_device,
                use_qk_fused=False,
            )
        )

        wh = _resolve_llama32_3b_wh_tuning(num_dev=num_dev, max_batch_size=max_batch_size)

        layers: list[TransformerBlock1D] = [
            _build_decoder_layer(
                idx=idx,
                hf_layer=base.layers[idx],
                mcfg=mcfg,
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                topology=topology,
                num_dev=num_dev,
                torch_dtype=torch_dtype,
                precision=precision,
                executor_mode=executor_mode,
                paged_block_size=block_size,
                paged_max_blocks=max_num_blocks,
                cache_path=cache_path,
                wh=wh,
                decode_residual_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            )
            for idx in range(n_layers)
        ]

        norm_lw = _lazy(
            weight_utils.rms_weight_torch(base.norm).to(torch_dtype),
            dtype=ttnn.bfloat16,
            cache=(cache_path / "norm", "final") if cache_path else None,
        )
        final_norm = RMSNorm1D.from_config(
            RMSNorm1DConfig(
                weight=norm_lw,
                mesh_device=mesh_device,
                eps=hf_cfg.rms_norm_eps,
                max_batch_size=max_batch_size,
                tt_ccl=tt_ccl,
            )
        )

        lm = _build_lm_head(
            mesh_device=mesh_device,
            hf_lm_head=hf.lm_head,
            mcfg=mcfg,
            lm_head_dtype=precision.lm_head_dtype,
            cache_path=cache_path,
        )

        del hf

        model = cls(mcfg, embedding, rope_setup, layers, final_norm, lm, mesh_device)
        if executor_mode:
            model.model_args = Llama32_3BExecutorRuntimeConfig(
                n_layers=n_layers,
                n_kv_heads=n_kv,
                head_dim=head_dim,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                cluster_shape=list(mesh_device.shape),
                model_cache_path=cache_path,
                kv_cache_dtype=precision.kv_cache_dtype,
            )
        return model

    # =========================================================================
    # KV cache binding
    # =========================================================================

    def set_kv_cache(self, kv_cache: list) -> None:
        assert len(kv_cache) == len(
            self.layers
        ), f"kv_cache has {len(kv_cache)} entries but model has {len(self.layers)} layers"
        for i, layer in enumerate(self.layers):
            layer.attention.config.kv_cache = tuple(kv_cache[i])

    # =========================================================================
    # Forward methods — take pre-embedded tensors
    # =========================================================================

    def decode_forward(
        self,
        x_embed: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        x = x_embed
        for i, layer in enumerate(self.layers):
            x = ttnn.to_memory_config(x, self.decode_residual_memcfg, self.activation_dtypes[i])
            x = layer.decode_forward(x, current_pos, rot_mats, page_table)

        x = _all_gather_rmsnorm_tensor(self.norm, x, memory_config=self.norm.config.decode_memory_config)
        x = self.norm.decode_forward(x)
        x = self.lm_head.forward(x)
        return x

    def prefill_forward(
        self,
        x_embed: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int = 0,
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        get_last_token: int = -1,
        batch_size: int = 1,
    ) -> ttnn.Tensor:
        # batch_size > 1: x_embed is the folded [1,1,B*S,dim] tensor (B users). The batched path always
        # returns the full hidden state (get_last_token == -1); the executor does per-slot last-token
        # extraction + norm/lm_head so those stages stay bit-identical to the single-user path.
        x = x_embed
        for i, layer in enumerate(self.layers):
            activation_dtype = self.activation_dtypes[i]
            if activation_dtype is not None and x.dtype != activation_dtype:
                old = x
                x = ttnn.typecast(x, activation_dtype)
                ttnn.deallocate(old)
            x = layer.prefill_forward(x, rot_mats, user_id, page_table, chunk_page_table, chunk_start_idx, batch_size)

        if get_last_token == -1:
            return x

        get_last_token_floor = (get_last_token // 32) * 32
        old = x
        x = ttnn.slice(x, (0, 0, get_last_token_floor, 0), (1, 1, get_last_token_floor + 32, x.shape[-1]))
        ttnn.deallocate(old)

        x = self.norm.prefill_forward(x)
        x = _all_gather_rmsnorm_tensor(self.norm, x)
        lm_head_memcfg = self.lm_head.config.input_memcfg
        if lm_head_memcfg is not None and lm_head_memcfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_memcfg)
        x = self.lm_head.forward(x)
        return ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

    def post_process_prefill_output(self, hidden_states: ttnn.Tensor, last_token_idx: int) -> ttnn.Tensor:
        get_last_token_floor = (last_token_idx // 32) * 32
        x = ttnn.slice(
            hidden_states,
            (0, 0, get_last_token_floor, 0),
            (1, 1, get_last_token_floor + 32, hidden_states.shape[-1]),
        )
        x = self.norm.prefill_forward(x)
        x = _all_gather_rmsnorm_tensor(self.norm, x)
        lm_head_memcfg = self.lm_head.config.input_memcfg
        if lm_head_memcfg is not None and lm_head_memcfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_memcfg)
        x = self.lm_head.forward(x)
        return ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos=None,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id: int = 0,
        mode: str = "decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token: int = -1,
        batch_size: int = 1,
    ) -> ttnn.Tensor:
        rot_mats = rot_mats_global
        if mode == "prefill":
            return self.prefill_forward(
                x,
                rot_mats,
                user_id=user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                get_last_token=get_last_token,
                batch_size=batch_size,
            )
        return self.decode_forward(x, current_pos, rot_mats, page_table=page_table)

    # =========================================================================
    # Embedding + output processing helpers (executor contract)
    # =========================================================================

    def embed_decode(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        x = self.embedding.forward(tokens)
        x = ttnn.unsqueeze_to_4D(x)
        return ttnn.to_memory_config(x, self.decode_residual_memcfg)

    def embed_prefill(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        x = self.embedding.forward(tokens)
        return ttnn.unsqueeze_to_4D(x)

    def gather_and_untilize_logits(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        if self.num_devices > 1 and self.tt_ccl is not None:
            logits = ttnn.experimental.all_gather_async(
                logits,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=1,
                memory_config=logits.memory_config(),
                topology=default_topology(self.mesh_device),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
        return ttnn.untilize(logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def increment_positions(self, current_pos: ttnn.Tensor, rot_mat_idxs: ttnn.Tensor) -> None:
        ttnn.plus_one(current_pos, skip_negative_entries=True)
        ttnn.plus_one(rot_mat_idxs)

    # =========================================================================
    # Smoke-test helpers (no executor, no page_table)
    # =========================================================================

    def prefill_from_token_ids(self, token_ids_tt: ttnn.Tensor, *, start_pos: int = 0, user_id: int = 0) -> ttnn.Tensor:
        x = self.embed_prefill(token_ids_tt)
        seq_len = x.shape[2]
        assert seq_len % 128 == 0, "prefill seq_len must be divisible by 128"
        rot = self.rope_setup.prefill_forward(start_pos, seq_len)
        h = x
        for layer in self.layers:
            h = layer.prefill_forward(
                h, rot, user_id=user_id, page_table=None, chunk_page_table=None, chunk_start_idx=None
            )
        h = self.norm.prefill_forward(h)
        return _all_gather_rmsnorm_tensor(self.norm, h)

    def decode_from_token_ids(self, token_ids_tt: ttnn.Tensor, *, current_pos: int) -> ttnn.Tensor:
        x = self.embedding.forward(token_ids_tt)
        x = ttnn.unsqueeze_to_4D(x)
        pos = torch.tensor([current_pos], dtype=torch.long)
        rot_idxs = prepare_rot_idxs(self.rope_setup.config, pos, on_host=False)
        rot = self.rope_setup.decode_forward(rot_idxs)
        cur = ttnn.from_torch(
            pos,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(self.mesh_device),
        )
        h = x
        for layer in self.layers:
            h = layer.decode_forward(h, cur, rot, page_table=None)
        h = _all_gather_rmsnorm_tensor(self.norm, h, memory_config=self.norm.config.decode_memory_config)
        return self.norm.decode_forward(h)


# =============================================================================
# Executor validation traversal
# =============================================================================


def _iter_llama_executor_named_modules(model):
    """Yield named submodules that declare executor input contracts."""
    if not hasattr(model, "layers"):
        return

    for i, layer in enumerate(model.layers):
        for suffix, submodule in [
            ("attn_norm", getattr(layer, "attention_norm", None)),
            ("attention", getattr(layer, "attention", None)),
            ("ff_norm", getattr(layer, "ff_norm", None)),
            ("mlp", getattr(layer, "feed_forward", None)),
        ]:
            if submodule is not None:
                yield f"layer[{i}].{suffix}", submodule

    if hasattr(model, "norm"):
        yield "final_norm", model.norm
    if hasattr(model, "lm_head"):
        yield "lm_head", model.lm_head


# =============================================================================
# EagerLlama32_3BExecutor — thin wrapper
# =============================================================================


class EagerLlama32_3BExecutor:
    """Thin wrapper: passes Llama32_3B model to EagerLLMExecutor."""

    def __init__(self, model: Llama32_3BTransformer1D, mesh_device: ttnn.MeshDevice, model_args=None):
        if model_args is not None:
            model.model_args = model_args
        self._engine = EagerLLMExecutor(model, mesh_device, iter_named_modules=_iter_llama_executor_named_modules)

    @property
    def model(self):
        return self._engine.model

    @property
    def mesh_device(self):
        return self._engine.mesh_device

    @property
    def model_args(self):
        return self._engine.model_args

    @property
    def mode(self):
        return self._engine.mode

    @mode.setter
    def mode(self, value):
        self._engine.mode = value

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        return self._engine.allocate_kv_cache(kv_cache_shape, dtype, num_layers)

    def _assert_kv_cache_identity(self, kv_cache):
        return self._engine._assert_kv_cache_identity(kv_cache)

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table):
        return self._engine.prepare_decode_inputs_host(tokens, current_pos, page_table)

    def prepare_decode_inputs_device(self, tokens, current_pos, page_table):
        return self._engine.prepare_decode_inputs_device(tokens, current_pos, page_table)

    def compile_prefill(
        self,
        *,
        tokens,
        page_table,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        start_pos=None,
        sampling_params=None,
    ):
        return self._engine.compile_prefill(
            tokens=tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            start_pos=start_pos,
            sampling_params=sampling_params,
        )

    def compile_decode(self, *, tokens, start_pos, page_table, kv_cache=None, sampling_params=None):
        return self._engine.compile_decode(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        sampling_params=None,
        start_pos=None,
    ):
        return self._engine.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            sampling_params=sampling_params,
            start_pos=start_pos,
        )

    def _prefill_single_user(self, tokens, page_table, user_id, last_token_idx, num_cached_tokens=0):
        return self._engine._prefill_single_user(tokens, page_table, user_id, last_token_idx, num_cached_tokens)

    def decode_forward(self, tokens, start_pos, page_table, kv_cache=None, read_from_device=True, sampling_params=None):
        return self._engine.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            read_from_device=read_from_device,
            sampling_params=sampling_params,
        )

    def cleanup(self):
        return self._engine.cleanup()


# =============================================================================
# TracedLlama32_3BExecutor — thin wrapper
# =============================================================================


class TracedLlama32_3BExecutor:
    """Thin wrapper: passes Llama32_3B model to TracedLLMExecutor."""

    def __init__(self, model: Llama32_3BTransformer1D, mesh_device: ttnn.MeshDevice, model_args=None):
        if model_args is not None:
            model.model_args = model_args
        self._engine = TracedLLMExecutor(model, mesh_device, iter_named_modules=_iter_llama_executor_named_modules)

    @property
    def model(self):
        return self._engine.model

    @property
    def mesh_device(self):
        return self._engine.mesh_device

    @property
    def model_args(self):
        return self._engine.model_args

    @property
    def mode(self):
        return self._engine.mode

    @mode.setter
    def mode(self, value):
        self._engine.mode = value

    @property
    def trace_id_prefill(self):
        return self._engine.trace_id_prefill

    @property
    def trace_ids_decode(self):
        return self._engine.trace_ids_decode

    @property
    def already_warmed_up_prefill(self):
        return self._engine.already_warmed_up_prefill

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        return self._engine.allocate_kv_cache(kv_cache_shape, dtype, num_layers)

    def warmup_model_prefill(self, seq_lens, make_tokens, make_page_table):
        return self._engine.warmup_model_prefill(seq_lens, make_tokens, make_page_table)

    def compile_prefill(
        self,
        *,
        tokens,
        page_table,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        start_pos=None,
        sampling_params=None,
    ):
        return self._engine.compile_prefill(
            tokens=tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            start_pos=start_pos,
            sampling_params=sampling_params,
        )

    def compile_decode(self, *, tokens, start_pos, page_table, kv_cache=None, sampling_params=None):
        return self._engine.compile_decode(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )

    def prefill_forward(
        self,
        tokens,
        page_table,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        sampling_params=None,
        start_pos=None,
    ):
        return self._engine.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            sampling_params=sampling_params,
            start_pos=start_pos,
        )

    def decode_forward(self, tokens, start_pos, page_table, kv_cache=None, read_from_device=True, sampling_params=None):
        return self._engine.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            read_from_device=read_from_device,
            sampling_params=sampling_params,
        )

    def cleanup(self):
        return self._engine.cleanup()


# =============================================================================
# Public exports
# =============================================================================

__all__ = [
    # Precision recipes
    "Llama32_3BPrecisionConfig",
    "LLAMA32_3B_ACCURACY",
    "LLAMA32_3B_PERFORMANCE",
    # Runtime config
    "Llama32_3BExecutorRuntimeConfig",
    "Llama32_3BConfig",
    # Model
    "Llama32_3BTransformer1D",
    # Executors
    "EagerLlama32_3BExecutor",
    "TracedLlama32_3BExecutor",
    # Building blocks
    "TransformerBlock1D",
    "TransformerBlock1DConfig",
    "_all_gather_rmsnorm_tensor",
    "_iter_llama_executor_named_modules",
]
