# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Llama 3.3-70B-Instruct — native stack (no ``models/tt_transformers`` imports).

Architecture: standard Llama 1D transformer, same topology as Llama 3.1-8B / 3.2-3B
(no QKV bias, no Q/K norm, GPT-NeoX rotate_half RoPE with llama3 scaling).
  hidden=8192, layers=80, n_heads=64, n_kv_heads=8, head_dim=128,
  intermediate=28672, vocab=128256, rope_theta=500000, RoPE llama3-scaled (factor=8).

Mesh compatibility: T3K (1×8) only. 64 attn heads / 8 = 8 per device, 8 KV heads / 8 = 1
per device; both tile-aligned. The port raises on any other mesh.

TTTv1 source for precision recipes:
  ``models/tt_transformers/tt/model_config.py :: DecodersPrecision``.
  ``get_base_model_name("…/Llama-3.3-70B-Instruct") == "Llama-3.3-70B"`` is NOT in the
  ``Llama-3.1-70B`` special-case list (model_config.py:119), so it resolves to the generic
  Llama-3 branch — identical recipe to Llama-3.2-3B: ``accuracy()`` BFP8 attention/KV/MLP +
  HIFI2_FP16 FF / HIFI4 SDPA-prefill (lines 130-159); ``performance()`` BFP4 FF1/FF3 + LOFI
  (lines 208-218).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
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
from models.common.modules.rope.rope_1d import Rope1DConfig, RotarySetup1D
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig
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
        self,
        x: ttnn.Tensor,
        rot_mats,
        user_id,
        page_table,
        chunk_page_table,
        chunk_start_idx,
        batch_size: int = 1,
        chunk_start_idx_tensor: ttnn.Tensor | None = None,
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
            chunk_start_idx_tensor=chunk_start_idx_tensor,
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
        if memory_config is not None:
            return ttnn.to_memory_config(x, memory_config)
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
# PrecisionConfig — TTTv1-matched recipes for Llama 3.3 70B Instruct
# =============================================================================

_LOFI_COMPUTE_KERNEL_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


@dataclass(frozen=True)
class Llama33_70BPrecisionConfig:
    """Per-layer precision + math-fidelity recipe for Llama 3.3 70B Instruct.

    Two module-level recipes: :data:`LLAMA33_70B_ACCURACY` and :data:`LLAMA33_70B_PERFORMANCE`.
    The Hugging Face adaptor selects one while building this provider-neutral graph.

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


# TTTv1 DecodersPrecision.accuracy("Llama-3.3-70B-Instruct") (model_config.py Llama-3 group):
#   wqkv=BFP8, wo=BFP8, kv_cache=BFP8, mlp_w1_w3=BFP8, mlp_w2=BFP8,
#   LI_FF1_FF3=HIFI2_FP16, LI_FF2=HIFI2_FP16, SDPA_prefill=HIFI4 (Attention1D default)
LLAMA33_70B_ACCURACY = Llama33_70BPrecisionConfig()

# TTTv1 DecodersPrecision.performance("Llama-3.3-70B-Instruct"):
#   FF1_FF3 → BFP4, LI_FF1_FF3 → LOFI; all other fields same as accuracy.
LLAMA33_70B_PERFORMANCE = Llama33_70BPrecisionConfig(
    mlp_w1_w3_dtype=ttnn.bfloat4_b,
    mlp_ff1_3_compute_kernel_cfg=_LOFI_COMPUTE_KERNEL_CFG,
)


# =============================================================================
# Runtime configs
# =============================================================================


@dataclass
class Llama33_70BPagedAttentionConfig:
    """Paged KV layout for ``Attention1D`` (duck-typed; matches Attention1D's expected interface)."""

    block_size: int
    max_num_blocks: int


@dataclass
class Llama33_70BTransformer1DConfig:
    """Complete provider-neutral tensor-graph configuration for Llama 3.3 70B."""

    n_layers: int
    vocab_size: int
    max_batch_size: int
    max_seq_len: int
    dim: int
    num_devices: int
    mesh_device: ttnn.MeshDevice
    embedding_config: Embedding1DConfig
    rope_config: Rope1DConfig
    block_configs: list[TransformerBlock1DConfig]
    norm_config: RMSNorm1DConfig
    lm_head_config: LMHead1DConfig
    sampling_config: Sampling1DConfig | None = None
    decode_residual_memcfg: ttnn.MemoryConfig | None = None
    prefill_residual_memcfg: ttnn.MemoryConfig | None = None
    activation_dtypes: list[ttnn.DataType | None] = field(default_factory=list)
    tt_ccl: Any = None
    cache_path: str | None = None


@dataclass(frozen=True)
class Llama33_70BModelParameters:
    dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_dim: int
    vocab_size: int
    rms_norm_eps: float
    max_batch_size: int
    max_seq_len: int


@dataclass(frozen=True)
class Llama33_70BLayerWeights:
    wqkv: torch.Tensor
    wo: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w3: torch.Tensor
    attention_norm: torch.Tensor
    ff_norm: torch.Tensor


@dataclass(frozen=True)
class Llama33_70BWeights:
    embedding: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    layers: tuple[Llama33_70BLayerWeights, ...]
    final_norm: torch.Tensor
    lm_head: torch.Tensor


# =============================================================================
# Wormhole tuning
# =============================================================================


@dataclass
class _Llama33_70BWHTuning:
    mlp_prefill_len_cutoff: int | None = None
    mlp_decode_spill_w1_to_dram: bool = False
    # Use ttnn.experimental.minimal_matmul for QKV + W2 prefill matmuls above seq_len > 128 (TTTv1
    # parity, PLAN_01). A/B escape hatch: set DISABLE_MINIMAL_MATMUL=1 to force ttnn.linear. Kept on
    # for consistency with the 1B/3B/family ports + long-prompt prefill; inert on the 128-bucket
    # short-context workload (only fires for seq_len > 128, e.g. long prompts).
    prefill_minimal_matmul: bool = True


def _resolve_llama33_70b_wh_tuning(*, num_dev: int, max_batch_size: int) -> _Llama33_70BWHTuning:
    """Pick WH L1 tuning knobs for Llama-3.3-70B on T3K.

    The 70B FF is wide (intermediate=28672 → 3584 per device on T3K); the prefill FF1/FF3 matmul is
    chunked at ``mlp_prefill_len_cutoff=1024`` = the shared engine's own Wormhole default (``mlp_1d.py``)
    and TTTv1's ``prefill_len_cutoff``. For the folded batch-32-ci FF prefill (``[1,1,B*S,dim]``,
    B*S=4096 at the 128 bucket) this runs 4 chunks of 1024 (``per_core_M=4``) instead of 16 chunks of
    256 (``per_core_M=1``) — matching TTTv1's blocking on the ~80%-FLOP block (this is the worst model,
    so the biggest absolute win). 1024 divides every folded prefill length ≥1024 (all power-of-2
    products of pb∈{1,2,4,8,16,32} × bucket∈{128,512,1024,2048}); shorter folds skip the reshape. The
    earlier 256 was a 7B-on-N300 inheritance (per-device FF shard 9472 ≫ the T3K 70B shard 3584), so
    its tighter-L1 motive does not apply here; 1024 fits — TTTv1 runs it for this exact model on this
    box. Output is unchanged: ``in0_block_w`` / the K-contraction are independent of the M-tiling, so
    only ``per_core_M`` changes. ``decode_spill_w1_to_dram`` stays off; re-evaluate if decode batch-32
    trips L1 circular-buffer validation.
    """
    t = _Llama33_70BWHTuning()
    t.mlp_prefill_len_cutoff = 1024
    t.mlp_decode_spill_w1_to_dram = False
    t.prefill_minimal_matmul = not os.environ.get("DISABLE_MINIMAL_MATMUL")
    logger.info(
        f"MLP tuning for Llama-3.3-70B on {num_dev} device(s): "
        f"prefill_len_cutoff={t.mlp_prefill_len_cutoff}, "
        f"decode_spill_w1_to_dram={t.mlp_decode_spill_w1_to_dram}"
    )
    return t


# =============================================================================
# Layer + head builders
# =============================================================================


def _post_attn_norm_decode_configs(
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
    memory_config = ttnn.create_sharded_memory_config(
        (tile_padded_batch_rows, dim // grid.num_cores),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return program_config, memory_config


def _build_decoder_layer(
    *,
    idx: int,
    weights: Llama33_70BLayerWeights,
    mcfg: Llama33_70BModelParameters,
    mesh_device: ttnn.MeshDevice,
    tt_ccl: Any,
    topology: Any,
    num_dev: int,
    precision: Llama33_70BPrecisionConfig,
    paged_attention_config: Llama33_70BPagedAttentionConfig,
    cache_path: Path | None,
    wh: _Llama33_70BWHTuning,
    decode_residual_memcfg: ttnn.MemoryConfig,
) -> TransformerBlock1DConfig:
    prefix = f"layer{idx}"

    lazy_wqkv = _lazy(
        weights.wqkv,
        dtype=precision.wqkv_dtype,
        cache=(cache_path / "attn", f"{prefix}_wqkv") if cache_path else None,
    )
    lazy_wo = _lazy(
        weights.wo,
        dtype=precision.wo_dtype,
        cache=(cache_path / "attn", f"{prefix}_wo") if cache_path else None,
    )

    attention_config = Attention1DConfig(
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
        use_vllm_paged_kv_cache=True,
        paged_attention_config=paged_attention_config,
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

    mlp_config = MLP1DConfig(
        w1=_lazy(
            weights.w1,
            dtype=precision.mlp_w1_w3_dtype,
            cache=(cache_path / "mlp", f"{prefix}_w1") if cache_path else None,
        ),
        w2=_lazy(
            weights.w2,
            dtype=precision.mlp_w2_dtype,
            cache=(cache_path / "mlp", f"{prefix}_w2") if cache_path else None,
        ),
        w3=_lazy(
            weights.w3,
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

    post_attn_decode_program_config, post_attn_decode_memory_config = _post_attn_norm_decode_configs(
        dim=mcfg.dim,
        hidden_dim=mcfg.hidden_dim,
        num_devices=num_dev,
        max_batch_size=mcfg.max_batch_size,
    )

    def _build_norm(weight: torch.Tensor, name: str, **extra: Any) -> RMSNorm1DConfig:
        lw = _lazy(
            weight,
            dtype=ttnn.bfloat16,
            cache=(cache_path / "norm", f"{prefix}_{name}") if cache_path else None,
        )
        return RMSNorm1DConfig(
            weight=lw,
            mesh_device=mesh_device,
            eps=mcfg.rms_norm_eps,
            max_batch_size=mcfg.max_batch_size,
            tt_ccl=tt_ccl,
            **extra,
        )

    attn_norm = _build_norm(weights.attention_norm, "pre_attn")
    ff_norm = _build_norm(
        weights.ff_norm,
        "post_attn",
        decode_program_config=post_attn_decode_program_config,
        decode_memory_config=post_attn_decode_memory_config,
    )

    return TransformerBlock1DConfig(
        attention_norm_config=attn_norm,
        attention_config=attention_config,
        ff_norm_config=ff_norm,
        mlp_config=mlp_config,
        decode_residual_memcfg=decode_residual_memcfg,
        prefill_residual_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        activation_dtype=None,
    )


def _build_lm_head_lazy_weights(
    mesh_device: ttnn.MeshDevice,
    lm_head_weight: torch.Tensor,
    *,
    dim: int,
    vocab_size: int,
    max_columns_per_device: int = 8192,
    dtype: ttnn.DataType = ttnn.bfloat8_b,
    cache_dir: Path | None = None,
) -> tuple[list[LazyWeight], list[int], list[ttnn.MemoryConfig]]:
    """Build provider-neutral column-split LM-head weights."""

    num_devices = mesh_device.get_num_devices()
    torch_w = lm_head_weight.T.contiguous().to(torch.bfloat16)
    padded_vocab_size = math.ceil(vocab_size / 32) * 32
    if vocab_size < padded_vocab_size:
        torch_w = torch.cat(
            [torch_w, torch.zeros(torch_w.shape[0], padded_vocab_size - vocab_size, dtype=torch_w.dtype)], dim=-1
        )

    size_per_device = padded_vocab_size // num_devices
    num_splits = math.ceil(size_per_device / max_columns_per_device)
    split_sizes = [min(size_per_device, max_columns_per_device)] * (num_splits - 1)
    split_sizes.append(size_per_device - sum(split_sizes))
    dram_size = mesh_device.dram_grid_size()
    dram_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_size.x - 1, dram_size.y - 1))}
    )

    output_weights = []
    weights_memcfgs = []
    for split_index, split_size in enumerate(split_sizes):
        device_splits = []
        for device_index in range(num_devices):
            start = device_index * size_per_device + sum(split_sizes[:split_index])
            device_splits.append(torch_w[:, start : start + split_size])
        combined = torch.cat(device_splits, dim=-1)
        padded_n = math.ceil((combined.shape[-1] // num_devices) / (TILE_SIZE * dram_size.x)) * (
            TILE_SIZE * dram_size.x
        )
        shard_spec = ttnn.ShardSpec(
            dram_grid,
            (dim, padded_n // dram_size.x),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)
        weights_memcfgs.append(memory_config)
        output_weights.append(
            LazyWeight(
                source=combined,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper_config=ttnn.MeshMapperConfig(
                    placements=[ttnn.PlacementShard(-1)],
                    mesh_shape_override=ttnn.MeshShape([num_devices]),
                ),
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
                cache_dir_weight_name=(
                    (cache_dir, f"lm_head_split_{split_index}_{combined.shape[-1]}") if cache_dir else None
                ),
            )
        )
    return output_weights, split_sizes, weights_memcfgs


def _build_lm_head(
    *,
    mesh_device: ttnn.MeshDevice,
    lm_head_weight: torch.Tensor,
    mcfg: Llama33_70BModelParameters,
    lm_head_dtype: ttnn.DataType,
    cache_path: Path | None,
) -> LMHead1DConfig:
    lm_splits, lm_split_sizes, lm_weights_memcfgs = _build_lm_head_lazy_weights(
        mesh_device,
        lm_head_weight,
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
    return LMHead1DConfig(
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


# =============================================================================
# Llama33_70BTransformer1D
# =============================================================================


def build_llama33_70b_transformer_1d_config(
    *,
    mesh_device: ttnn.MeshDevice,
    params: Llama33_70BModelParameters,
    weights: Llama33_70BWeights,
    n_layers: int,
    precision: Llama33_70BPrecisionConfig,
    cache_path: Path,
    paged_attention_config: Llama33_70BPagedAttentionConfig,
) -> Llama33_70BTransformer1DConfig:
    """Build the TT tensor graph from provider-neutral dimensions and tensors."""

    num_devices = mesh_device.get_num_devices()
    if num_devices != 8:
        raise ValueError(f"Llama-3.3-70B supports exactly 8 devices (T3K), got {num_devices}")
    if params.n_heads % num_devices or params.n_kv_heads % num_devices:
        raise ValueError(
            f"Checkpoint heads ({params.n_heads}/{params.n_kv_heads}) must be divisible by device count ({num_devices})"
        )
    if len(weights.layers) != n_layers:
        raise ValueError(f"Expected {n_layers} decoder layer weight sets, got {len(weights.layers)}")

    tt_ccl = get_tt_ccl(mesh_device)
    topology = default_topology(mesh_device)
    embedding_config = Embedding1DConfig(
        weights=_lazy(
            weights.embedding,
            dtype=ttnn.bfloat16,
            cache=(cache_path / "embedding", "tok_embeddings"),
        ),
        mesh_device=mesh_device,
        embed_scale=1.0,
    )
    rope_config = Rope1DConfig(
        cos_matrix=_lazy(weights.rope_cos, dtype=ttnn.bfloat16, cache=(cache_path / "rope", "cos")),
        sin_matrix=_lazy(weights.rope_sin, dtype=ttnn.bfloat16, cache=(cache_path / "rope", "sin")),
        max_batch_size=params.max_batch_size,
        head_dim=params.head_dim,
        device=mesh_device,
        use_qk_fused=False,
    )
    wh = _resolve_llama33_70b_wh_tuning(num_dev=num_devices, max_batch_size=params.max_batch_size)
    block_configs = [
        _build_decoder_layer(
            idx=index,
            weights=weights.layers[index],
            mcfg=params,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            topology=topology,
            num_dev=num_devices,
            precision=precision,
            paged_attention_config=paged_attention_config,
            cache_path=cache_path,
            wh=wh,
            decode_residual_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        )
        for index in range(n_layers)
    ]
    norm_config = RMSNorm1DConfig(
        weight=_lazy(weights.final_norm, dtype=ttnn.bfloat16, cache=(cache_path / "norm", "final")),
        mesh_device=mesh_device,
        eps=params.rms_norm_eps,
        max_batch_size=params.max_batch_size,
        tt_ccl=tt_ccl,
    )
    lm_head_config = _build_lm_head(
        mesh_device=mesh_device,
        lm_head_weight=weights.lm_head,
        mcfg=params,
        lm_head_dtype=precision.lm_head_dtype,
        cache_path=cache_path,
    )
    sampling_config = Sampling1DConfig(
        vocab_size=params.vocab_size,
        valid_vocab_size=params.vocab_size,
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        max_batch_size=_nearest_32(params.max_batch_size),
        allow_force_argmax=False,
        pad_to_power_of_2=True,
    )
    return Llama33_70BTransformer1DConfig(
        n_layers=n_layers,
        vocab_size=params.vocab_size,
        max_batch_size=params.max_batch_size,
        max_seq_len=params.max_seq_len,
        dim=params.dim,
        num_devices=num_devices,
        mesh_device=mesh_device,
        embedding_config=embedding_config,
        rope_config=rope_config,
        block_configs=block_configs,
        norm_config=norm_config,
        lm_head_config=lm_head_config,
        sampling_config=sampling_config,
        decode_residual_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        prefill_residual_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        activation_dtypes=[None] * n_layers,
        tt_ccl=tt_ccl,
        cache_path=str(cache_path),
    )


class Llama33_70BTransformer1D(LightweightModule):
    """Provider-neutral TTTv2 Llama 3.3 70B tensor model."""

    decode_residual_memcfg = ttnn.DRAM_MEMORY_CONFIG

    def __init__(self, config: Llama33_70BTransformer1DConfig):
        from tqdm import tqdm

        super().__init__()
        self.config = config
        self.embedding = Embedding1D.from_config(config.embedding_config)
        self.rope_setup = RotarySetup1D.from_config(config.rope_config)
        self.layers = [
            TransformerBlock1D.from_config(config.block_configs[index])
            for index in tqdm(range(config.n_layers), desc="Building layers")
        ]
        self.norm = RMSNorm1D.from_config(config.norm_config)
        self.lm_head = LMHead1D.from_config(config.lm_head_config)
        self.sampling = Sampling1D.from_config(config.sampling_config) if config.sampling_config is not None else None
        self.supports_on_device_sampling = self.sampling is not None
        self.mesh_device = config.mesh_device
        self.tt_ccl = config.tt_ccl
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.num_devices = config.num_devices
        self.decode_residual_memcfg = config.decode_residual_memcfg or ttnn.DRAM_MEMORY_CONFIG
        self.prefill_residual_memcfg = config.prefill_residual_memcfg or ttnn.DRAM_MEMORY_CONFIG
        self.activation_dtypes = config.activation_dtypes or [None] * config.n_layers
        self.model_args = None

    # =========================================================================
    # KV cache binding
    # =========================================================================

    def iter_executor_named_modules(self):
        layers = getattr(self, "layers", ())
        for i, layer in enumerate(layers):
            for suffix, submodule in (
                ("attn_norm", layer.attention_norm),
                ("attention", layer.attention),
                ("ff_norm", layer.ff_norm),
                ("mlp", layer.feed_forward),
            ):
                yield f"layer[{i}].{suffix}", submodule
        if hasattr(self, "norm"):
            yield "final_norm", self.norm
        if hasattr(self, "lm_head"):
            yield "lm_head", self.lm_head

    def configure_paged_attention(self, *, block_size: int, max_num_blocks: int) -> None:
        for name, value in (("block_size", block_size), ("max_num_blocks", max_num_blocks)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        live_configs = tuple(layer.attention.config for layer in self.layers)
        for layer_index, config in enumerate(live_configs):
            if not config.use_vllm_paged_kv_cache or config.paged_attention_config is None:
                raise RuntimeError(f"Model layer {layer_index} is not configured for externally managed paged KV cache")
            if config.kv_cache is not None or getattr(self.layers[layer_index].attention, "kv_cache", None) is not None:
                raise RuntimeError(f"Model layer {layer_index} already has a bound KV cache")
        construction_configs = tuple(block.attention_config for block in self.config.block_configs)
        for config in tuple({id(c): c for c in (*construction_configs, *live_configs)}.values()):
            config.paged_attention_config = replace(
                config.paged_attention_config, block_size=block_size, max_num_blocks=max_num_blocks
            )

    def set_kv_cache(self, kv_cache: list | None) -> None:
        if kv_cache is None:
            for layer in self.layers:
                layer.attention.config.kv_cache = None
                if hasattr(layer.attention, "kv_cache"):
                    layer.attention.kv_cache = None
            return
        if len(kv_cache) != len(self.layers):
            raise ValueError(f"kv_cache has {len(kv_cache)} entries but model has {len(self.layers)} layers")
        cache_pairs = []
        for i, value in enumerate(kv_cache):
            try:
                pair = tuple(value)
            except TypeError as error:
                raise TypeError(f"kv_cache layer {i} must provide an iterable K/V tensor pair") from error
            if len(pair) != 2:
                raise ValueError(f"kv_cache layer {i} must contain exactly two K/V tensors")
            cache_pairs.append(pair)
        for layer, pair in zip(self.layers, cache_pairs):
            layer.attention.config.kv_cache = pair
            if hasattr(layer.attention, "kv_cache"):
                layer.attention.kv_cache = pair

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
        chunk_start_idx_tensor: ttnn.Tensor | None = None,
        last_token_slice: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        last_token_index: ttnn.Tensor | None = None,
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
            x = layer.prefill_forward(
                x,
                rot_mats,
                user_id,
                page_table,
                chunk_page_table,
                chunk_start_idx,
                batch_size,
                chunk_start_idx_tensor,
            )

        if last_token_index is not None and last_token_slice is None:
            raise ValueError("last_token_index is required with a runtime last_token_slice")
        if get_last_token == -1 and last_token_slice is None:
            return x

        old = x
        if last_token_slice is None:
            get_last_token_floor = (get_last_token // 32) * 32
            x = ttnn.slice(x, (0, 0, get_last_token_floor, 0), (1, 1, get_last_token_floor + 32, x.shape[-1]))
        else:
            x = ttnn.slice(
                x,
                last_token_slice[0],
                last_token_slice[1],
                slice_dim=2,
                num_devices=int(x.shape[2]) // 32,
            )
        ttnn.deallocate(old)

        if last_token_index is not None:
            if x.dtype != ttnn.bfloat16:
                old = x
                x = ttnn.typecast(x, ttnn.bfloat16)
                ttnn.deallocate(old)
            old = x
            x = ttnn.embedding(last_token_index, x, layout=ttnn.TILE_LAYOUT)
            x = ttnn.unsqueeze_to_4D(x)
            ttnn.deallocate(old)

        x = self.norm.prefill_forward(x)
        x = _all_gather_rmsnorm_tensor(self.norm, x)
        lm_head_memcfg = self.lm_head.config.input_memcfg
        if lm_head_memcfg is not None and lm_head_memcfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_memcfg)
        x = self.lm_head.forward(x)
        return ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

    def post_process_prefill_output(
        self,
        hidden_states: ttnn.Tensor,
        last_token_idx: int,
        last_token_slice: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        last_token_index: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        if last_token_slice is None:
            get_last_token_floor = (last_token_idx // 32) * 32
            x = ttnn.slice(
                hidden_states,
                (0, 0, get_last_token_floor, 0),
                (1, 1, get_last_token_floor + 32, hidden_states.shape[-1]),
            )
        else:
            x = ttnn.slice(
                hidden_states,
                last_token_slice[0],
                last_token_slice[1],
                slice_dim=2,
                num_devices=int(hidden_states.shape[2]) // 32,
            )
        if last_token_index is not None:
            if x.dtype != ttnn.bfloat16:
                old = x
                x = ttnn.typecast(x, ttnn.bfloat16)
                ttnn.deallocate(old)
            old = x
            x = ttnn.embedding(last_token_index, x, layout=ttnn.TILE_LAYOUT)
            x = ttnn.unsqueeze_to_4D(x)
            ttnn.deallocate(old)
        x = self.norm.prefill_forward(x)
        x = _all_gather_rmsnorm_tensor(self.norm, x)
        lm_head_memcfg = self.lm_head.config.input_memcfg
        if lm_head_memcfg is not None and lm_head_memcfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_memcfg)
        x = self.lm_head.forward(x)
        return ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

    def post_process_batched_prefill_output(
        self,
        hidden_states: ttnn.Tensor,
        last_token_idx_list: list[int],
        padded_batch: int,
        prefill_seq_len: int,
        last_token_slice: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        last_token_index: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        del last_token_slice, last_token_index
        fold_len = padded_batch * prefill_seq_len
        selector = torch.zeros(1, 1, 32, fold_len, dtype=torch.bfloat16)
        for local_row, last_token_idx in enumerate(last_token_idx_list):
            selector[0, 0, local_row, local_row * prefill_seq_len + last_token_idx] = 1.0
        selector = ttnn.from_torch(
            selector,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        x = ttnn.matmul(
            selector,
            hidden_states,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                self.mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        ttnn.deallocate(selector)
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

    def prepare_prefill_rot_mats(self, position_indices: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        self.rope_setup.load_device_weights()
        cos = None
        sin = None
        try:
            cos = ttnn.embedding(position_indices, self.rope_setup.cos_matrix, layout=ttnn.TILE_LAYOUT)
            sin = ttnn.embedding(position_indices, self.rope_setup.sin_matrix, layout=ttnn.TILE_LAYOUT)
            return ttnn.unsqueeze_to_4D(cos), ttnn.unsqueeze_to_4D(sin)
        except BaseException:
            for tensor in (sin, cos):
                if tensor is not None:
                    try:
                        ttnn.deallocate(tensor)
                    except BaseException:
                        pass
            raise

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


# =============================================================================
# Public exports
# =============================================================================

__all__ = [
    "Llama33_70BPrecisionConfig",
    "LLAMA33_70B_ACCURACY",
    "LLAMA33_70B_PERFORMANCE",
    "Llama33_70BPagedAttentionConfig",
    "Llama33_70BLayerWeights",
    "Llama33_70BModelParameters",
    "Llama33_70BWeights",
    "Llama33_70BTransformer1DConfig",
    "Llama33_70BTransformer1D",
    "build_llama33_70b_transformer_1d_config",
    "TransformerBlock1D",
    "TransformerBlock1DConfig",
    "_all_gather_rmsnorm_tensor",
]
