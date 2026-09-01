# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen2.5-7B-Instruct — native TTTv2 stack (``Embedding1D``, ``RMSNorm1D``,
``Attention1D``, ``MLP1D``, ``RotarySetup1D``, ``LMHead1D``).

Tensor layout contracts:
  - **Prefill** hidden states: ``[1, 1, S, dim]`` TILE, ``S % 128 == 0``.
  - **Decode** hidden states: ``[1, 1, B, dim]`` TILE (``B`` padded to tile in modules).

The tensor model is provider-neutral. Hugging Face loading and conversion live
in :mod:`models.common.models.qwen25_7b.hf_adaptor`.
"""

from __future__ import annotations

import math
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
from models.common.modules.rope.rope_1d import Rope1DConfig, RotarySetup1D, prepare_rot_idxs
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig
from models.common.modules.tt_ccl import default_topology, get_tt_ccl
from models.common.tensor_utils import TILE_SIZE, get_padded_hidden_dim


def _lazy(
    tensor: torch.Tensor,
    *,
    dtype: ttnn.DataType,
    cache: tuple[Path, str] | None,
) -> LazyWeight:
    """Minimal LazyWeight; ``Attention1D`` / ``MLP1D`` / ``Embedding1D`` resolvers fill mesh + memory."""
    return LazyWeight(source=tensor, dtype=dtype, cache_dir_weight_name=cache)


@dataclass
class Qwen25PagedAttentionConfig:
    """Paged KV layout for ``Attention1D`` (``block_size`` / ``max_num_blocks`` only)."""

    block_size: int
    max_num_blocks: int


@dataclass
class Qwen25_7BTransformerConfig:
    """Complete provider-neutral tensor-graph configuration."""

    n_layers: int
    vocab_size: int
    max_batch_size: int
    max_seq_len: int
    dim: int
    num_devices: int
    mesh_device: ttnn.MeshDevice
    embedding_config: Embedding1DConfig
    rope_config: Rope1DConfig
    block_configs: list[Qwen25_7BDecoderLayerConfig]
    norm_config: RMSNorm1DConfig
    lm_head_config: LMHead1DConfig
    sampling_config: Sampling1DConfig | None = None
    decode_residual_memcfg: ttnn.MemoryConfig | None = None
    prefill_residual_memcfg: ttnn.MemoryConfig | None = None
    activation_dtypes: list[ttnn.DataType | None] = field(default_factory=list)
    tt_ccl: Any = None
    cache_path: str | None = None


@dataclass(frozen=True)
class Qwen25_7BModelParameters:
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
class Qwen25_7BLayerWeights:
    wqkv: torch.Tensor
    wo: torch.Tensor
    wqkv_bias: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w3: torch.Tensor
    attention_norm: torch.Tensor
    ff_norm: torch.Tensor


@dataclass(frozen=True)
class Qwen25_7BWeights:
    embedding: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    layers: tuple[Qwen25_7BLayerWeights, ...]
    final_norm: torch.Tensor
    lm_head: torch.Tensor


@dataclass(frozen=True)
class Qwen25_7BPrecisionConfig:
    """Per-layer precision + math-fidelity recipe for Qwen2.5-7B-Instruct.

    Mirrors the fields TTTv1's ``DecodersPrecision`` actually distinguishes for
    Qwen2.5-7B (Llama-family group in ``model_config.py`` for both ``accuracy()``
    and ``performance()``). Two module-level recipes are exposed:
    :data:`QWEN25_7B_ACCURACY` and :data:`QWEN25_7B_PERFORMANCE`. Pass one to
    the product adaptor via ``optimizations=``; use
    ``dataclasses.replace(QWEN25_7B_PERFORMANCE, perf_decode_tuning=False)`` to
    customize a single field (e.g. disable aggressive decode math during
    teacher-forcing parity runs).

    Attention compute-kernel configs (HIFI4 + fp32 dest acc for LI_QKV / LI_O /
    SDPA prefill, plus the decode variants and the LoFi perf-decode kernel) are
    resolved inside ``_resolve_qwen_wh_tuning`` from
    :attr:`perf_decode_tuning`; they are not exposed here because TTTv1 ships
    them as a coupled bundle for this model.
    """

    # Qwen2.5-7B keeps BF16 WQKV/WO in BOTH TTTv1 recipes — TTTv2's
    # ``Attention1D`` default of BFP8 silently downgrades QKV/WO precision and
    # broadens per-layer divergence on this model (see
    # debugging/numerical_divergence_vs_hf_2026-05-14.md).
    wqkv_dtype: ttnn.DataType = ttnn.bfloat16
    wo_dtype: ttnn.DataType = ttnn.bfloat16
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat8_b

    # MLP weights: BFP8 in both modes for Qwen2.5-7B (TTTv1 does not drop these
    # to BFP4 for this checkpoint).
    mlp_w1_w3_dtype: ttnn.DataType = ttnn.bfloat8_b
    mlp_w2_dtype: ttnn.DataType = ttnn.bfloat8_b

    lm_head_dtype: ttnn.DataType = ttnn.bfloat8_b

    # Decode-side throughput tuning: LoFi attention decode (LI_QKV / SDPA /
    # LI_O), HiFi2 MLP decode FF, SDPA exp-approx, no W1→DRAM spill for small
    # batch. Performance recipe only — keep ``False`` in accuracy mode and in
    # any teacher-forcing path (aggressive decode math drops top-1).
    perf_decode_tuning: bool = False


# TTTv1 ``DecodersPrecision.accuracy("Qwen2.5-7B-Instruct")`` (Llama-family
# group in ``model_config.py``): BF16 attention, BFP8 MLP, BF16 LM head, BF16
# KV cache, HiFi4 throughout. ``perf_decode_tuning`` stays off — decode keeps
# HiFi4 + fp32 dest acc.
QWEN25_7B_ACCURACY = Qwen25_7BPrecisionConfig(
    kv_cache_dtype=ttnn.bfloat16,
    lm_head_dtype=ttnn.bfloat16,
)

# TTTv1 ``DecodersPrecision.performance("Qwen2.5-7B-Instruct")``: same dtypes
# as accuracy except ``kv_cache_dtype`` / ``lm_head_dtype`` drop to BFP8, plus
# the decode-side throughput bundle (LoFi attn decode + SDPA exp-approx + HiFi2
# MLP decode FF) is engaged via ``perf_decode_tuning=True``.
QWEN25_7B_PERFORMANCE = Qwen25_7BPrecisionConfig(
    perf_decode_tuning=True,
)


def _qwen_wh_mlp_matmul_compute_kernel() -> ttnn.WormholeComputeKernelConfig:
    """HiFi4 lowers L1 circular-buffer footprint vs HiFi2 for wide FF matmuls on Wormhole."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _qwen_wh_mlp_decode_matmul_compute_kernel() -> ttnn.WormholeComputeKernelConfig:
    """HiFi2 decode FF matmuls: faster steady-state decode while prefill keeps HiFi4 (``_qwen_wh_mlp_matmul_compute_kernel``)."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _qwen_wh_decode_attn_lofi_kernel() -> ttnn.WormholeComputeKernelConfig:
    """LoFi decode attention matmuls + SDPA when ``perf_decode_tuning`` is enabled (performance demo only)."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _qwen_wh_attn_hifi4_kernel() -> ttnn.WormholeComputeKernelConfig:
    """HiFi4 + fp32 dest acc for Qwen2.5-7B attention matmuls (LI_QKV, LI_O, SDPA).

    Matches the TTTv1 ``ModelArgs.compute_kernel_config_hifi4`` used by both
    ``ModelOptimizations.performance`` and ``ModelOptimizations.accuracy`` for
    Qwen2.5-7B (see ``models/tt_transformers/tt/model_config.py``).
    The TTTv2 ``Attention1D`` defaults are HiFi2 with fp16 accumulation; that
    silently downgrades attention prefill QKV/WO and decode QKV/SDPA/WO matmul
    precision for this model, producing a broad per-layer divergence vs HF.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _slice_last_token_tile(x: ttnn.Tensor, last_token_idx: int) -> ttnn.Tensor:
    """Slice the 32-row tile containing ``last_token_idx`` from ``[1, 1, S, W]``.

    Width-sharded LM matmul M tile rows must equal ``LMHead1D`` program-config tile rows.
    """
    floor = (last_token_idx // 32) * 32
    return ttnn.slice(x, (0, 0, floor, 0), (1, 1, floor + 32, x.shape[-1]))


def _post_attn_norm_decode_configs(
    *,
    dim: int,
    hidden_dim: int,
    num_devices: int,
    max_batch_size: int,
) -> tuple[Any, ttnn.MemoryConfig]:
    """Resolve post-attention RMSNorm decode sharding so its output matches MLP1D's W1/W3 input.

    MLP1D decode uses ``_dram_shard_core_grid_k_n(dim, padded_hidden / num_devices)`` for W1/W3
    inputs, but the default RMSNorm program config is derived from ``_compute_norm_core_grid(dim)``
    alone. On N300 Qwen2.5-7B that DRAM-width-shard mismatch silently corrupts decode activations.
    """
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


def _all_gather_rmsnorm_tensor(
    norm: RMSNorm1D, x: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig | None = None
) -> ttnn.Tensor:
    cfg = norm.config
    if cfg.mesh_device.get_num_devices() == 1 or x.shape[-1] == cfg.weight.source.numel():
        return x

    if memory_config is None:
        memory_config = x.memory_config()

    tt_ccl = cfg.tt_ccl or get_tt_ccl(cfg.mesh_device)
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


@dataclass
class _Qwen25WHTuning:
    """Wormhole-specific MLP / LM-head / attention tuning resolved at build time.

    Populated for Qwen2.5-7B on N150 / N300 only; otherwise all fields stay ``None``
    (modules fall back to library defaults).
    """

    mlp_prefill_len_cutoff: int | None = None
    mlp_ff_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    mlp_decode_spill_w1_to_dram: bool = False
    mlp_decode_ff_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    lm_head_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    perf_decode_sdpa_cfg: ttnn.SDPAProgramConfig | None = None
    # Attention prefill compute kernels: always HiFi4 + fp32 dest acc on Qwen
    # (Attention1D defaults are HiFi2/fp16 acc — too lossy for this model).
    attn_li_qkv_prefill_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    attn_li_o_prefill_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    # Attention decode compute kernels: HiFi4 + fp32 dest acc by default; LoFi
    # under ``perf_decode_tuning`` (performance demo only).
    attn_li_qkv_decode_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    attn_sdpa_decode_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    attn_li_o_decode_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None


def _resolve_qwen_wh_tuning(
    *, hf_model_id: str, num_dev: int, max_batch_size: int, perf_decode_tuning: bool
) -> _Qwen25WHTuning:
    """Pick WH tuning knobs for Qwen2.5-7B-Instruct on N150 / N300.

    ``get_padded_prefill_len`` maps 129..1024 tokens to a 1024-wide tile. ``MLP1D`` then
    reshapes/chunks using ``prefill_len_cutoff``.

    Qwen2.5-7B preserves its validated 256-token MLP prefill cutoff. This halves the
    per-kernel M tile for the model's wide per-device FF shard and keeps its established
    circular-buffer footprint.

    Output is unchanged by the cutoff: ``in0_block_w`` and the K-contraction order are independent of the
    M-tiling, so only ``per_core_M`` changes. Decode never reads ``prefill_len_cutoff``.

    Decode W1→DRAM before W3 avoids L1 overlap between W1 activations and W3 matmul CBs
    (N300 batch 32).
    """
    t = _Qwen25WHTuning()
    model_slug = hf_model_id.split("/")[-1]
    if not (model_slug.startswith("Qwen2.5-7B") and num_dev in (1, 2)):
        return t

    t.mlp_prefill_len_cutoff = 256
    t.mlp_ff_compute_kernel_cfg = _qwen_wh_mlp_matmul_compute_kernel()
    t.mlp_decode_spill_w1_to_dram = max_batch_size >= 16
    t.lm_head_compute_kernel_cfg = _qwen_wh_mlp_matmul_compute_kernel()
    t.mlp_decode_ff_compute_kernel_cfg = (
        _qwen_wh_mlp_decode_matmul_compute_kernel() if perf_decode_tuning else t.mlp_ff_compute_kernel_cfg
    )

    # Attention prefill always uses HiFi4 + fp32 dest acc on Qwen (matches TTTv1).
    attn_hifi4 = _qwen_wh_attn_hifi4_kernel()
    t.attn_li_qkv_prefill_kernel_cfg = attn_hifi4
    t.attn_li_o_prefill_kernel_cfg = attn_hifi4
    # Attention decode defaults to HiFi4 + fp32 dest acc; LoFi only under perf_decode_tuning.
    t.attn_li_qkv_decode_kernel_cfg = attn_hifi4
    t.attn_sdpa_decode_kernel_cfg = attn_hifi4
    t.attn_li_o_decode_kernel_cfg = attn_hifi4

    if perf_decode_tuning:
        lo = _qwen_wh_decode_attn_lofi_kernel()
        t.perf_decode_sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=True,
            q_chunk_size=0,
            k_chunk_size=0,
        )
        t.attn_li_qkv_decode_kernel_cfg = lo
        t.attn_sdpa_decode_kernel_cfg = lo
        t.attn_li_o_decode_kernel_cfg = lo
    logger.info(
        f"MLP/LM/attention tuning for {hf_model_id} on {num_dev} device(s): "
        f"prefill_len_cutoff={t.mlp_prefill_len_cutoff}, FF prefill HiFi4, attn prefill+decode HiFi4+fp32, "
        f"decode spill W1→DRAM={t.mlp_decode_spill_w1_to_dram}, "
        f"perf_decode_tuning={perf_decode_tuning}"
    )
    return t


@dataclass
class Qwen25_7BDecoderLayerConfig:
    attention_norm_config: RMSNorm1DConfig
    attention_config: Attention1DConfig
    ff_norm_config: RMSNorm1DConfig
    mlp_config: MLP1DConfig


def _build_decoder_layer(
    *,
    idx: int,
    weights: Qwen25_7BLayerWeights,
    qcfg: Qwen25_7BModelParameters,
    mesh_device: ttnn.MeshDevice,
    tt_ccl: Any,
    topology: Any,
    num_dev: int,
    precision: Qwen25_7BPrecisionConfig,
    paged_cfg: Qwen25PagedAttentionConfig,
    cache_path: Path | None,
    wh: _Qwen25WHTuning,
) -> Qwen25_7BDecoderLayerConfig:
    """Build one decoder-layer config from provider-converted tensors."""
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

    bias_lw = LazyWeight(
        source=weights.wqkv_bias.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        cache_dir_weight_name=(cache_path / "attn", f"{prefix}_bias") if cache_path else None,
    )

    attention_config = Attention1DConfig(
        wqkv=lazy_wqkv,
        wo=lazy_wo,
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        topology=topology,
        n_heads=qcfg.n_heads,
        n_kv_heads=qcfg.n_kv_heads,
        head_dim=qcfg.head_dim,
        max_batch_size=qcfg.max_batch_size,
        max_seq_len=qcfg.max_seq_len,
        q_norm_config=None,
        k_norm_config=None,
        wqkv_bias=bias_lw,
        use_vllm_paged_kv_cache=True,
        paged_attention_config=paged_cfg,
        kv_cache=None,
        kv_cache_dtype=precision.kv_cache_dtype,
        decode_sdpa_prg_config=wh.perf_decode_sdpa_cfg,
        li_qkv_decode_compute_kernel_cfg=wh.attn_li_qkv_decode_kernel_cfg,
        sdpa_decode_compute_kernel_cfg=wh.attn_sdpa_decode_kernel_cfg,
        li_o_decode_compute_kernel_cfg=wh.attn_li_o_decode_kernel_cfg,
        li_qkv_prefill_compute_kernel_cfg=wh.attn_li_qkv_prefill_kernel_cfg,
        li_o_prefill_compute_kernel_cfg=wh.attn_li_o_prefill_kernel_cfg,
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
        max_batch_size=qcfg.max_batch_size,
        prefill_len_cutoff=wh.mlp_prefill_len_cutoff,
        ff1_3_compute_kernel_cfg=wh.mlp_ff_compute_kernel_cfg,
        ff2_compute_kernel_cfg=wh.mlp_ff_compute_kernel_cfg,
        decode_ff1_3_compute_kernel_cfg=wh.mlp_decode_ff_compute_kernel_cfg,
        decode_ff2_compute_kernel_cfg=wh.mlp_decode_ff_compute_kernel_cfg,
        decode_spill_w1_to_dram_before_w3=wh.mlp_decode_spill_w1_to_dram,
    )

    post_attn_decode_program_config, post_attn_decode_memory_config = _post_attn_norm_decode_configs(
        dim=qcfg.dim,
        hidden_dim=qcfg.hidden_dim,
        num_devices=num_dev,
        max_batch_size=qcfg.max_batch_size,
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
            eps=qcfg.rms_norm_eps,
            max_batch_size=qcfg.max_batch_size,
            tt_ccl=tt_ccl,
            **extra,
        )

    return Qwen25_7BDecoderLayerConfig(
        attention_norm_config=_build_norm(weights.attention_norm, "pre_attn"),
        attention_config=attention_config,
        ff_norm_config=_build_norm(
            weights.ff_norm,
            "post_attn",
            decode_program_config=post_attn_decode_program_config,
            decode_memory_config=post_attn_decode_memory_config,
        ),
        mlp_config=mlp_config,
    )


def _build_lm_head(
    *,
    mesh_device: ttnn.MeshDevice,
    lm_head_weight: torch.Tensor,
    qcfg: Qwen25_7BModelParameters,
    lm_head_dtype: ttnn.DataType,
    lm_head_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None,
    cache_path: Path | None,
) -> LMHead1DConfig:
    """Build the vocab-sharded LM head with DRAM-matmul program configs.

    LM head DRAM matmul is sized for decode batch tiles (``max_batch_size``). Prefill logits
    use a single 32-row tile via ``post_process_prefill_output`` / :func:`_slice_last_token_tile`.
    """
    lm_splits, lm_split_sizes, lm_weights_memcfgs = _build_lm_head_lazy_weights(
        mesh_device,
        lm_head_weight,
        dim=qcfg.dim,
        vocab_size=qcfg.vocab_size,
        dtype=lm_head_dtype,
        cache_dir=cache_path / "lm_head" if cache_path else None,
    )
    lm_head_core_grid = _dram_shard_core_grid(qcfg.dim)
    tile = ttnn.TILE_SIZE
    tile_padded_batch_rows = tile * math.ceil(qcfg.max_batch_size / tile)
    lm_prog_configs = [
        _dram_matmul_config(tile_padded_batch_rows, qcfg.dim, ss, lm_head_core_grid.num_cores) for ss in lm_split_sizes
    ]
    lm_input_memcfg = ttnn.create_sharded_memory_config(
        (tile_padded_batch_rows, _nearest_32(qcfg.dim // lm_head_core_grid.num_cores)),
        lm_head_core_grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return LMHead1DConfig(
        output_weights=lm_splits,
        mesh_device=mesh_device,
        dim=qcfg.dim,
        max_batch_size=qcfg.max_batch_size,
        lm_head_dtype=lm_head_dtype,
        program_configs=lm_prog_configs,
        compute_kernel_config=lm_head_compute_kernel_cfg,
        input_memcfg=lm_input_memcfg,
        weights_memcfgs=lm_weights_memcfgs,
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
    """Column-split a provider-neutral LM-head tensor for ``LMHead1D``."""
    num_devices = mesh_device.get_num_devices()
    torch_w = lm_head_weight.T.contiguous().to(torch.bfloat16)
    padded_vocab_size = math.ceil(vocab_size / TILE_SIZE) * TILE_SIZE
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

    def dram_sharded_memcfg(k: int, n: int) -> ttnn.MemoryConfig:
        padded_n = math.ceil(n / (TILE_SIZE * dram_size.x)) * (TILE_SIZE * dram_size.x)
        shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_n // dram_size.x), ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    output_weights = []
    weights_memcfgs = []
    for index, split_size in enumerate(split_sizes):
        device_splits = []
        for device_index in range(num_devices):
            start = device_index * size_per_device + sum(split_sizes[:index])
            device_splits.append(torch_w[:, start : start + split_size])
        combined = torch.cat(device_splits, dim=-1)
        mem_cfg = dram_sharded_memcfg(dim, math.ceil(combined.shape[-1] / num_devices))
        weights_memcfgs.append(mem_cfg)
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
                memory_config=mem_cfg,
                cache_dir_weight_name=(cache_dir, f"lm_head_split_{index}_{combined.shape[-1]}") if cache_dir else None,
            )
        )
    return output_weights, split_sizes, weights_memcfgs


def build_qwen25_7b_transformer_config(
    *,
    mesh_device: ttnn.MeshDevice,
    params: Qwen25_7BModelParameters,
    weights: Qwen25_7BWeights,
    n_layers: int,
    precision: Qwen25_7BPrecisionConfig,
    cache_path: Path,
    paged_attention_config: Qwen25PagedAttentionConfig,
    model_name: str = "Qwen2.5-7B-Instruct",
) -> Qwen25_7BTransformerConfig:
    """Build the TT graph from provider-neutral dimensions and converted tensors."""
    num_devices = mesh_device.get_num_devices()
    if num_devices != 2:
        raise ValueError(f"Qwen2.5-7B supports logical TP2 lanes only, got {num_devices} devices")
    if params.n_heads % num_devices or params.n_kv_heads % num_devices:
        raise ValueError(
            f"Checkpoint heads ({params.n_heads}/{params.n_kv_heads}) "
            f"must be divisible by device count ({num_devices})"
        )
    if len(weights.layers) != n_layers:
        raise ValueError(f"Expected {n_layers} decoder layer weight sets, got {len(weights.layers)}")

    tt_ccl = get_tt_ccl(mesh_device)
    topology = default_topology(mesh_device)
    wh = _resolve_qwen_wh_tuning(
        hf_model_id=model_name,
        num_dev=num_devices,
        max_batch_size=params.max_batch_size,
        perf_decode_tuning=precision.perf_decode_tuning,
    )
    embedding_config = Embedding1DConfig(
        weights=_lazy(weights.embedding, dtype=ttnn.bfloat16, cache=(cache_path / "embedding", "tok_embeddings")),
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
    block_configs = [
        _build_decoder_layer(
            idx=index,
            weights=weights.layers[index],
            qcfg=params,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            topology=topology,
            num_dev=num_devices,
            precision=precision,
            paged_cfg=paged_attention_config,
            cache_path=cache_path,
            wh=wh,
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
        qcfg=params,
        lm_head_dtype=precision.lm_head_dtype,
        lm_head_compute_kernel_cfg=wh.lm_head_compute_kernel_cfg,
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
    return Qwen25_7BTransformerConfig(
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


class Qwen25_7BDecoderLayer(LightweightModule):
    def __init__(
        self,
        *,
        input_layernorm: RMSNorm1D,
        self_attn: Attention1D,
        post_attention_layernorm: RMSNorm1D,
        mlp: MLP1D,
    ):
        super().__init__()
        self.input_layernorm = input_layernorm
        self.self_attn = self_attn
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp
        self.attention_norm = input_layernorm
        self.attention = self_attn
        self.ff_norm = post_attention_layernorm
        self.feed_forward = mlp

    @classmethod
    def from_config(cls, config: Qwen25_7BDecoderLayerConfig) -> Qwen25_7BDecoderLayer:
        return cls(
            input_layernorm=RMSNorm1D.from_config(config.attention_norm_config),
            self_attn=Attention1D.from_config(config.attention_config),
            post_attention_layernorm=RMSNorm1D.from_config(config.ff_norm_config),
            mlp=MLP1D.from_config(config.mlp_config),
        )

    def prefill_forward(
        self,
        x: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        *,
        user_id: int = 0,
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        batch_size: int = 1,
        chunk_start_idx_tensor: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        # For batched prefill (batch_size > 1) x is the folded [1,1,B*S,dim] hidden state; norm,
        # residual add and MLP are row-independent so they treat B*S as one long sequence unchanged.
        # Only attention unfolds the batch axis internally (see Attention1D.prefill_forward).
        # Match Llama ``TransformerBlock1D``: fractured embed / norm activations must be
        # all-gathered to full ``dim`` before Attention1D / MLP1D (QKV matmul expects width ``dim``).
        r = self.input_layernorm.prefill_forward(x)
        r = _all_gather_rmsnorm_tensor(self.input_layernorm, r)
        r = self.self_attn.prefill_forward(
            r,
            rot_mats,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            batch_size=batch_size,
            chunk_start_idx_tensor=chunk_start_idx_tensor,
        )
        h = ttnn.add(x, r, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        r2 = self.post_attention_layernorm.prefill_forward(h)
        r2 = _all_gather_rmsnorm_tensor(self.post_attention_layernorm, r2)
        r2 = self.mlp.prefill_forward(r2)
        return ttnn.add(h, r2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def decode_forward(
        self,
        x: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        xa = _all_gather_rmsnorm_tensor(
            self.input_layernorm, x, memory_config=self.input_layernorm.config.decode_memory_config
        )
        r = self.input_layernorm.forward(xa, "decode")
        r = self.self_attn.forward(r, current_pos, rot_mats, mode="decode", page_table=page_table)
        h = ttnn.add(x, r, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hf = _all_gather_rmsnorm_tensor(
            self.post_attention_layernorm, h, memory_config=self.post_attention_layernorm.config.decode_memory_config
        )
        r2 = self.post_attention_layernorm.forward(hf, "decode")
        r2 = self.mlp.forward(r2, "decode")
        return ttnn.add(h, r2, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class Qwen25_7B(LightweightModule):
    """
    Full decoder for Qwen2.5-7B-Instruct (TTTv2 modules only).

    Prefill/decode on **embedded** activations match ``EagerLLMExecutor``. Token embedding
    is ``embed_prefill`` / ``embed_decode``. Bind KV with ``set_kv_cache`` before first forward.
    """

    decode_residual_memcfg = ttnn.DRAM_MEMORY_CONFIG

    def __init__(self, config: Qwen25_7BTransformerConfig):
        from tqdm import tqdm

        super().__init__()
        self.config = config
        self.embed = Embedding1D.from_config(config.embedding_config)
        self.embedding = self.embed
        self.rope_setup = RotarySetup1D.from_config(config.rope_config)
        self.layers = [
            Qwen25_7BDecoderLayer.from_config(config.block_configs[index])
            for index in tqdm(range(config.n_layers), desc="Building layers")
        ]
        self.norm = RMSNorm1D.from_config(config.norm_config)
        self.lm_head = LMHead1D.from_config(config.lm_head_config)
        self.sampling = Sampling1D.from_config(config.sampling_config) if config.sampling_config is not None else None
        self.supports_on_device_sampling = self.sampling is not None
        self.mesh_device = config.mesh_device
        self.tt_ccl = config.tt_ccl or get_tt_ccl(config.mesh_device)
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.num_devices = config.num_devices
        self.decode_residual_memcfg = config.decode_residual_memcfg or ttnn.DRAM_MEMORY_CONFIG
        self.prefill_residual_memcfg = config.prefill_residual_memcfg or ttnn.DRAM_MEMORY_CONFIG
        self.activation_dtypes = config.activation_dtypes or [None] * config.n_layers
        self.model_args = None

    @property
    def n_kv_heads(self) -> int:
        return self.config.block_configs[0].attention_config.n_kv_heads

    def iter_executor_named_modules(self):
        if not hasattr(self, "layers"):
            return
        for index, layer in enumerate(self.layers):
            for suffix, submodule in (
                ("attn_norm", getattr(layer, "attention_norm", None)),
                ("attention", getattr(layer, "attention", None)),
                ("ff_norm", getattr(layer, "ff_norm", None)),
                ("mlp", getattr(layer, "feed_forward", None)),
            ):
                if submodule is not None:
                    yield f"layer[{index}].{suffix}", submodule
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
        for config in tuple({id(item): item for item in (*construction_configs, *live_configs)}.values()):
            config.paged_attention_config = replace(
                config.paged_attention_config,
                block_size=block_size,
                max_num_blocks=max_num_blocks,
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
        for index, value in enumerate(kv_cache):
            try:
                pair = tuple(value)
            except TypeError as error:
                raise TypeError(f"kv_cache layer {index} must provide an iterable K/V tensor pair") from error
            if len(pair) != 2:
                raise ValueError(f"kv_cache layer {index} must contain exactly two K/V tensors")
            cache_pairs.append(pair)
        for layer, pair in zip(self.layers, cache_pairs):
            layer.attention.config.kv_cache = pair
            if hasattr(layer.attention, "kv_cache"):
                layer.attention.kv_cache = pair

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
        x = self.embed.forward(tokens)
        x = ttnn.unsqueeze_to_4D(x)
        return ttnn.to_memory_config(x, self.decode_residual_memcfg)

    def embed_prefill(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        x = self.embed.forward(tokens)
        return ttnn.unsqueeze_to_4D(x)

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
        # batch_size > 1: x_embed is the folded [1,1,B*S,dim] tensor (B users). Without a runtime
        # last-token slice, the batched path returns the full hidden state (get_last_token == -1);
        # the executor does per-slot extraction + norm/lm_head so those stages stay bit-identical.
        x = x_embed
        for layer in self.layers:
            x = layer.prefill_forward(
                x,
                rot_mats,
                user_id=user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                batch_size=batch_size,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
            )

        if get_last_token == -1 and last_token_slice is None:
            return x

        # Slice + deallocate the full-sequence buffer before norm/LM head reduces peak L1.
        old = x
        if last_token_slice is None:
            x_tile = _slice_last_token_tile(old, get_last_token)
        else:
            x_tile = ttnn.slice(
                old,
                last_token_slice[0],
                last_token_slice[1],
                slice_dim=2,
                num_devices=int(old.shape[2]) // 32,
            )
        ttnn.deallocate(old)
        if last_token_index is not None:
            if x_tile.dtype != ttnn.bfloat16:
                old_tile = x_tile
                x_tile = ttnn.typecast(x_tile, ttnn.bfloat16)
                ttnn.deallocate(old_tile)
            old_tile = x_tile
            x_tile = ttnn.embedding(last_token_index, x_tile, layout=ttnn.TILE_LAYOUT)
            x_tile = ttnn.unsqueeze_to_4D(x_tile)
            ttnn.deallocate(old_tile)
        return self._last_tile_logits(x_tile)

    def post_process_prefill_output(
        self,
        hidden_states: ttnn.Tensor,
        last_token_idx: int,
        last_token_slice: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        last_token_index: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        if last_token_slice is None:
            x = _slice_last_token_tile(hidden_states, last_token_idx)
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
        return self._last_tile_logits(x)

    def _last_tile_logits(self, x_tile: ttnn.Tensor) -> ttnn.Tensor:
        """Final-norm + all-gather + LM-head on a 32-row tile. ``x_tile`` shape ``[1, 1, 32, dim]``."""
        x = self.norm.prefill_forward(x_tile)
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
        return self._last_tile_logits(x)

    def decode_forward(
        self,
        x_embed: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        x = x_embed
        for layer in self.layers:
            x = layer.decode_forward(x, current_pos, rot_mats, page_table=page_table)
        x = _all_gather_rmsnorm_tensor(self.norm, x, memory_config=self.norm.config.decode_memory_config)
        x = self.norm.decode_forward(x)
        return self.lm_head.forward(x)

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
                chunks_per_sync=24,
                num_workers_per_link=4,
                num_buffers_per_channel=2,
            )
        return ttnn.untilize(logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def increment_positions(self, current_pos: ttnn.Tensor, rot_mat_idxs: ttnn.Tensor) -> None:
        ttnn.plus_one(current_pos, skip_negative_entries=True)
        ttnn.plus_one(rot_mat_idxs)

    def prefill_from_token_ids(self, token_ids_tt: ttnn.Tensor, *, start_pos: int = 0, user_id: int = 0) -> ttnn.Tensor:
        """Legacy path: embed + RoPE + blocks + final norm (no page table). For tests only."""
        x = self.embed_prefill(token_ids_tt)
        seq_len = x.shape[2]
        assert seq_len % 128 == 0, "prefill seq_len must be divisible by 128"
        rot = self.rope_setup.prefill_forward(start_pos, seq_len)
        h = x
        for layer in self.layers:
            h = layer.prefill_forward(h, rot, user_id=user_id, page_table=None)
        h = self.norm.prefill_forward(h)
        return _all_gather_rmsnorm_tensor(self.norm, h)

    def decode_from_token_ids(self, token_ids_tt: ttnn.Tensor, *, current_pos: int) -> ttnn.Tensor:
        """Legacy path: single-token decode without paged ``page_table``."""
        x = self.embed.forward(token_ids_tt)
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
        return self.norm.forward(h, "decode")

    def lm_logits(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        """Project last hidden to logits (vocab-sharded on multi-device).

        Skip the explicit interleaved→shard if the caller already produced a sharded
        input (``decode_from_token_ids`` returns the decode-mode RMSNorm's width-sharded
        output, which already matches ``LMHead1D.config.input_memcfg``).
        """
        x = hidden
        lm_head_memcfg = self.lm_head.config.input_memcfg
        if lm_head_memcfg is not None and lm_head_memcfg.is_sharded() and not x.memory_config().is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_memcfg)
        return self.lm_head.forward(x)


__all__ = [
    "QWEN25_7B_ACCURACY",
    "QWEN25_7B_PERFORMANCE",
    "Qwen25PagedAttentionConfig",
    "Qwen25_7B",
    "Qwen25_7BDecoderLayer",
    "Qwen25_7BDecoderLayerConfig",
    "Qwen25_7BLayerWeights",
    "Qwen25_7BModelParameters",
    "Qwen25_7BPrecisionConfig",
    "Qwen25_7BTransformerConfig",
    "Qwen25_7BWeights",
    "build_qwen25_7b_transformer_config",
]
