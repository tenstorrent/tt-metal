# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen2-7B-Instruct — native TTTv2 stack (``Embedding1D``, ``RMSNorm1D``,
``Attention1D``, ``MLP1D``, ``RotarySetup1D``, ``LMHead1D``).

Tensor layout contracts:
  - **Prefill** hidden states: ``[1, 1, S, dim]`` TILE, ``S % 128 == 0``.
  - **Decode** hidden states: ``[1, 1, B, dim]`` TILE (``B`` padded to tile in modules).

Executor contract (``EagerLLMExecutor`` / ``TracedLLMExecutor``): pre-embedded forwards,
``set_kv_cache``, ``rope_setup``, ``page_table`` through attention, ``model_args`` holds a
:class:`Qwen2ExecutorRuntimeConfig` (not v1 ``ModelArgs``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.models.qwen2_7b import weight_utils
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


def _lazy(
    tensor: torch.Tensor,
    *,
    dtype: ttnn.DataType,
    cache: tuple[Path, str] | None,
) -> LazyWeight:
    """Minimal LazyWeight; ``Attention1D`` / ``MLP1D`` / ``Embedding1D`` resolvers fill mesh + memory."""
    return LazyWeight(source=tensor, dtype=dtype, cache_dir_weight_name=cache)


@dataclass
class Qwen2PagedAttentionConfig:
    """Paged KV layout for ``Attention1D`` (``block_size`` / ``max_num_blocks`` only)."""

    block_size: int
    max_num_blocks: int


@dataclass
class Qwen2ExecutorRuntimeConfig:
    """Engine-facing runtime knobs. Exposed as ``model.model_args`` for shared ``EagerLLMExecutor``."""

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

    def can_enable_trace(self, prefill_seq_len: int, num_cached_tokens: int = 0) -> bool:
        # Mirror TTTv1's prefill-trace gate (model_config.get_trace_prefill_supported_seq_lens):
        # only trace the seq lens TTTv1 lists -- bigger seq lens already have small op2op gaps, so
        # tracing buys nothing. Qwen2-7B has NO model-specific entry in TTTv1, so it falls back to
        # the family default: N150 -> [128], N300 -> [128, 1024]. (T3K is unsupported here anyway:
        # 8 does not divide the 4 KV heads.) Decode trace remains enabled at the engine layer.
        num_devices = int(self.cluster_shape[0]) * int(self.cluster_shape[1])
        allowed = {1: (128,), 2: (128, 1024)}.get(num_devices, (128,))
        return (
            prefill_seq_len in allowed
            and prefill_seq_len <= self.max_prefill_chunk_size
            and prefill_seq_len <= self.max_seq_len
        )


@dataclass
class Qwen2_7BConfig:
    """Resolved hyper-parameters for a loaded HF Qwen2-7B checkpoint."""

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


@dataclass(frozen=True)
class Qwen2_7BPrecisionConfig:
    """Per-layer precision + math-fidelity recipe for Qwen2-7B-Instruct.

    Mirrors the fields TTTv1's ``DecodersPrecision`` actually distinguishes for
    Qwen2-7B (Llama-family group in ``model_config.py`` for both ``accuracy()``
    and ``performance()``). Two module-level recipes are exposed:
    :data:`QWEN2_7B_ACCURACY` and :data:`QWEN2_7B_PERFORMANCE`. Pass one to
    :meth:`Qwen2_7B.from_pretrained` via ``precision=``; use
    ``dataclasses.replace(QWEN2_7B_PERFORMANCE, perf_decode_tuning=False)`` to
    customize a single field (e.g. disable aggressive decode math during
    teacher-forcing parity runs).

    Attention compute-kernel configs (HIFI4 + fp32 dest acc for LI_QKV / LI_O /
    SDPA prefill, plus the decode variants and the LoFi perf-decode kernel) are
    resolved inside ``_resolve_qwen_wh_tuning`` from
    :attr:`perf_decode_tuning`; they are not exposed here because TTTv1 ships
    them as a coupled bundle for this model.
    """

    # Qwen2-7B keeps BF16 WQKV/WO in BOTH TTTv1 recipes — TTTv2's
    # ``Attention1D`` default of BFP8 silently downgrades QKV/WO precision and
    # broadens per-layer divergence on this model (see
    # debugging/numerical_divergence_vs_hf_2026-05-14.md).
    wqkv_dtype: ttnn.DataType = ttnn.bfloat16
    wo_dtype: ttnn.DataType = ttnn.bfloat16
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat8_b

    # MLP weights: BFP8 in both modes for Qwen2-7B (TTTv1 does not drop these
    # to BFP4 for this checkpoint).
    mlp_w1_w3_dtype: ttnn.DataType = ttnn.bfloat8_b
    mlp_w2_dtype: ttnn.DataType = ttnn.bfloat8_b

    lm_head_dtype: ttnn.DataType = ttnn.bfloat8_b

    # Decode-side throughput tuning: LoFi attention decode (LI_QKV / SDPA /
    # LI_O), HiFi2 MLP decode FF, SDPA exp-approx, no W1→DRAM spill for small
    # batch. Performance recipe only — keep ``False`` in accuracy mode and in
    # any teacher-forcing path (aggressive decode math drops top-1).
    perf_decode_tuning: bool = False


# TTTv1 ``DecodersPrecision.accuracy("Qwen2-7B-Instruct")`` (Llama-family
# group in ``model_config.py``): BF16 attention, BFP8 MLP, BF16 LM head, BF16
# KV cache, HiFi4 throughout. ``perf_decode_tuning`` stays off — decode keeps
# HiFi4 + fp32 dest acc.
QWEN2_7B_ACCURACY = Qwen2_7BPrecisionConfig(
    kv_cache_dtype=ttnn.bfloat16,
    lm_head_dtype=ttnn.bfloat16,
)

# TTTv1 ``DecodersPrecision.performance("Qwen2-7B-Instruct")``: same dtypes
# as accuracy except ``kv_cache_dtype`` / ``lm_head_dtype`` drop to BFP8, plus
# the decode-side throughput bundle (LoFi attn decode + SDPA exp-approx + HiFi2
# MLP decode FF) is engaged via ``perf_decode_tuning=True``.
QWEN2_7B_PERFORMANCE = Qwen2_7BPrecisionConfig(
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
    """HiFi4 + fp32 dest acc for Qwen2-7B attention matmuls (LI_QKV, LI_O, SDPA).

    Matches the TTTv1 ``ModelArgs.compute_kernel_config_hifi4`` used by both
    ``ModelOptimizations.performance`` and ``ModelOptimizations.accuracy`` for
    Qwen2-7B (see ``models/tt_transformers/tt/model_config.py``).
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
    mlp: MLP1D,
    *,
    dim: int,
    hidden_dim: int,
    num_devices: int,
    max_batch_size: int,
) -> tuple[Any, ttnn.MemoryConfig]:
    """Resolve post-attention RMSNorm decode sharding so its output matches MLP1D's W1/W3 input.

    MLP1D decode uses ``_dram_shard_core_grid_k_n(dim, padded_hidden / num_devices)`` for W1/W3
    inputs, but the default RMSNorm program config is derived from ``_compute_norm_core_grid(dim)``
    alone. On N300 Qwen2-7B that DRAM-width-shard mismatch silently corrupts decode activations.
    """
    padded_hidden = get_padded_hidden_dim(hidden_dim, num_devices, TILE_SIZE)
    grid = _dram_shard_core_grid_k_n(dim, padded_hidden // num_devices)
    tile_padded_batch_rows = TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE)
    program_config = _create_sharded_norm_program_config(dim, grid, tile_padded_batch_rows, TILE_SIZE)
    return program_config, mlp.config.decode_input_memcfg


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
class _Qwen2WHTuning:
    """Wormhole-specific MLP / LM-head / attention tuning resolved at build time.

    Populated for Qwen2-7B on N150 / N300 only; otherwise all fields stay ``None``
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
) -> _Qwen2WHTuning:
    """Pick WH tuning knobs for Qwen2-7B-Instruct on N150 / N300.

    ``get_padded_prefill_len`` maps 129..1024 tokens to a 1024-wide tile. ``MLP1D``
    then reshapes/chunks using ``prefill_len_cutoff``. Cutoff 512 still trips
    ``validate_circular_buffer_region`` on WH (prefill multicast + LM DRAM matmul vs L1).
    256 halves the per-kernel M tile; HiFi4 shrinks CB vs HiFi2 for FF and LM DRAM linears.
    Decode W1→DRAM before W3 avoids L1 overlap between W1 activations and W3 matmul CBs
    (N300 batch 32).
    """
    t = _Qwen2WHTuning()
    model_slug = hf_model_id.split("/")[-1]
    if not (model_slug.startswith("Qwen2-7B") and num_dev in (1, 2)):
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
        f"prefill_len_cutoff=256, FF prefill HiFi4, attn prefill+decode HiFi4+fp32, "
        f"decode spill W1→DRAM={t.mlp_decode_spill_w1_to_dram}, "
        f"perf_decode_tuning={perf_decode_tuning}"
    )
    return t


def _build_decoder_layer(
    *,
    idx: int,
    hf_layer: Any,
    qcfg: Qwen2_7BConfig,
    mesh_device: ttnn.MeshDevice,
    tt_ccl: Any,
    topology: Any,
    num_dev: int,
    torch_dtype: torch.dtype,
    precision: Qwen2_7BPrecisionConfig,
    executor_mode: bool,
    paged_cfg: Qwen2PagedAttentionConfig | None,
    cache_path: Path | None,
    wh: _Qwen2WHTuning,
) -> Qwen2_7BDecoderLayer:
    """Construct one decoder layer (attention + MLP + the two RMSNorms) from an HF layer."""
    prefix = f"layer{idx}"

    wqkv, wo, qn, kn, wqkv_b = weight_utils.attention_wqkv_wo_from_hf_layer(hf_layer.self_attn, num_dev)
    lazy_wqkv = _lazy(
        wqkv, dtype=precision.wqkv_dtype, cache=(cache_path / "attn", f"{prefix}_wqkv") if cache_path else None
    )
    lazy_wo = _lazy(wo, dtype=precision.wo_dtype, cache=(cache_path / "attn", f"{prefix}_wo") if cache_path else None)

    def _qk_norm_cfg(weight: torch.Tensor | None, name: str) -> RMSNorm1DConfig | None:
        if weight is None:
            return None
        lw = _lazy(
            weight.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch_dtype),
            dtype=ttnn.bfloat16,
            cache=(cache_path / "attn", f"{prefix}_{name}") if cache_path else None,
        )
        return RMSNorm1DConfig(
            weight=lw,
            mesh_device=mesh_device,
            eps=qcfg.rms_norm_eps,
            decode_in_sharded=False,
            decode_out_sharded=False,
            prefill_distributed=False,
            tt_ccl=tt_ccl,
        )

    bias_lw = (
        LazyWeight(
            source=wqkv_b.to(torch_dtype),
            dtype=ttnn.bfloat16,
            cache_dir_weight_name=(cache_path / "attn", f"{prefix}_bias") if cache_path else None,
        )
        if wqkv_b is not None
        else None
    )

    attn = Attention1D.from_config(
        Attention1DConfig(
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
            q_norm_config=_qk_norm_cfg(qn, "qn"),
            k_norm_config=_qk_norm_cfg(kn, "kn"),
            wqkv_bias=bias_lw,
            use_vllm_paged_kv_cache=executor_mode,
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
            max_batch_size=qcfg.max_batch_size,
            prefill_len_cutoff=wh.mlp_prefill_len_cutoff,
            ff1_3_compute_kernel_cfg=wh.mlp_ff_compute_kernel_cfg,
            ff2_compute_kernel_cfg=wh.mlp_ff_compute_kernel_cfg,
            decode_ff1_3_compute_kernel_cfg=wh.mlp_decode_ff_compute_kernel_cfg,
            decode_ff2_compute_kernel_cfg=wh.mlp_decode_ff_compute_kernel_cfg,
            decode_spill_w1_to_dram_before_w3=wh.mlp_decode_spill_w1_to_dram,
        )
    )

    post_attn_decode_program_config, post_attn_decode_memory_config = _post_attn_norm_decode_configs(
        mlp,
        dim=qcfg.dim,
        hidden_dim=qcfg.hidden_dim,
        num_devices=num_dev,
        max_batch_size=qcfg.max_batch_size,
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
                eps=qcfg.rms_norm_eps,
                max_batch_size=qcfg.max_batch_size,
                tt_ccl=tt_ccl,
                **extra,
            )
        )

    return Qwen2_7BDecoderLayer(
        input_layernorm=_build_norm(hf_layer.input_layernorm, "pre_attn"),
        self_attn=attn,
        post_attention_layernorm=_build_norm(
            hf_layer.post_attention_layernorm,
            "post_attn",
            decode_program_config=post_attn_decode_program_config,
            decode_memory_config=post_attn_decode_memory_config,
        ),
        mlp=mlp,
    )


def _build_lm_head(
    *,
    mesh_device: ttnn.MeshDevice,
    hf_lm_head: torch.nn.Module,
    qcfg: Qwen2_7BConfig,
    lm_head_dtype: ttnn.DataType,
    lm_head_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None,
    cache_path: Path | None,
) -> LMHead1D:
    """Build the vocab-sharded LM head with DRAM-matmul program configs.

    LM head DRAM matmul is sized for decode batch tiles (``max_batch_size``). Prefill logits
    use a single 32-row tile via ``post_process_prefill_output`` / :func:`_slice_last_token_tile`.
    """
    lm_w = hf_lm_head.weight.detach().to(torch.bfloat16).clone()
    lm_splits, lm_split_sizes, lm_weights_memcfgs = weight_utils.build_lm_head_lazy_weights(
        mesh_device,
        lm_w,
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
    return LMHead1D.from_config(
        LMHead1DConfig(
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
    )


class Qwen2_7BDecoderLayer(LightweightModule):
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

    def prefill_forward(
        self,
        x: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        *,
        user_id: int = 0,
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
    ) -> ttnn.Tensor:
        # Match Llama ``TransformerBlock1D``: fractured embed / norm activations must be
        # all-gathered to full ``dim`` before Attention1D / MLP1D (QKV matmul expects width ``dim``).
        r = self.input_layernorm.prefill_forward(x)
        r = _all_gather_rmsnorm_tensor(self.input_layernorm, r)
        r = self.self_attn.forward(
            r,
            None,
            rot_mats,
            mode="prefill",
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
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


class Qwen2_7B(LightweightModule):
    """
    Full decoder for Qwen2-7B-Instruct (TTTv2 modules only).

    Prefill/decode on **embedded** activations match ``EagerLLMExecutor``. Token embedding
    is ``embed_prefill`` / ``embed_decode``. Bind KV with ``set_kv_cache`` before first forward.
    """

    decode_residual_memcfg = ttnn.DRAM_MEMORY_CONFIG

    def __init__(
        self,
        cfg: Qwen2_7BConfig,
        embed: Embedding1D,
        rope_setup: RotarySetup1D,
        layers: List[Qwen2_7BDecoderLayer],
        norm: RMSNorm1D,
        lm_head: LMHead1D,
        mesh_device: ttnn.MeshDevice,
    ):
        super().__init__()
        self.cfg = cfg
        self.embed = embed
        self.rope_setup = rope_setup
        self.layers = layers
        self.norm = norm
        self.lm_head = lm_head
        self.mesh_device = mesh_device
        self.model_args: Qwen2ExecutorRuntimeConfig | None = None

        self.vocab_size = cfg.vocab_size
        self.n_layers = cfg.num_hidden_layers
        self.num_devices = mesh_device.get_num_devices()
        self.tt_ccl = get_tt_ccl(mesh_device) if self.num_devices > 1 else None

        # The model owns its sampler (replacing the self.sampling = None placeholder); callers
        # select behavior per request via sampling_params. test_sampling1d_trace_capture shows
        # argmax + top-k both capture/replay on 1x1 and 1x2, so the gate is all 1D meshes. Buffers
        # are lazy (nothing materializes until the first on-device sampled decode), so this is
        # harmless when sampling_params is None (the host-argmax default the demo ships with).
        self.supports_on_device_sampling = self.num_devices >= 1
        self.sampling = (
            Sampling1D(
                vocab_size=self.vocab_size,
                mesh_device=mesh_device,
                tt_ccl=self.tt_ccl,
                max_batch_size=_nearest_32(cfg.max_batch_size),
                # Clone TTTv1's decision: default_sampling_force_argmax.allow_force_argmax=False for
                # all non-Galaxy meshes. The top-k perf recipe (temp=0, top_p=0.08, top_k=32) routes
                # through the cheap top-k op path, never the full-vocab argmax all-gather.
                allow_force_argmax=False,
                pad_to_power_of_2=True,
            )
            if self.supports_on_device_sampling
            else None
        )

    @property
    def n_kv_heads(self) -> int:
        return self.cfg.n_kv_heads

    @classmethod
    def from_pretrained(
        cls,
        mesh_device: ttnn.MeshDevice,
        hf_model_id: str = "Qwen/Qwen2-7B-Instruct",
        *,
        revision: str | None = None,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        num_layers: int | None = None,
        cache_dir: Path | str | None = None,
        precision: Qwen2_7BPrecisionConfig = QWEN2_7B_ACCURACY,
        block_size: int = 32,
        executor_mode: bool = False,
    ) -> Qwen2_7B:
        """
        Load HF weights on host and build TTNN modules (weights materialize on first forward).

        Args:
            mesh_device: Open mesh device (N150 ``(1,1)``, N300 ``(1,2)``, …).
            hf_model_id: Hugging Face hub id.
            max_batch_size: Decode batch / KV allocation (tile-padded internally).
            max_seq_len: KV cache sequence budget (per layer).
            num_layers: If set, truncate stack for smoke tests.
            cache_dir: Optional directory for ``LazyWeight`` tensor caches.
            precision: Per-layer precision + math-fidelity recipe (see :class:`Qwen2_7BPrecisionConfig`).
                Defaults to :data:`QWEN2_7B_ACCURACY` (mirrors TTTv1
                ``DecodersPrecision.accuracy`` for Qwen2-7B). Use
                :data:`QWEN2_7B_PERFORMANCE` for TTTv1's perf recipe (BFP8 KV cache + BFP8
                LM head + ``perf_decode_tuning``), or ``dataclasses.replace(...)`` to customize
                a single field.
            block_size: Paged attention block size (tokens per block).
            executor_mode: If True, use external paged KV (``set_kv_cache`` + shared executor).
                If False, internal KV tensors (smoke / ``prefill_from_token_ids`` without executor).
        """
        ttnn.SetDefaultDevice(mesh_device)
        cache_path = Path(cache_dir) if cache_dir else None
        num_dev = mesh_device.get_num_devices()
        tt_ccl = get_tt_ccl(mesh_device) if num_dev > 1 else None
        topology = default_topology(mesh_device)

        hf_cfg = AutoConfig.from_pretrained(hf_model_id, revision=revision)
        n_heads_hf = hf_cfg.num_attention_heads
        n_kv_hf = hf_cfg.num_key_value_heads
        if num_dev > 1 and (n_heads_hf % num_dev != 0 or n_kv_hf % num_dev != 0):
            raise ValueError(
                f"This checkpoint requires num_attention_heads ({n_heads_hf}) and "
                f"num_key_value_heads ({n_kv_hf}) to each be divisible by the mesh device "
                f"count ({num_dev}) for Attention1D sharding. "
                f"Use a smaller mesh (e.g. 2 or 4 devices) or a model whose head counts divide the mesh."
            )
        torch_dtype = torch.bfloat16
        logger.info(f"Loading HF weights: {hf_model_id} (revision={revision})")
        hf = AutoModelForCausalLM.from_pretrained(hf_model_id, revision=revision, torch_dtype=torch_dtype)
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
        paged_cfg = (
            Qwen2PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks) if executor_mode else None
        )

        qcfg = Qwen2_7BConfig(
            hf_model_id=hf_model_id,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv,
            head_dim=head_dim,
            hidden_dim=inter,
            vocab_size=vocab,
            rms_norm_eps=hf_cfg.rms_norm_eps,
            rope_theta=getattr(hf_cfg, "rope_theta", 1_000_000.0),
            num_hidden_layers=n_layers,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            rope_table_len=rope_len,
        )

        emb_src = weight_utils.embed_tokens_torch(base.embed_tokens)
        emb = Embedding1D.from_config(
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

        wh = _resolve_qwen_wh_tuning(
            hf_model_id=hf_model_id,
            num_dev=num_dev,
            max_batch_size=max_batch_size,
            perf_decode_tuning=precision.perf_decode_tuning,
        )

        layers: list[Qwen2_7BDecoderLayer] = [
            _build_decoder_layer(
                idx=idx,
                hf_layer=base.layers[idx],
                qcfg=qcfg,
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                topology=topology,
                num_dev=num_dev,
                torch_dtype=torch_dtype,
                precision=precision,
                executor_mode=executor_mode,
                paged_cfg=paged_cfg,
                cache_path=cache_path,
                wh=wh,
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
            qcfg=qcfg,
            lm_head_dtype=precision.lm_head_dtype,
            lm_head_compute_kernel_cfg=wh.lm_head_compute_kernel_cfg,
            cache_path=cache_path,
        )

        del hf

        model = cls(qcfg, emb, rope_setup, layers, final_norm, lm, mesh_device)
        if executor_mode:
            model.model_args = Qwen2ExecutorRuntimeConfig(
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

    def set_kv_cache(self, kv_cache: list) -> None:
        assert len(kv_cache) == len(
            self.layers
        ), f"kv_cache has {len(kv_cache)} entries but model has {len(self.layers)} layers"
        for i, layer in enumerate(self.layers):
            layer.self_attn.config.kv_cache = tuple(kv_cache[i])

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
        *,
        user_id: int = 0,
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        get_last_token: int = -1,
    ) -> ttnn.Tensor:
        x = x_embed
        for layer in self.layers:
            x = layer.prefill_forward(
                x,
                rot_mats,
                user_id=user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
            )

        if get_last_token == -1:
            return x

        # Slice + deallocate the full-sequence buffer before norm/LM head reduces peak L1.
        old = x
        x_tile = _slice_last_token_tile(old, get_last_token)
        ttnn.deallocate(old)
        return self._last_tile_logits(x_tile)

    def post_process_prefill_output(self, hidden_states: ttnn.Tensor, last_token_idx: int) -> ttnn.Tensor:
        return self._last_tile_logits(_slice_last_token_tile(hidden_states, last_token_idx))

    def _last_tile_logits(self, x_tile: ttnn.Tensor) -> ttnn.Tensor:
        """Final-norm + all-gather + LM-head on a 32-row tile. ``x_tile`` shape ``[1, 1, 32, dim]``."""
        x = self.norm.prefill_forward(x_tile)
        x = _all_gather_rmsnorm_tensor(self.norm, x)
        lm_head_memcfg = self.lm_head.config.input_memcfg
        if lm_head_memcfg is not None and lm_head_memcfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_memcfg)
        x = self.lm_head.forward(x)
        return ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

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
