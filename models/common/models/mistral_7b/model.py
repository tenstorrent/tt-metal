# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral-7B-Instruct-v0.3 — native TTTv2 stack (``Embedding1D``, ``RMSNorm1D``,
``Attention1D``, ``MLP1D``, ``RotarySetup1D``, ``LMHead1D``).

Tensor layout contracts:
  - **Prefill** hidden states: ``[1, 1, S, dim]`` TILE, ``S % 128 == 0``.
  - **Decode** hidden states: ``[1, 1, B, dim]`` TILE (``B`` padded to tile in modules).

The tensor graph is provider-neutral. ``hf_adaptor.py`` owns Hugging Face loading and
builds :class:`Mistral7BTransformerConfig`; the model exposes the strict duck-typed
surface consumed by ``models.common.llm_runtime``.
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
from models.common.models.mistral_7b import weight_utils
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


def _lazy(
    tensor: torch.Tensor,
    *,
    dtype: ttnn.DataType,
    cache: tuple[Path, str] | None,
) -> LazyWeight:
    """Minimal LazyWeight; ``Attention1D`` / ``MLP1D`` / ``Embedding1D`` resolvers fill mesh + memory."""
    return LazyWeight(source=tensor, dtype=dtype, cache_dir_weight_name=cache)


@dataclass
class Mistral7BPagedAttentionConfig:
    """Paged KV layout for ``Attention1D`` (``block_size`` / ``max_num_blocks`` only)."""

    block_size: int
    max_num_blocks: int


@dataclass
class Mistral7BTransformerConfig:
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
    block_configs: list[Mistral7BDecoderLayerConfig]
    norm_config: RMSNorm1DConfig
    lm_head_config: LMHead1DConfig
    sampling_config: Sampling1DConfig | None = None
    decode_residual_memcfg: ttnn.MemoryConfig | None = None
    prefill_residual_memcfg: ttnn.MemoryConfig | None = None
    activation_dtypes: list[ttnn.DataType | None] = field(default_factory=list)
    tt_ccl: Any = None
    cache_path: str | None = None


@dataclass(frozen=True)
class Mistral7BModelParameters:
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
class Mistral7BLayerWeights:
    wqkv: torch.Tensor
    wo: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w3: torch.Tensor
    attention_norm: torch.Tensor
    ff_norm: torch.Tensor


@dataclass(frozen=True)
class Mistral7BWeights:
    embedding: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    layers: tuple[Mistral7BLayerWeights, ...]
    final_norm: torch.Tensor
    lm_head: torch.Tensor


@dataclass(frozen=True)
class Mistral7BPrecisionConfig:
    """Per-layer precision + math-fidelity recipe for Mistral-7B-Instruct-v0.3.

    Mirrors the fields TTTv1's ``DecodersPrecision`` actually distinguishes for Mistral-7B
    (Llama-family group in ``model_config.py:130-159`` for ``accuracy()``, ``:208-218`` for
    ``performance()``). Two module-level recipes are exposed: :data:`MISTRAL_ACCURACY` and
    :data:`MISTRAL_PERFORMANCE`. Pass one to :meth:`Mistral7BForCausalLM.from_pretrained` via
    ``precision=``; use ``dataclasses.replace(MISTRAL_ACCURACY, lm_head_dtype=...)`` to
    customize a single field.

    Attention compute-kernel configs (LI_QKV_*, LI_O_*, SDPA_*) are intentionally absent:
    TTTv1 leaves them at engine defaults for Mistral-7B (HIFI2 QKV/O, HIFI4 SDPA prefill,
    HIFI2 SDPA decode), which coincide with ``Attention1D``'s TTTv2 defaults. Add fields
    here only when a future variant overrides them.
    """

    wqkv_dtype: ttnn.DataType = ttnn.bfloat8_b
    wo_dtype: ttnn.DataType = ttnn.bfloat8_b
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat8_b

    mlp_w1_w3_dtype: ttnn.DataType = ttnn.bfloat8_b
    mlp_w2_dtype: ttnn.DataType = ttnn.bfloat8_b
    # ``None`` ⇒ MLP1D resolves to HIFI2_FP16 (matches TTTv1 ``LI_FF1_FF3`` / ``LI_FF2`` for accuracy).
    mlp_ff1_3_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    mlp_ff2_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None

    # Not part of TTTv1's ``DecodersPrecision`` — TTTv2 accuracy mode tightens this to bf16
    # to hit PERF.md top-1 thresholds.
    lm_head_dtype: ttnn.DataType = ttnn.bfloat8_b


_LOFI_COMPUTE_KERNEL_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


# TTTv1 ``DecodersPrecision.accuracy("Mistral-7B-Instruct-v0.3")`` (Llama-family group at
# ``model_config.py:130-159``) keeps BFP8 attention + BFP8 KV cache + HIFI2_FP16 MLP; only
# the LM head is tightened to bf16 (TTTv2 addition; required for top-1 in this stack).
MISTRAL_ACCURACY = Mistral7BPrecisionConfig(
    lm_head_dtype=ttnn.bfloat16,
)

# TTTv1 ``DecodersPrecision.performance("Mistral-7B-Instruct-v0.3")`` (``model_config.py:208-218``):
# FF1_FF3 → BFP4 and LI_FF1_FF3 → LOFI. Everything else matches the accuracy base; LM head
# stays BFP8 in perf mode. This single delta is the bulk of TTTv1's perf-mode throughput uplift.
MISTRAL_PERFORMANCE = Mistral7BPrecisionConfig(
    mlp_w1_w3_dtype=ttnn.bfloat4_b,
    mlp_ff1_3_compute_kernel_cfg=_LOFI_COMPUTE_KERNEL_CFG,
)


def _slice_last_token_tile(x: ttnn.Tensor, last_token_idx: int) -> ttnn.Tensor:
    """Slice the 32-row tile containing ``last_token_idx`` from ``[1, 1, S, W]``."""
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

    Same gotcha as Qwen: MLP1D decode uses ``_dram_shard_core_grid_k_n(dim, padded_hidden / num_devices)``
    for W1/W3 inputs, but the default RMSNorm program config is derived from ``_compute_norm_core_grid(dim)``
    alone — that DRAM-width-shard mismatch silently corrupts decode activations.
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
        num_links=tt_ccl.get_num_links(),
        topology=default_topology(cfg.mesh_device),
        memory_config=memory_config,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        # CCL tuning: match the house default (CCL_CHUNKS_PER_SYNC / CCL_NUM_WORKERS_PER_LINK in
        # tt_ccl.py = 10 / 2) used by every shared module (Attention1D/MLP1D/RMSNorm1D) and the
        # same-architecture llama3_8b reference (32 heads / 8 KV). The original port shipped a
        # divergent 24 / 4 here. These two per-layer rmsnorm all-gathers run 64x/decode step, so on
        # the T3K 8-device ring their per-op cost is non-trivial; aligning to the validated house
        # default is correctness-neutral (all-gather result is identical) and removes an unexplained
        # divergence. (The earlier "~17% T3K decode gap vs TTTv1" this was suspected to drive turned
        # out to be a cross-box/cross-harness measurement artifact, not a real regression: on equal
        # footing -- same box, on-device sampling -- T3K decode is at parity with TTTv1.)
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )


@dataclass
class _Mistral7BWHTuning:
    """Wormhole-specific MLP tuning resolved at build time.

    Mistral 7B (TTTv1 ``DecodersPrecision.accuracy``) keeps HIFI2_FP16 / BFP8 across
    attention + FF, so we do **not** override the ``Attention1D`` / ``MLP1D`` math-fidelity
    defaults. Only the MLP prefill_len_cutoff is set here, matching TTTv1's WH cutoff for
    7B-class Llama-family models (512 on N150; 1024 on multi-device WH).
    """

    mlp_prefill_len_cutoff: int | None = None
    mlp_decode_spill_w1_to_dram: bool = False
    # Use ttnn.experimental.minimal_matmul for QKV + W2 prefill matmuls above seq_len > 128 (TTTv1
    # parity, PLAN_01). A/B escape hatch: set DISABLE_MINIMAL_MATMUL=1 to force ttnn.linear. On a 7B
    # the prefill matmuls are large, so minimal_matmul is a real prefill-TTFT win (unlike tiny 1B).
    prefill_minimal_matmul: bool = True


def _resolve_mistral_wh_tuning(*, num_dev: int, max_batch_size: int) -> _Mistral7BWHTuning:
    """Pick WH tuning knobs for Mistral-7B-Instruct-v0.3 on N150 / N300.

    TTTv1 ([model_config.py:516-589](../../../tt_transformers/tt/model_config.py#L516-L589)) sets
    ``prefill_len_cutoff=1024`` on WH by default and reduces it to 512 on N150 for Llama 3.1-8B /
    Mistral-7B / Phi-4. We mirror that here.
    """
    t = _Mistral7BWHTuning()
    t.mlp_prefill_len_cutoff = 512 if num_dev == 1 else 1024
    # Decode W1→DRAM spill: leave off by default; promote only if N150 batch decode trips L1.
    t.mlp_decode_spill_w1_to_dram = False
    t.prefill_minimal_matmul = not os.environ.get("DISABLE_MINIMAL_MATMUL")
    logger.info(
        f"MLP tuning for Mistral-7B on {num_dev} device(s): "
        f"prefill_len_cutoff={t.mlp_prefill_len_cutoff}, "
        f"decode_spill_w1_to_dram={t.mlp_decode_spill_w1_to_dram}, "
        f"prefill_minimal_matmul={t.prefill_minimal_matmul}"
    )
    return t


@dataclass
class Mistral7BDecoderLayerConfig:
    attention_norm_config: RMSNorm1DConfig
    attention_config: Attention1DConfig
    ff_norm_config: RMSNorm1DConfig
    mlp_config: MLP1DConfig


def _build_decoder_layer(
    *,
    idx: int,
    weights: Mistral7BLayerWeights,
    mcfg: Mistral7BModelParameters,
    mesh_device: ttnn.MeshDevice,
    tt_ccl: Any,
    topology: Any,
    num_dev: int,
    precision: Mistral7BPrecisionConfig,
    paged_cfg: Mistral7BPagedAttentionConfig,
    cache_path: Path | None,
    wh: _Mistral7BWHTuning,
) -> Mistral7BDecoderLayerConfig:
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
        q_norm_config=None,
        k_norm_config=None,
        wqkv_bias=None,
        use_vllm_paged_kv_cache=True,
        paged_attention_config=paged_cfg,
        kv_cache=None,
        kv_cache_dtype=precision.kv_cache_dtype,
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

    return Mistral7BDecoderLayerConfig(
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
    mcfg: Mistral7BModelParameters,
    lm_head_dtype: ttnn.DataType,
    cache_path: Path | None,
) -> LMHead1DConfig:
    """Build the vocab-sharded LM head with DRAM-matmul program configs."""
    lm_splits, lm_split_sizes, lm_weights_memcfgs = weight_utils.build_lm_head_lazy_weights(
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


def build_mistral_7b_transformer_config(
    *,
    mesh_device: ttnn.MeshDevice,
    params: Mistral7BModelParameters,
    weights: Mistral7BWeights,
    n_layers: int,
    precision: Mistral7BPrecisionConfig,
    cache_path: Path | None,
    paged_attention_config: Mistral7BPagedAttentionConfig,
) -> Mistral7BTransformerConfig:
    """Build the TT tensor graph from provider-neutral dimensions and converted tensors."""

    num_devices = mesh_device.get_num_devices()
    if params.n_heads % num_devices or params.n_kv_heads % num_devices:
        raise ValueError(
            f"Checkpoint heads ({params.n_heads}/{params.n_kv_heads}) "
            f"must be divisible by device count ({num_devices})"
        )
    if len(weights.layers) != n_layers:
        raise ValueError(f"Expected {n_layers} decoder layer weight sets, got {len(weights.layers)}")

    tt_ccl = get_tt_ccl(mesh_device) if num_devices > 1 else None
    topology = default_topology(mesh_device)
    wh = _resolve_mistral_wh_tuning(num_dev=num_devices, max_batch_size=params.max_batch_size)
    embedding_config = Embedding1DConfig(
        weights=_lazy(
            weights.embedding,
            dtype=ttnn.bfloat16,
            cache=(cache_path / "embedding", "tok_embeddings") if cache_path else None,
        ),
        mesh_device=mesh_device,
        embed_scale=1.0,
    )
    rope_config = Rope1DConfig(
        cos_matrix=_lazy(
            weights.rope_cos,
            dtype=ttnn.bfloat16,
            cache=(cache_path / "rope", "cos") if cache_path else None,
        ),
        sin_matrix=_lazy(
            weights.rope_sin,
            dtype=ttnn.bfloat16,
            cache=(cache_path / "rope", "sin") if cache_path else None,
        ),
        max_batch_size=params.max_batch_size,
        head_dim=params.head_dim,
        device=mesh_device,
        use_qk_fused=False,
    )
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
            paged_cfg=paged_attention_config,
            cache_path=cache_path,
            wh=wh,
        )
        for index in range(n_layers)
    ]
    norm_config = RMSNorm1DConfig(
        weight=_lazy(
            weights.final_norm,
            dtype=ttnn.bfloat16,
            cache=(cache_path / "norm", "final") if cache_path else None,
        ),
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
    return Mistral7BTransformerConfig(
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
        cache_path=str(cache_path) if cache_path is not None else None,
    )


class Mistral7BDecoderLayer(LightweightModule):
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
    def from_config(cls, config: Mistral7BDecoderLayerConfig) -> Mistral7BDecoderLayer:
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
        # Fractured embed/norm activations must be all-gathered to full ``dim`` before
        # Attention1D / MLP1D (QKV matmul expects width ``dim``).
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


class Mistral7B(LightweightModule):
    """
    Full decoder for Mistral-7B-Instruct-v0.3 (TTTv2 modules only).

    Prefill/decode on **embedded** activations are orchestrated by ``Mistral7BExecutor``. Token
    embedding is ``embed_prefill`` / ``embed_decode``. Bind KV with ``set_kv_cache`` before first forward.
    """

    decode_residual_memcfg = ttnn.DRAM_MEMORY_CONFIG

    def __init__(self, config: Mistral7BTransformerConfig):
        from tqdm import tqdm

        super().__init__()
        self.config = config
        self.embed = Embedding1D.from_config(config.embedding_config)
        self.embedding = self.embed
        self.rope_setup = RotarySetup1D.from_config(config.rope_config)
        self.layers = [
            Mistral7BDecoderLayer.from_config(config.block_configs[index])
            for index in tqdm(range(config.n_layers), desc="Building layers")
        ]
        self.norm = RMSNorm1D.from_config(config.norm_config)
        self.lm_head = LMHead1D.from_config(config.lm_head_config)
        self.sampling = Sampling1D.from_config(config.sampling_config) if config.sampling_config is not None else None
        self.supports_on_device_sampling = self.sampling is not None
        self.mesh_device = config.mesh_device
        self.tt_ccl = config.tt_ccl or (get_tt_ccl(config.mesh_device) if config.num_devices > 1 else None)
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
        # batch_size > 1: x_embed is the folded [1,1,B*S,dim] tensor (B users). The batched path always
        # returns the full hidden state when no runtime last-token slice is supplied; the executor does
        # per-slot last-token extraction + norm/lm_head so those stages stay bit-identical to the single-user path.
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

        if last_token_index is not None and last_token_slice is None:
            raise ValueError("last_token_index is required with a runtime last_token_slice")
        if get_last_token == -1 and last_token_slice is None:
            return x

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
                # House-default CCL tuning (10 / 2), matching llama3_8b. Only used in host
                # sampling mode (full-vocab gather); the original 24 / 4 is the suspected reason
                # T3K host mode was the worst TTTv2 mode. See _all_gather_rmsnorm_tensor for rationale.
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
        return ttnn.untilize(logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def increment_positions(self, current_pos: ttnn.Tensor, rot_mat_idxs: ttnn.Tensor) -> None:
        ttnn.plus_one(current_pos, skip_negative_entries=True)
        ttnn.plus_one(rot_mat_idxs)
