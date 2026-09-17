# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-32B — native TTTv2 stack (``Embedding1D``, ``RMSNorm1D``,
``Attention1D``, ``MLP1D``, ``RotarySetup1D``, ``LMHead1D``). Targets Wormhole
T3K (mesh ``(1, 8)``) and BlackHole P150x4 (mesh ``(1, 4)``).

Tensor layout contracts:
  - **Prefill** hidden states: ``[1, 1, S, dim]`` TILE, ``S % 128 == 0``.
  - **Decode** hidden states: ``[1, 1, B, dim]`` TILE (``B`` padded to tile in modules).

Model-owned executor contract: pre-embedded forwards, ``set_kv_cache``,
``rope_setup``, ``page_table`` through attention, ``model_args`` holds a
:class:`Qwen3_32BExecutorRuntimeConfig` (not v1 ``ModelArgs``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, List

import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.models.qwen3_32b import weight_utils
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
from models.common.modules.tt_ccl import get_tt_ccl
from models.common.tensor_utils import TILE_SIZE, get_padded_hidden_dim

# Pinned HF revision SHA for Qwen/Qwen3-32B (resolved 2026-06-03).
DEFAULT_HF_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
QWEN3_32B_INTERMEDIATE_SIZE = 25600

# Both physical Blackhole four-die products expose the canonical logical
# ``P150x4`` Qwen SKU.  Keep this product admission and its CCL recipe owned by
# the model: the shared CCL fallback does not know that P300_X2 has a physical
# Ring and would otherwise select Linear.
QWEN3_32B_BH_TP4_CLUSTER_TYPES = (
    ttnn.cluster.ClusterType.P150_X4,
    ttnn.cluster.ClusterType.P300_X2,
)


def _qwen3_ccl_topology(mesh_device) -> ttnn.Topology:
    """Return the fail-closed CCL topology for an admitted Qwen3-32B mesh."""

    arch = mesh_device.arch()
    cluster_type = ttnn.cluster.get_cluster_type()
    num_devices = mesh_device.get_num_devices()
    if (arch == ttnn.device.Arch.WORMHOLE_B0 and cluster_type == ttnn.cluster.ClusterType.T3K and num_devices == 8) or (
        arch == ttnn.device.Arch.BLACKHOLE and cluster_type in QWEN3_32B_BH_TP4_CLUSTER_TYPES and num_devices == 4
    ):
        return ttnn.Topology.Ring
    raise ValueError(
        "Qwen3-32B CCL supports physical Wormhole T3K or BlackHole P150_X4/P300_X2; "
        f"got arch={arch}, cluster_type={cluster_type}, num_devices={num_devices}"
    )


def _lazy(
    tensor: torch.Tensor,
    *,
    dtype: ttnn.DataType,
    cache: tuple[Path, str] | None,
) -> LazyWeight:
    """Minimal LazyWeight; ``Attention1D`` / ``MLP1D`` / ``Embedding1D`` resolvers fill mesh + memory."""
    return LazyWeight(source=tensor, dtype=dtype, cache_dir_weight_name=cache)


@dataclass
class Qwen3_32BPagedAttentionConfig:
    """Paged KV layout for ``Attention1D`` (``block_size`` / ``max_num_blocks`` only)."""

    block_size: int
    max_num_blocks: int


@dataclass
class Qwen3_32BExecutorRuntimeConfig:
    """Engine-facing runtime knobs exposed as ``model.model_args``."""

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
    # batches models whose prefill_forward threads ``batch_size`` — Qwen3-32B does, below). Qwen3's
    # per-head QK-norm is a row-independent RMSNorm on ``[B, n_heads, S, head_dim]`` (each user's rows
    # normalized independently), so the batched fold is bit-safe. ``max_prefill_batch_size`` is the
    # largest supported padded wave; 32 folds the whole batch-32 prefill in ONE 32-user pass (TTTv1 structural parity,
    # generator.py:679-700) so the eager norm+lm_head tail + full-vocab readback run once instead of 4×.
    # At S=128 the fold is 32*128=4096=2*2048, an exact multiple of MAX_QKV_MM_SEQ_LEN (reshape-safe).
    # ``disable_batched_prefill`` is the escape hatch back to the sequential loop;
    # ``max_prefill_chunk_size`` (above) drives the #45234 decline.
    supports_batched_prefill: bool = True
    max_prefill_batch_size: int = 32
    disable_batched_prefill: bool = False
    # When True (default), batched prefill runs norm+lm_head ONCE per group over the gathered last-token
    # rows (TTTv1 parity); False falls back to the bit-identical per-slot path (one lm_head per user).
    batched_prefill_batched_extract: bool = True
    # Both supported Qwen products preserve the TTTv1 Q128/Q1024 prefill trace
    # buckets.  Keep this material explicit because the executor snapshots it
    # independently from ``can_enable_trace`` when constructing warmup coverage.
    trace_prefill_supported_seq_lens: tuple[int, ...] = (128, 1024)

    def can_enable_trace(self, prefill_seq_len: int, num_cached_tokens: int = 0) -> bool:
        # Only trace the sequence lengths retained by the model-owned product.
        # Decode trace stays enabled at the engine layer regardless.
        return (
            num_cached_tokens == 0
            and prefill_seq_len in self.trace_prefill_supported_seq_lens
            and prefill_seq_len <= self.max_prefill_chunk_size
            and prefill_seq_len <= self.max_seq_len
        )


@dataclass
class Qwen3_32BConfig:
    """Resolved hyper-parameters for a loaded HF Qwen3-32B checkpoint."""

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
    num_devices: int = 8
    mesh_device: ttnn.MeshDevice | None = None
    n_layers: int | None = None
    block_configs: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_layers is None:
            self.n_layers = self.num_hidden_layers


_HIFI4_FP32_KERNEL = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
    dst_full_sync_en=False,
)
"""HiFi4 + fp32 dest acc for Qwen3-32B attention matmuls (LI_QKV, LI_O, SDPA).

TTTv1 ``DecodersPrecision.accuracy("Qwen3-32B")`` resolves to the generic ``else``
branch in ``model_config.py:160-177`` which forces all attention ops to ``HIFI4``. The
TTTv2 ``Attention1D`` default is ``HIFI2`` with fp16 accumulation; without this override,
attention QKV / WO / SDPA produce a broad per-layer drift vs HF (same regression debugged
on the Qwen2.5-7B port). Used in ``QWEN3_32B_ACCURACY``.
"""


_HIFI2_FP32_APPROX_KERNEL = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
    dst_full_sync_en=False,
)

_HIFI2_FP32_KERNEL = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
    dst_full_sync_en=False,
)

_HIFI2_FP16_KERNEL = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
    dst_full_sync_en=False,
)

_LOFI_COMPUTE_KERNEL_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
    dst_full_sync_en=False,
)
"""LoFi + packer L1 acc for the MLP FF1/FF3 matmuls in performance mode.

Mirrors TTTv1 ``DecodersPrecision.performance("Qwen3-32B")``: the generic ``else``
branch at ``model_config.py:208-218`` sets ``FF1_FF3 → BFP4`` and
``LI_FF1_FF3 → LOFI``. This single delta is the bulk of the perf-mode throughput uplift.
"""


@dataclass(frozen=True)
class Qwen3_32BPrecisionConfig:
    """Per-layer precision + math-fidelity recipe for Qwen3-32B.

    Mirrors the fields TTTv1's ``DecodersPrecision`` distinguishes for Qwen3-32B.
    The base model name resolves to ``Qwen3-32B`` (via ``common.get_base_model_name``),
    which falls into the generic ``else`` branch — **not** the >70B (``model_config.py:119``),
    Llama-3 / Mistral-7B / Phi (``:130``), or Qwen2.5-7B (``:187``) special cases. As a result:

      * **Accuracy** (``model_config.py:160-177``): BF16 ``WQKV`` / ``KV_CACHE`` / ``WO`` +
        ``HIFI4`` on every ``LI_QKV`` / ``SDPA`` / ``LI_O``. FF and LM head stay at engine
        defaults (BFP8 FF + ``HIFI2_FP16``).
      * **Performance** (``model_config.py:208-218``, generic ``else`` branch): only ``FF1_FF3 → BFP4``
        and ``LI_FF1_FF3 → LOFI``. Everything else reverts to TTTv1 defaults
        (BFP8 attention + ``HIFI2`` attention kernels + BFP8 KV cache).

    Two module-level recipes are exposed: :data:`QWEN3_32B_ACCURACY` (default) and
    :data:`QWEN3_32B_PERFORMANCE`. Pass one to :meth:`Qwen3_32B.from_pretrained`
    via ``precision=``; use ``dataclasses.replace(QWEN3_32B_ACCURACY, ...)`` to
    customize a single field. Defaults below mirror the accuracy recipe so ``Qwen3_32BPrecisionConfig()``
    is equivalent to :data:`QWEN3_32B_ACCURACY`.

    Every parity-critical compute slot is explicit even when its value equals a
    shared default, so a later shared-module change cannot silently alter this
    validated model profile.
    """

    # Attention weight / KV-cache dtypes. Accuracy overrides BF16; performance keeps BFP8 default.
    wqkv_dtype: ttnn.DataType = ttnn.bfloat16
    wo_dtype: ttnn.DataType = ttnn.bfloat16
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat16

    # MLP FF1/FF3 weight dtype. Accuracy keeps BFP8 default; performance overrides BFP4.
    mlp_w1_w3_dtype: ttnn.DataType = ttnn.bfloat8_b

    # Four explicit MLP operation slots. Accuracy uses the TTTv1 HIFI2_FP16
    # baseline; performance changes only FF1/FF3 to LOFI.
    mlp_prefill_ff1_ff3_kernel: ttnn.DeviceComputeKernelConfig = _HIFI2_FP16_KERNEL
    mlp_prefill_ff2_kernel: ttnn.DeviceComputeKernelConfig = _HIFI2_FP16_KERNEL
    mlp_decode_ff1_ff3_kernel: ttnn.DeviceComputeKernelConfig = _HIFI2_FP16_KERNEL
    mlp_decode_ff2_kernel: ttnn.DeviceComputeKernelConfig = _HIFI2_FP16_KERNEL

    # Six explicit attention slots. Accuracy is HIFI4/FP32 throughout. The
    # performance profile below spells out TTTv1's asymmetric default table.
    attn_decode_qkv_kernel: ttnn.DeviceComputeKernelConfig = _HIFI4_FP32_KERNEL
    attn_decode_sdpa_kernel: ttnn.DeviceComputeKernelConfig = _HIFI4_FP32_KERNEL
    attn_decode_wo_kernel: ttnn.DeviceComputeKernelConfig = _HIFI4_FP32_KERNEL
    attn_prefill_qkv_kernel: ttnn.DeviceComputeKernelConfig = _HIFI4_FP32_KERNEL
    attn_prefill_sdpa_kernel: ttnn.DeviceComputeKernelConfig = _HIFI4_FP32_KERNEL
    attn_prefill_wo_kernel: ttnn.DeviceComputeKernelConfig = _HIFI4_FP32_KERNEL

    # LM-head weight dtype. Both Qwen3-32B recipes use BFP8 (== TTTv1, which never upgrades the
    # LM head). The bf16 "accuracy tightening" the Mistral / Coder-32B ports added regresses
    # Qwen3-32B token-accuracy (see QWEN3_32B_ACCURACY below) — do not raise this to bf16 here.
    lm_head_dtype: ttnn.DataType = ttnn.bfloat8_b


# TTTv1 ``DecodersPrecision.accuracy("Qwen3-32B")`` (``model_config.py:160-177``):
# BF16 attention weights + BF16 KV cache + HIFI4 + fp32_dest_acc on every attention stage.
# FF and LM head sit at TTTv2/TTTv1 defaults (BFP8 + HIFI2_FP16). LM head stays BFP8 — bf16 regresses
# Qwen3-32B token-accuracy (86.3% vs 98.8%), unlike the Mistral / Coder-32B ports.
QWEN3_32B_ACCURACY = Qwen3_32BPrecisionConfig(
    # Qwen3-32B accuracy uses BFP8 LM head (== TTTv1 ``DecodersPrecision.accuracy``, which never
    # upgrades the LM head). A bf16 LM head — the TTTv2 "accuracy tightening" the Mistral / Coder-32B
    # ports added — *regresses* Qwen3-32B token-accuracy from 98.8% → 86.3% top-1. Bisected on T3K
    # 2026-06-04: with every other accuracy field held fixed (bf16 WQKV/WO/KV-cache, HiFi4+fp32 attn),
    # only `lm_head_dtype` bf16→bf8 moves top-1 86.3→98.8 / top-5 87.9→100. The bf16 DRAM-sharded
    # decode LM-head matmul on the (zero-)padded 152064 vocab corrupts the logit ranking; Coder-32B's
    # 152064 vocab was already tile-aligned per device, so its bf16 LM head never exercised the bug.
    lm_head_dtype=ttnn.bfloat8_b,
)

# TTTv1 ``DecodersPrecision.performance("Qwen3-32B")`` (``model_config.py:208-218``,
# non-7B branch): FF1_FF3 → BFP4 and LI_FF1_FF3 → LOFI; everything else reverts to engine defaults
# (BFP8 attention, HIFI2 attention kernels, BFP8 KV cache, BFP8 LM head).
QWEN3_32B_PERFORMANCE = Qwen3_32BPrecisionConfig(
    wqkv_dtype=ttnn.bfloat8_b,
    wo_dtype=ttnn.bfloat8_b,
    kv_cache_dtype=ttnn.bfloat8_b,
    mlp_w1_w3_dtype=ttnn.bfloat4_b,
    mlp_prefill_ff1_ff3_kernel=_LOFI_COMPUTE_KERNEL_CFG,
    mlp_decode_ff1_ff3_kernel=_LOFI_COMPUTE_KERNEL_CFG,
    attn_decode_qkv_kernel=_HIFI2_FP32_APPROX_KERNEL,
    attn_decode_sdpa_kernel=_HIFI2_FP32_APPROX_KERNEL,
    attn_decode_wo_kernel=_HIFI2_FP32_APPROX_KERNEL,
    attn_prefill_qkv_kernel=_HIFI2_FP32_APPROX_KERNEL,
    attn_prefill_sdpa_kernel=_HIFI4_FP32_KERNEL,
    attn_prefill_wo_kernel=_HIFI2_FP32_APPROX_KERNEL,
    lm_head_dtype=ttnn.bfloat8_b,
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
    alone. Mismatched DRAM-width-shard between RMSNorm output and MLP1D W1/W3 input silently
    corrupts decode activations (observed on the 7B port — same shape pattern on T3K Qwen3-32B).
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
        if memory_config is not None:
            return ttnn.to_memory_config(x, memory_config)
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
        topology=_qwen3_ccl_topology(cfg.mesh_device),
        memory_config=memory_config,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        chunks_per_sync=24,
        num_workers_per_link=4,
        num_buffers_per_channel=2,
    )


@dataclass(frozen=True, slots=True)
class _Qwen3_32BSKUOverlay:
    """Effective WH-T3K or BH-P150x4 topology and geometry policy."""

    architecture: str
    topology: ttnn.Topology
    dram_shard_grid_width: int
    mlp_prefill_len_cutoff: int
    mlp_prefill_grid: tuple[int, int]
    mlp_decode_spill_w1_to_dram: bool = False
    prefill_minimal_matmul: bool = True
    attention_prefill_qkv_grid: tuple[int, int] = (8, 8)
    attention_decode_create_qkv_head_grid: ttnn.CoreGrid | None = None
    attention_decode_transformation_grid: ttnn.CoreCoord | None = None
    lm_head_core_grid: ttnn.CoreGrid | None = None
    lm_head_max_columns_per_device: int = 8192
    distributed_rmsnorm_min_dim_exclusive: int | None = None
    disable_batched_prefill: bool = False


def _resolve_qwen3_32b_sku_overlay(*, arch, cluster_type, num_dev: int, mesh_device) -> _Qwen3_32BSKUOverlay:
    """Select the approved Qwen3-32B WH-T3K or BH-P150x4 overlay."""
    minimal = not os.environ.get("DISABLE_MINIMAL_MATMUL")
    if arch == ttnn.device.Arch.WORMHOLE_B0 and cluster_type == ttnn.cluster.ClusterType.T3K and num_dev == 8:
        overlay = _Qwen3_32BSKUOverlay(
            architecture="wormhole",
            topology=ttnn.Topology.Ring,
            # Preserve the established WH recipe, which shards over 8 DRAM
            # banks even though Wormhole physically exposes 12 DRAM cores.
            dram_shard_grid_width=8,
            mlp_prefill_len_cutoff=1024,
            mlp_prefill_grid=(8, 8),
            prefill_minimal_matmul=minimal,
            attention_prefill_qkv_grid=(8, 8),
            attention_decode_transformation_grid=mesh_device.compute_with_storage_grid_size(),
        )
    elif arch == ttnn.device.Arch.BLACKHOLE and cluster_type in QWEN3_32B_BH_TP4_CLUSTER_TYPES and num_dev == 4:
        overlay = _Qwen3_32BSKUOverlay(
            architecture="blackhole",
            topology=ttnn.Topology.Ring,
            dram_shard_grid_width=mesh_device.dram_grid_size().x,
            mlp_prefill_len_cutoff=512,
            mlp_prefill_grid=(8, 5),
            prefill_minimal_matmul=minimal,
            attention_prefill_qkv_grid=(8, 4),
            attention_decode_create_qkv_head_grid=ttnn.CoreGrid(x=8, y=4),
            attention_decode_transformation_grid=ttnn.CoreCoord(8, 8),
            lm_head_core_grid=ttnn.CoreGrid(x=8, y=5),
            lm_head_max_columns_per_device=4008,
            distributed_rmsnorm_min_dim_exclusive=4096,
            disable_batched_prefill=True,
        )
    else:
        raise ValueError(
            "Qwen3-32B supports Wormhole T3K (8 devices) or BlackHole P150_X4/P300_X2 (4 devices); "
            f"got arch={arch}, cluster_type={cluster_type}, num_devices={num_dev}"
        )
    logger.info(
        f"Qwen3-32B {overlay.architecture} SKU overlay on {num_dev} devices: "
        f"topology={overlay.topology}, mlp_grid={overlay.mlp_prefill_grid}, "
        f"attention_grid={overlay.attention_prefill_qkv_grid}, "
        f"cutoff={overlay.mlp_prefill_len_cutoff}, minimal_matmul={overlay.prefill_minimal_matmul}"
    )
    return overlay


def _qwen3_rmsnorm_config(common: RMSNorm1DConfig) -> RMSNorm1DConfig:
    """Apply the model-owned RMSNorm recipe without mutating the caller config."""
    return replace(common, compute_kernel_config=_HIFI2_FP32_KERNEL)


def _qwen3_attention_config(
    common: Attention1DConfig,
    *,
    sku: _Qwen3_32BSKUOverlay,
    precision: Qwen3_32BPrecisionConfig,
) -> Attention1DConfig:
    return replace(
        common,
        li_qkv_decode_compute_kernel_cfg=precision.attn_decode_qkv_kernel,
        sdpa_decode_compute_kernel_cfg=precision.attn_decode_sdpa_kernel,
        li_o_decode_compute_kernel_cfg=precision.attn_decode_wo_kernel,
        li_qkv_prefill_compute_kernel_cfg=precision.attn_prefill_qkv_kernel,
        sdpa_prefill_compute_kernel_cfg=precision.attn_prefill_sdpa_kernel,
        li_o_prefill_compute_kernel_cfg=precision.attn_prefill_wo_kernel,
        prefill_qkv_grid=sku.attention_prefill_qkv_grid,
        dram_shard_grid_width=sku.dram_shard_grid_width,
        decode_create_qkv_head_grid=sku.attention_decode_create_qkv_head_grid,
        decode_transformation_core_grid=sku.attention_decode_transformation_grid,
    )


def _qwen3_mlp_config(
    common: MLP1DConfig,
    *,
    sku: _Qwen3_32BSKUOverlay,
    precision: Qwen3_32BPrecisionConfig,
) -> MLP1DConfig:
    return replace(
        common,
        ff1_3_compute_kernel_cfg=precision.mlp_prefill_ff1_ff3_kernel,
        ff2_compute_kernel_cfg=precision.mlp_prefill_ff2_kernel,
        decode_ff1_3_compute_kernel_cfg=precision.mlp_decode_ff1_ff3_kernel,
        decode_ff2_compute_kernel_cfg=precision.mlp_decode_ff2_kernel,
        prefill_len_cutoff=sku.mlp_prefill_len_cutoff,
        prefill_dram_shard_grid_width=sku.dram_shard_grid_width,
        prefill_ff1_ff3_grid=sku.mlp_prefill_grid,
        prefill_ff2_grid=sku.mlp_prefill_grid,
    )


def _qwen3_lm_head_config(common: LMHead1DConfig) -> LMHead1DConfig:
    return replace(common, compute_kernel_config=_HIFI2_FP16_KERNEL)


def _build_decoder_layer(
    *,
    idx: int,
    hf_layer: Any,
    qcfg: Qwen3_32BConfig,
    mesh_device: ttnn.MeshDevice,
    tt_ccl: Any,
    topology: Any,
    num_dev: int,
    torch_dtype: torch.dtype,
    precision: Qwen3_32BPrecisionConfig,
    executor_mode: bool,
    paged_cfg: Qwen3_32BPagedAttentionConfig | None,
    cache_path: Path | None,
    sku: _Qwen3_32BSKUOverlay,
) -> Qwen3_32BDecoderLayer:
    """Construct one decoder layer (attention + MLP + the two RMSNorms) from an HF layer."""
    prefix = f"layer{idx}"

    wqkv, wo, qn, kn, wqkv_b = weight_utils.attention_wqkv_wo_from_hf_layer(hf_layer.self_attn, num_dev)
    lazy_wqkv = _lazy(
        wqkv, dtype=precision.wqkv_dtype, cache=(cache_path / "attn", f"{prefix}_wqkv") if cache_path else None
    )
    lazy_wo = _lazy(wo, dtype=precision.wo_dtype, cache=(cache_path / "attn", f"{prefix}_wo") if cache_path else None)

    def _qk_norm_cfg(weight: torch.Tensor | None, name: str):
        if weight is None:
            return None
        lw = _lazy(
            weight.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch_dtype),
            dtype=ttnn.bfloat16,
            cache=(cache_path / "attn", f"{prefix}_{name}") if cache_path else None,
        )
        common = RMSNorm1DConfig(
            weight=lw,
            mesh_device=mesh_device,
            eps=qcfg.rms_norm_eps,
            decode_in_sharded=False,
            decode_out_sharded=False,
            prefill_distributed=False,
            tt_ccl=tt_ccl,
        )
        return _qwen3_rmsnorm_config(common)

    bias_lw = (
        LazyWeight(
            source=wqkv_b.to(torch_dtype),
            dtype=ttnn.bfloat16,
            cache_dir_weight_name=(cache_path / "attn", f"{prefix}_bias") if cache_path else None,
        )
        if wqkv_b is not None
        else None
    )

    attention_common = Attention1DConfig(
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
        prefill_qkv_minimal_matmul=sku.prefill_minimal_matmul,
    )
    attention_config = _qwen3_attention_config(
        attention_common,
        sku=sku,
        precision=precision,
    )
    attn = Attention1D.from_config(attention_config)

    w1, w2, w3 = weight_utils.mlp_weights_from_hf_layer(hf_layer.mlp)
    mlp_common = MLP1DConfig(
        w1=_lazy(
            w1, dtype=precision.mlp_w1_w3_dtype, cache=(cache_path / "mlp", f"{prefix}_w1") if cache_path else None
        ),
        w2=_lazy(w2, dtype=ttnn.bfloat8_b, cache=(cache_path / "mlp", f"{prefix}_w2") if cache_path else None),
        w3=_lazy(
            w3, dtype=precision.mlp_w1_w3_dtype, cache=(cache_path / "mlp", f"{prefix}_w3") if cache_path else None
        ),
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        topology=topology,
        dim=qcfg.dim,
        hidden_dim=qcfg.hidden_dim,
        max_batch_size=qcfg.max_batch_size,
        w1_w3_dtype=precision.mlp_w1_w3_dtype,
        w2_dtype=ttnn.bfloat8_b,
        decode_spill_w1_to_dram_before_w3=sku.mlp_decode_spill_w1_to_dram,
        prefill_w2_minimal_matmul=sku.prefill_minimal_matmul,
    )
    mlp = MLP1D.from_config(
        _qwen3_mlp_config(
            mlp_common,
            sku=sku,
            precision=precision,
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
        common = RMSNorm1DConfig(
            weight=lw,
            mesh_device=mesh_device,
            eps=qcfg.rms_norm_eps,
            max_batch_size=qcfg.max_batch_size,
            tt_ccl=tt_ccl,
            **extra,
        )
        if sku.distributed_rmsnorm_min_dim_exclusive is not None:
            common.prefill_distributed = qcfg.dim > sku.distributed_rmsnorm_min_dim_exclusive
        return RMSNorm1D.from_config(_qwen3_rmsnorm_config(common))

    return Qwen3_32BDecoderLayer(
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
    qcfg: Qwen3_32BConfig,
    lm_head_dtype: ttnn.DataType,
    cache_path: Path | None,
    sku: _Qwen3_32BSKUOverlay,
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
        max_columns_per_device=sku.lm_head_max_columns_per_device,
        dtype=lm_head_dtype,
        cache_dir=cache_path / "lm_head" if cache_path else None,
    )
    lm_head_core_grid = sku.lm_head_core_grid or _dram_shard_core_grid(qcfg.dim)
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
    common = LMHead1DConfig(
        output_weights=lm_splits,
        mesh_device=mesh_device,
        dim=qcfg.dim,
        max_batch_size=qcfg.max_batch_size,
        lm_head_dtype=lm_head_dtype,
        program_configs=lm_prog_configs,
        output_split_sizes=lm_split_sizes,
        input_memcfg=lm_input_memcfg,
        weights_memcfgs=lm_weights_memcfgs,
    )
    return LMHead1D.from_config(_qwen3_lm_head_config(common))


class Qwen3_32BDecoderLayer(LightweightModule):
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


class Qwen3_32B(LightweightModule):
    """
    Full decoder for Qwen3-32B (TTTv2 modules only) on WH T3K or BH P150x4.

    Prefill/decode on **embedded** activations match the model-owned executor surface. Token embedding
    is ``embed_prefill`` / ``embed_decode``. Bind KV with ``set_kv_cache`` before first forward.
    """

    decode_residual_memcfg = ttnn.DRAM_MEMORY_CONFIG

    def __init__(
        self,
        cfg: Qwen3_32BConfig,
        embed: Embedding1D,
        rope_setup: RotarySetup1D,
        layers: List[Qwen3_32BDecoderLayer],
        norm: RMSNorm1D,
        lm_head: LMHead1D,
        mesh_device: ttnn.MeshDevice,
    ):
        super().__init__()
        self.cfg = cfg
        self.config = cfg
        self.config.mesh_device = mesh_device
        self.embed = embed
        self.embedding = self.embed
        self.rope_setup = rope_setup
        self.layers = layers
        self.norm = norm
        self.lm_head = lm_head
        self.mesh_device = mesh_device
        self.model_args: Qwen3_32BExecutorRuntimeConfig | None = None

        self.vocab_size = cfg.vocab_size
        self.n_layers = cfg.num_hidden_layers
        self.num_devices = mesh_device.get_num_devices()
        self.tt_ccl = get_tt_ccl(mesh_device) if self.num_devices > 1 else None
        self.config.num_devices = self.num_devices
        self.config.n_layers = cfg.num_hidden_layers
        self.config.block_configs = [
            type("_BlockConfig", (), {"attention_config": layer.self_attn.config})() for layer in self.layers
        ]
        # Same padded width the LM head uses (weight_utils is the single source of truth). The
        # sampler runs per-device top-k on the LM head's tile-aligned shards, so it MUST share this
        # width or its index_offsets (device_id * vocab // num_devices) miss the real shard boundary.
        self.padded_vocab_size = weight_utils.lm_head_padded_vocab_size(self.vocab_size, self.num_devices)

        # The model owns its sampler (replacing the self.sampling = None placeholder); callers pick
        # behavior per request via sampling_params. Buffers are lazy (nothing materializes until the
        # first on-device sampled decode), so this is harmless when sampling_params is None (the
        # host-argmax path, which stays the demo default). Both supported SKUs
        # are multi-device, so build the sampler unconditionally.
        self.supports_on_device_sampling = True
        self.sampling = Sampling1D(
            # Padded width for the per-device top-k offset math; valid_vocab_size carries the
            # real tokenizer vocab so #47021's mask zeroes out the trailing pad logits.
            vocab_size=self.padded_vocab_size,
            valid_vocab_size=self.vocab_size,
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            max_batch_size=_nearest_32(cfg.max_batch_size),
            # Qwen3's valid vocabulary is tile-aligned, so greedy argmax can slice
            # away the padded tail instead of invoking the unsupported tail mask.
            allow_force_argmax=True,
            pad_to_power_of_2=True,
        )

    @property
    def n_kv_heads(self) -> int:
        return self.cfg.n_kv_heads

    @classmethod
    def from_pretrained(
        cls,
        mesh_device: ttnn.MeshDevice,
        hf_model_id: str = "Qwen/Qwen3-32B",
        *,
        revision: str | None = DEFAULT_HF_REVISION,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        num_layers: int | None = None,
        cache_dir: Path | str | None = None,
        precision: Qwen3_32BPrecisionConfig = QWEN3_32B_ACCURACY,
        block_size: int = 32,
        executor_mode: bool = False,
        disable_batched_prefill: bool | None = None,
    ) -> Qwen3_32B:
        """
        Load HF weights on host and build TTNN modules (weights materialize on first forward).

        Args:
            mesh_device: Open Wormhole T3K ``(1, 8)`` or BlackHole P150x4 ``(1, 4)`` mesh.
            hf_model_id: Hugging Face hub id.
            revision: HF revision SHA (default pins to ``DEFAULT_HF_REVISION``).
            max_batch_size: Decode batch / KV allocation (tile-padded internally).
            max_seq_len: KV cache sequence budget (per layer).
            num_layers: If set, truncate stack for smoke tests.
            cache_dir: Optional directory for ``LazyWeight`` tensor caches.
            precision: Per-layer precision + math-fidelity recipe (see :class:`Qwen3_32BPrecisionConfig`).
                Defaults to :data:`QWEN3_32B_ACCURACY` (mirrors TTTv1 ``DecodersPrecision.accuracy``
                for Qwen3-32B). Use :data:`QWEN3_32B_PERFORMANCE` for TTTv1's perf recipe
                (BFP4 FF1/FF3 + LOFI), or ``dataclasses.replace(...)`` to customize a single field.
            block_size: Paged attention block size (tokens per block).
            executor_mode: If True, use external paged KV (``set_kv_cache`` + shared executor).
                If False, internal KV tensors (smoke / ``prefill_from_token_ids`` without executor).
        """
        ttnn.SetDefaultDevice(mesh_device)
        cache_path = Path(cache_dir) if cache_dir else None
        num_dev = mesh_device.get_num_devices()
        arch = mesh_device.arch()
        sku = _resolve_qwen3_32b_sku_overlay(
            arch=arch,
            cluster_type=ttnn.cluster.get_cluster_type(),
            num_dev=num_dev,
            mesh_device=mesh_device,
        )
        tt_ccl = get_tt_ccl(mesh_device)
        topology = sku.topology

        local_files_only = any(
            os.getenv(name, "").lower() in {"1", "true", "yes"}
            for name in ("CI", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        )
        hf_cfg = AutoConfig.from_pretrained(hf_model_id, revision=revision, local_files_only=local_files_only)
        n_heads_hf = hf_cfg.num_attention_heads
        n_kv_hf = hf_cfg.num_key_value_heads
        if n_heads_hf % num_dev != 0 or n_kv_hf % num_dev != 0:
            raise ValueError(
                f"This checkpoint requires num_attention_heads ({n_heads_hf}) and "
                f"num_key_value_heads ({n_kv_hf}) to each be divisible by the mesh device "
                f"count ({num_dev}) for Attention1D sharding."
            )
        torch_dtype = torch.bfloat16
        logger.info(f"Loading HF weights: {hf_model_id} (revision={revision})")
        hf = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            revision=revision,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )
        hf.eval()
        base = hf.model
        n_layers = num_layers if num_layers is not None else hf_cfg.num_hidden_layers
        dim = hf_cfg.hidden_size
        n_heads = hf_cfg.num_attention_heads
        n_kv = hf_cfg.num_key_value_heads
        # Qwen3 decouples head_dim from hidden_size (Qwen3-32B: head_dim=128, hidden/n_heads=80).
        # Always prefer the explicit config field; fall back to the Llama/Qwen2.5 coupled formula.
        head_dim = getattr(hf_cfg, "head_dim", None) or (dim // n_heads)
        inter = hf_cfg.intermediate_size
        if inter != QWEN3_32B_INTERMEDIATE_SIZE:
            raise ValueError(
                "Qwen3-32B requires the checked-in intermediate_size=25600 profile; "
                f"the loaded checkpoint reports {inter}"
            )
        vocab = hf_cfg.vocab_size
        rope_len = max(max_seq_len * 2, 8192)
        rope_len = (rope_len + 127) // 128 * 128

        blocks_per_user = (max_seq_len + block_size - 1) // block_size
        max_num_blocks = blocks_per_user * max_batch_size
        paged_cfg = (
            Qwen3_32BPagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)
            if executor_mode
            else None
        )

        qcfg = Qwen3_32BConfig(
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
            num_devices=num_dev,
            mesh_device=mesh_device,
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
                core_grid=sku.attention_decode_transformation_grid,
            )
        )

        layers: list[Qwen3_32BDecoderLayer] = [
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
                sku=sku,
            )
            for idx in range(n_layers)
        ]

        norm_lw = _lazy(
            weight_utils.rms_weight_torch(base.norm).to(torch_dtype),
            dtype=ttnn.bfloat16,
            cache=(cache_path / "norm", "final") if cache_path else None,
        )
        final_norm_common = RMSNorm1DConfig(
            weight=norm_lw,
            mesh_device=mesh_device,
            eps=hf_cfg.rms_norm_eps,
            max_batch_size=max_batch_size,
            tt_ccl=tt_ccl,
        )
        if sku.distributed_rmsnorm_min_dim_exclusive is not None:
            final_norm_common.prefill_distributed = dim > sku.distributed_rmsnorm_min_dim_exclusive
        final_norm = RMSNorm1D.from_config(_qwen3_rmsnorm_config(final_norm_common))

        lm = _build_lm_head(
            mesh_device=mesh_device,
            hf_lm_head=hf.lm_head,
            qcfg=qcfg,
            lm_head_dtype=precision.lm_head_dtype,
            cache_path=cache_path,
            sku=sku,
        )

        del hf

        model = cls(qcfg, emb, rope_setup, layers, final_norm, lm, mesh_device)
        if executor_mode:
            model.model_args = Qwen3_32BExecutorRuntimeConfig(
                n_layers=n_layers,
                n_kv_heads=n_kv,
                head_dim=head_dim,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                cluster_shape=list(mesh_device.shape),
                model_cache_path=cache_path,
                kv_cache_dtype=precision.kv_cache_dtype,
                # Match TTTv1's P150x4 policy. The cross-cardinality experiment in the
                # approval plan must pass before this construction-time guard is removed.
                disable_batched_prefill=(
                    sku.disable_batched_prefill if disable_batched_prefill is None else disable_batched_prefill
                )
                or bool(os.environ.get("DISABLE_BATCHED_PREFILL")),
                # A/B escape hatch: DISABLE_BATCHED_EXTRACT=1 forces the per-slot last-token extract
                # (one lm_head per user, bit-identical to the sequential path) instead of the default
                # gathered extract (one lm_head over the whole group).
                batched_prefill_batched_extract=not os.environ.get("DISABLE_BATCHED_EXTRACT"),
            )
        return model

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
        live_configs = tuple(layer.self_attn.config for layer in self.layers)
        for layer_index, config in enumerate(live_configs):
            if not getattr(config, "use_vllm_paged_kv_cache", False):
                raise RuntimeError("Cannot configure paged attention on a model built without executor_mode=True")
            if config.kv_cache is not None or getattr(self.layers[layer_index].self_attn, "kv_cache", None) is not None:
                raise RuntimeError(f"Model layer {layer_index} already has a bound KV cache")
        construction_configs = tuple(block.attention_config for block in getattr(self.config, "block_configs", ()))
        for config in tuple({id(item): item for item in (*construction_configs, *live_configs)}.values()):
            config.paged_attention_config = Qwen3_32BPagedAttentionConfig(
                block_size=block_size,
                max_num_blocks=max_num_blocks,
            )

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

    def set_kv_cache(self, kv_cache: list | None) -> None:
        if kv_cache is None:
            for layer in self.layers:
                layer.self_attn.config.kv_cache = None
                if hasattr(layer.self_attn, "kv_cache"):
                    layer.self_attn.kv_cache = None
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
            layer.self_attn.config.kv_cache = pair
            if hasattr(layer.self_attn, "kv_cache"):
                layer.self_attn.kv_cache = pair

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
        # returns the full hidden state (get_last_token == -1); the executor does per-slot last-token
        # extraction + norm/lm_head so those stages stay bit-identical to the single-user path.
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

        if get_last_token == -1:
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
                old = x_tile
                x_tile = ttnn.typecast(x_tile, ttnn.bfloat16)
                ttnn.deallocate(old)
            old = x_tile
            x_tile = ttnn.embedding(last_token_index, x_tile, layout=ttnn.TILE_LAYOUT)
            x_tile = ttnn.unsqueeze_to_4D(x_tile)
            ttnn.deallocate(old)
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(self.mesh_device),
        )
        x = ttnn.matmul(selector, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(selector)
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
        lm_head_memcfg = self.lm_head.config.input_memcfg
        if lm_head_memcfg is not None and lm_head_memcfg.is_sharded() and x.memory_config() != lm_head_memcfg:
            x = ttnn.reshard(x, lm_head_memcfg)
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
                topology=_qwen3_ccl_topology(self.mesh_device),
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
