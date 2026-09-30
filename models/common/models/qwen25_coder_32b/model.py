# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen2.5-Coder-32B-Instruct — native TTTv2 stack (``Embedding1D``, ``RMSNorm1D``,
``Attention1D``, ``MLP1D``, ``RotarySetup1D``, ``LMHead1D``). Targets T3K (mesh ``(1, 8)``).

Tensor layout contracts:
  - **Prefill** hidden states: ``[1, 1, S, dim]`` TILE, ``S % 128 == 0``.
  - **Decode** hidden states: ``[1, 1, B, dim]`` TILE (``B`` padded to tile in modules).

Executor contract (``EagerLLMExecutor`` / ``TracedLLMExecutor``): pre-embedded forwards,
``set_kv_cache``, ``rope_setup``, ``page_table`` through attention, ``model_args`` holds a
:class:`Qwen25Coder32BExecutorRuntimeConfig` (not v1 ``ModelArgs``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, List

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.models.qwen25_coder_32b import weight_utils
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

# Pinned HF revision SHA for Qwen/Qwen2.5-Coder-32B-Instruct (resolved 2026-05-19).
DEFAULT_HF_REVISION = "381fc969f78efac66bc87ff7ddeadb7e73c218a7"


def _lazy(
    tensor: torch.Tensor,
    *,
    dtype: ttnn.DataType,
    cache: tuple[Path, str] | None,
) -> LazyWeight:
    """Minimal LazyWeight; ``Attention1D`` / ``MLP1D`` / ``Embedding1D`` resolvers fill mesh + memory."""
    return LazyWeight(source=tensor, dtype=dtype, cache_dir_weight_name=cache)


@dataclass
class Qwen25Coder32BPagedAttentionConfig:
    """Paged KV layout for ``Attention1D`` (``block_size`` / ``max_num_blocks`` only)."""

    block_size: int
    max_num_blocks: int


@dataclass
class Qwen25Coder32BExecutorRuntimeConfig:
    """Engine-facing runtime knobs. Exposed as ``model.model_args`` for shared ``EagerLLMExecutor``."""

    n_layers: int
    n_kv_heads: int
    head_dim: int
    max_batch_size: int
    max_seq_len: int
    cluster_shape: list[int]
    max_prefill_chunk_size: int = 4096
    model_cache_path: Path | None = None
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    optimizations: Any = None
    # Batched prefill (parity caveat #12): fuse equal-length users into batched passes to close the
    # batch-32 TTFT gap. ``supports_batched_prefill`` is the per-model opt-in (the shared engine only
    # batches models whose prefill_forward threads ``batch_size`` — Qwen2.5-Coder-32B does, below).
    # Qwen2.5-Coder is a standard dense Qwen2.5 attention (NO QK-norm), so every prefill op is
    # row-independent and the batched fold is bit-safe (same as the qwen25_7b port).
    # ``max_prefill_batch_size`` is the largest supported padded wave; 32 folds batch-32 prefill in ONE
    # 32-user pass (TTTv1 structural parity) so the eager norm+lm_head tail + full-vocab readback run
    # once instead of 4×. At the natural 128 bucket the fold is 32*128=4096=2*2048, an exact multiple of
    # MAX_QKV_MM_SEQ_LEN (reshape-safe), and 4096 % mlp_prefill_len_cutoff(1024) == 0 for the FF reshape;
    # the DRAM guard (padded_batch*seq < 128K) passes with 4096. This model already runs the 1024 engine
    # default for the FF cutoff (_resolve_coder_32b_wh_tuning), so the fold is the only prefill lever
    # applied here. ``disable_batched_prefill`` is the escape hatch back to the sequential loop;
    # ``max_prefill_chunk_size`` (above) drives the #45234 chunked-prompt decline.
    supports_batched_prefill: bool = True
    max_prefill_batch_size: int = 32
    disable_batched_prefill: bool = False
    # When True (default), batched prefill runs norm+lm_head ONCE per group over the gathered last-token
    # rows (TTTv1 parity); False falls back to the bit-identical per-slot path (one lm_head per user).
    batched_prefill_batched_extract: bool = True

    def can_enable_trace(self, prefill_seq_len: int, num_cached_tokens: int) -> bool:
        # Mirror TTTv1's ModelArgs.get_trace_prefill_supported_seq_lens: on the only supported SKU
        # (T3K, 8 devices) the device default is [128, 1024]. Prefill compiles, captures and replays
        # for these seq lens on hardware.
        #
        # Gate tracing to COLD prefill (num_cached_tokens == 0). The shared traced-prefill path does
        # not thread the cached start position / chunk_start into the trace body, so a prefix-cached
        # request (num_cached_tokens > 0) that pads to a trace-eligible length would replay with
        # start_pos=0 -> wrong RoPE indices and KV written at cache offset 0 (silently wrong output).
        # The eager prefix-cache path offsets correctly, so fall through to it; full traced
        # prefix-cache prefill is the upstream work item (issue #32056).
        if num_cached_tokens != 0:
            return False
        num_devices = int(self.cluster_shape[0]) * int(self.cluster_shape[1])
        allowed = {1: (128,), 2: (128, 1024), 8: (128, 1024)}.get(num_devices, (128,))
        return (
            prefill_seq_len in allowed
            and prefill_seq_len <= self.max_prefill_chunk_size
            and prefill_seq_len <= self.max_seq_len
        )


@dataclass
class Qwen25Coder32BConfig:
    """Resolved hyper-parameters for a loaded HF Qwen2.5-Coder-32B checkpoint."""

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


@dataclass(frozen=True)
class Qwen25Coder32BLayerWeights:
    wqkv: torch.Tensor
    wo: torch.Tensor
    wqkv_bias: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w3: torch.Tensor
    attention_norm: torch.Tensor
    ff_norm: torch.Tensor


@dataclass(frozen=True)
class Qwen25Coder32BWeights:
    embedding: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    layers: tuple[Qwen25Coder32BLayerWeights, ...]
    final_norm: torch.Tensor
    lm_head: torch.Tensor


_QWEN_ATTN_HIFI4_FP32_KERNEL = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
"""HiFi4 + fp32 dest acc for Qwen2.5 attention matmuls (LI_QKV, LI_O, SDPA).

TTTv1 ``DecodersPrecision.accuracy("Qwen2.5-Coder-32B-Instruct")`` resolves to the
non-Llama / non-Mistral branch in ``model_config.py:160-177`` which forces all attention
ops to ``HIFI4``. The TTTv2 ``Attention1D`` default is ``HIFI2`` with fp16 accumulation;
without this override, attention QKV / WO / SDPA produce a broad per-layer drift vs HF
(same regression debugged on the Qwen2.5-7B port). Used in ``QWEN25_CODER_32B_ACCURACY``.
"""


_LOFI_COMPUTE_KERNEL_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)
"""LoFi + packer L1 acc for the MLP FF1/FF3 matmuls in performance mode.

Mirrors TTTv1 ``DecodersPrecision.performance("Qwen2.5-Coder-32B-Instruct")``: the
non-Qwen2.5-7B branch at ``model_config.py:208-218`` sets ``FF1_FF3 → BFP4`` and
``LI_FF1_FF3 → LOFI``. This single delta is the bulk of the perf-mode throughput uplift.
"""


_TTTV1_HIFI2_COMPUTE_KERNEL_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
"""TTTv1's ``compute_kernel_config_hifi2`` for HIFI2 attention ops in performance mode.

TTTv2's ``Attention1D`` default for HIFI2 ops is ``compute_kernel_hifi2_fp16``
(``math_approx=False``, ``fp32_dest_acc=False``) — that matches TTTv1's HIFI2_FP16
setting, **not** TTTv1's plain HIFI2 (``math_approx=True``, ``fp32_dest_acc=True``).
For Qwen2.5-Coder-32B perf mode, the TTTv1 OpFidelity defaults
(``model_config.py:292-297``) resolve attention decode kernels to plain HIFI2, so the
perf recipe explicitly pins this kernel instead of falling back to TTTv2's fp16 variant.
"""


@dataclass(frozen=True)
class Qwen25Coder32BPrecisionConfig:
    """Per-layer precision + math-fidelity recipe for Qwen2.5-Coder-32B-Instruct on T3K.

    Mirrors the fields TTTv1's ``DecodersPrecision`` distinguishes for Qwen2.5-Coder-32B.
    The base model name resolves to ``Qwen2.5-Coder-32B`` (via ``common.get_base_model_name``),
    which falls into the standard Qwen2.5-family branch — **not** the Qwen2.5-7B / Qwen2.5-VL-7B
    special case in ``model_config.py:187``. As a result:

      * **Accuracy** (``model_config.py:160-177``): BF16 ``WQKV`` / ``WO`` + ``HIFI4`` on every
        ``LI_QKV`` / ``SDPA`` / ``LI_O``. FF and LM head stay at engine defaults (BFP8 FF +
        ``HIFI2_FP16``). **KV cache is BFP8, not BF16** (diverges from TTTv1) — BF16 KV deadlocks
        the traced on-device-topk decode replay at full depth; BFP8 clears it and is loss-free
        here (see ``kv_cache_dtype`` below).
      * **Performance** (``model_config.py:208-218``, non-7B branch): only ``FF1_FF3 → BFP4``
        and ``LI_FF1_FF3 → LOFI``. Everything else reverts to TTTv1 defaults
        (BFP8 attention + ``HIFI2`` attention kernels + BFP8 KV cache).

    Two module-level recipes are exposed: :data:`QWEN25_CODER_32B_ACCURACY` (default) and
    :data:`QWEN25_CODER_32B_PERFORMANCE`. Pass one to :meth:`Qwen25Coder32B.from_pretrained`
    via ``precision=``; use ``dataclasses.replace(QWEN25_CODER_32B_ACCURACY, ...)`` to
    customize a single field. Defaults below mirror the accuracy recipe so ``Qwen25Coder32BPrecisionConfig()``
    is equivalent to :data:`QWEN25_CODER_32B_ACCURACY`.

    The ``mlp_w2_dtype`` / ``mlp_ff2_compute_kernel_cfg`` fields are absent because TTTv1 leaves
    them at engine defaults (``BFP8`` / ``HIFI2_FP16``) in *both* recipes for this model, and
    those defaults coincide with the TTTv2 ``MLP1D`` defaults.
    """

    # Attention weight dtypes: accuracy uses BF16 (default), performance overrides BFP8.
    wqkv_dtype: ttnn.DataType = ttnn.bfloat16
    wo_dtype: ttnn.DataType = ttnn.bfloat16
    # KV cache: BFP8 in BOTH recipes. TTTv1's accuracy config nominally uses BF16 KV; BFP8 is kept
    # here because it is loss-free for this model (teacher-forcing top1 98.6% / top5 100.0%, >= the
    # 95/99 targets -- the accuracy recipe's precision comes from BF16 attention weights + HIFI4 +
    # fp32 dest acc, not the KV dtype) AND it avoids BF16's doubled per-layer KV read traffic in the
    # memory-bound decode step. NOTE: an earlier bringup switched BF16 -> BFP8 KV to work around a
    # batch-1·accuracy·on_device_topk decode-trace-replay hang, attributing it to BF16 KV's larger
    # footprint tripping a "cumulative traffic ceiling". That was a layout band-aid: the true root
    # cause was on-device sampling buffers being allocated while a trace was already active and then
    # clobbered on replay, fixed generically in executor.py
    # (TracedLLMExecutor._prealloc_sampling_buffers, called before the prefill trace is captured).
    # BFP8 KV is now retained on its own merits (loss-free + less decode traffic), not for the hang.
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat8_b

    # MLP FF1/FF3 weight dtype. Accuracy keeps BFP8 default; performance overrides BFP4.
    mlp_w1_w3_dtype: ttnn.DataType = ttnn.bfloat8_b

    # MLP FF1/FF3 compute kernel. ``None`` → MLP1D default (HIFI2_FP16); performance overrides LOFI.
    mlp_ff1_3_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None

    # Attention compute kernels. Accuracy sets HIFI4 + fp32 dest acc on every stage; performance
    # leaves them at the Attention1D default (HIFI2 fp16 dest acc).
    attn_li_qkv_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = _QWEN_ATTN_HIFI4_FP32_KERNEL
    attn_sdpa_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = _QWEN_ATTN_HIFI4_FP32_KERNEL
    attn_li_o_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = _QWEN_ATTN_HIFI4_FP32_KERNEL

    # Not in TTTv1 DecodersPrecision; TTTv2 accuracy tightens to bf16 to lock in top-1 (matches the
    # Mistral-7B port). Performance keeps BFP8.
    lm_head_dtype: ttnn.DataType = ttnn.bfloat16


# TTTv1 ``DecodersPrecision.accuracy("Qwen2.5-Coder-32B-Instruct")`` (``model_config.py:160-177``):
# BF16 attention weights + HIFI4 + fp32_dest_acc on every attention stage. FF and LM head sit at
# TTTv2 defaults (BFP8 + HIFI2_FP16); LM head tightens to bf16 (TTTv2 addition). KV cache is BFP8
# (NOT TTTv1's BF16) to avoid the full-depth decode-trace-replay deadlock — see kv_cache_dtype above.
QWEN25_CODER_32B_ACCURACY = Qwen25Coder32BPrecisionConfig()

# TTTv1 ``DecodersPrecision.performance("Qwen2.5-Coder-32B-Instruct")`` (``model_config.py:208-218``,
# non-7B branch): FF1_FF3 → BFP4 and LI_FF1_FF3 → LOFI; everything else reverts to engine defaults
# (BFP8 attention, HIFI2 attention kernels, BFP8 KV cache, BFP8 LM head).
QWEN25_CODER_32B_PERFORMANCE = Qwen25Coder32BPrecisionConfig(
    wqkv_dtype=ttnn.bfloat8_b,
    wo_dtype=ttnn.bfloat8_b,
    kv_cache_dtype=ttnn.bfloat8_b,
    mlp_w1_w3_dtype=ttnn.bfloat4_b,
    mlp_ff1_3_compute_kernel_cfg=_LOFI_COMPUTE_KERNEL_CFG,
    # Pin TTTv1's plain HIFI2 (math_approx=True, fp32_dest_acc=True). Without this, the
    # Attention1D default for the three attention ops is HIFI2_FP16, which mismatches
    # TTTv1's compute_kernel_config_hifi2 (model_config.py:728-733).
    attn_li_qkv_kernel_cfg=_TTTV1_HIFI2_COMPUTE_KERNEL_CFG,
    attn_sdpa_kernel_cfg=_TTTV1_HIFI2_COMPUTE_KERNEL_CFG,
    attn_li_o_kernel_cfg=_TTTV1_HIFI2_COMPUTE_KERNEL_CFG,
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
    corrupts decode activations (observed on the 7B port — same shape pattern on T3K Coder-32B).
    """
    padded_hidden = get_padded_hidden_dim(hidden_dim, num_devices, TILE_SIZE)
    grid = _dram_shard_core_grid_k_n(dim, padded_hidden // num_devices)
    tile_padded_batch_rows = TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE)
    program_config = _create_sharded_norm_program_config(dim, grid, tile_padded_batch_rows, TILE_SIZE)
    return program_config, mlp.config.decode_input_memcfg


# Cast coder's two per-layer prefill norm-reconstruction all-gathers (pre-attn + post-attn) from bf16 to
# bfloat8_b, mirroring the sibling qwen25_72b H003 (commit 783d2945e2b, model.py:262). Every coder prefill
# decoder layer runs two full-activation all-gathers at bf16 to reconstruct the dim-fractured RMSNorm output
# to full ``dim`` before Attention1D / MLP1D (the QKV / W1/W3 matmuls expect width ``dim``). Each moves a
# 5120-wide activation across 8 devices, x2/layer x64 layers = 128 bf16 all-gathers on the batched b32-ci
# prefill critical path. Every matmul these gathers feed already quantizes activations to bf8_b (QKV/WO) or
# bf4_b (W1/W3), so the bf16 gather precision is largely unconsumed; casting the gather input to bf8_b halves
# each collective's cross-device payload (AllGatherAsync is payload-bound at dim=5120). PREFILL-ONLY: the
# decode + tail-norm call sites of _all_gather_rmsnorm_tensor pass no dtype -> byte-identical decode/tail.
# Unlike 72b there is no third fused pre-WO gather to cast: coder's dim=5120 fails the fused all-gather auto-
# gate ((5120//32//8) % 8 = 4 != 0), so only these two norm gathers apply (the fused hook is a no-op here).
# NOT bit-exact vs bf16 (mantissa drop) -> gated on token-accuracy + eval-32, NOT the 32-user byte-compare
# (INVALID on coder's tie-heavy on_device_topk). coder's perf recipe is BFP4-FF/LoFi (less precision-tolerant
# than 72b's >70B bf4/bf8 recipe), so the accuracy profile is re-gated separately. A/B escape hatch: set
# DISABLE_PREFILL_AG_BF8=1 to keep the gathers at bf16 (byte-identical to the pre-change HEAD). None disables.
_PREFILL_AG_CCL_DTYPE: ttnn.DataType | None = (
    None if os.environ.get("DISABLE_PREFILL_AG_BF8") == "1" else ttnn.bfloat8_b
)


def _all_gather_rmsnorm_tensor(
    norm: RMSNorm1D,
    x: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig | None = None,
    dtype: ttnn.DataType | None = None,
) -> ttnn.Tensor:
    cfg = norm.config
    if cfg.mesh_device.get_num_devices() == 1 or x.shape[-1] == cfg.weight.source.numel():
        if memory_config is not None:
            return ttnn.to_memory_config(x, memory_config)
        return x

    if memory_config is None:
        memory_config = x.memory_config()

    # Prefill-only opt-in (``dtype`` is set only at the two per-layer prefill call sites): cast the gather
    # input (bf16) to a smaller CCL dtype (bfloat8_b) to halve this collective's cross-device payload. Decode
    # and tail-norm call sites leave ``dtype=None`` -> byte-identical. An explicit typecast is required
    # (to_memory_config does not cast an already-DRAM tensor).
    if dtype is not None and x.dtype != dtype:
        x_cast = ttnn.typecast(x, dtype)
        ttnn.deallocate(x)
        x = x_cast

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
class _Qwen25Coder32BWHTuning:
    """Wormhole-specific L1 / cutoff tuning resolved at build time.

    Precision-vs-fidelity knobs live on :class:`Qwen25Coder32BPrecisionConfig`. This
    dataclass only carries non-mode-dependent L1 footprint controls: MLP prefill cutoff
    and the optional W1→DRAM spill on decode.
    """

    mlp_prefill_len_cutoff: int | None = None
    mlp_decode_spill_w1_to_dram: bool = False
    # Use ttnn.experimental.minimal_matmul for the QKV + W2 prefill matmuls above seq_len > 128 (TTTv1
    # parity, PLAN_01). A/B escape hatch: set DISABLE_MINIMAL_MATMUL=1 to force ttnn.linear. On a 32B the
    # batch-32-ci prefill is matmul-compute-bound (~80% of FLOPs = the 3 MLP matmuls), so minimal_matmul
    # is a real prefill-TTFT win — it narrows the batch-32-ci TTFT gap vs TTTv1 (which uses minimal_matmul
    # for the same matmuls). The shared plumbing (attention_1d.use_minimal_qkv_matmul / mlp_1d
    # use_minimal_w2_matmul, both gated seq_len>128) is already in the base; this flag just engages it.
    # Coder is a Qwen2.5 arch: the QKV bias is added AFTER the matmul, so it is unchanged by the minimal
    # path (same as the mistral_7b / deepseek Qwen2 ports). Independent of the FF-hidden DRAM-shard pad
    # (a decode fix); minimal_matmul is prefill-only, so decode throughput is unchanged.
    prefill_minimal_matmul: bool = True
    # Cast coder's per-layer prefill attention output reduce-scatter from bf16 to bfloat8_b, matching
    # TTTv1's ccl_dtype (model_config.py:1070). coder's dim=5120 fails the fused all-gather auto-gate
    # ((5120//32//8) % 8 = 4 != 0), so attention runs a SEPARATE per-layer reduce_scatter on the bf16 WO
    # output (attention_1d._all_reduce_output_prefill), x64 layers; a bf16 reduce moves 2x the cross-device
    # bytes of a bf8_b one and reduce_scatter on this shape is dtype-bound (PLAN_03: 2210us -> 1534us/layer
    # at dim=5120). TTTv1 already reduces this at bf8_b, so the result matches the shipped reference numerics
    # (not a degradation below it). PREFILL-ONLY: the decode reduce (_all_reduce_output_decode) never reads
    # prefill_reduce_ccl_dtype -> decode byte-identical. A/B escape hatch: set DISABLE_PREFILL_REDUCE_BF8=1
    # to keep the reduce at bf16 (byte-identical to the pre-change HEAD). Not bit-exact vs bf16 (mantissa
    # drop) -> gated on token-accuracy + eval-32, not the 32-user byte-compare (INVALID on coder's tie-heavy
    # on_device_topk).
    prefill_reduce_ccl_bf8: bool = True


def _resolve_qwen_coder_wh_tuning(*, num_dev: int, max_batch_size: int) -> _Qwen25Coder32BWHTuning:
    """Pick WH L1 tuning knobs for Qwen2.5-Coder-32B-Instruct on T3K.

    Use TTTv1's Wormhole default (``prefill_len_cutoff=1024`` at ``model_config.py:516``).
    Coder-32B is not in the "reduce to 512" override list (``model_config.py:583-589``), so
    TTTv1 keeps 1024 for this model on T3K. ``mlp_decode_spill_w1_to_dram`` is off on T3K
    because per-device FF shards (5120×3456 per chip) are smaller than 7B-on-N300.
    """
    t = _Qwen25Coder32BWHTuning(
        mlp_prefill_len_cutoff=1024,
        mlp_decode_spill_w1_to_dram=False,
    )
    t.prefill_minimal_matmul = not os.environ.get("DISABLE_MINIMAL_MATMUL")
    t.prefill_reduce_ccl_bf8 = not os.environ.get("DISABLE_PREFILL_REDUCE_BF8")
    logger.info(
        f"L1 tuning for Qwen2.5-Coder-32B on {num_dev} device(s): "
        f"prefill_len_cutoff={t.mlp_prefill_len_cutoff}, "
        f"decode_spill_w1_to_dram={t.mlp_decode_spill_w1_to_dram}, "
        f"prefill_minimal_matmul={t.prefill_minimal_matmul}, "
        f"prefill_reduce_ccl_bf8={t.prefill_reduce_ccl_bf8}"
    )
    return t


def _build_decoder_layer(
    *,
    idx: int,
    weights: Qwen25Coder32BLayerWeights,
    qcfg: Qwen25Coder32BConfig,
    mesh_device: ttnn.MeshDevice,
    tt_ccl: Any,
    topology: Any,
    num_dev: int,
    precision: Qwen25Coder32BPrecisionConfig,
    executor_mode: bool,
    paged_cfg: Qwen25Coder32BPagedAttentionConfig | None,
    cache_path: Path | None,
    wh: _Qwen25Coder32BWHTuning,
) -> Qwen25Coder32BDecoderLayer:
    """Construct one decoder layer (attention + MLP + the two RMSNorms) from an HF layer."""
    prefix = f"layer{idx}"

    lazy_wqkv = _lazy(
        weights.wqkv, dtype=precision.wqkv_dtype, cache=(cache_path / "attn", f"{prefix}_wqkv") if cache_path else None
    )
    lazy_wo = _lazy(
        weights.wo, dtype=precision.wo_dtype, cache=(cache_path / "attn", f"{prefix}_wo") if cache_path else None
    )

    def _qk_norm_cfg(weight: torch.Tensor | None, name: str) -> RMSNorm1DConfig | None:
        if weight is None:
            return None
        lw = _lazy(
            weight.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
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
            source=weights.wqkv_bias.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            cache_dir_weight_name=(cache_path / "attn", f"{prefix}_bias") if cache_path else None,
        )
        if weights.wqkv_bias is not None
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
            q_norm_config=_qk_norm_cfg(None, "qn"),
            k_norm_config=_qk_norm_cfg(None, "kn"),
            wqkv_bias=bias_lw,
            use_vllm_paged_kv_cache=executor_mode,
            paged_attention_config=paged_cfg,
            kv_cache=None,
            kv_cache_dtype=precision.kv_cache_dtype,
            li_qkv_prefill_compute_kernel_cfg=precision.attn_li_qkv_kernel_cfg,
            li_qkv_decode_compute_kernel_cfg=precision.attn_li_qkv_kernel_cfg,
            sdpa_decode_compute_kernel_cfg=precision.attn_sdpa_kernel_cfg,
            li_o_prefill_compute_kernel_cfg=precision.attn_li_o_kernel_cfg,
            li_o_decode_compute_kernel_cfg=precision.attn_li_o_kernel_cfg,
            prefill_qkv_minimal_matmul=wh.prefill_minimal_matmul,
            # Route the folded-batch prefill WO projection through minimal_matmul (completes PLAN_01's
            # QKV+FF2 minimal plumbing on coder). WO is the least-efficient prefill matmul on ttnn.linear;
            # the minimal op (already used for QKV/FF2) recovers most of that gap and makes TTTv2 beat
            # TTTv1's ttnn.linear WO (a shared op on both stacks). coder's WO is the non-fused shape
            # (dim=5120 fails the fused all-gather auto-gate), so this runs every layer. Gated by the
            # same DISABLE_MINIMAL_MATMUL escape hatch (OFF => byte-identical to the pre-change HEAD).
            prefill_wo_minimal_matmul=wh.prefill_minimal_matmul,
            # Cast the per-layer prefill attention output reduce-scatter (bf16 WO output) to bfloat8_b,
            # matching TTTv1's ccl_dtype. coder is on the NON-fused path (dim=5120 fails the fused
            # all-gather auto-gate), so this reduce runs every layer; a bf8_b reduce halves the
            # cross-device payload. Prefill-only (the decode reduce ignores this field). Gated by the
            # DISABLE_PREFILL_REDUCE_BF8 escape hatch (None => byte-identical to the pre-change HEAD).
            prefill_reduce_ccl_dtype=(ttnn.bfloat8_b if wh.prefill_reduce_ccl_bf8 else None),
        )
    )

    w1, w2, w3 = weights.w1, weights.w2, weights.w3
    # Pad the FF hidden dim to a grid-friendly per-device size so the DRAM-sharded decode FF
    # matmuls (W1/W3/W2) use a full multi-core grid instead of 4. The decode grid must divide both
    # the K-tile and N-tile counts (in0 is K-width-sharded on dim, weights are N-sharded on hidden);
    # with the raw per-device hidden 27648/8 -> 3456 = 108 tiles (2^2*27), gcd(dim_tiles=160, 108)=4
    # -> only 4 DRAM readers stream the FF weights, which dominate memory-bound decode. Padding per
    # device to 128 tiles (4096, total 32768) gives gcd(160, 128)=32 cores. The extra columns are
    # zeros (silu(0)=0, mul->0, and W2 contracts the padded rows to 0), so decode output is unchanged.
    _ff_align = TILE_SIZE * TILE_SIZE * num_dev  # 32*32*8 = 8192 on T3K
    _ff_pad = math.ceil(w1.shape[-1] / _ff_align) * _ff_align
    if _ff_pad != w1.shape[-1]:
        w1 = torch.nn.functional.pad(w1, (0, _ff_pad - w1.shape[-1]))
        w3 = torch.nn.functional.pad(w3, (0, _ff_pad - w3.shape[-1]))
        w2 = torch.nn.functional.pad(w2, (0, 0, 0, _ff_pad - w2.shape[-2]))
    mlp = MLP1D.from_config(
        MLP1DConfig(
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
            max_batch_size=qcfg.max_batch_size,
            prefill_len_cutoff=wh.mlp_prefill_len_cutoff,
            w1_w3_dtype=precision.mlp_w1_w3_dtype,
            w2_dtype=ttnn.bfloat8_b,
            ff1_3_compute_kernel_cfg=precision.mlp_ff1_3_compute_kernel_cfg,
            decode_ff1_3_compute_kernel_cfg=precision.mlp_ff1_3_compute_kernel_cfg,
            decode_spill_w1_to_dram_before_w3=wh.mlp_decode_spill_w1_to_dram,
            prefill_w2_minimal_matmul=wh.prefill_minimal_matmul,
        )
    )

    post_attn_decode_program_config, post_attn_decode_memory_config = _post_attn_norm_decode_configs(
        mlp,
        dim=qcfg.dim,
        # Use the padded FF hidden so the post-attn RMSNorm decode output is width-sharded on the
        # SAME (32-core) grid as MLP1D's W1/W3 decode input; a mismatch silently corrupts decode.
        hidden_dim=_ff_pad,
        num_devices=num_dev,
        max_batch_size=qcfg.max_batch_size,
    )

    def _build_norm(weight: torch.Tensor, name: str, **extra: Any) -> RMSNorm1D:
        lw = _lazy(
            weight.to(torch.bfloat16),
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

    return Qwen25Coder32BDecoderLayer(
        input_layernorm=_build_norm(weights.attention_norm, "pre_attn"),
        self_attn=attn,
        post_attention_layernorm=_build_norm(
            weights.ff_norm,
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
    qcfg: Qwen25Coder32BConfig,
    lm_head_dtype: ttnn.DataType,
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
            input_memcfg=lm_input_memcfg,
            weights_memcfgs=lm_weights_memcfgs,
        )
    )


def build_qwen25_coder_32b_model(
    *,
    mesh_device: ttnn.MeshDevice,
    config: Qwen25Coder32BConfig,
    weights: Qwen25Coder32BWeights,
    precision: Qwen25Coder32BPrecisionConfig,
    cache_path: Path | None,
    paged_attention_config: Qwen25Coder32BPagedAttentionConfig | None,
) -> Qwen25Coder32B:
    """Build the TT tensor graph from provider-neutral Qwen2.5-Coder-32B dimensions and tensors."""

    ttnn.SetDefaultDevice(mesh_device)
    num_devices = mesh_device.get_num_devices()
    if num_devices != 8:
        raise ValueError(
            f"Qwen2.5-Coder-32B-Instruct port targets T3K (mesh (1, 8) = 8 devices) only. "
            f"Got mesh_device with {num_devices} device(s). Open a T3K mesh with MESH_DEVICE=T3K."
        )
    if config.n_heads % num_devices != 0 or config.n_kv_heads % num_devices != 0:
        raise ValueError(
            f"Checkpoint heads ({config.n_heads}/{config.n_kv_heads}) must be divisible by "
            f"device count ({num_devices})"
        )
    if len(weights.layers) != config.num_hidden_layers:
        raise ValueError(f"Expected {config.num_hidden_layers} decoder layer weights, got {len(weights.layers)}")

    tt_ccl = get_tt_ccl(mesh_device)
    topology = default_topology(mesh_device)
    emb = Embedding1D.from_config(
        Embedding1DConfig(
            weights=_lazy(
                weights.embedding,
                dtype=ttnn.bfloat16,
                cache=(cache_path / "embedding", "tok_embeddings") if cache_path else None,
            ),
            mesh_device=mesh_device,
            embed_scale=1.0,
        )
    )
    rope_setup = RotarySetup1D.from_config(
        Rope1DConfig(
            cos_matrix=_lazy(
                weights.rope_cos, dtype=ttnn.bfloat16, cache=(cache_path / "rope", "cos") if cache_path else None
            ),
            sin_matrix=_lazy(
                weights.rope_sin, dtype=ttnn.bfloat16, cache=(cache_path / "rope", "sin") if cache_path else None
            ),
            max_batch_size=config.max_batch_size,
            head_dim=config.head_dim,
            device=mesh_device,
            use_qk_fused=False,
        )
    )

    wh = _resolve_qwen_coder_wh_tuning(num_dev=num_devices, max_batch_size=config.max_batch_size)
    layers = [
        _build_decoder_layer(
            idx=idx,
            weights=weights.layers[idx],
            qcfg=config,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            topology=topology,
            num_dev=num_devices,
            precision=precision,
            executor_mode=paged_attention_config is not None,
            paged_cfg=paged_attention_config,
            cache_path=cache_path,
            wh=wh,
        )
        for idx in range(config.num_hidden_layers)
    ]
    final_norm = RMSNorm1D.from_config(
        RMSNorm1DConfig(
            weight=_lazy(
                weights.final_norm, dtype=ttnn.bfloat16, cache=(cache_path / "norm", "final") if cache_path else None
            ),
            mesh_device=mesh_device,
            eps=config.rms_norm_eps,
            max_batch_size=config.max_batch_size,
            tt_ccl=tt_ccl,
        )
    )
    lm_head = _build_lm_head(
        mesh_device=mesh_device,
        hf_lm_head=type("_LMHeadWeight", (), {"weight": weights.lm_head})(),
        qcfg=config,
        lm_head_dtype=precision.lm_head_dtype,
        cache_path=cache_path,
    )
    return Qwen25Coder32B(config, emb, rope_setup, layers, final_norm, lm_head, mesh_device)


class Qwen25Coder32BDecoderLayer(LightweightModule):
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
        r = _all_gather_rmsnorm_tensor(self.input_layernorm, r, dtype=_PREFILL_AG_CCL_DTYPE)
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
        r2 = _all_gather_rmsnorm_tensor(self.post_attention_layernorm, r2, dtype=_PREFILL_AG_CCL_DTYPE)
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


class Qwen25Coder32B(LightweightModule):
    """
    Full decoder for Qwen2.5-Coder-32B-Instruct (TTTv2 modules only) on T3K.

    Prefill/decode on **embedded** activations match ``EagerLLMExecutor``. Token embedding
    is ``embed_prefill`` / ``embed_decode``. Bind KV with ``set_kv_cache`` before first forward.
    """

    decode_residual_memcfg = ttnn.DRAM_MEMORY_CONFIG

    def __init__(
        self,
        cfg: Qwen25Coder32BConfig,
        embed: Embedding1D,
        rope_setup: RotarySetup1D,
        layers: List[Qwen25Coder32BDecoderLayer],
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
        self.model_args: Qwen25Coder32BExecutorRuntimeConfig | None = None

        self.vocab_size = cfg.vocab_size
        self.n_layers = cfg.num_hidden_layers
        self.num_devices = mesh_device.get_num_devices()
        self.tt_ccl = get_tt_ccl(mesh_device) if self.num_devices > 1 else None
        self.config.block_configs = [
            type("_BlockConfig", (), {"attention_config": layer.self_attn.config})() for layer in self.layers
        ]

        # On-device sampling. The model owns its sampler; callers only pick behavior per request
        # via ``sampling_params`` (the executor routes greedy/argmax vs the top-k op path). Buffers
        # are lazy -- nothing materializes until the first on-device sampled decode -- so this is
        # harmless when ``sampling_params is None`` (the host-argmax path, which stays the demo
        # default). Qwen2.5-Coder-32B is a T3K-only port (8 devices), where Sampling1D's all-gather
        # uses a barrier-free Ring and is trace-capture-safe.
        #
        # Coder's LM head exposes 128 padded rows, beyond the top-k sampling
        # kernel's 32-user limit. Route greedy requests through force-argmax.
        self.supports_on_device_sampling = self.num_devices >= 1
        self.sampling = (
            Sampling1D(
                vocab_size=self.vocab_size,
                mesh_device=mesh_device,
                tt_ccl=self.tt_ccl,
                max_batch_size=_nearest_32(cfg.max_batch_size),
                allow_force_argmax=True,
                pad_to_power_of_2=True,
            )
            if self.supports_on_device_sampling
            else None
        )

    @property
    def n_kv_heads(self) -> int:
        return self.cfg.n_kv_heads

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
            if not config.use_vllm_paged_kv_cache or config.paged_attention_config is None:
                raise RuntimeError(f"Model layer {layer_index} is not configured for externally managed paged KV cache")
            if config.kv_cache is not None or getattr(self.layers[layer_index].self_attn, "kv_cache", None) is not None:
                raise RuntimeError(f"Model layer {layer_index} already has a bound KV cache")
        construction_configs = tuple(block.attention_config for block in getattr(self.config, "block_configs", ()))
        for config in tuple({id(item): item for item in (*construction_configs, *live_configs)}.values()):
            config.paged_attention_config = replace(
                config.paged_attention_config,
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

    @classmethod
    def from_pretrained(
        cls,
        mesh_device: ttnn.MeshDevice,
        hf_model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        *,
        revision: str | None = DEFAULT_HF_REVISION,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        num_layers: int | None = None,
        cache_dir: Path | str | None = None,
        precision: Qwen25Coder32BPrecisionConfig = QWEN25_CODER_32B_ACCURACY,
        block_size: int = 32,
        executor_mode: bool = False,
    ) -> Qwen25Coder32B:
        """Compatibility constructor; provider loading lives in ``hf_adaptor``."""

        from models.common.models.qwen25_coder_32b.hf_adaptor import from_pretrained

        optimizations: str | Qwen25Coder32BPrecisionConfig
        if precision == QWEN25_CODER_32B_PERFORMANCE:
            optimizations = "performance"
        elif precision == QWEN25_CODER_32B_ACCURACY:
            optimizations = "accuracy"
        else:
            optimizations = precision
        product = from_pretrained(
            mesh_device,
            hf_model=hf_model_id,
            hf_revision=revision,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=optimizations,
            n_layers=num_layers,
            paged_attention_config=(
                Qwen25Coder32BPagedAttentionConfig(
                    block_size=block_size,
                    max_num_blocks=((max_seq_len + block_size - 1) // block_size) * max_batch_size,
                )
                if executor_mode
                else None
            ),
            cache_dir=cache_dir,
        )
        return product.model

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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(self.mesh_device),
        )
        x = ttnn.matmul(selector, hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
