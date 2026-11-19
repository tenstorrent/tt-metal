# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure CPU planning arithmetic for the full48 TP4 shared KV pool.

The tensor constants come from doc/full_model/capacity_plan.json. The inherited
DRAM limit and shape-dependent runtime forecast are planning, not hardware proof.
This helper must not reduce the model's advertised 262144-token context.
"""

from __future__ import annotations

import operator
from dataclasses import asdict, dataclass

SUPPORTED_CONTEXT = 262144
PAGE_TOKENS = 32
ATTENTION_LAYERS = 12
INHERITED_ALLOCATABLE_BYTES = 34178731008
PLANNING_RUNTIME_RESERVE_BYTES = 2**31
HISTORICAL_INTERLEAVED_HEAD_STATIC_BYTES = 28289880064
LM_HEAD_DRAM_BANK_PADDING_DELTA_BYTES = 655360
KNOWN_STATIC_BYTES = HISTORICAL_INTERLEAVED_HEAD_STATIC_BYTES + LM_HEAD_DRAM_BANK_PADDING_DELTA_BYTES
FIXED_PRIVATE_NON_KV_BYTES = 34675200
PREFILL_OUTPUT_BYTES = 3981312  # One independent workspace, shared by all fixed slots.
PREFILL_CHECKPOINT_BYTES_PER_SLOT = 34576896
BYTES_PER_PAIR_BLOCK = 32768
BYTES_PER_FULL_STACK_BLOCK = ATTENTION_LAYERS * BYTES_PER_PAIR_BLOCK
PREFILL_TRACE_SHORT_CONTEXT_BYTES = 765526016
NONTRACE_RUNTIME_ALLOWANCE_BYTES = 2**28
SAMPLING_TRACE_ALLOWANCE_BYTES = 2 * 327680


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def _ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


def factored_workspace_bytes(batch):
    """Source-exact bank reservations; no owner/buffers for configured B1."""
    if batch <= 1:
        return 0
    buckets = (batch - 1).bit_length() + 1
    return batch * (3 * 163840 + 16384) + buckets * 163840 + 1024


def runtime_forecast(context, batch):
    """Measured-component planning forecast, never a validated physical limit.

    Short-context capture sizes come from full48 C384 prefill and four-layer
    C128 bucketed decode. QSA indexer dispatch unrolls ceil(C/32768) blocks.
    Unmeasured B>2 multiplies QSA attention and the whole prefill allowance
    by that factor conservatively. B2 uses actual full-context adapter evidence.
    B1 keeps its prior2GiB reserve, supported by actual C262144 capture evidence.
    """
    if not 1 <= context <= SUPPORTED_CONTEXT or not 1 <= batch <= 32:
        raise ValueError("Runtime forecast requires supported context and1..32 slots")
    if batch == 1:
        return dict(
            reserve_bytes=PLANNING_RUNTIME_RESERVE_BYTES,
            historical_actual_max_context_capture_growth_bytes=1085865984,
            measured_projection=False,
            qsa_indexer_blocks=_ceildiv(context, 32768),
            provenance="full_accuracy_v2 actual C262144; preserve configured-B1 allowance",
            validated_capacity_claim=False,
        )
    if batch == 2:
        # Full48, selected precision, scheduler pool8195, all six prefill
        # captures and both decode widths. No growth on repeated replay.
        # The reserve also covers the small sampler-mode trace and transient
        # workspaces; it exceeds observed runtime growth by >1 GiB/rank.
        return dict(
            reserve_bytes=PLANNING_RUNTIME_RESERVE_BYTES,
            measured_runtime_growth_bytes=898926080,
            measured_context=SUPPORTED_CONTEXT,
            measured_pool_blocks=8195,
            qsa_indexer_blocks=_ceildiv(context, 32768),
            measured_projection=True,
            long_context_extrapolation=False,
            provenance="doc/vllm_integration/capacity_full_v1.json: pool_allocated to replayed",
            validated_capacity_claim=False,
        )
    buckets = (batch - 1).bit_length() + 1
    blocks = _ceildiv(context, 32768)
    attention = batch * (35 * 1179648 + 2490368 + 12 * 1703936 * blocks)
    moe = 48 * (393216 + (buckets - 1) * 458752)
    terminal = batch * 327680 + 8192
    prefill = PREFILL_TRACE_SHORT_CONTEXT_BYTES * blocks
    total = attention + moe + terminal + prefill + SAMPLING_TRACE_ALLOWANCE_BYTES + NONTRACE_RUNTIME_ALLOWANCE_BYTES
    return dict(
        reserve_bytes=max(PLANNING_RUNTIME_RESERVE_BYTES, total),
        attention_trace_bytes=attention,
        moe_trace_bytes=moe,
        terminal_trace_bytes=terminal,
        prefill_trace_bytes=prefill,
        sampling_trace_allowance_bytes=SAMPLING_TRACE_ALLOWANCE_BYTES,
        nontrace_runtime_allowance_bytes=NONTRACE_RUNTIME_ALLOWANCE_BYTES,
        historical_prepared_minus_listed_tensor_bytes=232719360,
        component_total_bytes=total,
        minimum_reserve_bytes=PLANNING_RUNTIME_RESERVE_BYTES,
        qsa_indexer_blocks=blocks,
        physical_moe_bucket_count=buckets,
        measured_projection=True,
        provenance="factored_decode_buckets_tracked_v1 C128 + prefill_signatures_full48_v2 C384; full48 fit pending",
        long_context_extrapolation=blocks > 1,
        validated_capacity_claim=False,
    )


@dataclass(frozen=True)
class CacheBudget:
    max_model_len: int
    max_num_seqs: int
    physical_capacity: int
    private_non_kv_bytes_per_cache: int
    prefill_checkpoint_bytes: int
    prefill_output_bytes: int
    factored_workspace_bytes: int
    shared_index_bytes: int
    known_resident_before_pool_bytes: int
    allocatable_bytes: int
    runtime_reserve_bytes: int
    runtime_forecast: dict
    available_pool_bytes: int
    maximum_pool_blocks: int
    plugin_output_reservation_blocks: int
    null_blocks: int = 1

    def to_dict(self):
        return asdict(self)

    def plugin_max_tokens_all_users(self):
        """Budget before pinned worker adds one output page for every slot.

        The pinned worker rounds this return value to pages after adding
        PAGE_TOKENS * B. Reserve that exact addition, and keep one null block
        inside our returned budget. A smaller shared total than B*C is valid;
        inability to admit even one full C is an explicit configuration error.
        """
        request_blocks = _ceildiv(self.max_model_len, PAGE_TOKENS)
        available_before_plugin = self.maximum_pool_blocks - self.plugin_output_reservation_blocks
        minimum_before_plugin = request_blocks + self.null_blocks
        minimum_warmup = 2 * self.max_num_seqs
        if available_before_plugin < minimum_before_plugin or self.maximum_pool_blocks < minimum_warmup:
            raise ValueError(
                "The unmeasured full-model memory plan cannot admit one complete "
                f"{self.max_model_len}-token request with {self.max_num_seqs} fixed slots: "
                f"pool budget={self.maximum_pool_blocks} blocks, "
                f"needed>={minimum_before_plugin + self.plugin_output_reservation_blocks} "
                f"including null/output reservations; warmup needs {minimum_warmup}. "
                "Use a measured memory plan or fewer fixed slots; advertised context is unchanged."
            )
        desired_before_plugin = self.max_num_seqs * request_blocks + self.null_blocks
        return min(desired_before_plugin, available_before_plugin) * PAGE_TOKENS


def plan_shared_kv_cache(
    max_model_len,
    max_num_seqs,
    *,
    allocatable_bytes=INHERITED_ALLOCATABLE_BYTES,
    runtime_reserve_bytes=None,
    known_static_bytes=KNOWN_STATIC_BYTES,
    kv_cache_dtype="bfloat16",
):
    """Compute exact listed tensor bytes and a separately stated runtime reserve.

    Shared pool setup must defer private KV allocation. Every fixed slot and
    the relocation scratch slot retain recurrent/compressed-index/private state.
    K and V use the explicit BF16/BFP8 tiled policy; no fallback is budgeted.
    allocatable_bytes is the total pre-model allocatable DRAM pool, never the
    post-load free value; passing post-load free bytes would charge weights twice.
    """
    context = _integer(max_model_len, "max_model_len")
    batch = _integer(max_num_seqs, "max_num_seqs")
    limit = _integer(allocatable_bytes, "allocatable_bytes")
    static = _integer(known_static_bytes, "known_static_bytes")
    if not 1 <= context <= SUPPORTED_CONTEXT:
        raise ValueError(f"max_model_len must be in 1..{SUPPORTED_CONTEXT}")
    if not 1 <= batch <= 32:
        raise ValueError("max_num_seqs must be in 1..32")
    forecast = runtime_forecast(context, batch)
    reserve = (
        forecast["reserve_bytes"]
        if runtime_reserve_bytes is None
        else _integer(runtime_reserve_bytes, "runtime_reserve_bytes")
    )
    forecast["explicit_reserve_override"] = runtime_reserve_bytes is not None
    if limit < 0 or reserve < 0 or static < 0:
        raise ValueError("Memory plan inputs must be nonnegative")
    physical = _ceildiv(context, 128) * 128
    private = FIXED_PRIVATE_NON_KV_BYTES + _ceildiv(physical, 256) * 196608 + _ceildiv(physical, 512) * 6144
    shared = (_ceildiv(physical, 1024) + _ceildiv(physical, 256)) * 32768
    private_copies = batch + int(batch > 1)
    checkpoint = batch * PREFILL_CHECKPOINT_BYTES_PER_SLOT
    workspace = factored_workspace_bytes(batch)
    resident = static + shared + private_copies * private + PREFILL_OUTPUT_BYTES + checkpoint + workspace
    available = limit - resident - reserve
    if kv_cache_dtype not in ("bfloat16", "bfloat8_b"):
        raise ValueError("KV cache dtype must be bfloat16 or bfloat8_b")
    stack_block_bytes = ATTENTION_LAYERS * 2 * 8 * (2048 if kv_cache_dtype == "bfloat16" else 1088)
    return CacheBudget(
        max_model_len=context,
        max_num_seqs=batch,
        physical_capacity=physical,
        private_non_kv_bytes_per_cache=private,
        prefill_checkpoint_bytes=checkpoint,
        prefill_output_bytes=PREFILL_OUTPUT_BYTES,
        factored_workspace_bytes=workspace,
        shared_index_bytes=shared,
        known_resident_before_pool_bytes=resident,
        allocatable_bytes=limit,
        runtime_reserve_bytes=reserve,
        runtime_forecast=forecast,
        available_pool_bytes=available,
        maximum_pool_blocks=max(0, available // stack_block_bytes),
        plugin_output_reservation_blocks=batch,
    )
