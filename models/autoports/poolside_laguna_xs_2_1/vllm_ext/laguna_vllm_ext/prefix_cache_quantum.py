# SPDX-License-Identifier: Apache-2.0
"""Canonical-prefix admission for Laguna's shape-sensitive prefill kernels.

Laguna's long prefill is evaluated in fixed outer chunks (8K tokens in the
qualified p150x2 profile).  A KV block produced by a smaller/differently
partitioned prefill is mathematically valid, but low-precision kernel
partitioning can make it numerically different from the same block produced as
part of the canonical long-prefill chunk.  Reusing an arbitrary 64-token vLLM
cache hit can therefore perturb greedy generation.

This extension keeps vLLM's physical KV block size at 64 while admitting cache
hits only in whole canonical outer chunks.  Crucially, truncation happens in
``KVCacheManager.get_computed_blocks`` *before* ``allocate_slots`` increments
references: the rejected tail receives fresh writable blocks and the model
never mutates shared cached blocks.

Only complete canonical prompt chunks are inserted into the prefix hash map.
This prevents both partial prompt chunks and token-by-token decode KV from
winning vLLM's oldest-first duplicate lookup and poisoning a later canonical
hit.
"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

PATCH_MARKER = "_laguna_prefix_cache_quantum_patch"
DEFAULT_PREFIX_QUANTUM = 8192
QUALIFIED_KV_BLOCK_SIZE = 64
QUALIFIED_MAX_NUM_SEQS = 1


def prefix_cache_quantum_enabled() -> bool:
    """Return whether this process is serving the explicit Laguna APC profile."""

    return os.environ.get("TT_LAGUNA_PREFIX_CACHE", "0") == "1"


def canonical_prefix_quantum() -> int:
    """Return the qualified outer-prefill/cache-admission quantum."""

    raw = os.environ.get("TT_LAGUNA_PREFILL_FAST_CHUNK", str(DEFAULT_PREFIX_QUANTUM))
    try:
        quantum = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"invalid TT_LAGUNA_PREFILL_FAST_CHUNK={raw!r}") from exc
    if quantum != DEFAULT_PREFIX_QUANTUM:
        raise RuntimeError(
            "Laguna canonical prefix caching requires "
            f"TT_LAGUNA_PREFILL_FAST_CHUNK={DEFAULT_PREFIX_QUANTUM}, got {quantum}"
        )
    return quantum


def validate_prefix_cache_vllm_config(vllm_config: Any) -> None:
    """Fail closed unless the scheduler matches the qualified APC envelope.

    The KV group itself is unavailable until cache planning, so
    :func:`_validate_manager_geometry` separately checks the realized manager.
    """

    if not prefix_cache_quantum_enabled():
        return

    quantum = canonical_prefix_quantum()
    scheduler_config = vllm_config.scheduler_config
    cache_config = vllm_config.cache_config
    errors = []
    if not bool(cache_config.enable_prefix_caching):
        errors.append("vLLM prefix caching is not enabled")
    if int(cache_config.block_size) != QUALIFIED_KV_BLOCK_SIZE:
        errors.append(
            f"KV block size is {cache_config.block_size}, expected {QUALIFIED_KV_BLOCK_SIZE}"
        )
    if bool(scheduler_config.enable_chunked_prefill):
        errors.append("scheduler chunked prefill is enabled")
    if int(scheduler_config.max_num_seqs) != QUALIFIED_MAX_NUM_SEQS:
        errors.append(
            f"max_num_seqs is {scheduler_config.max_num_seqs}, expected {QUALIFIED_MAX_NUM_SEQS}"
        )
    if vllm_config.speculative_config is not None:
        errors.append("vLLM speculative decoding is configured")
    if getattr(vllm_config, "kv_transfer_config", None) is not None:
        errors.append("an external KV-transfer connector is configured")
    if os.environ.get("TT_LAGUNA_SPEC_DECODE", ""):
        errors.append("TT_LAGUNA_SPEC_DECODE is set")
    if os.environ.get("TT_LAGUNA_PREFILL_FAST", "1") != "1":
        errors.append("TT_LAGUNA_PREFILL_FAST is not 1")
    if os.environ.get("TT_LAGUNA_PREFILL_SDPA_CHUNK", str(quantum)) != str(quantum):
        errors.append(f"TT_LAGUNA_PREFILL_SDPA_CHUNK is not {quantum}")
    if os.environ.get("TT_LAGUNA_HYBRID_KV", "0") != "0":
        errors.append("TT_LAGUNA_HYBRID_KV is not 0")
    if errors:
        raise RuntimeError(
            f"Laguna canonical prefix-cache policy (quantum={quantum}) rejected the vLLM config: "
            + "; ".join(errors)
        )


def _full_attention_spec_type() -> type:
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    return FullAttentionSpec


def _canonical_cache_limit(request: Any, quantum: int) -> int:
    """Largest complete canonical *prompt* checkpoint eligible for hashing.

    Capping at ``num_prompt_tokens`` deliberately excludes generated assistant
    tokens, whose KV was produced one token at a time by the decode kernel.
    """

    prompt_tokens = int(getattr(request, "num_prompt_tokens"))
    if prompt_tokens < 0:
        raise RuntimeError(f"Laguna request has negative num_prompt_tokens={prompt_tokens}")
    return prompt_tokens // quantum * quantum


def _validate_manager_geometry(manager: Any, quantum: int) -> None:
    if quantum != DEFAULT_PREFIX_QUANTUM:
        raise RuntimeError(
            f"Laguna prefix-cache quantum must be {DEFAULT_PREFIX_QUANTUM}, got {quantum}"
        )
    if not bool(manager.enable_caching):
        raise RuntimeError("Laguna canonical prefix caching requires KV caching to be enabled")
    if bool(getattr(manager, "use_eagle", False)):
        raise RuntimeError("Laguna canonical prefix caching does not support EAGLE/MTP cache groups")
    coordinator = manager.coordinator
    scheduler_block_size = int(coordinator.scheduler_block_size)
    if scheduler_block_size != QUALIFIED_KV_BLOCK_SIZE:
        raise RuntimeError(
            f"Laguna prefix caching requires scheduler block size {QUALIFIED_KV_BLOCK_SIZE}, "
            f"got {scheduler_block_size}"
        )
    groups = manager.kv_cache_config.kv_cache_groups
    if len(groups) != 1:
        raise RuntimeError(
            "Laguna canonical prefix caching requires exactly one uniform KV cache group, "
            f"got {len(groups)}"
        )
    group = groups[0]
    spec = group.kv_cache_spec
    if not isinstance(spec, _full_attention_spec_type()):
        raise RuntimeError(
            "Laguna canonical prefix caching requires one FullAttentionSpec KV group, "
            f"got {type(spec).__name__}"
        )
    if bool(getattr(group, "is_eagle_group", False)):
        raise RuntimeError("Laguna canonical prefix caching does not support an EAGLE KV group")
    block_size = int(spec.block_size)
    if block_size != QUALIFIED_KV_BLOCK_SIZE:
        raise RuntimeError(
            f"Laguna prefix caching requires KV group block size {QUALIFIED_KV_BLOCK_SIZE}, "
            f"got {block_size}"
        )
    if quantum % block_size:
        raise RuntimeError(
            f"Laguna prefix-cache quantum {quantum} is not divisible by KV block size {block_size}"
        )


def _adjust_recorded_hits(manager: Any, request: Any, dropped_tokens: int) -> None:
    """Correct the stock metric, which was recorded before this wrapper ran."""

    if dropped_tokens <= 0 or not manager.log_stats:
        return
    stats = manager.prefix_cache_stats
    if stats is None:
        raise RuntimeError("Laguna prefix-cache stats are enabled but unavailable")
    field = "preempted_hits" if int(getattr(request, "num_preemptions", 0)) > 0 else "hits"
    value = int(getattr(stats, field))
    if value < dropped_tokens:
        raise RuntimeError(
            f"Laguna cannot adjust prefix-cache metric {field}: value={value}, dropped={dropped_tokens}"
        )
    setattr(stats, field, value - dropped_tokens)


def _truncate_computed_blocks(manager: Any, request: Any, blocks: Any, raw_tokens: int, quantum: int):
    """Return an ownership-safe canonical prefix from a stock vLLM lookup."""

    raw_tokens = int(raw_tokens)
    if raw_tokens <= 0:
        return blocks, raw_tokens
    _validate_manager_geometry(manager, quantum)
    accepted_tokens = raw_tokens // quantum * quantum
    if accepted_tokens == raw_tokens:
        return blocks, raw_tokens

    groups = tuple(blocks.blocks)
    if not groups:
        raise RuntimeError(f"vLLM returned {raw_tokens} cached tokens without KV blocks")
    expected_group_count = len(manager.kv_cache_config.kv_cache_groups)
    if len(groups) != expected_group_count:
        raise RuntimeError(
            f"vLLM returned {len(groups)} cached KV groups, expected {expected_group_count}"
        )

    kept_groups = []
    for index, (group_blocks, group) in enumerate(zip(groups, manager.kv_cache_config.kv_cache_groups)):
        block_size = int(group.kv_cache_spec.block_size)
        if raw_tokens % block_size:
            raise RuntimeError(
                f"raw cache hit {raw_tokens} is not aligned to KV group {index} block size {block_size}"
            )
        raw_block_count = raw_tokens // block_size
        if len(group_blocks) != raw_block_count:
            raise RuntimeError(
                f"KV group {index} returned {len(group_blocks)} blocks for {raw_tokens} tokens "
                f"at block size {block_size}"
            )
        keep = accepted_tokens // block_size
        kept_groups.append(list(group_blocks[:keep]))

    dropped = raw_tokens - accepted_tokens
    _adjust_recorded_hits(manager, request, dropped)
    accepted = manager.create_kv_cache_blocks(tuple(kept_groups))
    logger.info(
        "Laguna canonical prefix-cache admission: raw_tokens=%d accepted_tokens=%d quantum=%d",
        raw_tokens,
        accepted_tokens,
        quantum,
    )
    return accepted, accepted_tokens


def _patch_kv_cache_manager(manager_class: type) -> bool:
    """Wrap vLLM's cache lookup/admission methods once."""

    if manager_class.__dict__.get(PATCH_MARKER, False):
        return False

    original_init = manager_class.__init__
    original_get = manager_class.get_computed_blocks
    original_allocate = manager_class.allocate_slots
    original_cache = manager_class.cache_blocks

    @functools.wraps(original_init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if prefix_cache_quantum_enabled():
            _validate_manager_geometry(self, canonical_prefix_quantum())

    @functools.wraps(original_get)
    def get_computed_blocks(self: Any, request: Any):
        blocks, raw_tokens = original_get(self, request)
        if not prefix_cache_quantum_enabled():
            return blocks, raw_tokens
        quantum = canonical_prefix_quantum()
        return _truncate_computed_blocks(self, request, blocks, raw_tokens, quantum)

    @functools.wraps(original_allocate)
    def allocate_slots(
        self: Any,
        request: Any,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Any | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
        reserved_blocks: int = 0,
        has_scheduled_reqs: bool = True,
    ):
        if not prefix_cache_quantum_enabled():
            return original_allocate(
                self,
                request,
                num_new_tokens,
                num_new_computed_tokens,
                new_computed_blocks,
                num_lookahead_tokens,
                num_external_computed_tokens,
                delay_cache_blocks,
                num_encoder_tokens,
                full_sequence_must_fit,
                reserved_blocks,
                has_scheduled_reqs,
            )

        quantum = canonical_prefix_quantum()
        _validate_manager_geometry(self, quantum)
        if num_external_computed_tokens:
            raise RuntimeError("Laguna canonical prefix caching does not support external KV connectors")
        if num_lookahead_tokens:
            raise RuntimeError("Laguna canonical prefix caching does not support speculative lookahead slots")
        if num_encoder_tokens:
            raise RuntimeError("Laguna canonical prefix caching supports decoder-only requests")

        # Suppress the stock method's direct coordinator.cache_blocks call, then
        # commit only the complete canonical prompt checkpoints after allocation
        # succeeds. Preserve a caller's own delayed-cache contract (P/D).
        allocated = original_allocate(
            self,
            request,
            num_new_tokens,
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_tokens,
            num_external_computed_tokens,
            True,
            num_encoder_tokens,
            full_sequence_must_fit,
            reserved_blocks,
            has_scheduled_reqs,
        )
        if allocated is None or delay_cache_blocks or not self.enable_caching:
            return allocated

        num_local_computed_tokens = int(request.num_computed_tokens) + int(num_new_computed_tokens)
        total_computed_tokens = min(num_local_computed_tokens, int(self.max_model_len))
        requested_cache_tokens = min(
            total_computed_tokens + int(num_new_tokens),
            int(request.num_tokens),
        )
        cache_tokens = min(requested_cache_tokens, _canonical_cache_limit(request, quantum))
        if cache_tokens > 0:
            self.coordinator.cache_blocks(request, cache_tokens)
        return allocated

    @functools.wraps(original_cache)
    def cache_blocks(self: Any, request: Any, num_computed_tokens: int) -> None:
        if not prefix_cache_quantum_enabled():
            return original_cache(self, request, num_computed_tokens)
        quantum = canonical_prefix_quantum()
        _validate_manager_geometry(self, quantum)
        cache_tokens = min(int(num_computed_tokens), _canonical_cache_limit(request, quantum))
        if cache_tokens > 0:
            return original_cache(self, request, cache_tokens)
        return None

    manager_class.__init__ = __init__
    manager_class.get_computed_blocks = get_computed_blocks
    manager_class.allocate_slots = allocate_slots
    manager_class.cache_blocks = cache_blocks
    setattr(manager_class, PATCH_MARKER, True)
    return True


def install_prefix_cache_quantum_patch() -> bool:
    """Install canonical cache admission in the current vLLM process."""

    from vllm.v1.core.kv_cache_manager import KVCacheManager

    installed = _patch_kv_cache_manager(KVCacheManager)
    if installed:
        logger.info("Installed Laguna canonical prefix-cache admission")
    return installed


def prefix_cache_quantum_patch_is_installed() -> bool:
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    return bool(KVCacheManager.__dict__.get(PATCH_MARKER, False))
