# SPDX-License-Identifier: Apache-2.0
"""Fail-closed vLLM/TT-plugin compatibility for Laguna hybrid KV.

The pinned TT plugin disables scheduler chunked prefill for Laguna and sizes a
hybrid pool with a generic sliding-window heuristic. Those choices are safe for
the qualified uniform cache, but they are incompatible with Laguna's opt-in
four-group layout: an unchunked 131K prompt makes all three sliding groups hold
the entire prompt in the allocation step, and the generic heuristic does not
reserve enough shared block IDs even for an 8192-token scheduler chunk.

This extension is active only when ``TT_LAGUNA_HYBRID_KV=1``. It temporarily
admits ``model_type=laguna`` through the TT plugin's chunked-prefill policy,
validates the cache-off/8192-token envelope (1..32 sequences), and raises the
worker's block count to the full-plus-three-sliding requirement (including
vLLM's globally reserved null block). No installed package is modified, and all
other TT models retain the pinned plugin behavior.
"""

from __future__ import annotations

import logging
import os
from math import ceil
from typing import Any

logger = logging.getLogger(__name__)

HYBRID_ENV = "TT_LAGUNA_HYBRID_KV"
MODEL_TYPE = "laguna"
QUALIFIED_BLOCK_SIZE = 64
QUALIFIED_SCHEDULER_CHUNK = 8192
QUALIFIED_SLIDING_WINDOW = 512
QUALIFIED_NUM_LAYERS = (40, 48)  # Laguna-XS-2.1, Laguna-S-2.1
QUALIFIED_SLIDING_GROUPS = 3
MAX_NUM_SEQS = 32  # one decode tile of rows

_PLATFORM_PATCH_MARKER = "_laguna_hybrid_kv_platform_patch"
_WORKER_PATCH_MARKER = "_laguna_hybrid_kv_worker_patch"


def _enabled() -> bool:
    return os.environ.get(HYBRID_ENV, "0") == "1"


def _text_config(vllm_config: Any) -> Any:
    hf_config = vllm_config.model_config.hf_config
    return getattr(hf_config, "text_config", hf_config)


def _expected_layer_types(num_layers: int) -> tuple[str, ...]:
    return tuple("full_attention" if layer % 4 == 0 else "sliding_attention" for layer in range(num_layers))


def validate_hybrid_kv_vllm_config(vllm_config: Any) -> None:
    """Require the exact scheduler/cache contract used by the block formula."""

    if not _enabled():
        return
    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    scheduler_config = vllm_config.scheduler_config
    hf_config = model_config.hf_config
    model_type = getattr(hf_config, "model_type", None)
    if model_type != MODEL_TYPE:
        raise RuntimeError(f"{HYBRID_ENV}=1 is Laguna-specific; got model_type={model_type!r}")
    if bool(cache_config.enable_prefix_caching):
        raise RuntimeError("Laguna hybrid KV qualification requires prefix caching disabled")
    if int(cache_config.block_size) != QUALIFIED_BLOCK_SIZE:
        raise RuntimeError(
            f"Laguna hybrid KV requires block_size={QUALIFIED_BLOCK_SIZE}, " f"got {cache_config.block_size}"
        )
    if not bool(scheduler_config.enable_chunked_prefill):
        raise RuntimeError("Laguna hybrid KV requires scheduler chunked prefill")
    if int(scheduler_config.max_num_batched_tokens) != QUALIFIED_SCHEDULER_CHUNK:
        raise RuntimeError(
            "Laguna hybrid KV requires max_num_batched_tokens="
            f"{QUALIFIED_SCHEDULER_CHUNK}, got {scheduler_config.max_num_batched_tokens}"
        )
    if not 1 <= int(scheduler_config.max_num_seqs) <= MAX_NUM_SEQS:
        raise RuntimeError(
            f"Laguna hybrid KV requires 1 <= max_num_seqs <= {MAX_NUM_SEQS}, got {scheduler_config.max_num_seqs}"
        )

    text_config = _text_config(vllm_config)
    layer_types = tuple(getattr(text_config, "layer_types", ()) or ())
    if len(layer_types) not in QUALIFIED_NUM_LAYERS or layer_types != _expected_layer_types(len(layer_types)):
        raise RuntimeError(
            "Laguna hybrid KV requires the exact 40-layer (XS) or 48-layer (S) "
            "full/sliding/sliding/sliding attention pattern"
        )
    sliding_window = int(getattr(text_config, "sliding_window", 0) or 0)
    if sliding_window != QUALIFIED_SLIDING_WINDOW:
        raise RuntimeError(
            f"Laguna hybrid KV requires sliding_window={QUALIFIED_SLIDING_WINDOW}, " f"got {sliding_window}"
        )


def exact_hybrid_kv_num_blocks(vllm_config: Any) -> int:
    """vLLM-visible pool floor for one full and three sliding groups, for 1..32 sequences.

    The four groups use disjoint IDs in one ``BlockPool``. The full group retains every token of every
    running sequence; vLLM's token budget for the whole pool is ``max_model_len`` (the adapter's
    ``get_max_tokens_all_users``), so the full group needs ``ceil(max_model_len / block)`` IDs however
    many sequences share it. Each sliding group retains, per sequence, ``window - 1`` old tokens plus that
    sequence's freshly scheduled tokens, and ``SlidingWindowSpec`` adds one block for an unaligned window
    boundary. The scheduled tokens of one step total at most ``max_num_batched_tokens`` across all
    sequences, so with N sequences one sliding group needs at most
    ``N * (ceil((window - 1) / block) + 2) + ceil(max_num_batched_tokens / block)`` IDs, and never more
    than N times the one-sequence requirement. vLLM also removes block 0 from the free queue as its global
    null block.

    One sequence at the qualified 131K context gives 2460 (2459 live IDs plus the null block), exactly as
    before; at 1,048,576 it gives 16,796. 32 sequences at 1,048,576 give 17,729.

    Do not add Laguna's prefill-padding block here. The adapter allocates that private scratch row outside
    vLLM's ID space, so a vLLM-visible ``num_blocks`` of 2460 deliberately becomes a physical tensor first
    dimension of 2461.
    """

    validate_hybrid_kv_vllm_config(vllm_config)
    block_size = int(vllm_config.cache_config.block_size)
    max_model_len = int(vllm_config.model_config.max_model_len)
    max_num_batched_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
    num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
    full_blocks = ceil(max_model_len / block_size)
    one_sequence_tokens = min(QUALIFIED_SLIDING_WINDOW - 1 + max_num_batched_tokens, max_model_len)
    one_sequence_blocks = ceil(one_sequence_tokens / block_size) + 1
    shared_chunk_bound = num_seqs * (ceil((QUALIFIED_SLIDING_WINDOW - 1) / block_size) + 2) + ceil(
        max_num_batched_tokens / block_size
    )
    sliding_blocks = min(num_seqs * one_sequence_blocks, shared_chunk_bound)
    null_blocks = 1
    return full_blocks + QUALIFIED_SLIDING_GROUPS * sliding_blocks + null_blocks


def _patch_platform(platform_module: Any, platform_class: type) -> bool:
    """Wrap TTPlatform once so only an enabled Laguna run bypasses its denylist."""

    if platform_class.__dict__.get(_PLATFORM_PATCH_MARKER, False):
        return False
    original_method = platform_class.check_and_update_config
    original_function = getattr(original_method, "__func__", None)
    if original_function is None:
        raise TypeError("TTPlatform.check_and_update_config is not a classmethod")

    def check_and_update_config(cls: type, vllm_config: Any) -> None:
        admitted_types = platform_module._CHUNKED_PREFILL_MODEL_TYPES
        already_admitted = MODEL_TYPE in admitted_types
        if _enabled():
            admitted_types.add(MODEL_TYPE)
        try:
            original_function(cls, vllm_config)
        finally:
            if _enabled() and not already_admitted:
                admitted_types.discard(MODEL_TYPE)
        validate_hybrid_kv_vllm_config(vllm_config)

    check_and_update_config.__name__ = original_function.__name__
    check_and_update_config.__doc__ = original_function.__doc__
    setattr(
        platform_class,
        "check_and_update_config",
        classmethod(check_and_update_config),
    )
    setattr(platform_class, _PLATFORM_PATCH_MARKER, True)
    return True


def _patch_worker(worker_module: Any) -> bool:
    """Raise the TT worker's global block heuristic to Laguna's exact floor."""

    if getattr(worker_module, _WORKER_PATCH_MARKER, False):
        return False
    original = worker_module.get_num_available_blocks_tt

    def get_num_available_blocks_tt(vllm_config: Any, num_devices: int = 1) -> int:
        proposed = int(original(vllm_config, num_devices))
        if not _enabled():
            return proposed
        required = exact_hybrid_kv_num_blocks(vllm_config)
        selected = max(proposed, required)
        logger.warning(
            "Laguna hybrid KV block pool: plugin_heuristic=%d exact_floor=%d selected=%d",
            proposed,
            required,
            selected,
        )
        return selected

    get_num_available_blocks_tt.__name__ = original.__name__
    get_num_available_blocks_tt.__doc__ = original.__doc__
    worker_module.get_num_available_blocks_tt = get_num_available_blocks_tt
    setattr(worker_module, _WORKER_PATCH_MARKER, True)
    return True


def install_hybrid_kv_patch() -> bool:
    """Install both runtime wrappers once in the current process."""

    import vllm_tt_plugin.platform as platform_module
    import vllm_tt_plugin.worker as worker_module

    platform_installed = _patch_platform(platform_module, platform_module.TTPlatform)
    worker_installed = _patch_worker(worker_module)
    if platform_installed or worker_installed:
        logger.info("Installed fail-closed Laguna hybrid-KV compatibility")
    return platform_installed or worker_installed


def hybrid_kv_patch_is_installed() -> bool:
    """Return whether both model-local wrappers are active in this process."""

    import vllm_tt_plugin.platform as platform_module
    import vllm_tt_plugin.worker as worker_module

    return bool(
        platform_module.TTPlatform.__dict__.get(_PLATFORM_PATCH_MARKER, False)
        and getattr(worker_module, _WORKER_PATCH_MARKER, False)
    )
