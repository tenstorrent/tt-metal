# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE 5 Hz LM decode: unify WIDTH_SHARDED activation specs to cut ``ReshardDeviceOperation`` churn.

Tracy on eager decode shows thousands of L1→L1 reshards with mismatched shard widths (e.g.
``[32; 96]`` MLP intermediates vs ``[32; 64]`` residual). ``tt_transformers`` matmul output getters
often return generic ``L1_WIDTH_SHARDED_MEMORY_CONFIG`` while ``get_residual_mem_config(DECODE)``
uses a fixed core grid — TTNN inserts reshards between ops.

Decode matmul output shard specs are derived from each op's DRAM-sharded program config
(core grid + per-core ``N``). Forcing ``get_residual_mem_config(DECODE)`` (``[32, 64]`` on
32 cores) onto attention gather / wo / all-gather getters makes matmul log config mismatches
and ignore the provided layout (computed shapes include ``[32, 128]`` and ``[32, 32]``).

This module keeps the patch hook for future getters that truly match the residual grid; the
getter list is intentionally empty until such a case is validated on silicon.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from models.tt_transformers.tt.common import Mode

# ``(getter_name, index of prefetcher in positional args, or None if kw-only/default trailing)``
_DECODE_RESIDUAL_SHARD_GETTERS: tuple[tuple[str, int | None], ...] = ()


def _prefetcher_from_call(
    args: tuple,
    kwargs: dict,
    *,
    prefetcher_index: int | None,
) -> Any:
    pf = kwargs.get("prefetcher")
    if pf is not None:
        return pf
    if prefetcher_index is not None and len(args) > prefetcher_index:
        return args[prefetcher_index]
    return None


def _patch_decode_residual_shard_getter(model_args: Any, name: str, *, prefetcher_index: int | None) -> None:
    original: Callable = getattr(model_args, name)
    if hasattr(original, "cache_clear"):
        original.cache_clear()

    @functools.wraps(original)
    def patched(*args, **kwargs):
        mode = args[0] if args else kwargs.get("mode")
        if mode != Mode.DECODE:
            return original(*args, **kwargs)
        if getattr(model_args, "is_galaxy", False):
            return original(*args, **kwargs)
        if _prefetcher_from_call(args, kwargs, prefetcher_index=prefetcher_index) is not None:
            return original(*args, **kwargs)
        return model_args.get_residual_mem_config(Mode.DECODE, None)

    if hasattr(original, "cache_clear"):
        patched = functools.lru_cache(maxsize=None)(patched)  # type: ignore[assignment]

    setattr(model_args, name, patched)


def ace_step_patch_model_args_decode_unified_shard(model_args: Any) -> None:
    """Apply decode residual-shard overrides for vetted matmul output getters (none by default)."""
    if getattr(model_args, "is_galaxy", False):
        return
    for name, pf_index in _DECODE_RESIDUAL_SHARD_GETTERS:
        if hasattr(model_args, name):
            _patch_decode_residual_shard_getter(model_args, name, prefetcher_index=pf_index)


__all__ = ["ace_step_patch_model_args_decode_unified_shard"]
