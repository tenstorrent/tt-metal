# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class KVCachePhase(str, Enum):
    """KV cache write discipline for DiffusionGemma's multi-phase forward."""

    PREFILL_WRITE = "prefill_write"
    DENOISE_READONLY = "denoise_readonly"
    COMMIT_APPEND = "commit_append"


def coerce_kv_cache_phase(value, *, is_decode: bool) -> KVCachePhase:
    if value is None:
        return KVCachePhase.COMMIT_APPEND if is_decode else KVCachePhase.PREFILL_WRITE
    if isinstance(value, KVCachePhase):
        phase = value
    else:
        phase = KVCachePhase(value)
    if is_decode and phase is KVCachePhase.DENOISE_READONLY:
        raise ValueError(
            "DENOISE_READONLY is a prefill-only KV phase; decode must write or append the current token KV"
        )
    return phase
