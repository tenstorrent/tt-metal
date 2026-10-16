# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Golden cache helpers for the Llama 3.1 8B reference model.

Provides save/load utilities for per-module golden tensors used in PCC
validation against device outputs. A frozen dataclass cache key ensures
that different configurations (sequence length, number of layers, dtype,
random seed, etc.) produce distinct cache files and stale results are
never reused silently.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger

# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceCacheKey:
    """All parameters that affect reference output identity.

    Changing any field produces a different cache filename.
    """

    model_name: str
    hidden_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    seq_len: int
    dtype: str  # e.g. "bfloat16"
    seed: int

    def __str__(self) -> str:
        return (
            f"{self.model_name}"
            f"_h{self.hidden_size}"
            f"_l{self.num_layers}"
            f"_kv{self.num_kv_heads}"
            f"_d{self.head_dim}"
            f"_s{self.seq_len}"
            f"_{self.dtype}"
            f"_seed{self.seed}"
        )


# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_ENV = "TT_LLAMA31_8B_REF_CACHE"
_DEFAULT_CACHE_DIR = "/tmp/llama31_8b_ref_cache"


def _cache_dir() -> Path:
    """Return the directory used for reference caches."""
    return Path(os.environ.get(_DEFAULT_CACHE_ENV, _DEFAULT_CACHE_DIR))


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_reference_cache(
    cache_key: ReferenceCacheKey,
    kv_cache: dict[int, dict[str, torch.Tensor]],
    logits: torch.Tensor | None = None,
) -> Path:
    """Save per-layer golden tensors to disk.

    Args:
        cache_key: Frozen dataclass identifying this reference run.
        kv_cache: Dict mapping layer_idx to {"k": k_tensor, "v": v_tensor}.
        logits: Optional logits tensor to save alongside the KV cache.

    Returns:
        Path to the saved cache file.
    """
    cache_path = _cache_dir() / f"{cache_key}.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {"kv_cache": kv_cache}
    if logits is not None:
        payload["logits"] = logits

    torch.save(payload, cache_path)
    n_layers = len(kv_cache)
    logger.info(f"Saved reference cache to {cache_path} ({n_layers} layers)")
    return cache_path


def load_reference_cache(
    cache_key: ReferenceCacheKey,
) -> tuple[dict[int, dict[str, torch.Tensor]], torch.Tensor | None]:
    """Load per-layer golden tensors from disk.

    Args:
        cache_key: Frozen dataclass identifying the reference run.

    Returns:
        (kv_cache, logits) where:
            kv_cache: {layer_idx: {"k": k_tensor, "v": v_tensor}}
            logits: Optional logits tensor (None if not saved).

    Raises:
        FileNotFoundError: If no cache exists for this key.
        ValueError: If the cache file has an unexpected format.
    """
    cache_path = _cache_dir() / f"{cache_key}.pt"

    if not cache_path.exists():
        raise FileNotFoundError(f"Reference cache not found: {cache_path}")

    cached = torch.load(cache_path, weights_only=False)

    if "kv_cache" not in cached:
        raise ValueError(f"Invalid cache format in {cache_path} (missing 'kv_cache' key)")

    logger.info(f"Loaded reference cache from {cache_path}")
    return cached["kv_cache"], cached.get("logits")


def check_reference_cache_exists(cache_key: ReferenceCacheKey) -> bool:
    """Check whether a cache file exists for the given key.

    Args:
        cache_key: Frozen dataclass identifying the reference run.

    Returns:
        True if a valid cache file exists on disk.
    """
    cache_path = _cache_dir() / f"{cache_key}.pt"
    exists = cache_path.exists()
    if exists:
        logger.info(f"Reference cache found: {cache_path}")
    else:
        logger.debug(f"Reference cache not found: {cache_path}")
    return exists
