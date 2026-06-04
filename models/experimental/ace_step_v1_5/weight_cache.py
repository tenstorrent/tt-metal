# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host weight cache for ACE-Step demos.

Avoids re-reading the same ``.safetensors`` shards from disk on every CLI invocation.
The full checkpoint is loaded **once** into a process-global torch state dict; filtered
numpy views (condition encoder, audio detokenizer, etc.) are derived from that copy.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, TypeVar, Union

from loguru import logger as log

_T = TypeVar("_T")

# Process-global caches (path_key -> payload)
_torch_sd_cache: Dict[tuple, Dict[str, Any]] = {}
_np_dict_cache: Dict[tuple, Any] = {}
_lock = threading.Lock()


def weight_cache_enabled() -> bool:
    """Return False only when ``ACE_STEP_DISABLE_WEIGHT_CACHE=1``."""
    return os.environ.get("ACE_STEP_DISABLE_WEIGHT_CACHE", "0") not in ("1", "true", "True", "yes")


def log_weight_load(component: str, path: str | None = None) -> None:
    tag = f"  path={path}" if path else ""
    log.info("⏳ LOAD   {:<30}{}", component, tag)


def log_weight_load_done(component: str, *, num_keys: int, elapsed_s: float) -> None:
    log.info("✓ LOAD done {} ({} tensors, {:.1f}s)", component, num_keys, elapsed_s)


def log_weight_reuse(component: str, *, source: str = "memory") -> None:
    if source == "memory":
        where = "already loaded in memory"
    elif source == "device":
        where = "already in TTNN device memory"
    else:
        where = f"reusing cache ({source})"
    log.info("♻  REUSE  {:<30}[{}]", component, where)


def log_weights_ready() -> None:
    log.info("✅ Weights ready – subsequent runs reuse cached host tensors (no safetensors I/O).")


def _path_cache_key(path: str) -> tuple[str, float, int]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(path)
    st = p.stat()
    return (str(p), float(st.st_mtime_ns), int(st.st_size))


def _tensor_to_f32_numpy(value: Any):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().to(torch.float32).cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def cached_call(
    component: str,
    path: str,
    *,
    tag: str,
    loader: Callable[[], _T],
) -> _T:
    """Load *loader* result once per checkpoint revision; log LOAD vs REUSE."""
    if not weight_cache_enabled():
        log_weight_load(component, path)
        t0 = time.perf_counter()
        fresh = loader()
        n = len(fresh) if isinstance(fresh, dict) else 1
        log_weight_load_done(component, num_keys=int(n), elapsed_s=time.perf_counter() - t0)
        return fresh

    path_key = _path_cache_key(path)
    mem_key = (path_key, tag)

    with _lock:
        hit = _np_dict_cache.get(mem_key)
        if hit is not None:
            log_weight_reuse(component, source="memory")
            return hit  # type: ignore[return-value]

    log_weight_load(component, path)
    t0 = time.perf_counter()
    fresh = loader()
    elapsed = time.perf_counter() - t0
    n = len(fresh) if isinstance(fresh, dict) else 1
    log_weight_load_done(component, num_keys=int(n), elapsed_s=elapsed)

    with _lock:
        _np_dict_cache[mem_key] = fresh  # type: ignore[assignment]

    return fresh


def load_prefix_weights_np(
    safetensors_path: str,
    prefixes: Union[str, Sequence[str]],
    *,
    component: str,
    tag: str,
) -> Dict[str, np.ndarray]:
    """Load tensors whose keys start with *prefixes* from the shared checkpoint cache."""

    if isinstance(prefixes, str):
        prefix_tuple = (prefixes,)
    else:
        prefix_tuple = tuple(prefixes)

    def _load() -> Dict[str, np.ndarray]:
        sd = get_torch_state_dict(
            safetensors_path,
            component="safetensors-checkpoint",
        )
        return {k: _tensor_to_f32_numpy(v) for k, v in sd.items() if k.startswith(prefix_tuple)}

    out = cached_call(component, str(safetensors_path), tag=tag, loader=_load)
    if not out:
        raise KeyError(
            f"No weights matching prefixes {prefix_tuple!r} in {safetensors_path}. "
            "Checkpoint may be wrong or incomplete."
        )
    return out  # type: ignore[return-value]


def get_torch_state_dict(path: str, *, component: str = "safetensors") -> Dict[str, Any]:
    """Full safetensors file as CPU torch tensors (shared across all readers)."""
    if not weight_cache_enabled():
        log_weight_load(component, path)
        t0 = time.perf_counter()
        fresh = _load_torch_state_dict_uncached(path)
        log_weight_load_done(component, num_keys=len(fresh), elapsed_s=time.perf_counter() - t0)
        return fresh

    path_key = _path_cache_key(path)
    with _lock:
        hit = _torch_sd_cache.get(path_key)
        if hit is not None:
            log_weight_reuse(component, source="memory")
            return hit

    log_weight_load(component, path)
    t0 = time.perf_counter()
    fresh = _load_torch_state_dict_uncached(path)
    log_weight_load_done(component, num_keys=len(fresh), elapsed_s=time.perf_counter() - t0)
    with _lock:
        _torch_sd_cache[path_key] = fresh
    return fresh


def _load_torch_state_dict_uncached(path: str) -> Dict[str, Any]:
    try:
        from safetensors.torch import load_file as torch_load_file  # type: ignore

        return {k: v.detach().cpu() for k, v in torch_load_file(path, device="cpu").items()}
    except Exception:
        from safetensors.numpy import load_file  # type: ignore

        return load_file(path)


def clear_memory_cache() -> None:
    """Drop in-process caches (e.g. after checkpoint swap in tests)."""
    with _lock:
        _torch_sd_cache.clear()
        _np_dict_cache.clear()
