# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""On-disk memoization of expensive CPU reference results, so full-depth goldens are computed once per source."""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

import torch

from models.experimental.alpamayo2_edge.reference.config import num_layers_override

DEFAULT_CACHE_DIR = "/localdev/ctr-mdukic/alpamayo_cache"
_REFERENCE_DIR = Path(__file__).parent


def cache_dir() -> Path:
    return Path(os.environ.get("ALPAMAYO_CACHE_DIR") or DEFAULT_CACHE_DIR)


def _reference_source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(_REFERENCE_DIR.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cached(name: str, key: dict, fn: Callable[[], Any]) -> Any:
    """Return `fn()`, stored under `name` and keyed on `key`, the reference sources and ALPAMAYO_NUM_LAYERS.

    Any edit to a file in `reference/` invalidates every entry; stale files are left behind, not deleted.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"cache name must be a plain file stem, got {name!r}")
    full_key = {"key": key, "num_layers": num_layers_override(), "source": _reference_source_hash()}
    digest = hashlib.sha256(json.dumps(full_key, sort_keys=True, default=str).encode()).hexdigest()[:16]
    path = cache_dir() / f"{name}-{digest}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    result = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(result, tmp)
    os.replace(tmp, path)
    return result
