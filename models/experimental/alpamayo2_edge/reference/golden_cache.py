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

from models.experimental.alpamayo2_edge.reference.config import Alpamayo2EdgeConfig

_REFERENCE_DIR = Path(__file__).parent
# The checkpoint files that pin its tensors and architecture; hashing the multi-GB shards themselves is too slow.
_CHECKPOINT_KEY_FILES = ("config.json", "model.safetensors.index.json")


def cache_dir() -> Path:
    """`ALPAMAYO_CACHE_DIR`, else a per-user `${XDG_CACHE_HOME:-~/.cache}/alpamayo2_edge`."""
    explicit = os.environ.get("ALPAMAYO_CACHE_DIR")
    if explicit:
        return Path(explicit)
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "alpamayo2_edge"


def _reference_source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(_REFERENCE_DIR.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _checkpoint_hash(model_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in _CHECKPOINT_KEY_FILES:
        digest.update((Path(model_dir) / name).read_bytes())
    return digest.hexdigest()


def _defining_file_hash(fn: Callable) -> str:
    """Hash of the source file that defines `fn`, so an edit to a test-side golden function invalidates its entries."""
    code = getattr(fn, "__code__", None)
    if code is None:
        raise TypeError(f"golden_cache needs a plain function or lambda to locate its source, got {fn!r}")
    return hashlib.sha256(Path(code.co_filename).read_bytes()).hexdigest()


def _canonical(value: Any) -> Any:
    """JSON-able exact form of a key value; tensors are hashed by dtype, shape and bytes."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest = hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        return {"tensor": [str(tensor.dtype), list(tensor.shape), digest]}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    raise TypeError(
        f"golden_cache cannot key on a {type(value).__name__}; pass tensors, numbers, strings or containers"
    )


def cached(name: str, key: dict, fn: Callable[[], Any], *, cfg: Alpamayo2EdgeConfig) -> Any:
    """Return `fn()`, stored under `name` and keyed on `key`, the model `cfg` that `fn` runs and the sources.

    `cfg` contributes the checkpoint (its config and weight index) and the vision, LM and expert depths. Any edit to a
    file in `reference/` or to the file that defines `fn`, or another torch version, invalidates the entry; stale
    files are left behind, not deleted. Results load with `weights_only=True`, so they must be containers of tensors,
    numbers and strings.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"cache name must be a plain file stem, got {name!r}")
    full_key = {
        "key": _canonical(key),
        "checkpoint": _checkpoint_hash(cfg.model_dir),
        "layers": [cfg.vision.layers, cfg.text.layers, cfg.expert.layers],
        "source": _reference_source_hash(),
        "fn_source": _defining_file_hash(fn),
        "torch": torch.__version__,
    }
    digest = hashlib.sha256(json.dumps(full_key, sort_keys=True).encode()).hexdigest()[:16]
    path = cache_dir() / f"{name}-{digest}.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    result = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(result, tmp)
    os.replace(tmp, path)
    return result
