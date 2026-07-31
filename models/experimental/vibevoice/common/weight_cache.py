# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Byte-exact on-disk cache for preprocessed (tilized) VibeVoice device weights.

Motivation
----------
``TTVibeVoiceModel.from_checkpoint`` uploads the language-model matmul weights, the
speech connectors and the diffusion head to the device on every process start.  The
raw safetensors are mmap'd (so re-reading them is cheap), but the per-tensor
transpose/reshape + fp32->bf16 cast + host tilize + device transfer are recomputed
each time — several seconds that short runs pay in full before any inference.

``ttnn.as_tensor(cache_file_name=...)`` already serialises the *device-ready* (tiled)
tensor to a flatbuffer on first build and reloads it straight to device on later runs,
skipping the host tilize.  This helper wraps that mechanism so that on a cache hit the
torch tensor is *never constructed* — the checkpoint bytes are not touched at all —
and adds checkpoint-keyed cache directories plus hit/miss accounting.

Correctness
-----------
The cached flatbuffer is byte-identical to the tensor the uncached path would have
uploaded (same dump that ``ttnn.as_tensor`` performs internally), so enabling the cache
cannot change model output.  A load failure (corrupt/partial file) falls back to a
fresh build, so a bad cache degrades to the current behaviour rather than breaking.
"""

import hashlib
import os
from pathlib import Path
from typing import Callable, Optional

import ttnn

# Bump when the on-disk key scheme or the meaning of a cached tensor changes, so that
# stale caches from an older layout are simply missed (and rebuilt) rather than reused.
_CACHE_FORMAT_VERSION = "v1"

# Safetensors shard suffix, used to build the checkpoint signature.
_SHARD_SUFFIX = ".safetensors"


def resolve_weight_cache(
    model_path: str, submodule: str = "", *, weight_cache_dir: Optional[str] = None
) -> "WeightCache":
    """Vibevoice default weight-cache resolution, shared by ``from_checkpoint`` and the tests.

    Enabled by default for every caller; ``VV_DISABLE_WEIGHT_CACHE=1`` turns it off. The directory
    is ``weight_cache_dir`` → ``$VV_WEIGHT_CACHE_DIR`` → ``$TT_CACHE_PATH/vibevoice/weight_cache`` →
    ``generated/ttnn/vibevoice/weight_cache``. The cache key folds in the checkpoint identity and the
    weight-value-affecting flags (``VV_FUSED_ROPE`` → ``rope{0,1}``, ``VV_POST_SCALE_FOLD`` →
    ``fold{0,1}``). Pass ``submodule`` to get the namespaced child (``"lm"``, ``"acoustic_tokenizer"``,
    …) matching what ``from_checkpoint`` uses, so component tests share its cache files.
    """
    if os.environ.get("VV_DISABLE_WEIGHT_CACHE", "").lower() in ("1", "true", "yes"):
        return WeightCache(None, enabled=False)
    cache_dir = weight_cache_dir or os.environ.get("VV_WEIGHT_CACHE_DIR") or default_weight_cache_root()
    rope = 1 if os.environ.get("VV_FUSED_ROPE", "0") == "1" else 0
    fold = 1 if os.environ.get("VV_POST_SCALE_FOLD", "") == "1" else 0
    wc = WeightCache.for_checkpoint(cache_dir, model_path, enabled=True, variant=f"rope{rope}_fold{fold}")
    return wc.child(submodule) if submodule else wc


def default_weight_cache_root() -> str:
    """tt-metal-standard root for tiled weight caches.

    ``$TT_CACHE_PATH`` if set, else ``generated/ttnn`` — the same convention used by
    ``tt_transformers`` and the Devstral-2 port (``generated/ttnn/<model>/weight_cache``),
    so a repo clean wipes it and ``TT_CACHE_PATH`` persists it across builds.
    """
    root = os.environ.get("TT_CACHE_PATH") or os.path.join("generated", "ttnn")
    return os.path.join(root, "vibevoice", "weight_cache")


class WeightCache:
    """Namespaced, checkpoint-keyed cache of preprocessed device weights.

    A single instance is threaded from ``from_checkpoint`` down through the
    ``preprocess_*`` functions.  A *disabled* instance (``enabled=False`` or
    ``cache_dir=None``) is a transparent pass-through: it builds and uploads exactly
    as the uncached code did, so callers can always pass one unconditionally.
    """

    def __init__(self, cache_dir: Optional[str], enabled: bool = True, prefix: str = ""):
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.enabled = bool(enabled) and self.cache_dir is not None
        self.prefix = prefix
        self.hits = 0
        self.misses = 0
        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

    # ── construction helpers ──────────────────────────────────────────────
    @classmethod
    def for_checkpoint(
        cls,
        cache_dir: Optional[str],
        model_path: str,
        enabled: bool = True,
        variant: str = "",
    ) -> "WeightCache":
        """Build a cache rooted at a subdirectory unique to this checkpoint (+ variant).

        The subdirectory name folds in the checkpoint's shard filenames + sizes + mtimes,
        the cache-format version, and ``variant`` — a string identifying any *weight-value*
        affecting options (e.g. RoPE permutation, scale folding). A different or updated
        checkpoint, or a different variant, never reuses another's cached tensors.
        """
        if cache_dir is None or not enabled:
            return cls(None, enabled=False)
        sig = _checkpoint_signature(model_path)
        name = f"{_CACHE_FORMAT_VERSION}_{sig}"
        if variant:
            name = f"{name}_{variant}"
        root = os.path.join(str(cache_dir), name)
        return cls(root, enabled=True)

    def child(self, prefix: str) -> "WeightCache":
        """Return a view whose keys are prefixed with ``prefix.`` (same cache dir).

        The child starts its own counters at zero; each hit/miss also propagates up to this
        parent (see ``_bump``), so the root accumulates the total across all children.
        """
        wc = WeightCache(self.cache_dir, enabled=self.enabled, prefix=self._join(prefix))
        wc._parent = self
        return wc

    # ── the one operation callers use ─────────────────────────────────────
    def as_tensor(
        self,
        name: str,
        build: Callable[[], "object"],
        *,
        dtype,
        layout,
        memory_config,
        device,
    ) -> ttnn.Tensor:
        """Return a device tensor for ``name``.

        On a cache hit the pre-tiled flatbuffer is loaded straight to ``device`` and
        ``build`` is never called (the checkpoint is not touched).  On a miss ``build``
        is invoked to produce the host torch tensor, which is uploaded and dumped.
        """
        if not self.enabled:
            return ttnn.as_tensor(build(), dtype=dtype, layout=layout, memory_config=memory_config, device=device)

        base = os.path.join(self.cache_dir, self._join(name))
        final = f"{base}_dtype_{dtype.name}_layout_{layout.name}.tensorbin"
        if os.path.isfile(final):
            try:
                t = ttnn.load_tensor(final, device=device)
                self._bump("hits")
                if memory_config is not None:
                    t = ttnn.to_memory_config(t, memory_config)
                return t
            except RuntimeError:
                # Corrupt/partial cache entry — rebuild and overwrite below.
                pass

        t = ttnn.as_tensor(
            build(),
            dtype=dtype,
            layout=layout,
            memory_config=memory_config,
            device=device,
            cache_file_name=base,
        )
        self._bump("misses")
        return t

    # ── internals ─────────────────────────────────────────────────────────
    def _join(self, name: str) -> str:
        return f"{self.prefix}.{name}" if self.prefix else name

    def _bump(self, which: str) -> None:
        node = self
        while node is not None:
            setattr(node, which, getattr(node, which) + 1)
            node = getattr(node, "_parent", None)

    _parent: Optional["WeightCache"] = None

    def summary(self) -> str:
        return f"{self.hits} hit / {self.misses} miss"


def _checkpoint_signature(model_path: str) -> str:
    """Short stable hash of the checkpoint's safetensors shards (name, size, mtime)."""
    p = Path(model_path)
    parts = []
    if p.is_dir():
        shards = sorted(f for f in os.listdir(p) if f.endswith(_SHARD_SUFFIX))
        for f in shards:
            st = (p / f).stat()
            parts.append(f"{f}:{st.st_size}:{int(st.st_mtime)}")
    if not parts:
        # Fall back to the resolved path so distinct checkpoints still key apart.
        parts.append(str(p.resolve()))
    digest = hashlib.sha1("|".join(parts).encode()).hexdigest()
    return digest[:16]
