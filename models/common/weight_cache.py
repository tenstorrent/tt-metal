# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared, ModelArgs-agnostic warm ttnn weight-cache helpers (generalizes PR #50550 / #48531
to forked model loaders — issue #45400 follow-up).

On a warm cache, a model can be built from its on-disk ``.tensorbin`` files without the
expensive host-side HF ``from_pretrained`` load (the load that OOMs/hangs during prefill,
#48509): ``ttnn.as_tensor(torch_weight, cache_file_name=...)`` loads the cached tensor and
ignores ``torch_weight`` on a cache hit (see ttnn/operations/core.py). So most weights only
need a dataless placeholder (``torch.empty`` of the right shape/dtype) to satisfy the modules'
host-side reshape ops before ``as_tensor``.

Some forks (e.g. gemma4) additionally consume a *small* set of weights on the host — token
embeddings used via ``F.embedding``, per-layer scalars read via ``.item()``, etc. Those must be
real. ``mark_weight_cache_complete`` persists exactly those tensors to a sidecar at cold-build
time (write-access run), and ``build_cached_state_dict`` serves them real on later warm runs
while placeholdering the rest — a HYBRID state_dict. The host subset is a tiny fraction of the
weight bytes, so the full from_pretrained (and its OOM) is still avoided.
"""

import collections.abc
import json
import os
from pathlib import Path

import torch
from loguru import logger

WEIGHT_CACHE_MARKER = ".weights_complete"
HOST_WEIGHTS_SIDECAR = ".host_weights.pt"
# Bump when the set/naming/layout of cached weights, or this marker schema, changes such that an
# existing cache would not satisfy a new build. Kept at 2 to stay compatible with the markers
# PR #50550 already writes for the tt_transformers models (same required fields).
WEIGHT_CACHE_FORMAT_VERSION = 2

DEFAULT_FORCE_ENV = "TT_TRANSFORMERS_FORCE_MODEL_LOAD"


def _dtype_from_str(s):
    return getattr(torch, s.rsplit(".", 1)[-1])


def weight_cache_is_complete(cache_path, *, model_name, n_layers, mesh_shape, force_env=DEFAULT_FORCE_ENV):
    """True when the on-disk ttnn weight cache at ``cache_path`` was fully built by a previous run
    for this (model_name, n_layers, mesh_shape) and carries a weight manifest (and, if it declared
    host weights, the sidecar holding them). ``force_env=...=1`` forces a cold load."""
    if force_env and os.getenv(force_env) == "1":
        return False
    cache_path = Path(cache_path)
    marker = cache_path / WEIGHT_CACHE_MARKER
    if not marker.is_file():
        return False
    try:
        meta = json.loads(marker.read_text())
    except (ValueError, OSError):
        return False
    if meta.get("format_version") != WEIGHT_CACHE_FORMAT_VERSION:
        return False
    if meta.get("model_name") != model_name or meta.get("n_layers") != n_layers:
        return False
    if meta.get("mesh_shape") != str(mesh_shape):
        return False
    if not meta.get("weights"):
        return False
    # If host weights were captured, the sidecar must still be present to rebuild them.
    if meta.get("host_weights") and not (cache_path / HOST_WEIGHTS_SIDECAR).is_file():
        return False
    return any(cache_path.glob("*.tensorbin"))


def mark_weight_cache_complete(
    cache_path, state_dict, *, model_name, n_layers, mesh_shape, is_moe=False, is_host_weight=None
):
    """Record that the ttnn weight cache at ``cache_path`` is fully built.

    Writes a ``.weights_complete`` marker holding a ``{key: [shape, dtype]}`` manifest of every
    weight. If ``is_host_weight(key)`` is provided, the (real) tensors it matches are also saved
    to a ``.host_weights.pt`` sidecar so a later warm run can serve them for real (hybrid)."""
    cache_path = Path(cache_path)
    marker = cache_path / WEIGHT_CACHE_MARKER
    weights = {}
    host = {}
    for k, v in state_dict.items():
        shape = getattr(v, "shape", None)
        dt = getattr(v, "dtype", None)
        if shape is None or dt is None:
            continue  # skip non-tensor entries
        weights[k] = [list(shape), str(dt)]
        if is_host_weight is not None and is_host_weight(k):
            host[k] = v
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        if host:
            torch.save(host, cache_path / HOST_WEIGHTS_SIDECAR)
        marker.write_text(
            json.dumps(
                {
                    "format_version": WEIGHT_CACHE_FORMAT_VERSION,
                    "model_name": model_name,
                    "n_layers": n_layers,
                    "mesh_shape": str(mesh_shape),
                    "is_moe": bool(is_moe),
                    "host_weights": sorted(host.keys()),
                    "weights": weights,
                }
            )
        )
        logger.info(
            f"Marked ttnn weight cache complete: {marker} ({len(weights)} weights, {len(host)} host-loaded)"
        )
    except OSError as e:
        logger.warning(f"Could not write weight-cache completion marker {marker}: {e}")


class CachedStateDict(collections.abc.MutableMapping):
    """A stand-in ``state_dict`` for warm-cache builds.

    Serves the real tensor for keys captured in the host-weights sidecar; for every other key it
    returns a fresh dataless ``torch.empty`` of the manifest shape/dtype (which ``ttnn.as_tensor``
    discards on the guaranteed cache hit). Mutable (some loaders ``setdefault`` missing KV-shared
    weights) and truthy (some loaders gate real-weight loading on ``if state_dict:``)."""

    def __init__(self, manifest, host):
        self._manifest = manifest  # key -> (shape, dtype_str)
        self._host = dict(host or {})  # key -> real torch.Tensor
        self._overrides = {}  # keys set by the caller at build time
        self._deleted = set()

    def __getitem__(self, key):
        if key in self._deleted:
            raise KeyError(key)
        if key in self._overrides:
            return self._overrides[key]
        if key in self._host:
            return self._host[key]
        spec = self._manifest.get(key)
        if spec is None:
            raise KeyError(key)
        shape, dt = spec
        return torch.empty(tuple(shape), dtype=_dtype_from_str(dt))

    def __setitem__(self, key, value):
        self._deleted.discard(key)
        self._overrides[key] = value

    def __delitem__(self, key):
        if key not in self:
            raise KeyError(key)
        self._overrides.pop(key, None)
        if key in self._host or key in self._manifest:
            self._deleted.add(key)

    def __iter__(self):
        seen = set()
        for k in list(self._overrides) + list(self._host) + list(self._manifest):
            if k in self._deleted or k in seen:
                continue
            seen.add(k)
            yield k

    def __len__(self):
        return sum(1 for _ in self)


def build_cached_state_dict(cache_path):
    """Build the warm-cache stand-in ``state_dict`` from the marker manifest + host sidecar."""
    cache_path = Path(cache_path)
    meta = json.loads((cache_path / WEIGHT_CACHE_MARKER).read_text())
    manifest = meta["weights"]
    host = {}
    sidecar = cache_path / HOST_WEIGHTS_SIDECAR
    if sidecar.is_file():
        host = torch.load(sidecar, map_location="cpu", weights_only=True)
    logger.info(
        f"Warm ttnn weight cache: built state_dict for {len(manifest)} weights "
        f"({len(host)} real host weights, no full HF load)."
    )
    return CachedStateDict(manifest, host)
