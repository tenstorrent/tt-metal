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
import hashlib
import json
import os
from pathlib import Path

import torch
from loguru import logger

WEIGHT_CACHE_MARKER = ".weights_complete"
HOST_WEIGHTS_SIDECAR = ".host_weights.pt"
# Bump when the set/naming/layout of cached weights, or this marker schema, changes such that an
# existing cache would not satisfy a new build. A marker written by an older format is rejected,
# so the run cold-loads and regenerates rather than building from an incompatible cache.
#   v2: model/n_layers/mesh_shape validation + a {key: [shape, dtype]} manifest.
#   v3: canonical mesh_shape encoding shared with ModelArgs (the two writers previously encoded it
#       differently and each rejected the other's marker), a `components` field so a text-only seed
#       cannot certify a cache for a build that also needs the vision tower, and `cache_files` --
#       the recursive list of .tensorbin files the completed build actually produced, verified
#       per-file on read. That last one is load-bearing: ttnn.as_tensor PERSISTS whatever tensor it
#       is handed on a cache miss, so a marker that outlives some of its tensorbins would otherwise
#       dump placeholders to disk as real cache entries -- silent, permanent corruption. Verifying
#       the recorded file set turns every such case back into a plain cold load. Also `build_variant`
#       -- the build options (prefetcher, precision) that change an as_tensor cache FILENAME, matched
#       exactly, because a different variant needs different files rather than fewer.
WEIGHT_CACHE_FORMAT_VERSION = 3

DEFAULT_FORCE_ENV = "TT_TRANSFORMERS_FORCE_MODEL_LOAD"


def _variant_digest(build_variant):
    """Stable short digest of a build_variant dict ("none" for None)."""
    if build_variant is None:
        return "none"
    return hashlib.sha1(json.dumps(build_variant, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _variant_unverifiable(build_variant):
    return bool(build_variant) and bool(build_variant.get("unverifiable"))


def marker_path(cache_path, build_variant=None):
    """The marker file for one (cache dir, build variant).

    The variant digest is part of the FILENAME, not just a field compared inside one shared
    marker. A cache dir legitimately serves several build variants (the Llama CI job runs
    eval-32 with and without the DRAM prefetcher against the same instruct cache), and a single
    marker matched exactly would make each variant's seed evict the other's on every run -- both
    then cold-load forever with nothing going red. One marker per variant lets them coexist.
    (#45400 review, finding B3)"""
    return Path(cache_path) / f"{WEIGHT_CACHE_MARKER}.{_variant_digest(build_variant)}"


def _dtype_from_str(s):
    return getattr(torch, s.rsplit(".", 1)[-1])


def normalize_mesh_shape(mesh_shape):
    """Canonical marker encoding for a mesh shape.

    ``ttnn.MeshShape`` stringifies as ``MeshShape([1, 8])`` while callers that pass a plain tuple
    stringify as ``(1, 8)``. Both writers must agree or each rejects the other's marker and the
    model cold-loads forever (gemma3 inherits ModelArgs but its demos call this module). Normalize
    everything to a plain tuple-of-ints string."""
    try:
        return str(tuple(int(d) for d in mesh_shape))
    except TypeError:
        return str(mesh_shape)


def _normalize_components(components):
    """Canonical component list. ``None`` means "the whole model as this loader builds it" and is
    encoded as a single implicit component so old-style callers stay self-consistent."""
    if components is None:
        return ["all"]
    if isinstance(components, str):
        return [components]
    return sorted(str(c) for c in components)


def list_cache_files(cache_path):
    """Every ``.tensorbin`` under ``cache_path``, recursively, as sorted relative POSIX paths.

    Recursive because forked loaders nest per-layer weights in subdirectories (qwen36
    ``layers.{n}/``, gemma4 ``layer_{i}/``); a top-level ``glob`` would call a cache complete when
    only the root-level ``output.weight`` survived an interrupted seed."""
    cache_path = Path(cache_path)
    return sorted(p.relative_to(cache_path).as_posix() for p in cache_path.rglob("*.tensorbin"))


# One-entry cache so the completeness gate's validation load is reused by
# build_cached_state_dict instead of torch.load-ing the same multi-GB file twice per warm run
# (gemma-4-31b's embedding alone is ~2.8 GB). Keyed on (path, mtime, size) so a republished
# sidecar is never served stale; the builder consumes the entry so the tensors are not pinned
# past the build. (#45400 review, finding R1)
_SIDECAR_CACHE = {}


def load_host_sidecar(cache_path, *, consume=False):
    """Load the host-weights sidecar, or None if absent/unreadable.

    ``consume=True`` drops the memoized entry after returning it (the caller takes ownership)."""
    sidecar = Path(cache_path) / HOST_WEIGHTS_SIDECAR
    if not sidecar.is_file():
        return None
    try:
        st = sidecar.stat()
        key = (str(sidecar), st.st_mtime_ns, st.st_size)
        host = _SIDECAR_CACHE.get(key)
        if host is None:
            host = torch.load(sidecar, map_location="cpu", weights_only=True)
            _SIDECAR_CACHE.clear()
            _SIDECAR_CACHE[key] = host
        if consume:
            _SIDECAR_CACHE.pop(key, None)
        return host
    except Exception:
        return None


def weight_cache_is_complete(
    cache_path,
    *,
    model_name,
    n_layers,
    mesh_shape,
    components=None,
    build_variant=None,
    force_env=DEFAULT_FORCE_ENV,
):
    """True when the on-disk ttnn weight cache at ``cache_path`` was fully built by a previous run
    for this exact build, and every tensorbin that build produced is still present.

    ``components`` names the model parts this build will construct (e.g. ``"text"`` vs
    ``"text+vision"``); a marker written by a narrower build does not satisfy a wider one, because
    the wider build needs tensorbins the narrower one never wrote. ``force_env=...=1`` forces a
    cold load."""
    if force_env and os.getenv(force_env) == "1":
        return False
    # A variant we could not compute is a variant we cannot verify: accepting it could hand a
    # placeholder to a build whose cache-filename set we did not check, and as_tensor would
    # persist that placeholder to disk. Fail closed, loudly. (#45400 review, finding R3)
    if _variant_unverifiable(build_variant):
        logger.warning(
            f"Warm-cache check for {cache_path}: build_variant could not be computed "
            f"({build_variant.get('error', 'unknown error')}); forcing a cold load."
        )
        return False
    cache_path = Path(cache_path)
    marker = marker_path(cache_path, build_variant)
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
    if meta.get("mesh_shape") != normalize_mesh_shape(mesh_shape):
        return False
    # The recorded build must cover every component this build needs. Superset is fine (a
    # text+vision seed wrote the text tensorbins too, so it satisfies a text-only build); a subset
    # is not (a text-only seed never wrote the vision tower's tensorbins, and accepting it would
    # make as_tensor dump placeholders for them).
    if not set(_normalize_components(components)).issubset(set(meta.get("components") or [])):
        return False
    # Build options that change an as_tensor cache FILENAME (prefetcher, precision) must match
    # exactly. A superset rule is wrong here: a different variant does not need fewer files, it
    # needs DIFFERENT ones, and any it is missing would be regenerated from the placeholder.
    if meta.get("build_variant") != build_variant:
        return False
    if not meta.get("weights"):
        return False
    # Every tensorbin the completed build produced must still be on disk. Any missing file would
    # otherwise be regenerated by as_tensor FROM THE PLACEHOLDER we are about to hand it, writing
    # garbage into the cache permanently. Missing file => cold load, which rebuilds it correctly.
    recorded = meta.get("cache_files")
    if not recorded:
        return False
    present = set(list_cache_files(cache_path))
    if not all(f in present for f in recorded):
        return False
    # If host weights were captured, the sidecar must be present AND loadable. A torn/corrupt
    # sidecar (interrupted or racing seed) must fall back to a cold load -- the way a torn marker
    # already does via the except above -- rather than pass this gate and then crash torch.load on
    # every subsequent run, bricking the cache dir. Checked LAST so the load it performs is
    # memoized only when the gate is about to pass, for build_cached_state_dict to consume.
    # (#45400 review)
    if meta.get("host_weights") and load_host_sidecar(cache_path) is None:
        return False
    return True


def mark_weight_cache_complete(
    cache_path,
    state_dict,
    *,
    model_name,
    n_layers,
    mesh_shape,
    components=None,
    build_variant=None,
    is_moe=False,
    is_host_weight=None,
):
    """Record that the ttnn weight cache at ``cache_path`` is fully built.

    Writes a ``.weights_complete`` marker holding a ``{key: [shape, dtype]}`` manifest of every
    weight plus the recursive list of ``.tensorbin`` files this build produced (verified per-file
    on read). If ``is_host_weight(key)`` is provided, the (real) tensors it matches are also saved
    to a ``.host_weights.pt`` sidecar so a later warm run can serve them for real (hybrid).

    Call this only AFTER the model has been constructed, so the tensorbins exist to be recorded."""
    if _variant_unverifiable(build_variant):
        # Never certify a cache under an identity we could not compute -- a later run computing
        # the same error string would otherwise warm-match it. (#45400 review, finding R3)
        logger.warning(
            f"Not marking weight cache complete at {cache_path}: build_variant could not be "
            f"computed ({build_variant.get('error', 'unknown error')})."
        )
        return
    cache_path = Path(cache_path)
    marker = marker_path(cache_path, build_variant)
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
        cache_files = list_cache_files(cache_path)
        if not cache_files:
            logger.warning(f"Not marking weight cache complete: no .tensorbin files under {cache_path}")
            return
        # Write both the sidecar and the marker atomically (temp file + os.replace, atomic on
        # POSIX). Two jobs can seed the same (model, dtype, mesh) dir on one host concurrently, and
        # an interrupted write must never leave a torn file that a later run picks up: a half-written
        # sidecar would otherwise pass the is_file() gate and crash torch.load on every subsequent
        # run. The temp name is pid-unique so two concurrent seeders cannot write the SAME temp
        # inode -- with a fixed name, B could publish the file while A was still writing into it.
        # Sidecar first, then marker, so the completeness gate only appears once its sidecar is
        # fully in place. (#45400 review)
        uniq = os.getpid()
        if host:
            sidecar = cache_path / HOST_WEIGHTS_SIDECAR
            sidecar_tmp = sidecar.with_suffix(sidecar.suffix + f".tmp.{uniq}")
            torch.save(host, sidecar_tmp)
            os.replace(sidecar_tmp, sidecar)
        marker_body = json.dumps(
            {
                "format_version": WEIGHT_CACHE_FORMAT_VERSION,
                "model_name": model_name,
                "n_layers": n_layers,
                "mesh_shape": normalize_mesh_shape(mesh_shape),
                "components": _normalize_components(components),
                "build_variant": build_variant,
                "cache_files": cache_files,
                "is_moe": bool(is_moe),
                "host_weights": sorted(host.keys()),
                "weights": weights,
            }
        )
        marker_tmp = marker.with_suffix(marker.suffix + f".tmp.{uniq}")
        marker_tmp.write_text(marker_body)
        os.replace(marker_tmp, marker)
        logger.info(f"Marked ttnn weight cache complete: {marker} ({len(weights)} weights, {len(host)} host-loaded)")
    except Exception as e:
        # Deliberately broad: this function only RECORDS completion -- failing to record must
        # never kill a build that already succeeded. The concrete case: on a read-only
        # /mnt/MLPerf, torch.save of the host sidecar raises RuntimeError from torch's C++
        # serializer (inline_container.cc "Read-only file system"), not OSError, and the narrow
        # except crashed every read-only cold run of the sidecar models (gemma4/gemma3) right
        # after a successful build. (#45400 review, finding R5; seen on Gemma-4-E2B bh_p150,
        # run 32511945147)
        logger.warning(f"Could not write weight-cache completion marker {marker}: {e}")


class CachedStateDict(collections.abc.MutableMapping):
    """A stand-in ``state_dict`` for warm-cache builds.

    Serves the real tensor for keys captured in the host-weights sidecar; for every other key it
    returns a fresh dataless ``torch.empty`` of the manifest shape/dtype (which ``ttnn.as_tensor``
    discards on the guaranteed cache hit). Mutable (some loaders ``setdefault`` missing KV-shared
    weights) and truthy (some loaders gate real-weight loading on ``if state_dict:``)."""

    # Explicit marker that this is a warm-cache stand-in, NOT real weights. Callers that must tell
    # "warm-cache placeholder" apart from "real weights" MUST branch on this attribute, never on
    # truthiness: this mapping is truthy (non-zero __len__) but tt_transformers' _PlaceholderStateDict
    # is falsy (__bool__ -> False), so a truthiness test silently means opposite things for the two.
    # If tt_transformers is ever collapsed onto this class (a listed follow-up), the attribute keeps
    # `if is_placeholder(...)` reload sites (e.g. test_model_prefill) correct. (#45400 review)
    is_placeholder = True

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

    # Mapping's default __contains__/get/items route through __getitem__, which allocates a
    # full-size torch.empty for EVERY key touched -- including multi-GB ones like lm_head.weight.
    # substate() (models/tt_dit/utils/substate.py) iterates .items() and filters by prefix, so a
    # 62-layer gemma4 build would allocate the entire model once per layer just to discard it.
    # Answer membership from the key sets, and make items() lazy so only matching keys materialize.
    def __contains__(self, key):
        if key in self._deleted:
            return False
        return key in self._overrides or key in self._host or key in self._manifest

    def keys(self):
        return list(self)

    def items(self):
        for k in self:
            yield k, self[k]

    def get(self, key, default=None):
        if key not in self:
            return default
        return self[key]


def build_cached_state_dict(cache_path, host=None, args=None, build_variant=None):
    """Build the warm-cache stand-in ``state_dict`` from the marker manifest + host sidecar.

    ``host`` may be a sidecar dict already loaded by ``weight_cache_is_complete``'s validation, to
    avoid a second multi-GB ``torch.load`` of the same file on every warm run (gemma-4-31b's
    embedding alone is ~2.8 GB).

    ``args`` (a ModelArgs-like) has ``is_mixture_of_experts`` restored from the marker. That flag is
    normally set as a side effect of ``load_state_dict`` (by sniffing for ``.experts.`` keys), which
    the warm path skips -- so without this a MoE checkpoint would build a dense decoder and die on a
    missing ``feed_forward.w1.weight``. (#45400 review)"""
    cache_path = Path(cache_path)
    meta = json.loads(marker_path(cache_path, build_variant).read_text())
    manifest = meta["weights"]
    if args is not None and hasattr(args, "__dict__"):
        args.is_mixture_of_experts = bool(meta.get("is_moe", False))
        # fuse_qkv / fuse_mlp are normally sniffed from the checkpoint keys inside load_state_dict,
        # which the warm path skips -- leaving them at their __init__ defaults and silently changing
        # how the decoder is built. The manifest holds the same key set, so derive them identically.
        keys = manifest.keys()
        args.fuse_qkv = any("qkv" in k for k in keys)
        args.fuse_mlp = any("gate_up" in k for k in keys)
        if args.is_mixture_of_experts:
            args.moe = True
            expert_indices = [int(k[-11]) + 1 for k in keys if "block_sparse_moe.experts" in k]
            if expert_indices:
                args.num_experts = max(expert_indices)
            elif hasattr(args, "num_local_experts"):
                args.num_experts = args.num_local_experts
    if host is None and meta.get("host_weights"):
        # consume=True: reuse the load the completeness gate just performed and release the
        # memoized entry, so the sidecar is read from NAS once per warm run, not twice. (R1)
        host = load_host_sidecar(cache_path, consume=True)
    host = host or {}
    logger.info(
        f"Warm ttnn weight cache: built state_dict for {len(manifest)} weights "
        f"({len(host)} real host weights, no full HF load)."
    )
    return CachedStateDict(manifest, host)
