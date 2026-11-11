# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Container-package boot for the resident DFlash2 adapter: build the resident engine's converted weights on first
start, so a fresh tt-model container needs no pre-built cache.

The engine reads per-chip BF4 / BF8 weight files from $GRES_CACHE (the `<engine package>/weights.py` layout, raw:
unfolded norms). ensure_resident_cache() makes sure that directory is complete and was built for exactly these weights,
this converter and this precision before the adapter builds anything:
  - <cache>/COMPLETE.json holds the key (weights repo + revision, engine package, sha256 of the converter modules and of
    the whole engine package, the converter's per-entry dtypes, the declared precision, the converter arguments) and
    the file manifest (every file with its size and sha256);
  - a boot accepts the cache only if the marker parses, its key equals the current key, and the files on disk are
    exactly the manifest's: same set, regular files (no symlinks), same sizes, and the same sha256 too with
    GEMMA4_RESIDENT_CACHE_VERIFY=sha256;
  - anything else rebuilds: the stale marker is deleted first, then the directory, then the conversion runs in
    <cache>.partial, which gets its marker after an fsync of everything written and is renamed into place. A kill at
    any point leaves no directory a later boot accepts (a .partial is always discarded and rebuilt).
Package-only (changh95/gemma4-31b-it-dflash): the verified adapters are unchanged; the package's
vllm_models/resident_spec bundle names Gemma4ResidentSpecPackaged.
"""

import fcntl
import hashlib
import importlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger

from models.demos.gemma4_31b_qb2.tt.generator_resident_spec_vllm import Gemma4ResidentSpecForCausalLM

MARKER = "COMPLETE.json"
FORMAT = 1
# The converter: the engine package's modules that decide what a cache file holds (weights.py writes them, common.py
# defines ENTRIES, the bank layouts and the quantization dtypes).
CONVERTER_MODULES = ("weights.py", "common.py")
ENGINE_SOURCES = (".py", ".cpp", ".hpp", ".h")
# The converter call: raw caches (norms applied on the slices, every BF4 / BF8 block equal to the ttnn path's),
# converted from the checkpoint alone (no `reuse` links into another cache).
CONVERT_ARGS = {"fold": False, "head": True, "embed": True, "reuse": None}
PRECISIONS = {"bf4": "bf4", "bf8": "bf8"}  # declared label -> the dtype every projection entry must have


def _sha256(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def _tree_sha256(root, suffixes):
    """One hash over the relative paths and contents of root's files with these suffixes (no __pycache__)."""
    h = hashlib.sha256()
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file() and p.suffix in suffixes):
        rel = path.relative_to(root)
        if "__pycache__" in rel.parts:
            continue
        h.update(f"{rel.as_posix()}\0{_sha256(path)}\n".encode())
    return h.hexdigest()


def weights_identity(checkpoint):
    """(repo, revision) of an HF hub snapshot directory (.../models--org--name/snapshots/<sha>), else the path."""
    path = Path(checkpoint)
    if path.parent.name == "snapshots" and path.parent.parent.name.startswith("models--"):
        return path.parent.parent.name[len("models--") :].replace("--", "/"), path.name
    return "local", f"path:{path.resolve()}"


def cache_key(package, checkpoint, precision):
    """Everything a converted cache depends on. Equal keys = interchangeable caches."""
    pkg = importlib.import_module(f"models.demos.gemma4_31b_qb2.{package}")
    common = importlib.import_module(f"models.demos.gemma4_31b_qb2.{package}.common")
    pkg_dir = Path(pkg.__file__).parent
    entry_dtypes = {name: dtype for name, _, dtype, _ in common.ENTRIES}
    wanted = PRECISIONS.get(precision)
    if wanted is None:
        raise ValueError(f"GEMMA4_RESIDENT_PRECISION={precision!r}: expected one of {sorted(PRECISIONS)}")
    projections = {name: dtype for name, dtype in entry_dtypes.items() if name != "head"}
    if set(projections.values()) != {wanted}:
        raise ValueError(f"Declared precision {precision} but {package}'s converter writes {projections}")
    repo, revision = weights_identity(checkpoint)
    h = hashlib.sha256()
    for name in CONVERTER_MODULES:
        h.update(f"{name}\0{_sha256(pkg_dir / name)}\n".encode())
    return {
        "format": FORMAT,
        "weights_repo": repo,
        "weights_revision": revision,
        "engine_package": package,
        "converter_sha256": h.hexdigest(),
        "engine_sha256": _tree_sha256(pkg_dir, ENGINE_SOURCES),
        "precision": precision,
        "entry_dtypes": entry_dtypes,
        "layers": common.LAYERS,
        "convert_args": CONVERT_ARGS,
    }


def _files(root):
    """{relative path: lstat} of everything under root except the marker; directories are not entries."""
    out = {}
    for dirpath, _, names in os.walk(root):
        for name in names:
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if rel != MARKER:
                out[rel] = path.lstat()
    return out


def file_manifest(root, workers=8):
    """[[relative path, size, sha256], ...] sorted, for every file under root but the marker."""
    entries = _files(root)
    if any(not os.path.isfile(Path(root) / rel) or os.path.islink(Path(root) / rel) for rel in entries):
        raise ValueError(f"{root} holds a symlink or special file; the cache must hold regular files only")
    rels = sorted(entries)
    with ThreadPoolExecutor(workers) as pool:
        digests = list(pool.map(lambda rel: _sha256(Path(root) / rel), rels))
    return [[rel, entries[rel].st_size, d] for rel, d in zip(rels, digests)]


def check_cache(cache, key, verify="sizes"):
    """None if the cache at `cache` is complete and built for `key`, else why not (for the log)."""
    cache = Path(cache)
    if not cache.exists():
        return "absent"
    marker = cache / MARKER
    if not marker.is_file():
        return f"no {MARKER} (an unfinished or foreign directory)"
    try:
        record = json.loads(marker.read_text())
        recorded_key, files = record["key"], record["files"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        return f"unreadable {MARKER} ({e!r})"
    if recorded_key != key:
        fields = sorted(k for k in set(key) | set(recorded_key) if key.get(k) != recorded_key.get(k))
        return f"built for another key (differs in {', '.join(fields)})"
    on_disk = _files(cache)
    listed = {rel: (size, digest) for rel, size, digest in files}
    if set(on_disk) != set(listed):
        missing, extra = sorted(set(listed) - set(on_disk)), sorted(set(on_disk) - set(listed))
        return f"files differ from the manifest ({len(missing)} missing, {len(extra)} extra: {(missing + extra)[:3]})"
    for rel, st in on_disk.items():
        if not os.path.isfile(cache / rel) or os.path.islink(cache / rel):
            return f"{rel} is not a regular file"
        if st.st_size != listed[rel][0]:
            return f"{rel} has {st.st_size} bytes, the manifest {listed[rel][0]}"
    if verify == "sha256":
        bad = [rel for rel, _, digest in file_manifest(cache) if digest != listed[rel][1]]
        if bad:
            return f"{len(bad)} files differ from their manifest sha256 ({bad[:3]})"
    return None


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def discard(cache):
    """Remove a cache a boot rejected: its marker first (so a kill mid-removal leaves an unaccepted directory)."""
    cache = Path(cache)
    if cache.is_symlink():  # never touch what a symlinked cache points at
        cache.unlink()
        return
    if not cache.exists():
        return
    marker = cache / MARKER
    if marker.exists() or marker.is_symlink():
        marker.unlink()
        _fsync_dir(cache)
    shutil.rmtree(cache)


def _write_marker(directory, record):
    tmp = Path(directory) / (MARKER + ".tmp")
    with open(tmp, "w") as f:
        json.dump(record, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, Path(directory) / MARKER)
    _fsync_dir(directory)


def ensure_resident_cache(cache=None, package=None, precision=None, verify=None, convert=None):
    """Make $GRES_CACHE a complete cache for $HF_MODEL, the engine package and the declared precision; build it if not.

    Returns the cache path. `convert` replaces the engine package's weights.convert (tests)."""
    cache = Path(cache or os.environ.get("GRES_CACHE") or "")
    if not cache.name:
        raise ValueError("GRES_CACHE must name the resident weight cache directory")
    package = package or os.environ.get("GEMMA4_RESIDENT_ENGINE_PACKAGE")
    if not package:
        raise ValueError("GEMMA4_RESIDENT_ENGINE_PACKAGE must name the resident engine package")
    precision = precision or os.environ.get("GEMMA4_RESIDENT_PRECISION", "bf4")
    verify = verify or os.environ.get("GEMMA4_RESIDENT_CACHE_VERIFY", "sizes")
    checkpoint = os.environ.get("HF_MODEL", "")
    if not os.path.isfile(os.path.join(checkpoint, "model.safetensors.index.json")):
        raise ValueError(f"HF_MODEL must name the checkpoint directory the cache is converted from, got {checkpoint!r}")
    key = cache_key(package, checkpoint, precision)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache.with_name(cache.name + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        problem = check_cache(cache, key, verify)
        if problem is None:
            logger.info("Resident weight cache {}: complete for {}@{} ({}, {})", cache, key["weights_repo"],
                        key["weights_revision"][:12], package, precision)
            return cache
        logger.info("Resident weight cache {}: {}; converting {} into it ({}, {})", cache, problem, checkpoint, package,
                    precision)
        discard(cache)
        partial = cache.with_name(cache.name + ".partial")
        if partial.is_symlink():
            partial.unlink()
        elif partial.exists():
            shutil.rmtree(partial)
        if convert is None:
            convert = importlib.import_module(f"models.demos.gemma4_31b_qb2.{package}.weights").convert
        t0 = time.time()
        convert(partial, **CONVERT_ARGS)
        converted = time.time() - t0
        os.sync()  # every converted byte on disk before the marker that vouches for it
        files = file_manifest(partial)
        _write_marker(partial, {"key": key, "files": files, "convert_seconds": round(converted, 1),
                                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        os.rename(partial, cache)
        _fsync_dir(cache.parent)
        logger.info("Resident weight cache {}: converted in {:.0f} s ({} files, {:.1f} GB), manifest in {:.0f} s", cache,
                    converted, len(files), sum(size for _, size, _ in files) / 1e9, time.time() - t0 - converted)
    return cache


class Gemma4ResidentSpecPackaged(Gemma4ResidentSpecForCausalLM):
    """Gemma4ResidentSpecForCausalLM that first makes sure the resident weight cache exists (the package's class)."""

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, **kwargs):
        ensure_resident_cache()
        return super().initialize_vllm_model(hf_config, mesh_device, max_batch_size, **kwargs)
