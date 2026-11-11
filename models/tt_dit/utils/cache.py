# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import shutil
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from loguru import logger

import ttnn

from ..layers.module import Module
from . import walltime

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

CACHE_DICT_FILE = "cache_dict.json"

# Manifest schema. Bump to make every content-keyed cache miss and rebuild.
CACHE_FORMAT = 1

# Bump when weight-prep code changes the VALUES it writes without changing any parameter's
# shape or dtype — e.g. GemmaFeatureExtractor's D-major→layer-major column permutation.
# Shape, dtype, layout and sharding changes need no bump: `module_signature` already sees them.
CACHE_VERSION = 1

# A cache hit up to this many bytes of tensorbins is scanned for NaN/inf before it is served
# (TT_DIT_CACHE_VERIFY=1): connectors and VAEs, not a transformer, whose publish-time verify covers it.
SCAN_MAX_BYTES_DEFAULT = 8 << 30
# A clean scan leaves this stamp, keyed on the tensorbins' names, sizes and mtimes, so the next process
# skips the scan (it costs ~90 s of a cs04 worker start) until a tensorbin is rewritten.
SCAN_OK_FILE = ".nonfinite-scan-ok"
_scanned_cache_dirs: set[str] = set()


class MissingCacheError(Exception):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __str__(self) -> str:
        return f"cache does not exist at '{self.path}'"


def config_id(parallel_config):
    config_id = ""
    for n, v in parallel_config._asdict().items():
        if v is not None:
            config_id += f"{''.join([w[0].upper() for w in n.split('_')])}{v.factor}_{v.mesh_axis}_"
    return config_id


def cache_dir_is_set() -> bool:
    return _cache_root() is not None


def source_id(path: str | Path) -> str:
    """Content identity of a source checkpoint, cheap enough to recompute every process start.

    A HuggingFace hub blob is stored under its own git-lfs sha256, so for the common case the
    resolved filename *is* an exact content hash and costs one `readlink` — no read of the 46GB
    file. Anything else falls back to size, mtime, ctime and inode. A re-download or a swapped
    file changes at least one; ctime closes the aliasing gap a plain size+mtime key leaves, since
    the kernel bumps it on any write and userspace cannot reset it, so even a metadata-preserving
    copy that restores size and mtime is still caught (its ctime and inode differ).
    """
    p = Path(path).resolve()
    st = p.stat()
    if p.parent.name == "blobs" and len(p.name) == 64 and all(c in "0123456789abcdef" for c in p.name):
        return f"sha256:{p.name}"
    return f"stat:{st.st_size}:{st.st_mtime_ns}:{st.st_ctime_ns}:{st.st_ino}"


def module_signature(module: Module) -> str:
    """Hash the structure of every parameter the cache will hold.

    Read off the built-but-unloaded module, so it costs no weight I/O. It covers exactly what
    determines a tensorbin's bytes — name, shape, dtype, layout, mesh sharding — which makes it
    the authority on weight dtype: a quant preset that only retunes compute (fidelity, SDPA
    inputs) leaves every `Parameter.dtype` alone and so keeps its cache, while one that restores
    a weight gets a new key. Value-only prep changes are invisible here; `CACHE_VERSION` covers
    those.
    """
    parts = [
        f"{name}|{p.total_shape}|{p.dtype}|{p.layout}|{p.mesh_axes}"
        for name, p in sorted(_walk_parameters(module), key=lambda kv: kv[0])
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def content_key(module: Module, sources: Sequence[str | Path] = ()) -> str:
    """Cache-key component binding an artifact to the weights and the code that produced it.

    Invalidates when the source checkpoint changes (`source_id`), when the module's weight
    structure or dtype changes (`module_signature`), or when weight-prep code changes values
    under a fixed structure (`CACHE_VERSION`). Every input is O(1) in the weight bytes.
    """
    h = hashlib.sha256(f"fmt{CACHE_FORMAT}|v{CACHE_VERSION}".encode())
    for s in sources:
        h.update(f"|src:{source_id(s)}".encode())
    h.update(f"|sig:{module_signature(module)}".encode())
    return h.hexdigest()[:12]


def _walk_parameters(module: Module, prefix: str = ""):
    for name, child in module.named_children():
        yield from _walk_parameters(child, f"{prefix}{name}.")
    for name, parameter in module.named_parameters():
        yield f"{prefix}{name}", parameter


def _kernel_capture_only() -> bool:
    """True during a kernel-prewarm capture pass, which records compile recipes with dispatch off.

    Gated on TT_METAL_KERNEL_CAPTURE_ONLY: any non-empty value enables it.
    """
    return os.environ.get("TT_METAL_KERNEL_CAPTURE_ONLY", "") != ""


def load_model(
    tt_model: Module,
    *,
    model_name: str,
    subfolder: str,
    parallel_config: NamedTuple,
    mesh_shape: Sequence[int],
    mesh_device: ttnn.MeshDevice,
    dtype: str = "bf16",
    is_fsdp: bool = False,
    sources: Sequence[str | Path] = (),
    get_torch_state_dict: Callable[[], dict] | None = None,
    create_cache: bool = True,
    post_load_hook: Callable[[Module], None] | None = None,
) -> None:
    """
    Load model weights from cache or PyTorch state dict.

    Attempts to load from cache first. If the cache does not exist, loads from PyTorch state dict
    (if provided) and optionally creates the cache. Raises `MissingCacheError` if neither is
    available. Finally, any module that needs to be offloaded is taken care of.

    Args:
        `tt_model`: TT model instance to load weights into.
        `model_name`: Model name (e.g., "flux1-dev", "stable-diffusion-3.5").
        `subfolder`: Subfolder within model cache directory (e.g., "transformer", "vae").
        `parallel_config`: Parallelism configuration (tensor/sequence parallel).
        `mesh_shape`: Device mesh shape.
        `mesh_device`: Mesh device used to derive the multi-host ownership cache suffix.
        `dtype`: Data type for cached weights (default: "bf16").
        `is_fsdp`: Whether FSDP is used (default: False).
        `sources`: Checkpoint files the weights are prepared from. Supplying them keys the cache
            on their content (see `content_key`) and turns on manifest verification, so an
            artifact built from other weights, other prep code or a torn write cannot be served.
            Omitting them keeps the legacy content-blind key and its existing on-disk caches.
        `get_torch_state_dict`: Optional callable returning PyTorch state dict. Enables lazy
            evaluation - PyTorch model only loads if the cache does not exist. If `None`, cache
            must exist or `MissingCacheError` is raised.
        `create_cache`: Create cache after loading from PyTorch (default: True).

    Raises:
        `MissingCacheError`: Cache does not exist and `get_torch_state_dict` is `None`.
        `RuntimeError`: `TT_DIT_CACHE_DIR` is not set and `get_torch_state_dict` is `None`.
    """
    if tt_model.is_loaded():
        return

    key = content_key(tt_model, sources) if sources else None

    key_args = {
        "model_name": model_name,
        "subfolder": subfolder,
        "parallel_config": parallel_config,
        "mesh_shape": mesh_shape,
        "mesh_device": mesh_device,
        "dtype": dtype,
        "is_fsdp": is_fsdp,
    }
    cache_dir = model_cache_dir(**key_args, content=key, required=get_torch_state_dict is None)

    if cache_dir is None:
        assert get_torch_state_dict is not None

        logger.info(
            "Loading transformer weights from PyTorch state dict. "
            "To use caching, set the TT_DIT_CACHE_DIR environment variable."
        )
        with walltime.timed("weight_load", f"{model_name}/{subfolder}", cached=False, detail="TT_DIT_CACHE_DIR unset"):
            tt_model.load_torch_state_dict(get_torch_state_dict())
        if post_load_hook is not None:
            post_load_hook(tt_model)
        ttnn.distributed_context_barrier()
        return

    if _cache_is_complete(cache_dir, tt_model, key) and not (
        get_torch_state_dict is not None and _set_aside_if_nonfinite(tt_model, cache_dir)
    ):
        logger.info(f"loading cache at '{cache_dir}'.")
        with walltime.timed("weight_load", f"{model_name}/{subfolder}", cached=True):
            tt_model.load(cache_dir)
        if post_load_hook is not None:
            post_load_hook(tt_model)
        ttnn.distributed_context_barrier()
        return

    if get_torch_state_dict is None:
        raise MissingCacheError(cache_dir)

    logger.info("Cache does not exist. Loading PyTorch state dict.")
    with walltime.timed("weight_load", f"{model_name}/{subfolder}", cached=False, detail=miss_reason(cache_dir)):
        tt_model.load_torch_state_dict(get_torch_state_dict())

    # Hook (e.g. quant typecast) must run BEFORE save so the cache holds the post-hook
    # weight dtype; otherwise a dynamic-load reload reads stale-dtype tensorbins into a
    # hook-mutated module and the dtype check fails.
    if post_load_hook is not None:
        post_load_hook(tt_model)

    # If distributed, ensure that all processes have completed the check whether cache_dir exists,
    # before any rank might proceed to create that dir to save.
    ttnn.distributed_context_barrier()

    if create_cache and _kernel_capture_only():
        # Weight prep runs on device, and this pass has dispatch disabled, so the tensors above hold
        # whatever the skipped ops left behind. Publishing them poisons the cache for every later run:
        # a content key would at least refuse the artifact, but the legacy content-blind entries have
        # no manifest to check it against, so the garbage is served silently. Re-prepping costs one
        # weight load; a bad cache costs every run until someone deletes it by hand.
        logger.warning(f"kernel capture-only pass: NOT writing weight cache to '{cache_dir}'")
    elif create_cache:
        if key is not None:
            _reclaim_superseded(model_cache_dir(**key_args, content=None, required=False))
        logger.info(f"Writing cache to '{cache_dir}'.")
        _publish_cache(tt_model, cache_dir, key)
        _place_like_a_cache_hit(tt_model, cache_dir, key, post_load_hook)


def _place_like_a_cache_hit(
    tt_model: Module, cache_dir: Path, key: str | None, post_load_hook: Callable[[Module], None] | None
) -> None:
    """After a miss has published the cache, reload the weights from it, so they sit on the device
    exactly where every later cache hit puts them.

    A miss builds the weights from the state dict, which allocates differently from a cache load. A
    trace captured on those weights bakes their addresses, and a dynamic-load mesh reloads the module
    from the cache before the next replay: the replay then reads memory the reload did not put there.
    On f07cs04 a job whose transformer missed in-gen captured its denoise traces on the state-dict
    weights, and the next job, reloaded from the cache, replayed them into pure noise (jobs 2749-2750,
    2026-10-18). Reloading here costs one cache read, on a miss only. A cache that was not published
    (capture-only pass, failed verify, no disk room) keeps the weights as built.
    """
    if not _cache_is_complete(cache_dir, tt_model, key):
        return
    tt_model.deallocate_weights()
    logger.info(f"reloading from the cache just written at '{cache_dir}', as a cache hit places it.")
    tt_model.load(cache_dir)
    if post_load_hook is not None:
        post_load_hook(tt_model)
    ttnn.distributed_context_barrier()


def _verification_parameters(model: Module, prefix: str):
    for name, child in model.named_children():
        yield from _verification_parameters(child, f"{prefix}{name}.")
    for name, parameter in model.named_parameters():
        yield f"{prefix}{name}", parameter


@contextmanager
def _verification_phase(stats: dict, phase: str):
    started = time.monotonic()
    try:
        yield
    finally:
        stats[phase] += time.monotonic() - started


def _verify_shard_pair(resident, fresh, stats: dict, *, native: bool = False) -> tuple[bool, bool]:
    # Workers use only the qualified host conversion; decorated serial conversions
    # and device readbacks stay on the caller. Release decoded storage on every exit.
    import torch

    a = b = None
    try:
        with _verification_phase(stats, "convert_s"):
            a = resident.to_torch() if native else ttnn.to_torch(resident)
            b = fresh.to_torch() if native else ttnn.to_torch(fresh)
        with _verification_phase(stats, "compare_s"):
            equal = a.shape == b.shape and torch.equal(a, b)
            finite = bool(torch.isfinite(b).all()) if equal else True
        return equal, finite
    finally:
        # An escaping exception's traceback must not retain decoded storage either.
        a = b = resident = fresh = None


_VERIFY_WORKSPACE_BYTES = 512 << 20
_VERIFY_TO_TORCH = ttnn.to_torch


@lru_cache(maxsize=1)
def _verification_native_build() -> bool:
    # This experiment is qualified only on these immutable native host converters.
    # Unknown builds keep the public decorated serial path; extending this list needs
    # native parity/concurrency evidence. Hash once, never once per parameter.
    expected = {
        "ttnn/ttnn/_ttnn.so": "8d3bb729319e9c2e5c26a97c7e5085c59b3a2c6faf92cd331e5b4cf664e4c879",
        "build/lib/_ttnncpp.so": "d2904ca48fbf9f897a7ba8f7f53bda96b63d2538fda4399a5b1c6a1da03b4e49",
        "build/lib/libtt_metal.so": "80ff3f818e086373bc3da70a425f60406cfa54ef2e308125f13268144dc5168a",
    }
    try:
        root = Path(ttnn.__file__).parents[2]
        for name, digest in expected.items():
            hasher = hashlib.sha256()
            with (root / name).open("rb") as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    hasher.update(block)
            if hasher.hexdigest() != digest:
                return False
        return True
    except (OSError, AttributeError, IndexError):
        return False


def _verification_workspace(a, b):
    # Read-only host handles (or independent blocking device copies) stay alive until
    # drain. Neither the caller nor a worker mutates their packed storage.
    try:
        if (
            ttnn.to_torch is not _VERIFY_TO_TORCH
            or ttnn.CONFIG.enable_logging
            or ttnn.CONFIG.enable_comparison_mode
            or ttnn.tracer.ENABLE_TRACER
            or ttnn.graph.is_graph_capture_active()
            or ttnn.graph.is_python_io_recording_enabled()
            or ttnn.decorators.PRE_OPERATION_HOOKS
            or ttnn.decorators.POST_OPERATION_HOOKS
            or not _verification_native_build()
        ):
            return None
        total = 0
        for tensor in (a, b):
            if (
                type(tensor) is not ttnn.Tensor
                or tensor.is_sharded()
                or tensor.dtype not in (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b)
                or tensor.layout not in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT)
            ):
                return None
            # Per padded element: packed snapshot <=4B, unpack/untilize/strip/output
            # <=16B, masks/abs intermediates <=9B; round up to32B plus64KiB overhead.
            total += 32 * math.prod(tensor.padded_shape) + (64 << 10)
        return total
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None


def _verification_worker(a, b):
    stats = dict(convert_s=0.0, compare_s=0.0)
    cpu = time.thread_time()
    try:
        equal, finite = _verify_shard_pair(a, b, stats, native=True)
        return equal, finite, stats, None
    except Exception as err:
        return False, True, stats, f"{type(err).__name__}: {err}"
    finally:
        stats["cpu_s"] = time.thread_time() - cpu
        stats["thread_id"] = threading.get_ident()
        a = b = None


class _VerificationPool:
    def __init__(self, workers, counters):
        self.workers, self.counters = workers, counters
        self.executor = None
        self.pending = deque()
        self.bytes = 0
        self.threads = set()

    def take(self):
        index, future, size = self.pending.popleft()
        try:
            with _verification_phase(self.counters, "wait_s"):
                equal, finite, phases, error = future.result()
            for phase in ("convert_s", "compare_s", "cpu_s"):
                self.counters["worker_" + phase] += phases[phase]
            self.threads.add(phases["thread_id"])
            self.counters["worker_threads"] = len(self.threads)
            self.counters["shards_completed"] += int(error is None)
            if error:
                logger.error(f"cache worker failed at shard {index}: {error}")
            return index, equal, finite
        except Exception as err:
            logger.error(f"cache worker cancelled/failed at shard {index}: {err}")
            return index, False, True
        finally:
            self.bytes -= size

    def pairs(self, resident, fresh):
        try:
            for index, (a, b) in enumerate(zip(resident, fresh)):
                size = _verification_workspace(a, b) if self.workers > 1 else None
                if size is None or size > _VERIFY_WORKSPACE_BYTES:
                    while self.pending:
                        yield self.take()
                    self.counters["serial_pairs"] += 1
                    if size is not None:
                        self.counters["oversized_pairs"] += 1
                        self.counters["largest_serial_workspace_bytes"] = max(
                            self.counters["largest_serial_workspace_bytes"], size
                        )
                    equal, finite = _verify_shard_pair(a, b, self.counters)
                    self.counters["shards_completed"] += 1
                    yield index, equal, finite
                else:
                    while self.pending and (
                        len(self.pending) >= self.workers or self.bytes + size > _VERIFY_WORKSPACE_BYTES
                    ):
                        yield self.take()
                    with _verification_phase(self.counters, "snapshot_s"):
                        a = a.cpu(blocking=True)
                        b = b.cpu(blocking=True)
                    if self.executor is None:
                        self.executor = ThreadPoolExecutor(self.workers, thread_name_prefix="cache-verify")
                    self.pending.append((index, self.executor.submit(_verification_worker, a, b), size))
                    self.bytes += size
                    self.counters["parallel_pairs"] += 1
                    self.counters["peak_inflight_pairs"] = max(self.counters["peak_inflight_pairs"], len(self.pending))
                    self.counters["peak_workspace_bytes"] = max(self.counters["peak_workspace_bytes"], self.bytes)
            while self.pending:
                yield self.take()
        finally:
            # Also drain if the caller or a snapshot is interrupted. No future may
            # retain a temporary device tensor or spill into the next parameter.
            while self.pending:
                self.take()
            a = b = None

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)


def verify_saved_model(
    tt_model: Module,
    cache_dir: str | Path,
    /,
    *,
    prefix: str = "",
    stats: dict | None = None,
    workers: int | None = None,
) -> bool:
    """Reload independently; opt in with workers or TT_DIT_CACHE_VERIFY_WORKERS (1..4).

    Caller owns all loads, blocking snapshots and counters. Only qualified immutable
    host pairs use one bounded pool. Unsupported paths remain decorated/serial.
    Parallel workspace is capped at512MiB; an oversized serial pair is reported
    separately, not claimed bounded. Model weights and the reloaded parameter are
    additional. Worker phases overlap: do not add them to wall_s or walltime's ledger.
    load_s/snapshot_s/convert_s include lazy faults, not isolated disk I/O.
    Callers must keep resident weights and instrumentation unchanged during verification.
    """
    try:
        workers = max(
            1, min(4, int(os.environ.get("TT_DIT_CACHE_VERIFY_WORKERS", "1") if workers is None else workers))
        )
    except (TypeError, ValueError):
        workers = 1
    started = last_report = time.monotonic()
    counters = dict(
        parameters_completed=0,
        parameters_failed=0,
        shards_completed=0,
        tensorbin_bytes=0,
        load_s=0.0,
        convert_s=0.0,
        compare_s=0.0,
        completed=False,
        workers=workers,
        parallel_pairs=0,
        serial_pairs=0,
        oversized_pairs=0,
        peak_workspace_bytes=0,
        largest_serial_workspace_bytes=0,
        peak_inflight_pairs=0,
        snapshot_s=0.0,
        wait_s=0.0,
        worker_convert_s=0.0,
        worker_compare_s=0.0,
        worker_cpu_s=0.0,
        worker_threads=0,
    )

    pool = _VerificationPool(workers, counters)

    def report_progress():
        nonlocal last_report
        now = time.monotonic()
        if now - last_report >= 5.0:
            last_report = now
            logger.info("cache verify progress: " + json.dumps(counters, sort_keys=True))

    cache_dir = Path(cache_dir)
    ok = True
    try:
        for name, parameter in _verification_parameters(tt_model, prefix):
            path = cache_dir / f"{name}.tensorbin"
            reloaded = resident = fresh = pairs = None
            parameter_ok = True
            try:
                if not path.is_file():
                    parameter_ok = False
                    logger.error(f"cache incomplete: no tensorbin for '{name}' at '{path}'")
                else:
                    counters["tensorbin_bytes"] += path.stat().st_size
                    with _verification_phase(counters, "load_s"):
                        reloaded = ttnn.load_tensor(path, device=None if parameter.on_host else parameter.device)
                    resident = ttnn.get_device_tensors(parameter.data)
                    fresh = ttnn.get_device_tensors(reloaded)
                    if len(resident) != len(fresh):
                        parameter_ok = False
                        logger.error(f"cache mismatch in '{path.name}': shard count {len(resident)} vs {len(fresh)}")
                    # Do not deduplicate replicated storage: its ordered logical coordinates
                    # are part of the comparison. Only native shard handles are retained here.
                    pairs = pool.pairs(resident, fresh)
                    for i, equal, finite in pairs:
                        if not equal or not finite:
                            parameter_ok = False
                            reason = "mismatch" if not equal else "NaN or inf"
                            logger.error(f"cache {reason} in '{path.name}': shard {i}")
                        report_progress()
                    del resident, fresh
            except (RuntimeError, OSError, ValueError, TypeError) as err:
                parameter_ok = False
                logger.error(f"cache unreadable: '{path.name}' failed to load or compare: {err}")
            finally:
                if pairs is not None:
                    pairs.close()
                resident = fresh = pairs = None
                if reloaded is not None and not parameter.on_host:
                    ttnn.deallocate(reloaded)
                reloaded = None
            ok &= parameter_ok
            counters["parameters_completed"] += 1
            counters["parameters_failed"] += int(not parameter_ok)
            report_progress()
        counters["completed"] = True
        return ok
    finally:
        pool.close()
        counters["wall_s"] = time.monotonic() - started
        if stats is not None:
            stats.update(counters)
        logger.info("cache verify summary: " + json.dumps(counters, sort_keys=True))


def _verify_env_enabled() -> bool:
    return os.environ.get("TT_DIT_CACHE_VERIFY", "0") in ("1", "true", "True")


def model_cache_dir(
    *,
    model_name: str,
    subfolder: str,
    parallel_config: NamedTuple,
    mesh_shape: Sequence[int],
    mesh_device: ttnn.MeshDevice,
    dtype: str = "bf16",
    is_fsdp: bool = False,
    content: str | None = None,
    required: bool = True,
) -> Path | None:
    cache_dir = _cache_root()
    if cache_dir is None:
        if required:
            msg = "Cache is required. Set the TT_DIT_CACHE_DIR environment variable."
            raise RuntimeError(msg)
        return None

    parallel_key = config_id(parallel_config)
    mesh_key = "x".join(str(x) for x in mesh_shape)

    key = f"{parallel_key}mesh{mesh_key}_{dtype}"
    if is_fsdp:
        key += "_FSDP"
    # Content lives in the path, not beside it, so weights that disagree with the key land in
    # different directories: a stale artifact misses and rebuilds instead of being read back
    # under a key that no longer describes it.
    if content is not None:
        key += f"_c{content}"

    path = Path(cache_dir) / model_name / subfolder / key

    ownership_suffix = _cache_ownership_suffix(mesh_device)
    if ownership_suffix:
        path = path / ownership_suffix

    return path


def nonfinite_tensorbins(tt_model: Module, cache_dir: str | Path) -> list[str]:
    """The parameters of `tt_model` whose tensorbin in `cache_dir` holds NaN or inf, read on the host.

    Only logical values count: to_torch drops tile padding. A file that does not load is listed too.
    """
    import torch

    bad = []
    for name, _ in _walk_parameters(tt_model):
        try:
            stored = ttnn.load_tensor(Path(cache_dir) / f"{name}.tensorbin", device=None)
            if not all(torch.isfinite(ttnn.to_torch(t)).all() for t in ttnn.get_device_tensors(stored)):
                bad.append(name)
        except RuntimeError as err:
            logger.warning(f"cache tensorbin '{name}' in '{cache_dir}' does not load on the host: {err}")
            bad.append(name)
    return bad


def _scan_signature(cache_dir: Path) -> str:
    entries = sorted((f.name, st.st_size, st.st_mtime_ns) for f in cache_dir.glob("*.tensorbin") for st in [f.stat()])
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def _set_aside_if_nonfinite(tt_model: Module, cache_dir: Path) -> bool:
    """Whether a cache hit holds NaN or inf; if so it is moved aside, so `load_model` rebuilds it.

    The manifest check sees only sizes, so a cache written from bad weights (a faulty device, a
    torn prep) is otherwise served on every run until someone deletes it by hand (#208: an ltx VAE
    encoder cache of inf/NaN). Opt-in with TT_DIT_CACHE_VERIFY=1, like the publish-time verify; once
    per cache dir per process; single-process meshes only, since a rebuild on one rank alone would
    leave the others waiting at its barriers; and only up to TT_DIT_CACHE_SCAN_MAX_BYTES of tensorbins.
    """
    if not _verify_env_enabled() or str(cache_dir) in _scanned_cache_dirs or _distributed_world_size() > 1:
        return False
    _scanned_cache_dirs.add(str(cache_dir))
    size = sum(f.stat().st_size for f in cache_dir.glob("*.tensorbin"))
    if size > int(os.environ.get("TT_DIT_CACHE_SCAN_MAX_BYTES", SCAN_MAX_BYTES_DEFAULT)):
        return False
    stamp = _scan_signature(cache_dir)
    try:
        if (cache_dir / SCAN_OK_FILE).read_text() == stamp:
            return False
    except OSError:
        pass
    t0 = time.time()
    bad = nonfinite_tensorbins(tt_model, cache_dir)
    logger.info(f"scanned cache at '{cache_dir}' ({size / 2**30:.2f} GiB) for NaN/inf in {time.time() - t0:.1f}s")
    if not bad:
        try:
            tmp = cache_dir / f"{SCAN_OK_FILE}.{os.getpid()}"
            tmp.write_text(stamp)
            tmp.replace(cache_dir / SCAN_OK_FILE)
        except OSError as err:
            logger.debug(f"cannot stamp the clean scan of '{cache_dir}' ({err}); the next load scans it again")
        return False
    aside = cache_dir.with_name(f"{cache_dir.name}.bad-nonfinite-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}")
    logger.warning(
        f"cache at '{cache_dir}' holds NaN or inf in {len(bad)} tensors ({', '.join(bad[:5])}"
        f"{', ...' if len(bad) > 5 else ''}); moving it to '{aside}' and rebuilding"
    )
    try:
        cache_dir.rename(aside)
    except FileNotFoundError:
        pass  # another process moved it first
    except OSError as err:
        # Not ours to move (a cache root shared read-only): rebuilding could not publish over it either.
        logger.error(f"cannot move the NaN/inf cache at '{cache_dir}' aside ({err}); serving it as it is")
        return False
    return True


def miss_reason(cache_dir: str | Path) -> str:
    """Why `load_model` is rebuilding `cache_dir`, for the wall-time anomaly list."""
    if _kernel_capture_only():
        # The capture pass never publishes, so its misses say nothing about the next run's hits.
        return f"kernel capture-only pass, cache not written to '{cache_dir}'"
    if not (Path(cache_dir) / CACHE_DICT_FILE).is_file():
        reason = f"no cache at '{cache_dir}' yet, writing it"
        if Path("/tmp") in Path(cache_dir).resolve().parents:
            # systemd-tmpfiles empties /tmp at boot, so a reboot silently turns a warm cache cold.
            reason += "; /tmp is wiped at every boot, keep TT_DIT_CACHE_DIR on persistent storage"
        return reason
    return f"cache at '{cache_dir}' rejected by the manifest check, rebuilding"


def _cache_is_complete(cache_dir: str | Path, tt_model: Module, key: str | None) -> bool:
    """Whether `cache_dir` holds an artifact that is safe to serve.

    Without a content key the caller has opted out of verification and only the completion
    marker is checked, which keeps every pre-existing cache loadable. With one, the manifest
    must agree with the module we are about to load into — a disagreement is reported as a miss
    so the weights are rebuilt, never quietly read back as something they are not.
    """
    marker = Path(cache_dir) / CACHE_DICT_FILE
    if not marker.is_file():
        return False
    if key is None:
        return True

    try:
        manifest = json.loads(marker.read_text())
    except (OSError, ValueError) as err:
        logger.warning(f"cache at '{cache_dir}' has an unreadable manifest ({err}); rebuilding.")
        return False

    if manifest.get("format") != CACHE_FORMAT or manifest.get("content_key") != key:
        logger.warning(f"cache at '{cache_dir}' was built for another key; rebuilding.")
        return False

    recorded = manifest.get("parameters", {})
    if {name: dict(spec, bytes=None) for name, spec in recorded.items()} != _structure(tt_model):
        logger.warning(f"cache at '{cache_dir}' does not describe this module; rebuilding.")
        return False

    for name, spec in recorded.items():
        path = Path(cache_dir) / f"{name}.tensorbin"
        if not path.is_file() or path.stat().st_size != spec["bytes"]:
            logger.warning(f"cache at '{cache_dir}' is missing or truncated at '{name}'; rebuilding.")
            return False

    return True


def _cache_ownership_suffix(mesh_device: ttnn.MeshDevice) -> str:
    """Multi-host cache dir suffix keyed by local mesh-coordinate ownership.

    Single-host / no distributed context: empty (same unsuffixed path as before).
    Multi-host: ``host_coords_r{r0}-{r1}_c{c0}-{c1}`` for the local coord bounding box.
    """
    if _distributed_world_size() <= 1:
        return ""

    view = mesh_device.get_view()
    rows = []
    cols = []
    for coord in ttnn.MeshCoordinateRange(view.shape()):
        if view.is_local(coord):
            rows.append(int(coord[0]))
            cols.append(int(coord[1]))
    return f"host_coords_r{min(rows)}-{max(rows)}_c{min(cols)}-{max(cols)}"


def _structure(tt_model: Module) -> dict:
    """What each tensorbin must contain, as recorded in the manifest (byte count filled at save)."""
    return {
        name: {"shape": list(p.total_shape), "dtype": str(p.dtype), "layout": str(p.layout), "bytes": None}
        for name, p in _walk_parameters(tt_model)
    }


def _reclaim_superseded(legacy_dir: Path | None) -> None:
    """Drop the content-blind directory that a content key replaces, before rebuilding over it.

    Once a module keys on content, nothing constructs its old path again, so whatever sits there
    is unreachable — and for the 22B transformer it is 37GB. Freeing it first makes the one-time
    migration disk-neutral; leaving it would need room for two copies of a checkpoint that only
    just fits once. It is not a fallback: an unverifiable artifact is exactly what is being
    retired, so there is nothing to keep.
    """
    if legacy_dir is None or not legacy_dir.is_dir():
        return
    logger.info(f"Reclaiming superseded content-blind cache at '{legacy_dir}'.")
    shutil.rmtree(legacy_dir, ignore_errors=True)


def _publish_cache(tt_model: Module, cache_dir: str | Path, key: str | None) -> None:
    """Write the cache to a scratch directory and move it into place in one step.

    The tensorbins and their completion marker become visible together or not at all, so a build
    that dies partway, or two processes racing to fill the same key, can never leave a
    half-written directory that later reads as complete.
    """
    cache_dir = Path(cache_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    # Not tempfile.mkdtemp: it creates the directory 0700 and the rename below publishes that mode.
    # Under a default ACL (a cache root shared by several users) the 0700 also sets the leaf's ACL
    # mask to ---, so every other user, the serving account included, is denied the cache. A plain
    # mkdir gets 0777 less the umask, the parent's default ACL and its setgid group like any other
    # directory; the pid and random suffix keep concurrent builders apart as mkdtemp did.
    staging = cache_dir.parent / f".{cache_dir.name}.staging-{os.getpid()}-{secrets.token_hex(4)}"
    staging.mkdir()

    try:
        tt_model.save(staging)
        # Opt-in (TT_DIT_CACHE_VERIFY=1): only a cache that reads back exactly is published.
        # Without this a handful of flipped bytes on the way to disk is reused on every later run,
        # and nothing downstream can tell a damaged weight from a real one. Off by default because
        # it re-reads the whole cache once at creation.
        if _verify_env_enabled() and not verify_saved_model(tt_model, staging):
            logger.error(f"cache for {cache_dir} did not verify; not publishing it, so it is rebuilt next time.")
            return

        manifest = {"format": CACHE_FORMAT}
        if key is not None:
            parameters = _structure(tt_model)
            for name, spec in parameters.items():
                spec["bytes"] = (staging / f"{name}.tensorbin").stat().st_size
            manifest |= {"content_key": key, "parameters": parameters}
        (staging / CACHE_DICT_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True))

        # A concurrent process may have finished publishing this key while we prepared weights (a
        # long, on-device window since load_model's completeness check). Its artifact is already
        # complete — for a content key its bytes are ours by construction — so adopt it rather than
        # delete a valid cache out from under a concurrent reader mid-load.
        if _cache_is_complete(cache_dir, tt_model, key):
            logger.info(f"cache at '{cache_dir}' was published concurrently; adopting it.")
            return
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        os.replace(staging, cache_dir)
    except OSError:
        # A racing process may have published this exact key first. Its content key matches ours,
        # so its bytes are ours; adopt them rather than fight over the directory.
        if _cache_is_complete(cache_dir, tt_model, key):
            logger.info(f"cache at '{cache_dir}' was published concurrently; adopting it.")
        else:
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _distributed_world_size() -> int:
    if not ttnn.distributed_context_is_initialized():
        return 1
    return int(ttnn.distributed_context_world_size())


def _cache_root() -> str | None:
    return os.environ.get("TT_DIT_CACHE_DIR")
