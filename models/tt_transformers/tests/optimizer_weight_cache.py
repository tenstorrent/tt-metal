# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A converted-weight store for the optimizer harness: each weight is converted once, ever.

Every harness run (gate, timing, profile) used to convert every Hugging Face weight into ttnn's
format from scratch: ~115 s of every Qwen3-32B run, ~30 s of every Llama-3.1-8B run, mostly the
host-side tile packing and block-float quantization inside `ttnn.from_torch`.

What makes a converted weight is exactly the arguments of the `ttnn.as_tensor` call that converts it:
the torch tensor's bytes, shape and dtype, the target dtype and layout, the memory config, and how
the mesh mapper splits it over the chips. This store keys each weight by a hash of exactly those
(plus the ttnn build that does the conversion), so it is correct under any edit to the model code:

  - an edit that changes a weight (dtype, padding, fusing, sharding, preprocessing) changes the
    torch bytes or the arguments, so the key changes and that weight is converted fresh;
  - an edit that does not change a weight (a matmul's program config, fidelity, a new op) leaves
    its key alone and the converted file is reused;
  - a call the store cannot describe exactly (an unknown mesh mapper, a `preprocess` callable, a
    non-torch input) is not stored at all and converts fresh, exactly as without the store.

The model's torch work on the checkpoint (its transposes, concatenations, pads) does not run for a
weight the store already has. During the model build (`RunCache.build`) every checkpoint tensor is a
deferred tensor (optimizer_lazy_weights) that records the ops applied to it, and a weight reaches
`as_tensor` as a recipe: the ops, their arguments, and the content of the checkpoint tensors it
starts from. The store keeps, per recipe key, the content key the recipe produced; a known recipe
loads that file without running the ops, and an unknown one is replayed into an ordinary tensor and
keyed by its bytes exactly as above. Before this, the torch work ran on every run to make the bytes
to hash: 32.5 s of a warm Qwen3-32B run once the model merged and padded its gate/up weights.

Loading a stored weight onto the mesh. `load_tensor_flatbuffer` maps the file read-only and private,
and the tensor's host shards point into that mapping. `enqueue_write_tensor` pins host memory of a
tensor above 32 MB for the upload (`PinnedMemoryCache::try_pin`, read-only, since #52893), and on
QB2 (IOMMU on, KMD 2.11.0, kernel 7.0.0) the kernel's long-term read-only pin of those private file
pages does not return: `tenstorrent 0000:04:00.0: could only pin 512 of N pages` in `journalctl -k`
at the moment each stalled run was killed (Llama-3.1-8B's embedding since 2026-09-09, Qwen3-32B's
wqkv and embedding on 2026-09-25). That was the "warm-cache hang" the harness used to avoid by
never reusing a cache. Freshly converted weights live in anonymous heap memory and pin fine. So
before a stored weight is moved, `_privatize` gives the process its own copy of every page of the
mapping (a write of each page's first byte back to itself, under a temporary PROT_WRITE, which the
kernel serves by copy-on-write), checks in /proc/self/pagemap that every page is now anonymous, and
only then moves the tensor exactly as a fresh build does: `host.to(device, memory_config)`. The
bytes are unchanged by construction. If any step fails, the weight is converted fresh.

Files are written by ttnn's own fresh path into a scratch name and renamed into place, so a run that
dies mid-write leaves nothing a later run could load. The store is pruned to OPTIMIZER_WEIGHT_STORE_GB
(least recently used first). OPTIMIZER_WEIGHT_STORE=0 turns it off; OPTIMIZER_WEIGHT_STORE_LAZY=0 keeps
the store but runs the model's torch work eagerly; OPTIMIZER_WEIGHT_STORE_VERIFY=1 replays every known
recipe anyway and checks it still produces the content key on record.
"""

from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import mmap
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from models.tt_transformers.tests import optimizer_lazy_weights as lazy

REPO_ROOT = Path(__file__).resolve().parents[3]
STORE_ROOT = REPO_ROOT / "generated" / "optimizer_weight_store"
FORMAT = "optimizer-weight-store-v2"  # v2: the key hashes a tensor's memory in place, with its strides
STORE_GB = float(os.environ.get("OPTIMIZER_WEIGHT_STORE_GB", "120"))
# Scratch TT_CACHE_PATH directories left by runs that were killed (a timeout, a stop) are removed
# once they are this old; a live run touches its own directory as it converts.
STALE_SCRATCH_S = 6 * 3600

_PROT_READ, _PROT_WRITE = 1, 2
_libc = ctypes.CDLL(None, use_errno=True)
_libc.mprotect.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
_libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
_MADV_POPULATE_WRITE = 23  # Linux 5.14+


def enabled() -> bool:
    return os.environ.get("OPTIMIZER_WEIGHT_STORE", "1") != "0"


def lazy_enabled() -> bool:
    return enabled() and os.environ.get("OPTIMIZER_WEIGHT_STORE_LAZY", "1") != "0"


def _verify() -> bool:
    return os.environ.get("OPTIMIZER_WEIGHT_STORE_VERIFY") == "1"


# ---- describing a mesh mapper exactly ------------------------------------------------------------
# ttnn's mappers are C++ objects with no readable configuration, so the store records the arguments
# each one was made with. Only mappers made through these factories after `install()` are known.
_MAPPER_FACTORIES = ("replicate_tensor_to_mesh_mapper", "shard_tensor_to_mesh_mapper", "create_mesh_mapper")
_mapper_args: dict[int, tuple[object, str]] = {}  # id(mapper) -> (mapper kept alive, description)


def _describe_arg(value) -> str | None:
    import ttnn

    if isinstance(value, ttnn.MeshDevice):
        return f"MeshDevice(shape={tuple(value.shape)})"
    text = repr(value)
    return None if " at 0x" in text else text  # a repr with an address says nothing stable


def _wrap_factory(name: str):
    import ttnn

    original = getattr(ttnn, name)
    if getattr(original, "_optimizer_weight_store", False):
        return

    def factory(*args, **kwargs):
        mapper = original(*args, **kwargs)
        described = [_describe_arg(a) for a in args] + [_describe_arg(v) for _, v in sorted(kwargs.items())]
        if all(d is not None for d in described):  # otherwise unknown: its weights convert fresh
            names = [""] * len(args) + [f"{k}=" for k in sorted(kwargs)]
            _mapper_args[id(mapper)] = (mapper, f"{name}({', '.join(n + d for n, d in zip(names, described))})")
        return mapper

    factory._optimizer_weight_store = True
    setattr(ttnn, name, factory)
    # A module that ran `from ttnn import <factory>` before this holds the original, and its
    # mappers were unknown to the store, so their weights converted every run (rope.py's
    # replicate_tensor_to_mesh_mapper: "not_stored=3" on Qwen3-32B). Point those names here too.
    import sys

    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("models.") and getattr(module, name, None) is original:
            setattr(module, name, factory)


def _describe_mapper(mesh_mapper) -> str | None:
    import ttnn

    if mesh_mapper is None:
        return "none"
    if isinstance(mesh_mapper, ttnn.ReplicateTensorToMeshWrapper):
        # as_tensor converts and stores a replicated weight unsplit; `.to(mesh)` replicates it.
        return "replicate-unsplit"
    entry = _mapper_args.get(id(mesh_mapper))
    return entry[1] if entry is not None and entry[0] is mesh_mapper else None


# ---- the key -------------------------------------------------------------------------------------
def _ttnn_build() -> str:
    """Identity of the code that converts: the ttnn extension and libraries, and ttnn's Python."""
    import ttnn

    digest = hashlib.sha256()
    pkg = Path(ttnn.__file__).resolve().parent
    for path in sorted(pkg.rglob("*.so")) + sorted((REPO_ROOT / "build" / "lib").glob("*.so")):
        st = path.stat()
        digest.update(f"{path.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    for rel in ("operations/core.py", "distributed/distributed.py"):
        digest.update((pkg / rel).read_bytes())
    return digest.hexdigest()


_BUILD: str | None = None


def _key(tensor, dtype, layout, memory_config, mapper: str) -> str | None:
    import torch
    import xxhash

    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        return None
    data = tensor.detach()
    span, strides = _memory_span(data)
    head = json.dumps(
        [FORMAT, _BUILD, repr(dtype), repr(layout), repr(memory_config), mapper, str(data.dtype), list(data.shape),
         strides]
    )
    h = xxhash.xxh3_128()
    h.update(head.encode())
    if span is not None and span.numel():
        h.update(span.view(torch.uint8).numpy())
    return h.hexdigest()


# The model hands most weights over as views (a transpose, a slice), and `.contiguous()` copied
# every one only to hash it: 23.6 of a warm Qwen3-32B build's 46.5 weight seconds (py-spy,
# 2026-09-26). So the key hashes the tensor's memory span in place, with its strides. Two tensors with
# the same contents in different layouts get different keys, which only costs a conversion.
_memory_span = lazy.memory_span


def _recipe_key(tensor, dtype, layout, memory_config, mapper: str) -> str:
    """The key of a deferred tensor's recipe with this conversion: known before any torch work runs."""
    import torch
    import xxhash

    head = json.dumps(
        [FORMAT, "recipe", _BUILD, torch.__version__, repr(dtype), repr(layout), repr(memory_config), mapper,
         str(tensor.dtype), list(tensor.shape), list(tensor.stride()), tensor.recipe_digest()]
    )
    return xxhash.xxh3_128(head.encode()).hexdigest()


# ---- loading a stored file into memory that pins ------------------------------------------------
def _mappings_of(path: str) -> list[tuple[int, int, str]]:
    found = []
    with open("/proc/self/maps") as maps:
        for line in maps:
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].rstrip("\n") == path:
                lo, hi = (int(x, 16) for x in fields[0].split("-"))
                found.append((lo, hi, fields[1]))
    return found


_PAGE_PRESENT, _PAGE_FILE_OR_SHARED = 1 << 63, 1 << 61


def _all_pages_private(lo: int, hi: int) -> bool:
    """True if every page in [lo, hi) is present and anonymous (/proc/self/pagemap, 8 bytes a page).

    Not /proc/self/smaps: the kernel builds that by walking every mapping of the process, including
    the mapped checkpoint, which cost ~0.4 s per weight on Qwen3-32B.
    """
    import numpy as np

    page = mmap.PAGESIZE
    with open("/proc/self/pagemap", "rb") as pagemap:
        pagemap.seek(lo // page * 8)
        entries = np.frombuffer(pagemap.read((hi - lo) // page * 8), dtype=np.uint64)
    if len(entries) != (hi - lo) // page:
        return False
    present = (entries & np.uint64(_PAGE_PRESENT)) != 0
    shared = (entries & np.uint64(_PAGE_FILE_OR_SHARED)) != 0
    return bool(present.all() and not shared.any())


def _privatize(path: str) -> bool:
    """Give this process its own copy of every page ttnn mapped from `path`; True if all are anonymous."""
    import numpy as np

    ranges = _mappings_of(path)
    if not ranges:
        return False
    page = mmap.PAGESIZE
    for lo, hi, perms in ranges:
        if perms[3] != "p":  # only a private mapping can be copied on write
            return False
        if _libc.mprotect(lo, hi - lo, _PROT_READ | _PROT_WRITE) != 0:
            return False
        try:
            # One call faults every page in writable, so the kernel copies each for us: 7.6 s of
            # page-at-a-time stores per warm Qwen3-32B build before (py-spy, 2026-09-26).
            if _libc.madvise(lo, hi - lo, _MADV_POPULATE_WRITE) != 0:
                pages = np.frombuffer((ctypes.c_ubyte * (hi - lo)).from_address(lo), dtype=np.uint8)
                pages[::page] = pages[::page].copy()  # a store to each page does the same
        finally:
            _libc.mprotect(lo, hi - lo, _PROT_READ)
        if not _all_pages_private(lo, hi):
            return False
    return True


# ---- the store -----------------------------------------------------------------------------------
class _Stats:
    def __init__(self):
        self.reused = self.stored = self.passed = self.failed = 0
        self.reused_bytes = 0
        # Seconds, for the report: where a model build's weight time goes. `between` is the time
        # outside as_tensor from the first call to the last -- the model's own torch work.
        self.key_s = self.load_s = self.move_s = self.fresh_s = self.between_s = 0.0
        self.first = self.last_exit = None
        # Deferred tensors: recipes loaded without their torch work, recipes replayed (and how long
        # the replay took), and under OPTIMIZER_WEIGHT_STORE_VERIFY recipes checked and found wrong.
        self.recipe_hits = self.recipe_replays = self.verified = self.mismatched = 0
        self.replay_s = 0.0
        self.redone: str | None = None  # why a deferred build was redone eagerly
        self.not_stored: list[str] = []  # "name: reason", for the report


STATS = _Stats()


def _spec(t) -> dict:
    return {"shape": list(t.shape), "dtype": repr(t.dtype), "layout": repr(t.layout), "topology": repr(t.tensor_topology())}


def install(root: Path = STORE_ROOT) -> None:
    """Wrap ttnn.as_tensor and the mesh-mapper factories. Idempotent; a no-op when disabled."""
    global _BUILD
    import ttnn

    if not enabled() or getattr(ttnn.as_tensor, "_optimizer_weight_store", False):
        return
    _BUILD = _ttnn_build()
    root = Path(root)
    (root / "tmp").mkdir(parents=True, exist_ok=True)
    for name in _MAPPER_FACTORIES:
        _wrap_factory(name)
    original = ttnn.as_tensor

    def as_tensor(tensor, dtype=None, *, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, memory_config=None,
                  cache_file_name=None, preprocess=None, mesh_mapper=None):
        entered = time.perf_counter()
        if STATS.last_exit is not None:
            STATS.between_s += entered - STATS.last_exit
        STATS.first = STATS.first or entered
        try:
            return _as_tensor(tensor, dtype, layout, device, memory_config, cache_file_name, preprocess, mesh_mapper)
        finally:
            STATS.last_exit = time.perf_counter()

    def _reuse(key, device, memory_config):
        """The stored weight `key` placed on the device, or None if it is not stored or does not load cleanly."""
        final = root / key[:2] / f"{key}.tensorbin"
        meta = final.with_suffix(".json")
        if not (final.is_file() and meta.is_file()):
            return None
        try:
            expected = json.loads(meta.read_text())
            started = time.perf_counter()
            host = ttnn._ttnn.tensor.load_tensor_flatbuffer(str(final), device=None)
            ok = _spec(host) == expected["spec"] and _privatize(str(final.resolve()))
            STATS.load_s += time.perf_counter() - started
            if ok:
                started = time.perf_counter()
                placed = host.to(device, memory_config)
                STATS.move_s += time.perf_counter() - started
                os.utime(final)  # most recently used, for pruning
                STATS.reused += 1
                STATS.reused_bytes += final.stat().st_size
                return placed
        except (OSError, RuntimeError, ValueError, KeyError):
            pass
        STATS.failed += 1  # a stored file that does not load cleanly is converted fresh
        return None

    def _replay(tensor):
        started = time.perf_counter()
        try:
            return lazy.materialize(tensor)
        finally:
            STATS.recipe_replays += 1
            STATS.replay_s += time.perf_counter() - started

    def _as_tensor(tensor, dtype, layout, device, memory_config, cache_file_name, preprocess, mesh_mapper):
        call = dict(dtype=dtype, layout=layout, device=device, memory_config=memory_config,
                    preprocess=preprocess, mesh_mapper=mesh_mapper)
        mapper = _describe_mapper(mesh_mapper)
        storable = cache_file_name is not None and device is not None and preprocess is None and mapper is not None
        recipe = recorded = None
        if lazy.is_lazy(tensor):
            if storable and lazy.STATE.taint is None:
                started = time.perf_counter()
                recipe = _recipe_key(tensor, dtype, layout, memory_config, mapper)
                recorded = _read_recipe(root, recipe)
                STATS.key_s += time.perf_counter() - started
                if recorded is not None and not _verify():
                    placed = _reuse(recorded, device, memory_config)
                    if placed is not None:
                        STATS.recipe_hits += 1
                        return placed
            tensor = _replay(tensor)
        key = None
        if storable:
            started = time.perf_counter()
            key = _key(tensor, dtype, layout, memory_config, mapper)
            STATS.key_s += time.perf_counter() - started
        if key is None:
            STATS.passed += 1
            why = ("no cache name" if cache_file_name is None else "no device" if device is None
                   else "preprocess" if preprocess is not None else "unknown mapper" if mapper is None
                   else "not a CPU torch tensor")
            STATS.not_stored.append(f"{Path(str(cache_file_name)).name if cache_file_name else '?'}: {why}")
            return original(tensor, cache_file_name=cache_file_name, **call)
        if recorded is not None:  # only under OPTIMIZER_WEIGHT_STORE_VERIFY, or when that file failed to load
            STATS.verified += 1
            if recorded != key:
                STATS.mismatched += 1
                print(f"WEIGHT_STORE recipe {recipe} recorded {recorded} but replays to {key}", flush=True)
        placed = _reuse(key, device, memory_config)
        if placed is not None:
            if recipe is not None and recorded != key:
                _write_recipe(root, recipe, key)
            return placed
        final = root / key[:2] / f"{key}.tensorbin"
        meta = final.with_suffix(".json")
        # Fresh: ttnn's own path converts, dumps the host tensor to our scratch name, and moves it.
        scratch = root / "tmp" / uuid.uuid4().hex
        started = time.perf_counter()
        placed = original(tensor, cache_file_name=str(scratch), **call)
        STATS.fresh_s += time.perf_counter() - started
        dumped = Path(f"{scratch}_dtype_{dtype.name if dtype is not None else 'None'}"
                      f"_layout_{layout.name if layout is not None else 'None'}.tensorbin")
        try:
            host = ttnn._ttnn.tensor.load_tensor_flatbuffer(str(dumped), device=None)
            spec = _spec(host)
            del host
            final.parent.mkdir(parents=True, exist_ok=True)
            tmp_meta = meta.with_suffix(f".json.{uuid.uuid4().hex}")
            tmp_meta.write_text(json.dumps({"spec": spec, "format": FORMAT}))
            os.replace(tmp_meta, meta)
            os.replace(dumped, final)  # the commit point: the tensorbin appears whole or not at all
            STATS.stored += 1
            if recipe is not None:
                _write_recipe(root, recipe, key)
        except (OSError, RuntimeError):
            dumped.unlink(missing_ok=True)
        return placed

    as_tensor._optimizer_weight_store = True
    ttnn.as_tensor = as_tensor

    # A deferred tensor has no memory of its own, so every other way a torch tensor enters ttnn
    # replays it first. ttnn.from_torch is the one the model code uses (as_tensor's fresh path too).
    from_torch = ttnn.from_torch

    def from_torch_replayed(tensor, *args, **kwargs):
        return from_torch(lazy.materialize(tensor) if lazy.is_lazy(tensor) else tensor, *args, **kwargs)

    ttnn.from_torch = from_torch_replayed


# ---- recipes: a deferred tensor's recipe key -> the content key it produced ---------------------
def _recipe_path(root: Path, recipe: str) -> Path:
    return Path(root) / "recipes" / recipe[:2] / f"{recipe}.json"


def _read_recipe(root: Path, recipe: str) -> str | None:
    try:
        key = json.loads(_recipe_path(root, recipe).read_text())["key"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return key if isinstance(key, str) else None


def _write_recipe(root: Path, recipe: str, key: str) -> None:
    if lazy.STATE.taint is not None:  # this build is redone eagerly; record nothing from it
        return
    path = _recipe_path(root, recipe)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".json.{uuid.uuid4().hex}")
        tmp.write_text(json.dumps({"key": key, "format": FORMAT}))
        os.replace(tmp, path)
    except OSError:
        pass


def report() -> str:
    eager = ", ".join(f"{op} x{n}" for op, n in lazy.STATE.eager.most_common(4))
    return (
        f"WEIGHT_STORE reused={STATS.reused} ({STATS.reused_bytes / 1e9:.1f} GB) stored={STATS.stored} "
        f"not_stored={STATS.passed} bad_files={STATS.failed} | seconds: torch_between={STATS.between_s:.1f} "
        f"key={STATS.key_s:.1f} replay={STATS.replay_s:.1f} load={STATS.load_s:.1f} to_device={STATS.move_s:.1f} "
        f"fresh={STATS.fresh_s:.1f} span={(STATS.last_exit or 0) - (STATS.first or 0):.1f}"
        f" | deferred: sources={lazy.STATE.sources} recipe_hits={STATS.recipe_hits} "
        f"replayed={STATS.recipe_replays} verified={STATS.verified} mismatched={STATS.mismatched}"
        + (f" redone_eagerly=({STATS.redone})" if STATS.redone else "")
        + (f" eager_ops=({eager})" if eager else "")
        + (f" | not stored: {'; '.join(STATS.not_stored[:6])}" if STATS.not_stored else "")
    )


def prune(root: Path = STORE_ROOT, limit_gb: float = STORE_GB) -> None:
    """Remove least-recently-used weights beyond the size limit, and stale scratch files."""
    root = Path(root)
    now = time.time()
    for stale in (root / "tmp").glob("*"):
        try:
            if now - stale.stat().st_mtime > STALE_SCRATCH_S:
                stale.unlink()
        except OSError:
            pass
    files = []
    for f in root.glob("??/*.tensorbin"):
        try:
            st = f.stat()
            files.append((st.st_mtime, st.st_size, f))
        except OSError:
            pass
    total = sum(size for _, size, _ in files)
    for _, size, f in sorted(files):
        if total <= limit_gb * 1e9:
            break
        f.unlink(missing_ok=True)
        f.with_suffix(".json").unlink(missing_ok=True)
        total -= size
    for recipe in root.glob("recipes/??/*.json"):  # a recipe whose weight was pruned points at nothing
        key = _read_recipe(root, recipe.stem)
        if key is None or not (root / key[:2] / f"{key}.tensorbin").is_file():
            recipe.unlink(missing_ok=True)


def _deferring(raw):
    """`raw` (a class attribute: function, staticmethod or classmethod) returning a deferred state dict.

    The descriptor kind is kept: wrapping a staticmethod as a plain function would bind `self` and
    shift every argument by one.
    """
    if isinstance(raw, staticmethod):
        inner = raw.__func__
        return staticmethod(lambda *a, **k: lazy.wrap_state_dict(inner(*a, **k)))
    if isinstance(raw, classmethod):
        inner = raw.__func__
        return classmethod(lambda cls, *a, **k: lazy.wrap_state_dict(inner(cls, *a, **k)))
    return lambda self, *a, **k: lazy.wrap_state_dict(raw(self, *a, **k))


class RunCache:
    """This run's TT_CACHE_PATH (a scratch directory, removed afterwards) with the store installed.

    Use: `cache = RunCache(root, prefix)`, set TT_CACHE_PATH to `cache.path`, build the model with
    `cache.build(lambda: create_tt_model(...))`, then `cache.cleanup()` in a finally block. tt-transformers finds its per-run cache directory empty, so
    it loads the Hugging Face checkpoint in full, and every weight goes through the store.
    """

    def __init__(self, root: Path, prefix: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        now = time.time()
        for old in self.root.glob("*-*"):
            try:
                if old.is_dir() and now - old.stat().st_mtime > STALE_SCRATCH_S:
                    shutil.rmtree(old, ignore_errors=True)
            except OSError:
                pass
        self.path = tempfile.mkdtemp(prefix=prefix, dir=str(self.root))
        install()
        print(f"WEIGHT_STORE {'on' if enabled() else 'off'} root={STORE_ROOT} scratch={self.path}", flush=True)

    def build(self, build_model, loaders=None):
        """`build_model()` with the checkpoint's tensors deferred, so stored weights skip the torch work.

        The checkpoint dict each checkpoint loader returns is made of deferred tensors for the
        duration of the call. `loaders` names them as (owner class, attribute) pairs; the default is
        tt-transformers' `ModelArgs.load_state_dict`. A model with its own args class passes its own
        loader: gemma4 loads through `Gemma4ModelArgs.load_state_dict`, a staticmethod, and with only
        the default hooked every gemma4 build reported `deferred: sources=0` and paid the full torch
        load (~9 s per run, 2026-09-28). If deferral cannot reproduce eager execution exactly (a write
        into a checkpoint tensor, its memory exposed; see optimizer_lazy_weights), or the build raises,
        the model is built again with eager tensors: the result is always the one the eager build gives.
        """
        if not lazy_enabled():
            return build_model()
        if loaders is None:
            from models.tt_transformers.tt.model_config import ModelArgs

            loaders = [(ModelArgs, "load_state_dict")]

        lazy.reset()
        restore = []
        for owner, name in loaders:
            raw = owner.__dict__.get(name)
            if raw is None:
                continue
            restore.append((owner, name, raw))
            setattr(owner, name, _deferring(raw))
        failure = None
        try:
            result = build_model()
            failure = lazy.STATE.taint
        except Exception as exc:  # noqa: BLE001 -- redone eagerly below, which raises it again if it is real
            if not lazy.STATE.sources:
                raise
            result, failure = None, lazy.STATE.taint or f"{type(exc).__name__}: {exc}"
        finally:
            for owner, name, raw in reversed(restore):
                setattr(owner, name, raw)
        if failure is None:
            return result
        STATS.redone = failure[:200]
        print(f"WEIGHT_STORE deferred build redone eagerly: {failure}", flush=True)
        result = None
        gc.collect()
        return build_model()

    def loaded(self) -> None:
        if enabled():
            print(report(), flush=True)
            prune()

    def cleanup(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)
