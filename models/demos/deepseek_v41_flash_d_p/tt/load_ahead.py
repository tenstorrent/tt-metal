# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""LOAD_THREADS (load time only; tt/settings.py): a pool of worker threads that works ahead of the model build.

The device side of a layer's build (its uploads, programs, DRAM / L1 allocations) stays on the main thread in layer
order: buffer addresses follow the allocation order, and one thread drives the command queue. The pool takes the host
work the main thread would otherwise wait on:

* reads (:class:`LoadAhead`): the bytes the next layers' build reads, into the page cache, in layer order - their
  routed experts' weight cache files (MOE_BLOCK_V2's .tensorbin: ttnn.load_tensor maps the file and reads it during
  the upload) and their dense tensors' ranges of the checkpoint shards - at most LOAD_AHEAD_GB past the layer being
  built (the next layer always). Half the threads read (one reader already saturates a slow disk; several keep a
  fast one's queue full), so the other half is free for:
* dense_dtype's bfp8 statistics (tt/common.py: a host bfp8 round trip of every dense matrix, for its log line), run on
  the pool (common.set_stats_pool) instead of before each upload.

The main thread's own reads then hit the page cache; nothing it builds changes (same calls, same order, same bytes)."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger

from .settings import settings

PIECE = 64 << 20  # the most bytes one read (one pread into the reading thread's buffer)
_GAP = 1 << 20  # checkpoint ranges closer than this are read as one (the gap's bytes too)
_CHOSEN: dict = {}  # "suffixes": the routed-expert cache files' names past a layer's prefix, as a layer's hit took them


def note_expert_files(prefix: str | None, paths) -> None:
    """The MoE adapter's weight cache hit (tt/moe_block_v2_adapter.py): the file names past ``prefix``
    (``.moe_blk.<key>.w_gu.tensorbin`` ...), so later layers' reads take the same variant when a layer's cache holds
    several."""
    if prefix is not None:
        _CHOSEN["suffixes"] = sorted(str(p)[len(prefix) :] for p in paths)


def threads() -> int:
    return max(0, int(settings.get("LOAD_THREADS")))


def expert_files(model, layer: int) -> list[Path]:
    """The routed-expert weight cache files the layer's MoE block will load on a hit: its prefix's
    ``.moe_blk.*`` / ``.moe_main.*`` .tensorbin pair - the variant an earlier layer's hit took, else the one pair the
    cache holds; [] when there is no cache, no pair, or several and no layer has chosen yet."""
    from .experts import expert_cache_prefix

    if model.weight_cache is None:
        return []
    prefix = expert_cache_prefix(
        model.weight_cache, model.ck, layer, settings.get("EXPERTS_IMPL"), model.mesh, model.cfg.n_experts
    )
    if prefix is None:
        return []
    suffixes = _CHOSEN.get("suffixes")
    if suffixes:
        files = [Path(prefix + s) for s in suffixes]
        return files if all(f.is_file() for f in files) else []
    stem = Path(prefix)
    if not stem.parent.is_dir():
        return []
    found = sorted(stem.parent.glob(f"{stem.name}.moe_*.tensorbin"))
    return found if len(found) == 2 else []


def dense_ranges(ck, layer: int) -> list[tuple[str, int, int]]:
    """(file, offset, bytes) of the layer's stored tensors but its routed experts and its engram table (neither is
    read by a cached build: the experts come from the weight cache, the table from tt/engram.py), merged per file."""
    p = f"layers.{layer}."
    spans = []
    for name, fname in ck.weight_map.items():
        if not name.startswith(p) or ".ffn.experts." in name or ".engram.embed." in name:
            continue
        _, _, off, n = ck.info(name)
        spans.append((os.path.join(ck.path, fname), off, off + n))
    spans.sort()
    merged: list[list] = []
    for f, a, b in spans:
        if merged and merged[-1][0] == f and a - merged[-1][2] <= _GAP:
            merged[-1][2] = max(merged[-1][2], b)
        else:
            merged.append([f, a, b])
    return [(f, a, b - a) for f, a, b in merged]


def _pieces(spans):
    for f, off, n in spans:
        for a in range(0, n, PIECE):
            yield f, off + a, min(PIECE, n - a)


class LoadAhead:
    """The pool over one model build of ``layers`` (in build order): ``building(layer)`` as each layer starts,
    ``close()`` after the last (stops the reads, waits for the ones running)."""

    def __init__(self, model, layers, n_threads: int, budget_bytes: int):
        from . import common

        self.model, self.layers = model, [int(i) for i in layers]
        self.n_threads, self.budget = int(n_threads), int(budget_bytes)
        self.pool = ThreadPoolExecutor(self.n_threads, thread_name_prefix="load-ahead")
        self.readers = threading.BoundedSemaphore(max(1, self.n_threads // 2))
        self.cv = threading.Condition()
        self.at = -1  # index in self.layers of the layer being built
        self.stop = False
        self.ahead: dict[int, int] = {}  # layer index -> bytes read (or being read) for it
        self.read_bytes, self.read_layers = 0, set()
        self.t0 = time.time()
        self.local = threading.local()
        common.set_stats_pool(self.pool)
        self.sched = threading.Thread(target=self._schedule, name="load-ahead-schedule", daemon=True)
        self.sched.start()

    @classmethod
    def start(cls, model, layers) -> "LoadAhead | None":
        n = threads()
        if n <= 0 or not layers:
            return None
        return cls(model, layers, n, int(float(settings.get("LOAD_AHEAD_GB")) * 1e9))

    def building(self, layer: int) -> None:
        with self.cv:
            self.at = self.layers.index(int(layer))
            self.cv.notify_all()

    def _room(self, k: int, n: int) -> bool:
        if k <= self.at + 1:  # (the layer being built and the next: always)
            return True
        past = sum(b for j, b in self.ahead.items() if j > self.at)
        return past + n <= self.budget

    def _schedule(self) -> None:
        try:
            for k, layer in enumerate(self.layers):
                spans = expert_files_spans(self.model, layer) + dense_ranges(self.model.ck, layer)
                for path, off, n in _pieces(spans):
                    with self.cv:
                        while not self.stop and k >= self.at and not self._room(k, n):
                            self.cv.wait(0.5)
                        if self.stop:
                            return
                        if k < self.at:  # (the build is past this layer: its reads are done)
                            break
                        self.ahead[k] = self.ahead.get(k, 0) + n
                    self.readers.acquire()
                    try:
                        self.pool.submit(self._read, k, path, off, n)
                    except RuntimeError:  # (the pool shut down under us: close() ran)
                        self.readers.release()
                        return
        except Exception as e:  # (load-ahead is only a speedup: the build reads what it needs itself)
            logger.warning(f"load-ahead: stopped reading ahead ({e!r}); the build reads its own files")

    def _read(self, k: int, path: str, off: int, n: int) -> None:
        try:
            buf = getattr(self.local, "buf", None)
            if buf is None:
                buf = self.local.buf = bytearray(PIECE)
            fd = os.open(path, os.O_RDONLY)
            try:
                view, pos, end = memoryview(buf), off, off + n
                while pos < end:
                    got = os.preadv(fd, [view[: end - pos]], pos)
                    if got <= 0:
                        break
                    pos += got
            finally:
                os.close(fd)
            with self.cv:
                self.read_bytes += n
                self.read_layers.add(k)
        except OSError as e:
            logger.debug(f"load-ahead: {path} @ {off}: {e!r}")
        finally:
            self.readers.release()

    def close(self) -> None:
        from . import common

        try:
            common.drain_stats()  # (BFP8_STATS whole before the model is handed out)
        finally:
            common.set_stats_pool(None)
            with self.cv:
                self.stop = True
                self.cv.notify_all()
            self.sched.join()
            self.pool.shutdown(wait=True)
        logger.info(
            f"load-ahead: {self.n_threads} threads read {self.read_bytes / 1e9:.1f} GB of {len(self.read_layers)} "
            f"layers' weight files into the page cache over the {time.time() - self.t0:.0f} s build"
        )


def expert_files_spans(model, layer: int) -> list[tuple[str, int, int]]:
    return [(str(f), 0, f.stat().st_size) for f in expert_files(model, layer)]
