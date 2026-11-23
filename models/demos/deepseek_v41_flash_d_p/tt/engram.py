# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Engram rows: the model's one host input besides the token ids (components.yaml engram_rows, tag CPU; owner rule).

Per chunk and engram layer, on the host: hash the chunk's natural-order token ids with the carried 3-id tail
(reference/ngram.py:EngramHasher, bit-equal to the checkpoint's inference/engram.py), read the memory-mapped fp8 rows
times their e8m0 group scales (reference/weights.py:EngramTable; never the whole 98 GB table), and send the result to
the device once: [1, 1, S/SP, 24 x 256] bf16 per chip, rows split over mesh axis 0 (SP), replicated over axis 1 (TP),
TILE, DRAM. bf16 holds e4m3 x 2^k exactly, so the device copy equals the reference's fp32 rows.

Rows at or past actual_end (engine pad, PAD_ID 0xFFFFFFFF) are not hashed: they are zero and never reach a cache;
the tail is the compressed ids of the last 3 real positions. The device keeps no engram state: the id history (tail)
is host state per slot, returned to the caller.

This runs in the device model's ``begin_chunk(tokens, start)`` (agent rule 5: a host input), never inside a layer.

Where the table is read from (ENGRAM_TABLE; bringup/ENGRAM_HOST_PATH.md): mmap = the checkpoint mapping with
MADV_RANDOM over the table's bytes (one 4 KB page a row; the kernel's read-ahead would fault in up to read_ahead_kb a
random 256-byte row: ~7 s a 5120-token chunk cold on the QB, so the mapping never reads ahead); resident = the table held in
anonymous memory from build on (4 KB pages unless ENGRAM_HUGEPAGES=on; the file pages it read dropped from the page cache;
with ENGRAM_BUILD=background built on one thread while the weights load, the first chunk waits for it): with
ENGRAM_ROWS=bfp8 already as bfloat8_b (272 B a row, exponents next to the magnitudes; tt/engram_bfp8.py), the chunk's
rows gathered straight into TILE bytes; with bf16 the 8-bit bytes. The gather is split over ENGRAM_GATHER_THREADS
Python threads (torch indexing releases the GIL). Every mode gives EngramTable.rows' values bit for bit.
"""

from __future__ import annotations

import hashlib
import mmap
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from loguru import logger

import ttnn

from .cores import from_torch
from .settings import settings

_HASHERS = {}  # model path -> EngramHasher (the compressed token map takes seconds to build)
_RESIDENT = {}  # (checkpoint path, tensor name) -> resident uint8 tensor (one copy a process, whoever builds it)
_RESIDENT_LOCK = threading.Lock()
_BUILDER = []  # ENGRAM_BUILD=background: the one thread a process that builds the resident tables, one after another
_BUILDER_LOCK = threading.Lock()  # (not _RESIDENT_LOCK: a build holds that for its whole run)
# ENGRAM_ASYNC=on (P.6): the worker thread's host -> device calls. They are begin_chunk's host input (agent rule 5),
# issued from the worker while the main thread dispatches layers; testing/host_transfers.py counts a layer's own
# transfers by wrapping the ttnn module attributes for the whole process (any thread), so the worker calls the
# functions bound here at import, and the counter keeps counting what the layer calls do themselves.
_COPY_TO_DEVICE = ttnn.copy_host_to_device_tensor
# tt-metal's pinned-write threshold (tt_metal/impl/tensor/pinned_upload.hpp k_pin_write_threshold_bytes): a host -> device
# tensor write of more bytes takes the pinned path, whose enqueue_write_shards(blocking=True) waits for the device to
# finish all queued work while holding the command queue (PB.3K, TtEngramRows.upload_pieces)
PIN_WRITE_BYTES = 32 * 1024 * 1024
_FROM_TORCH = ttnn.from_torch


def _host_impl():
    """ENGRAM_HOST_IMPL: the module that does the engram host work, numba (tt/engram_bfp8.py) or C++
    (tt/engram_host_cpp.py); same functions, same bytes (tests/test_engram_host_cpp.py)."""
    if settings.get("ENGRAM_HOST_IMPL") == "cpp":
        from . import engram_host_cpp as impl
    else:
        from . import engram_bfp8 as impl
    return impl


def _build_in_background(fn):
    """fn() on the builder thread (FIX.10: the resident tables build while the weights load) -> its Future."""
    with _BUILDER_LOCK:
        if not _BUILDER:
            _BUILDER.append(ThreadPoolExecutor(1, thread_name_prefix="engram-build"))
    return _BUILDER[0].submit(fn)


def engram_hasher(model_path: str, cfg):
    """One EngramHasher per checkpoint (from its tokenizer and config)."""
    key = str(model_path)
    if key not in _HASHERS:
        from transformers import AutoTokenizer

        from models.demos.deepseek_v41_flash_d_p.reference.ngram import EngramHasher

        _HASHERS[key] = EngramHasher.build(cfg.raw, AutoTokenizer.from_pretrained(key))
    return _HASHERS[key]


def _anonymous(nbytes: int, huge: bool):
    buf = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    if huge:
        try:
            buf.madvise(mmap.MADV_HUGEPAGE)
        except (AttributeError, OSError):
            pass
    return buf


def _drop_cache(fd: int, off: int, nbytes: int):
    """Drop a range of a file we read once from the page cache (FIX.10): the table bytes are never read again, and
    leaving ~200 GB of them cached evicts the weight cache the load streams next."""
    try:
        os.posix_fadvise(fd, off, nbytes, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def _shm_rows_path(ck, layer: int, nbytes: int) -> str | None:
    """Where ENGRAM_SHM_DIR keeps the layer's packed rows (None: off): named by the table's place in the checkpoint
    (file, dtype, shape, offset, bytes of weight and scale), so another checkpoint or row format never matches."""
    root = settings.get("ENGRAM_SHM_DIR")
    if root == "off":
        return None
    h = hashlib.sha1(repr(("bfp8 rows v1", nbytes)).encode())
    for name in _names(layer):
        h.update(repr((name, os.path.basename(ck.weight_map[name]), *map(str, ck.info(name)))).encode())
    return os.path.join(root, f"L{layer:02d}.bfp8.{h.hexdigest()[:12]}.rows")


def _shm_rows_file(path: str, nbytes: int):
    """(name, shared mapping) of a new ``nbytes`` file next to ``path`` for a build to fill, its space reserved; it is
    renamed to ``path`` once whole. (None, None), logged, when the file system has no room for it."""
    root = os.path.dirname(path)
    os.makedirs(root, exist_ok=True)
    st = os.statvfs(root)
    free = st.f_bavail * st.f_frsize
    if free < nbytes:
        logger.warning(
            f"{root}: {free / 1e9:.1f} GB free, the packed engram rows need {nbytes / 1e9:.1f} GB: building them in "
            f"anonymous memory (rebuilt every run)"
        )
        return None, None
    tmp = f"{path}.building.{os.getpid()}"
    fd = os.open(tmp, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.posix_fallocate(fd, 0, nbytes)
        return tmp, mmap.mmap(fd, nbytes, flags=mmap.MAP_SHARED)
    except OSError as e:
        os.unlink(tmp)
        logger.warning(f"{tmp}: cannot reserve {nbytes / 1e9:.1f} GB ({e}): building in anonymous memory")
        return None, None
    finally:
        os.close(fd)


def resident_bfp8(ck, layer: int, threads: int = 16, block: int = 1 << 20, huge: bool | None = None):
    """The layer's table as packed bfloat8_b rows [N, 272] uint8, once a process: mapped from ENGRAM_SHM_DIR when an
    earlier run left them there, else built -- into a new file there when it has room, else into anonymous memory --
    from blocks of ``block`` rows read with pread (``threads`` ahead; each block's file pages dropped from the page
    cache after it) and packed by tt/engram_bfp8.pack_into. ``huge``: MADV_HUGEPAGE on anonymous rows (default
    ENGRAM_HUGEPAGES)."""
    from collections import deque

    import numpy as np

    pack_into = _host_impl().pack_into

    key = (str(ck.path), "bfp8", layer)
    with _RESIDENT_LOCK:
        if key in _RESIDENT:
            return _RESIDENT[key]
        wname, sname = _names(layer)
        _, (n, d), woff, _ = ck.info(wname)
        _, (_, sd), soff, _ = ck.info(sname)
        width = d // 16 + d
        stored = _shm_rows_path(ck, layer, n * width)
        if stored is not None and os.path.isfile(stored) and os.path.getsize(stored) == n * width:
            t0 = time.perf_counter()
            fd = os.open(stored, os.O_RDWR)
            try:
                # Shared, so populating faults the tmpfs pages in instead of copying them.
                buf = mmap.mmap(fd, n * width, flags=mmap.MAP_SHARED | mmap.MAP_POPULATE)
            finally:
                os.close(fd)
            logger.info(
                f"engram layer {layer}: packed rows mapped from {stored} "
                f"({n * width / 1e9:.1f} GB in {time.perf_counter() - t0:.1f} s)"
            )
            t = torch.from_numpy(np.frombuffer(buf, dtype=np.uint8).reshape(n, width))
            t._engram_buf = buf  # (keeps the mapping alive with the tensor)
            _RESIDENT[key] = t
            return t
        building, buf = _shm_rows_file(stored, n * width) if stored is not None else (None, None)
        if buf is None:
            buf = _anonymous(n * width, settings.get("ENGRAM_HUGEPAGES") == "on" if huge is None else huge)
        t0 = time.perf_counter()
        packed = np.frombuffer(buf, dtype=np.uint8).reshape(n, width)
        fw = os.open(os.path.join(ck.path, ck.weight_map[wname]), os.O_RDONLY)
        fs = os.open(os.path.join(ck.path, ck.weight_map[sname]), os.O_RDONLY)

        def read(a):
            m = min(block, n - a)
            w, s = torch.empty(m, d, dtype=torch.uint8), torch.empty(m, sd, dtype=torch.uint8)
            assert os.preadv(fw, [w.numpy()], woff + a * d) == m * d
            assert os.preadv(fs, [s.numpy()], soff + a * sd) == m * sd
            _drop_cache(fw, woff + a * d, m * d)
            _drop_cache(fs, soff + a * sd, m * sd)
            return a, w, s

        try:
            with ThreadPoolExecutor(threads) as ex:
                starts, todo = iter(range(0, n, block)), deque()
                for a in starts:
                    todo.append(ex.submit(read, a))
                    if len(todo) >= threads:
                        break
                while todo:
                    a, w, s = todo.popleft().result()
                    nxt = next(starts, None)
                    if nxt is not None:
                        todo.append(ex.submit(read, nxt))
                    pack_into(w, s, packed[a : a + w.shape[0]], threads)
        except BaseException:
            if building is not None:
                os.unlink(building)
            raise
        finally:
            os.close(fw)
            os.close(fs)
        if building is not None:
            os.rename(building, stored)
        logger.info(
            f"engram layer {layer}: packed rows built in {time.perf_counter() - t0:.1f} s"
            + (f", kept in {stored} for the next run" if building is not None else "")
        )
        t = torch.from_numpy(packed)
        t._engram_buf = buf  # (keeps the mapping alive with the tensor)
        _RESIDENT[key] = t
        return t


def _warm_tiles(packed, threads: int):
    """One tiny tile_rows call: numba loads the gather kernel (~0.5 s on the first call; ENGRAM_HOST_IMPL=cpp: the
    extension is built / loaded) off the first chunk."""
    tile_rows = _host_impl().tile_rows

    tile_rows(packed, torch.zeros(32, 24, dtype=torch.int64), 32, threads)


def _names(layer: int):
    return (f"layers.{layer}.engram.embed.weight", f"layers.{layer}.engram.embed.scale")


def madvise_random(ck, layer: int):
    """MADV_RANDOM over the byte range of the layer's engram weight + scale in the checkpoint mapping (the rest of the
    file keeps its read-ahead)."""
    page = mmap.PAGESIZE
    for name in _names(layer):
        _, _, off, nbytes = ck.info(name)
        a = off - off % page
        ck._map(ck.weight_map[name])._mmap.madvise(mmap.MADV_RANDOM, a, off + nbytes - a)


def resident_copy(ck, name: str, readers: int = 16, huge: bool | None = None) -> torch.Tensor:
    """A stored tensor's bytes copied once into anonymous memory, as uint8 of its shape's rows (file pages dropped from
    the page cache after each read; ``huge``: MADV_HUGEPAGE, default ENGRAM_HUGEPAGES)."""
    key = (str(ck.path), name)
    with _RESIDENT_LOCK:
        if key in _RESIDENT:
            return _RESIDENT[key]
        _, shape, off, nbytes = ck.info(name)
        buf = _anonymous(nbytes, settings.get("ENGRAM_HUGEPAGES") == "on" if huge is None else huge)
        fd = os.open(os.path.join(ck.path, ck.weight_map[name]), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, off, nbytes, os.POSIX_FADV_SEQUENTIAL)
            step, view = 64 << 20, memoryview(buf)

            def read(k):
                a = k * step
                n = min(step, nbytes - a)
                assert os.preadv(fd, [view[a : a + n]], off + a) == n
                _drop_cache(fd, off + a, n)

            with ThreadPoolExecutor(readers) as ex:
                list(ex.map(read, range((nbytes + step - 1) // step)))
        finally:
            os.close(fd)
        t = torch.frombuffer(buf, dtype=torch.uint8).view(shape[0], -1)
        t._engram_buf = buf  # (keeps the mapping alive with the tensor)
        _RESIDENT[key] = t
        return t


class TtEngramRows:
    """Host hash + table lookup for one engram layer, and the one host -> device transfer of its rows."""

    def __init__(
        self,
        mesh,
        hasher,
        table,
        layer: int,
        mode: str | None = None,
        threads: int | None = None,
        rows_format: str | None = None,
    ):
        self.mesh, self.hasher, self.table, self.layer = mesh, hasher, table, layer
        self.width = hasher.tail_len * hasher.n_heads * table.dim  # 3 x 8 x 256 = 6144
        self.vocab = int(hasher.token_map.shape[0])
        self.mode = mode or settings.get("ENGRAM_TABLE")
        self.threads = threads or settings.get("ENGRAM_GATHER_THREADS")
        self.format = rows_format or settings.get("ENGRAM_ROWS")
        self.weight, self.scale, self._packed = table.weight, table.scale, None
        self._staged = None  # ENGRAM_ASYNC: the bfp8 staging host tensor (_staging)
        self._staged_tp = None  # HOST_HIDE=on (P.78): the per-chip staging host tensor (_staging_tp)
        self._lane_outs = {}  # PB.3 (PACKED=on): (id(outs), first chip, chips) -> (outs, the lane's share views)
        ck = getattr(table, "ckpt", None)
        if self.mode == "mmap":
            madvise_random(ck, layer)
        elif self.mode == "resident" and self.format == "bfp8":

            def build():
                packed = resident_bfp8(ck, layer, max(self.threads, 16)).numpy()
                _warm_tiles(packed, self.threads)  # (the compiled gather loaded here, not in the first chunk)
                return packed

            if settings.get("ENGRAM_BUILD") == "background":
                self._packed = _build_in_background(build)
            else:
                self._packed = build()
        elif self.mode == "resident":
            self.weight, self.scale = (resident_copy(ck, n) for n in _names(layer))
        self.pool = ThreadPoolExecutor(self.threads) if self.threads > 1 else None

    @property
    def packed(self):
        """The resident bfp8 rows [N, 272] (numpy uint8), else None; ENGRAM_BUILD=background: waits for the build."""
        p = self._packed
        if p is not None and not hasattr(p, "shape"):
            p = self._packed = p.result()
        return p

    def gather(self, ids: torch.Tensor):
        """The 8-bit table bytes of ids [...] in their order: (e4m3 [n, dim], e8m0 [n, dim / 32]) uint8, the gather
        split over self.threads, from self.weight / scale."""
        flat = ids.reshape(-1)
        n = flat.numel()
        assert int(flat.min()) >= 0 and int(flat.max()) < self.table.num_rows, "engram row out of range"
        order = torch.argsort(flat)  # sorted reads: neighbouring rows share pages
        sids = flat[order]
        w = torch.empty(n, self.table.dim, dtype=torch.uint8)
        s = torch.empty(n, self.scale.shape[1], dtype=torch.uint8)
        k = max(1, min(self.threads, n))
        edges = [n * j // k for j in range(k + 1)]

        def part(j):
            a, b = edges[j], edges[j + 1]
            w[order[a:b]] = self.weight[sids[a:b]]
            s[order[a:b]] = self.scale[sids[a:b]]

        if self.pool is None or k == 1:
            part(0) if k == 1 else [part(j) for j in range(k)]
        else:
            list(self.pool.map(part, range(k)))
        return w, s

    def rows(self, ids: torch.Tensor) -> torch.Tensor:
        """EngramTable.rows(ids), bit for bit."""
        from models.demos.deepseek_v41_flash_d_p.reference.weights import BLOCK

        w, s = self.gather(ids)
        n = w.shape[0]
        v = w.view(torch.float8_e4m3fn).float().view(n, -1, BLOCK)
        v = v * s.view(torch.float8_e8m0fnu).float().unsqueeze(-1)
        return v.view(*ids.shape, self.table.dim)

    def new_tail(self) -> torch.Tensor:
        return self.hasher.new_tail()

    def _hashes(self, ids: torch.Tensor, tail: torch.Tensor, fast: bool = False, threads: int | None = None):
        """The table rows of ids [valid] with the carried tail -> (rows [valid, slots] int64, new tail). ``fast``
        (HOST_HIDE=on, P.78): the numba form (tt/engram_bfp8.ngram_hashes, bit-equal; tests/test_engram_hash.py)."""
        if fast:
            return _host_impl().ngram_hashes(self.hasher, ids, self.layer, tail.long(), threads or self.threads)
        return self.hasher.hashes(ids, self.layer, tail.long())

    def _valid(self, ids: torch.Tensor, valid: int | None) -> int:
        s = ids.shape[0]
        if valid is None:  # the engine pads past actual_end with 0xFFFFFFFF (or any id outside the vocabulary)
            bad = ((ids < 0) | (ids >= self.vocab)).nonzero()
            valid = int(bad[0]) if bad.numel() else s
        return max(0, min(int(valid), s))

    def host_rows(self, tokens: torch.Tensor, tail: torch.Tensor, valid: int | None = None):
        """tokens [S] natural order (pad past ``valid``), tail [3] -> (rows [S, width] bf16, new tail [3])."""
        ids = tokens.reshape(-1).long()
        s = ids.shape[0]
        valid = self._valid(ids, valid)
        rows = torch.zeros(s, self.width, dtype=torch.bfloat16)
        if valid == 0:
            return rows, tail.clone()
        h, new_tail = self.hasher.hashes(ids[:valid], self.layer, tail.long())
        rows[:valid] = self.rows(h).flatten(1).to(torch.bfloat16)  # exact (e4m3 x 2^k)
        return rows, new_tail

    def to_device(self, rows: torch.Tensor) -> ttnn.Tensor:
        """Host rows [S, width] -> device [1, 1, S/SP, width] bf16 per chip (rows over axis 0, replicated over 1)."""
        s, w = rows.shape
        sp = self.mesh.shape[0]
        assert s % sp == 0, f"{s} rows do not split over {sp} mesh rows"
        return from_torch(
            rows.reshape(1, 1, s, w).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(2, None)),
        )

    def host_bfp8(
        self,
        tokens: torch.Tensor,
        tail: torch.Tensor,
        valid: int | None = None,
        times: dict | None = None,
        threads: int | None = None,
        fast: bool = False,
    ):
        """tokens [S], tail [3] -> (exponents [S, width / 16] uint8, magnitudes [S, width] uint8, new tail): the rows
        as bfloat8_b (tt/engram_bfp8.py), from the table's bytes by integer operations only; pad rows are 0.
        ``times``: ms of hash / lookup / pack added (ENGRAM_ASYNC's report). ``threads`` / ``fast``: HOST_HIDE (P.78),
        the pack's thread count (default ENGRAM_GATHER_THREADS) and the numba hash."""
        pack_rows = _host_impl().pack_rows

        t0 = time.perf_counter()
        ids = tokens.reshape(-1).long()
        s = ids.shape[0]
        valid = self._valid(ids, valid)
        exps = torch.zeros(s, self.width // 16, dtype=torch.uint8)
        mags = torch.zeros(s, self.width, dtype=torch.uint8)
        if valid == 0:
            return exps, mags, tail.clone()
        h, new_tail = self._hashes(ids[:valid], tail, fast, threads)
        t1 = time.perf_counter()
        w, sc = self.gather(h)
        t2 = time.perf_counter()
        e, m = pack_rows(w, sc, threads=threads or self.threads)
        exps[:valid], mags[:valid] = e.view(valid, -1), m.view(valid, -1)
        if times is not None:
            times.update(hash=(t1 - t0) * 1e3, lookup=(t2 - t1) * 1e3, pack=(time.perf_counter() - t2) * 1e3)
        return exps, mags, new_tail

    def to_device_bfp8(self, exps: torch.Tensor, mags: torch.Tensor) -> ttnn.Tensor:
        """bfloat8_b rows -> device [1, 1, S/SP, width] bfloat8_b per chip (rows over axis 0, replicated over 1): each
        mesh row's TILE bytes wrapped as a host tensor (no float conversion), one upload."""
        from .engram_bfp8 import tile_bytes

        s, w = mags.shape
        rows_ax, cols_ax = tuple(self.mesh.shape)
        assert s % (32 * rows_ax) == 0, f"{s} rows do not split into tiles over {rows_ax} mesh rows"
        n = s // rows_ax
        wrap = ttnn.experimental.disaggregation.tensor_from_bfp8_bytes
        shards = []
        for r in range(rows_ax):
            raw = tile_bytes(exps[r * n : (r + 1) * n], mags[r * n : (r + 1) * n]).numpy().tobytes()
            shards += [wrap(raw, [1, 1, n, w])] * cols_ax
        host = ttnn.from_host_shards(shards, self.mesh.shape)
        return ttnn.to_device(host, self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def tiles_from_packed(
        self,
        tokens: torch.Tensor,
        tail: torch.Tensor,
        valid: int | None = None,
        times=None,
        outs=None,
        threads: int | None = None,
        fast: bool = False,
    ):
        """Resident bfp8 table: tokens [S], tail [3] -> (per mesh row the TILE bytes of its [S/SP, width] rows, new
        tail); the bytes equal tile_bytes(*host_bfp8(...)) of each mesh row's share. ``times``: ms of hash and of
        lookup + pack (one numba pass: the resident rows are already bfloat8_b) added; ``outs``: per mesh row the
        uint8 tensor to write the bytes into (ENGRAM_ASYNC's staging). ``threads`` / ``fast``: HOST_HIDE (P.78), the
        gather's thread count (default ENGRAM_GATHER_THREADS) and the numba hash."""
        tile_rows = _host_impl().tile_rows

        t0 = time.perf_counter()
        packed = self.packed  # (ENGRAM_BUILD=background: the first chunk waits for the build here, not in "hash")
        t0b = time.perf_counter()
        ids = tokens.reshape(-1).long()
        s = ids.shape[0]
        valid = self._valid(ids, valid)
        rows_ax = tuple(self.mesh.shape)[0] if self.mesh is not None else 1
        assert s % (32 * rows_ax) == 0, f"{s} rows do not split into tiles over {rows_ax} mesh rows"
        n = s // rows_ax
        if valid:
            h, new_tail = self._hashes(ids[:valid], tail, fast, threads)
        else:
            h, new_tail = torch.empty(0, self.width // self.table.dim, dtype=torch.int64), tail.clone()
        assert valid == 0 or (int(h.min()) >= 0 and int(h.max()) < self.table.num_rows), "engram row out of range"
        t1 = time.perf_counter()
        outs = outs or [None] * rows_ax
        th = threads or self.threads
        tiles = [tile_rows(packed, h[r * n : (r + 1) * n], n, th, outs[r]) for r in range(rows_ax)]
        if times is not None:
            times.update(
                build_wait=(t0b - t0) * 1e3, hash=(t1 - t0b) * 1e3, lookup_pack=(time.perf_counter() - t1) * 1e3
            )
        return tiles, new_tail

    def upload_tiles(self, tiles: list) -> ttnn.Tensor:
        """Per mesh row TILE bytes -> device [1, 1, S/SP, width] bfloat8_b per chip, replicated over axis 1."""
        cols_ax = tuple(self.mesh.shape)[1]
        n = tiles[0].numel() * 16 // 17 // self.width
        wrap = ttnn.experimental.disaggregation.tensor_from_bfp8_bytes
        shards = [x for t in tiles for x in [wrap(t.numpy().tobytes(), [1, 1, n, self.width])] * cols_ax]
        return ttnn.to_device(
            ttnn.from_host_shards(shards, self.mesh.shape), self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    # ------------------------------------------------------------------ ENGRAM_ASYNC=on (P.6)
    def allocate(self, s: int) -> ttnn.Tensor:
        """Main thread, in begin_chunk: the chunk's device rows tensor (the allocation the off path's upload makes, in
        the same order), filled later by ``fill`` on the worker."""
        rows_ax = tuple(self.mesh.shape)[0]
        assert s % (32 * rows_ax) == 0, f"{s} rows do not split into tiles over {rows_ax} mesh rows"
        dtype = ttnn.bfloat8_b if self.format == "bfp8" else ttnn.bfloat16
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, s // rows_ax, self.width]), dtype, ttnn.TILE_LAYOUT, self.mesh, ttnn.DRAM_MEMORY_CONFIG
        )

    def next_tail(self, tokens: torch.Tensor, tail: torch.Tensor, valid: int | None = None) -> torch.Tensor:
        """The tail the chunk leaves (what host_rows / host_bfp8 / tiles_from_packed return), without the hash: the
        compressed ids of the last 3 real positions, the old tail's where the chunk has fewer."""
        ids = tokens.reshape(-1).long()
        valid = self._valid(ids, valid)
        if valid == 0:
            return tail.clone()
        last = ids[max(0, valid - self.hasher.tail_len) : valid]
        return torch.cat([tail.long(), self.hasher.token_map[last]])[-self.hasher.tail_len :].clone()

    def _staging(self, n: int):
        """ENGRAM_ASYNC, bfp8 rows: (the mesh host tensor of [1, 1, n, width] bfloat8_b per chip, per mesh row a
        writable uint8 view of its TILE bytes), made once a layer. The mesh tensor's shards share the mesh rows'
        bytes (replicated over axis 1, as ``upload_tiles``), so the worker writes the chunk's bytes in place: no
        Python bytes copy and no wrap copy a chunk (both hold the GIL: ~35 ms a 5120-token layer on the Galaxy host).
        Reused every chunk: copy_host_to_device_tensor of an unpinned host buffer copies it into the command queue
        before it returns (fd_mesh_command_queue.cpp:write_shard_to_device, pinned_memory null)."""
        if self._staged is None or self._staged[0] != n:
            rows_ax, cols_ax = tuple(self.mesh.shape)
            nbytes = n * self.width * 17 // 16
            wrap = ttnn.experimental.disaggregation.tensor_from_bfp8_bytes
            rows = [wrap(bytes(nbytes), [1, 1, n, self.width]) for _ in range(rows_ax)]
            host = ttnn.from_host_shards([x for t in rows for x in [t] * cols_ax], self.mesh.shape)
            dist = host.host_buffer()
            bufs = [dist.get_shard(ttnn.MeshCoordinate(r, 0)) for r in range(rows_ax)]
            views = [torch.from_dlpack(b) for b in bufs]  # (the shard's own bytes, no copy)
            assert all(v.numel() == nbytes and v.dtype == torch.uint8 for v in views), [v.shape for v in views]
            self._staged = (n, host, views, (rows, bufs))
        return self._staged[1], self._staged[2]

    def host_tensor(
        self,
        tokens: torch.Tensor,
        tail: torch.Tensor,
        valid: int | None = None,
        times=None,
        threads: int | None = None,
        fast: bool = False,
    ):
        """Everything but the upload: (the multi-device host tensor of the chunk's rows, new tail); the bytes the off
        path's ``__call__`` sends. ``times``: per phase ms added (hash, lookup, pack / lookup_pack, wrap).
        ``threads`` / ``fast``: HOST_HIDE (P.78; the bfp8 rows only): the gather / pack threads and the numba hash."""
        s = tokens.reshape(-1).shape[0]
        rows_ax = tuple(self.mesh.shape)[0]
        if self.format != "bfp8":
            rows, new_tail = self.host_rows(tokens, tail, valid)
            t0 = time.perf_counter()
            host = _FROM_TORCH(
                rows.reshape(1, 1, s, self.width).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(2, None)),
            )
            if times is not None:
                times["wrap"] = (time.perf_counter() - t0) * 1e3
            return host, new_tail
        assert s % (32 * rows_ax) == 0, f"{s} rows do not split into tiles over {rows_ax} mesh rows"
        host, views = self._staging(s // rows_ax)
        if self._packed is not None:
            _, new_tail = self.tiles_from_packed(tokens, tail, valid, times, outs=views, threads=threads, fast=fast)
        else:
            from .engram_bfp8 import tile_bytes

            exps, mags, new_tail = self.host_bfp8(tokens, tail, valid, times, threads=threads, fast=fast)
            t0 = time.perf_counter()
            n = s // rows_ax
            for r in range(rows_ax):
                views[r].copy_(tile_bytes(exps[r * n : (r + 1) * n], mags[r * n : (r + 1) * n]))
            if times is not None:
                times["pack"] = times.get("pack", 0.0) + (time.perf_counter() - t0) * 1e3
        return host, new_tail

    def fill(self, dev: ttnn.Tensor, tokens, tail, valid=None, want_tail=None, threads=None, fast=False) -> dict:
        """Worker thread: the chunk's rows into ``dev`` (allocated by ``allocate`` on the main thread) -> per phase
        ms. The write goes into CQ 0 (every mesh this model opens has one CQ: the harness's device_params,
        tt-d-gen's runner_utils.open_mesh_device), in order, so it lands before the ops the main thread enqueues
        after waiting for this job. ``want_tail``: the tail begin_chunk already stored (checked). ``threads`` /
        ``fast``: HOST_HIDE (P.78), see host_tensor."""
        times = {}
        host, new_tail = self.host_tensor(tokens, tail, valid, times, threads=threads, fast=fast)
        assert want_tail is None or torch.equal(new_tail.long(), want_tail.long()), (self.layer, new_tail, want_tail)
        t0 = time.perf_counter()
        _COPY_TO_DEVICE(host, dev)  # (GIL released for the copy)
        times["upload"] = (time.perf_counter() - t0) * 1e3
        return times

    # ------------------------------------------------------------------ HOST_HIDE=on (P.78): rows sharded over TP
    def allocate_tp(self, s: int, pieces: int = 1):
        """Main thread, in begin_chunk: the chunk's device rows, each chip its own share [1, 1, S/(SP TP), width]
        bfloat8_b (chip (r, c): rows [(r TP + c) q, (r TP + c + 1) q) of the chunk, q = S / (SP TP)); ``fill_tp``
        writes them, the model gathers them over TP into the replicated layout ``allocate`` gives (a data move).
        ``pieces`` > 1 (PACKED_HOST_FAST=on, ``upload_pieces``): a list of that many [1, 1, q / pieces, width] tensors,
        the share's row blocks in order (TtDS41Model.take_engram_rows concatenates them on the device)."""
        rows_ax, cols_ax = tuple(self.mesh.shape)
        assert self.format == "bfp8", "HOST_HIDE's sharded upload sends bfloat8_b rows (ENGRAM_ROWS=bfp8)"
        assert s % (32 * rows_ax * cols_ax * pieces) == 0, f"{s} rows do not split into tiles over the chips / pieces"

        def one(rows: int) -> ttnn.Tensor:
            return ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, rows, self.width]),
                ttnn.bfloat8_b,
                ttnn.TILE_LAYOUT,
                self.mesh,
                ttnn.DRAM_MEMORY_CONFIG,
            )

        q = s // (rows_ax * cols_ax)
        return one(q) if pieces == 1 else [one(q // pieces) for _ in range(pieces)]

    def upload_pieces(self, s: int) -> int:
        """PACKED_HOST_FAST=on (PB.3K, main thread, read per chunk): the number of uploads the TP-sharded rows of an
        S-row chunk go up in, each below tt-metal's pinned-write threshold (PIN_WRITE_BYTES). One upload above it takes
        the pinned path, a BLOCKING write: enqueue_write_shards(blocking=True) waits for the device to finish
        everything queued before it while holding the command queue's lock, so the main thread's dispatch stalls for
        the device's whole backlog (PB.3K: 6.3-6.9 ms a 53.5 MB upload inside a packed chunk, 0.7 ms on an idle
        device) and the device then idles until the host refills it. Below it the write is non-blocking. 1: off, an
        upload that fits already, the mmap table (no resident staging) or rows that do not split into whole tiles."""
        if settings.get("PACKED_HOST_FAST") != "on" or self._packed is None:
            return 1
        n = int(self.mesh.get_num_devices())
        q = s // n
        total = q * self.width * 17 // 16 * n
        p = 1
        while total > p * PIN_WRITE_BYTES:
            p *= 2
        return p if p > 1 and q % (32 * p) == 0 else 1

    def _staging_tp(self, q: int, pieces: int = 1):
        """HOST_HIDE=on: (the mesh host tensor of [1, 1, q, width] bfloat8_b per chip, every chip its own bytes; per chip
        in row-major mesh order a writable uint8 view, and the same views as a tuple of numpy arrays), made once a
        layer (as ``_staging``: no copy a chunk). ``pieces`` > 1 (``upload_pieces``): a list of that many mesh host
        tensors of [1, 1, q / pieces, width] (the share's row blocks in order), and the views / arrays chip-major
        (chip k's block j at k pieces + j: the host jobs fill out k pieces + j with rows [k q + j q', ..), q' =
        q / pieces, i.e. chip k's rows in order)."""
        if self._staged_tp is None or self._staged_tp[0] != (q, pieces):
            n_chips = int(self.mesh.get_num_devices())
            qp = q // pieces
            nbytes = qp * self.width * 17 // 16
            wrap = ttnn.experimental.disaggregation.tensor_from_bfp8_bytes
            rows_ax, cols_ax = tuple(self.mesh.shape)
            hosts, keep, per = [], [], []
            for _ in range(pieces):
                shards = [wrap(bytes(nbytes), [1, 1, qp, self.width]) for _ in range(n_chips)]
                host = ttnn.from_host_shards(shards, self.mesh.shape)
                dist = host.host_buffer()
                bufs = [dist.get_shard(ttnn.MeshCoordinate(r, c)) for r in range(rows_ax) for c in range(cols_ax)]
                hosts.append(host)
                keep.append((shards, bufs))
                per.append([torch.from_dlpack(b) for b in bufs])  # (the shard's own bytes, no copy)
            views = [per[j][k] for k in range(n_chips) for j in range(pieces)]
            assert all(v.numel() == nbytes and v.dtype == torch.uint8 for v in views), [v.shape for v in views]
            assert len({v.data_ptr() for v in views}) == n_chips * pieces, "a chip's staging bytes are shared"
            host = hosts[0] if pieces == 1 else hosts
            self._staged_tp = ((q, pieces), host, views, tuple(v.numpy() for v in views), keep)
        return self._staged_tp[1:4]

    def fill_tp(self, dev: ttnn.Tensor, tokens, tail, valid=None, want_tail=None, threads=None) -> dict:
        """Worker thread, HOST_HIDE=on (P.78): ``fill`` into ``allocate_tp``'s rows: the numba hash, one compiled
        gather + tiling call for every chip's share (resident table; mmap: host_bfp8 then the tile bytes per share)
        and one upload of 1 / TP of ``fill``'s bytes (each chip its own share; ``fill`` sends every mesh row's rows to
        all TP chips of the row). The bytes of chip (r, c) are ``fill``'s rows [c q, (c+1) q) of mesh row r."""
        from .engram_bfp8 import tile_bytes

        tile_rows_chips = _host_impl().tile_rows_chips
        times = {}
        t0 = time.perf_counter()
        if self._packed is not None and settings.get("ENGRAM_HOST_IMPL") == "cpp":
            # ENGRAM_HOST_IMPL=cpp: the whole host job in ONE C++ call (valid prefix, tail check, hash, range check,
            # gather into every chip's share), no torch operation around it: the GIL is taken back once after it and
            # once after the upload (tt/engram_host_cpp.fill_job; the bytes the numba branch below writes)
            from .engram_host_cpp import fill_job

            q = tokens.numel() // int(self.mesh.get_num_devices())
            host, views, outs = self._staging_tp(q)
            packed = self.packed
            t0b = time.perf_counter()
            _, t = fill_job(
                self.hasher, self.layer, tokens, tail, packed, outs, q, threads or self.threads, valid, want_tail
            )
            times.update(build_wait=(t0b - t0) * 1e3, **t)
            t0 = time.perf_counter()
            _COPY_TO_DEVICE(host, dev)  # (GIL released for the copy)
            times["upload"] = (time.perf_counter() - t0) * 1e3
            return times
        ids = tokens.reshape(-1).long()
        s = ids.shape[0]
        valid = self._valid(ids, valid)
        n_chips = int(self.mesh.get_num_devices())
        q = s // n_chips
        host, views, outs = self._staging_tp(q)
        if self._packed is not None:
            packed = self.packed
            t0b = time.perf_counter()
            if valid:
                h, new_tail = self._hashes(ids[:valid], tail, fast=True, threads=threads)
            else:
                h, new_tail = torch.empty(0, self.width // self.table.dim, dtype=torch.int64), tail.clone()
            t1 = time.perf_counter()
            tile_rows_chips(packed, h, valid, outs, q, threads or self.threads)  # (the row range check: in the kernel)
            times.update(
                build_wait=(t0b - t0) * 1e3, hash=(t1 - t0b) * 1e3, lookup_pack=(time.perf_counter() - t1) * 1e3
            )
        else:
            exps, mags, new_tail = self.host_bfp8(tokens, tail, valid, times, threads=threads, fast=True)
            t1 = time.perf_counter()
            for k in range(n_chips):
                views[k].copy_(tile_bytes(exps[k * q : (k + 1) * q], mags[k * q : (k + 1) * q]))
            times["pack"] = times.get("pack", 0.0) + (time.perf_counter() - t1) * 1e3
        assert want_tail is None or torch.equal(new_tail.long(), want_tail.long()), (self.layer, new_tail, want_tail)
        t0 = time.perf_counter()
        _COPY_TO_DEVICE(host, dev)  # (GIL released for the copy)
        times["upload"] = (time.perf_counter() - t0) * 1e3
        return times

    def fill_tp_lanes(self, dev: ttnn.Tensor, lanes: list, threads=None) -> dict:
        """Worker thread, PB.3 (PACKED=on): ``fill_tp`` of a packed chunk. ``lanes``: per lane (first chip, chips,
        the lane's ids [lane rows], its slot's tail, valid rows, the tail it leaves or None): each lane hashed with its
        own tail into its chips' shares (rows past valid zero, as a single chunk's pad rows), by ``fill_tp``'s own
        branch (ENGRAM_HOST_IMPL=cpp with the resident table: the C++ job; resident: numba hash + tile_rows_chips;
        mmap: host_bfp8 + tile_bytes); then one upload, as ``fill_tp``."""
        from .engram_bfp8 import tile_bytes
        from .engram_host_cpp import fill_job

        cpp = self._packed is not None and settings.get("ENGRAM_HOST_IMPL") == "cpp"
        times = {}
        t0 = time.perf_counter()
        n_chips = int(self.mesh.get_num_devices())
        assert sum(nc for _, nc, *_ in lanes) == n_chips, [(c0, nc) for c0, nc, *_ in lanes]
        q = lanes[0][2].numel() // lanes[0][1]
        p = len(dev) if isinstance(dev, list) else 1  # (PB.3K: the upload in pieces, ``upload_pieces``)
        assert p == 1 or self._packed is not None, "upload pieces: the resident table's paths only"
        host, views, outs = self._staging_tp(q, p)
        if p > 1:  # (each chip's share as p outputs of q / p rows, in order: the jobs fill output k p + j)
            lanes = [(c0 * p, nc * p, *rest) for c0, nc, *rest in lanes]
            q //= p
        packed = self.packed if self._packed is not None else None
        t0b = time.perf_counter()
        agg = {}
        for c0, nc, ids, tail, valid, want in lanes:
            assert ids.numel() == nc * q, (ids.numel(), nc, q)
            key = (id(outs), c0, nc)
            sub = self._lane_outs.get(key)
            if sub is None or sub[0] is not outs:
                sub = self._lane_outs[key] = (outs, tuple(outs[c0 : c0 + nc]))
            if cpp:
                _, t = fill_job(
                    self.hasher, self.layer, ids, tail, packed, sub[1], q, threads or self.threads, valid, want
                )
                for k, v in t.items():
                    agg[k] = agg.get(k, 0.0) + v
                continue
            if packed is not None:
                if valid:
                    h, new_tail = self._hashes(ids[:valid], tail, fast=True, threads=threads)
                else:
                    h, new_tail = torch.empty(0, self.width // self.table.dim, dtype=torch.int64), tail.clone()
                _host_impl().tile_rows_chips(packed, h, valid, sub[1], q, threads or self.threads)
            else:
                exps, mags, new_tail = self.host_bfp8(ids, tail, valid, None, threads=threads, fast=True)
                for j in range(nc):
                    views[c0 + j].copy_(tile_bytes(exps[j * q : (j + 1) * q], mags[j * q : (j + 1) * q]))
            assert want is None or torch.equal(new_tail.long(), want.long()), (self.layer, new_tail, want)
        times.update(build_wait=(t0b - t0) * 1e3, **agg)
        t0 = time.perf_counter()
        if p == 1:
            _COPY_TO_DEVICE(host, dev)  # (GIL released for the copy)
        else:  # (each below the pinned-write threshold: non-blocking writes, ``upload_pieces``)
            for h, d in zip(host, dev):
                _COPY_TO_DEVICE(h, d)
        times["upload"] = (time.perf_counter() - t0) * 1e3
        return times

    def fill_tp_glued(self, dev, segs: list, chunk: int, threads=None) -> dict:
        """Worker thread, PB.7 (PACKED_GLUED=on): ``fill_tp`` of a glued chunk (tt/packed.GluedChunk). ``segs``: per
        segment (model row offset, its ids [rows], its slot's tail, the tail it leaves or None), rows back to back from
        row 0, every offset and row count whole tiles (32). A segment's rows may start inside a chip's share and run
        over several chips: the staging's shares are cut into tile rows (a share's bytes are its tile rows in order:
        TILE layout, tiles row-major), and each segment's job (fill_tp's own branch: ENGRAM_HOST_IMPL=cpp's C++ job, or
        numba's hash + tile_rows_chips; ENGRAM_TABLE=mmap: host_bfp8's rows as tile bytes) writes its tile rows only,
        hashed with its own tail; the tail pad's tile rows are written zero (no valid row). Then one upload, as
        ``fill_tp``."""
        from . import engram_host_cpp
        from .engram_bfp8 import tile_bytes
        from .engram_host_cpp import fill_job

        resident = self._packed is not None
        cpp = resident and settings.get("ENGRAM_HOST_IMPL") == "cpp"
        times = {}
        t0 = time.perf_counter()
        n_chips = int(self.mesh.get_num_devices())
        q = chunk // n_chips
        p = len(dev) if isinstance(dev, list) else 1
        host, views, outs = self._staging_tp(q, p)
        qp = q // p
        assert qp % 32 == 0, (q, p)
        key = (id(outs), "tilerows")
        hit = self._lane_outs.get(key)
        if hit is None or hit[0] is not outs:  # (every tile row of the chunk, model order: made once a staging)
            per = qp // 32
            rows = tuple(o[j * (o.size // per) : (j + 1) * (o.size // per)] for o in outs for j in range(per))
            hit = self._lane_outs[key] = (outs, rows)
        tiles = hit[1]
        assert len(tiles) * 32 == chunk, (len(tiles), chunk)
        packed = self.packed if resident else None
        agg = {}
        jobs = list(segs)
        used = sum(len(ids) for _, ids, *_ in segs)
        if used < chunk:  # (the tail pad: zero rows)
            pad = torch.full((chunk - used,), 0, dtype=torch.int64)
            jobs.append((used, pad, self.new_tail().long().contiguous(), None, 0))
        for job in jobs:
            off, ids, tail, want = job[:4]
            valid = job[4] if len(job) > 4 else None
            n = int(ids.numel())
            if n == 0:
                continue
            assert off % 32 == 0 and n % 32 == 0, (off, n)
            sub = tiles[off // 32 : (off + n) // 32]
            v = n if valid is None else int(valid)
            if cpp:
                _, t = fill_job(
                    self.hasher, self.layer, ids.contiguous(), tail, packed, sub, 32, threads or self.threads, v, want
                )
                engram_host_cpp._OUTS.pop(id(sub), None)  # (a tuple of this call only)
                for k2, x in t.items():
                    agg[k2] = agg.get(k2, 0.0) + x
                continue
            if not resident:  # (mmap: host_bfp8's rows, as fill_tp_lanes, copied tile row by tile row)
                if v:
                    exps, mags, new_tail = self.host_bfp8(ids, tail, v, None, threads=threads, fast=True)
                    b = tile_bytes(exps, mags).numpy()
                    tr = b.size // len(sub)
                    for j, dst in enumerate(sub):
                        dst[:] = b[j * tr : (j + 1) * tr]
                else:
                    new_tail = tail.clone()
                    for dst in sub:
                        dst[:] = 0
                assert want is None or torch.equal(new_tail.long(), want.long()), (self.layer, new_tail, want)
                continue
            if v:
                h, new_tail = self._hashes(ids[:v], tail, fast=True, threads=threads)
            else:
                h, new_tail = torch.empty(0, self.width // self.table.dim, dtype=torch.int64), tail.clone()
            _host_impl().tile_rows_chips(packed, h, v, sub, 32, threads or self.threads)
            assert want is None or torch.equal(new_tail.long(), want.long()), (self.layer, new_tail, want)
        times.update(build_wait=0.0, **agg)
        times["fill"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        if p == 1:
            _COPY_TO_DEVICE(host, dev)
        else:
            for h_, d in zip(host, dev):
                _COPY_TO_DEVICE(h_, d)
        times["upload"] = (time.perf_counter() - t0) * 1e3
        return times

    def __call__(self, tokens: torch.Tensor, tail: torch.Tensor, valid: int | None = None):
        """begin_chunk work for this layer: (device rows [1, 1, S/SP, width] in ENGRAM_ROWS' format, new tail)."""
        if self._packed is not None:
            tiles, new_tail = self.tiles_from_packed(tokens, tail, valid)
            return self.upload_tiles(tiles), new_tail
        if self.format == "bfp8":
            exps, mags, new_tail = self.host_bfp8(tokens, tail, valid)
            return self.to_device_bfp8(exps, mags), new_tail
        rows, new_tail = self.host_rows(tokens, tail, valid)
        return self.to_device(rows), new_tail


def build_engram_rows(mesh, ck, cfg, layer: int, hasher=None) -> TtEngramRows:
    """``hasher``: an EngramHasher already built (the reference's), else one is built from the checkpoint."""
    from models.demos.deepseek_v41_flash_d_p.reference.weights import EngramTable

    assert cfg.is_engram(layer), f"layer {layer} has no engram"
    return TtEngramRows(mesh, hasher or engram_hasher(ck.path, cfg), EngramTable(ck, layer), layer)
