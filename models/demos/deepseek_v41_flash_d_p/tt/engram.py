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

Where the table is read from (ENGRAM_TABLE; bringup/ENGRAM_HOST_PATH.md): mmap = the checkpoint mapping with the
kernel's read-ahead (a random 256-byte row faults in up to read_ahead_kb: ~7 s a 5120-token chunk cold on the QB);
random = the same mapping with MADV_RANDOM over the table's bytes (one 4 KB page a row); resident = the table held in
anonymous memory from build on (4 KB pages unless ENGRAM_HUGEPAGES=on; the file pages it read dropped from the page cache;
with ENGRAM_BUILD=background built on one thread while the weights load, the first chunk waits for it): with
ENGRAM_ROWS=bfp8 already as bfloat8_b (272 B a row, exponents next to the magnitudes; tt/engram_bfp8.py), the chunk's
rows gathered straight into TILE bytes; with bf16 the 8-bit bytes. The gather is split over ENGRAM_GATHER_THREADS
Python threads (torch indexing releases the GIL). Every mode gives EngramTable.rows' values bit for bit.
"""

from __future__ import annotations

import mmap
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import torch

import ttnn

from .settings import settings

_HASHERS = {}  # model path -> EngramHasher (the compressed token map takes seconds to build)
_RESIDENT = {}  # (checkpoint path, tensor name) -> resident uint8 tensor (one copy a process, whoever builds it)
_RESIDENT_LOCK = threading.Lock()
_BUILDER = []  # ENGRAM_BUILD=background: the one thread a process that builds the resident tables, one after another
_BUILDER_LOCK = threading.Lock()  # (not _RESIDENT_LOCK: a build holds that for its whole run)


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


def resident_bfp8(ck, layer: int, threads: int = 16, block: int = 1 << 20, huge: bool | None = None):
    """The layer's table as packed bfloat8_b rows [N, 272] uint8 in anonymous memory, built once a process: blocks of
    ``block`` rows read with pread (``threads`` ahead; each block's file pages dropped from the page cache after it)
    and packed by tt/engram_bfp8.pack_into. ``huge``: MADV_HUGEPAGE on the rows (default ENGRAM_HUGEPAGES)."""
    from collections import deque

    import numpy as np

    from .engram_bfp8 import pack_into

    key = (str(ck.path), "bfp8", layer)
    with _RESIDENT_LOCK:
        if key in _RESIDENT:
            return _RESIDENT[key]
        wname, sname = _names(layer)
        _, (n, d), woff, _ = ck.info(wname)
        _, (_, sd), soff, _ = ck.info(sname)
        width = d // 16 + d
        buf = _anonymous(n * width, settings.get("ENGRAM_HUGEPAGES") == "on" if huge is None else huge)
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
        finally:
            os.close(fw)
            os.close(fs)
        t = torch.from_numpy(packed)
        t._engram_buf = buf  # (keeps the mapping alive with the tensor)
        _RESIDENT[key] = t
        return t


def _warm_tiles(packed, threads: int):
    """One tiny tile_rows call: numba loads the gather kernel (~0.5 s on the first call) off the first chunk."""
    from .engram_bfp8 import tile_rows

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
        ck = getattr(table, "ckpt", None)
        if self.mode == "random":
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
        return ttnn.from_torch(
            rows.reshape(1, 1, s, w).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(2, None)),
        )

    def host_bfp8(self, tokens: torch.Tensor, tail: torch.Tensor, valid: int | None = None):
        """tokens [S], tail [3] -> (exponents [S, width / 16] uint8, magnitudes [S, width] uint8, new tail): the rows
        as bfloat8_b (tt/engram_bfp8.py), from the table's bytes by integer operations only; pad rows are 0."""
        from .engram_bfp8 import pack_rows

        ids = tokens.reshape(-1).long()
        s = ids.shape[0]
        valid = self._valid(ids, valid)
        exps = torch.zeros(s, self.width // 16, dtype=torch.uint8)
        mags = torch.zeros(s, self.width, dtype=torch.uint8)
        if valid == 0:
            return exps, mags, tail.clone()
        h, new_tail = self.hasher.hashes(ids[:valid], self.layer, tail.long())
        e, m = pack_rows(*self.gather(h), threads=self.threads)
        exps[:valid], mags[:valid] = e.view(valid, -1), m.view(valid, -1)
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

    def tiles_from_packed(self, tokens: torch.Tensor, tail: torch.Tensor, valid: int | None = None):
        """Resident bfp8 table: tokens [S], tail [3] -> (per mesh row the TILE bytes of its [S/SP, width] rows, new
        tail); the bytes equal tile_bytes(*host_bfp8(...)) of each mesh row's share."""
        from .engram_bfp8 import tile_rows

        ids = tokens.reshape(-1).long()
        s = ids.shape[0]
        valid = self._valid(ids, valid)
        rows_ax = tuple(self.mesh.shape)[0] if self.mesh is not None else 1
        assert s % (32 * rows_ax) == 0, f"{s} rows do not split into tiles over {rows_ax} mesh rows"
        n = s // rows_ax
        if valid:
            h, new_tail = self.hasher.hashes(ids[:valid], self.layer, tail.long())
        else:
            h, new_tail = torch.empty(0, self.width // self.table.dim, dtype=torch.int64), tail.clone()
        assert valid == 0 or (int(h.min()) >= 0 and int(h.max()) < self.table.num_rows), "engram row out of range"
        return [tile_rows(self.packed, h[r * n : (r + 1) * n], n, self.threads) for r in range(rows_ax)], new_tail

    def upload_tiles(self, tiles: list) -> ttnn.Tensor:
        """Per mesh row TILE bytes -> device [1, 1, S/SP, width] bfloat8_b per chip, replicated over axis 1."""
        cols_ax = tuple(self.mesh.shape)[1]
        n = tiles[0].numel() * 16 // 17 // self.width
        wrap = ttnn.experimental.disaggregation.tensor_from_bfp8_bytes
        shards = [x for t in tiles for x in [wrap(t.numpy().tobytes(), [1, 1, n, self.width])] * cols_ax]
        return ttnn.to_device(
            ttnn.from_host_shards(shards, self.mesh.shape), self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

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
