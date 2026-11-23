# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The host half of Engram: token ids -> the n-gram table rows each Engram layer reads.

A position is hashed with the tokens before it by the checkpoint's own ``inference/engram.py``
(``NgramHashState``), giving ``n_hash_cols`` row ids per Engram layer. The rows come from that
layer's table -- ~100 GB of fp8 E4M3 with one E8M0 scale per 32 values, held in host RAM -- and
are dequantized exactly as ``ParallelEngramEmbedding`` does it, to bf16
``[B, L, n_hash_cols * head_dim]``. The device half (:mod:`.decode.engram`) receives them over an
H2D socket.

:func:`stage_engram_tables` copies the tables' checkpoint files into :data:`STAGE_DIR` (tmpfs, so the copy
stays in RAM between processes) and :func:`load_engram_table` then maps the copy instead of reading the
checkpoint: one copy of the ~203 GB, shared by every process instead of one private copy each. The demo and
the full-model tests call it before building the model, so repeated and concurrent runs share it; it is a
no-op once the files are staged, and falls back to reading the checkpoint when they do not fit. Set
``DEEPSEEK_V41_ENGRAM_STAGE=0`` to turn it off, and ``tools/stage_engram_tables.py`` to stage (or ``--remove``
the copies) ahead of time.
"""

import contextlib
import ctypes
import fcntl
import importlib.util
import json
import math
import mmap
import os
import shutil
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

# One E8M0 scale per this many consecutive values of a table row.
TABLE_BLOCK = 32
# Rows packed per worker task (~264 MB at head_dim=256).
PACK_CHUNK_ROWS = 1 << 20
# Headroom above the Engram tables a run needs, whether it stages a copy of them or reads them: the copy holds
# the source's page cache plus the new tmpfs pages while it is made, and the build needs room above either.
ENGRAM_RAM_MARGIN = 1.25
# Where tools/stage_engram_tables.py and :func:`stage_engram_tables` keep the tables' checkpoint files. tmpfs
# pages are not reclaimed under memory pressure the way page cache is, so a staged table stays resident until
# removed or reboot.
STAGE_DIR = Path(os.environ.get("DEEPSEEK_V41_ENGRAM_STAGE_DIR", "/dev/shm/deepseek_v41_engram"))
# Linux 5.14+: fault a range's pages into the page tables without touching them from user space.
_MADV_POPULATE_READ = 22
_libc = ctypes.CDLL(None, use_errno=True)
_libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]


def host_mem_available() -> int:
    """Bytes of host RAM ``MemAvailable`` currently reports."""
    with open("/proc/meminfo") as meminfo:
        return next(int(line.split()[1]) * 1024 for line in meminfo if line.startswith("MemAvailable:"))


def _drop_page_cache(path) -> None:
    """Drop ``path`` from the page cache (``POSIX_FADV_DONTNEED``), so a staged copy of it does not hold the
    source's pages as well. Pages another running process has mapped stay as long as it maps them."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def tensor_region(path, name: str) -> tuple[int, list]:
    """Byte offset and shape of tensor ``name`` in the safetensors file ``path``."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        meta = json.loads(f.read(header_len))[name]
    return 8 + header_len + meta["data_offsets"][0], meta["shape"]


def read_into_memory(
    path, offset: int, shape, dtype: torch.dtype, threads: int = 16, chunk: int = 1 << 30, progress=None
):
    """The tensor at ``offset`` in ``path`` copied into process memory, read in ``chunk``-sized pieces by
    ``threads`` readers (a single reader takes minutes for a 100 GB table). ``progress(n)``, if given, is called
    from the readers with each piece's byte count as it lands."""
    nbytes = math.prod(shape) * dtype.itemsize
    buf = torch.empty(nbytes, dtype=torch.uint8)
    view = memoryview(buf.numpy())

    def read(begin):
        end = min(begin + chunk, nbytes)
        with open(path, "rb", buffering=0) as f:
            f.seek(offset + begin)
            while begin < end:
                n = f.readinto(view[begin:end])
                assert n, f"short read of {path} at {offset + begin}"
                begin += n
                if progress is not None:
                    progress(n)

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(read, range(0, nbytes, chunk)))
    return buf.view(dtype).reshape(shape)


def map_into_memory(
    path, offset: int, shape, dtype: torch.dtype, threads: int = 16, chunk: int = 1 << 30, progress=None
):
    """The tensor at ``offset`` in ``path`` mapped rather than copied: a private mapping, which shares the
    file's resident pages with every other process mapping it. ``threads`` ``MADV_POPULATE_READ`` calls of
    ``chunk`` bytes fault every page in up front; a lookup would otherwise take ~60 us per new page on the
    decode path. ``progress`` as :func:`read_into_memory`'s."""
    nbytes = math.prod(shape) * dtype.itemsize
    file = torch.from_file(str(path), shared=False, size=os.path.getsize(path), dtype=torch.uint8)
    base = file.data_ptr()
    end = offset + nbytes

    def populate(begin):
        stop = min(begin + chunk, end)
        if _libc.madvise(base + begin, stop - begin, _MADV_POPULATE_READ):
            raise OSError(ctypes.get_errno(), f"MADV_POPULATE_READ of {path} at {begin}")
        if progress is not None:
            progress(stop - max(begin, offset))

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(populate, range(offset - offset % mmap.PAGESIZE, end, chunk)))
    return file[offset:end].view(dtype).reshape(shape)


def staged_copy(path) -> Optional[Path]:
    """The :data:`STAGE_DIR` copy of checkpoint file ``path``, or ``None`` if it has not been staged.

    A copy is named after the file ``path`` resolves to -- in the HuggingFace cache a blob named by its
    SHA-256, so a copy matches only the file it was made from -- and must be as long as it."""
    path = Path(path).resolve()
    copy = STAGE_DIR / path.name
    if copy.is_file() and copy.stat().st_size == path.stat().st_size:
        return copy
    return None


def _table_regions(loader: DeepseekV4WeightLoader, layer_idx: int):
    """``(w_path, w_offset, s_path, s_offset, rows, head_dim, scale_cols)`` of layer ``layer_idx``'s table."""
    w_name = f"layers.{layer_idx}.engram.embed.weight"
    s_name = f"layers.{layer_idx}.engram.embed.scale"
    w_path = Path(loader.shard_of(w_name, translate=False)).resolve()
    s_path = Path(loader.shard_of(s_name, translate=False)).resolve()
    w_offset, w_shape = tensor_region(w_path, w_name)
    s_offset, s_shape = tensor_region(s_path, s_name)
    rows, head_dim = w_shape
    assert s_shape == [rows, head_dim // TABLE_BLOCK], (w_shape, s_shape)
    return w_path, w_offset, s_path, s_offset, rows, head_dim, s_shape[1]


def staged_packed_path(loader: DeepseekV4WeightLoader, layer_idx: int) -> Path:
    """Path in :data:`STAGE_DIR` where layer ``layer_idx``'s row-packed ``[weight | scale]`` table lives."""
    w_path = Path(loader.shard_of(f"layers.{layer_idx}.engram.embed.weight", translate=False)).resolve()
    return STAGE_DIR / f"{w_path.name}.layer{layer_idx}.packed"


def staged_packed_copy(loader: DeepseekV4WeightLoader, layer_idx: int) -> Optional[Path]:
    """The :data:`STAGE_DIR` row-packed copy of layer ``layer_idx``'s table, or ``None`` if absent or truncated."""
    copy = staged_packed_path(loader, layer_idx)
    if copy.is_file() and copy.stat().st_size == engram_table_nbytes(loader, layer_idx):
        return copy
    return None


def _pread_exact(fd: int, nbytes: int, offset: int) -> bytes:
    """Read exactly ``nbytes`` at ``offset`` from ``fd`` via ``os.pread``."""
    data = os.pread(fd, nbytes, offset)
    if len(data) == nbytes:
        return data
    chunks = [data]
    got = len(data)
    while got < nbytes:
        piece = os.pread(fd, nbytes - got, offset + got)
        assert piece, f"short pread at {offset + got}"
        chunks.append(piece)
        got += len(piece)
    return b"".join(chunks)


def _pack_into_array(
    w_path: Path,
    w_offset: int,
    s_path: Path,
    s_offset: int,
    dst: np.ndarray,
    head_dim: int,
    scale_cols: int,
    threads: int = 16,
    chunk_rows: int = PACK_CHUNK_ROWS,
    progress=None,
) -> None:
    """Stream ``weight`` and ``scale`` from checkpoint files into ``dst`` ``[rows, head_dim + scale_cols]`` uint8."""
    rows = dst.shape[0]
    row_bytes = head_dim + scale_cols
    w_fd = os.open(w_path, os.O_RDONLY)
    s_fd = w_fd if s_path == w_path else os.open(s_path, os.O_RDONLY)
    try:

        def pack(begin: int) -> None:
            end = min(begin + chunk_rows, rows)
            n = end - begin
            w_raw = _pread_exact(w_fd, n * head_dim, w_offset + begin * head_dim)
            s_raw = _pread_exact(s_fd, n * scale_cols, s_offset + begin * scale_cols)
            dst[begin:end, :head_dim] = np.frombuffer(w_raw, dtype=np.uint8).reshape(n, head_dim)
            dst[begin:end, head_dim:] = np.frombuffer(s_raw, dtype=np.uint8).reshape(n, scale_cols)
            if progress is not None:
                progress(n * row_bytes)

        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(pack, range(0, rows, chunk_rows)))
    finally:
        os.close(w_fd)
        if s_fd != w_fd:
            os.close(s_fd)


def _pack_layer_to_file(loader: DeepseekV4WeightLoader, layer_idx: int, dst: Path, threads: int = 16) -> int:
    """Stream layer ``layer_idx``'s table into a contiguous ``[weight | scale]`` packed file at ``dst``."""
    w_path, w_offset, s_path, s_offset, rows, head_dim, scale_cols = _table_regions(loader, layer_idx)
    row_bytes = head_dim + scale_cols
    size = rows * row_bytes
    with open(dst, "w+b") as f:
        f.truncate(size)
        mm = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_WRITE)
        try:
            arr = np.frombuffer(mm, dtype=np.uint8).reshape(rows, row_bytes)
            try:
                _pack_into_array(w_path, w_offset, s_path, s_offset, arr, head_dim, scale_cols, threads=threads)
            finally:
                del arr
        finally:
            mm.close()
    _drop_page_cache(w_path)
    if s_path != w_path:
        _drop_page_cache(s_path)
    return size


def engram_shards(loader: DeepseekV4WeightLoader, layer_ids) -> list[Path]:
    """The checkpoint files holding ``layer_ids``' tables, deduplicated in layer order. A file holds one layer's
    ``weight`` and ``scale`` and nothing else, so this is one file per layer on the shipped checkpoint."""
    return list(
        dict.fromkeys(
            Path(loader.shard_of(f"layers.{i}.engram.embed.{part}", translate=False)).resolve()
            for i in layer_ids
            for part in ("weight", "scale")
        )
    )


@contextlib.contextmanager
def _staging_lock():
    """An exclusive lock on :data:`STAGE_DIR`, so two processes starting at once copy the tables once: the loser
    waits here, then finds the files staged and copies nothing."""
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STAGE_DIR / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def stage_engram_tables(loader: DeepseekV4WeightLoader, layer_ids, threads: int = 16) -> tuple[list[Path], int]:
    """Stage ``layer_ids``' tables in :data:`STAGE_DIR` as contiguous row-packed ``[weight | scale]`` files so
    every process maps one copy instead of reading the checkpoint into memory itself: the copies outlive the
    process, and with them a run takes no Engram RAM of its own (:func:`engram_ram_needed` returns 0 for a
    staged layer). Each row packs its 256 B fp8 weights followed immediately by its 8 B E8M0 scales (264 B/row,
    same total size as the raw tensors, halving random DRAM/TLB misses per lookup).

    Skips layers already staged, and returns the staged paths and the bytes this call wrote (0 when nothing had
    to be). Falls back to leaving the files alone -- the model then reads the checkpoint as before -- when they
    do not fit next to the RAM a build needs, or when ``DEEPSEEK_V41_ENGRAM_STAGE=0``. Files go to a partial
    name first and are renamed into place, and the whole staging runs under a lock on :data:`STAGE_DIR`, so
    processes starting at once share the copy rather than racing for it.

    Must be called before the model reads the tables (``DeepSeekV41Model.prepare_static_decode``).
    """
    if os.environ.get("DEEPSEEK_V41_ENGRAM_STAGE", "1") in ("0", "", "false", "False"):
        return [], 0
    ids = list(layer_ids)
    paths = [staged_packed_path(loader, i) for i in ids]
    with _staging_lock():
        todo = [i for i in ids if staged_packed_copy(loader, i) is None]
        if not todo:
            return paths, 0
        need = sum(engram_table_nbytes(loader, i) for i in todo)
        # The copy holds the bytes twice while it is made (the source's page cache and the new tmpfs pages), and
        # the build still needs its own headroom above what is staged.
        parent = STAGE_DIR
        while not parent.exists():
            parent = parent.parent
        available = min(shutil.disk_usage(parent).free, host_mem_available())
        if available < ENGRAM_RAM_MARGIN * need:
            logger.warning(
                f"Engram tables of layers {ids} not staged: {need / 1e9:.0f} GB to pack, "
                f"{available / 1e9:.0f} GB available in {parent} and host RAM; reading the checkpoint instead"
            )
            return paths, 0
        logger.info(f"staging packed Engram tables of layers {ids} in {STAGE_DIR}: {need / 1e9:.0f} GB to write")
        copied = 0
        for layer_idx in todo:
            dst = staged_packed_path(loader, layer_idx)
            partial = dst.with_name(f".{dst.name}.{os.getpid()}.partial")
            start = time.monotonic()
            try:
                written = _pack_layer_to_file(loader, layer_idx, partial, threads=threads)
                partial.chmod(0o644)
                os.replace(partial, dst)
            finally:
                partial.unlink(missing_ok=True)
            seconds = time.monotonic() - start
            copied += written
            logger.info(
                f"staged packed {dst.name}: {written / 1e9:.0f} GB in {seconds:.0f}s "
                f"({written / 1e9 / max(seconds, 1e-9):.1f} GB/s)"
            )
        return paths, copied


def engram_table_nbytes(loader: DeepseekV4WeightLoader, layer_idx: int) -> int:
    """Bytes of layer ``layer_idx``'s n-gram table as stored (fp8 weight plus E8M0 scales, one byte each)."""
    total = 0
    for part in ("weight", "scale"):
        name = f"layers.{layer_idx}.engram.embed.{part}"
        _, shape = tensor_region(loader.shard_of(name, translate=False), name)
        total += math.prod(shape)
    return total


def engram_ram_needed(loader: DeepseekV4WeightLoader, layer_ids) -> int:
    """Bytes of free host RAM loading the tables of ``layer_ids`` takes. A staged table is already
    resident (and not counted in ``MemAvailable``), so it needs none."""
    return sum(
        engram_table_nbytes(loader, i)
        for i in layer_ids
        if staged_packed_copy(loader, i) is None
        and staged_copy(loader.shard_of(f"layers.{i}.engram.embed.weight", translate=False)) is None
    )


def load_engram_table(
    loader: DeepseekV4WeightLoader, layer_idx: int, progress=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Layer ``layer_idx``'s table in host RAM: ``weight`` ``[rows, head_dim]`` fp8 E4M3 and ``scale``
    ``[rows, head_dim / 32]`` E8M0 (~101 GB together). Mapped as strided views of a contiguous row-packed
    ``[weight | scale]`` buffer from :func:`staged_packed_copy` when staged (or packed directly on read when
    not staged), falling back to a legacy raw :func:`staged_copy` if one is present. ``progress`` as
    :func:`read_into_memory`'s."""
    w_path, w_offset, s_path, s_offset, rows, head_dim, scale_cols = _table_regions(loader, layer_idx)
    row_bytes = head_dim + scale_cols
    packed_copy = staged_packed_copy(loader, layer_idx)
    if packed_copy is not None:
        packed = map_into_memory(packed_copy, 0, (rows, row_bytes), torch.uint8, progress=progress)
        return packed[:, :head_dim].view(torch.float8_e4m3fn), packed[:, head_dim:].view(torch.float8_e8m0fnu)
    # Fall back to a legacy raw shard staged in STAGE_DIR if present.
    w_copy, s_copy = staged_copy(w_path), staged_copy(s_path)
    if w_copy is not None and s_copy is not None:
        return (
            map_into_memory(w_copy, w_offset, (rows, head_dim), torch.float8_e4m3fn, progress=progress),
            map_into_memory(s_copy, s_offset, (rows, scale_cols), torch.float8_e8m0fnu, progress=progress),
        )
    packed = torch.empty((rows, row_bytes), dtype=torch.uint8)
    _pack_into_array(w_path, w_offset, s_path, s_offset, packed.numpy(), head_dim, scale_cols, progress=progress)
    return packed[:, :head_dim].view(torch.float8_e4m3fn), packed[:, head_dim:].view(torch.float8_e8m0fnu)


def _engram_module(snapshot_dir: Path):
    """The checkpoint's ``inference/engram.py``."""
    spec = importlib.util.spec_from_file_location("deepseek_v41_engram", Path(snapshot_dir) / "inference" / "engram.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EngramHostLookup:
    """Token ids -> ``{layer_id: rows [B, L, n_hash_cols * head_dim] bf16}`` for every Engram layer in ``tables``.

    Keeps each user's compressed-token history across calls, so decode calls it once per step with that
    step's token at its position, after prefill called it with the prompt from position 0.

    ``sessions`` histories ``[batch, max_seq]`` are kept, one per decode session slot;
    :meth:`use_session` picks the one the next calls read and extend.
    """

    def __init__(self, config, snapshot_dir, tables: dict, batch: int, max_seq: int, tokenizer=None, sessions: int = 1):
        """``tables`` maps an Engram layer id to its ``(weight, scale)`` from :func:`load_engram_table`.
        ``tokenizer`` defaults to the checkpoint's; it only feeds the normalized-token map."""
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
        engram = _engram_module(snapshot_dir)
        args = SimpleNamespace(
            engram_layer_ids=config.engram_layer_ids,
            engram_num_embeddings=config.engram_num_embeddings,
            engram_max_ngram_size=config.engram_max_ngram_size,
            engram_vocab_size=config.engram_vocab_size,
            engram_n_heads=config.engram_n_heads,
            engram_head_dim=config.engram_head_dim,
            engram_compressed_vocab_size=config.engram_compressed_vocab_size,
            engram_pad_id=config.engram_pad_token_id,
            max_batch_size=batch,
            max_seq_len=max_seq,
        )
        self.layout = engram.EngramLayout.from_args(args)
        self.hash = engram.NgramHashState(args, self.layout, tokenizer)
        self.vocab_size = config.vocab_size
        for layer_id, (weight, scale) in tables.items():
            rows = self.layout.num_embeddings[self.layout.layer_ids.index(layer_id)]
            assert tuple(weight.shape) == (rows, self.layout.head_dim), (layer_id, tuple(weight.shape))
            assert tuple(scale.shape) == (rows, self.layout.head_dim // TABLE_BLOCK), (layer_id, tuple(scale.shape))
        self.tables = tables
        self._histories = [self.hash.cache] + [torch.empty_like(self.hash.cache) for _ in range(sessions - 1)]
        # The decode packet writer hashes on its own thread while the posting thread switches sessions.
        self._lock = threading.Lock()

    def use_session(self, slot: int) -> None:
        """Hash against session slot ``slot``'s token history from now on (a pointer swap, no copy)."""
        with self._lock:
            self.hash.cache = self._histories[slot]

    def export_history(self, slot: int, pos: int) -> torch.Tensor:
        """A copy of session slot ``slot``'s first ``pos`` compressed tokens, ``[batch, pos]``."""
        with self._lock:
            return self._histories[slot][:, :pos].clone()

    def import_history(self, slot: int, history: torch.Tensor) -> None:
        """Make ``history`` (an :meth:`export_history`, of any slot) the start of session slot ``slot``'s history."""
        with self._lock:
            self._histories[slot][:, : history.shape[1]] = history

    def _compressed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """``input_ids`` ``[B, L]`` as the history holds them: through the compressed token map, image-span ids dead."""
        text = input_ids < self.vocab_size
        return torch.where(text, self.hash.token_map[torch.where(text, input_ids, 0)], self.hash.DEAD)

    def extend_history(self, slot: int, input_ids: torch.Tensor, start_pos: int) -> None:
        """Leave session slot ``slot``'s history as :meth:`hash_session` would, without computing the hashes: a
        prefill commit only needs the history the next decode step hashes against."""
        compressed = self._compressed(input_ids)
        with self._lock:
            self._histories[slot][:, start_pos : start_pos + input_ids.shape[1]] = compressed

    def _hash(self, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        """``self.hash`` with image-span tokens (ids past the vocabulary, see :mod:`.vision`) cached as dead, so no
        n-gram spans them -- the reference's ``token_mask``. Call with ``_lock`` held."""
        text = input_ids < self.vocab_size
        if bool(text.all()):
            return self.hash(input_ids, start_pos)
        return self.hash(torch.where(text, input_ids, 0), start_pos, token_mask=text)

    @torch.no_grad()
    def hash_session(self, slot: int, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        """``self.hash`` against session slot ``slot``'s history, leaving the :meth:`use_session` choice as it was.
        An image-span token's hashes are of an empty n-gram: zero its rows (:meth:`__call__` does)."""
        with self._lock:
            picked, self.hash.cache = self.hash.cache, self._histories[slot]
            try:
                return self._hash(input_ids, start_pos)
            finally:
                self.hash.cache = picked

    def _rows(self, input_ids: torch.Tensor, hashes: torch.Tensor) -> dict:
        """Each table's rows ``[B, L, kin]`` for hashes ``[B, L, n_layers, n_cols]``. Image-span tokens are zero:
        the reference gates their Engram output to 0, and zero rows give the device half exactly that."""
        image = (input_ids >= self.vocab_size).unsqueeze(-1)
        rows = {}
        for layer_id in self.tables:
            found = self.gather(layer_id, hashes[:, :, self.layout.layer_ids.index(layer_id)]).flatten(-2)
            rows[layer_id] = found.masked_fill(image, 0) if bool(image.any()) else found
        return rows

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor, start_pos: int) -> dict:
        """``input_ids`` ``[B, L]`` at positions ``start_pos ..`` -> each table layer's rows ``[B, L, n_hash_cols * head_dim]``.

        An image-span token's rows are zero: the reference gates its Engram output to 0, and zero rows give the
        device half exactly that."""
        with self._lock:
            hashes = self._hash(input_ids, start_pos)  # [B, L, n_engram_layers, n_hash_cols]
        return self._rows(input_ids, hashes)

    @torch.no_grad()
    def lookup_history(self, history: torch.Tensor, input_ids: torch.Tensor, start_pos: int) -> dict:
        """``__call__`` against a copy of ``history`` ``[B, H]`` (positions ``0 .. H``), not a live session.

        The server looks a waiting prompt up this way, so the gather is done before the prefill runs and the
        session a decode step is hashing is left alone. ``input_ids`` ``[B, L]`` sit at ``start_pos``."""
        if history.ndim != 2 or input_ids.ndim != 2 or history.shape[0] != input_ids.shape[0]:
            raise ValueError(
                f"history {tuple(history.shape)} and ids {tuple(input_ids.shape)} must both be [batch, length]"
            )
        scratch = torch.zeros_like(self.hash.cache)
        if history.shape[1] > scratch.shape[1]:
            raise ValueError(f"history of {history.shape[1]} tokens is past max_seq {scratch.shape[1]}")
        scratch[:, : history.shape[1]] = history.to(dtype=scratch.dtype)
        with self._lock:
            picked, self.hash.cache = self.hash.cache, scratch
            try:
                hashes = self._hash(input_ids, start_pos)
            finally:
                self.hash.cache = picked
        return self._rows(input_ids, hashes)

    def gather(self, layer_id: int, ids: torch.Tensor) -> torch.Tensor:
        """Rows ``ids`` of ``layer_id``'s table, dequantized to bf16: ``[..., head_dim]``."""
        weight, scale = self.tables[layer_id]
        values = F.embedding(ids, weight).float().unflatten(-1, (-1, TABLE_BLOCK))
        return (values * F.embedding(ids, scale).float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
