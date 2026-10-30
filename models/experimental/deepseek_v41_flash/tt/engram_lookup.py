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

``tools/stage_engram_tables.py`` copies the tables' checkpoint files into :data:`STAGE_DIR` (tmpfs, so
the copy stays in RAM between processes); :func:`load_engram_table` then maps the copy instead of reading
the checkpoint.
"""

import ctypes
import importlib.util
import json
import math
import mmap
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F

from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

# One E8M0 scale per this many consecutive values of a table row.
TABLE_BLOCK = 32
# Where tools/stage_engram_tables.py keeps the tables' checkpoint files. tmpfs pages are not reclaimed
# under memory pressure the way page cache is, so a staged table stays resident until removed or reboot.
STAGE_DIR = Path(os.environ.get("DEEPSEEK_V41_ENGRAM_STAGE_DIR", "/dev/shm/deepseek_v41_engram"))
# Linux 5.14+: fault a range's pages into the page tables without touching them from user space.
_MADV_POPULATE_READ = 22
_libc = ctypes.CDLL(None, use_errno=True)
_libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]


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
        if staged_copy(loader.shard_of(f"layers.{i}.engram.embed.weight", translate=False)) is None
    )


def load_engram_table(
    loader: DeepseekV4WeightLoader, layer_idx: int, progress=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Layer ``layer_idx``'s table in host RAM, as stored: ``weight`` ``[rows, head_dim]`` fp8 E4M3 and
    ``scale`` ``[rows, head_dim / 32]`` E8M0 (~101 GB together). Mapped from its :func:`staged_copy` if
    there is one, else read from the checkpoint. ``progress`` as :func:`read_into_memory`'s."""
    tensors = []
    for part, dtype in (("weight", torch.float8_e4m3fn), ("scale", torch.float8_e8m0fnu)):
        name = f"layers.{layer_idx}.engram.embed.{part}"
        path = loader.shard_of(name, translate=False)
        region = tensor_region(path, name)
        copy = staged_copy(path)
        if copy is None:
            tensors.append(read_into_memory(path, *region, dtype, progress=progress))
        else:
            tensors.append(map_into_memory(copy, *region, dtype, progress=progress))
    return tuple(tensors)


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

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor, start_pos: int) -> dict:
        """``input_ids`` ``[B, L]`` at positions ``start_pos ..`` -> each table layer's rows ``[B, L, n_hash_cols * head_dim]``.

        An image-span token's rows are zero: the reference gates its Engram output to 0, and zero rows give the
        device half exactly that."""
        with self._lock:
            hashes = self._hash(input_ids, start_pos)  # [B, L, n_engram_layers, n_hash_cols]
        image = (input_ids >= self.vocab_size).unsqueeze(-1)
        rows = {}
        for layer_id in self.tables:
            found = self.gather(layer_id, hashes[:, :, self.layout.layer_ids.index(layer_id)]).flatten(-2)
            rows[layer_id] = found.masked_fill(image, 0) if bool(image.any()) else found
        return rows

    def gather(self, layer_id: int, ids: torch.Tensor) -> torch.Tensor:
        """Rows ``ids`` of ``layer_id``'s table, dequantized to bf16: ``[..., head_dim]``."""
        weight, scale = self.tables[layer_id]
        values = F.embedding(ids, weight).float().unflatten(-1, (-1, TABLE_BLOCK))
        return (values * F.embedding(ids, scale).float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
