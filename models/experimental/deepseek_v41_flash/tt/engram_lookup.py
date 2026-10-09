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
"""

import importlib.util
import json
import math
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

# One E8M0 scale per this many consecutive values of a table row.
TABLE_BLOCK = 32


def tensor_region(path, name: str) -> tuple[int, list]:
    """Byte offset and shape of tensor ``name`` in the safetensors file ``path``."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        meta = json.loads(f.read(header_len))[name]
    return 8 + header_len + meta["data_offsets"][0], meta["shape"]


def read_into_memory(path, offset: int, shape, dtype: torch.dtype, threads: int = 16, chunk: int = 1 << 30):
    """The tensor at ``offset`` in ``path`` copied into process memory, read in ``chunk``-sized pieces by
    ``threads`` readers (a single reader takes minutes for a 100 GB table)."""
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

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(read, range(0, nbytes, chunk)))
    return buf.view(dtype).reshape(shape)


def load_engram_table(loader: DeepseekV4WeightLoader, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Layer ``layer_idx``'s table in host RAM, as stored: ``weight`` ``[rows, head_dim]`` fp8 E4M3 and
    ``scale`` ``[rows, head_dim / 32]`` E8M0 (~101 GB together)."""
    tensors = []
    for part, dtype in (("weight", torch.float8_e4m3fn), ("scale", torch.float8_e8m0fnu)):
        name = f"layers.{layer_idx}.engram.embed.{part}"
        path = loader.shard_of(name, translate=False)
        tensors.append(read_into_memory(path, *tensor_region(path, name), dtype))
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
    """

    def __init__(self, config, snapshot_dir, tables: dict, batch: int, max_seq: int, tokenizer=None):
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
        for layer_id, (weight, scale) in tables.items():
            rows = self.layout.num_embeddings[self.layout.layer_ids.index(layer_id)]
            assert tuple(weight.shape) == (rows, self.layout.head_dim), (layer_id, tuple(weight.shape))
            assert tuple(scale.shape) == (rows, self.layout.head_dim // TABLE_BLOCK), (layer_id, tuple(scale.shape))
        self.tables = tables

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor, start_pos: int) -> dict:
        """``input_ids`` ``[B, L]`` at positions ``start_pos ..`` -> each table layer's rows ``[B, L, n_hash_cols * head_dim]``."""
        hashes = self.hash(input_ids, start_pos)  # [B, L, n_engram_layers, n_hash_cols]
        return {
            layer_id: self.gather(layer_id, hashes[:, :, self.layout.layer_ids.index(layer_id)]).flatten(-2)
            for layer_id in self.tables
        }

    def gather(self, layer_id: int, ids: torch.Tensor) -> torch.Tensor:
        """Rows ``ids`` of ``layer_id``'s table, dequantized to bf16: ``[..., head_dim]``."""
        weight, scale = self.tables[layer_id]
        values = F.embedding(ids, weight).float().unflatten(-1, (-1, TABLE_BLOCK))
        return (values * F.embedding(ids, scale).float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
