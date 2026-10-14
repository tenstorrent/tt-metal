# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint access for DeepSeek-V4.1-Flash, shared by the CPU reference and the HF oracle.

Storage (config quantization_config: fp8 e4m3, weight_block_size [32, 32], scale_fmt ue8m0, expert_dtype fp4):
    FP8 e4m3 ``<m>.weight`` + e8m0 ``<m>.scale`` [ceil(out/32), ceil(in/32)]: attention wq_a / wq_b / wkv / wo_a / wo_b,
        indexer wq_b, shared experts, engram wkv. W = fp8(w) * 2^e[r // 32, c // 32] (exact in fp32 and in bf16).
    FP4 e2m1 packed 2 per int8 (low nibble = even column) + e8m0 ``scale`` [out, in/32]: routed experts w1 / w2 / w3.
        W = e2m1(w) * 2^e[r, c // 32] (exact in fp32 and in bf16).
    Engram tables ``layers.{1,14}.engram.embed.weight`` [~384M, 256] fp8 + ``.scale`` [~384M, 8] e8m0 (one per 32):
        203 GB together; never loaded, only memory-mapped and read row by row (``EngramTable``).
    BF16: norms, router weight, embed, head, compressor wkv / wgate, indexer wk / weights_proj, engram q / k weights.
    FP32: attn_sink, router bias, mHC fn / base / scale.
Every tensor is found through ``model.safetensors.index.json`` (the subset trim renames shards) and read through a
memory map of its shard, so nothing larger than the tensor asked for is ever resident.
"""

from __future__ import annotations

import json
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

BLOCK = 32  # fp8 block (32 x 32) and fp4 / engram scale group (32 along the row)
_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I32": torch.int32,
    "I64": torch.int64,
}
# e2m1 code -> value (sign bit 3); a byte holds two codes, the low nibble first (inference/convert.py FP4_TABLE)
FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)
FP4_BYTE_TABLE = torch.stack([FP4_VALUES[torch.arange(256) & 15], FP4_VALUES[torch.arange(256) >> 4]], dim=-1)


class Checkpoint:
    """Name -> tensor over the safetensors shards, via the index and one read-only (copy-on-write) map per shard."""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._headers: dict[str, tuple[dict, int]] = {}
        self._maps: dict[str, np.memmap] = {}
        self._lock = threading.Lock()

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def _header(self, fname: str) -> tuple[dict, int]:
        with self._lock:
            if fname not in self._headers:
                with open(os.path.join(self.path, fname), "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    self._headers[fname] = (json.loads(f.read(n)), 8 + n)
            return self._headers[fname]

    def _map(self, fname: str) -> np.memmap:
        with self._lock:
            if fname not in self._maps:
                self._maps[fname] = np.memmap(os.path.join(self.path, fname), dtype=np.uint8, mode="c")
            return self._maps[fname]

    def info(self, name: str) -> tuple[torch.dtype, list[int], int, int]:
        """(dtype, shape, absolute byte offset, byte length) of a stored tensor."""
        fname = self.weight_map[name]
        hdr, base = self._header(fname)
        h = hdr[name]
        b0, b1 = h["data_offsets"]
        return _DTYPES[h["dtype"]], list(h["shape"]), base + b0, b1 - b0

    def view(self, name: str) -> torch.Tensor:
        """A zero-copy tensor over the memory map (pages are read on access; never written)."""
        assert not name.startswith(("vision.", "aligner.", "mtp.", "image_")), f"text decoder only: {name}"
        dtype, shape, off, nbytes = self.info(name)
        mm = self._map(self.weight_map[name])
        return torch.from_numpy(mm[off : off + nbytes]).view(dtype).view(shape)

    def get(self, name: str) -> torch.Tensor:
        """A resident copy of a stored tensor, in its stored dtype."""
        return self.view(name).clone()

    def is_fp8(self, name: str) -> bool:
        return (
            name.endswith(".weight")
            and self.has(name[: -len("weight")] + "scale")
            and self.info(name)[0] == (torch.float8_e4m3fn)
        )

    def weight(self, name: str, dtype=torch.float32) -> torch.Tensor:
        """A weight as the model computes with it: fp8 blocks dequantized (exact), anything else cast."""
        if self.is_fp8(name):
            return fp8_block_dequant(self.view(name), self.view(name[: -len("weight")] + "scale"), dtype)
        return self.view(name).to(dtype)

    def expert(self, layer: int, e: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Routed expert e of a layer in its stored form: {w1, w2, w3: (packed uint8 [out, in/2], e8m0 [out, in/32])},
        resident copies (read from the map in one go each)."""
        p = f"layers.{layer}.ffn.experts.{e}."
        return {
            n: (self.get(p + n + ".weight").view(torch.uint8), self.get(p + n + ".scale")) for n in ("w1", "w2", "w3")
        }


def fp8_block_dequant(w: torch.Tensor, scale: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """W = fp8(w) * scale[r // 32, c // 32] in fp32 (exact: an e4m3 value times a power of two), then cast."""
    rows, cols = w.shape
    s = scale.float()
    if rows % BLOCK == 0 and cols % BLOCK == 0:
        wf = w.float().view(rows // BLOCK, BLOCK, cols // BLOCK, BLOCK)
        return (wf * s[:, None, :, None]).view(rows, cols).to(dtype)
    s = s.repeat_interleave(BLOCK, 0)[:rows].repeat_interleave(BLOCK, 1)[:, :cols]
    return (w.float() * s).to(dtype)


def fp4_dequant(packed: torch.Tensor, scale: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """Packed e2m1 [out, in/2] (uint8, low nibble = even column) with e8m0 scales [out, in/32] -> [out, in] (exact)."""
    rows, half = packed.shape
    cols = 2 * half
    vals = torch.index_select(FP4_BYTE_TABLE, 0, packed.reshape(-1).to(torch.int32)).view(rows, cols // BLOCK, BLOCK)
    vals.mul_(scale.float().unsqueeze(-1))
    return vals.view(rows, cols).to(dtype)


EXPERT_WORKERS = 8  # threads reading + dequantizing routed experts ahead of their GEMMs
_POOL = None


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(EXPERT_WORKERS, thread_name_prefix="ds41-expert")
    return _POOL


class ExpertPrefetcher:
    """Reads and dequantizes one layer's routed experts ahead of use, in the order they run (ascending id), at most
    2 * EXPERT_WORKERS at a time: get(e) returns {w1, w2, w3: (packed uint8, e8m0 scale, fp32 weight)}."""

    def __init__(self, ck: Checkpoint, layer: int, experts: list[int]):
        self.ck, self.layer, self.order = ck, layer, list(experts)
        self.pos = {e: k for k, e in enumerate(self.order)}
        self.futs: dict[int, object] = {}
        self.next = 0

    def _load(self, e: int) -> dict:
        p = self.ck.expert(self.layer, e)
        return {n: (w, s, fp4_dequant(w, s)) for n, (w, s) in p.items()}

    def get(self, e: int) -> dict:
        k = self.pos[e]
        while self.next < min(len(self.order), k + 2 * EXPERT_WORKERS):
            self.futs[self.next] = _pool().submit(self._load, self.order[self.next])
            self.next += 1
        return self.futs.pop(k).result()


class EngramTable:
    """One engram table (fp8 rows + e8m0 scale per 32), memory-mapped; rows are read when looked up."""

    def __init__(self, ckpt: Checkpoint, layer: int):
        self.weight = ckpt.view(f"layers.{layer}.engram.embed.weight").view(torch.uint8)  # [N, 256] fp8 bits
        self.scale = ckpt.view(f"layers.{layer}.engram.embed.scale").view(torch.uint8)  # [N, 8] e8m0 bits
        self.num_rows, self.dim = self.weight.shape

    def rows(self, ids: torch.Tensor) -> torch.Tensor:
        """ids [...] int64 -> rows [..., dim] fp32 (bf16-exact): fp8 value times its group's scale, as the inference
        ParallelEngramEmbedding computes it (then casts to bf16, which keeps every value)."""
        flat = ids.reshape(-1)
        assert int(flat.min()) >= 0 and int(flat.max()) < self.num_rows, "engram row out of range"
        order = torch.argsort(flat)  # sorted reads: neighbouring rows share pages
        w = torch.empty(flat.numel(), self.dim, dtype=torch.uint8)
        s = torch.empty(flat.numel(), self.scale.shape[1], dtype=torch.uint8)
        w[order] = self.weight[flat[order]]
        s[order] = self.scale[flat[order]]
        v = w.view(torch.float8_e4m3fn).float().view(flat.numel(), -1, BLOCK)
        v = v * s.view(torch.float8_e8m0fnu).float().unsqueeze(-1)
        return v.view(*ids.shape, self.dim)
