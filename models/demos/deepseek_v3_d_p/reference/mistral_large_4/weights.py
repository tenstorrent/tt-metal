# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint access for Mistral Large 4: the consolidated (Mistral-format) safetensors shards, FP8 block
dequantization, and the routed-expert store shared by the CPU reference and the HF oracle.

The checkpoint is 1 TB of FP8 (e4m3) weights with one float32 scale per 128 x 128 block (``<name>.qscale_weight``,
powers of two, ``use_ue8m0_for_qscales``), and bf16 for the tensors the quantization config leaves out (norms,
kv_a / kv_b, the indexer's wk and weights_proj, the router, embeddings, the LM head). An FP8 value times a power of
two is exact in bf16, so dequantizing to bf16 loses nothing.

Reads go straight to the shards with parallel O_DIRECT preads (2+ GB/s from the NFS checkpoint, page cache untouched:
a forward over 40 MoE layers scans more bytes than the page cache holds, so cached reads would evict exactly what the
next chunk needs).

Routed experts (14.5 GB of FP8 per MoE layer, 128 experts x w1/w2/w3) never fit in host memory all together. The
store keeps a layer's raw FP8 weights and scales (the checkpoint's own bytes) resident while a byte budget allows, and
reads the other layers again each time they run, prefetching the next layer while the current one computes. On a
resident layer, an expert is dequantized to bf16 (exact) the first time a chunk routes to it and kept for the process
while a second budget lasts (component tests call one layer's experts step many times); other experts are
dequantized for the GEMM that uses them (``dequant``).
"""

from __future__ import annotations

import ctypes
import json
import mmap
import os
import struct
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import torch

DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "I64": torch.int64,
    "I32": torch.int32,
    "U8": torch.uint8,
}
ALIGN = 4096  # O_DIRECT offset / length / buffer alignment
PIECE = 32 << 20  # bytes per pread task
QBLOCK = 128  # FP8 block size (quantization.block_size_out / block_size_in)
IO_THREADS = 16  # 16 parallel preads: 2.25 GB/s from the NFS checkpoint (32: 1.4 GB/s)


def keep_heap() -> None:
    """Serve large allocations from glibc's heap and keep freed memory there (mallopt M_MMAP_MAX 0, M_TRIM_THRESHOLD
    max). By default every tensor above 32 MB is a fresh mmap whose pages the kernel faults in and zeroes on first
    touch: an expert GEMM's temporaries (and the tests' bfloat8_b rounding of each expert weight) ran 7x slower on page
    faults than on the arithmetic. A no-op where glibc is absent."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(-4, 0)  # M_MMAP_MAX
        libc.mallopt(-1, 2**31 - 1)  # M_TRIM_THRESHOLD
    except (OSError, AttributeError):
        pass


def mem_available() -> int:
    """MemAvailable in bytes (0 when unknown)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class Checkpoint:
    """Lazy reader over the sharded checkpoint: ``read(names)`` returns {name: tensor} in the stored dtype."""

    def __init__(self, path):
        self.path = Path(path)
        self.where = json.loads((self.path / "consolidated.safetensors.index.json").read_text())["weight_map"]
        self._headers: dict[str, tuple[dict, int]] = {}
        self._lock = threading.Lock()
        self.pool = ThreadPoolExecutor(IO_THREADS, thread_name_prefix="ckpt-read")
        self._direct = True

    def has(self, name: str) -> bool:
        return name in self.where

    def _header(self, fname: str) -> tuple[dict, int]:
        with self._lock:
            if fname not in self._headers:
                with open(self.path / fname, "rb") as f:
                    n = struct.unpack("<Q", f.read(8))[0]
                    self._headers[fname] = (json.loads(f.read(n)), 8 + n)
            return self._headers[fname]

    def meta(self, name: str) -> tuple[torch.dtype, list[int]]:
        h, _ = self._header(self.where[name])
        return DTYPES[h[name]["dtype"]], list(h[name]["shape"])

    def _span(self, name: str) -> tuple[str, int, int]:
        fname = self.where[name]
        h, base = self._header(fname)
        a, b = h[name]["data_offsets"]
        return fname, base + a, base + b

    def _pread(self, fname: str, buf: memoryview, offset: int) -> None:
        """Fill buf from the file at offset (stops at end of file)."""
        flags = os.O_RDONLY | (os.O_DIRECT if self._direct else 0)
        try:
            fd = os.open(self.path / fname, flags)
        except OSError:
            self._direct = False
            fd = os.open(self.path / fname, os.O_RDONLY)
        try:
            done = 0
            while done < len(buf):
                try:
                    n = os.preadv(fd, [buf[done:]], offset + done)
                except OSError:  # a filesystem without O_DIRECT: buffered from here on
                    if not self._direct:
                        raise
                    self._direct = False
                    os.close(fd)
                    fd = os.open(self.path / fname, os.O_RDONLY)
                    continue
                if n <= 0:
                    break
                done += n
        finally:
            os.close(fd)

    def read(self, names: list[str]) -> dict[str, torch.Tensor]:
        """Read tensors in parallel (each split into PIECE-sized preads over the shared pool)."""
        jobs, out = [], {}
        for name in names:
            fname, start, end = self._span(name)
            s0 = start // ALIGN * ALIGN
            e0 = -(-end // ALIGN) * ALIGN
            buf = mmap.mmap(-1, e0 - s0)  # page-aligned, as O_DIRECT needs
            mv = memoryview(buf)
            for off in range(0, e0 - s0, PIECE):
                jobs.append(self.pool.submit(self._pread, fname, mv[off : off + PIECE], s0 + off))
            out[name] = (buf, start - s0, end - start)
        for j in jobs:
            j.result()
        res = {}
        for name, (buf, off, nbytes) in out.items():
            dtype, shape = self.meta(name)
            t = (
                torch.frombuffer(buf, dtype=torch.uint8, offset=off, count=nbytes)
                if nbytes
                else torch.empty(0, dtype=torch.uint8)
            )
            res[name] = t.view(dtype).view(shape)
        return res

    def get(self, name: str) -> torch.Tensor:
        return self.read([name])[name]


def dequant(w8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, out: torch.Tensor | None = None) -> torch.Tensor:
    """FP8 [rows, cols] with float32 block scales [ceil(rows / 128), ceil(cols / 128)] -> ``dtype`` (exact for bf16 and
    fp32: the scales are powers of two). ``out`` (optional) is a preallocated [rows, cols] buffer of ``dtype``."""
    rows, cols = w8.shape
    out = torch.empty(rows, cols, dtype=dtype) if out is None else out
    out.copy_(w8)
    sr, sc = scale.shape
    if rows == sr * QBLOCK and cols == sc * QBLOCK:
        out.view(sr, QBLOCK, sc, QBLOCK).mul_(scale.to(dtype)[:, None, :, None])
    else:
        full = scale.repeat_interleave(QBLOCK, 0)[:rows].repeat_interleave(QBLOCK, 1)[:, :cols]
        out.mul_(full.to(dtype))
    return out


def load_dense(ckpt: Checkpoint, names: dict[str, str], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """{key: checkpoint name} -> {key: tensor in dtype}; FP8 tensors are dequantized with their ``.qscale_weight``."""
    want = list(names.values())
    want += [
        n.removesuffix(".weight") + ".qscale_weight" for n in names.values() if ckpt.meta(n)[0] == torch.float8_e4m3fn
    ]
    raw = ckpt.read(want)
    out = {}
    for key, n in names.items():
        t = raw[n]
        if t.dtype == torch.float8_e4m3fn:
            out[key] = dequant(t, raw[n.removesuffix(".weight") + ".qscale_weight"], dtype)
        else:
            out[key] = t.to(dtype).clone() if t.dtype == dtype else t.to(dtype)
    return out


EXPERT_MATS = ("w1", "w2", "w3")  # w1 gate, w3 up (both [I, H]), w2 down ([H, I])


class LayerExperts:
    """One MoE layer's routed experts as stored: ``w[(e, m)]`` FP8 and ``s[(e, m)]`` its block scales, m in w1/w2/w3.
    On a resident layer (``store`` set), an expert a chunk routes to is dequantized once into bf16 (exact) and kept in
    ``deq`` while the store's dequant budget lasts: later calls only widen it to the compute dtype."""

    def __init__(self, w: dict, s: dict):
        self.w, self.s = w, s
        self.store: ExpertStore | None = None
        self.deq: dict[tuple[int, str], torch.Tensor] = {}

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in list(self.w.values()) + list(self.s.values()))

    def weight(self, e: int, m: str, dtype: torch.dtype, out: torch.Tensor | None = None) -> torch.Tensor:
        key = (e, m)
        w = self.deq.get(key)
        if w is None and self.store is not None and self.store.reserve(2 * self.w[key].numel()):
            w = self.deq[key] = dequant(self.w[key], self.s[key], torch.bfloat16)
        if w is None:
            return dequant(self.w[key], self.s[key], dtype, out)
        return w.to(dtype) if out is None else out.copy_(w)


class ExpertStore:
    """Routed experts by layer: resident while the byte budget lasts (first come), else read on every use; ``get(i,
    next_layer=j)`` also starts reading layer j in the background."""

    def __init__(self, ckpt: Checkpoint, num_experts: int, budget: int | None = None, deq_budget: int | None = None):
        self.ckpt = ckpt
        self.num_experts = num_experts
        self.budget = int(0.4 * mem_available()) if budget is None else budget
        self.deq_budget = int(0.15 * mem_available()) if deq_budget is None else deq_budget
        self.resident: dict[int, LayerExperts] = {}
        self.used = self.deq_used = 0
        self._pending: dict[int, Future] = {}
        self._lock = threading.Lock()
        self._bg = ThreadPoolExecutor(1, thread_name_prefix="expert-prefetch")

    def _names(self, i: int) -> list[str]:
        out = []
        for e in range(self.num_experts):
            for m in EXPERT_MATS:
                out += [f"layers.{i}.experts.{e}.{m}.weight", f"layers.{i}.experts.{e}.{m}.qscale_weight"]
        return out

    def _load(self, i: int) -> LayerExperts:
        raw = self.ckpt.read(self._names(i))
        w, s = {}, {}
        for e in range(self.num_experts):
            for m in EXPERT_MATS:
                w[(e, m)] = raw[f"layers.{i}.experts.{e}.{m}.weight"]
                s[(e, m)] = raw[f"layers.{i}.experts.{e}.{m}.qscale_weight"]
        return LayerExperts(w, s)

    def _start(self, i: int) -> None:
        with self._lock:
            if i not in self.resident and i not in self._pending:
                self._pending[i] = self._bg.submit(self._load, i)

    def get(self, i: int, next_layer: int | None = None) -> LayerExperts:
        if i in self.resident:
            le = self.resident[i]
        else:
            self._start(i)
            with self._lock:
                fut = self._pending[i]
            le = fut.result()
            with self._lock:
                self._pending.pop(i, None)
                n = le.nbytes()
                if self.used + n <= self.budget:
                    self.resident[i] = le
                    self.used += n
                    le.store = self
        if next_layer is not None:
            self._start(next_layer)
        return le

    def reserve(self, nbytes: int) -> bool:
        """Take nbytes of the dequantized-expert budget (LayerExperts.weight); False once it is spent."""
        with self._lock:
            if self.deq_used + nbytes > self.deq_budget:
                return False
            self.deq_used += nbytes
            return True


_STORES: dict[tuple[str, int], ExpertStore] = {}


def expert_store(ckpt: Checkpoint, num_experts: int) -> ExpertStore:
    """One store per checkpoint and process: the HF oracle and the reference of one gate share its resident layers."""
    key = (str(ckpt.path), num_experts)
    if key not in _STORES:
        _STORES[key] = ExpertStore(ckpt, num_experts)
    return _STORES[key]


_CKPTS: dict[str, Checkpoint] = {}


def checkpoint(path) -> Checkpoint:
    key = str(Path(path).resolve())
    if key not in _CKPTS:
        _CKPTS[key] = Checkpoint(path)
    return _CKPTS[key]
