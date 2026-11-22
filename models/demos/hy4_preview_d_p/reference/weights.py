# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint access for Hy4 Preview (shared by the CPU reference and the HF oracle hook).

Storage (tencent/Hy4-preview @ 705d81ee): every weight BF16 except the hyper-connection parameters
(``hc_*``), the attention sinks (``learnable_sink_param``) and the router bias (``e_score_correction_bias``), stored
F32. (HF also keeps the indexer's ``k_norm`` / ``weights_proj`` and the LM head in fp32 modules; they are stored
BF16, so that is an exact upcast.) The routed experts
of an MoE layer are two fused tensors, ``experts.gate_up_proj`` [256, 2 * 2048, 6144] (gate rows first, then up) and
``experts.down_proj`` [256, 6144, 2048]. No quantization: the "dequantization" is a dtype cast.

Every tensor is looked up through ``model.safetensors.index.json`` (never by shard name): R.4 trims the checkpoint to
the subset's layers and renames mixed shards. ``ExpertSlab`` reads one expert of a fused expert tensor straight from
its byte range (parallel os.preadv), so the full 1.5 TB model can run with only the selected experts in
memory.
"""

from __future__ import annotations

import json
import os
import struct

import torch
from safetensors import safe_open

_DT = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}


class WeightLoader:
    """Lazy safetensors accessor keyed by checkpoint tensor name (never touches model.mtp_layers.*)."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._handles = {}
        self._headers = {}

    def _file(self, name: str) -> str:
        assert not name.startswith("model.mtp_layers."), f"MTP is out of scope: {name}"
        return os.path.join(self.model_path, self.weight_map[name])

    def get(self, name: str) -> torch.Tensor:
        fname = self._file(name)
        if fname not in self._handles:
            self._handles[fname] = safe_open(fname, framework="pt")
        return self._handles[fname].get_tensor(name)

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def header(self, name: str) -> tuple[str, int, dict]:
        """(file, absolute byte offset of the data section, the tensor's header entry)."""
        fname = self._file(name)
        if fname not in self._headers:
            with open(fname, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._headers[fname] = (8 + n, json.loads(f.read(n)))
        base, hdr = self._headers[fname]
        return fname, base, hdr[name]


_POOL = None
_PIECE = 8 << 20  # bytes per parallel read


def _pool():
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _POOL = ThreadPoolExecutor(max_workers=16, thread_name_prefix="hy4-expert-read")
    return _POOL


def _pread_into(fd: int, mv: memoryview, pos: int) -> None:
    off = 0
    while off < len(mv):
        n = os.preadv(fd, [mv[off:]], pos + off)
        assert n > 0, "short read"
        off += n


class ExpertSlab:
    """One fused expert tensor [E, a, b] of the checkpoint; ``slab[e]`` reads expert e ([a, b]) from disk.

    ``prefetch(ids)`` starts reading those experts in a thread pool (16 parallel 8 MB preads), so a layer's expert
    reads overlap with each other and with compute; ``slab[e]`` then waits for its own bytes. The full model's
    experts (1.5 TB) do not fit the page cache, so the HF sanity run is bound by this read rate."""

    def __init__(self, loader: WeightLoader, name: str, dtype=torch.bfloat16):
        self.name = name
        self.dtype = dtype
        fname, base, entry = loader.header(name)
        self.src_dtype = _DT[entry["dtype"]]
        self.shape = tuple(entry["shape"])
        begin, end = entry["data_offsets"]
        self.offset = base + begin
        self.stride = (end - begin) // self.shape[0]
        self.fd = os.open(fname, os.O_RDONLY)
        self._pending = {}

    def __len__(self) -> int:
        return self.shape[0]

    def _start(self, e: int):
        buf = bytearray(self.stride)
        mv, pos = memoryview(buf), self.offset + e * self.stride
        futs = [
            _pool().submit(_pread_into, self.fd, mv[o : o + _PIECE], pos + o) for o in range(0, self.stride, _PIECE)
        ]
        return buf, futs

    def prefetch(self, ids) -> None:
        for e in ids:
            e = int(e)
            if e not in self._pending:
                self._pending[e] = self._start(e)

    def raw(self, e: int) -> torch.Tensor:
        e = int(e)
        buf, futs = self._pending.pop(e, None) or self._start(e)
        for f in futs:
            f.result()
        return torch.frombuffer(buf, dtype=self.src_dtype).view(self.shape[1:])

    def __getitem__(self, e) -> torch.Tensor:
        return self.raw(int(e)).to(self.dtype)
