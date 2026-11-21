# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bonsai PQ2_0 weights straight from the GGUF, in their own values: no dequantisation, no unrotation.

A PQ2_0 block is 34 bytes: fp16 scale d, then 32 bytes of 2-bit codes q (4 per byte, lowest bits first), weight
(q - 1) * d, 128 weights per block along the input dimension. The GGUF stores W' with W_dense = W' @ blockdiag(H) @ D
(H the normalized 1024-point Sylvester Hadamard, D explicit signs per input width), so y = W_dense x = W' (H D x)
(models/demos/blackhole/bonsai2/convert_gguf_to_hf.py, Alejandro's converter, documents the fold).
"""

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PQ2_0 = 142
GGUF_TYPES = {0: (1, 4), 1: (1, 2), 30: (1, 2), 142: (128, 34), 143: (128, 28)}


def _read(path):
    f = open(path, "rb")
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    assert f.read(4) == b"GGUF"
    struct.unpack("<I", f.read(4))
    n_t, n_kv = struct.unpack("<QQ", f.read(16))

    def rstr():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    scal = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def rval(t):
        if t in scal:
            fmt = scal[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        if t == 8:
            return rstr()
        if t == 9:
            (et,) = struct.unpack("<I", f.read(4))
            (n,) = struct.unpack("<Q", f.read(8))
            return [rval(et) for _ in range(n)]
        raise ValueError(t)

    kv = {}
    for _ in range(n_kv):
        k = rstr()
        (t,) = struct.unpack("<I", f.read(4))
        kv[k] = rval(t)
    infos = []
    for _ in range(n_t):
        name = rstr()
        (nd,) = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        (tt,) = struct.unpack("<I", f.read(4))
        (off,) = struct.unpack("<Q", f.read(8))
        infos.append((name, dims, tt, off))
    align = kv.get("general.alignment", 32)
    base = (f.tell() + align - 1) // align * align
    tensors = {}
    for name, dims, tt, off in infos:
        be, bb = GGUF_TYPES[tt]
        n = int(np.prod(dims))
        tensors[name] = (dims, tt, mm[base + off : base + off + n // be * bb])
    return kv, tensors


@dataclass
class BonsaiGGUF:
    path: Path

    def __post_init__(self):
        self.kv, self.tensors = _read(self.path)
        widths, vals = self.kv["prism.hadamard.sign_widths"], self.kv["prism.hadamard.sign_values"]
        self.signs, off = {}, 0
        for w in widths:
            self.signs[w] = np.asarray(vals[off : off + w], dtype=np.float32)
            off += w
        self.folded = set(self.kv["prism.hadamard.weight_names"])
        self.block = int(self.kv["prism.hadamard.block_size"])

    def ternary(self, name):
        """(t_kn [K, N] int8 in {-1, 0, 1}, s_nb [N, K/128] float16) of a PQ2_0 tensor; W'[n, k] = t[k, n] s[n, k/128]."""
        dims, tt, raw = self.tensors[name]
        assert tt == PQ2_0, (name, tt)
        N, K = int(dims[1]), int(dims[0])
        blk = raw.reshape(-1, 34)
        d = blk[:, :2].copy().view("<f2").reshape(N, K // 128)
        qs = blk[:, 2:]
        q = np.stack([(qs >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(N, K)
        t = (q.astype(np.int8) - 1).T.copy()
        return t, d.astype(np.float16)

    def rotate(self, x: np.ndarray) -> np.ndarray:
        """x' = blockdiag(H) D x over the last dimension (the activation the folded weights expect)."""
        K = x.shape[-1]
        xd = x * self.signs[K]
        return fwht_blocks(xd, self.block)


def hadamard(n: int) -> np.ndarray:
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def fwht_blocks(x: np.ndarray, block: int) -> np.ndarray:
    h = hadamard(block)
    xs = x.reshape(*x.shape[:-1], -1, block)
    return (xs @ h).reshape(x.shape)  # h is symmetric
