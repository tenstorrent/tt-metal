# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reader for the ternary GGUF types used by Ternary-Bonsai-2-27B.

Three containers ship the same weights in different layouts:

  type 142  PQ2_0   34 B / 128 w : fp16 ``d``, then 32 B of 2-bit codes
  type  42  Q2_0    18 B /  64 w : fp16 ``d``, then 16 B of 2-bit codes
  type 143  PTQ1_0  28 B / 128 w : 26 B of base-3 trits, then fp16 ``d``

All three decode to the same canonical form::

    codes  int8  [N, K]      values in {-1, 0, +1}
    scales fp16  [N, K // G] one ``d`` per group, broadcast over the group
    G      int               group size (128 for PQ2_0 / PTQ1_0, 64 for Q2_0)

so ``W_dequant[n, k] = codes[n, k] * scales[n, k // G]``.

The Hadamard rotation is *not* applied here. The GGUF stores ``W'`` such that the
runtime computes ``y = W' @ (H (D x))``; ``hadamard.py`` supplies ``H`` and the
sign vector ``D``.

Row permutations: the GGUF follows llama.cpp's qwen35 layout for the GDN
value-head grouping. Rather than re-deriving it, the converter's fixups are
reused (see ``convert_gguf_to_hf.py``); this module exposes the same
``v_to_hf`` index helper so callers can move between the two orders.

Usage::

    python -m models.demos.blackhole.bonsai2.native.gguf_ternary dump FILE.gguf
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# GGUF container
# ---------------------------------------------------------------------------

# type id -> (elements per block, bytes per block)
GGML_BLOCK = {
    0: (1, 4),  # F32
    1: (1, 2),  # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    8: (32, 34),  # Q8_0
    30: (1, 2),  # BF16
    42: (64, 18),  # Q2_0   (PrismML ternary, 64-weight groups)
    142: (128, 34),  # PQ2_0 (PrismML ternary, 128-weight groups)
    143: (128, 28),  # PTQ1_0 (PrismML ternary-packed-trit, 128-weight groups)
}

#: Formats this module decodes to ternary codes.
TERNARY_TYPES = {42: 64, 142: 128, 143: 128}

_SCALARS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


@dataclass
class GGUFTensor:
    name: str
    dims: tuple  # GGUF order: fastest-varying first
    ggml_type: int
    offset: int
    nbytes: int

    @property
    def shape(self) -> tuple:
        """Torch/numpy order: [rows, cols] for a 2-D tensor."""
        return tuple(int(d) for d in self.dims[::-1])


@dataclass
class GGUF:
    path: Path
    kv: dict
    tensors: dict = field(default_factory=dict)  # name -> GGUFTensor
    _mm: np.ndarray | None = None
    _data_start: int = 0

    def raw(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        return self._mm[self._data_start + t.offset : self._data_start + t.offset + t.nbytes]

    def prism_metadata(self) -> dict:
        return {k: v for k, v in self.kv.items() if k.startswith("prism.")}


def read_gguf(path) -> GGUF:
    """Parse the GGUF header and tensor index (no tensor data is read)."""
    path = Path(path)
    f = open(path, "rb")
    try:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        (version,) = struct.unpack("<I", f.read(4))
        if version not in (2, 3):
            raise ValueError(f"{path}: unsupported GGUF version {version}")
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        def rstr() -> str:
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def rval(t):
            if t in _SCALARS:
                fmt = _SCALARS[t]
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
            if t == 8:
                return rstr()
            if t == 9:
                (et,) = struct.unpack("<I", f.read(4))
                (n,) = struct.unpack("<Q", f.read(8))
                return [rval(et) for _ in range(n)]
            raise ValueError(f"unsupported GGUF metadata type {t}")

        kv = {}
        for _ in range(n_kv):
            k = rstr()
            (t,) = struct.unpack("<I", f.read(4))
            kv[k] = rval(t)

        entries = []
        for _ in range(n_tensors):
            name = rstr()
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
            (tt,) = struct.unpack("<I", f.read(4))
            (off,) = struct.unpack("<Q", f.read(8))
            entries.append((name, dims, tt, off))
        header_end = f.tell()
    finally:
        f.close()

    align = int(kv.get("general.alignment", 32))
    data_start = (header_end + align - 1) // align * align

    gguf = GGUF(path=path, kv=kv)
    for name, dims, tt, off in entries:
        if tt not in GGML_BLOCK:
            raise ValueError(f"{name}: unsupported ggml type {tt}")
        be, bb = GGML_BLOCK[tt]
        n = int(np.prod(dims))
        if n % be:
            raise ValueError(f"{name}: {n} elements is not a multiple of the {be}-element block of type {tt}")
        gguf.tensors[name] = GGUFTensor(name, dims, tt, off, n // be * bb)

    gguf._mm = np.memmap(path, dtype=np.uint8, mode="r")
    gguf._data_start = data_start

    end = max(data_start + t.offset + t.nbytes for t in gguf.tensors.values())
    if end > path.stat().st_size:
        raise ValueError(f"{path}: tensor data ends at {end} but file is {path.stat().st_size} bytes")
    return gguf


# ---------------------------------------------------------------------------
# Block decoders -> (codes int8 [-1,0,1], fp16 scale per block)
# ---------------------------------------------------------------------------


def _decode_2bit_codes(qs: np.ndarray) -> np.ndarray:
    """2-bit codes -> int8.  Shared by PQ2_0 and Q2_0.

    ``qs`` is uint8 [..., nbytes] holding 4 codes per byte, LSB first.
    ``00 -> -1, 01 -> 0, 10 -> +1, 11 -> +2``; +2 cannot occur in a ternary
    weight, so it is rejected rather than silently mapped.
    """
    shifts = np.arange(4, dtype=np.uint8) * 2  # (0, 2, 4, 6)
    q = (qs[..., None] >> shifts) & 0x3  # [..., nbytes, 4]
    q = q.reshape(*qs.shape[:-1], qs.shape[-1] * 4)
    if q.max() == 3:
        raise ValueError("2-bit code 3 (+2) found in a ternary weight block")
    return (q.astype(np.int8) - 1)


def decode_pq2_0(raw: np.ndarray, shape=None) -> tuple[np.ndarray, np.ndarray]:
    """Type 142.  Block = fp16 d, then 32 B of 2-bit codes (128 weights)."""
    blk = raw.reshape(-1, 34)
    d = blk[:, :2].copy().view("<f2")
    codes = _decode_2bit_codes(blk[:, 2:])
    if shape is not None:
        codes = codes.reshape(shape)
        d = d.reshape(*shape[:-1], shape[-1] // 128)
    return codes, d


def decode_q2_0(raw: np.ndarray, shape=None) -> tuple[np.ndarray, np.ndarray]:
    """Type 42.  Block = fp16 d, then 16 B of 2-bit codes (64 weights)."""
    blk = raw.reshape(-1, 18)
    d = blk[:, :2].copy().view("<f2")
    codes = _decode_2bit_codes(blk[:, 2:])
    if shape is not None:
        codes = codes.reshape(shape)
        d = d.reshape(*shape[:-1], shape[-1] // 64)
    return codes, d


def decode_ptq1_0(raw: np.ndarray, shape=None) -> tuple[np.ndarray, np.ndarray]:
    """Type 143.  Block = 26 B of base-3 trits, then fp16 d (128 weights).

    The 26 bytes hold 5+5+4+4+4+4 trits in stride-16-ish order.  Following the
    PR's converter: bytes [0,16) carry 5 trits, [16,24) carry 5, [24,26) carry 4.
    Each trit is ``((p * 3**t) & 255) * 3 >> 8``.

    This is the reference decode.  It is *not* the kernel decode: on Tensix the
    same arithmetic runs per lane and is the reason PTQ1_0 is a research spike.
    """
    blk = raw.reshape(-1, 28)
    d = blk[:, 26:28].copy().view("<f2")
    pieces = []
    for lo, hi, count in ((0, 16, 5), (16, 24, 5), (24, 26, 4)):
        packed = blk[:, lo:hi].astype(np.uint16)
        for trit in range(count):
            rem = (packed * (3**trit)) & 255
            pieces.append(((rem * 3) >> 8).astype(np.int8) - 1)
    codes = np.concatenate(pieces, axis=1)
    if shape is not None:
        codes = codes.reshape(shape)
        d = d.reshape(*shape[:-1], shape[-1] // 128)
    return codes, d


_DECODERS = {142: decode_pq2_0, 42: decode_q2_0, 143: decode_ptq1_0}


# ---------------------------------------------------------------------------
# Canonical form
# ---------------------------------------------------------------------------


@dataclass
class TernaryTensor:
    """One ternary weight matrix in canonical form.

    ``codes`` is int8 [N, K] in {-1, 0, +1}; ``scales`` is fp16 [N, K // G].
    ``codes * scales[..., k // G]`` reproduces the GGUF values exactly.
    """

    name: str
    codes: np.ndarray
    scales: np.ndarray
    group: int
    ggml_type: int

    @property
    def shape(self) -> tuple:
        return self.codes.shape

    @property
    def nbytes(self) -> int:
        """Bytes the weight occupies once packed for the kernel: 2-bit codes + fp16 scales."""
        return self.codes.size // 4 + self.scales.size * 2

    def expanded_scales(self, dtype=np.float32) -> np.ndarray:
        """Scales broadcast to ``[N, K]``, so ``codes * expanded_scales`` is the dense weight."""
        k = self.codes.shape[1]
        s = self.scales.astype(dtype)
        if s.shape[1] * self.group != k:
            raise ValueError(f"{s.shape[1]} scales x group {self.group} != K={k}")
        return np.repeat(s, self.group, axis=1)

    def dequant(self) -> np.ndarray:
        """Dense float32 ``[N, K]``: ``codes * scale``, the weight as the matmul sees it."""
        return self.codes.astype(np.float32) * self.expanded_scales()

    def regrouped_scales(self, group: int) -> np.ndarray:
        """The scale table this weight would have with a different group size.

        Only meaningful where one group's scale subdivides or repeats exactly:
        widening (``group`` a multiple of the stored one) replicates, narrowing
        splits evenly.  Used by the Q2_0 dedupe check.
        """
        stored = self.group
        k = self.codes.shape[1]
        if k % group:
            raise ValueError(f"K={k} is not a multiple of the requested group {group}")
        factor = group // stored
        if factor > 0 and group % stored == 0:
            return np.repeat(self.scales, factor, axis=1)
        if stored % group == 0:
            return np.repeat(self.scales, stored // group, axis=1)
        raise ValueError(f"group {group} and stored group {stored} are not commensurate")



def read_ternary(gguf: GGUF, name: str) -> TernaryTensor:
    """Decode one GGUF tensor to canonical ternary form."""
    t = gguf.tensors[name]
    if t.ggml_type not in _DECODERS:
        raise ValueError(f"{name}: ggml type {t.ggml_type} is not ternary")
    shape = t.shape
    if len(shape) != 2:
        raise ValueError(f"{name}: expected a 2-D tensor, got shape {shape}")
    group = TERNARY_TYPES[t.ggml_type]
    if shape[1] % group:
        raise ValueError(f"{name}: K={shape[1]} is not a multiple of the {group}-weight group")
    codes, scales = _DECODERS[t.ggml_type](gguf.raw(name), shape)
    return TernaryTensor(name, codes, scales, group, t.ggml_type)


def read_all_ternary(gguf: GGUF) -> dict[str, TernaryTensor]:
    return {n: read_ternary(gguf, n) for n, t in gguf.tensors.items() if t.ggml_type in TERNARY_TYPES}


# ---------------------------------------------------------------------------
# llama.cpp <-> HF row order for the GDN value heads
# ---------------------------------------------------------------------------


def v_head_index(num_v_heads: int, num_k_heads: int, unit: int) -> np.ndarray:
    """Index array ``g`` with ``hf = gguf[g]`` for the GGUF v-head regrouping.

    llama.cpp stores the v heads tiled as [nv // nk, nk, unit]; HF expects
    [nk, nv // nk, unit].  ``unit`` is the row width of one head (head_dim for
    qkv / out, 1 for the scalar gates).
    """
    assert num_v_heads % num_k_heads == 0
    return np.arange(num_v_heads * unit).reshape(num_v_heads // num_k_heads, num_k_heads, unit).transpose(1, 0, 2).reshape(-1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _dump(path) -> None:
    gguf = read_gguf(path)
    print(f"{path}")
    print(f"  GGUF v{gguf.kv.get('general.version', '?')}  {len(gguf.tensors)} tensors")
    for k in sorted(gguf.kv):
        if k.startswith("prism."):
            v = gguf.kv[k]
            if isinstance(v, list) and len(v) > 8:
                print(f"  {k} = list[{len(v)}] head={v[:4]}")
            else:
                print(f"  {k} = {v}")
    types = {}
    for t in gguf.tensors.values():
        types[t.ggml_type] = types.get(t.ggml_type, 0) + 1
    print(f"  types: {types}")
    total = 0
    for name in sorted(gguf.tensors):
        t = gguf.tensors[name]
        total += t.nbytes
        print(f"  {name:44s} type={t.ggml_type:3d} shape={t.shape} bytes={t.nbytes}")
    print(f"  total tensor bytes: {total / 1e9:.3f} GB")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3 or sys.argv[1] != "dump":
        print(__doc__)
        raise SystemExit(2)
    _dump(sys.argv[2])
