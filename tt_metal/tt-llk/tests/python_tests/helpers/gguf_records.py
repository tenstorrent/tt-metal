# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GGUF weights as the Blackhole SFPU decoders read them: ggml blocks, records, L1 pages, level tables and goldens."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, quants

# helpers -> python_tests -> tests -> tt-llk
_PAGE_H = (
    Path(__file__).resolve().parents[3]
    / "tt_llk_blackhole/common/inc/experimental/ckernel_gguf_page.h"
)


def _layout_args(values):
    """make_layout's arguments, code_offset (default 0) included."""
    args = tuple(int(a) for a in values.split(","))
    return args + (0,) * (6 - len(args))


def _header_layouts():
    """The format ids and page_layout arguments of ckernel_gguf_page.h (the one table of the page layouts)."""
    text = _PAGE_H.read_text()
    ids = {name: int(v) for name, v in re.findall(r"^\s+(\w+)\s+= (\d+),$", text, re.M)}
    args = {
        name: _layout_args(values)
        for name, values in re.findall(
            r"case (\w+):\s+return make_layout\(([^)]*)\);", text
        )
    }
    assert ids and set(ids) == set(args), f"{_PAGE_H}: format ids and layouts differ"
    return ids, args


_IDS, _LAYOUTS = _header_layouts()


@dataclass(frozen=True)
class GgufFormat:
    """One decoder format: GGUF block size and bytes, the page layout of ckernel_gguf_page.h."""

    name: str
    id: int  # ckernel::gguf::Format
    block_elems: int
    block_bytes: int
    group_tiles: int
    page_tiles: int
    code_bytes: int
    scale_rows: int
    fp16_rows: int
    code_offset: int = 0
    table_entries: int = 0

    @property
    def pages_per_group(self) -> int:
        return self.group_tiles // self.page_tiles

    @property
    def scale_offset(self) -> int:
        return -(-(self.code_offset + self.code_bytes) // 256) * 256

    @property
    def scale_bytes(self) -> int:
        if self.fp16_rows:
            return (-(-self.scale_rows // 4) * 4 + 8 * self.fp16_rows) * 64
        return self.scale_rows * 64

    @property
    def page_rows(self) -> int:
        rows = 16
        while rows * 64 < self.scale_offset + self.scale_bytes:
            rows *= 2
        return rows

    @property
    def page_bytes(self) -> int:
        return self.page_rows * 64

    @property
    def codebook(self) -> bool:
        return self.table_entries > 0

    @property
    def bits_per_weight(self) -> float:
        return 8 * self.block_bytes / self.block_elems


def _format(name, block_elems, block_bytes, table_entries=0):
    layout = _LAYOUTS[name]
    return GgufFormat(
        name, _IDS[name], block_elems, block_bytes, *layout, table_entries
    )


FORMATS = {
    f.name: f
    for f in (
        _format("Q8_0", 32, 34),
        _format("Q4_0", 32, 18),
        _format("IQ4_NL", 32, 18),
        _format("MXFP4", 32, 17),
        _format("NVFP4", 64, 36),
        _format("Q6_K", 256, 210),
        _format("Q4_K", 256, 144),
        _format("Q5_K", 256, 176),
        _format("Q3_K", 256, 110),
        _format("Q2_K", 256, 84),
        _format("IQ4_XS", 256, 136),
        _format("TQ2_0", 256, 66),
        _format("Q2_0", 64, 18),
        _format("Q1_0", 128, 18),
        _format("IQ2_XXS", 256, 66, 256),
        _format("IQ1_S", 256, 50, 2048),
        _format("IQ2_S", 256, 82, 1024),
        _format("IQ3_S", 256, 110, 512),
        _format("IQ3_XXS", 256, 98, 256),
        _format("IQ2_XS", 256, 74, 512),
        _format("IQ1_M", 256, 56, 2048),
        _format("Q4_1", 32, 20),
        _format("Q5_0", 32, 22),
        _format("Q5_1", 32, 24),
        _format("TQ1_0", 256, 54),
    )
}

# ------------------------------------------------------------------------------------------------- random blocks

# Byte offsets of each format's fp16 scale fields in a block (IQ1_M's d is spread over its scale words' top nibbles).
_FP16_FIELDS = {
    "Q8_0": (0,),
    "Q4_0": (0,),
    "IQ4_NL": (0,),
    "Q4_1": (0, 2),
    "Q5_0": (0,),
    "Q5_1": (0, 2),
    "Q2_0": (0,),
    "Q1_0": (0,),
    "Q2_K": (80, 82),
    "Q3_K": (108,),
    "Q4_K": (0, 2),
    "Q5_K": (0, 2),
    "Q6_K": (208,),
    "IQ4_XS": (0,),
    "TQ2_0": (64,),
    "TQ1_0": (52,),
    "IQ2_XXS": (0,),
    "IQ2_XS": (0,),
    "IQ2_S": (0,),
    "IQ3_XXS": (0,),
    "IQ3_S": (0,),
    "IQ1_S": (0,),
}


# Formats whose elements carry their own sign bits: ggml's quantizers write d >= 0 and the decoder takes the sign from
# the sign bits alone.
_UNSIGNED_D = ("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S")


def _fp16_bits(rng, shape, low_exp, high_exp, signed=True):
    """fp16 bit patterns of magnitude 2^e, e uniform in [low_exp, high_exp] (subnormals below -14)."""
    mag = np.exp2(rng.uniform(low_exp, high_exp, size=shape))
    sign = rng.choice((-1.0, 1.0), size=shape) if signed else 1.0
    return (sign * mag).astype(np.float16).view(np.uint16)


def random_blocks(
    fmt: GgufFormat, n: int, k: int, rng, wide_scales: bool = True
) -> np.ndarray:
    """(n, k / block_elems, block_bytes) uint8 ggml blocks: random code bits, finite scale fields (non-negative where
    ggml's quantizers write them so).

    wide_scales: scales from fp16 subnormals to 4 (decode tests); otherwise about 2^-7 .. 2^-4, so that the weights
    stay in the range of a trained layer (matmul tests).
    """
    nb = k // fmt.block_elems
    blocks = rng.integers(0, 256, size=(n, nb, fmt.block_bytes), dtype=np.uint8)
    low, high = (-24.0, 2.0) if wide_scales else (-7.0, -4.0)
    for off in _FP16_FIELDS.get(fmt.name, ()):
        d = _fp16_bits(rng, (n, nb), low, high, signed=fmt.name not in _UNSIGNED_D)
        blocks[:, :, off : off + 2] = d.view(np.uint8).reshape(n, nb, 2)
    if fmt.name == "IQ1_M":
        d = _fp16_bits(rng, (n, nb), low, high).astype(np.uint16)
        sc = blocks[:, :, 48:56].copy().view(np.uint16)
        for i in range(4):
            sc[:, :, i] = (sc[:, :, i] & 0x0FFF) | (((d >> (4 * i)) & 0xF) << 12)
        blocks[:, :, 48:56] = sc.view(np.uint8)
    elif fmt.name == "MXFP4":
        # E8M0; 2 .. 250 keeps every value a normal fp32 number.
        blocks[:, :, 0] = (
            rng.integers(2, 251, size=(n, nb))
            if wide_scales
            else rng.integers(119, 123, size=(n, nb))
        )
    elif fmt.name == "NVFP4":
        # Unsigned E4M3 (bit 7 clear).
        blocks[:, :, 0:4] = (
            rng.integers(0, 128, size=(n, nb, 4))
            if wide_scales
            else rng.integers(8, 24, size=(n, nb, 4))
        )
    return blocks


# --------------------------------------------------------------------------------------------------- golden


def dequantize(fmt: GgufFormat, blocks: np.ndarray) -> np.ndarray:
    """ggml's dequantization of (n, nb, block_bytes) blocks: (n, k) float32."""
    n, nb, _ = blocks.shape
    if fmt.name in ("Q1_0", "Q2_0"):
        # Not in gguf-py: dequantize_row_q1_0 / dequantize_row_q2_0 of ggml-quants.c.
        d = np.ascontiguousarray(blocks[:, :, 0:2]).view(np.float16).astype(np.float32)
        qs = blocks[:, :, 2:]
        w = np.arange(fmt.block_elems)
        if fmt.name == "Q1_0":
            bit = (qs[:, :, w // 8] >> (w % 8)) & 1
            return np.where(bit != 0, d, -d).astype(np.float32).reshape(n, -1)
        q = ((qs[:, :, w // 4] >> (2 * (w % 4))) & 3).astype(np.float32)
        return ((q - 1.0) * d).astype(np.float32).reshape(n, -1)
    data = np.ascontiguousarray(blocks).reshape(n, nb * fmt.block_bytes)
    return (
        quants.dequantize(data, GGMLQuantizationType[fmt.name])
        .reshape(n, -1)
        .astype(np.float32)
    )


def level_table(fmt: GgufFormat) -> np.ndarray:
    """A codebook format's level table: grid entry -> the level index of each of its elements, element i at bits w i
    (IQ2: values (8, 25, 43); IQ1: (-1, 0, 1); IQ3_S: (1, 3, .., 15), w = 3; IQ3_XXS: (4, 12, .., 52, 62) as
    f = 0..6 and 8 for 62, w = 4, so that the value is 65 - |61 - 8 f|)."""
    q = getattr(quants, "IQ1_S" if fmt.name == "IQ1_M" else fmt.name)
    q.init_grid()
    entries, per = q.grid_shape
    w = 4 if fmt.name == "IQ3_XXS" else 3 if fmt.name == "IQ3_S" else 2
    idx = np.searchsorted(
        np.array(q.grid_map, dtype=np.float32), q.grid.reshape(entries, per)
    )
    if fmt.name == "IQ3_XXS":
        idx = np.where(idx == 7, 8, idx)
    shifts = (w * np.arange(per)).astype(np.uint32)
    return (idx.astype(np.uint32) << shifts).sum(axis=1).astype(np.uint16)


# ------------------------------------------------------------------------------------------------- records


# SFPU lane (r, c)'s output o is element (_out_k(o, r), _out_n(o, c)) of a 32x32 [K][N] tile; word-vector v of a page
# is rows 4 (v / 2) .. + 3, columns of parity v % 2 (_lane_word); scale word w = 2c + p covers columns 16p + 2c + b.
def _lane_col(w):  # scale word w covers columns _lane_col(w) + b
    return 16 * (w % 2) + 2 * (w // 2)


def _idx_col(c, k):  # byte k of lane column c's index word
    return 16 * (k // 2) + 2 * c + k % 2


def _nib_col(m):  # half m % 2 of high-bits word m / 2
    return 16 * ((m % 4) // 2) + 2 * (m // 4) + m % 2


def _out_k(o, r):
    return 16 * (o // 16) + 4 * ((o // 2) % 4) + r


def _out_n(o, c):
    return 16 * ((o // 8) % 2) + o % 2 + 2 * c


def _lane_word(wv, r, c):
    return (4 * (wv // 2) + r) * 16 + 2 * c + wv % 2


class _Tile:
    """The blocks of one N tile: block(n, b) is column n's block b."""

    def __init__(self, blocks: np.ndarray, nt: int):
        self.b = blocks[32 * nt : 32 * nt + 32]

    def block(self, n, b):
        return self.b[n, b]

    def rd16(self, n, b, off):
        x = self.b[n, b]
        return int(x[off]) | int(x[off + 1]) << 8

    def rd32(self, n, b, off):
        return self.rd16(n, b, off) | self.rd16(n, b, off + 2) << 16


def _words(*rows) -> bytes:
    return np.array(rows, dtype=np.uint32).tobytes()


def _pair16(t, b, off):
    return [
        t.rd16(_lane_col(w), b, off) | t.rd16(_lane_col(w) + 1, b, off) << 16
        for w in range(16)
    ]


def _quad8(t, b0, o0, b1, o1):
    row = []
    for w in range(16):
        n = _lane_col(w)
        row.append(
            int(t.block(n, b0)[o0])
            | int(t.block(n + 1, b0)[o0]) << 8
            | int(t.block(n, b1)[o1]) << 16
            | int(t.block(n + 1, b1)[o1]) << 24
        )
    return row


def _col_words(t, b, off):
    return [t.rd32(n, b, off) for n in range(32)]


def _sign_word(aux):
    return sum(
        (((aux >> (7 * l)) & 0x7F) | (((aux >> (28 + l)) & 1) << 7)) << (8 * l)
        for l in range(4)
    )


def _idx_words(t, b, off) -> bytes:
    out = bytearray(128)
    for l in range(4):
        for c in range(8):
            for k in range(4):
                out[4 * (8 * l + c) + k] = t.block(_idx_col(c, k), b)[off + l]
    return bytes(out)


def _idx3_words(t, b, off) -> bytes:
    out = bytearray(256)
    for m in range(64):
        a, g, c = m // 32, m // 8 % 4, m % 8
        for k in range(4):
            out[4 * m + k] = t.block(16 * a + 2 * c + k % 2, b)[off + g + 4 * (k // 2)]
    return bytes(out)


def _lane_highs(hn, w):
    pos = ((8, 8 + w, 24, 24 + w), (0, w, 16, 16 + w))
    hi = [0] * 16
    for m in range(32):
        for k in range(4):
            hi[m // 2] |= hn[m][k] << pos[m % 2][k]
    return hi


def _planes(name, blk):
    """Element w of a block is (block[byte[w]] >> shift[w]) & (2^bits - 1), for the low and high code planes."""
    lo_bits = (
        8
        if name == "Q8_0"
        else (
            1
            if name == "Q1_0"
            else 2 if name in ("Q3_K", "Q2_K", "TQ2_0", "Q2_0") else 4
        )
    )
    hi_bits = (
        2 if name == "Q6_K" else 1 if name in ("Q5_K", "Q3_K", "Q5_0", "Q5_1") else 0
    )
    lo, hi = [], []
    for w in range(blk):
        h, t, e, j = w // 128, w % 128 // 32, w % 32, w // 32
        lb = ls = hb = hs = 0
        if name == "Q8_0":
            lb = 2 + w
        elif name in ("Q4_0", "IQ4_NL"):
            lb, ls = 2 + w % 16, 4 * (w // 16)
        elif name == "Q4_1":
            lb, ls = 4 + w % 16, 4 * (w // 16)
        elif name == "Q5_0":
            lb, ls, hb, hs = 6 + w % 16, 4 * (w // 16), 2 + w // 8, w % 8
        elif name == "Q5_1":
            lb, ls, hb, hs = 8 + w % 16, 4 * (w // 16), 4 + w // 8, w % 8
        elif name == "MXFP4":
            lb, ls = 1 + w % 16, 4 * (w // 16)
        elif name == "NVFP4":
            lb, ls = 4 + 8 * (w // 16) + w % 8, 4 * (w % 16 // 8)
        elif name == "Q6_K":
            lb, ls, hb, hs = (
                64 * h + w % 64,
                4 * (w % 128 // 64),
                128 + 32 * h + e,
                2 * t,
            )
        elif name == "Q4_K":
            lb, ls = 16 + 32 * (j // 2) + e, 4 * (j % 2)
        elif name == "Q5_K":
            lb, ls, hb, hs = 48 + 32 * (j // 2) + e, 4 * (j % 2), 16 + e, j
        elif name == "Q3_K":
            lb, ls, hb, hs = 32 + 32 * h + e, 2 * t, e, j
        elif name == "Q2_K":
            lb, ls = 16 + 32 * h + e, 2 * t
        elif name == "IQ4_XS":
            lb, ls = 8 + 16 * j + w % 16, 4 * (w % 32 // 16)
        elif name == "TQ2_0":
            lb, ls = 32 * h + e, 2 * t
        elif name == "Q2_0":
            lb, ls = 2 + w // 4, 2 * (w % 4)
        elif name == "Q1_0":
            lb, ls = 2 + w // 8, w % 8
        lo.append((lb, ls))
        hi.append((hb, hs))
    return lo_bits, lo, hi_bits, hi


def _code(block, plane, bits, w):
    byte, shift = plane[w]
    return (int(block[byte]) >> shift) & ((1 << bits) - 1)


def _pack_codes(fmt, planes, t, page):
    """The codes of K tiles page_tiles * page .. as a page's code words: code q = lo | hi << lo_bits as one B-bit
    field, tile i's B word-vectors from word-vector B i (ckernel_sfpu_gguf_decode.h mrow, mword, mlsb). Stream
    position k is row o = 16 (k // 8 % 2) + 8 (k // 16) + k % 8. When B divides 32, row o is field o % F of word
    o // F (F = 32 // B); otherwise position k < F B is field k % F of word k // F, and positions F B .. 31 take the
    words' top two bits: bit s of that string (word 0 bits 30, 31, word 1 bits 30, 31, ..) is bit s % B of position
    F B + s // B."""
    lo_bits, lo, hi_bits, hi = planes
    words = [0] * (fmt.code_bytes // 4)
    B = lo_bits + hi_bits
    F = 32 // B
    clean = F * B if 32 % B else 32
    for i in range(fmt.page_tiles):
        kt = fmt.page_tiles * page + i
        b = 32 * kt // fmt.block_elems
        w0 = 32 * kt % fmt.block_elems
        for k in range(32):
            o = 16 * (k // 8 % 2) + 8 * (k // 16) + k % 8
            for r in range(4):
                w = w0 + _out_k(o, r)
                for c in range(8):
                    blk = t.block(_out_n(o, c), b)
                    q = _code(blk, lo, lo_bits, w)
                    if hi_bits:
                        q |= _code(blk, hi, hi_bits, w) << lo_bits
                    if 32 % B == 0:
                        words[_lane_word(B * i + o // F, r, c)] |= q << (B * (o % F))
                    elif k < clean:
                        words[_lane_word(B * i + k // F, r, c)] |= q << (B * (k % F))
                    else:
                        for j in range(B):
                            bit = B * (k - clean) + j
                            words[_lane_word(B * i + bit // 2, r, c)] |= (
                                (q >> j) & 1
                            ) << (clean + bit % 2)
    return _words(*words)


# GGUF code -> the page's block-float code. MXFP4 / NVFP4: the sign stays, E2M1 magnitudes 0..7 (0, 0.5, 1, 1.5, 2,
# 3, 4, 6) go to 0, 1, 4, 5, 6, 7, 2, 3 (ckernel_sfpu_gguf_decode_fp4.h). IQ4_NL / IQ4_XS: the values in order go to
# x = 1.75 .. -1.75 (codes 7 .. 1, 0, 9 .. 15) and the largest to code 8 (ckernel_sfpu_gguf_decode_iq4.h).
_FP4_TO_PAGE = tuple((q & 8) | (0, 1, 4, 5, 6, 7, 2, 3)[q & 7] for q in range(16))
_IQ4_TO_PAGE = tuple(7 - q if q <= 7 else q + 1 if q <= 14 else 8 for q in range(16))
_BFP_TO_PAGE = {
    "MXFP4": _FP4_TO_PAGE,
    "NVFP4": _FP4_TO_PAGE,
    "IQ4_NL": _IQ4_TO_PAGE,
    "IQ4_XS": _IQ4_TO_PAGE,
}


def _pack_codes_bfp(fmt, planes, t, kt):
    """MXFP4, NVFP4, IQ4_NL, IQ4_XS: K tile kt's codes as 4-bit block-float datums of the 32x32 [K][N] tile in face
    order (element e in bits 4 (e % 2) .. of byte e // 2), mapped by _BFP_TO_PAGE.
    """
    to_page = _BFP_TO_PAGE[fmt.name]
    lo_bits, lo, _, _ = planes
    out = bytearray(512)
    b = 32 * kt // fmt.block_elems
    w0 = 32 * kt % fmt.block_elems
    for k in range(32):
        for n in range(32):
            q = _code(t.block(n, b), lo, lo_bits, w0 + k)
            e = 512 * (k // 16) + 256 * (n // 16) + 16 * (k % 16) + n % 16
            out[e // 2] |= to_page[q] << (4 * (e % 2))
    return bytes(out)


def _scale_page_planes(fmt, t, g):
    name, S = fmt.name, []
    if name in ("Q8_0", "Q4_0", "IQ4_NL", "Q2_0", "Q1_0"):
        S = [_pair16(t, g, 0)]
    elif name == "TQ2_0":
        S = [_pair16(t, g, 64)]
    elif name == "Q4_1":
        S = [_pair16(t, g, 0), _pair16(t, g, 2)]
    elif name == "Q5_0":
        S = [_pair16(t, 2 * g, 0), _pair16(t, 2 * g + 1, 0)]
    elif name == "Q5_1":
        for i in range(2):
            S += [_pair16(t, 2 * g + i, 0), _pair16(t, 2 * g + i, 2)]
    elif name == "MXFP4":
        S = [_quad8(t, 2 * g, 0, 2 * g + 1, 0)]
    elif name == "NVFP4":
        S = [_quad8(t, g // 2, 2 * (g % 2), g // 2, 2 * (g % 2) + 1)]
    elif name in ("Q6_K", "Q2_K"):
        sc = 192 if name == "Q6_K" else 0
        S = [_quad8(t, g, sc + 2 * i, g, sc + 2 * i + 1) for i in range(8)]
        S.append(_pair16(t, g, 208 if name == "Q6_K" else 80))
        if name == "Q2_K":
            S.append(_pair16(t, g, 82))
    elif name in ("Q4_K", "Q5_K", "Q3_K"):
        sc = 96 if name == "Q3_K" else 4
        S = [
            [t.rd32(_lane_col(w) + q // 3, g, sc + 4 * (q % 3)) for w in range(16)]
            for q in range(6)
        ]
        S.append(_pair16(t, g, 108 if name == "Q3_K" else 0))
        if name != "Q3_K":
            S.append(_pair16(t, g, 2))
    elif name == "IQ4_XS":
        rows = [[0] * 16 for _ in range(3)]
        for w in range(16):
            ls = [0, 0]
            for b in range(2):
                n = _lane_col(w) + b
                for j in range(8):
                    v = ((int(t.block(n, g)[4 + j // 2]) >> (4 * (j % 2))) & 15) | (
                        (t.rd16(n, g, 2) >> (2 * j)) & 3
                    ) << 4
                    ls[b] |= v << (6 * j)
            rows[0][w] = ls[0] & 0xFFFFFFFF
            rows[1][w] = (ls[0] >> 32 | ls[1] << 16) & 0xFFFFFFFF
            rows[2][w] = (ls[1] >> 16) & 0xFFFFFFFF
        S = rows + [_pair16(t, g, 0)]
    return _words(*S)


def _group_planes(fmt, planes, t, g):
    scale = _scale_page_planes(fmt, t, g)
    pack = _pack_codes_bfp if fmt.name in _BFP_TO_PAGE else _pack_codes
    pages = [
        pack(fmt, planes, t, fmt.pages_per_group * g + i)
        for i in range(fmt.pages_per_group)
    ]
    # One codes page per group: codes first, then the scale fields, in the page's order.
    return [scale] + pages if fmt.pages_per_group > 1 else pages + [scale]


def _group_tq1_0(t, g):
    words = [0] * 384
    for v in range(12):
        a = v // 4 if v < 8 else (v - 8) // 2
        b = v % 2
        first = 16 * (v // 2 % 2) if v < 8 else 32
        for r in range(4):
            for c in range(8):
                for j in range(4):
                    words[_lane_word(v, r, c)] |= int(
                        t.block(16 * a + 2 * c + b, g)[first + 4 * j + r]
                    ) << (8 * j)
    S = [
        [t.rd32(_lane_col(w), g, 48) for w in range(16)],
        [t.rd32(_lane_col(w) + 1, g, 48) for w in range(16)],
    ]
    S.append(_pair16(t, g, 52))
    return [_words(*words), _words(*S)]


def _group_iq2_xxs(t, g):
    out = [_words(*_pair16(t, g, 0))]
    for s in range(8):
        signs = _words(*[_sign_word(t.rd32(n, g, 6 + 8 * s)) for n in range(32)])
        out.append(signs + _idx_words(t, g, 2 + 8 * s))
    return out


def _group_iq1_s(t, g):
    out = [_words(*_pair16(t, g, 0))]
    for s in range(8):
        hn = [
            [
                (t.rd16(_idx_col(m % 8, k), g, 34 + 2 * s) >> (3 * (m // 8))) & 7
                for k in range(4)
            ]
            for m in range(32)
        ]
        hi = _lane_highs(hn, 3)
        for m in range(32):
            nib = t.rd16(_nib_col(m), g, 34 + 2 * s) >> 12
            o = 16 * (m % 2)
            hi[m // 2] |= (
                (nib & 3) << (o + 6)
                | ((nib >> 2) & 1) << (o + 14)
                | (nib >> 3) << (o + 15)
            )
        out.append(_idx_words(t, g, 2 + 4 * s) + _words(*hi))
    return out


def _group_iq2_s(t, g):
    out = [_words(*_pair16(t, g, 0))]
    for s in range(8):
        hn = [
            [
                (int(t.block(_idx_col(m % 8, k), g)[66 + s]) >> (2 * (m // 8))) & 3
                for k in range(4)
            ]
            for m in range(32)
        ]
        hi = _lane_highs(hn, 2)
        for m in range(32):
            s8 = int(t.block(_nib_col(m), g)[74 + s])
            o = 16 * (m % 2)
            hi[m // 2] |= (s8 & 15) << (o + 4) | (s8 >> 4) << (o + 12)
        out.append(
            _words(*_col_words(t, g, 34 + 4 * s))
            + _idx_words(t, g, 2 + 4 * s)
            + _words(*hi)
        )
    return out


def _group_iq3_s(t, g):
    out = [_words(*_pair16(t, g, 0))]
    for s in range(8):
        W = [0] * 12
        for m in range(64):
            a, gg, c = m // 32, m // 8 % 4, m % 8
            for k in range(4):
                qh = int(t.block(16 * a + 2 * c + k % 2, g)[66 + s])
                W[m // 8] |= ((qh >> (gg + 4 * (k // 2))) & 1) << (8 * k + m % 8)
        scb = bytearray(16)
        sh = 4 * (s % 2)
        for i in range(16):
            n = 16 * (i // 8) + 2 * (i % 8)
            scb[i] = ((int(t.block(n, g)[106 + s // 2]) >> sh) & 15) | (
                (int(t.block(n + 1, g)[106 + s // 2]) >> sh) & 15
            ) << 4
        W[8:12] = np.frombuffer(bytes(scb), dtype=np.uint32).tolist()
        out.append(
            _words(*_col_words(t, g, 74 + 4 * s))
            + _idx3_words(t, g, 2 + 8 * s)
            + _words(*W)
        )
    return out


def _group_iq3_xxs(t, g):
    out = [_words(*_pair16(t, g, 0))]
    for s in range(8):
        signs = _words(*[_sign_word(t.rd32(n, g, 66 + 4 * s)) for n in range(32)])
        out.append(signs + _idx3_words(t, g, 2 + 8 * s))
    return out


def _group_iq2_xs(t, g):
    S = [[0] * 16 for _ in range(5)]
    for u in range(4):
        for j in range(4):
            for w in range(16):
                S[u][w] |= int(
                    t.block(_lane_col(w) + j % 2, g)[66 + 2 * u + j // 2]
                ) << (8 * j)
    S[4] = _pair16(t, g, 0)
    out = [_words(*S)]
    for s in range(8):
        page = bytearray(256)
        for m in range(32):
            l, c = m // 8, m % 8
            sign = 16 * (l // 2) + l % 2 + 2 * c
            for k in range(4):
                e = t.rd16(_idx_col(c, k), g, 2 + 8 * s + 2 * l)
                page[4 * sign + k] = (e >> 9) | ((e >> 8) & 1) << 7
                page[128 + 4 * m + k] = e & 0xFF
        out.append(bytes(page))
    return out


def _group_iq1_m(t, g):
    S = [[0] * 16 for _ in range(4)]
    for w in range(16):
        for f in range(32):
            j = 2 * (f // 4) + f % 2
            v = (
                t.rd16(_lane_col(w) + f // 2 % 2, g, 48 + 2 * (j // 4)) >> (3 * (j % 4))
            ) & 7
            if f < 30:
                S[f // 10][w] |= v << (3 * (f % 10))
            else:
                S[f - 30][w] |= (v & 3) << 30
                S[2][w] |= (v >> 2) << f
        for b in range(2):
            n = _lane_col(w) + b
            sc = [t.rd16(n, g, 48 + 2 * i) for i in range(4)]
            d = (
                sc[0] >> 12
                | (sc[1] >> 8 & 0xF0)
                | (sc[2] >> 4 & 0xF00)
                | (sc[3] & 0xF000)
            )
            S[3][w] |= d << (16 * b)
    out = [_words(*S)]
    for s in range(8):

        def qh(n, l):
            return (int(t.block(n, g)[32 + 2 * s + l // 2]) >> (4 * (l % 2))) & 15

        hn = [[qh(_idx_col(m % 8, k), m // 8) & 7 for k in range(4)] for m in range(32)]
        hi = _lane_highs(hn, 3)
        for m in range(32):
            for e, pos in enumerate((6, 7, 14, 15)):
                hi[m // 2] |= (qh(_nib_col(m), e) >> 3) << (16 * (m % 2) + pos)
        out.append(_idx_words(t, g, 4 * s) + _words(*hi))
    return out


_CODEBOOK_GROUPS = {
    "IQ2_XXS": _group_iq2_xxs,
    "IQ1_S": _group_iq1_s,
    "IQ2_S": _group_iq2_s,
    "IQ3_S": _group_iq3_s,
    "IQ3_XXS": _group_iq3_xxs,
    "IQ2_XS": _group_iq2_xs,
    "IQ1_M": _group_iq1_m,
}


def records(fmt: GgufFormat, blocks: np.ndarray) -> list[list[list[bytes]]]:
    """records[nt][g]: group g of N tile nt as its parts in storage order (the scale fields and the codes pages; a
    group with one codes page stores it first). Their sizes add up to the blocks' size.
    """
    n, nb, _ = blocks.shape
    k = nb * fmt.block_elems
    assert (
        n % 32 == 0 and k % (32 * fmt.group_tiles) == 0
    ), f"{fmt.name}: no records for [{n}, {k}]"
    planes = _planes(fmt.name, fmt.block_elems)
    out = []
    for nt in range(n // 32):
        t = _Tile(blocks, nt)
        groups = []
        for g in range(k // 32 // fmt.group_tiles):
            if fmt.name in _CODEBOOK_GROUPS:
                groups.append(_CODEBOOK_GROUPS[fmt.name](t, g))
            elif fmt.name == "TQ1_0":
                groups.append(_group_tq1_0(t, g))
            else:
                groups.append(_group_planes(fmt, planes, t, g))
        out.append(groups)
    return out


def _fp16_to_fp32(h: np.ndarray) -> np.ndarray:
    """fp16 bits to fp32 bits, exact for every value (subnormals normalized, inf and NaN payloads kept)."""
    h = h.astype(np.uint32)
    sign, mag = (h & 0x8000) << 16, h & 0x7FFF
    special = 0x7F800000 | (mag & 0x3FF) << 13
    subnormal = (mag.astype(np.float32) * np.float32(2.0**-24)).view(np.uint32)
    bits = np.where(mag >= 0x7C00, special, (mag << 13) + 0x38000000)
    return (sign | np.where(mag < 0x400, subnormal, bits)).astype(np.uint32)


def _grid_indices(fmt: GgufFormat, w: np.ndarray) -> np.ndarray:
    """[index word m][byte k]: a codebook page's grid indices with their high bits (ckernel_sfpu_gguf_decode_iq.h):
    IQ3_S bit 8k + m % 8 of word m / 8 after the index words; IQ1_S, IQ1_M, IQ2_S bits 8 (1 - m % 2) + 16 (k / 2) +
    w (k % 2) of word m / 2 after them; IQ2_XS bit 8k + 7 of sign word 16 (m / 16) + 2 (m % 8) + (m / 8) % 2.
    """
    base = fmt.code_offset // 4 + (0 if fmt.name in ("IQ1_S", "IQ1_M") else 32)
    n = 64 if fmt.name in ("IQ3_S", "IQ3_XXS") else 32
    m, k = np.arange(n)[:, None], np.arange(4)[None]
    high = 0
    if fmt.name == "IQ3_S":
        high = w[base + 64 + m // 8] >> 8 * k + m % 8 & 1
    elif fmt.name in ("IQ1_S", "IQ1_M", "IQ2_S"):
        bits = 2 if fmt.name == "IQ2_S" else 3
        shift = 8 * (1 - m % 2) + 16 * (k // 2) + bits * (k % 2)
        high = w[base + 32 + m // 2] >> shift & (1 << bits) - 1
    elif fmt.name == "IQ2_XS":
        sign_word = base - 32 + 16 * (m // 16) + 2 * (m % 8) + m // 8 % 2
        high = w[sign_word] >> 8 * k + 7 & 1
    return w[base : base + n, None] >> 8 * k & 0xFF | high << 8


def prepare_page(fmt: GgufFormat, page: bytes, table: np.ndarray = None) -> bytes:
    """A page completed for the decoder: the fp32 copies of its fp16 scale rows (row scale_rows - fp16_rows + f, half b
    to rows R + 8f + 4b .. + 3, R = scale_rows rounded up to 4) and, for a codebook format, two 16-bit level codes per
    word from byte 0: word 2m + j holds the table entries of index word m's bytes 2j and 2j + 1 (IQ3_S: each column's
    4-bit scale ORed into bits 12..15 of the codes of index words m, (m / 8) % 4 == 0).
    """
    w = np.frombuffer(page, dtype=np.uint32).copy()
    s, r = fmt.scale_offset // 4, -(-fmt.scale_rows // 4) * 4
    for f in range(fmt.fp16_rows):
        row = w[s + 16 * (fmt.scale_rows - fmt.fp16_rows + f) :][:16].copy()
        for b, half in enumerate((row & 0xFFFF, row >> 16)):
            w[s + 16 * (r + 8 * f + 4 * b) :][:64] = np.tile(_fp16_to_fp32(half), 4)
    if fmt.codebook:
        table = level_table(fmt) if table is None else table
        g = table[_grid_indices(fmt, w)].astype(np.uint32)
        if fmt.name == "IQ3_S":
            m = np.arange(64)
            scales = w[fmt.code_offset // 4 + 104 :][:4].view(np.uint8)
            sc = np.where(m // 8 % 4 == 0, scales[8 * (m // 32) + m % 8], 0)
            sc = sc.astype(np.uint32)
            g[:, 0] |= (sc & 15) << 12
            g[:, 1] |= (sc >> 4) << 12
        w[: 2 * len(g)] = (g[:, 0::2] | g[:, 1::2] << 16).reshape(-1)
    return w.tobytes()


def pages(fmt: GgufFormat, recs) -> list[list[bytes]]:
    """pages[nt][p]: page p of N tile nt (K tiles page_tiles * p ..), as the decoder reads it in L1: the codes at
    code_offset, the group's scale fields at scale_offset, completed by prepare_page, the rest zero.
    """
    table = level_table(fmt) if fmt.codebook else None
    out = []
    for groups in recs:
        tile_pages = []
        for parts in groups:
            if fmt.pages_per_group > 1:
                scale, codes = parts[0], parts[1:]
            else:
                codes, scale = parts[:-1], parts[-1]
            for c in codes:
                page = bytearray(fmt.page_bytes)
                page[fmt.code_offset : fmt.code_offset + len(c)] = c
                page[fmt.scale_offset : fmt.scale_offset + len(scale)] = scale
                tile_pages.append(prepare_page(fmt, bytes(page), table))
        out.append(tile_pages)
    return out


def tiles_kn(weights: np.ndarray, nt: int, kt: int) -> np.ndarray:
    """The [K][N] tile (nt, kt) of an [N, K] weight, as the decoder writes it."""
    return weights[32 * nt : 32 * nt + 32, 32 * kt : 32 * kt + 32].T
