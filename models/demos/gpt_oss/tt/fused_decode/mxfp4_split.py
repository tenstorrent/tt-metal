# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TEMP experiment (GPT_OSS_DECODE_EXPERTS_SPLIT=1): the routed experts' MXFP4 weights held EXACTLY by the decode
streams, as a BFP4 part plus a BFP2 part (the compressed custom_mm reads both formats in one stream).

MXFP4 stores each weight as sign x {0, 0.5, 1, 1.5, 2, 3, 4, 6} x 2^s, one s per 32 input (K) values of an output
column. In units u = 2^(s - 1): magnitudes {0, 1, 2, 3, 4, 6, 8, 12}. BFP4 (16 values share an exponent, 3-bit
magnitudes) cannot hold 1 and 12 in one group, so the device's bfloat4_b conversion rounds. Split each weight:
    v = small + big,   small = v if |v| <= 6 u, else v - 8 u sign(v)   (magnitudes {0, 1, 2, 3, 4, 6}: BFP4-exact)
                       big   = 0 or 8 u sign(v)                        (magnitudes {0, 8}:            BFP2-exact)
when every 16-value BFP group shares one u. BFP groups run along a tile's rows, so the weights are stored transposed
(stored[n][k] = W[k][n], 16 consecutive K of one output column per group; custom_mm transposes on unpack), and a
16-value K group always lies inside one 32-value MXFP4 group (TP shards of the intermediate start at multiples of 16).
u per group: 2^(floor(log2 max|v|) - 3) makes every magnitude a valid code (a group whose largest code is c <= 6 gets
u = u_true / 2^j with all codes scaled into {0, ..., 12}).

Per weight column the stream holds, back to back: the BFP4 small K tiles + one BFP4 bias tile (two residual terms per
output column, at K positions 0 and 16, each in its own exponent group: the activation's bias tile carries its
multipliers there), then the BFP2 big K tiles. The compute runs two compressed custom_mm calls per column into the same
DST tile (all-BFP4, then all-BFP2: a BFP2 <-> other format switch costs a reconfig + stall per tile); each call reads
the activation from its first tile. The per-tile metadata (format + whether to load the next activation tile) is the
same for every column.

Byte layouts (tt_metal blockfloat_common.cpp, is_exp_a = false): 64 exponent bytes (face-major, one per 16-value face
row: the largest FP32 exponent of the row), then per face row 16 values packed into 32-bit words, element 0 in the
low bits: BFP4 = (sign << 3) | m3, value m3 x 2^(E - 129); BFP2 = (sign << 1) | m1, value m1 x 2^(E - 127).
"""

import torch

TILE = 32
FACE = 16
BFP4_BYTES = 576
BFP2_BYTES = 320
FMT_ZERO, FMT_BFP2, FMT_BFP4, FMT_BFP8 = 0, 1, 2, 3


def split_exact(w):
    """w [..., K, N] exact MXFP4 values (K = input) -> (small, big), small + big == w, per 16-value K group."""
    *lead, K, N = w.shape
    assert K % FACE == 0
    g = w.float().reshape(*lead, K // FACE, FACE, N)
    m = g.abs().amax(-2, keepdim=True)
    u = torch.where(m > 0, torch.pow(2.0, torch.floor(torch.log2(m.clamp_min(1e-38))) - 3), torch.ones_like(m))
    c = g / u
    codes = torch.tensor([0.0, 1, 2, 3, 4, 6, 8, 12])
    assert torch.isin(c.abs(), codes).all(), "not MXFP4 values on a shared 16-value grid"
    big = torch.where(c.abs() >= 8, torch.sign(c) * 8, torch.zeros_like(c))
    small = c - big
    return (small * u).reshape(w.shape), (big * u).reshape(w.shape)


def _exp_fp32(x):
    """Biased FP32 exponent field of x (0 for 0)."""
    return (x.float().contiguous().view(torch.int32) >> 23) & 0xFF


def _faces(tiles):
    """[n, 32, 32] -> [n, 4, 16, 16] faces in tile order (face 0: rows 0-15 x cols 0-15, 1: cols 16-31, 2, 3)."""
    n = tiles.shape[0]
    return tiles.reshape(n, 2, FACE, 2, FACE).permute(0, 1, 3, 2, 4).reshape(n, 4, FACE, FACE)


def pack_bfp(tiles, bits):
    """[n, 32, 32] values exactly representable in BFP{4,2}_b (bits = 3 or 1 magnitude bits) -> [n, bytes] uint8."""
    f = _faces(tiles.float())  # [n, 4, 16, 16]
    E = _exp_fp32(f).amax(-1)  # [n, 4, 16] shared exponent per face row
    scale = torch.pow(2.0, (E.float() - 127 - (bits - 1)).unsqueeze(-1))
    mag = f.abs() / scale
    q = torch.round(mag)
    assert torch.equal(q, mag) and (q <= (1 << bits) - 1).all(), "value not exact in this BFP format"
    sign = ((f < 0) & (q > 0)).to(torch.int64)
    code = (sign << bits) | q.to(torch.int64)  # [n, 4, 16, 16]
    per_word = 32 // (bits + 1)  # 8 (BFP4) or 16 (BFP2) values per 32-bit word
    code = code.reshape(*code.shape[:-1], FACE // per_word, per_word)
    shifts = torch.arange(per_word) * (bits + 1)
    words = (code << shifts).sum(-1)  # [n, 4, 16, words per row], element 0 in the low bits
    words = words.reshape(words.shape[0], -1).to(torch.int64)
    wbytes = torch.stack([(words >> (8 * i)) & 0xFF for i in range(4)], -1).reshape(words.shape[0], -1)
    ebytes = E.reshape(E.shape[0], -1)  # 64: face-major, then row
    return torch.cat([ebytes, wbytes], -1).to(torch.uint8)


def unpack_bfp(raw, bits):
    """Inverse of pack_bfp: [n, bytes] uint8 -> [n, 32, 32] float."""
    n = raw.shape[0]
    E = raw[:, :64].to(torch.int64).reshape(n, 4, FACE)
    b = raw[:, 64:].to(torch.int64).reshape(n, -1, 4)
    words = (b * (torch.tensor([1, 256, 65536, 16777216]))).sum(-1)  # [n, words]
    per_word = 32 // (bits + 1)
    shifts = torch.arange(per_word) * (bits + 1)
    code = (words.unsqueeze(-1) >> shifts) & ((1 << (bits + 1)) - 1)
    code = code.reshape(n, 4, FACE, FACE)
    q = (code & ((1 << bits) - 1)).float()
    sign = torch.where((code >> bits) & 1 == 1, -1.0, 1.0)
    f = sign * q * torch.pow(2.0, (E.float() - 127 - (bits - 1)).unsqueeze(-1))
    return f.reshape(n, 2, 2, FACE, FACE).permute(0, 1, 3, 2, 4).reshape(n, TILE, TILE)


def _q4_single(x):
    """BFP4 value of x alone in its exponent group (the device packer: magnitude min(round(|x| / step), 7))."""
    e = torch.floor(torch.log2(x.abs().clamp_min(1e-38)))
    step = torch.pow(2.0, e - 2)
    return torch.where(x == 0, x, torch.sign(x) * torch.clamp(torch.round(x.abs() / step), max=7) * step)


def bias_tiles_t(bias):
    """[..., 32] biases -> [..., 32, 32] stored (transposed) BFP4 bias tiles: row n holds output column n's two
    residual terms at K positions 0 and 16 (separate exponent groups)."""
    b = bias.float()
    t1 = _q4_single(b)
    t2 = _q4_single(b - t1)
    out = torch.zeros(*b.shape, TILE)
    out[..., 0] = t1
    out[..., FACE] = t2
    return out


def stored_tiles(w):
    """w [E, K, N] -> [E, N/32, K/32, 32, 32] stored tiles (stored[n][k] = W[k][n])."""
    E, K, N = w.shape
    return w.reshape(E, K // TILE, TILE, N // TILE, TILE).permute(0, 3, 1, 4, 2)


def round_k(w):
    """w [..., K, N] -> the device bfloat4_b rounding (RNE, magnitude <= 7 steps) with 16-value groups along K, the
    grouping of the transposed stored tiles."""
    *lead, K, N = w.shape
    g = w.float().reshape(*lead, K // FACE, FACE, N)
    m = g.abs().amax(-2, keepdim=True)
    step = torch.pow(2.0, torch.floor(torch.log2(m.clamp_min(1e-38))) - 2)
    return torch.where(m > 0, torch.clamp(torch.round(g / step), -7, 7) * step, g).reshape(w.shape)


def pack_columns(w, bias, with_round=False, w_round=None, b_round=None):
    """w [E, K, N] exact MXFP4 values, bias [E, N] -> (p4 [E, N/32, (K/32 + 1) * 576], p2 [E, N/32, K/32 * 320]) uint8:
    per column the BFP4 small tiles + the BFP4 bias tile, and the BFP2 big tiles. with_round: also r4 (as p4) = the
    weights rounded to BFP4 along K (round_k) + the same bias tile, read for the routed ranks held inexactly
    (GPT_OSS_DECODE_SPLIT_RANKS). w_round: that part's weights instead of round_k(w), already on the K-grouped
    bfloat4_b grid (a GPTQ rounding, GPT_OSS_DECODE_SPLIT_ROUND_DIR)."""
    small, big = split_exact(w)
    s, g = stored_tiles(small), stored_tiles(big)
    E, nt, kt = s.shape[:3]
    ps = pack_bfp(s.reshape(-1, TILE, TILE), 3).reshape(E, nt, kt * BFP4_BYTES)
    pg = pack_bfp(g.reshape(-1, TILE, TILE), 1).reshape(E, nt, kt * BFP2_BYTES)
    pb = pack_bfp(bias_tiles_t(bias.reshape(E, nt, TILE)).reshape(-1, TILE, TILE), 3).reshape(E, nt, BFP4_BYTES)
    if not with_round:
        return torch.cat([ps, pb], -1), pg
    wr = round_k(w) if w_round is None else w_round
    pr = pack_bfp(stored_tiles(wr).reshape(-1, TILE, TILE), 3).reshape(E, nt, kt * BFP4_BYTES)
    pbr = pb  # b_round: the rounded part's own bias (its columns in another neuron order)
    if b_round is not None:
        pbr = pack_bfp(bias_tiles_t(b_round.reshape(E, nt, TILE)).reshape(-1, TILE, TILE), 3).reshape(E, nt, BFP4_BYTES)
    return torch.cat([ps, pb], -1), pg, torch.cat([pr, pbr], -1)


def _q4_rows_n(x):
    """bfloat4_b RNE of the rows of x [..., N] with 16-value groups along N (the device conversion)."""
    g = x.float().reshape(*x.shape[:-1], -1, FACE)
    m = g.abs().amax(-1, keepdim=True)
    step = torch.pow(2.0, torch.floor(torch.log2(m.clamp_min(1e-38))) - 2)
    return torch.where(m > 0, torch.clamp(torch.round(g / step), -7, 7) * step, g).reshape(x.shape)


def pack_columns_pkg(w, bias, nbias=4):
    """The package's bfloat4_b tiles in their normal (not transposed) orientation, for the inexact routed ranks
    (GPT_OSS_DECODE_SPLIT_REST_PKG): w [E, K, N] already on the bfloat4_b grid with 16-value groups along N (the GPTQ
    build), bias [E, N] -> [E, N/32, (K/32 + 1) * 576] uint8: per column the K tiles, then a bias tile whose rows
    0..nbias-1 hold the bias as nbias BFP4 residual terms (the activation's bias tile carries the multipliers there)."""
    E, K, N = w.shape
    tiles = w.float().reshape(E, K // TILE, TILE, N // TILE, TILE).permute(0, 3, 1, 2, 4)  # [E, nt, kt, k, n]
    nt, kt = tiles.shape[1:3]
    pw = pack_bfp(tiles.reshape(-1, TILE, TILE), 3).reshape(E, nt, kt * BFP4_BYTES)
    rows, r = [], bias.float()
    for _ in range(nbias):
        t = _q4_rows_n(r)
        rows.append(t)
        r = r - t
    bt = torch.zeros(E, N, TILE)
    bt[..., :nbias] = torch.stack(rows, -1)
    bt = bt.reshape(E, nt, TILE, TILE).transpose(-1, -2)  # tile [k = bias row, n]
    pb = pack_bfp(bt.reshape(-1, TILE, TILE), 3).reshape(E, nt, BFP4_BYTES)
    return torch.cat([pw, pb], -1)


def split_spec(weight_dtype):
    """(is a split stream, exact routed ranks or None = all exact, no rounded part) of a decode stream weight dtype:
    "split" or "split_r<k>"."""
    if not isinstance(weight_dtype, str) or not weight_dtype.startswith("split"):
        return False, None
    return True, (int(weight_dtype[len("split_r") :]) if weight_dtype.startswith("split_r") else None)


def metadata_words(seq):
    """compressed custom_mm metadata for a tile sequence [(use_activation, format)]: 10 tiles per 32-bit word after the
    previous tile's format (bits 1:0; the first word starts from the zero-format sentinel), then per tile the
    use-activation bit and the 2-bit format (_llk_unpack_AB_compressed_custom_mm_)."""
    assert len(seq) % 2 == 0 and 2 <= len(seq) <= 256, len(seq)
    words, prev = [], FMT_ZERO
    for i in range(0, len(seq), 10):
        w = prev
        for j, (use_b, fmt) in enumerate(seq[i : i + 10]):
            w |= (use_b << (3 * j + 2)) | (fmt << (3 * j + 3))
            prev = fmt
        words.append(w)
    return words


def gate_up_sequences(k_tiles):
    """Gate|up column: call 1 = BFP4 small tiles + bias tile (+ a zero tile to make the count even), call 2 = BFP2
    big tiles; both read the activation from its first tile. A zero tile still takes one activation tile (use_b = 1):
    the math thread waits for and clears a SrcB bank per zero tile (_llk_math_compressed_custom_mm_)."""
    s4 = [(1, FMT_BFP4)] * (k_tiles + 1)
    if len(s4) % 2:
        s4.append((1, FMT_ZERO))
    s2 = [(1, FMT_BFP2)] * k_tiles
    if len(s2) % 2:
        s2.append((1, FMT_ZERO))
    return s4, s2


def down_sequences(seg_k_tiles, num_sel, ranks=None, compact=False):
    """Down column block (num_sel expert segments of seg_k_tiles + 1 activation tiles): call 1 = per expert BFP4 small
    (or rounded) tiles + bias tile; call 2 = for the first `ranks` experts (default all) BFP2 big tiles + one zero tile
    that steps past the bias multiplier tile (+ a zero tile to make the count even)."""
    s4 = [(1, FMT_BFP4)] * ((seg_k_tiles + 1) * num_sel)
    exact = num_sel if ranks is None else ranks
    if (
        exact == 0
    ):  # no exact rank: the BFP2 call is skipped (SPLIT_NO_BFP2); two placeholder tiles keep the arrays valid
        return s4, [(1, FMT_ZERO), (1, FMT_ZERO)]
    if compact:  # the BFP2 call reads a compact activation buffer without the bias tiles: BFP2 tiles only, no stalls
        s2 = [(1, FMT_BFP2)] * (seg_k_tiles * exact)
    else:
        s2 = ([(1, FMT_BFP2)] * seg_k_tiles + [(1, FMT_ZERO)]) * exact
    if len(s2) % 2:
        s2.append((1, FMT_ZERO))
    return s4, s2


def even_bfp4(n_tiles):
    """n_tiles BFP4 tiles reading the activation from its first tile (+ a zero tile to make the count even)."""
    s = [(1, FMT_BFP4)] * n_tiles
    if len(s) % 2:
        s.append((1, FMT_ZERO))
    return s


def page_bytes(total, max_packet=16384, align=64):
    """Largest NOC packet (multiple of align, <= max_packet) dividing total."""
    return max(d for d in range(align, max_packet + 1, align) if total % d == 0)
