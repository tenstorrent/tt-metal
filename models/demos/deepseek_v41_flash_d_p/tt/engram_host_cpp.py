# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ENGRAM_HOST_IMPL=cpp: the engram host work of tt/engram_bfp8.py (n-gram hash, bfloat8_b pack, gather of packed rows
into TILE bytes) in C++ (tt/csrc/engram_host.cpp), bit for bit (tests/test_engram_host_cpp.py), so the serving
process needs neither numba nor a short GIL switch interval: each call releases the GIL once for all its work and
runs it on a persistent pool of std::threads (``threads`` a call). Same signatures and results as the numba
functions: ``ngram_hashes``, ``tile_rows_chips``, ``tile_rows``, ``pack_rows``, ``pack_into``; plus
``hash_tile_chips``, HOST_HIDE's whole job of a layer (hash, then the gather into every chip's share) in one call.

Built on first use with torch.utils.cpp_extension.load (g++ -O3, no ttnn / CMake build; cached under
$TORCH_EXTENSIONS_DIR or ~/.cache/torch_extensions, rebuilt when the source changes; ``build()`` warms it).
"""

from __future__ import annotations

import os
import threading

import numpy as np
import torch

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "engram_host.cpp")
_LIB = None
_LOCK = threading.Lock()
_HASH_ARGS = {}  # (id(hasher), layer) -> (hasher, contiguous int64 numpy arrays of the layer's hash constants)


def build(verbose: bool = False):
    """The compiled module (built once a source version; every later process loads the cached .so)."""
    global _LIB
    if _LIB is None:
        with _LOCK:
            if _LIB is None:
                from torch.utils.cpp_extension import load

                _LIB = load(
                    name="ds41_engram_host",
                    sources=[_SRC],
                    extra_cflags=["-O3", "-std=c++17"],
                    extra_ldflags=["-lpthread"],
                    verbose=verbose,
                )
    return _LIB


def _ptr(x, dtype, writable: bool = False) -> int:
    """Address of a C-contiguous numpy array / torch CPU tensor of ``dtype`` (a numpy dtype)."""
    if isinstance(x, torch.Tensor):
        assert x.device.type == "cpu" and x.is_contiguous(), (x.device, x.stride())
        assert x.dtype == torch.from_numpy(np.empty(0, dtype)).dtype, (x.dtype, dtype)
        return x.data_ptr()
    assert isinstance(x, np.ndarray) and x.dtype == dtype and x.flags.c_contiguous, (type(x), x.dtype)
    assert not writable or x.flags.writeable
    return x.ctypes.data


def _hash_consts(hasher, layer: int):
    key = (id(hasher), layer)
    got = _HASH_ARGS.get(key)
    if got is None or got[0] is not hasher:
        li = hasher.layer_ids.index(layer)
        consts = tuple(
            np.ascontiguousarray(t.contiguous().numpy(), dtype=np.int64)
            for t in (hasher.token_map, hasher.multipliers[li], hasher.primes[li], hasher.offsets[li])
        )
        got = _HASH_ARGS[key] = (hasher, consts)
    return got[1]


def _new_tail(hasher, ids: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
    k = hasher.tail_len
    return torch.cat([tail, hasher.token_map[ids[-k:]]])[-k:].clone()  # (= the reference's comp[-k:])


def _threads(threads) -> int:
    return max(1, int(threads or 1))


def ngram_hashes(hasher, tokens: torch.Tensor, layer: int, tail: torch.Tensor, threads: int | None = None):
    """tt/engram_bfp8.ngram_hashes (= reference EngramHasher.hashes) bit for bit: (rows [S, slots] int64, new tail)."""
    from models.demos.deepseek_v41_flash_d_p.reference.ngram import DEAD

    ids = tokens.reshape(-1).long().contiguous()
    tail = tail.long().contiguous()
    assert tail.shape[0] == hasher.tail_len, (tail.shape, hasher.tail_len)
    tmap, mult, primes, offsets = _hash_consts(hasher, layer)
    s, m = int(ids.shape[0]), hasher.max_ngram_size
    out = torch.empty(s, (m - 1) * hasher.n_heads, dtype=torch.int64)
    bad = build().ngram_hash(
        ids.data_ptr(),
        tail.data_ptr(),
        tmap.ctypes.data,
        mult.ctypes.data,
        primes.ctypes.data,
        offsets.ctypes.data,
        s,
        hasher.tail_len,
        int(tmap.shape[0]),
        m,
        hasher.n_heads,
        int(hasher.pad_id),
        int(DEAD),
        out.data_ptr(),
        _threads(threads),
    )
    assert bad == 0, f"token id outside the vocabulary ({bad} reads)"
    return out, _new_tail(hasher, ids, tail)


def _check_tiles(packed, ids: torch.Tensor, valid: int, outs, q: int):
    assert isinstance(packed, np.ndarray) and packed.dtype == np.uint8 and packed.ndim == 2, type(packed)
    assert packed.shape[1] % 17 == 0 and packed.strides == (packed.shape[1], 1), (packed.shape, packed.strides)
    assert ids.shape[0] >= valid and ids.dtype == torch.int64, (ids.shape, ids.dtype, valid)
    slots = ids.shape[1]
    nbytes = q * slots * (packed.shape[1] // 17) * 16 * 17 // 16
    assert q % 32 == 0 and (slots * (packed.shape[1] // 17)) % 2 == 0, (q, slots)
    for o in outs:
        assert (o.numel() if isinstance(o, torch.Tensor) else o.size) == nbytes, (nbytes,)
    return [_ptr(o, np.uint8, writable=True) for o in outs]


def tile_rows_chips(packed, ids: torch.Tensor, valid: int, outs: tuple, q: int, threads: int | None = None) -> None:
    """tt/engram_bfp8.tile_rows_chips bit for bit: rows [k q, (k+1) q) of the chunk -> the TILE bytes of
    [q, slots * 256] in outs[k] (numpy uint8 arrays or uint8 tensors; rows past ``valid`` 0). An id outside the table
    raises (checked before any byte is written)."""
    ids = ids.contiguous()
    ptrs = _check_tiles(packed, ids, int(valid), outs, int(q))
    bad = build().tile_chips(
        packed.ctypes.data,
        packed.shape[0],
        packed.shape[1],
        ids.data_ptr(),
        ids.shape[1],
        int(valid),
        ptrs,
        int(q),
        _threads(threads),
    )
    assert bad == 0, f"engram row out of range ({bad} ids)"


def tile_rows(packed, ids: torch.Tensor, n: int, threads: int | None = None, out=None) -> torch.Tensor:
    """tt/engram_bfp8.tile_rows bit for bit (ids [valid <= n, slots] -> TILE bytes of [n, slots * 256])."""
    w = ids.shape[1] * (packed.shape[1] // 17) * 16
    if out is None:
        out = torch.empty(n * w * 17 // 16, dtype=torch.uint8)
    assert out.dtype == torch.uint8 and out.numel() == n * w * 17 // 16, (out.dtype, out.numel(), n, w)
    tile_rows_chips(packed, ids, ids.shape[0], (out,), n, threads)
    return out


def hash_tile_chips(
    hasher, tokens: torch.Tensor, layer: int, tail: torch.Tensor, packed, outs: tuple, q: int, threads=None
) -> torch.Tensor:
    """HOST_HIDE's job of one layer (TtEngramRows.fill_tp, resident table) in ONE call and one GIL release:
    ngram_hashes of tokens [valid] then tile_rows_chips of those rows into outs (len(outs) * q >= valid; rows past
    valid 0). Returns the new tail. Same bytes as the two numba calls."""
    from models.demos.deepseek_v41_flash_d_p.reference.ngram import DEAD

    ids = tokens.reshape(-1).long().contiguous()
    tail = tail.long().contiguous()
    valid = int(ids.shape[0])
    assert tail.shape[0] == hasher.tail_len, (tail.shape, hasher.tail_len)
    if valid == 0:
        tile_rows_chips(
            packed, torch.empty(0, hasher.tail_len * hasher.n_heads, dtype=torch.int64), 0, outs, q, threads
        )
        return tail.clone()
    tmap, mult, primes, offsets = _hash_consts(hasher, layer)
    m = hasher.max_ngram_size
    h = torch.empty(valid, (m - 1) * hasher.n_heads, dtype=torch.int64)
    ptrs = _check_tiles(packed, h, valid, outs, int(q))
    bad_ids, bad_rows = build().hash_tile_chips(
        ids.data_ptr(),
        tail.data_ptr(),
        tmap.ctypes.data,
        mult.ctypes.data,
        primes.ctypes.data,
        offsets.ctypes.data,
        valid,
        hasher.tail_len,
        int(tmap.shape[0]),
        m,
        hasher.n_heads,
        int(hasher.pad_id),
        int(DEAD),
        h.data_ptr(),
        packed.ctypes.data,
        packed.shape[0],
        packed.shape[1],
        ptrs,
        int(q),
        _threads(threads),
    )
    assert bad_ids == 0, f"token id outside the vocabulary ({bad_ids} reads)"
    assert bad_rows == 0, f"engram row out of range ({bad_rows} ids)"
    return _new_tail(hasher, ids, tail)


def fill_job(
    hasher,
    layer: int,
    ids: torch.Tensor,
    tail: torch.Tensor,
    packed,
    outs: tuple,
    q: int,
    threads=None,
    valid: int | None = None,
    want_tail: torch.Tensor | None = None,
):
    """TtEngramRows.fill_tp's host work (HOST_HIDE=on, resident table) in ONE C++ call: ``valid`` (None: up to the
    first id outside the vocabulary, TtEngramRows._valid), the new tail (TtEngramRows.next_tail; checked against
    ``want_tail``), the hash of ids [valid] and the gather of those rows into every chip's share (outs, rows past valid
    0). -> (new tail [tail_len] int64 numpy, {"hash": ms, "lookup_pack": ms} measured in C++).

    No torch operation on the way (each releases the GIL, and every re-take waits up to a switch interval while another
    thread runs Python): ids / tail / want_tail are read through data_ptr (ids int64 or int32, 1-D contiguous; tail and
    want_tail int64 contiguous), so the job takes the GIL back once, after the C++ call."""
    from models.demos.deepseek_v41_flash_d_p.reference.ngram import DEAD

    if ids.dtype not in (torch.int64, torch.int32) or not ids.is_contiguous() or ids.dim() != 1:
        ids = ids.reshape(-1).long().contiguous()  # (torch operations: only for ids the caller did not prepare)
    assert tail.dtype == torch.int64 and tail.is_contiguous() and tail.numel() == hasher.tail_len, (
        tail.dtype,
        tail.shape,
    )
    if want_tail is not None:
        assert want_tail.dtype == torch.int64 and want_tail.is_contiguous() and want_tail.numel() == hasher.tail_len
    slots = hasher.tail_len * hasher.n_heads
    s = ids.numel()
    assert len(outs) * q >= s, (len(outs), q, s)
    ptrs = _OUTS.get(id(outs))
    if ptrs is None or ptrs[0] is not outs:
        nbytes = q * slots * (packed.shape[1] // 17) * 16 * 17 // 16
        assert q % 32 == 0 and packed.dtype == np.uint8 and packed.strides == (packed.shape[1], 1), (q, packed.strides)
        for o in outs:
            assert (o.numel() if isinstance(o, torch.Tensor) else o.size) == nbytes, (nbytes,)
        ptrs = _OUTS[id(outs)] = (outs, [_ptr(o, np.uint8, writable=True) for o in outs])
    tmap, mult, primes, offsets = _hash_consts(hasher, layer)
    new_tail = np.empty(hasher.tail_len, dtype=np.int64)
    n_valid, bad_ids, bad_rows, tail_ok, hash_us, tile_us = build().engram_job(
        ids.data_ptr(),
        ids.dtype == torch.int32,
        s,
        -1 if valid is None else int(valid),
        tail.data_ptr(),
        0 if want_tail is None else want_tail.data_ptr(),
        new_tail.ctypes.data,
        tmap.ctypes.data,
        mult.ctypes.data,
        primes.ctypes.data,
        offsets.ctypes.data,
        hasher.tail_len,
        int(tmap.shape[0]),
        hasher.max_ngram_size,
        hasher.n_heads,
        int(hasher.pad_id),
        int(DEAD),
        packed.ctypes.data,
        packed.shape[0],
        packed.shape[1],
        ptrs[1],
        int(q),
        _threads(threads),
    )
    assert bad_ids == 0, f"token id outside the vocabulary ({bad_ids} reads)"
    assert bad_rows == 0, f"engram row out of range ({bad_rows} ids)"
    assert tail_ok, (layer, new_tail, want_tail)
    return new_tail, {"hash": hash_us / 1e3, "lookup_pack": tile_us / 1e3}


_OUTS = {}  # id(outs tuple) -> (the tuple, its addresses): the staging views are made once a layer


def _pack(w, s, exps, e_stride, mags, m_stride, threads):
    from .engram_bfp8 import _BYTE, _EXP, _GAPS, _ZERO

    global _TABLES
    if _TABLES is None:
        _TABLES = (np.ascontiguousarray(_EXP.int().numpy()), np.ascontiguousarray(_ZERO.numpy()), _BYTE.numpy().copy())
    et, zt, bt = _TABLES
    assert bt.shape[1] == _GAPS
    n, d = w.shape
    assert w.dtype == torch.uint8 and s.dtype == torch.uint8 and s.shape[0] == n and d % s.shape[1] == 0
    assert w.stride(1) == 1 and s.stride(1) == 1, (w.stride(), s.stride())
    build().pack(
        w.data_ptr(),
        w.stride(0),
        s.data_ptr(),
        s.stride(0),
        n,
        d,
        s.shape[1],
        exps,
        e_stride,
        mags,
        m_stride,
        et.ctypes.data,
        zt.ctypes.data,
        bt.ctypes.data,
        _GAPS,
        _threads(threads),
    )


_TABLES = None


def pack_rows(w: torch.Tensor, s: torch.Tensor, threads: int | None = None):
    """tt/engram_bfp8.pack_rows bit for bit: w [N, 256], s [N, 8] uint8 -> (exponents [N, 16], magnitudes [N, 256])."""
    n, d = w.shape
    exps = torch.empty(n, d // 16, dtype=torch.uint8)
    mags = torch.empty(n, d, dtype=torch.uint8)
    _pack(w, s, exps.data_ptr(), d // 16, mags.data_ptr(), d, threads)
    return exps, mags


def pack_into(w: torch.Tensor, s: torch.Tensor, out, threads: int | None = None):
    """tt/engram_bfp8.pack_into bit for bit: packed rows into out [N, 272] (numpy uint8 rows, exponents then
    magnitudes; a row-slice view of the resident table)."""
    n, d = w.shape
    g = d // 16
    assert isinstance(out, np.ndarray) and out.dtype == np.uint8 and out.shape == (n, g + d), (out.shape, n, d)
    assert out.strides[1] == 1 and out.flags.writeable, out.strides
    base = out.ctypes.data
    _pack(w, s, base, out.strides[0], base + g, out.strides[0], threads)
