# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU (no device): tt/engram_host_cpp.py (ENGRAM_HOST_IMPL=cpp, the C++ engram host work) equals the numba path
(tt/engram_bfp8.py) bit for bit: the n-gram hash (and so the reference EngramHasher), the bfloat8_b pack of table
rows, the gather of packed rows into every chip's TILE bytes, and the fused hash + gather of HOST_HIDE's job.

Real inputs when this host has them: the canonical prompt's ids (spec galaxy_8x4-full), the checkpoint's tokenizer
(the real compressed token map, multipliers, primes) and both engram layers' memory-mapped tables (the rows a chunk
reads, packed); else synthetic ones of the same format. Edge cases: a fresh (DEAD) tail and chunks shorter than the
tail (start < 3), engine pad past ``valid``, ids outside the vocabulary / table (refused before any byte is written).

    python_env/bin/python -m pytest --noconftest -q models/demos/deepseek_v41_flash_d_p/tests/test_engram_host_cpp.py
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pytest
import torch

from models.demos.deepseek_v41_flash_d_p.reference.ngram import DEAD, EngramHasher
from models.demos.deepseek_v41_flash_d_p.tt import engram_bfp8 as nb
from models.demos.deepseek_v41_flash_d_p.tt import engram_host_cpp as cpp

SPEC = "models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml"
_raises = pytest.raises  # allow-pytest.raises: host-side assertions, a CPU test run with --noconftest (no fixture)
CHUNK, CHIPS = 5120, 32  # the Galaxy's chunk, 8 x 4 chips: q = 160 rows a chip


@lru_cache(None)
def real():
    """(hasher, prompt ids, {layer: EngramTable}) from the spec's checkpoint, or None when this host lacks it."""
    try:
        from transformers import AutoTokenizer

        from models.demos.common.bringup.core.spec import Spec
        from models.demos.common.bringup.reference.golden import hf_path
        from models.demos.common.bringup.reference.prompt import tokens
        from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config
        from models.demos.deepseek_v41_flash_d_p.reference.weights import Checkpoint, EngramTable

        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
        spec = Spec.load(os.path.join(repo, SPEC))
        path = hf_path(spec)
        cfg = Config.from_dir(path)
        hasher = EngramHasher.build(cfg.raw, AutoTokenizer.from_pretrained(path))
        ids = tokens(spec, 4 * CHUNK)
        ck = Checkpoint(path)
        return hasher, ids, {i: EngramTable(ck, i) for i in hasher.layer_ids}
    except Exception as e:  # (no checkpoint / prompt on this host)
        print(f"real inputs unavailable: {e!r}")
        return None


def synthetic_hasher(seed: int) -> EngramHasher:
    g = torch.Generator().manual_seed(seed)
    m, h = 4, 8
    return EngramHasher(
        layer_ids=(1, 14),
        max_ngram_size=m,
        n_heads=h,
        token_map=torch.randint(0, 100_000, (129_280,), generator=g),
        pad_id=2,
        multipliers=torch.randint(1, 2**61, (2, m), generator=g) * 2 + 1,  # (products overflow int64)
        primes=torch.randint(1_000_000, 4_000_000, (2, m - 1, h), generator=g),
        offsets=torch.randint(0, 10_000_000, (2, (m - 1) * h), generator=g),
    )


def hashers():
    out = [("synthetic", synthetic_hasher(0), None)]
    r = real()
    if r is not None:
        out.insert(0, ("real", r[0], r[1]))
    return out


def chunks(ids: torch.Tensor):
    """Chunk sizes that start a request below the tail length (1, then 2: start < 3), then full chunks and a tail."""
    sizes, a = [1, 2, 37, CHUNK, CHUNK, 3, 2 * CHUNK], 0
    for n in sizes:
        if a + n > ids.numel():
            break
        yield ids[a : a + n]
        a += n


# ------------------------------------------------------------------------------------------------------- the hash
@pytest.mark.parametrize("threads", [1, 7, 24])
def test_hash_bit_identical(threads):
    for name, hs, prompt in hashers():
        ids_all = prompt if prompt is not None else torch.randint(0, 129_280, (4 * CHUNK,))
        rnd = torch.randint(0, hs.token_map.numel(), (4 * CHUNK,), generator=torch.Generator().manual_seed(threads))
        for ids_src in (ids_all, rnd):
            for layer in hs.layer_ids:
                tail_n = tail_c = hs.new_tail()  # (DEAD: the request's start)
                for k, ids in enumerate(chunks(ids_src)):
                    want, want_tail = nb.ngram_hashes(hs, ids, layer, tail_n, threads)
                    got, got_tail = cpp.ngram_hashes(hs, ids, layer, tail_c, threads)
                    assert torch.equal(got, want), (name, layer, k, ids.numel(), (got != want).sum())
                    assert torch.equal(got_tail, want_tail), (name, layer, k)
                    if k < 3:  # (and the reference itself, on the short chunks)
                        ref, ref_tail = hs.hashes(ids, layer, tail_n)
                        assert torch.equal(got, ref) and torch.equal(got_tail, ref_tail), (name, layer, k)
                    tail_n, tail_c = want_tail, got_tail


def test_hash_partial_tails():
    hs = synthetic_hasher(1)
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(0, 129_280, (CHUNK,), generator=g)
    for tail in (
        torch.tensor([DEAD, DEAD, int(hs.token_map[5])]),
        torch.tensor([DEAD, 17, 99]),
        hs.token_map[torch.randint(0, 129_280, (3,), generator=g)],
    ):
        for layer in hs.layer_ids:
            want = nb.ngram_hashes(hs, ids, layer, tail, 4)
            got = cpp.ngram_hashes(hs, ids, layer, tail, 4)
            assert all(torch.equal(a, b) for a, b in zip(got, want)), (tail, layer)


def test_hash_refuses_pad_ids():
    """Engine pad (0xFFFFFFFF, -1, any id outside the vocabulary) is cut by TtEngramRows._valid before the hash; were
    one to reach the C++ hash it raises (numba would read outside the token map)."""
    hs = synthetic_hasher(2)
    for bad in (-1, 0xFFFFFFFF, hs.token_map.numel()):
        ids = torch.arange(64)
        ids[40] = bad
        with _raises(AssertionError, match="outside the vocabulary"):
            cpp.ngram_hashes(hs, ids, 1, hs.new_tail(), 4)


# --------------------------------------------------------------------------------------------- pack and gather
def packed_from(table, rows: torch.Tensor):
    """The real table's ``rows`` (unique ids) packed as the resident bfp8 table packs them (numba pack_into) and by
    the C++ pack_into; checked equal and returned (one compact [U, 272] table)."""
    w = table.weight[rows].contiguous()
    s = table.scale[rows].contiguous()
    a = np.full((rows.numel(), 272), 0xAB, dtype=np.uint8)
    b = np.full((rows.numel(), 272), 0xCD, dtype=np.uint8)
    nb.pack_into(w, s, a, 8)
    cpp.pack_into(w, s, b, 8)
    assert np.array_equal(a, b), "pack_into: C++ != numba"
    e1, m1 = nb.pack_rows(w, s, threads=8)
    e2, m2 = cpp.pack_rows(w, s, threads=8)
    assert torch.equal(e1, e2) and torch.equal(m1, m2), "pack_rows: C++ != numba"
    return a


def _tables():
    """[(name, hasher, packed [U, 272] numpy, ids of a chunk [S], {layer: (rows of the chunk remapped into packed)})]"""
    out = []
    r = real()
    if r is not None:
        hs, prompt, tabs = r
        ids = prompt[CHUNK : 2 * CHUNK]
        tail = hs.token_map[prompt[CHUNK - 3 : CHUNK]]
        for layer, tab in tabs.items():
            h, _ = nb.ngram_hashes(hs, ids, layer, tail, 8)
            uniq, inv = torch.unique(h, return_inverse=True)
            out.append((f"real L{layer}", h, inv.view_as(h), packed_from(tab, uniq)))
    g = torch.Generator().manual_seed(11)
    w = torch.randint(0, 256, (1 << 16, 256), dtype=torch.uint8, generator=g)
    w[w == 0x7F] = 0  # (e4m3fn NaN, never stored)
    w[w == 0xFF] = 0
    s = torch.randint(100, 140, (1 << 16, 8), dtype=torch.uint8, generator=g)
    packed = np.empty((1 << 16, 272), dtype=np.uint8)
    nb.pack_into(w, s, packed, 8)
    b = np.empty_like(packed)
    cpp.pack_into(w, s, b, 8)
    assert np.array_equal(packed, b)
    ids = torch.randint(0, 1 << 16, (CHUNK, 24), generator=g)
    out.append(("synthetic", ids, ids, packed))
    return out


@pytest.mark.parametrize("threads", [1, 8, 24])
def test_tile_rows_chips_bit_identical(threads):
    for name, _, ids, packed in _tables():
        for chips, q in ((CHIPS, CHUNK // CHIPS), (4, CHUNK // 4), (1, CHUNK)):
            for valid in (CHUNK, CHUNK - 37, 5, 0):  # (engine pad past valid: zero rows)
                nbytes = q * 24 * 256 * 17 // 16
                want = tuple(np.full(nbytes, 0xAB, dtype=np.uint8) for _ in range(chips))
                got = tuple(np.full(nbytes, 0xCD, dtype=np.uint8) for _ in range(chips))
                nb.tile_rows_chips(packed, ids[:valid], valid, want, q, threads)
                cpp.tile_rows_chips(packed, ids[:valid], valid, got, q, threads)
                for k in range(chips):
                    assert np.array_equal(got[k], want[k]), (name, chips, valid, k)
        want = nb.tile_rows(packed, ids[:300], 320, 4)
        got = cpp.tile_rows(packed, ids[:300], 320, 4)
        assert torch.equal(got, want), name


def test_hash_tile_chips_fused():
    """HOST_HIDE's job in one C++ call (hash of ids[:valid] + gather into every chip) == numba's two calls."""
    r = real()
    cases = []
    if r is not None:
        hs, prompt, tabs = r
        for layer, tab in tabs.items():
            ids = prompt[2 * CHUNK : 3 * CHUNK]
            tail = hs.token_map[prompt[2 * CHUNK - 3 : 2 * CHUNK]]
            h, _ = nb.ngram_hashes(hs, ids, layer, tail, 8)
            uniq = torch.unique(h)
            cases.append((hs, layer, ids, tail, tab, uniq))
    hs = synthetic_hasher(3)
    for name_case in cases[:2]:
        hs_r, layer, ids, tail, tab, uniq = name_case
        # a table holding exactly the rows the real hashes name, indexed by the real row id (numpy view over a
        # sparse anonymous mapping: only the touched pages are ever allocated)
        n_rows = int(tab.num_rows)
        import mmap

        buf = mmap.mmap(
            -1, n_rows * 272, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | getattr(mmap, "MAP_NORESERVE", 0x4000)
        )
        packed = np.frombuffer(buf, dtype=np.uint8).reshape(n_rows, 272)
        packed[uniq.numpy()] = packed_from(tab, uniq)
        for valid in (CHUNK, CHUNK - 100):
            want = tuple(np.full(160 * 6528, 0xAB, dtype=np.uint8) for _ in range(CHIPS))
            got = tuple(np.full(160 * 6528, 0xCD, dtype=np.uint8) for _ in range(CHIPS))
            h, want_tail = nb.ngram_hashes(hs_r, ids[:valid], layer, tail, 8)
            nb.tile_rows_chips(packed, h, valid, want, 160, 8)
            got_tail = cpp.hash_tile_chips(hs_r, ids[:valid], layer, tail, packed, got, 160, 24)
            assert torch.equal(got_tail, want_tail)
            assert all(np.array_equal(a, b) for a, b in zip(got, want)), (layer, valid)
        del packed
        buf.close()
    # synthetic: hash rows modulo a small table via offsets / primes small enough
    hs.primes = torch.randint(1000, 4000, hs.primes.shape, generator=torch.Generator().manual_seed(4))
    hs.offsets = torch.cat([torch.zeros(2, 1, dtype=torch.int64), hs.primes.view(2, -1)[:, :-1].cumsum(-1)], -1)
    packed = np.random.default_rng(0).integers(0, 256, (int(hs.primes.view(2, -1).sum(-1).max()), 272), dtype=np.uint8)
    ids = torch.randint(0, 129_280, (CHUNK,), generator=torch.Generator().manual_seed(9))
    for layer in hs.layer_ids:
        for valid, tail in (
            (CHUNK, hs.new_tail()),
            (2, hs.new_tail()),
            (CHUNK - 1, hs.token_map[:3].clone()),
            (0, hs.new_tail()),
        ):
            want = tuple(np.full(160 * 6528, 0xAB, dtype=np.uint8) for _ in range(CHIPS))
            got = tuple(np.full(160 * 6528, 0xCD, dtype=np.uint8) for _ in range(CHIPS))
            if valid:
                h, want_tail = nb.ngram_hashes(hs, ids[:valid], layer, tail, 8)
            else:
                h, want_tail = torch.empty(0, 24, dtype=torch.int64), tail.clone()
            nb.tile_rows_chips(packed, h, valid, want, 160, 8)
            got_tail = cpp.hash_tile_chips(hs, ids[:valid], layer, tail, packed, got, 160, 16)
            assert torch.equal(got_tail, want_tail), (layer, valid)
            assert all(np.array_equal(a, b) for a, b in zip(got, want)), (layer, valid)


def test_tile_range_check():
    packed = np.zeros((64, 272), dtype=np.uint8)
    ids = torch.zeros(4 * 32, 24, dtype=torch.int64)
    ids[70, 3] = 64  # (one past the table)
    outs = tuple(np.full(32 * 24 * 256 * 17 // 16, 7, dtype=np.uint8) for _ in range(4))
    with _raises(AssertionError, match=r"out of range \(1 ids\)"):
        cpp.tile_rows_chips(packed, ids, 128, outs, 32, 4)
    assert all((o == 7).all() for o in outs), "bytes written before the range check"
    ids[71, 0] = -5
    with _raises(AssertionError, match=r"out of range \(2 ids\)"):
        cpp.tile_rows_chips(packed, ids, 128, outs, 32, 4)
    cpp.tile_rows_chips(packed, ids, 70, outs, 32, 4)  # (rows past valid are never read)


def test_concurrent_callers():
    """Two Python threads calling at once (the engram worker and the table builder): same bytes as alone."""
    from concurrent.futures import ThreadPoolExecutor

    hs = synthetic_hasher(6)
    ids = torch.randint(0, 129_280, (CHUNK,), generator=torch.Generator().manual_seed(1))
    want = [cpp.ngram_hashes(hs, ids, 1 + 13 * (k % 2), hs.new_tail(), 8)[0] for k in range(2)]
    with ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(cpp.ngram_hashes, hs, ids, 1 + 13 * (k % 2), hs.new_tail(), 3 + k) for k in range(40)]
        for k, f in enumerate(futs):
            assert torch.equal(f.result()[0], want[k % 2])


def _numba_fill_tp(hs, layer, ids, tail, packed, outs, q, valid=None):
    """TtEngramRows.fill_tp's host work today (HOST_HIDE=on, resident table, numba): _valid, the hash of ids[:valid],
    the gather into every chip's share, next_tail."""
    vocab = hs.token_map.numel()
    ids = ids.reshape(-1).long()
    if valid is None:
        bad = ((ids < 0) | (ids >= vocab)).nonzero()
        valid = int(bad[0]) if bad.numel() else ids.numel()
    valid = max(0, min(int(valid), ids.numel()))
    if valid:
        h, new_tail = nb.ngram_hashes(hs, ids[:valid], layer, tail, 8)
    else:
        h, new_tail = torch.empty(0, hs.tail_len * hs.n_heads, dtype=torch.int64), tail.clone()
    nb.tile_rows_chips(packed, h, valid, outs, q, 8)
    return new_tail


def test_fill_job_equals_numba_fill_tp():
    """ENGRAM_HOST_IMPL=cpp's whole fill_tp job (one C++ call) == the numba job: bytes of every chip and the new tail;
    engine pad (0xFFFFFFFF, -1) ends the valid prefix; int32 ids; explicit valid; a wrong want_tail raises."""
    r = real()
    if r is not None:
        hs, prompt, tabs = r
        layer, tab = 14, tabs[14]
        ids = prompt[3 * CHUNK : 4 * CHUNK].clone()
        tail = hs.token_map[prompt[3 * CHUNK - 3 : 3 * CHUNK]].clone()
        n_rows = int(tab.num_rows)
    else:
        hs = synthetic_hasher(8)
        layer, tab = 1, None
        ids = torch.randint(0, 129_280, (CHUNK,), generator=torch.Generator().manual_seed(2))
        tail = hs.token_map[:3].clone()
        n_rows = int(hs.offsets[0, -1] + hs.primes[0, -1, -1])
    import mmap

    buf = mmap.mmap(
        -1, n_rows * 272, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS | getattr(mmap, "MAP_NORESERVE", 0x4000)
    )
    packed = np.frombuffer(buf, dtype=np.uint8).reshape(n_rows, 272)
    h, _ = nb.ngram_hashes(hs, ids, layer, tail, 8)
    uniq = torch.unique(h)
    if tab is not None:
        packed[uniq.numpy()] = packed_from(tab, uniq)
    else:
        packed[uniq.numpy()] = np.random.default_rng(1).integers(0, 256, (uniq.numel(), 272), dtype=np.uint8)
    q = CHUNK // CHIPS
    padded = ids.clone()
    padded[CHUNK - 300 :] = 0xFFFFFFFF  # (engine pad past actual_end)
    neg = ids.clone()
    neg[2:] = -1  # (2 real tokens: valid < tail_len)
    cases = [
        (ids, tail, None),
        (padded, tail, None),
        (neg, hs.new_tail(), None),  # (the request's first chunk, 2 tokens)
        (ids.to(torch.int32), hs.new_tail(), None),
        (ids, tail, 1000),
        (ids, tail, 0),
    ]
    for k, (x, t, valid) in enumerate(cases):
        want = tuple(np.full(q * 6528, 0xAB, dtype=np.uint8) for _ in range(CHIPS))
        got = tuple(np.full(q * 6528, 0xCD, dtype=np.uint8) for _ in range(CHIPS))
        want_tail = _numba_fill_tp(hs, layer, x, t, packed, want, q, valid)
        got_tail, times = cpp.fill_job(
            hs, layer, x, t, packed, got, q, 24, valid, want_tail=want_tail.long().contiguous()
        )
        assert np.array_equal(got_tail, want_tail.numpy()), k
        assert all(np.array_equal(a, b) for a, b in zip(got, want)), k
        assert set(times) == {"hash", "lookup_pack"}
    with _raises(AssertionError):
        cpp.fill_job(hs, layer, ids, tail, packed, got, q, 8, want_tail=torch.tensor([1, 2, 3]))
    del packed
    buf.close()
