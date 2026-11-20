# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""V41_PREFILL_HOST_IDS (tt-blaze DS41F-0163 WS13) on the host: no chips, no ttnn.

    python -m pytest -c /dev/null --noconftest -p no:cacheprovider models/demos/deepseek_v3_d_p/tests/v41_prefix_reuse/test_v41_host_ids.py

* ``host_ids.py``: the shared-memory ring round trip, wrap-around, header matching across interleaved slots, a re-initialized
  file (new session), a missing file;
* ``engram_prefetch.py`` with the checkpoint's REAL ``NgramHashState`` (inference/engram.py of DeepSeek-V4.1-Flash-1017) under an
  EngramHost-like host whose ``hash_ids`` is EngramHost's: the early (scratch) hash equals the real one at every chunk boundary,
  including starts below the n-gram look-back and dead (image) ids, without touching the slot's history; a matching prefetch
  hands back exactly the rows a gather now would; a wrong guess (other ids / slot / start) is rejected and leaves every later
  hash unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
MODEL_DIR = os.environ.get("DSV41_TEST_MODEL_DIR", "/mnt/tt-data/sdawle/models/DeepSeek-V4.1-Flash-1017")


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


host_ids = _load("v41_host_ids", "models/demos/common/prefill/runners/host_ids.py")
engram_prefetch = _load("v41_engram_prefetch", "models/demos/deepseek_v3_d_p/tt/v41/engram_prefetch.py")


# ---- the shared-memory ring ------------------------------------------------------------------------------------------
def _ring(n=4, chunk=8):
    d = tempfile.mkdtemp()
    path = os.path.join(d, "ids")
    return path, host_ids.HostIdsWriter(path, chunk, n), host_ids.HostIdsReader(path)


def test_round_trip_and_match():
    path, w, r = _ring()
    w.write(0x20000001, 0, 8, list(range(8)))
    w.write(0x20000002, 0, 5, [7, 7, 7, 7, 7])
    w.write(0x60000001, 8, 12, [1, 2, 3, 4])
    assert r.match(0x20000002, 0, 5) == 1  # interleaved slots: the first entry with this header at / after the cursor
    assert r.cursor == 2
    word, start, end, ids = r.entry(2)
    assert (word, start, end, ids.tolist()) == (0x60000001, 8, 12, [1, 2, 3, 4])
    assert r.match(0x20000001, 0, 8) is None  # behind the cursor: not matched again
    assert r.entry(3) is None  # not written yet


def test_wrap_around_and_overwrite():
    path, w, r = _ring(n=4)
    for s in range(10):
        w.write(s, s * 8, s * 8 + 8, [s] * 8)
    assert r.entry(5) is None  # overwritten by chunk 9
    assert r.entry(9)[3].tolist() == [9] * 8
    assert r.match(7, 56, 64) == 7  # the cursor jumps to the last n_entries written
    assert r.entry(10) is None


def test_torn_entry_is_not_read():
    path, w, r = _ring()
    s = w.write(1, 0, 8, [5] * 8)
    off = host_ids.HEADER_BYTES + (s % w.n) * w.ebytes
    import struct

    struct.pack_into("<Q", w._mm, off, 2 * s + 1)  # a writer mid-update
    assert r.entry(s) is None
    struct.pack_into("<Q", w._mm, off, 2 * s + 2)
    assert r.entry(s)[3].tolist() == [5] * 8


def test_new_session_resets_cursor_and_missing_file():
    path, w, r = _ring()
    for s in range(3):
        w.write(s, 0, 8, [s] * 8)
    assert r.match(2, 0, 8) == 2 and r.cursor == 3
    w2 = host_ids.HostIdsWriter(path, 8, 4)  # an engine restart re-initializes the file
    w2.write(9, 0, 8, [9] * 8)
    assert r.match(9, 0, 8) == 0 and r.cursor == 1
    missing = host_ids.HostIdsReader(os.path.join(tempfile.mkdtemp(), "none"))
    assert missing.match(0, 0, 8) is None and missing.entry(0) is None


# ---- the Engram prefetch ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def eg_module():
    path = os.path.join(MODEL_DIR, "inference", "engram.py")
    if not os.path.exists(path):
        pytest.skip(f"no checkpoint engram.py at {path}")
    return _load_abs("ckpt_engram", path)


def _load_abs(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TinyEngramHost:
    """EngramHost's hash side for real (``hash_ids`` copied from tt-blaze engram_host.py, the checkpoint's NgramHashState), with
    a deterministic stand-in for the table gather (row = a function of the table row index and the layer)."""

    def __init__(self, eg, n_slots: int, max_seq_len: int):
        from transformers import AutoTokenizer

        cfg = json.load(open(os.path.join(MODEL_DIR, "inference", "config.json")))
        args = types.SimpleNamespace(**{k: v for k, v in cfg.items() if k.startswith("engram_")})
        args.max_batch_size, args.max_seq_len = 1, max_seq_len
        tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.vocab_size = int(cfg.get("vocab_size", 129280))
        self.layout = eg.EngramLayout.from_args(args)
        self.layer_ids = tuple(self.layout.layer_ids)
        self._hash = [eg.NgramHashState(args, self.layout, tok) for _ in range(n_slots)]
        self.head_dim = 4
        self.row_elems = (self.layout.max_ngram_size - 1) * self.layout.n_heads * self.head_dim
        self._layer_pool = ThreadPoolExecutor(2)

    def hash_ids(self, slot, token_ids, start_pos):
        ids = torch.as_tensor(token_ids, dtype=torch.long).reshape(1, -1)
        live = ids < self.vocab_size
        if bool(live.all()):
            return self._hash[slot](ids, start_pos)[0]
        return self._hash[slot](torch.where(live, ids, 0), start_pos, live)[0]

    def _dead(self, token_ids):
        d = torch.as_tensor(token_ids, dtype=torch.long).reshape(-1) >= self.vocab_size
        return d if bool(d.any()) else None

    def _gather(self, lid, idx):
        v = (np.asarray(idx, dtype=np.int64)[:, None] * 31 + lid * 7 + np.arange(self.head_dim)) % 997
        return torch.from_numpy(v.astype(np.float32)).to(torch.bfloat16)

    def rows_for(self, slot, token_ids, start_pos, layers=None):
        raise AssertionError("not used here")


def _prompt(rng, n, vocab, dead=()):
    ids = rng.integers(0, vocab, size=n).tolist()
    for i in dead:
        ids[i] = vocab + 5  # an image-span virtual id
    return ids


def test_scratch_hash_equals_real_and_leaves_history(eg_module):
    eh = TinyEngramHost(eg_module, n_slots=2, max_seq_len=512)
    pf = engram_prefetch.EngramPrefetch(eh, chunk_size=64)
    rng = np.random.default_rng(7)
    ids = _prompt(rng, 300, eh.vocab_size, dead=(70, 71, 140))
    bounds = [0, 1, 3, 5, 69, 70, 72, 133, 197, 261, 300]  # starts below the look-back, across dead ids, ragged widths
    for a, b in zip(bounds[:-1], bounds[1:]):
        part = ids[a:b]
        before = eh._hash[0].cache.clone()
        early = pf._scratch_hash(0, part, a)
        assert torch.equal(eh._hash[0].cache, before), "the early hash wrote the slot's history"
        real = eh.hash_ids(0, part, a)
        assert torch.equal(early, real), f"early != real for [{a}, {b})"


def test_matching_prefetch_returns_the_rows_a_gather_would(eg_module):
    eh = TinyEngramHost(eg_module, n_slots=2, max_seq_len=512)
    ref = TinyEngramHost(eg_module, n_slots=2, max_seq_len=512)
    pf = engram_prefetch.EngramPrefetch(eh, chunk_size=64)
    rng = np.random.default_rng(11)
    ids = _prompt(rng, 192, eh.vocab_size, dead=(100,))
    for a in range(0, 192, 64):
        part = ids[a : a + 64]
        got = {k: f.result() for k, f in pf.rows_async(0, part, a).items()}  # a > 0: prefetched in the previous step
        h = ref.hash_ids(0, part, a)
        want = {
            lid: ref._gather(lid, h[:, i, :].numpy().astype(np.int64).reshape(-1)).reshape(len(part), ref.row_elems)
            for i, lid in enumerate(ref.layer_ids)
        }
        d = ref._dead(part)
        for lid in want:
            if d is not None:
                want[lid][d] = 0
            assert torch.equal(got[lid], want[lid]), (a, lid)
        if a + 64 < 192:
            pf.prefetch(0, ids[a + 64 : a + 128], a + 64)  # the next chunk, while "this one runs"
    assert pf.stats["hits"] == 2 and pf.stats["misses"] == 0


def test_wrong_guess_is_rejected_and_changes_nothing(eg_module):
    eh = TinyEngramHost(eg_module, n_slots=2, max_seq_len=512)
    ref = TinyEngramHost(eg_module, n_slots=2, max_seq_len=512)
    pf = engram_prefetch.EngramPrefetch(eh, chunk_size=64)
    rng = np.random.default_rng(3)
    a_ids = _prompt(rng, 128, eh.vocab_size)
    other = _prompt(rng, 64, eh.vocab_size)
    for host in (pf, ref):
        if host is pf:
            list(pf.rows_async(0, a_ids[:64], 0).values())
        else:
            ref.hash_ids(0, a_ids[:64], 0)
    for wrong in ((0, other, 64), (1, a_ids[64:], 64), (0, a_ids[64:], 0)):
        slot1 = eh._hash[1].cache.clone()
        pf.prefetch(*wrong)
        assert torch.equal(eh._hash[1].cache, slot1), "a prefetch wrote another slot's history"
        rows = {k: f.result() for k, f in pf.rows_async(0, a_ids[64:], 64).items()}
        h = ref.hash_ids(0, a_ids[64:], 64)
        for i, lid in enumerate(ref.layer_ids):
            want = ref._gather(lid, h[:, i, :].numpy().astype(np.int64).reshape(-1)).reshape(64, ref.row_elems)
            assert torch.equal(rows[lid], want)
        assert torch.equal(eh._hash[0].cache[:, :128], ref._hash[0].cache[:, :128])
    assert pf.stats["hits"] == 0 and pf.stats["misses"] == 3


# ---- the runtime's wiring (fake ttnn: fake_v41.py) -------------------------------------------------------------------
def test_runtime_matches_the_chunk_and_prefetches_the_next():
    import sys

    sys.path.insert(0, HERE)
    sys.path.insert(0, ROOT)
    import fake_v41  # noqa: F401 -- installs the fake ttnn before runtime.py imports it

    from models.demos.common.prefill.prefix_reuse import split_slot_word
    from models.demos.deepseek_v3_d_p.tt.v41.runtime import V41PrefillRuntime

    path, w, r = _ring(n=8, chunk=16)
    marked, last, dec = 1 << 29, 1 << 30, 1 << 28
    a = list(range(100, 116))
    b = list(range(200, 210))
    w.write(marked | 3, 0, 16, a)
    w.write(marked | last | dec | 5, 32, 42, b + [1] * 6)  # another slot, a ragged decoder chunk

    class Rec:
        stats = {"hits": 0, "misses": 0}

        def __init__(self):
            self.calls = []

        def prefetch(self, slot, ids, start):
            self.calls.append((slot, list(ids), start))

    rt = V41PrefillRuntime.__new__(V41PrefillRuntime)
    rt._hostids, rt._eprefetch, rt.num_users, rt.chunk_size = r, Rec(), 8, 16
    rt._hi = {"matched": 0, "unmatched": 0, "ids_differ": 0, "next_missing": 0}
    cur = rt._host_ids_match(marked | 3, 0, 16, a)
    assert cur == 0 and rt._hi["matched"] == 1 and rt._hi["ids_differ"] == 0
    rt._prefetch_next(cur)
    assert rt._eprefetch.calls == [(split_slot_word(marked | last | dec | 5)[0], b, 32)]
    assert rt._eprefetch.calls[0][0] == 5
    assert rt._host_ids_match(marked | 3, 0, 16, a) is None and rt._hi["unmatched"] == 1  # consumed: not matched twice
    rt._prefetch_next(1)  # nothing pushed after it yet
    assert rt._hi["next_missing"] == 1 and len(rt._eprefetch.calls) == 1
    cur2 = rt._host_ids_match(
        marked | last | dec | 5, 32, 42, [9] * 10
    )  # the device read other ids: counted, never trusted
    assert cur2 == 1 and rt._hi["ids_differ"] == 1
