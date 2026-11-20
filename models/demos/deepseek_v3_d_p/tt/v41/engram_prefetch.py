# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""V41_PREFILL_HOST_IDS (tt-blaze DS41F-0163 WS13): gather the NEXT chunk's Engram rows while this chunk runs.

A chunk's Engram rows (layers 1 and 14) come from remote row servers; layer 1 needs them right after the embedding, so today
the device idles at every chunk start while the host waits for them (``engram.wait`` 41-43 ms a chunk on the final prefill
set, WS7 host profile). With the next chunk's ids known host-side (``host_ids.HostIdsReader``), its rows can be gathered
during this chunk's ~0.3 s of device time.

Exactness, by construction: ``EngramPrefetch.rows_async`` (what the islands path calls) ALWAYS computes the chunk's real hash
with ``EngramHost.hash_ids`` -- the same call, in the same order, as ``EngramHost.rows_async`` -- and hands back the early
gather only when its table rows (per layer) and token ids equal the real ones. The early hash never touches the slot's n-gram
history: it runs on a scratch ``NgramHashState`` seeded with a copy of the history window it reads
(``max_ngram_size - 1`` positions before the chunk), so a wrong guess about the next chunk can never change a later hash.
``_submit`` is ``EngramHost.rows_async``'s gather loop (identical in tt-blaze's plain and split engram_host builds).

V41_PREFILL_HOST_IDS_VERIFY=1 also gathers every hit again in the background and compares the rows off the critical path (next
call); a difference is logged and counted in ``verify_fail`` (testing only; the extra gathers load the row servers).
"""

from __future__ import annotations

import copy
import os

import numpy as np
import torch
from loguru import logger

VERIFY = os.environ.get("V41_PREFILL_HOST_IDS_VERIFY", "0") == "1"
_NEED = ("hash_ids", "_hash", "_gather", "_dead", "_layer_pool", "row_elems", "layer_ids", "layout", "vocab_size")


class EngramPrefetch:
    def __init__(self, engram_host, chunk_size: int):
        missing = [a for a in _NEED if not hasattr(engram_host, a)]
        if missing:
            raise RuntimeError(f"EngramPrefetch: this EngramHost lacks {missing}")
        self._e = engram_host
        self._n = int(engram_host.layout.max_ngram_size)
        scratch = copy.deepcopy(engram_host._hash[0])
        # a chunk plus its look-back window: positions [0, n - 1 + chunk)
        scratch.register_buffer(
            "cache", torch.zeros(1, self._n + int(chunk_size) + 8, dtype=torch.int64), persistent=False
        )
        self._scratch = scratch
        self._pending = None
        self.stats = {"prefetched": 0, "hits": 0, "misses": 0, "unused": 0, "verified": 0, "verify_fail": 0}
        self._verify = []  # (prefetched futures, reference futures) awaiting comparison

    def __getattr__(self, name):
        return getattr(self._e, name)

    # ---- early side ---------------------------------------------------------------------------------------------------
    def _scratch_hash(self, slot: int, token_ids, start_pos: int) -> torch.Tensor:
        """``EngramHost.hash_ids(slot, token_ids, start_pos)`` without writing the slot's history: the n-grams of positions
        [start_pos, start_pos + L) read the history only at [start_pos - (n - 1), start_pos), copied here to the scratch
        cache's front; with s0 = min(start_pos, n - 1) every look-back index and every ``positions < shift`` test maps 1:1.
        """
        e = self._e
        s0 = min(int(start_pos), self._n - 1)
        hist = e._hash[slot].cache
        st = self._scratch
        if s0:
            st.cache[0, :s0] = hist[0, int(start_pos) - s0 : int(start_pos)]
        ids = torch.as_tensor(token_ids, dtype=torch.long).reshape(1, -1)
        live = ids < e.vocab_size
        if bool(live.all()):
            return st(ids, s0)[0]
        return st(torch.where(live, ids, 0), s0, live)[0]

    def prefetch(self, slot: int, token_ids, start_pos: int) -> None:
        """Start gathering the rows of a chunk that has not arrived yet. Replaces an unused earlier prefetch (its gathers that
        have not started are cancelled: a run of skipped chunks must not queue gathers ahead of the next computed chunk's).
        """
        if self._pending is not None:
            self.stats["unused"] += 1
            for f in self._pending["futures"].values():
                f.cancel()
        ids = [int(v) for v in token_ids]
        h = self._scratch_hash(int(slot), ids, int(start_pos))
        idx = {lid: h[:, i, :].numpy().astype(np.int64) for i, lid in enumerate(self._e.layer_ids)}
        self._pending = {
            "key": (int(slot), int(start_pos), len(ids)),
            "ids": np.asarray(ids, dtype=np.int64),
            "idx": idx,
            "futures": self._submit(idx, ids),
        }
        self.stats["prefetched"] += 1

    # ---- the chunk's own call -----------------------------------------------------------------------------------------
    def _submit(self, idx: dict, token_ids, layers=None) -> dict:
        """``EngramHost.rows_async``'s gather loop for already-hashed rows: {layer_id: Future of [L, 6144] bf16}."""
        e = self._e
        out = {}
        for lid in e.layer_ids:
            if layers is not None and lid not in layers:
                continue
            ix = idx[lid]
            dead = e._dead(token_ids)

            def _rows(lid=lid, ix=ix, dead=dead):
                r = e._gather(lid, ix.reshape(-1)).reshape(ix.shape[0], e.row_elems)
                if dead is not None:
                    r[dead] = 0
                return r

            out[lid] = e._layer_pool.submit(_rows)
        return out

    def rows_async(self, slot: int, token_ids, start_pos: int, layers=None) -> dict:
        """``EngramHost.rows_async``: the real hash first (it advances the slot's history, as always), then the early gather if
        it gathered exactly these rows, else a gather now."""
        e = self._e
        self._check_verify()
        p, self._pending = self._pending, None
        h = e.hash_ids(slot, token_ids, start_pos)  # [L, n_layers, 24]
        idx = {lid: h[:, i, :].numpy().astype(np.int64) for i, lid in enumerate(e.layer_ids)}
        if (
            p is not None
            and layers is None
            and p["key"] == (int(slot), int(start_pos), len(token_ids))
            and np.array_equal(p["ids"], np.asarray([int(v) for v in token_ids], dtype=np.int64))
            and all(np.array_equal(p["idx"][lid], idx[lid]) for lid in e.layer_ids)
        ):
            self.stats["hits"] += 1
            if VERIFY:
                self._verify.append((p["futures"], self._submit(idx, token_ids)))
            return p["futures"]
        if p is not None:
            self.stats["misses"] += 1
        return self._submit(idx, token_ids, layers)

    def rows_for(self, slot: int, token_ids, start_pos: int, layers=None) -> dict:
        """Synchronous callers (the non-island path, prefix-reuse replays) never use a prefetch; drop a pending one only if
        it is for this chunk (its gather is still valid for a later chunk otherwise)."""
        p = self._pending
        if p is not None and p["key"][0] == int(slot) and p["key"][1] == int(start_pos):
            self._pending = None
            self.stats["unused"] += 1
        return self._e.rows_for(slot, token_ids, start_pos, layers=layers)

    def _check_verify(self, wait: bool = False) -> None:
        keep = []
        for got, ref in self._verify:
            if not wait and not all(f.done() for f in list(got.values()) + list(ref.values())):
                keep.append((got, ref))
                continue
            ok = all(torch.equal(got[lid].result(), ref[lid].result()) for lid in got)
            self.stats["verified"] += 1
            if not ok:
                self.stats["verify_fail"] += 1
                logger.error(
                    "[v41 runtime] V41_PREFILL_HOST_IDS_VERIFY: prefetched Engram rows differ from a fresh gather"
                )
        self._verify = keep

    def summary(self) -> str:
        s = self.stats
        return (
            f"engram prefetch: {s['prefetched']} started, {s['hits']} used, {s['misses']} not matching, {s['unused']} unused"
            + (f", {s['verified']} verified, {s['verify_fail']} differ" if VERIFY else "")
        )


def log_summary_every(pf: EngramPrefetch, n_chunks: int, every: int = 64) -> None:
    if n_chunks and n_chunks % every == 0:
        logger.info(f"[v41 runtime] {pf.summary()}")
