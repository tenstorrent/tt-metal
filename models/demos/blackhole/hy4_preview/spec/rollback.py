# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-user KV / indexer-key / MTP-KV length bookkeeping of a speculative round, and the rollback rule (pure python, no ttnn).

Hy4 KV is additive: a decode step APPENDS one latent row (576 values) per layer at position p and one 128-wide key per full-indexer layer; nothing is
compressed, no ring, no halo, no compressor state (contrast DSV4: ring rows, prev_cs). Therefore a rejected speculative write needs NO restore:
  rollback == "the committed length does not advance over the rejected rows". Stale rows beyond the length are never read (every sparse row sees only keys
  <= its own position: the indexer's candidate range and the top-k validity count are per query row), and they are overwritten in place by the next round
  because the next round's rows start exactly at the committed length. Pages: the pages that hold the stale rows stay allocated (they are needed again within
  a round or two); only ``length`` moves. The page pool never has to un-allocate on rejection; it must allocate AHEAD (see ``pages_needed``).

Positions in a round (P = committed length = position of the pending token x_P):
  main verify rows       positions P .. P+k          (k+1 rows; row j sees keys <= P+j)         writes main KV + index keys of 21 full layers at those positions
  MTP step 0 rows        positions P .. P+k          (same rows; the pair (h_row, next token))   writes MTP KV (layer 78) + MTP index keys
  MTP draft step j=1..k-1 position (P+m)+j           (one row per user; transient)              writes MTP KV + keys beyond the new committed length
Commit with m accepted: main length = MTP length = P+m+1. MTP rows P+m+1.. (draft steps) and main rows P+m+1..P+k are stale.
"""
from dataclasses import dataclass

PAGE = 512


def pages_for(n_tokens: int, page: int = PAGE) -> int:
    return -(-n_tokens // page)


@dataclass
class UserSpecState:
    """Lengths of one user. ``length`` is the single committed length shared by the main KV, its 21 index-key layers, the MTP KV and the MTP keys (all stores
    are written at the same positions by construction; ``mtp_len`` is tracked separately only so the tests can assert that the invariant mtp_len == length holds
    after every commit, and ``mtp_len < length`` is the legitimate state between prefill and the seed step)."""

    length: int = 0
    mtp_len: int = 0
    pages: int = 0  # pages allocated to this user (high-water: never shrinks during a request)
    in_round: bool = False
    _P: int = 0
    _k: int = 0
    max_len: int = 1 << 20  # pool contract (context cap)

    def pages_needed(self, k: int) -> int:
        """Pages that must exist BEFORE a round of k drafts starts: the verify rows write positions length .. length+k (k+1 rows)."""
        return pages_for(self.length + k + 1)

    def begin_round(self, k: int) -> int:
        """Start a round: returns the number of extra pages the allocator must hand out before the trace runs (0 usually; 1 at a page boundary)."""
        assert not self.in_round, "round already open"
        assert (
            self.length + k + 1 <= self.max_len
        ), "round would write past the context cap: caller must clamp k (see clamp_k)"
        self.in_round, self._P, self._k = True, self.length, k
        extra = max(0, self.pages_needed(k) - self.pages)
        self.pages += extra
        return extra

    def row_nvalid(self, j: int) -> int:
        """Keys visible to verify row j (0..k): positions 0..P+j (including its own, written before the attention). The indexer / sparse kernel validity count."""
        assert self.in_round and 0 <= j <= self._k
        return self._P + j + 1

    def commit(self, m: int) -> int:
        """Accept m drafts (0..k): the committed length becomes P+m+1; everything written at positions >= that is stale (= rolled back). Returns new length."""
        assert self.in_round and 0 <= m <= self._k, (m, self._k)
        self.length = self._P + m + 1
        self.mtp_len = self.length
        self.in_round = False
        return self.length

    def stale_rows(self, m: int) -> range:
        """Positions written by the round but not committed (for tests / debugging)."""
        return range(self._P + m + 1, self._P + self._k + 1)

    def clamp_k(self, k: int, tokens_left: int) -> int:
        """Largest k <= requested such that the round fits the context cap and emits no more than ``tokens_left`` tokens (a round emits up to k+1)."""
        return max(0, min(k, self.max_len - self.length - 1, tokens_left - 1))

    def seed_after_prefill(self, prompt_len: int, prefill_mtp_rows: int) -> None:
        """After prefill of L tokens: main length L. The MTP prefill filled rows 0..L-2 (row p pairs h_p with token p+1); row L-1 needs the first generated
        token, so it is written by the SEED step (MTP step 0 on one row with (h_{L-1}, x_L)), after which mtp_len == L (the pending token x_L is position L).
        """
        assert prefill_mtp_rows == prompt_len - 1
        self.length = prompt_len
        self.mtp_len = prefill_mtp_rows
        self.pages = max(self.pages, pages_for(prompt_len))

    def seed_done(self) -> None:
        assert self.mtp_len == self.length - 1
        self.mtp_len = self.length


class SimStore:
    """Append-only row store used by the tests as the stand-in of one (layer) KV / key pool: a dict position -> value plus the committed length.
    Reads are only allowed below ``length_of_reader`` (the causal rule); writes can go anywhere below capacity (stale rows allowed).
    """

    def __init__(self):
        self.rows = {}

    def write(self, pos, val):
        self.rows[pos] = val

    def read_prefix(self, n):
        return [self.rows[i] for i in range(n)]
