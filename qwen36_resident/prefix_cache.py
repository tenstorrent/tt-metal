# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-tier prefix cache (Strata S1, option C — pending the user's pick): snapshots of a decode slot's state at
prefix lengths that are multiples of BLOCK tokens, kept in host memory under an LRU byte budget.

An entry's key is the hash chain of the prefix's BLOCK-token blocks (block i's key hashes block i - 1's key and its
tokens, as vLLM's prefix caching does), so a lookup walks the prompt's blocks and stops at the first miss. The value
is whatever the engine stores for that prefix — ResidentModel.snapshot_to_host(snapshot_slot(s, n_tokens=h)) — and
its size in bytes. A hit h restores the snapshot into the request's slot (restore_slot), then PPPrefill.load_prefix /
run_from prefill only the rest (h is a multiple of BLOCK = 1,024: every PP chunk size divides it).
"""
import hashlib
from collections import OrderedDict

BLOCK = 1024


def block_keys(tokens, block=BLOCK, start=0, prev=b""):
    """the hash chain keys of the prompt's whole blocks: keys[i] covers tokens[: (start + i + 1) * block]; start /
    prev continue a chain (prev = the key of tokens[: start * block])"""
    keys = []
    for i in range(start, len(tokens) // block):
        h = hashlib.sha256(prev)
        h.update(b",".join(str(int(t)).encode() for t in tokens[i * block : (i + 1) * block]))
        prev = h.digest()
        keys.append(prev)
    return keys


class PrefixCache:
    """LRU over prefix snapshots: insert(tokens, h, value, nbytes) for a prefix of h tokens (h % block == 0),
    lookup(tokens) -> (h, value) of the longest cached prefix that leaves at least one token to prefill (h <
    len(tokens)), or (0, None). Entries past the byte budget are evicted least recently used first (a hit or an insert
    refreshes an entry); an entry larger than the whole budget is not kept."""

    def __init__(self, budget_bytes, block=BLOCK):
        self.budget, self.block = int(budget_bytes), block
        self._entries = OrderedDict()  # key -> (h, value, nbytes, its chain keys)
        self._chains = {}  # every stored entry's chain keys -> the entries through them (split prompts' lookups)
        self.bytes = 0
        self.stats = dict(lookups=0, hits=0, hit_tokens=0, inserts=0, evictions=0)

    def __len__(self):
        return len(self._entries)

    def lookup(self, tokens):
        self.stats["lookups"] += 1
        keys = block_keys(tokens[: len(tokens) - 1], self.block)  # (a hit leaves at least one token to prefill)
        best = None
        for k in keys:
            if k not in self._entries:
                continue  # (a shorter entry may be missing while a longer one is kept: keep walking)
            best = k
        if best is None:
            return 0, None
        self._entries.move_to_end(best)
        h, value = self._entries[best][:2]
        self.stats["hits"] += 1
        self.stats["hit_tokens"] += h
        return h, value

    def peek(self, tokens):
        """the prefix lookup() would restore (0: none), without counting it or touching the LRU order"""
        keys = block_keys(tokens[: len(tokens) - 1], self.block)
        return max((self._entries[k][0] for k in keys if k in self._entries), default=0)

    def insert(self, tokens, h, value, nbytes):
        assert h > 0 and h % self.block == 0 and h <= len(tokens), h
        chain = block_keys(tokens[:h], self.block)
        key = chain[-1]
        if key in self._entries:
            self._drop(key)
        if nbytes > self.budget:
            return False
        self._entries[key] = (h, value, int(nbytes), chain)
        for k in chain:
            self._chains[k] = self._chains.get(k, 0) + 1
        self.bytes += int(nbytes)
        self.stats["inserts"] += 1
        while self.bytes > self.budget:
            self._drop(next(iter(self._entries)))
            self.stats["evictions"] += 1
        return True

    def _drop(self, key):
        _, _, nb, chain = self._entries.pop(key)
        self.bytes -= nb
        for k in chain:
            self._chains[k] -= 1
            if not self._chains[k]:
                del self._chains[k]

    def extends(self, chain_key):
        """whether a stored entry's prefix runs through the prefix whose chain key this is (a split prompt's span:
        keep deferring while it does)"""
        return chain_key in self._chains

    def boundaries(self, n_tokens):
        """the prefix lengths worth snapshotting for a prompt of n_tokens: its last whole block (the prompt's end
        rounded down), the point a re-sent prompt or the next turn of the conversation hits"""
        h = (n_tokens // self.block) * self.block
        return [h] if h else []


class SwapStore:
    """Strata S3: preempted users' decode states in host memory, LRU under a byte budget. vLLM's preemption frees a
    running request's blocks and later resumes it by prefilling prompt + output again (recompute); a model that kept
    the state restores it instead. An entry: the tokens the state consumed (positions < q), the next token (position
    q: the device's draw, or the host's when it saw it), the host snapshot. A resumed prompt replays either exactly the
    q tokens (its next token is the entry's) or the q tokens and the next one (one decode step from the state); any
    other prompt is not a match (a plain prefill)."""

    def __init__(self, budget_bytes):
        self.budget = int(budget_bytes)
        self._entries = OrderedDict()  # key -> (tokens, next token, value, nbytes)
        self.bytes = 0
        self.stats = dict(inserts=0, hits=0, hit_tokens=0, evictions=0, refused=0)

    def __len__(self):
        return len(self._entries)

    @staticmethod
    def _key(tokens):
        return hashlib.sha256(b",".join(str(int(t)).encode() for t in tokens)).digest()

    def insert(self, tokens, nxt, value, nbytes):
        tokens = tuple(int(t) for t in tokens)
        key = self._key(tokens)
        if key in self._entries:
            self._drop(key)
        if nbytes > self.budget:
            self.stats["refused"] += 1
            return False
        self._entries[key] = (tokens, int(nxt), value, int(nbytes))
        self.bytes += int(nbytes)
        self.stats["inserts"] += 1
        while self.bytes > self.budget:
            self._drop(next(iter(self._entries)))
            self.stats["evictions"] += 1
        return True

    def _drop(self, key):
        self.bytes -= self._entries.pop(key)[3]

    def candidate(self, prefix, total):
        """whether a resumed prompt of `total` tokens whose first tokens are `prefix` (a split prompt's first span) may
        replay an entry (decided at its last span by match())"""
        prefix = tuple(int(t) for t in prefix)
        n = len(prefix)
        return any(
            len(tok) in (total, total - 1) and n <= len(tok) and tok[:n] == prefix
            for tok, _, _, _ in self._entries.values()
        )

    def match(self, ids):
        """the entry the prompt `ids` replays, as (key, q, next token), or None (not removed: take() does)"""
        ids = tuple(int(t) for t in ids)
        for q in (len(ids), len(ids) - 1):
            if q <= 0:
                continue
            key = self._key(ids[:q])
            e = self._entries.get(key)
            if e is not None and e[0] == ids[:q] and (q == len(ids) or ids[q] == e[1]):
                return key, q, e[1]
        return None

    def take(self, key):
        """remove and return an entry's host snapshot (its state goes back into a slot once)"""
        tokens, _, value, nb = self._entries.pop(key)
        self.bytes -= nb
        self.stats["hits"] += 1
        self.stats["hit_tokens"] += len(tokens)
        return value
