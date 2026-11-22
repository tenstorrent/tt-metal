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


def block_keys(tokens, block=BLOCK):
    """the hash chain keys of the prompt's whole blocks: keys[i] covers tokens[: (i + 1) * block]"""
    keys, prev = [], b""
    for i in range(len(tokens) // block):
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
        self._entries = OrderedDict()  # key -> (h, value, nbytes)
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
        h, value, _ = self._entries[best]
        self.stats["hits"] += 1
        self.stats["hit_tokens"] += h
        return h, value

    def insert(self, tokens, h, value, nbytes):
        assert h > 0 and h % self.block == 0 and h <= len(tokens), h
        key = block_keys(tokens[:h], self.block)[-1]
        if key in self._entries:
            self.bytes -= self._entries.pop(key)[2]
        if nbytes > self.budget:
            return False
        self._entries[key] = (h, value, int(nbytes))
        self.bytes += int(nbytes)
        self.stats["inserts"] += 1
        while self.bytes > self.budget:
            _, (_, _, nb) = self._entries.popitem(last=False)
            self.bytes -= nb
            self.stats["evictions"] += 1
        return True

    def boundaries(self, n_tokens):
        """the prefix lengths worth snapshotting for a prompt of n_tokens: its last whole block (the prompt's end
        rounded down), the point a re-sent prompt or the next turn of the conversation hits"""
        h = (n_tokens // self.block) * self.block
        return [h] if h else []
