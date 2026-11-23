# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of the host-tier prefix cache index (qwen36_resident/prefix_cache.py)."""
import random

from qwen36_resident.prefix_cache import PrefixCache, block_keys


def toks(n, seed):
    r = random.Random(seed)
    return [r.randrange(150000) for _ in range(n)]


def test_block_keys_chain():
    a = toks(3000, 1)
    ka, kb = block_keys(a, 1024), block_keys(a[:2048] + toks(952, 2), 1024)
    assert len(ka) == 2 and ka[0] == kb[0] and ka[1] == kb[1]  # (same first 2048 tokens)
    c = list(a)
    c[5] += 1
    assert block_keys(c, 1024)[0] != ka[0] and block_keys(c, 1024)[1] != ka[1]  # (a change propagates)


def test_lookup_longest_prefix_and_one_token_left():
    pc = PrefixCache(10**9)
    prompt = toks(5000, 3)
    pc.insert(prompt, 2048, "s2048", 100)
    pc.insert(prompt, 4096, "s4096", 100)
    assert pc.lookup(prompt + toks(300, 4)) == (4096, "s4096")  # (turn 2: the longest)
    assert pc.lookup(prompt[:4096]) == (2048, "s2048")  # (a hit must leave a token to prefill)
    assert pc.lookup(prompt[:4097]) == (4096, "s4096")
    assert pc.lookup(toks(5000, 5)) == (0, None)
    other = prompt[:3000] + toks(2000, 6)
    assert pc.lookup(other) == (2048, "s2048")  # (a shared prefix: the block before the divergence)


def test_lru_budget():
    pc = PrefixCache(250)
    p = [toks(2100, 10 + i) for i in range(3)]
    for i in range(3):
        assert pc.insert(p[i], 2048, i, 100)
    assert len(pc) == 2 and pc.lookup(p[0] + [1])[0] == 0  # (the oldest evicted)
    pc.lookup(p[1] + [1])  # (refreshes p[1])
    pc.insert(toks(2100, 20), 2048, 9, 100)
    assert pc.lookup(p[1] + [1])[1] == 1 and pc.lookup(p[2] + [1])[0] == 0
    assert not pc.insert(toks(2100, 21), 2048, 0, 300)  # (larger than the budget: not kept)
    assert pc.bytes <= pc.budget


def test_boundaries():
    pc = PrefixCache(1)
    assert pc.boundaries(1000) == [] and pc.boundaries(2600) == [2048] and pc.boundaries(4096) == [4096]


def test_peek_matches_lookup_without_side_effects():
    pc = PrefixCache(1e9, block=1024)
    a = toks(5000, 7)
    pc.insert(a, 4096, "snap", 10)
    stats = dict(pc.stats)
    order = list(pc._entries)
    assert pc.peek(a) == 4096 and pc.peek(toks(5000, 8)) == 0
    assert dict(pc.stats) == stats and list(pc._entries) == order  # (no count, no LRU touch)
    assert pc.lookup(a)[0] == pc.peek(a)
