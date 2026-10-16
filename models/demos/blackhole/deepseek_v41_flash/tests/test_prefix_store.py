# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-free test of tt/prefix_cache.PrefixStore (lookup rule, LRU budget).

    pytest --noconftest models/demos/blackhole/deepseek_v41_flash/tests/test_prefix_store.py
"""

import torch

from models.demos.blackhole.deepseek_v41_flash.tt.prefix_cache import PrefixSnapshot, PrefixStore


def snap(tokens, s, nbytes=1000):
    return PrefixSnapshot(s=s, toks=tokens[:s].clone(), pool={0: torch.zeros(nbytes // 2, dtype=torch.bfloat16)})


def test_lookup_takes_the_largest_boundary_that_leaves_the_last_128_positions():
    t = torch.arange(5000)
    st = PrefixStore(512, 10**6)
    for s in (512, 1024, 1536):
        st.put(snap(t, s))
    assert st.lookup(t, 4000, 4000).s == 1536
    assert (
        st.lookup(t, 4000, 1600).s == 1472 // 512 * 512 == 1024
    )  # end - 128 = 1472: 1024 is the largest boundary below it
    assert st.lookup(t, 1100, 4000).s == 1024  # a hit longer than vLLM's num_computed_tokens is not used
    assert st.lookup(t, 4000, 600) is None  # 600 - 128 = 472 < 512
    t2 = t.clone()
    t2[700] += 1  # diverges inside the second chunk: only the first boundary matches
    assert st.lookup(t2, 4000, 4000).s == 512
    assert (st.hits, st.misses) == (4, 1)


def test_signature_separates_configurations():
    t = torch.arange(3000)
    a, b = PrefixStore(512, 10**6, "cfgA"), PrefixStore(512, 10**6, "cfgB")
    a.put(snap(t, 1024))
    assert a.lookup(t, 2000, 2000) is not None and b.lookup(t, 2000, 2000) is None


def test_lru_budget_evicts_the_least_recently_used():
    t = torch.arange(5000)
    st = PrefixStore(512, 2500)
    st.put(snap(t, 512))
    st.put(snap(t, 1024))
    assert st.lookup(t, 700, 4000).s == 512  # touch 512
    st.put(snap(t, 1536))  # 3000 > 2500: 1024 (older than the touched 512) goes
    assert sorted(k[1] for k in st.snaps) == [512, 1536] and st.evictions == 1
    assert not st.put(snap(t, 2048, nbytes=10**5))  # larger than the whole budget
    st.clear()
    assert st.bytes == 0 and not st.snaps
