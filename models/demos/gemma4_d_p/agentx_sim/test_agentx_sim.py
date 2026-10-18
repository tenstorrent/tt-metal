# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the AgentX simulator: the prefix cache, the cache state a request sent mid-step sees, and the
--config validation. No device and no traces: `pytest models/demos/gemma4_d_p/agentx_sim/test_agentx_sim.py`."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentx_sim import BLOCK, PrefixCache, parse_config, run  # noqa: E402
from cost_model import Model  # noqa: E402


def test_prefix_cache_longest_prefix():
    cache = PrefixCache()
    cache.insert(0, [1, 2, 3])
    assert cache.lookup(0, [1, 2, 3, 4]) == 3 * BLOCK
    assert cache.lookup(0, [1, 9]) == BLOCK
    assert cache.lookup(1, [1, 2, 3]) == 0, "sessions do not share blocks"


def test_prefix_cache_lru_evicts_tails_first():
    cache = PrefixCache(3 * BLOCK)
    cache.insert(0, [1, 2, 3])
    cache.insert(1, [7])  # one block over capacity: session 0's tail goes, its prefix stays
    assert cache.lookup(0, [1, 2, 3]) == 2 * BLOCK
    assert cache.lookup(1, [7]) == BLOCK


def test_prefix_cache_lookup_refreshes_recency():
    cache = PrefixCache(2 * BLOCK)
    cache.insert(0, [1])
    cache.insert(1, [2])
    assert cache.lookup(0, [1]) == BLOCK  # A was just read, so B is now the least recent
    cache.insert(2, [3])
    assert cache.lookup(0, [1]) == BLOCK
    assert cache.lookup(1, [2]) == 0


class _OneSecondSteps:
    def step_s(self, batch, C, budget, calibrated=True):
        return 1.0, 0.0


def _node(n_in, hashes, deps=(), t=0.0, out=0):
    return {"in": n_in, "out": out, "t": t, "end": 0.0, "hashes": hashes, "deps": list(deps)}


def test_request_sent_mid_step_misses_that_steps_completions():
    """Steps take 1 s, one 64-token request per step. A finishes in step 1, B (blocks [1]) in step 2. C depends on A
    and is sent 0.5 s into step 2, so it must look the cache up before B's blocks are inserted."""
    session = [
        _node(BLOCK, [10]),  # A
        _node(BLOCK, [1]),  # B
        _node(2 * BLOCK, [1, 5], deps=[0], t=0.5),  # C: A's recorded end is 0, so it thinks for 0.5 s
    ]
    res = run(_OneSecondSteps(), [session], BLOCK, BLOCK, "fcfs", 1)
    c = next(x for x in res["_stats"] if x.n_in == 2 * BLOCK)
    assert c.cached == 0, "C saw B's blocks, which were inserted after C was sent"


@pytest.mark.parametrize("cfg", ["2048x8192:pack", "8192x8192:fcfs", "16384x16384:fcfs", "1024x4096:fcfs"])
def test_parse_config_accepts_measured_units(cfg):
    parse_config(cfg, Model.load())


@pytest.mark.parametrize(
    "cfg",
    [
        "3072x3072:fcfs",  # unit without measured data
        "2048x3000:pack",  # budget not a multiple of the unit
        "2048x32768:pack",  # past the largest measured step
        "2048x8192:packed",  # unknown policy
        "2048:fcfs",  # no budget
    ],
)
def test_parse_config_rejects(cfg):
    with pytest.raises(SystemExit, match="bad --config"):
        parse_config(cfg, Model.load())


def test_next_session_starts_when_the_first_response_ends():
    """Two of three sessions run at once, one request each; session 1 decodes for 100 s, session 0 not at all. The
    third session takes the slot whose response ends first, whichever prefill finished first."""
    sessions = [[_node(BLOCK, [1])], [_node(BLOCK - 1, [2], out=10_000)], [_node(2 * BLOCK, [3])]]
    res = run(_OneSecondSteps(), sessions, BLOCK, BLOCK, "fcfs", 2, decode_tps=100.0)
    by_len = {x.n_in: x for x in res["_stats"]}
    first_response_end = min(by_len[BLOCK].done, by_len[BLOCK - 1].done + 100.0)
    assert by_len[2 * BLOCK].send == pytest.approx(first_response_end)
