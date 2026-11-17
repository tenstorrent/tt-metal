# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the AgentX simulator: the prefix cache, the cache state a request sent mid-step sees, the
--config validation and the --ttft-out records. No device and no traces:
`pytest models/demos/gemma4_d_p/agentx_sim/test_agentx_sim.py`."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentx_sim import BLOCK, PrefixCache, json_record, parse_config, parse_sessions, run, ttft_record  # noqa: E402
from cost_model import Model  # noqa: E402
from plot_ttft_cdf import labels, load_runs  # noqa: E402


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


@pytest.mark.parametrize(
    "align, start, busy",
    [("chunk", 0, 2.0), ("resume", 5 * BLOCK, 2.5), ("exact", 5 * BLOCK, 2.5)],
)
def test_align_rounds_the_resume_and_mid_chunk_steps_cost_more(align, start, busy):
    """B resumes on A's 5 cached blocks inside one 8-block unit: chunk alignment rounds the start down to 0; resume
    (a slab, unit / 8 = one block) and exact keep it, and then B's step starts mid-unit and costs 1 + mid_cost."""
    session = [_node(5 * BLOCK, [1, 2, 3, 4, 5]), _node(6 * BLOCK, [1, 2, 3, 4, 5, 6], deps=[0])]
    res = run(_OneSecondSteps(), [session], 8 * BLOCK, 8 * BLOCK, "fcfs", 1, align=align, mid_cost=0.5)
    assert next(x for x in res["_stats"] if x.n_in == 6 * BLOCK).start == start
    assert res["busy_s"] == pytest.approx(busy)


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
    # The run ends at the last step, not at the last response or past it.
    assert res["makespan_s"] == pytest.approx(by_len[2 * BLOCK].done)
    assert res["tok_s_wall"] > 0


def test_main_request_joins_only_sub_agents_that_ended_before_it(tmp_path):
    """A sub-agent launched after main 0 ends at 2.5 s: main 1, sent at 2.0 s, must not wait for it; main 2 does."""

    def req(t, kind="main"):
        return {"type": kind, "t": t, "in": BLOCK, "out": 0, "hash_ids": [1], "api_time": 0.0}

    sub = {"type": "subagent", "t": 1.0, "duration_ms": 1500, "requests": [req(1.0, "subagent")]}
    trace = {"block_size": BLOCK, "hash_id_scope": "local", "requests": [req(0.0), sub, req(2.0), req(5.0)]}
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(trace) + "\n")
    nodes = parse_sessions(str(path))[0]
    mains = [i for i, n in enumerate(nodes) if n["t"] in (0.0, 2.0, 5.0)]
    sub_node = next(i for i, n in enumerate(nodes) if n["t"] == 1.0)
    assert nodes[mains[1]]["deps"] == [mains[0]]
    assert sorted(nodes[mains[2]]["deps"]) == sorted([mains[1], sub_node])


def test_resume_rewinds_so_the_last_padded_chunk_fits_the_slot():
    """B has 15 of its 16 blocks cached in a 16-block slot: resuming at block 15 would pad its 8-block chunk past the
    slot, which the runtime rejects, so it resumes at block 8 and its last chunk ends at the slot's end."""
    session = [_node(15 * BLOCK, list(range(1, 16))), _node(16 * BLOCK, list(range(1, 17)), deps=[0])]
    res = run(_OneSecondSteps(), [session], 8 * BLOCK, 8 * BLOCK, "fcfs", 1, align="resume", max_seq_len=16 * BLOCK)
    b = next(x for x in res["_stats"] if x.n_in == 16 * BLOCK)
    assert b.start == 8 * BLOCK and b.start + b.rows <= 16 * BLOCK


def test_ttft_exports_from_before_mid_cost_plot_with_new_ones(tmp_path):
    """An export without mid_cost loads as no cost, so it plots beside a run that has one."""
    res = run(_OneSecondSteps(), [[_node(BLOCK, [1])]], BLOCK, BLOCK, "fcfs", 1)
    knobs = dict(config="64x64:fcfs", concurrency=32, cache="inf", align="chunk", max_think_s=60.0)
    path = tmp_path / "ttft.jsonl"
    path.write_text(
        json.dumps(ttft_record(knobs, res["_stats"]))
        + "\n"
        + json.dumps(ttft_record({**knobs, "align": "resume", "mid_cost": 0.045}, res["_stats"]))
        + "\n"
    )
    runs = load_runs([str(path)], [])
    assert [r["mid_cost"] for r in runs] == [0.0, 0.045]
    assert labels(runs)[0][1].endswith("mid-chunk cost +4.5%")


def test_json_record_is_strict_json_for_empty_buckets():
    """One request leaves the other size buckets empty; their NaN percentiles must export as null."""
    res = run(_OneSecondSteps(), [[_node(BLOCK, [1])]], BLOCK, BLOCK, "fcfs", 1)
    res.pop("_stats")
    record = json.loads(json.dumps(json_record(res), allow_nan=False))
    assert record["p50_>16K"] is None
    assert record["ttft_p50"] == res["ttft_p50"]


class _RecordedSteps(_OneSecondSteps):
    """One-second steps that record each step's lanes and check it fits the step's token budget."""

    def __init__(self):
        self.steps = []

    def step_s(self, batch, C, budget, calibrated=True):
        assert sum(width for width, _ in batch) <= budget, f"step of {len(batch)} lanes over the {budget}-token budget"
        self.steps.append(len(batch))
        return super().step_s(batch, C, budget, calibrated)


def _by_rows(res):
    return {x.n_in: x for x in res["_stats"]}


def test_pack_gives_a_lone_request_every_lane():
    model = _RecordedSteps()
    res = run(model, [[_node(4 * BLOCK, [1])]], BLOCK, 4 * BLOCK, "pack", 1)
    (stat,) = res["_stats"]
    assert model.steps == [4] and stat.done == pytest.approx(stat.send + 1.0)


def test_pack_serves_the_shortest_request_first():
    """A 1-chunk request sent during an 8-chunk request's first step goes first on the next step; the long one takes
    the other 3 lanes. Sessions start at seeded times: session 1 (long) before session 0 (short)."""
    model = _RecordedSteps()
    res = run(model, [[_node(BLOCK, [1])], [_node(8 * BLOCK, [2])]], BLOCK, 4 * BLOCK, "pack", 2)
    short, long_ = _by_rows(res)[BLOCK], _by_rows(res)[8 * BLOCK]
    assert long_.send < short.send < long_.send + 1.0, "precondition: short sent during the long one's first step"
    assert model.steps == [4, 4, 1]
    assert short.done == pytest.approx(long_.send + 2.0)
    assert long_.done == pytest.approx(long_.send + 3.0)


def test_fcfs_keeps_admitted_requests_ahead_of_later_arrivals():
    """Two slots, one lane each: A and B (6 chunks each) hold them, so C (1 chunk, sent later) waits until A finishes.
    Seeded starts: session 2 (A) at 4.2 s, session 1 (B) at 7.6 s, session 0 (C) at 8.4 s."""
    model = _RecordedSteps()
    sessions = [[_node(BLOCK, [1])], [_node(6 * BLOCK - 1, [2])], [_node(6 * BLOCK, [3])]]
    res = run(model, sessions, BLOCK, 2 * BLOCK, "fcfs", 3)
    a, b, c = (_by_rows(res)[n] for n in (6 * BLOCK, 6 * BLOCK - 1, BLOCK))
    assert a.send < b.send < c.send < a.done, "precondition: C is sent while A still holds a slot"
    assert c.done == pytest.approx(a.done + 1.0), "C took A's slot on the step after A finished"
    assert max(model.steps) <= 2


def test_ttft_out_records_each_request_and_filters_by_knob(tmp_path):
    """--ttft-out keeps every request's TTFT and new tokens; plot_ttft_cdf reads them back, filters by knob and labels
    the curves by the knobs that differ."""
    sessions = [[_node(BLOCK, [1]), _node(2 * BLOCK, [1, 2], deps=[0], t=1.0)]]
    res = run(_OneSecondSteps(), sessions, BLOCK, BLOCK, "fcfs", 1)
    path = tmp_path / "ttft.jsonl"
    with open(path, "w") as f:
        for conc in (32, 64):
            knobs = dict(config="64x64:fcfs", concurrency=conc, cache="inf", align="chunk", max_think_s=60.0)
            f.write(json.dumps(ttft_record(knobs, res["_stats"])) + "\n")
    runs = load_runs([str(path)], [])
    assert runs[0]["ttft_s"] == [pytest.approx(x.done - x.send) for x in res["_stats"]]
    assert runs[0]["new_tokens"] == [BLOCK, BLOCK], "the second request's first block is cached"
    assert [r["concurrency"] for r in load_runs([str(path)], [("concurrency", "64"), ("max_think_s", "60")])] == [64]
    names, shared = labels(runs)
    assert names == ["32 sessions", "64 sessions"] and "unlimited cache" in shared
    assert labels([{**runs[0], "config": c} for c in ("8192x8192:fcfs", "2048x8192:pack")])[0] == [
        "8k",
        "2k x4 lanes, shortest-first",
    ]
