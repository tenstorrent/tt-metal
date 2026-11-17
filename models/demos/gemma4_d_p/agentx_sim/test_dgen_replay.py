# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of dgen_replay.py: parsing a tt-d-gen run's logs, matching its requests, and the replay's timing,
scheduling and caches. Small fixtures, no device and no run directory:
`pytest models/demos/gemma4_d_p/agentx_sim/test_dgen_replay.py`."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgen_replay import (  # noqa: E402
    BLOCK,
    CHUNK,
    Engine,
    Slots,
    Tokens,
    attach_traces,
    load_aiperf,
    load_engine,
    load_routes,
    main,
    match_engine,
    replay,
    scaled_steps,
)

T0 = 1793138516.0  # 2026-10-27T22:01:56Z


def stamp(t, frac=3):
    """The logs' UTC timestamp for epoch second t."""
    import datetime as dt

    s = dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")
    return s[: len(s) - 6 + frac]


def inject(t, slot, start, end):
    return f"{stamp(t)} INFO     Pipeline  python/1 prefill_pipeline.cpp:219  prefill inject slot_id={slot} start={start} end={end}\n"


def route(t, rid, phase):
    # As the frontend writes it: ANSI colours, a Z suffix, and request_id fields besides x_request_id.
    return (
        f"\x1b[2m{stamp(t, 6)}Z\x1b[0m \x1b[32m INFO\x1b[0m kv_router.route_request: route request completed "
        f'request.outcome="success" x_request_id="{rid}" request_id=abc dp_rank=0 phase={phase} worker_id=1\n'
    )


def aiperf(rid, conv, send, isl, ttft_ms, phase="profiling", kind="weka_main", trace=None, outer=0):
    return {
        "metadata": {
            "x_request_id": rid,
            "conversation_id": conv,
            "phase_kind": phase,
            "source_kind": kind,
            "source_trace_id": trace or conv,
            "source_outer_idx": outer,
            "request_start_ns": int(send * 1e9),
        },
        "metrics": {"input_sequence_length": {"value": isl}, "time_to_first_token": {"value": ttft_ms}},
    }


def write(path, lines):
    with open(path, "w") as f:
        f.writelines(lines)
    return str(path)


def test_load_engine_groups_a_slots_injects_into_requests(tmp_path):
    path = write(
        tmp_path / "prefill.log",
        [
            "unrelated line\n",
            inject(T0, 0, 0, CHUNK),
            inject(T0 + 0.1, 1, 4096, 4096 + CHUNK),
            inject(T0 + 0.2, 0, CHUNK, CHUNK + 100),  # short: slot 0's request is done
            inject(T0 + 0.3, 0, 0, 500),  # a new request on slot 0
            inject(T0 + 0.4, 1, 4096 + CHUNK, 4096 + CHUNK + 7),
        ],
    )
    reqs = load_engine(path)
    assert [(slot, start, end, n) for slot, _, start, end, n in reqs] == [
        (0, 0, CHUNK + 100, 2),
        (1, 4096, 4096 + CHUNK + 7, 2),
        (0, 0, 500, 1),
    ]
    assert reqs[1][1] == pytest.approx(T0 + 0.1, abs=1e-3), "seated at its first inject"


def test_load_engine_starts_a_request_when_the_slot_does_not_continue(tmp_path):
    path = write(tmp_path / "prefill.log", [inject(T0, 0, 0, CHUNK), inject(T0 + 1, 0, 1024, 1024 + CHUNK)])
    assert [(r[2], r[4]) for r in load_engine(path)] == [(0, 1), (1024, 1)]


def test_load_routes_reads_coloured_frontend_lines(tmp_path):
    path = write(
        tmp_path / "frontend.log", ["noise\n", route(T0 + 2, "r1", "Decode"), route(T0 + 1.5, "r1", "Prefill")]
    )
    routes = load_routes(path)
    assert routes["r1"]["Prefill"] == pytest.approx(T0 + 1.5, abs=1e-6)
    assert routes["r1"]["Decode"] == pytest.approx(T0 + 2, abs=1e-6)


def test_load_aiperf_sorts_by_send_and_reads_ttft(tmp_path):
    path = write(
        tmp_path / "export.jsonl",
        [json.dumps(aiperf("b", "c", T0 + 5, 300, 250)) + "\n", json.dumps(aiperf("a", "c", T0, 200, 1500)) + "\n"],
    )
    reqs = load_aiperf(path)
    assert [r["rid"] for r in reqs] == ["a", "b"]
    assert reqs[0]["first"] == pytest.approx(T0 + 1.5) and reqs[0]["isl"] == 200


def test_match_engine_pairs_by_length_seated_after_send():
    reqs = [dict(rid="a", send=10.0, isl=500), dict(rid="b", send=11.0, isl=500), dict(rid="c", send=12.0, isl=9)]
    engine = [(0, 9.0, 0, 500, 1), (0, 10.5, 0, 500, 1), (1, 11.5, 0, 500, 1), (2, 12.5, 0, 700, 1)]
    routes = {"a": {"Prefill": 10.8}}
    assert match_engine(reqs, engine, routes) == 2
    assert reqs[0]["eng"] is engine[1], "the one seated before its send is another request"
    assert reqs[1]["eng"] is engine[2]
    assert reqs[2]["eng"] is None, "no engine request of its length"
    assert reqs[0]["route_p"] == 10.8 and reqs[1]["route_p"] is None


def test_attach_traces_gives_aux_sidecars_their_own_session_and_picks_the_closest_subagent(tmp_path):
    traces = [
        {
            "id": "t1",
            "requests": [
                {"in": 128, "hash_ids": [1, 2]},
                {"requests": [{"in": 64, "hash_ids": [5]}, {"in": 640, "hash_ids": [6]}]},
            ],
        },
        {"id": "t2", "requests": [{"in": 64, "hash_ids": [9]}]},
    ]
    path = write(tmp_path / "traces.jsonl", [json.dumps(t) + "\n" for t in traces])
    reqs = [
        dict(trace="t1", outer=0, kind="weka_main", conv="t1", isl=130),
        dict(trace="t1", outer=1, kind="weka_subagent", conv="t1:sub:0", isl=600),
        dict(trace="t1", outer=0, kind="weka_main", conv="t1:aux:0", isl=130),
        dict(trace="t2", outer=0, kind="weka_main", conv="t2", isl=64),
    ]
    attach_traces(reqs, path)
    assert reqs[0]["hashes"] == [1, 2] and reqs[1]["hashes"] == [6]
    assert reqs[0]["sess"] == reqs[1]["sess"], "a subagent shares its trace's session"
    assert len({r["sess"] for r in reqs}) == 3, "the sidecar's session is its own"


def test_attach_traces_names_the_trace_ids_a_wrong_corpus_lacks(tmp_path):
    path = write(tmp_path / "traces.jsonl", ["\n", '{"models": []}\n', json.dumps({"id": "t1", "requests": []}) + "\n"])
    with pytest.raises(SystemExit, match="no entry for 1 trace ids of the run, e.g. t9"):
        attach_traces([dict(trace="t9", outer=0, kind="weka_main", conv="t9", isl=1)], path)


class _OneSecondSteps:
    def __call__(self, batch, C, budget):
        self.seen = getattr(self, "seen", []) + [batch[0][1]]
        return 1.0, 0.0


def req(rid, conv, send, isl, route_p=None, sess=0, hashes=(), eng=None):
    return dict(
        rid=rid, conv=conv, send=send, isl=isl, first=send, route_p=route_p, sess=sess, hashes=list(hashes), eng=eng
    )


def test_overhead_delays_the_request_by_a_constant_plus_its_length():
    r = req("a", "c", 10.0, 3 * CHUNK)
    out, busy = replay(
        [r], _OneSecondSteps(), Tokens(), "rr", request_overhead_s=0.5, request_overhead_per_token_s=1e-4
    )
    sent, start, done = out["a"]
    assert (sent, start) == (10.0, 0)
    assert done == pytest.approx(10.0 + 0.5 + 3 * CHUNK * 1e-4 + 3)
    assert busy == 3.0


def test_a_later_turn_keeps_its_measured_gap_after_the_replayed_prefill():
    first = req("a", "c", 0.0, 2 * CHUNK, route_p=1.0)  # measured: done at 1 s
    second = req("b", "c", 3.0, CHUNK)  # sent 2 s after that
    out, _ = replay([first, second], _OneSecondSteps(), Tokens(), "rr")
    assert out["a"][2] == 2.0, "replayed: done at 2 s, 1 s later than measured"
    assert out["b"][0] == 4.0, "so the next turn is sent 1 s later"
    early = req("a", "c", 0.0, CHUNK, route_p=5.0)
    out, _ = replay([early, req("b", "c", 7.0, CHUNK)], _OneSecondSteps(), Tokens(), "rr")
    assert out["b"][0] == 7.0, "never before its measured send"


def test_rr_interleaves_the_seated_requests_and_fcfs_runs_them_in_order():
    def two():
        return [req("a", "x", 0.0, 2 * CHUNK), req("b", "y", 0.0, 2 * CHUNK, sess=1)]

    rr, _ = replay(two(), _OneSecondSteps(), Tokens(), "rr", inject_depth=4)
    fcfs, _ = replay(two(), _OneSecondSteps(), Tokens(), "fcfs", inject_depth=4)
    assert (rr["a"][2], rr["b"][2]) == (3.0, 4.0)
    assert (fcfs["a"][2], fcfs["b"][2]) == (2.0, 4.0)


def test_a_new_request_waits_behind_the_chunks_already_queued():
    long = req("a", "x", 0.0, 4 * CHUNK)
    new = req("b", "y", 0.5, CHUNK, sess=1)
    deep, _ = replay([long, new], _OneSecondSteps(), Tokens(), "rr", inject_depth=3)
    shallow, _ = replay(
        [req("a", "x", 0.0, 4 * CHUNK), req("b", "y", 0.5, CHUNK, sess=1)], _OneSecondSteps(), Tokens(), "rr"
    )
    assert deep["b"][2] == 4.0, "three of a's chunks were queued when b arrived"
    assert shallow["b"][2] == 2.0


def test_slots_bound_the_seated_requests():
    reqs = [req("a", "x", 0.0, CHUNK), req("b", "y", 0.0, CHUNK, sess=1)]
    out, _ = replay(reqs, _OneSecondSteps(), Slots(1), "rr", slots=1, inject_depth=9)
    assert (out["a"][2], out["b"][2]) == (1.0, 2.0)


def test_resume_is_the_cached_prefix_rounded_down_to_align():
    first = req("a", "c", 0.0, 40 * BLOCK, route_p=0.0, hashes=range(40))
    second = req("b", "c", 0.0, 50 * BLOCK, hashes=range(50))
    out, _ = replay([first, second], _OneSecondSteps(), Tokens(), "rr", align=1024)
    assert out["b"][1] == 40 * BLOCK // 1024 * 1024
    out, _ = replay(
        [
            req("a", "c", 0.0, 40 * BLOCK, route_p=0.0, hashes=range(40)),
            req("b", "c", 0.0, 50 * BLOCK, hashes=range(50)),
        ],
        _OneSecondSteps(),
        Slots(2),
        "rr",
        align=BLOCK,
    )
    assert out["b"][1] == 40 * BLOCK, "the slot holding the prefix is reused"


def test_engine_cache_resumes_where_the_engine_did():
    steps = _OneSecondSteps()
    r = req("a", "c", 0.0, CHUNK + 3000, eng=(0, 0.1, 3072, CHUNK + 3000, 1))
    out, _ = replay([r], steps, Engine(), "rr")
    assert out["a"][1] == 3072 and steps.seen == [3072]


def test_scaled_steps_scale_by_the_ratio_of_chunk_times():
    class Model:
        def step_s(self, batch, C, budget):
            return 0.2, 0.01

    step_s = scaled_steps(Model(), (150.0, 1.0), (100.0, 0.5))
    assert step_s([(CHUNK, 0)], CHUNK, CHUNK) == pytest.approx((0.3, 0.01))
    assert step_s([(CHUNK, 100_000)], CHUNK, CHUNK)[0] == pytest.approx(0.2 * 250 / 150)


def write_run(tmp_path):
    """A two-request run: one conversation, two turns, 8k chunks."""
    run = tmp_path / "run"
    (run / "aiperf").mkdir(parents=True)
    write(
        run / "aiperf" / "profile_export.jsonl",
        [
            json.dumps(aiperf("r1", "t1", T0, CHUNK + 100, 400)) + "\n",
            json.dumps(aiperf("r2", "t1", T0 + 5, CHUNK + 300, 300, outer=1)) + "\n",
        ],
    )
    write(run / "frontend.log", [route(T0 + 0.35, "r1", "Prefill"), route(T0 + 5.25, "r2", "Prefill")])
    write(
        run / "prefill.log",
        [
            inject(T0 + 0.02, 0, 0, CHUNK),
            inject(T0 + 0.15, 0, CHUNK, CHUNK + 100),
            inject(T0 + 5.02, 0, 8192, CHUNK + 300),
        ],
    )
    hashes = list(range((CHUNK + 300) // BLOCK + 1))
    traces = write(
        tmp_path / "traces.jsonl",
        [
            json.dumps(
                {
                    "id": "t1",
                    "requests": [
                        {"in": CHUNK + 100, "hash_ids": hashes[:129]},
                        {"in": CHUNK + 300, "hash_ids": hashes},
                    ],
                }
            )
            + "\n"
        ],
    )
    return str(run), traces


def test_main_replays_a_run_and_writes_json(tmp_path):
    run, traces = write_run(tmp_path)
    out = tmp_path / "replay.json"
    rows = main([run, "--traces", traces, "--cache", "slot,engine", "--inject-depth", "1,9", "--json", str(out)])
    assert [r["name"].split()[0] for r in rows] == ["measured", "slot", "slot", "engine", "engine"]
    m = rows[0]
    assert m["requests"] == 2
    assert m["max"] == pytest.approx(0.35, abs=1e-3), "send to the Prefill route span: 0.35 s and 0.25 s"
    assert m["hit"] == pytest.approx(8192 / (2 * CHUNK + 400))
    assert all(r["same_start"] == 2 for r in rows[1:]), "both caches resume r2 where the engine did"
    assert json.loads(out.read_text())["rows"][1]["evictions"] == 0


@pytest.mark.parametrize(
    "flag, value",
    [
        ("--cache", "slot,lru"),
        ("--policy", "pack"),
        ("--inject-depth", "9,0"),
        ("--request-overhead-ms", "-1"),
        ("--request-overhead-ms-per-1k", "x"),
        ("--chunk-time", "150"),
        ("--chunk-time", "0,1"),
        ("--slots", "0"),
    ],
)
def test_main_rejects_bad_knobs_before_reading_the_run(tmp_path, flag, value):
    with pytest.raises(SystemExit) as e:
        main([str(tmp_path / "missing"), "--traces", "missing.jsonl", flag, value])
    assert e.value.code == 2
