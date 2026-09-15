# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Single-prefill latency from the FIXED metric, on a process-warm run. Asserts; does not print and hope.

Why this exists next to `analyze_prefill_latency.py` rather than replacing it: that script reads
`E2E_CLOCK`, whose `last_compute_end` is stamped after `_forward_shutdown` and the fabric drain, and
it *refuses* any log with more than one request. Both are now wrong the other way round:

  * the honest end of compute is `E2E_CLOCK_V2 last_chunk_end`, stamped inside the sentinel branch
    before the forward and the drain;
  * the request that should be reported is request 2 of 2 in ONE process, so that trace capture and
    first-chunk compile fall outside the measured request. `PREFILL_PRODUCER_MAX_REQUESTS=2`.
    Running a cell twice does NOT achieve this -- each run is a fresh process and re-captures.

What it reports, per cell:
  * the measured (last) request's latency: rank 0's first chunk of that request -> last rank's
    `last_chunk_end`;
  * N per-chunk intervals for N chunks -- the last one for the first time, which also makes a
    single-chunk ISL measurable at all;
  * request 1 minus request 2, which is a direct measurement of trace-capture cost per topology;
  * the drain gap (`E2E_DRAIN_GAP`), i.e. how much the old `E2E_CLOCK` metric was over-reporting.

Gates (handoff section 4). Any failure exits non-zero with GATE FAIL on stderr.
    1  measured request index >= 2                      (the warm gate that matters)
    2  JIT cache stats 100% hits present
    3  no rank-level Python failure in the log
    5  interval closure: sum(per-chunk intervals) == last_chunk_end - first_chunk_start
   10  tail sanity: the last chunk is not wildly out of line with the others. `last_chunk_end` is
        stamped on SENTINEL RECEIPT, so on a pipeline it is max(own completion, sentinel arrival)
        and every upstream `_forward_shutdown` delays it. Warm that is noise; cold it over-reported
        a 1-chunk PP=4 latency by 48x with every other gate green.

Usage: analyze_prefill_latency_v2.py <runner.log> [--min-request N] [--json]
"""

import json
import re
import statistics
import sys

CHUNK_START = re.compile(r"\[pp rank (\d+)\] CHUNK_START c=(\d+) compute_start=([0-9.]+) slot=(-?\d+) \[(\d+),(\d+)\)")
E2E_V2 = re.compile(
    r"\[pp rank (\d+)\] E2E_CLOCK_V2 first_compute_start=([0-9.na/]+) last_chunk_end=([0-9.]+) chunks=(\d+)"
)
E2E_V1 = re.compile(r"\[pp rank (\d+)\] E2E_CLOCK first_compute_start=([0-9.na/]+) last_compute_end=([0-9.]+)")
DRAIN_GAP = re.compile(r"\[pp rank (\d+)\] E2E_DRAIN_GAP gap_ms=([0-9.-]+)")
JIT = re.compile(r"JIT cache stats: (\d+)/(\d+) hits \(([0-9.]+)%\)")
# Rank teardown legitimately logs `TT_FATAL: cq_id 0 is out of range` from the D2D stream-service
# destructors after the device is closed, so a bare TT_FATAL grep fails every healthy run.
BENIGN = re.compile(r"cq_id \d+ is out of range")
FAILURE = re.compile(r"Traceback \(most recent call last\)|AssertionError|TT_FATAL")

FAILS = []


def gate(n, ok, msg):
    tag = "ok  " if ok else "FAIL"
    print(f"  gate {n}: [{tag}] {msg}")
    if not ok:
        FAILS.append(f"gate {n}: {msg}")
    return ok


def parse(path):
    txt = open(path, errors="replace").read()
    chunks = {}
    for rk, c, t, slot, st, en in CHUNK_START.findall(txt):
        chunks.setdefault(int(rk), []).append((int(c), float(t), int(st), int(en)))
    for rk in chunks:
        chunks[rk].sort()
    v2 = {int(r): (a, float(b), int(n)) for r, a, b, n in E2E_V2.findall(txt)}
    v1 = {int(r): (a, float(b)) for r, a, b in E2E_V1.findall(txt)}
    gaps = {int(r): float(g) for r, g in DRAIN_GAP.findall(txt)}
    jit = [(int(a), int(b), float(c)) for a, b, c in JIT.findall(txt)]
    bad = [ln for ln in txt.splitlines() if FAILURE.search(ln) and not BENIGN.search(ln)]
    kv = re.search(r"PREFILL_KV_ONLY_LAST_LAYER *= *(\w+)", txt)
    return chunks, v2, v1, gaps, jit, bad, (kv.group(1).lower() == "true") if kv else None


def split_requests(rows):
    """[(req_index, [(chunk_index_in_request, t)])] -- a new request starts at actual_start == 0."""
    out, cur = [], []
    for c, t, st, en in rows:
        if st == 0 and cur:
            out.append(cur)
            cur = []
        cur.append((st, t))
    if cur:
        out.append(cur)
    return list(enumerate(out))


def main():
    path = sys.argv[1]
    min_req = 2
    if "--min-request" in sys.argv:
        min_req = int(sys.argv[sys.argv.index("--min-request") + 1])
    chunks, v2, v1, gaps, jit, bad, kv_only = parse(path)

    print(f"log        : {path}")
    if not chunks:
        raise SystemExit("no CHUNK_START records -- is this a runner log from a completed run?")
    last = max(chunks)
    reqs = {rk: split_requests(rows) for rk, rows in chunks.items()}
    n_req = len(reqs[0])
    n_chunks = len(reqs[last][-1][1])
    isl = max(en for rows in chunks.values() for *_, en in rows)
    print(f"ranks      : 0..{last}   requests: {n_req}   chunks in measured request: {n_chunks}   ISL: {isl}")
    print(f"last layer : {'kv-only (no token emitted)' if kv_only else 'FULL (norm + LM head)'}")

    print("\ngates")
    gate(
        1,
        n_req >= min_req,
        f"measured request index = {n_req} (need >= {min_req}); "
        f"trace capture is {'outside' if n_req >= 2 else 'INSIDE'} the measured request",
    )
    if jit:
        h, t, pct = jit[-1]
        gate(2, pct >= 99.999, f"JIT cache stats: {h}/{t} hits ({pct}%)")
    else:
        gate(2, False, "no `JIT cache stats:` line in the log")
    gate(
        3,
        not bad,
        f"{len(bad)} rank-level Python failure lines" + (f" -- first: {bad[0].strip()[:110]}" if bad else ""),
    )

    if last not in v2:
        gate(5, False, f"no E2E_CLOCK_V2 on rank {last} -- the metric fix is not on the branch that ran this")
        print("\nRESULT: unmeasurable (old metric only).")
        raise SystemExit(1 if FAILS else 0)

    _, end, v2_chunks = v2[last]
    m_last = reqs[last][-1][1]  # measured request, on the last rank
    m_first = reqs[0][-1][1]  # measured request, on rank 0
    starts = [t for _, t in m_last]
    ivs = [(b - a) * 1000.0 for a, b in zip(starts, starts[1:])] + [(end - starts[-1]) * 1000.0]
    closure = sum(ivs) - (end - starts[0]) * 1000.0
    gate(
        5,
        abs(closure) < 5.0,
        f"interval closure: {len(ivs)} intervals for {n_chunks} chunks, "
        f"sum {sum(ivs):.1f} ms vs span {(end - starts[0])*1000.0:.1f} ms, delta {closure:+.3f} ms",
    )

    # GATE 10 -- the tail the metric is still exposed to, and the reason it must be asserted.
    #
    # `last_chunk_end` is stamped when a rank RECEIVES the shutdown sentinel, because that is the
    # only moment the loop learns the stream has ended. On a pipeline the sentinel reaches rank i
    # only after every upstream rank ran `_forward_shutdown`, so the last rank's stamp is really
    # max(its own completion, sentinel arrival) -- and `_forward_shutdown` is not prefill work.
    #
    # Warm it is milliseconds and this is noise. Cold it is not: measured 2026-09-15 on a
    # 40%-warm kernel cache, rank 0's `_forward_shutdown` cost 12,849 ms, so ranks 1-3 each stamped
    # ~12.8 s late and the 1-chunk PP=4 latency read 13.32 s against a true ~0.25 s -- a 48x
    # over-report that every other gate passed. Assert it, because it is silent and it is enormous.
    others = ivs[:-1] if len(ivs) > 1 else [(m_last[0][1] - reqs[last][-2][1][0][1]) * 1000.0]
    ref = statistics.median([x for x in others if x > 0]) if any(x > 0 for x in others) else float("nan")
    tail_ratio = ivs[-1] / ref if ref and ref == ref else float("inf")
    upstream = {rk: g for rk, g in gaps.items() if rk != last}
    worst = max(upstream.values()) if upstream else 0.0
    gate(
        10,
        tail_ratio <= 3.0,
        f"tail sanity: last chunk {ivs[-1]:.1f} ms vs {ref:.1f} ms reference "
        f"({'1-chunk request, so the reference is request 1 on that rank' if len(ivs) == 1 else 'median of the other intervals'})"
        f" -> {tail_ratio:.1f}x"
        + (
            f"; worst upstream _forward_shutdown is {worst:.1f} ms, which delays the sentinel and "
            f"therefore this stamp"
            if worst > 50.0
            else ""
        ),
    )

    lat = end - m_first[0][1]
    fill = (starts[0] - m_first[0][1]) * 1000.0
    print(f"\nmeasured request ({n_req} of {n_req}), WARM")
    print(f"  latency (rank0 first chunk -> rank{last} last_chunk_end): {lat:.4f} s")
    print(f"  {isl} tokens -> {isl/lat:,.0f} tok/s")
    if last > 0:
        print(f"  pipeline fill (rank0 start -> rank{last} first chunk): {fill:.1f} ms ({fill/lat/10:.1f}% of latency)")
    print(f"  per-chunk intervals on rank{last} (ms): " + " ".join(f"{x:.1f}" for x in ivs))
    print(f"  LAST chunk: {ivs[-1]:.1f} ms  <- unobtainable before the metric fix")
    if len(ivs) > 1:
        print(
            f"  old harness would have estimated it as the previous interval, {ivs[-2]:.1f} ms "
            f"({(ivs[-1]-ivs[-2])/ivs[-2]*100:+.1f}%)"
        )

    if n_req >= 2:
        # Request k's end is on the LAST rank, not rank 0. Measuring request 1 as rank0-start to
        # rank0-start gives only rank 0's slice of it, which on a 4-stage pipeline is smaller than
        # the whole request -- it made request1 - request2 come out NEGATIVE on a synthetic PP log.
        # The last rank goes straight from finishing request k's last chunk into request k+1's
        # first chunk, so its request-boundary stamp is the end of request k.
        r1_start = reqs[0][-2][1][0][1]
        r1_end = reqs[last][-1][1][0][1]
        r1_lat = r1_end - r1_start
        print("\ncapture cost, request 1 minus request 2 (same process, same slot)")
        print(f"  request 1 latency (rank0 start -> rank{last} start of request 2): {r1_lat:.4f} s")
        print(f"  request 2 latency (the reported, warm one)                     : {lat:.4f} s")
        print(f"  difference = trace capture + first-chunk compile               : {r1_lat - lat:+.4f} s")

        # Per-stage decomposition. Every rank stamps its own E2E_CLOCK_V2, so the measured request's
        # per-stage time is available for ALL stages -- which is the quantity a saturated pipeline
        # never reveals (it only shows max_i t[i][c]) and which the estimator had to model.
        print(f"\n  per-stage, measured request ({n_chunks} chunk(s) each):")
        print(f"    {'rank':>5} {'warm stage time':>16} {'req1 stage+capture':>19} {'capture':>10}")
        for rk in sorted(reqs):
            if rk not in v2:
                print(f"    {rk:>5}   no E2E_CLOCK_V2")
                continue
            warm = (v2[rk][1] - reqs[rk][-1][1][0][1]) * 1000.0
            r1 = (reqs[rk][-1][1][0][1] - reqs[rk][-2][1][0][1]) * 1000.0
            print(f"    {rk:>5} {warm:15.1f}ms {r1:18.1f}ms {r1 - warm:9.1f}ms")
        print(
            "    (warm = last_chunk_end - first chunk start of the measured request; req1 = that "
            "rank's\n     request-1 span, which for a 1-chunk request is its stage time plus its "
            "own capture)"
        )
        neg = [rk for rk in sorted(reqs) if rk in v2 and reqs[rk][-1][1][0][1] < reqs[rk][-2][1][0][1]]
        if neg:
            print(
                f"    WARNING ranks {neg}: request 2 starts BEFORE request 1 on that rank, which one "
                f"slot\n    cannot do -- the request split is wrong, so the capture column is junk."
            )

    if last in v1:
        old = v1[last][1] - m_first[0][1]
        print(f"\nold metric, for comparison")
        print(f"  E2E_CLOCK span (same start, post-drain end): {old:.4f} s  ({(old/lat - 1)*100:+.1f}%)")
        if last in gaps:
            print(f"  E2E_DRAIN_GAP on rank{last}: {gaps[last]:.1f} ms")
        for rk in sorted(gaps):
            print(f"    rank{rk} drain gap: {gaps[rk]:.1f} ms")

    if "--json" in sys.argv:
        print(
            "\nJSON "
            + json.dumps(
                dict(
                    log=path,
                    isl=isl,
                    ranks=last + 1,
                    requests=n_req,
                    chunks=n_chunks,
                    latency_s=lat,
                    fill_ms=fill,
                    intervals_ms=ivs,
                    closure_ms=closure,
                    gates_failed=FAILS,
                )
            )
        )
    if FAILS:
        print("\nGATE FAIL: " + "; ".join(FAILS), file=sys.stderr)
        raise SystemExit(2)
    print("\nall gates passed")


if __name__ == "__main__":
    main()
