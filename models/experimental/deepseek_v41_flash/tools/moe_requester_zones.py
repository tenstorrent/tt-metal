# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The persistent TP12 MoE as its requesters see it, from a streaming-profiler zone CSV.

Capture (MOE_PERF_FUSE_HC=1 for requests with the hyper-connection post step fused, as the model sends):

    TT_METAL_STREAMING_PROFILER=1 TT_METAL_STREAMING_PROFILER_ZONE_CSV=/tmp/fe_zones.csv \\
    MOE_PERF_PERSISTENT_ONLY=1 MOE_PERF_WARMUP=5 MOE_PERF_STEPS=40 MOE_PERF_FUSE_HC=1 \\
    pytest models/experimental/deepseek_v41_flash/tests/decode/test_moe_persistent.py::test_moe_persistent_tp12_perf

    python models/experimental/deepseek_v41_flash/tools/moe_requester_zones.py /tmp/fe_zones.csv --warmup 5

Every persistent_moe_request op on every requester chip, timed on that chip's clock from the op's start (its
first core's kernel start):

- send:     MOE_REQ_SEND end (chips that send; a single-sender row has one);
- arrive:   the last sum core's MOE_SUM_WAIT end -- every MoE chip's partials are in: the MoE's latency as the
            requester sees it;
- out:      the last MOE_SUM_WRITE end -- the op's output written;
- end:      the last core's kernel end;
- tail:     out - arrive, the requester's own work once the partials are in (the critical-path sum / HC post);
- post:     the last MOE_SUM / MOE_HC_POST end - arrive (the compute's share of the tail);
- mix:      the last MOE_HC_MIX end (fused only): it should be well before arrive, hidden in the wait;
- gap:      from the previous request op's end on the chip to this one's start (dispatch; in the model other ops
            fill it).

An op whose program runs once is eager; a program replayed (one trace per (requester, layer)) is traced. The
perf test runs 1 requester, then 3: a run starts where a requester mesh not yet seen in the current one sends
(program run ids are global and increase over time), and each run is reported on its own.
"""

import argparse
import csv
import statistics
from collections import defaultdict

ZONES = ("MOE_REQ_SEND", "MOE_SUM_WAIT", "MOE_SUM", "MOE_HC_MIX", "MOE_HC_POST", "MOE_SUM_WRITE")


def _load(path):
    """``(mhz, spans)``: spans[(chip, run)][(core, risc, name)] = [(start, end), ...] in issue order, for the MOE_
    zones and the kernel zones of the programs (runs) holding them."""
    raw = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        mhz = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)
        pending = {}
        for row in reader:
            name = row[10]
            if not (name in ZONES or name.endswith("-KERNEL")):
                continue
            chip, run = int(row[0]), row[7]
            # Logical core coordinates in a streaming capture; the device profiler's log has only physical ones.
            core = (int(row[15]), int(row[16])) if len(row) > 16 else (int(row[1]), int(row[2]))
            key = (core, row[3], name)
            if row[11] == "ZONE_START":
                pending[(chip, run, key)] = int(row[5])
            elif row[11] == "ZONE_END" and (chip, run, key) in pending:
                raw[(chip, run)][key].append((pending.pop((chip, run, key)), int(row[5])))
    spans = {k: v for k, v in raw.items() if any(n in ZONES for (_, _, n) in v)}
    return mhz, spans


def _instances(zones):
    """One dict per run of the program, in issue order."""
    counts = [len(v) for (_, _, name), v in zones.items() if name == "MOE_SUM_WAIT"]
    if not counts:  # e.g. the stop request, which gets no partials
        return []
    n = min(counts)
    out = []
    for k in range(n):
        at = lambda name: [v[k] for (_, _, nm), v in zones.items() if nm == name and k < len(v)]
        kernels = [v[k] for (_, _, nm), v in zones.items() if nm.endswith("-KERNEL") and k < len(v)]
        waits, writes = at("MOE_SUM_WAIT"), at("MOE_SUM_WRITE")
        if not (kernels and waits and writes):
            continue
        posts = at("MOE_HC_POST") or at("MOE_SUM")
        mixes = at("MOE_HC_MIX")
        sends = at("MOE_REQ_SEND")
        out.append(
            {
                "start": min(s for s, _ in kernels),
                "end": max(e for _, e in kernels),
                "send": max(e for _, e in sends) if sends else None,
                "arrive": max(e for _, e in waits),
                "out": max(e for _, e in writes),
                "post_end": max(e for _, e in posts) if posts else None,
                "mix_end": max(e for _, e in mixes) if mixes else None,
                "fused": bool(mixes),
            }
        )
    return out


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv")
    parser.add_argument("--warmup", type=int, default=5, help="request ops skipped at the start of each chip's loop")
    args = parser.parse_args()

    mhz, spans = _load(args.csv)
    us = lambda cycles: None if cycles is None else cycles / mhz
    insts_of = {key: _instances(zones) for key, zones in spans.items()}
    # A program (run id) runs on every chip of one requester mesh.
    meshes = defaultdict(set)
    for (chip, run), insts in insts_of.items():
        if insts:
            meshes[int(run)].add(chip)
    session_of = {}
    session, seen, runs_in = 0, set(), 0
    for run in sorted(meshes):
        mesh = frozenset(meshes[run])
        # A new mesh after a while of the others sending is the next run; within a run's first round, it joins.
        if mesh not in seen and runs_in > 2 * max(len(seen), 1):
            session, seen, runs_in = session + 1, set(), 0
        seen.add(mesh)
        runs_in += 1
        session_of[run] = session
    # session -> chip -> mode -> instances
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (chip, run), insts in insts_of.items():
        if insts:
            mode = "traced" if len(insts) > 1 else "eager"
            per[session_of[int(run)]][chip][mode] += insts
    if not per:
        raise SystemExit("no persistent_moe_request zones (MOE_SUM_WAIT) in the capture")

    cols = ("send", "arrive", "post", "out", "end", "tail", "mix", "gap")
    print(f"{mhz:.0f} MHz; median us per request op, from its start on each requester chip")
    for session in sorted(per):
        per_chip = per[session]
        groups = {frozenset(meshes[r]) for r, s_ in session_of.items() if s_ == session}
        print(f"\n=== run {session}: {len(groups)} requester mesh{'es' if len(groups) > 1 else ''}")
        for mode in ("eager", "traced"):
            rows = []
            for chip in sorted(per_chip):
                insts = sorted(per_chip[chip][mode], key=lambda i: i["start"])
                for prev, cur in zip([None] + insts[:-1], insts):
                    cur["gap"] = None if prev is None else cur["start"] - prev["end"]
                insts = insts[args.warmup :]
                if not insts:
                    continue
                rel = lambda i, k: None if i[k] is None else i[k] - i["start"]
                vals = {
                    "send": [us(rel(i, "send")) for i in insts],
                    "arrive": [us(rel(i, "arrive")) for i in insts],
                    "post": [None if i["post_end"] is None else us(i["post_end"] - i["arrive"]) for i in insts],
                    "out": [us(rel(i, "out")) for i in insts],
                    "end": [us(rel(i, "end")) for i in insts],
                    "tail": [us(i["out"] - i["arrive"]) for i in insts],
                    "mix": [us(rel(i, "mix_end")) for i in insts],
                    "gap": [us(i["gap"]) for i in insts],
                }
                kind = "fused" if any(i["mix_end"] is not None for i in insts) else "plain"
                rows.append((chip, len(insts), kind, {c: _median(v) for c, v in vals.items()}))
            if not rows:
                continue
            print(f"  {mode}:")
            print(f"    {'chip':>4} {'ops':>4} {'kind':>6}" + "".join(f"{c:>9}" for c in cols))
            for chip, n, kind, med in rows:
                print(f"    {chip:>4} {n:>4} {kind:>6}" + "".join(f"{med[c]:>9.2f}" for c in cols))
            allm = {c: _median([r[3][c] for r in rows]) for c in cols}
            print(f"    {'all':>4} {'':>4} {'':>6}" + "".join(f"{allm[c]:>9.2f}" for c in cols))


if __name__ == "__main__":
    main()
