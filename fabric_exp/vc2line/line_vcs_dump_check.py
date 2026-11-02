#!/usr/bin/env python3
"""line_vcs_dump_check.py [CASE ...]: checks the host's router args for line VCs on 1D fabrics (LINE_VCS_HOST_DESIGN.md).

Reads the arg dumps of the mock captures (vc2l_offline.py cases *_lvc<N>[_1erisc]: TT_FABRIC_EXP_DUMP_ARGS, one JSON
line per router ERISC, COV/argdump/<case>.jsonl) and the slot log (COV/cases/<case>/mock.log), and checks every router
ERISC against the design: N line VCs, each with a worker and a forwarding channel and a receiver; the worker role on each
VC's first channel; every line VC serviced, routing as a line, forwarding on its own VC; first-level acks and injection
flags on rings (deadlock avoidance) and not on lines; transaction IDs per receiver channel; distinct stream registers;
the flat bases; every line VC opening its downstream where VC0 does; the slot row. Prints one line per case and every
failed check. (For N >= 2 the router does not compile yet, so a capture stops at the first device's kernel build: the
dump holds that device's routers.)"""
import collections
import glob
import json
import os
import re
import sys

COV = os.environ.get("TT_COV_DIR", "/home/aagarwal/.tt-metal-vc2line-cov")
SLOT_ROWS = {1: None, 2: (13, 12, 16), 3: (10, 9, 8)}  # (worker, forwarding, receiver) per line VC


def ids_per_receiver_channel(channels, deadlock_avoidance):
    if channels == 1:
        return 8 if deadlock_avoidance else 16
    return 8 if deadlock_avoidance and channels <= 2 else 4


def check_record(rec, n, ring, fail):
    a = rec["named_ct_args"]
    where = f"device {rec['device']} eth {rec['eth_chan']} risc {rec['risc']}"

    def expect(name, want):
        got = a.get(name)
        if got != want:
            fail(f"{where}: {name} = {got}, expected {want}")

    workers = {2 * vc for vc in range(n)}
    for vc in range(3):
        line = vc < n
        expect(f"ACTUAL_VC{vc}_SENDER_CHANNELS", 2 if line else 0)
        expect(f"VC{vc}_SERVICED", 1 if line else 0)
        expect(f"VC{vc}_ROUTES_AS_LINE", 1 if line else 0)
        if line:
            expect(f"ENABLE_FIRST_LEVEL_ACK_VC{vc}", a.get("ENABLE_DEADLOCK_AVOIDANCE"))
    expect("ENABLE_DEADLOCK_AVOIDANCE", 1 if ring else 0)
    expect("VC2_LINE_MODE", 1 if n == 3 else 0)
    for i in range(10):
        expect(f"SENDER_CH_{i}_IS_WORKER_CHANNEL", 1 if i in workers else 0)
        if i < 2 * n:
            expect(f"SENDER_CH_{i}_IS_INJECTION", 1 if ring and i in workers else 0)
    for r in range(3):
        expect(f"RX_CH_{r}_FORWARDS_ON_VC", r if r < n else 0)
    expect("NUM_TRANSACTION_IDS_PER_RECEIVER_CHANNEL", ids_per_receiver_channel(n, bool(a.get("ENABLE_DEADLOCK_AVOIDANCE"))))
    if n >= 2:
        expect("VC1_FABRIC_POSITION_START", 2)
    if n == 3:
        expect("VC2_FABRIC_POSITION_START", 4)
    streams = [a.get(f"SENDER_CHANNEL_{i}_FREE_SLOTS_STREAM_ID") for i in range(2 * n)]
    streams += [a.get(f"TO_RECEIVER_{r}_PKTS_SENT_ID") for r in range(n)]
    if None in streams or len(set(streams)) != len(streams) or max(streams) > 31:
        fail(f"{where}: sender/receiver stream ids not distinct or out of range: {streams}")
    opens = [a.get(f"OPEN_DOWNSTREAM_VC{vc}_CONNECTIONS") for vc in range(n)]
    if len(set(opens)) != 1:
        fail(f"{where}: line VCs open their downstreams unequally: {opens}")
    downstreams = [a.get(f"NUM_DOWNSTREAM_SENDERS_VC{vc}") for vc in range(n)]
    if len(set(downstreams)) != 1:
        fail(f"{where}: line VCs have unequal downstream counts: {downstreams}")
    return opens[0]


def check_case(case):
    m = re.search(r"_lvc(\d)", case)
    n, ring = int(m.group(1)), "_ring" in case
    path = f"{COV}/argdump/{case}.jsonl"
    failures = []
    if not os.path.exists(path):
        return f"{case}: no arg dump", ["missing " + path]
    recs = [json.loads(l) for l in open(path) if l.strip()]
    opened = collections.Counter()
    for rec in recs:
        opened[check_record(rec, n, ring, failures.append)] += 1
    log = f"{COV}/cases/{case}/mock.log"
    rows = set()
    if os.path.exists(log):
        text = re.sub(r"\x1b\[[0-9;]*m", "", open(log, errors="replace").read())
        rows = set(re.findall(r"VC(\d) sender slots \{(\d+), (\d+)\} receiver slots \{(\d+)\}", text))
    want = SLOT_ROWS[n]
    if want is not None:
        # The router config's rows; the builder's per-direction MUX-mode configs (unused here) log rows with a 0 depth.
        got = {(int(w), int(f), int(r)) for vc, w, f, r in rows if int(vc) < n and 0 not in (int(w), int(f), int(r))}
        if got != {want}:
            failures.append(f"slot rows {sorted(got)}, expected {want} for each line VC")
    routers = len({(r["device"], r["eth_chan"]) for r in recs})
    summary = (f"{case}: N={n} {'ring' if ring else 'line'}, {len(recs)} router ERISCs on {routers} routers, "
               f"downstreams opened {dict(opened)}, slot row {want or 'as today'}: "
               f"{'all checks pass' if not failures else str(len(failures)) + ' failures'}")
    return summary, failures


def main():
    cases = sys.argv[1:] or sorted(os.path.basename(p)[:-6] for p in glob.glob(f"{COV}/argdump/*_lvc*.jsonl"))
    bad = 0
    for case in cases:
        summary, failures = check_case(case)
        print(summary)
        for f in failures[:10]:
            print("   ", f)
        bad += bool(failures)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
