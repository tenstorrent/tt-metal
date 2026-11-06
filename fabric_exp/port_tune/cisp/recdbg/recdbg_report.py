#!/usr/bin/env python3
"""recdbg_report.py RESULTS_DIR: the recorder ring (cisp/recdbg: per sender ERISC and iteration, at L1 0x60000) from
cisp_recdbg.sh's captures, lined up with the run's telemetry. Per router, the ring's entries are the iterations in which
it had a counted round, in order; they are matched to the log's telemetry rows with packets_sent > 0 and checked
(entry busy == the row's cycles). Stall events: rows more than 300k cycles above the off arm's median for the same test
and router (cisp_router_map.py's rule). Prints, for event router-iterations against quiet ones (on arm), each field."""
import collections, json, os, re, statistics, struct, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cisp_router_map as rmap  # noqa: E402

D = sys.argv[1]
RING = 0x60000
FIELDS = ["busy", "max_wall", "max_ts", "max_start", "max_end", "first_send", "zero_tx_sum", "max_gap", "max_tx",
          "max_held", "max_fw", "zero_tx_rounds", "counted", "backwards"] + [f"free{i}" for i in range(7)] + \
         ["huge_steps", "seq", "magic"]
FMT = "<8Q6I7I3I"
assert struct.calcsize(FMT) == 128
TEL = re.compile(r"Telemetry from MeshCoordinate\(\[(\d+), (\d+)\]\) core 0-(\d+): .*cycles=(\d+), eth_words_sent=\d+, "
                 r"packets_sent=(\d+)")


def rows(log):
    """[(test, iteration, router, cycles, packets)] in log order."""
    out, cur, k = [], None, collections.Counter()
    for line in open(log, errors="replace"):
        if m := rmap.RUN.search(line):
            cur = m.group(1)
            k[cur] += 1
        elif cur and (m := TEL.search(line)):
            r, c, ch, cyc, pk = map(int, m.groups())
            out.append((cur, k[cur] - 1, ((r, c), ch), cyc, pk))
    return out


def ring(cap):
    idx = json.load(open(os.path.join(cap, "index.json")))
    out = {}
    for chip, crec in idx["chips"].items():
        for chan in crec["cores"]:
            f = os.path.join(cap, f"chip{chip}", f"eth{int(chan):02d}_l1.bin")
            if not os.path.exists(f):
                continue
            b = open(f, "rb").read()
            n, magic = struct.unpack_from("<II", b, RING)
            if magic != 0x5EC0DB61:
                continue
            ents = []
            for i in range(1, n + 1):
                e = dict(zip(FIELDS, struct.unpack_from(FMT, b, RING + 128 * i)))
                ents.append(e)
            out[(int(chip), int(chan))] = ents
    return out


def main():
    peer, c2chip = rmap.topology()
    data = {}
    for arm in ("on", "off"):
        rs = rows(os.path.join(D, f"recdbg_{arm}.log"))
        rg = ring(os.path.join(D, f"recdbg_{arm}_cap"))
        per = collections.defaultdict(list)
        for t, k, rt, cyc, pk in rs:
            if pk > 0:
                per[rt].append((t, k, cyc, pk))
        matched, ok, bad = {}, 0, 0
        for rt, its in per.items():
            (r, c), ch = rt
            ents = rg.get((c2chip[(r, c)], ch), [])
            for (t, k, cyc, pk), e in zip(its, ents):
                if e["magic"] != 0x5EC0DB61:
                    bad += 1
                    continue
                if e["busy"] == cyc:
                    ok += 1
                else:
                    bad += 1
                matched[(t, k, rt)] = (cyc, e)
            if len(ents) != len(its):
                print(f"   {arm} {rt}: {len(its)} telemetry rows with packets, {len(ents)} ring entries")
        print(f"== {arm}: {len(rs)} telemetry rows, {sum(len(v) for v in per.values())} with packets; ring cores "
              f"{len(rg)}; entries matched {len(matched)}, busy == cycles {ok}, mismatched {bad}")
        data[arm] = (rs, matched)
    off_rows = collections.defaultdict(list)
    for t, k, rt, cyc, pk in data["off"][0]:
        if pk > 0:
            off_rows[(t, rt)].append(cyc)
    ev, qu = [], []
    for (t, k, rt), (cyc, e) in data["on"][1].items():
        if (t, rt) not in off_rows:
            continue
        excess = cyc - statistics.median(off_rows[(t, rt)])
        (ev if excess > 300e3 else qu).append((excess, e, t, k, rt))
    print(f"\n== on arm: {len(ev)} event router-iterations (> 300k over the off median), {len(qu)} quiet")

    def stat(xs, f):
        v = [f(e) for _, e, *_ in xs]
        return f"median {statistics.median(v):>12.0f}  mean {statistics.mean(v):>12.1f}  max {max(v):>12.0f}" if v else "-"
    checks = [
        ("excess over off median", lambda e: 0),
        ("longest round: wall delta", lambda e: e["max_wall"]),
        ("longest round: get_timestamp delta", lambda e: e["max_ts"]),
        ("longest round: wall - timestamp", lambda e: e["max_wall"] - e["max_ts"]),
        ("longest round: tx_packets", lambda e: e["max_tx"]),
        ("longest round: any_held", lambda e: e["max_held"]),
        ("longest round: fw_ran", lambda e: e["max_fw"]),
        ("zero-tx counted rounds", lambda e: e["zero_tx_rounds"]),
        ("zero-tx counted rounds: summed wall", lambda e: e["zero_tx_sum"]),
        ("largest step between wall reads", lambda e: e["max_gap"]),
        ("backwards wall reads", lambda e: e["backwards"]),
        ("steps > 2^31", lambda e: e["huge_steps"]),
        ("counted rounds", lambda e: e["counted"]),
        ("first send - longest round start", lambda e: (e["first_send"] - e["max_start"]) if e["first_send"] else 0),
    ]
    for name, f in checks[1:]:
        print(f"   {name:38s} events: {stat(ev, f)}")
        print(f"   {'':38s} quiet:  {stat(qu, f)}")
    if ev:
        ex = [x for x, *_ in ev]
        print(f"   excess (events): median {statistics.median(ex):.0f}, mean {statistics.mean(ex):.0f}")
        print("   events, first 15: test it router | excess | longest round wall / ts | tx held fw | zero-tx rounds, sum |"
              " first_send - longest start")
        for x, e, t, k, rt in sorted(ev, key=lambda z: -z[0])[:15]:
            print(f"     {re.sub(r'NeighborExchangeUnicast', 'NX', t)[:52]} it{k + 1} {rt} | {x:.0f} | {e['max_wall']} / "
                  f"{e['max_ts']} | {e['max_tx']} {e['max_held']} {e['max_fw']} | {e['zero_tx_rounds']}, {e['zero_tx_sum']} |"
                  f" {(e['first_send'] - e['max_start']) if e['first_send'] else '-'}")


if __name__ == "__main__":
    main()
