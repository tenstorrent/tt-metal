# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Offline analysis of a device-profiler CSV (profile_log_device.csv) written by test_subdevice_signal_wait.py runs
with TT_METAL_DEVICE_PROFILER=1: per traced replay, every program's FW span (min ZONE_START .. max ZONE_END of the
*-FW zones over all its cores and RISCs), mapped to the variant's program labels; prints median gaps in us.

usage: python subdevice_sync_profile.py <profile_log_device.csv> "<variant spec as passed to --variants>" [--mhz 1350]
       (2-CQ variants: name:2cq:<cq0 order>|<cq1 order>[:rounds]; traces are matched to variants in capture order)
"""
import argparse
import csv
import statistics
from collections import defaultdict


def program_labels(order, nrounds, credit=False):
    labels = []
    for r in range(nrounds):
        for tok in order.split():
            if tok in ("K", "k"):
                labels += [tok + "1", tok + "2", tok + "3"]
            elif tok == "s" and not (credit or r < nrounds - 1):
                continue
            elif tok == "w" and not (credit or r > 0):
                continue
            else:
                labels.append(tok if tok != "Wt" else "W")
    return labels


def load(path):
    progs = defaultdict(lambda: [None, None, set()])  # (chip, trace, counter, rid) -> [start, end, cores]
    with open(path) as f:
        next(f)
        rd = csv.reader(f)
        hdr = [h.strip() for h in next(rd)]
        ix = {h: i for i, h in enumerate(hdr)}
        for row in rd:
            zone = row[ix["zone name"]].strip()
            # start = first KERNEL zone start (the FW zone can open before the go when the launch is preloaded);
            # end = last FW zone end (all RISCs done, just before the core reports completion)
            is_kernel, is_fw = zone.endswith("-KERNEL"), zone.endswith("-FW")
            if not (is_kernel or is_fw):
                continue
            tr = row[ix["trace id"]].strip()
            if tr == "":
                continue
            key = (int(row[ix["PCIe slot"]]), int(tr), int(row[ix["trace id counter"]]), int(row[ix["run host ID"]]))
            t = int(row[ix["time[cycles since reset]"]])
            p = progs[key]
            typ = row[ix["type"]].strip()
            if typ == "ZONE_START" and is_kernel:
                p[0] = t if p[0] is None else min(p[0], t)
            elif typ == "ZONE_END" and is_fw:
                p[1] = t if p[1] is None else max(p[1], t)
            p[2].add((int(row[ix["core_x"]]), int(row[ix["core_y"]])))
    return progs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("variants")
    ap.add_argument("--mhz", type=float, default=1350.0)
    a = ap.parse_args()
    progs = load(a.csv)
    by_chip_trace = defaultdict(lambda: defaultdict(list))  # chip -> trace -> [(counter, rid, st, en, ncores)]
    for (chip, tr, cnt, rid), (st, en, cores) in progs.items():
        if st is not None and en is not None:
            by_chip_trace[chip][tr].append((cnt, rid, st, en, len(cores)))
    specs = []
    for spec in a.variants.split(","):
        p = spec.split(":")
        if p[1] == "2cq":
            prod, cons = p[2].split("|")
            nr = int(p[3]) if len(p) > 3 else 1
            specs.append((p[0], [program_labels(prod, nr, True), program_labels(cons, nr, True)]))
        else:
            nr = int(p[2]) if len(p) > 2 else 1
            specs.append((p[0], [program_labels(p[1], nr)]))
    for chip, traces in sorted(by_chip_trace.items()):
        tids = sorted(traces)
        ti = 0
        for name, label_lists in specs:
            mine = tids[ti : ti + len(label_lists)]
            ti += len(label_lists)
            vals = defaultdict(list)
            nrep = 0
            per_trace = []
            for tid, labels in zip(mine, label_lists):
                by_cnt = defaultdict(list)
                for cnt, rid, st, en, nc in traces[tid]:
                    by_cnt[cnt].append((rid, st, en, nc))
                per_trace.append((labels, by_cnt))
            counters = sorted(set.intersection(*[set(bc.keys()) for _, bc in per_trace])) if per_trace else []
            for cnt in counters:
                rec, seq, ok = {}, defaultdict(list), True
                for labels, by_cnt in per_trace:
                    rows = sorted(by_cnt[cnt])
                    if len(rows) != len(labels):
                        if cnt == counters[0]:
                            print(f"  chip {chip} {name}: counter {cnt}: {len(rows)} programs vs {len(labels)} labels")
                        ok = False
                        break
                    for lab, (rid, st, en, nc) in zip(labels, rows):
                        rec.setdefault(lab, (st, en, nc))
                        seq[lab].append(st)
                if not ok:
                    continue
                nrep += 1
                a_end = rec["A"][1] if "A" in rec else None
                for lab, (st, en, nc) in rec.items():
                    vals[f"dur({lab})"].append((en - st) / a.mhz)
                    if a_end is not None:
                        vals[f"start({lab})-end(A)"].append((st - a_end) / a.mhz)
                        vals[f"end({lab})-end(A)"].append((en - a_end) / a.mhz)
                for lab, sts in seq.items():
                    if len(sts) > 2:
                        vals[f"period({lab})"].append(statistics.median([(b - x) / a.mhz for x, b in zip(sts, sts[1:])]))
            if len(per_trace) == 2:
                # 2 CQs: the streams' replay counters are not aligned (the producer may run a replay ahead), so
                # match each consumer wait to the latest producer A / S that ended before the wait ended.
                import bisect

                def ends(bc, labels, lab):
                    out = []
                    for cnt, rows in bc.items():
                        rows = sorted(rows)
                        if len(rows) == len(labels):
                            out += [en for l, (rid, st, en, nc) in zip(labels, rows) if l == lab]
                    return sorted(out)

                (pl, pbc), (cl, cbc) = per_trace
                a_ends, s_ends = ends(pbc, pl, "A"), ends(pbc, pl, "S")
                for cnt, rows in cbc.items():
                    rows = sorted(rows)
                    if len(rows) != len(cl):
                        continue
                    rec = {l: (st, en) for l, (rid, st, en, nc) in zip(cl, rows)}
                    if "W" not in rec:
                        continue
                    w_end = rec["W"][1]
                    i = bisect.bisect_right(a_ends, w_end) - 1
                    j = bisect.bisect_right(s_ends, w_end) - 1
                    if i < 0 or j < 0:
                        continue
                    vals["2cq end(S)-end(A)"].append((s_ends[j] - a_ends[i]) / a.mhz)
                    vals["2cq end(W)-end(A)"].append((w_end - a_ends[i]) / a.mhz)
                    for lab in ("K1", "W", "s"):
                        if lab in rec:
                            vals[f"2cq start({lab})-end(A)"].append((rec[lab][0] - a_ends[i]) / a.mhz)
            print(f"chip {chip} {name}: traces {mine} replays {nrep}")
            for k in sorted(vals, key=lambda k: (k.split("(")[0], k)):
                print(f"    {k:28s} {statistics.median(vals[k]):9.3f} us")


if __name__ == "__main__":
    main()
