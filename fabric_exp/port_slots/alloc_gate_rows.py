#!/usr/bin/env python3
"""alloc_gate_rows.py STAGED_DIR TIER_DIR: the staged allocator's gate, rows (SLOT_ALLOCATOR.md). Each DIR holds one
directory per mock case (rc_capture.sh's RC/port, or cov.py's COV/cases) whose logs carry the slot log
(TT_FABRIC_EXP_LOG_SLOTS=1). Per case: the rows each arm logged (VC0 W, F.., R; VC1 F.., R; VC2 W, R; the channel
buffer), the staged arm's against staged_sketch.py's table where the case has a sketch config, and the tier arm's."""
import collections
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../consolidation/slotalloc"))
import staged_sketch as sk  # noqa: E402  (prints its table on import)

CLEAN = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"VC(\d) sender slots \{([^}]*)\} receiver slots \{([^}]*)\} \(channel buffer (\d+) B\)")


def rows(case_dir):
    """The allocator calls of the run: each a dict vc -> (senders, receivers) plus the channel buffer, in log order;
    an allocator's VC lines come together (VC0 first)."""
    calls, cur = collections.Counter(), {}
    for log in glob.glob(os.path.join(case_dir, "**", "*.log"), recursive=True):
        for line in open(log, errors="replace"):
            m = LINE.search(CLEAN.sub("", line))
            if not m:
                continue
            vc = int(m.group(1))
            if vc == 0 and cur:
                calls[tuple(sorted(cur.items()))] += 1
                cur = {}
            cur[vc] = (tuple(int(x) for x in m.group(2).split(",")), tuple(int(x) for x in m.group(3).split(",")),
                       int(m.group(4)))
        if cur:
            calls[tuple(sorted(cur.items()))] += 1
            cur = {}
    return calls


def fmt(call):
    out = []
    for vc, (tx, rx, buf) in call:
        fw = tx[1:] if vc == 0 else tx
        w = str(tx[0]) if vc in (0, 2) else ""
        f = "/".join(map(str, fw)) if len(set(fw)) > 1 else (str(fw[0]) if fw else "")
        if vc == 2:
            w, f = str(tx[0]), ""
        out.append(",".join(x for x in (w, f, ",".join(map(str, rx))) if x))
    return " + ".join(out) + f" ({call[0][1][2]} B)"


def is_mux(call):
    # the Tensix mux path's allocations: only the worker channel has slots
    tx = dict(call)[0][0]
    return len(tx) > 1 and all(x == 0 for x in tx[1:])


def sketch_cell(call):
    vcs = dict(call)
    tx0 = vcs[0][0]
    if len(tx0) < 2:
        return None
    cfg = {"1d": len(tx0) == 2, "vc1": 1 in vcs, "vc2": 2 in vcs, "nf": len(tx0) - 1}
    if 1 in vcs:
        cfg["nf1"] = len(vcs[1][0])
    if cfg["1d"] and (cfg["vc1"] or cfg["vc2"]):
        return None
    # N: every slot the staged arm gave out (its stages always place all N; none here reaches the 127 cap)
    n = sum(sum(tx) + sum(rx) for tx, rx, _ in vcs.values())
    out, f, fs, left = sk.allocate(cfg, n, sk.leftover(cfg))
    return f"{sk.fmt(cfg, out)} (N {n})"


def main():
    staged, tier = sys.argv[1:3]
    cases = sorted({os.path.basename(p) for d in (staged, tier) for p in glob.glob(os.path.join(d, "*")) if os.path.isdir(p)})
    print("\n| case | staged rows (x allocator calls) | sketch | tier rows |")
    print("| --- | --- | --- | --- |")
    for c in cases:
        a, b = rows(os.path.join(staged, c)), rows(os.path.join(tier, c))
        sa = "; ".join(f"{fmt(k)} x{v}" for k, v in sorted(a.items()) if not is_mux(k)) or "-"
        sb = "; ".join(f"{fmt(k)} x{v}" for k, v in sorted(b.items()) if not is_mux(k)) or "-"
        sketch = "; ".join(sorted({s for k in a if not is_mux(k) for s in [sketch_cell(k)] if s})) or "-"
        print(f"| {c} | {sa} | {sketch} | {sb} |")


if __name__ == "__main__":
    main()
