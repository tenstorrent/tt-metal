#!/usr/bin/env python3
"""Parse SPEC_SHADOW lines of a server log (phase 2 step S8) and print the top-p certificate failure rates.

Each DP worker process logs cumulative counters (prefix SPEC_SHADOW, key=value pairs, list values joined with "/"). This script keeps
the LAST line per pid (the counters are cumulative), sums the workers and prints, for each p in the line's ps list:
  - position fail rate  = failpos / pos           (examined verify positions, q < p + eps)
  - row fail rate       = failrow / rows          (a live sampled row-step with any failing examined position)
  - step fail rate      = failstep / steps        (a probe step with any failing row; overstates a lone special row)
for q_T (the certificate at the row temperature) and q_1 (T = 1). Then the estimated tok/s of a normal row under failure policy (a)
(work/research_1026/67 section 5.5): step = 48 + 45 f ms, 2.4 tokens per step, against 41.65 ms plain. f = the row fail rate.

Usage: shadow_parse.py LOGFILE [LOGFILE ...]     (or "-" for stdin)
"""

import re
import sys

PREFIX = "SPEC_SHADOW"
TOKEN = re.compile(r"(\S+?)=(\S*)")
LIST_KEYS = {"edges", "ps"}  # float lists; the other "/" lists are int lists
TOK_PER_STEP = 2.4
BASE_MS, FAIL_MS, PLAIN_MS = 48.0, 45.0, 41.65


def parse_line(line):
    """dict of one SPEC_SHADOW line (key -> int / float / list), or None when the line has no prefix."""
    i = line.find(PREFIX + " ")
    if i < 0:
        return None
    out = {}
    for k, v in TOKEN.findall(line[i + len(PREFIX) :]):
        if "/" in v or k in LIST_KEYS:
            out[k] = [float(x) if k in LIST_KEYS else int(x) for x in v.split("/") if x != ""]
        else:
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = float(v)
    return out if "pid" in out else None


def last_per_pid(lines):
    """pid -> the last parsed line (cumulative counters)."""
    last = {}
    for line in lines:
        d = parse_line(line)
        if d is not None:
            last[d["pid"]] = d
    return last


def _add(a, b):
    return [x + y for x, y in zip(a, b)] if isinstance(a, list) else a + b


SUM_SCALAR = ("steps", "rows", "pos", "allpos")
SUM_LISTS = (
    "hist_qT",
    "hist_q1",
    "failpos_qT",
    "failpos_q1",
    "failrow_qT",
    "failrow_q1",
    "failstep_qT",
    "failstep_q1",
    "allfail_qT",
)


def combine(workers):
    """Sum the counters of the workers (dict pid -> line). The per-temperature keys T<x>_* are summed too."""
    tot = {}
    for d in workers.values():
        for k, v in d.items():
            if k in ("pid", "every", "M", "eps", "edges", "ps"):
                tot.setdefault(k, v)
            elif k in ("qT_mean", "q1_mean"):
                continue
            else:
                tot[k] = _add(tot[k], v) if k in tot else v
    n = max(tot.get("pos", 0), 1)
    for key in ("qT_mean", "q1_mean"):  # position-weighted mean over the workers
        tot[key] = sum(d.get(key, 0.0) * d.get("pos", 0) for d in workers.values()) / n
    return tot


def tokps(f):
    return TOK_PER_STEP * 1000.0 / (BASE_MS + FAIL_MS * f)


def report(tot, workers=1):
    ps = tot.get("ps", [])
    lines = [
        f"workers={workers} every={tot.get('every')} M={tot.get('M')} probe_steps={tot['steps']} rows={tot['rows']} "
        f"examined_positions={tot['pos']} live_positions={tot['allpos']} mean_qT={tot['qT_mean']:.4f} mean_q1={tot['q1_mean']:.4f}"
    ]
    base = TOK_PER_STEP * 1000.0 / PLAIN_MS
    lines.append(f"plain normal row: {base:.1f} tok/s (step {PLAIN_MS} ms)")
    for kind in ("qT", "q1"):
        lines.append(f"-- {kind} ({'row temperature, L_T' if kind == 'qT' else 'T = 1, L_raw'})")
        lines.append("   p     pos_fail  row_fail  step_fail  est_tok/s(a, f=row_fail)  est_tok/s(a, f=step_fail)")
        for i, p in enumerate(ps):
            pf = tot["failpos_" + kind][i] / max(tot["pos"], 1)
            rf = tot["failrow_" + kind][i] / max(tot["rows"], 1)
            sf = tot["failstep_" + kind][i] / max(tot["steps"], 1)
            lines.append(f"   {p:<5} {pf:8.4f}  {rf:8.4f}  {sf:9.4f}  {tokps(rf):22.1f}  {tokps(sf):24.1f}")
    edges = tot.get("edges", [])
    for kind in ("qT", "q1"):
        h = tot["hist_" + kind]
        n = max(sum(h), 1)
        bins = [
            f"[{edges[i]:g},{edges[i + 1]:g})" if i + 1 < len(edges) else f"[{edges[i]:g},1]" for i in range(len(h))
        ]
        lines.append(f"hist {kind}: " + " ".join(f"{b}:{c / n:.3f}" for b, c in zip(bins, h)))
    temps = sorted({k[1 : k.index("_")] for k in tot if re.match(r"T[\d.]+_pos$|Tother_pos$", k)})
    if temps:
        lines.append("-- per temperature bucket: examined positions, position fail rate qT at each p")
        for t in temps:
            n = tot[f"T{t}_pos"]
            fr = " ".join(f"p{p}:{c / max(n, 1):.4f}" for p, c in zip(ps, tot[f"T{t}_failpos_qT"]))
            lines.append(f"   T={t:<5} pos={n:<8} rows={tot[f'T{t}_rows']:<7} {fr}")
    return "\n".join(lines)


def main(argv):
    lines = []
    for path in argv or ["-"]:
        lines += sys.stdin.read().splitlines() if path == "-" else open(path, errors="replace").read().splitlines()
    workers = last_per_pid(lines)
    if not workers:
        print("no SPEC_SHADOW line found")
        return 1
    print(report(combine(workers), len(workers)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
