#!/usr/bin/env python3
"""ccl_matrix_report.py [PASS ...]: op time (us) per CCL / num_links / max payload, main_fw against each v4 router,
from the summaries ccl_matrix_runs.sh leaves in fabric_exp/results/ccl/<router>/<pass>/. Mean over the passes given
(default a b), each pass's value in brackets."""
import glob, os, re, sys, collections
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "ccl")
passes = sys.argv[1:] or ["a", "b"]
data = collections.defaultdict(dict)  # (ccl, config, payload) -> {(router, pass): us}
# Rows come from each summary's "Per-run overview" table, read by its header: num_links for the link-sweeping CCLs,
# (num_l_chunks, block_size) for SDPA reduce-to-all, max_payload_size_bytes and op_time_us for all.
for router_dir in sorted(glob.glob(f"{R}/*")):
    router = os.path.basename(router_dir)
    for p in passes:
        for f in sorted(glob.glob(f"{router_dir}/{p}/ccl_trace_perf_summary_*.md")):
            ccl, header, in_overview = None, None, False
            for line in open(f):
                m = re.match(r"## (\w+)", line)
                if m:
                    ccl, header, in_overview = m.group(1), None, False
                    continue
                if line.startswith("### "):
                    in_overview, header = line.strip() == "### Per-run overview", None
                    continue
                if not (in_overview and line.startswith("|")):
                    continue
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if header is None:
                    header = cells
                    continue
                if set("".join(cells)) <= set("- "):
                    continue
                row = dict(zip(header, cells))
                if "num_l_chunks" in row:
                    config = f"l{row['num_l_chunks']},b{row['block_size']}"
                else:
                    config = row.get("num_links", "")
                try:
                    data[(ccl, config, int(row["max_payload_size_bytes"]))][(router, p)] = float(row["op_time_us"])
                except (KeyError, ValueError):
                    continue
routers = sorted({r for v in data.values() for (r, _) in v}, key=lambda r: (r != "main_fw", r))
print(f"{'ccl':20} {'config':>6} {'payload':>7} " + " ".join(f"{r:>22}" for r in routers) + "   v4/main")
for k in sorted(data):
    row, means = [], {}
    for r in routers:
        vals = [data[k][(r, p)] for p in passes if (r, p) in data[k]]
        if vals:
            means[r] = sum(vals) / len(vals)
            row.append(f"{means[r]:8.3f} ({'/'.join(f'{v:.3f}' for v in vals)})".rjust(22))
        else:
            row.append(" " * 22)
    ratios = [f"{means[r] / means['main_fw']:.3f}" for r in routers if r != "main_fw" and r in means and "main_fw" in means]
    print(f"{k[0]:20} {k[1]:>6} {k[2]:>7} " + " ".join(row) + "   " + " ".join(ratios))
