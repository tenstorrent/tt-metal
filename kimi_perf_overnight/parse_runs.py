#!/usr/bin/env python3
# Per-RUN op2op + kernel summary for the MLA profiling test, segmented by mla_run_{it}_start/_end
# signposts. Run 0 (cold/compile) is SKIPPED. For each warm run it reports, on the SLOWEST device
# (chosen by total kernel across all warm runs), the op-kernel time and the op-to-op (dispatch) gap:
#   op2op_raw    = signed sum (negatives = cross-op device overlap, e.g. a disjoint-core op the
#                  dispatcher front-runs so its start precedes the prior op's end)
#   op2op_pos    = negatives clamped to 0 -> honest dispatch-idle total
# Also writes a per-op breakdown of the LAST warm run (same format as 9L_with_service_sdpafix_log.new).
# Usage: parse_runs.py <ops_csv> <label> [perop_out.log]
import csv, sys, re
from collections import defaultdict

ops_csv = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else ops_csv
perop_out = sys.argv[3] if len(sys.argv) > 3 else None


def fnum(s):
    try:
        return float(s)
    except Exception:
        return 0.0


rows = []
start_sig = {}
end_sig = {}
re_s = re.compile(r"^mla_run_(\d+)_start$")
re_e = re.compile(r"^mla_run_(\d+)_end$")
with open(ops_csv) as f:
    for i, r in enumerate(csv.DictReader(f)):
        oc = (r.get("OP CODE") or "").strip()
        ms = re_s.match(oc)
        me = re_e.match(oc)
        if ms:
            start_sig[int(ms.group(1))] = i
            continue
        if me:
            end_sig[int(me.group(1))] = i
            continue
        try:
            dev = int(r["DEVICE ID"])
        except Exception:
            continue
        if not oc:
            continue
        rows.append(
            (
                i,
                oc,
                dev,
                fnum(r.get("DEVICE KERNEL DURATION [ns]")),
                fnum(r.get("DEVICE FW DURATION [ns]")),
                fnum(r.get("OP TO OP LATENCY [ns]")),
                (r.get("CORE COUNT") or "").strip(),
            )
        )

runs = sorted(k for k in start_sig if k in end_sig)
if not runs:
    sys.stderr.write("no mla_run_* signposts found\n")
    sys.exit(1)
warm = [it for it in runs if it >= 1]  # skip run 0 (compile)
if not warm:
    sys.stderr.write("only a compile run found (need >=2 iters)\n")
    sys.exit(1)


def run_ops(it):
    lo, hi = start_sig[it], end_sig[it]
    return [x for x in rows if lo < x[0] < hi]


# slowest device = max total kernel across all warm runs
ks = defaultdict(float)
for it in warm:
    for x in run_ops(it):
        ks[x[2]] += x[3]
slow = max(ks, key=ks.get)

print(f"# PER-RUN MLA profile — {label}")
print(f"#   slowest DEVICE ID = {slow}; runs profiled = {warm} (compile run 0 skipped)")
print(f"#   'boundary_ms' = op2op of each run's FIRST op (the inter-run synchronize_device + host input")
print(f"#   upload that segments the runs — a harness artifact, NOT intra-run dispatch). 'op2op_intra' =")
print(f"#   op2op summed over the run's remaining ops (the real within-run dispatch gap); _pos clamps")
print(f"#   negatives (cross-op overlap) to 0.")
print(
    f"{'run':>4} {'nops':>5} {'kernel_ms':>10} {'boundary_ms':>12} {'op2op_intra_raw':>16} "
    f"{'op2op_intra_pos':>16} {'pos%kernel':>11}"
)
tk = tr = tp = tb = 0.0
nwarm = 0
for it in warm:
    ops = [x for x in run_ops(it) if x[2] == slow]
    if not ops:
        continue
    k = sum(x[3] for x in ops)
    boundary = ops[0][5]  # first op = inter-run sync/host-upload gap (artifact)
    intra = ops[1:]
    o_raw = sum(x[5] for x in intra)
    o_pos = sum(x[5] for x in intra if x[5] > 0)
    pct = 100.0 * o_pos / k if k else 0.0
    print(
        f"{it:>4} {len(ops):>5} {k/1e6:>10.3f} {boundary/1e6:>12.3f} {o_raw/1e6:>16.3f} {o_pos/1e6:>16.3f} {pct:>10.1f}%"
    )
    tk += k
    tr += o_raw
    tp += o_pos
    tb += boundary
    nwarm += 1
print("-" * 80)
if nwarm:
    print(
        f"{'mean':>4} {'':>5} {tk/nwarm/1e6:>10.3f} {tb/nwarm/1e6:>12.3f} {tr/nwarm/1e6:>16.3f} "
        f"{tp/nwarm/1e6:>16.3f} {100.0*tp/tk if tk else 0:>10.1f}%"
    )
    print(f"{'tot':>4} {'':>5} {tk/1e6:>10.3f} {'':>12} {tr/1e6:>16.3f} {tp/1e6:>16.3f}   (over {nwarm} warm runs)")

# Per-op breakdown of ALL warm runs (same format as the *_log.new files), compile run 0 excluded.
if perop_out:
    lines = [
        f"# MLA per-op profile, SLOWEST device — {label}",
        f"# slowest DEVICE ID={slow}; warm runs {warm} (compile run 0 excluded)",
    ]
    for it in warm:
        ops = [x for x in run_ops(it) if x[2] == slow]
        ks2 = sum(x[3] for x in ops) / 1e6
        fs2 = sum(x[4] for x in ops) / 1e6
        o2 = sum(x[5] for x in ops) / 1e6
        o2x = sum(x[5] for x in ops[1:]) / 1e6  # excluding the boundary (first) op
        lines.append(
            f"\n## warm run {it}: {len(ops)} ops  kernel_sum={ks2:.3f}ms fw_sum={fs2:.3f}ms "
            f"op2op_sum={o2:.3f}ms op2op_intra={o2x:.3f}ms (first-op boundary={ops[0][5]/1e3:.1f}us)"
        )
        lines.append(f"  {'idx':>3} {'OP CODE':42s} {'KERNEL_us':>10} {'FW_us':>10} {'OP2OP_us':>9} {'cores':>6}")
        for j, x in enumerate(ops):
            lines.append(f"  {j:3d} {x[1][:42]:42s} {x[3]/1e3:10.2f} {x[4]/1e3:10.2f} {x[5]/1e3:9.2f} {x[6]:>6}")
    open(perop_out, "w").write("\n".join(lines) + "\n")
    print(f"# wrote per-op breakdown of ALL {len(warm)} warm runs -> {perop_out}")
