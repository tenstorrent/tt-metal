"""Parse a raw NoC-event trace (no tt-npe needed) and summarize read/write traffic by NoC, proc, and
DRAM bank -- to see injection balance for the large-input read. Capture the trace with:
  TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 \
  TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH=<dir>  python <run-one-op>
then: python tools/mm_sweep/parse_noc_trace.py <dir>/noc_trace_dev0_ID*.json
"""
import json, sys, collections

evs = json.load(open(sys.argv[1]))
for kind in ("READ", "WRITE"):
    es = [e for e in evs if e.get("type") == kind]
    if not es:
        continue
    tot = sum(e["num_bytes"] for e in es)
    print(f"\n=== {kind}: {len(es)} events, {tot/1e6:.2f} MB ===")
    byn = collections.Counter()
    byp = collections.Counter()
    byb = collections.Counter()
    for e in es:
        byn[e["noc"]] += e["num_bytes"]
        byp[(e["proc"], e["noc"])] += e["num_bytes"]
        byb[(e["dx"], e["dy"])] += e["num_bytes"]
    print("  by NoC:", {k: f"{v/1e6:.2f}MB ({100*v/tot:.0f}%)" for k, v in byn.most_common()})
    print("  by proc/NoC:", {f"{k[0]}/{k[1]}": f"{v/1e6:.2f}MB" for k, v in byp.most_common()})
    print(f"  dest banks: {len(byb)} banks, " f"max {max(byb.values())/1e6:.2f}MB / min {min(byb.values())/1e6:.2f}MB")
