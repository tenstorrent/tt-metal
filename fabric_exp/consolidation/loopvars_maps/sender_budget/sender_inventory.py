"""sender_inventory.py BUILD: per coverage case, E0's sender loop (two ERISCs: sender::run's pass loop; one ERISC:
single::sender_pass): carried values (live at the pass loop's head) by class, the peak, registers vs stack slots;
from loopvars.liveness with role sender."""
import sys, os, re, glob, json, collections, concurrent.futures
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import loopvars as L
CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"
OUT = os.path.dirname(os.path.abspath(__file__))

def classify(v):
    n = " ".join(v["names"]); o = v["origin"]
    if re.search(r"workers\.", n):
        m = re.search(r"workers\.(vc\d|first|rest[\w.]*)\.?(\w+)?", n)
        return "worker channel state", n.split(",")[0][:40]
    if re.search(r"forwarding\.records|forwarding\.", n): return "forwarding state (memory by design)", "record"
    if re.search(r"far\.", n): return "remote receiver", n.split(",")[0][:40]
    if "in_flight" in n: return "in_flight", "in_flight"
    if re.search(r"round\.", n): return "round (memory by design)", n.split(",")[0][:40]
    if re.search(r"connections|interface", n): return "worker interfaces (memory by design)", "iface"
    if re.search(r"pending|semaphores|attempted", n): return "pass", n.split(",")[0][:40]
    if o.startswith("invariant"): return "invariant (constant/pointer)", o[11:60]
    if re.search(r"run router_sender.hpp:7\d\d", o) and "-1" in o: return "pass counter", "n"
    if re.search(r"noc_nonblocking_api|eth_txq_reg_write|tt_eth_api", o): return "promoted / temporaries (NoC, TXQ)", o[:50]
    if n: return "other named", n[:50]
    return "other", o[:60]

def one(path):
    names = L.sh([L.B + "nm", path])
    if "6sender3run" not in names and "6single11sender_pass" not in names:
        return None
    try:
        r = L.liveness(path, "sender")
    except Exception as e:
        return (path, {"error": str(e)[:200]})
    head = collections.Counter(); headr = collections.Counter()
    for v in r["head"]["values"]:
        c, d = classify(v); head[c] += 1
        if not v["loc"].startswith("sp+"):
            headr[c] += 1
    return (path, {"pass": r["pass_insns"], "carried": r["head"]["live"], "carried_regs": r["head"]["regs"],
                   "carried_slots": r["head"]["slots"], "peak": r["peak"]["live"], "peak_regs": r["peak"]["regs"],
                   "peak_slots": r["peak"]["slots"], "peak_line": r["peak"]["line"], "classes": dict(head),
                   "classes_regs": dict(headr),
                   "head_values": [(v["loc"], classify(v)[0], " ".join(v["names"])[:60]) for v in r["head"]["values"]]})

if __name__ == "__main__":
    b = sys.argv[1]
    paths = sorted(glob.glob(f"{CB}/{b}/*/active_erisc/active_erisc.elf") +
                   [p for p in glob.glob(f"{CB}/{b}/*/subordinate_active_erisc/subordinate_active_erisc.elf") if "1erisc" in p or "mux" in p or "udm" in p])
    with concurrent.futures.ProcessPoolExecutor(16) as ex:
        res = [x for x in ex.map(one, paths) if x]
    out = {os.path.relpath(p, f"{CB}/{b}"): r for p, r in res}
    json.dump(out, open(f"{OUT}/inv_{b}.json", "w"), indent=0)
    fams = collections.defaultdict(list)
    for rel, r in out.items():
        if "error" not in r:
            fams[rel.split("__")[0]].append(r)
        else:
            print("ERR", rel, r["error"])
    print(f"== {b}: {len(out)} sender images")
    print(f"{'case':28} {'n':>3} {'pass':>9} {'carried regs':>13} {'carried slots':>14} {'peak regs':>10} {'peak slots':>11}  carried registers by class")
    def rng(xs): return f"{min(xs)}-{max(xs)}" if min(xs) != max(xs) else f"{min(xs)}"
    for f, rs in sorted(fams.items()):
        cls = collections.defaultdict(list)
        for r in rs:
            for c in set(list(r["classes_regs"])):
                pass
        allc = sorted({c for r in rs for c in r["classes_regs"]})
        cs = ", ".join(f"{c.split(' (')[0]} {rng([r['classes_regs'].get(c, 0) for r in rs])}" for c in allc)
        print(f"{f:28} {len(rs):3} {rng([r['pass'] for r in rs]):>9} {rng([r['carried_regs'] for r in rs]):>13} {rng([r['carried_slots'] for r in rs]):>14} {rng([r['peak_regs'] for r in rs]):>10} {rng([r['peak_slots'] for r in rs]):>11}  {cs}")
