"""inventory.py BUILD [BUILD...]: per coverage build and config family, E1's receiver loop: carried values (live at the
pass loop's head) by class, the peak of live values, and stack slots in use; from loopvars.liveness."""
import sys, os, re, glob, json, collections, concurrent.futures
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import loopvars as L
CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"

def classify(v):
    n = " ".join(v["names"]); o = v["origin"]
    if re.search(r"cursor|position|counts|tables\.ch|records", n):
        m = re.search(r"(ready|available|slot_addr|slot\b|index|uncredited|uncompleted|idle_passes\w*)", n)
        return "channel state", (m.group(1) if m else n)
    if "frame_due" in n: return "pass: frame due", "credit_frame_due"
    if "txq_status" in n or "txq" in n: return "pass: TX-queue status", "txq_status"
    if "arrived" in n: return "pass: arrived", "arrived"
    if "downstreams" in n or re.search(r"\bme\b|me\.", n) or "routers" in n: return "invariant (pointer/coords)", n.split(",")[0]
    if o.startswith("invariant"): return "invariant (constant/pointer)", o[11:60]
    if "router_receiver.hpp:7" in o and "-1" in o: return "pass counter", "n"
    if re.search(r"noc_nonblocking_api|count_local_writes|router_local_delivery|router_downstream|accumulate_first_level_acks|next_ring_slot", o):
        return "promoted memory (NoC counters / downstream cursors)", re.sub(r".*\((\w+) ([\w.]+:\d+)\).*", r"\1 \2", o)[:50]
    if n: return "other named", n[:50]
    return "other", o[:60]

def one(path):
    names = L.sh([L.B + "nm", path])
    if "8receiver3run" not in names and "6single13receiver_pass" not in names:
        return None
    try:
        r = L.liveness(path, "receiver")
    except Exception as e:
        return (path, {"error": str(e)})
    head = collections.Counter(); kinds = collections.Counter()
    for v in r["head"]["values"]:
        c, d = classify(v); head[c] += 1
        kinds[(c, v["kind"])] += 1
    return (path, {"pass": r["pass_insns"], "carried": r["head"]["live"], "carried_regs": r["head"]["regs"],
                   "carried_slots": r["head"]["slots"], "peak": r["peak"]["live"], "peak_regs": r["peak"]["regs"],
                   "peak_slots": r["peak"]["slots"], "peak_line": r["peak"]["line"], "classes": dict(head),
                   "kinds": {f"{a}|{b}": n for (a, b), n in kinds.items()},
                   "head_values": [(v["loc"], classify(v)) for v in r["head"]["values"]]})

def family(rel):
    case = rel.split("__")[0]
    return case

if __name__ == "__main__":
    out = {}
    for b in sys.argv[1:]:
        paths = sorted(glob.glob(f"{CB}/{b}/*/subordinate_active_erisc/subordinate_active_erisc.elf"))
        with concurrent.futures.ProcessPoolExecutor(16) as ex:
            res = [x for x in ex.map(one, paths) if x]
        out[b] = {os.path.relpath(p, f"{CB}/{b}"): r for p, r in res}
        json.dump(out, open(f"/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/budget/inv_{'_'.join(sys.argv[1:])}.json", "w"))
        fams = collections.defaultdict(list)
        for rel, r in out[b].items():
            if "error" not in r:
                fams[family(rel)].append(r)
        print(f"== {b}: {len(out[b])} E1 images")
        for f, rs in sorted(fams.items()):
            cl = collections.defaultdict(list)
            for r in rs:
                for c, n in r["classes"].items():
                    cl[c].append(n)
            rng = lambda xs: f"{min(xs)}-{max(xs)}" if min(xs) != max(xs) else f"{min(xs)}"
            print(f"  {f:28s} n={len(rs):3d} carried {rng([r['carried'] for r in rs])} (regs {rng([r['carried_regs'] for r in rs])}, slots {rng([r['carried_slots'] for r in rs])})"
                  f"  peak {rng([r['peak'] for r in rs])} (regs {rng([r['peak_regs'] for r in rs])}, slots {rng([r['peak_slots'] for r in rs])})")
            print("      " + "; ".join(f"{c} {rng(v)}" for c, v in sorted(cl.items())))
