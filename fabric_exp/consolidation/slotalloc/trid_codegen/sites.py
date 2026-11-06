import sys, json, re, collections
sys.path.insert(0, sys.argv[1])
from annot import annotate
C = sys.argv[1]; reps = json.load(open(f"{C}/reps.json"))
def site(chain):
    # (category, call-site line) for a per-packet piece of the receiver
    names = [f for f, _ in chain]
    for i, (f, l) in enumerate(chain):
        if f.startswith("local_write<"):
            caller = chain[i + 1] if i + 1 < len(chain) else ("?", "?")
            return f"{f} from deliver line {caller[1].split(':')[-1]}" if caller[0].startswith("deliver") else f"{f} from {caller[0]}"
        if f.startswith("forward<") and "router_downstream" in l:
            return "forward (one downstream)"
        if f.startswith("decode_action") or f == "action":
            return "route decode"
        if f.startswith("accumulate_completion_credit"):
            return "credit accumulate"
        if f.startswith("credit_completions") or f.startswith("oldest_uncredited") or f.startswith("trid_writes_sent"):
            return "credit proof (status read)"
        if f.startswith("of_slot"):
            return "slot ID (of_slot)"
    return None
for key in sys.argv[2:]:
    out = {}
    for arm, elf in zip(("idoff", "id"), reps[key]["elfs"]):
        _, rows = annotate(elf)
        c = collections.defaultdict(lambda: collections.Counter())
        for a, t, ch in rows:
            s = site(ch)
            if not s: continue
            c[s]["insns"] += 1
            op = t.split()[0]
            if "(sp)" in t and op.startswith("l"): c[s]["sp load"] += 1
            if "(sp)" in t and op.startswith("s"): c[s]["sp store"] += 1
            if "STATUS_READ" in " ".join(f for f, _ in ch) and op == "lw" and "(sp)" not in t: c[s]["status lw"] += 1
        out[arm] = c
    print(f"== {key}")
    for s in sorted(set(out["idoff"]) | set(out["id"])):
        a, b = out["idoff"][s], out["id"][s]
        fmt = lambda x: f"{x['insns']} insns" + "".join(f", {k} {x[k]}" for k in ("sp load", "sp store", "status lw") if x[k])
        mark = "" if a == b else "   <-- differs"
        print(f"   {s:50s} idoff {fmt(a):40s} id {fmt(b)}{mark}")
