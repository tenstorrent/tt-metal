"""ffstatic.py BASE NEW: per E1 image (and one-ERISC image) of two coverage builds: the receiver's pass loop (two
ERISCs: receiver::run's largest strongly connected region with the round's end_round call removed, i.e. the passes
only; one ERISC: single::receiver_pass) and the whole round loop, instructions and sp accesses, grouped by whether the
config has a first-level-ack receiver channel. Also single::run's size on one-ERISC images."""
import sys, os, re, glob, collections, concurrent.futures
sys.path.insert(0, "/home/aagarwal/tt-metal/fabric_exp/consolidation")
import loopvars as L
CB = "/home/aagarwal/.tt-metal-v3r-builds/cov_m/builds"

def fla(cfgdir):
    t = open(f"{cfgdir}/named_ct_arg_map_generated.h").read()
    a = dict((k, int(v)) for k, v in re.findall(r'\{"(\w+)",(-?\d+)\}', t))
    return any(a.get(f"ENABLE_FIRST_LEVEL_ACK_VC{v}", 0) and a.get(f"IS_RECEIVER_CHANNEL_{v}_SERVICED", 0) for v in range(3))

def sp_count(body, region):
    return sum(1 for a, op, raw in body if a in region and re.fullmatch(r"\w+,-?\d+\(sp\)", raw) and (op in L.LOADS or op in L.STORES))

def one(job):
    b, rel = job
    e = f"{CB}/{b}/{rel}"
    fns = L.functions(e)
    syms = {}
    for l in L.sh([L.B + "nm", e]).splitlines():
        p = l.split()
        if len(p) == 3:
            syms[int(p[0], 16)] = p[2]
    out = {}
    run = [n for n in fns if "8receiver3run" in n]
    if run:
        body = fns[run[0]]
        addrs = [a for a, *_ in body]
        succ, _ = L.cfg(body)
        rounds = L.largest_scc(addrs, succ)
        cut = {a for a, op, raw in body if op == "jal" and "end_round" in syms.get(int(raw.split(",")[-1], 16), "")}
        succ2 = {a: ([] if a in cut else [x for x in s if x not in cut]) for a, s in succ.items()}
        passes = L.largest_scc([a for a in addrs if a not in cut], succ2)
        out = {"round": (len(rounds), sp_count(body, rounds)), "pass": (len(passes), sp_count(body, passes)),
               "fn": len(body)}
    sp = [n for n in fns if "6single13receiver_pass" in n]
    if sp:
        body = fns[sp[0]]
        out = {"pass": (len(body), sp_count(body, {a for a, *_ in body})), "fn": len(body), "single": True}
        sr = [n for n in fns if "6single3run" in n]
        out["round"] = (len(fns[sr[0]]), 0) if sr else (0, 0)
    return (b, rel, out) if out else None

base, new = sys.argv[1], sys.argv[2]
rels = sorted(os.path.relpath(p, f"{CB}/{new}") for p in glob.glob(f"{CB}/{new}/*/*/*.elf"))
rels = [r for r in rels if os.path.exists(f"{CB}/{base}/{r}")]
with concurrent.futures.ProcessPoolExecutor(16) as ex:
    res = [x for x in ex.map(one, [(b, r) for r in rels for b in (base, new)]) if x]
by = collections.defaultdict(dict)
for b, rel, o in res:
    by[rel][b] = o
groups = collections.defaultdict(collections.Counter)
ex1 = {}
for rel, d in by.items():
    if base not in d or new not in d:
        continue
    cfg = os.path.dirname(os.path.dirname(f"{CB}/{new}/{rel}"))
    kind = ("one ERISC (pass = receiver_pass, round = single::run size)" if d[new].get("single")
            else "two ERISCs") + (" FLA" if fla(cfg) else " no-FLA")
    o0, o1 = d[base], d[new]
    key = (o1["pass"][0] - o0["pass"][0], o1["pass"][1] - o0["pass"][1], o1["round"][0] - o0["round"][0],
           o1["round"][1] - o0["round"][1], o1["fn"] - o0["fn"])
    groups[kind][key] += 1
    ex1.setdefault((kind, key), rel)
for kind, c in sorted(groups.items()):
    print(f"== {kind}: {sum(c.values())} images; deltas NEW - BASE: pass-loop insns / sp, round-loop insns / sp, function insns")
    for k, n in c.most_common(40):
        print(f"  {n:4d}  pass {k[0]:+d} / {k[1]:+d}   round {k[2]:+d} / {k[3]:+d}   fn {k[4]:+d}   e.g. {ex1[(kind, k)]}")
