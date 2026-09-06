import json, random, subprocess, sys, time, collections
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/tmp/timer_bench")
from derived import REAL, HOST_BOUND, observe, stats, derived_bounds

RNG = random.Random(1234)
CEIL = 10800
OPS = ["profile", "pcc", "build", "agent"] + sorted(HOST_BOUND)


def op_med(ops, op):
    return max(20, ops["build"] * 0.35) if op in HOST_BOUND else ops[op]


def gen(n):
    out = []
    for i in range(n):
        base, label, ops = RNG.choice(REAL)
        op = RNG.choice(OPS)
        med = op_med(ops, op)
        hist = observe(med, 24, RNG)
        s = stats(hist)
        health = RNG.choice(["healthy", "healthy", "hung_flat", "zombie", "spin"])
        hq = op in HOST_BOUND
        if health == "healthy":
            actual = med * RNG.lognormvariate(0, 0.45)
            in_op = RNG.uniform(0.05, 1.0) * actual
            cpu = [0] * 5 if hq else [int(RNG.uniform(2000, 9000)) for _ in range(5)]
            txt = (
                [int(RNG.uniform(0, 300)) for _ in range(5)]
                if hq
                else [int(RNG.uniform(2000, 40000)) for _ in range(5)]
            )
            # NEW SIGNAL: distinct action hashes in window / total actions
            acts = RNG.randint(3, 9)
            uniq = acts if RNG.random() < 0.9 else max(1, acts - 1)
            truth = "wait"
        elif health == "hung_flat":
            in_op = s["p99"] * RNG.uniform(3, 12)
            cpu = [0] * 5
            txt = [0] * 5
            acts, uniq = 0, 0
            truth = "kill"
        elif health == "zombie":
            in_op = s["p99"] * RNG.uniform(3, 12)
            cpu = [int(RNG.uniform(1, 20))] * 5
            txt = [0] * 5
            acts, uniq = 0, 0
            truth = "kill"
        else:  # spin: log grows, actions REPEAT
            in_op = s["p99"] * RNG.uniform(3, 12)
            cpu = [0] * 5
            txt = [int(RNG.uniform(3000, 9000)) for _ in range(5)]
            acts = RNG.randint(6, 14)
            uniq = 1 if RNG.random() < 0.8 else 2  # same action over and over
            truth = "kill"
        out.append(
            dict(
                id=f"G{i:03d}",
                model=label,
                base=base,
                op=op,
                truth=truth,
                health=health,
                in_op=round(in_op, 1),
                since_commit=round(in_op, 1),
                cpu=cpu,
                txt=txt,
                actions=acts,
                distinct_actions=uniq,
                p50=round(s["p50"], 1),
                p95=round(s["p95"], 1),
                p99=round(s["p99"], 1),
                bounds=derived_bounds(hist, CEIL),
            )
        )
    return out


P = """Watchdog: KEEP WAITING or KILL this optimization round.

model: {model} | operation: {op} | running: {in_op}s
OBSERVED history for this operation on this model: p50={p50}s p95={p95}s p99={p99}s
device CPU per window (oldest->newest): {cpu}
transcript bytes per window: {txt}
actions taken in window: {actions}  of which DISTINCT: {distinct_actions}
operator ceiling: 10800s

Notes: host-bound work (kernel_compile, weight_load, thermal_cool, device_reset, git_op, api_backoff,
jit_compile) uses NO device CPU and may emit almost no log, yet is healthy.
Constant tiny CPU with zero log growth can be a zombie.
IMPORTANT: many actions but only 1-2 DISTINCT means it is repeating itself = a spin/retry loop, not progress.
Judge against the observed history, not a fixed number.

Reply ONLY: {{"decision":"wait"|"kill"}}"""


def d_agent(s):
    try:
        r = subprocess.run(
            ["claude", "-p", P.format(**s), "--output-format", "text", "--model", "claude-sonnet-4-6"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        o = (r.stdout or "").strip()
        i, j = o.find("{"), o.rfind("}")
        return json.loads(o[i : j + 1]).get("decision", "?")
    except Exception:
        return "ERR"


def d_hybrid(ag, s):
    b = s["bounds"]
    alive = s["cpu"][-1] > 0 or s["txt"][-1] > 0
    novel = s["distinct_actions"] > 1 or s["actions"] <= 1  # derived: repetition => not progress
    if ag == "kill" and s["in_op"] < b["grace"] and alive and novel:
        return "wait"
    if ag == "wait" and s["since_commit"] > b["ceiling"]:
        return "kill"
    if ag == "wait" and s["since_commit"] > b["flat"] and not alive:
        return "kill"
    if ag == "wait" and s["since_commit"] > b["flat"] and not novel:
        return "kill"  # NEW: spin guard
    return ag


allsc = gen(120)
EVAL = allsc[36:]
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    ags = list(ex.map(d_agent, EVAL))
el = time.time() - t0
rows = [
    dict(id=s["id"], truth=s["truth"], health=s["health"], op=s["op"], model=s["model"], agent=a, hybrid=d_hybrid(a, s))
    for s, a in zip(EVAL, ags)
]


def sc(k):
    ok = sum(1 for r in rows if r[k] == r["truth"])
    fk = sum(1 for r in rows if r["truth"] == "wait" and r[k] == "kill")
    fw = sum(1 for r in rows if r["truth"] == "kill" and r[k] == "wait")
    return ok, fk, fw


print(f"HELD-OUT {len(rows)} scenarios — WITH the novelty signal (distinct actions)\n")
print(f"{'decider':<40}{'correct':>10}{'acc':>7}{'falseKILL':>11}{'falseWAIT':>11}")
print("-" * 79)
for n, k in (("agent + novelty signal", "agent"), ("agent + novelty + DERIVED bounds", "hybrid")):
    ok, fk, fw = sc(k)
    print(f"{n:<40}{ok:>7}/{len(rows)}{100*ok/len(rows):>6.0f}%{fk:>11}{fw:>11}")
print("-" * 79)
print("prev run (no novelty signal):  agent 80/84 95% · hybrid 81/84 96%")
cls = collections.defaultdict(collections.Counter)
for r in rows:
    for k in ("agent", "hybrid"):
        cls[r["health"]][k] += r[k] == r["truth"]
        cls[r["health"]]["n"] += 1
print(f"\n  {'class':<12}{'n':>4}{'agent':>8}{'hybrid':>8}")
for h, c in sorted(cls.items()):
    print(f"  {h:<12}{c['n']//2:>4}{c['agent']:>8}{c['hybrid']:>8}")
print("\nremaining misses:")
for r in rows:
    if r["hybrid"] != r["truth"]:
        print(
            f"  hybrid {'FALSE-KILL' if r['truth']=='wait' else 'FALSE-WAIT'} {r['id']} health={r['health']} op={r['op']} model={r['model']}"
        )
print(f"\ncost: {len(EVAL)} calls in {el:.0f}s")
