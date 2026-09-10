import json, random, subprocess, sys, time, collections
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/tmp/timer_bench")
from derived import REAL, HOST_BOUND, observe, stats, derived_bounds

RNG = random.Random(1234)
CEIL = 10800
OPS = ["profile", "pcc", "build", "agent"] + sorted(HOST_BOUND)


def op_med(ops, op):
    if op in HOST_BOUND:  # host-bound ops anchored to build cost (compile/load class)
        return max(20, ops["build"] * 0.35)
    return ops[op]


def gen(n):
    out = []
    for i in range(n):
        base, label, ops = RNG.choice(REAL)
        op = RNG.choice(OPS)
        med = op_med(ops, op)
        hist = observe(med, 24, RNG)
        s = stats(hist)
        health = RNG.choice(["healthy", "healthy", "hung_flat", "zombie", "spin"])
        host_quiet = op in HOST_BOUND
        if health == "healthy":
            actual = med * RNG.lognormvariate(0, 0.45)
            in_op = RNG.uniform(0.05, 1.0) * actual
            cpu = [0] * 5 if host_quiet else [int(RNG.uniform(2000, 9000)) for _ in range(5)]
            txt = (
                [int(RNG.uniform(2000, 40000)) for _ in range(5)]
                if not host_quiet
                else [int(RNG.uniform(0, 300)) for _ in range(5)]
            )
            truth = "wait"
        elif health == "hung_flat":
            in_op = s["p99"] * RNG.uniform(3, 12)
            cpu = [0] * 5
            txt = [0] * 5
            truth = "kill"
        elif health == "zombie":
            in_op = s["p99"] * RNG.uniform(3, 12)
            cpu = [int(RNG.uniform(1, 20))] * 5
            txt = [0] * 5
            truth = "kill"
        else:  # spin: log moves, no real progress
            in_op = s["p99"] * RNG.uniform(3, 12)
            cpu = [0] * 5
            txt = [int(RNG.uniform(3000, 9000)) for _ in range(5)]
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
                p50=round(s["p50"], 1),
                p95=round(s["p95"], 1),
                p99=round(s["p99"], 1),
                bounds=derived_bounds(hist, CEIL),
            )
        )
    return out


# ---- deciders ----
def d_fixed(s):
    cap = min(CEIL, max(2400, int(3 * s["base"])))
    alive = s["cpu"][-1] > 0 or s["txt"][-1] > 0
    if not alive and s["in_op"] > 600:
        return "kill"
    return "kill" if s["since_commit"] > cap else "wait"


PROMPT = """Watchdog decision: KEEP WAITING or KILL this optimization round.

model: {model} | operation in flight: {op} | running for: {in_op}s
OBSERVED history for THIS operation on THIS model (from prior runs):
  p50={p50}s  p95={p95}s  p99={p99}s
device CPU per window (oldest->newest): {cpu}
agent transcript bytes per window: {txt}
absolute operator ceiling: 10800s

Notes: host-bound work (kernel_compile, weight_load, thermal_cool, device_reset, git_op, api_backoff, jit_compile)
uses NO device CPU and may emit almost no log, yet is healthy. A constant tiny CPU trickle with zero log growth
can be a zombie. Log growth with no commit for far beyond p99 can be a spin loop.
Judge against the OBSERVED history, not a fixed number.

Reply ONLY: {{"decision":"wait"|"kill"}}"""


def d_agent(s):
    try:
        r = subprocess.run(
            ["claude", "-p", PROMPT.format(**s), "--output-format", "text", "--model", "claude-sonnet-4-6"],
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
    if ag == "kill" and s["in_op"] < b["grace"] and alive:
        return "wait"  # derived grace
    if ag == "wait" and s["since_commit"] > b["ceiling"]:
        return "kill"  # operator ceiling
    if ag == "wait" and s["since_commit"] > b["flat"] and not alive:
        return "kill"  # derived flat bound
    return ag


N = 120
allsc = gen(N)
CAL, EVAL = allsc[:36], allsc[36:]  # 30% calibration (unused by deciders), 70% held-out eval
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    ags = list(ex.map(d_agent, EVAL))
el = time.time() - t0

rows = []
for s, ag in zip(EVAL, ags):
    rows.append(
        dict(
            id=s["id"],
            truth=s["truth"],
            health=s["health"],
            op=s["op"],
            model=s["model"],
            fixed=d_fixed(s),
            agent=ag,
            hybrid=d_hybrid(ag, s),
        )
    )


def sc(k):
    ok = sum(1 for r in rows if r[k] == r["truth"])
    fk = sum(1 for r in rows if r["truth"] == "wait" and r[k] == "kill")
    fw = sum(1 for r in rows if r["truth"] == "kill" and r[k] == "wait")
    return ok, fk, fw


print(f"HELD-OUT EVAL: {len(rows)} scenarios (calibration set of {len(CAL)} kept separate, unused)\n")
print(f"{'decider':<34}{'correct':>10}{'acc':>8}{'false KILL':>12}{'false WAIT':>12}")
print("-" * 78)
for n, k in (
    ("current fixed timers", "fixed"),
    ("claude agent (given observed p50/95/99)", "agent"),
    ("agent + DERIVED bounds (no constants)", "hybrid"),
):
    ok, fk, fw = sc(k)
    print(f"{n:<34}{ok:>7}/{len(rows)}{100*ok/len(rows):>7.0f}%{fk:>12}{fw:>12}")
print("-" * 78)
print(f"agent cost: {len(EVAL)} calls in {el:.0f}s ({el/len(EVAL):.1f}s/call, 8 parallel)\n")
print("accuracy by health class (held-out):")
cls = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    for k in ("fixed", "agent", "hybrid"):
        cls[r["health"]][k] += r[k] == r["truth"]
        cls[r["health"]]["n_" + k] += 1
print(f"  {'class':<12}{'n':>4}{'fixed':>9}{'agent':>9}{'hybrid':>9}")
for h, c in sorted(cls.items()):
    n = c["n_fixed"]
    print(f"  {h:<12}{n:>4}{c['fixed']:>9}{c['agent']:>9}{c['hybrid']:>9}")
print("\naccuracy by model size (held-out):")
bym = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    for k in ("fixed", "agent", "hybrid"):
        bym[r["model"]][k] += r[k] == r["truth"]
        bym[r["model"]]["n"] += 1
for m, c in bym.items():
    print(f"  {m:<24}n={c['n']//3:<4} fixed={c['fixed']:<4} agent={c['agent']:<4} hybrid={c['hybrid']}")
json.dump(rows, open("/tmp/timer_bench/results3.json", "w"), indent=1)
