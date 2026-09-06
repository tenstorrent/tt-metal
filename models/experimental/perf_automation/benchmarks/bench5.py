import json, random, subprocess, sys, time, collections
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/tmp/timer_bench")
from derived import REAL, HOST_BOUND, observe, stats, derived_bounds

RNG = random.Random(1234)
CEIL = 10800
OPS = ["profile", "pcc", "build", "agent"] + sorted(HOST_BOUND)


def op_med(ops, op):
    return max(20, ops["build"] * 0.35) if op in HOST_BOUND else ops[op]


ACT = {
    "profile": ["measure_candidate", "Read", "record_kernel_attempt"],
    "pcc": ["check_pcc", "Read"],
    "build": ["Write", "run_perf_test", "Edit", "Read"],
    "agent": ["Read", "Grep", "Edit", "termination_check"],
    "kernel_compile": ["Bash(ninja)"],
    "weight_load": ["check_pcc"],
    "thermal_cool": ["_await_cool"],
    "device_reset": ["tt-smi -r"],
    "git_op": ["git_commit"],
    "api_backoff": ["(waiting)"],
    "jit_compile": ["measure_candidate"],
}
LOGS = {
    "healthy": [
        "kernel cache miss -> compiling",
        "op 1247/1903 dispatched",
        "PCC 0.9998 vs reference",
        "committed perf(_tt_common)",
    ],
    "hung_flat": [],
    "zombie": ["(no new lines)"],
    "spin": [
        "retrying: shard width must match per core N",
        "retrying: shard width must match per core N",
        "retrying: shard width must match per core N",
    ],
}


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
        pool = ACT.get(op, ["Read"])
        if health == "healthy":
            actual = med * RNG.lognormvariate(0, 0.45)
            in_op = RNG.uniform(0.05, 1.0) * actual
            cpu = [0] * 5 if hq else [int(RNG.uniform(2000, 9000)) for _ in range(5)]
            txt = (
                [int(RNG.uniform(0, 300)) for _ in range(5)]
                if hq
                else [int(RNG.uniform(2000, 40000)) for _ in range(5)]
            )
            seq = [RNG.choice(pool) for _ in range(RNG.randint(3, 9))]
            truth = "wait"
        elif health == "hung_flat":
            in_op = s["p99"] * RNG.uniform(3, 12)
            cpu = [0] * 5
            txt = [0] * 5
            seq = []
            truth = "kill"
        elif health == "zombie":
            in_op = s["p99"] * RNG.uniform(3, 12)
            cpu = [int(RNG.uniform(1, 20))] * 5
            txt = [0] * 5
            seq = []
            truth = "kill"
        else:
            in_op = s["p99"] * RNG.uniform(3, 12)
            cpu = [0] * 5
            txt = [int(RNG.uniform(3000, 9000)) for _ in range(5)]
            seq = [pool[0]] * RNG.randint(6, 14)
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
                actions=len(seq),
                distinct_actions=len(set(seq)),
                action_seq=seq[:14],
                p50=round(s["p50"], 1),
                p95=round(s["p95"], 1),
                p99=round(s["p99"], 1),
                samples=[round(x, 1) for x in sorted(hist)[:6]] + ["..."] + [round(x, 1) for x in sorted(hist)[-3:]],
                bounds=derived_bounds(hist, CEIL),
                log_tail=LOGS[health],
                round_no=RNG.randint(1, 3),
                commits_this_round=0 if health != "healthy" else RNG.randint(0, 2),
                chips=RNG.choice([1, 1, 1, 4]),
                host_cores=RNG.choice([32, 96]),
                op_count=RNG.choice([413, 1505, 1903, 3747]),
                prior_attempts=RNG.randint(0, 6),
                prior_wedges=RNG.randint(0, 3),
                device_fds=0 if health == "hung_flat" else RNG.choice([0, 4]),
            )
        )
    return out


FULL = """Watchdog: KEEP WAITING or KILL this optimization round. You have the complete evidence.

MODEL/RUN
  model/pipeline: {model}      baseline profile: {base}s      op_count in profile: {op_count}
  hardware: {chips} chip(s), {host_cores} host cores
  round number: {round_no}     commits so far this round: {commits_this_round}
  prior attempts on this op: {prior_attempts} (of which wedged: {prior_wedges})

OPERATION IN FLIGHT
  op: {op}      running for: {in_op}s      time since last commit/attempt: {since_commit}s
  device fds held by the run: {device_fds}

OBSERVED DURATION HISTORY for this op on this model (24 prior samples, sorted)
  {samples}
  p50={p50}s  p95={p95}s  p99={p99}s

ACTIVITY (5 sampling windows, oldest -> newest)
  device CPU jiffies : {cpu}
  transcript bytes   : {txt}
  actions taken      : {actions}   distinct: {distinct_actions}
  action sequence    : {action_seq}
  last log lines     : {log_tail}

LIMITS
  operator ceiling: 10800s

Judge from the evidence, not a fixed number. Consider: host-bound work (kernel_compile, weight_load,
thermal_cool, device_reset, git_op, api_backoff, jit_compile) uses no device CPU and may emit no log yet
is healthy; a constant tiny CPU trickle with no log growth can be a zombie; repeated identical actions
mean a spin/retry loop, not progress; compare elapsed against the observed history.

Reply ONLY: {{"decision":"wait"|"kill","reason":"<=14 words"}}"""

CURATED = """Watchdog: KEEP WAITING or KILL.
model: {model} | op: {op} | running: {in_op}s
observed history: p50={p50}s p95={p95}s p99={p99}s
device CPU/window: {cpu}
transcript bytes/window: {txt}
actions: {actions} distinct: {distinct_actions}
ceiling: 10800s
Host-bound ops are quiet but healthy; tiny CPU trickle with no log = zombie; repeated identical actions = spin loop.
Reply ONLY: {{"decision":"wait"|"kill"}}"""


def ask(tmpl, s):
    try:
        r = subprocess.run(
            ["claude", "-p", tmpl.format(**s), "--output-format", "text", "--model", "claude-sonnet-4-6"],
            capture_output=True,
            text=True,
            timeout=200,
        )
        o = (r.stdout or "").strip()
        i, j = o.find("{"), o.rfind("}")
        return json.loads(o[i : j + 1]).get("decision", "?")
    except Exception:
        return "ERR"


def hybrid(ag, s):
    b = s["bounds"]
    alive = s["cpu"][-1] > 0 or s["txt"][-1] > 0
    novel = s["distinct_actions"] > 1 or s["actions"] <= 1
    if ag == "kill" and s["in_op"] < b["grace"] and alive and novel:
        return "wait"
    if ag == "wait" and s["since_commit"] > b["ceiling"]:
        return "kill"
    if ag == "wait" and s["since_commit"] > b["flat"] and not alive:
        return "kill"
    if ag == "wait" and s["since_commit"] > b["flat"] and not novel:
        return "kill"
    return ag


EVAL = gen(120)[36:]
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    full = list(ex.map(lambda s: ask(FULL, s), EVAL))
    cur = list(ex.map(lambda s: ask(CURATED, s), EVAL))
el = time.time() - t0
rows = [
    dict(
        id=s["id"],
        truth=s["truth"],
        health=s["health"],
        op=s["op"],
        model=s["model"],
        full=f,
        curated=c,
        full_hy=hybrid(f, s),
        cur_hy=hybrid(c, s),
    )
    for s, f, c in zip(EVAL, full, cur)
]


def sc(k):
    ok = sum(1 for r in rows if r[k] == r["truth"])
    fk = sum(1 for r in rows if r["truth"] == "wait" and r[k] == "kill")
    fw = sum(1 for r in rows if r["truth"] == "kill" and r[k] == "wait")
    return ok, fk, fw


print(f"HELD-OUT {len(rows)} scenarios — FULL evidence vs CURATED evidence\n")
print(f"{'decider':<44}{'correct':>10}{'acc':>7}{'falseKILL':>11}{'falseWAIT':>11}")
print("-" * 83)
for n, k in (
    ("agent, CURATED evidence", "curated"),
    ("agent, FULL evidence (all details)", "full"),
    ("agent CURATED + derived bounds", "cur_hy"),
    ("agent FULL + derived bounds", "full_hy"),
):
    ok, fk, fw = sc(k)
    print(f"{n:<44}{ok:>7}/{len(rows)}{100*ok/len(rows):>6.0f}%{fk:>11}{fw:>11}")
print("-" * 83)
cls = collections.defaultdict(collections.Counter)
for r in rows:
    for k in ("curated", "full", "cur_hy", "full_hy"):
        cls[r["health"]][k] += r[k] == r["truth"]
        cls[r["health"]]["n"] += 1
print(f"\n  {'class':<12}{'n':>4}{'curated':>9}{'full':>7}{'cur+g':>7}{'full+g':>8}")
for h, c in sorted(cls.items()):
    print(f"  {h:<12}{c['n']//4:>4}{c['curated']:>9}{c['full']:>7}{c['cur_hy']:>7}{c['full_hy']:>8}")
print("\nmisses with FULL+guardrails:")
for r in rows:
    if r["full_hy"] != r["truth"]:
        print(
            f"  {'FALSE-KILL' if r['truth']=='wait' else 'FALSE-WAIT'} {r['id']} health={r['health']} op={r['op']} model={r['model']}"
        )
print(f"\ncost: {2*len(EVAL)} calls in {el:.0f}s")
