import json, subprocess, sys, time, collections
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/tmp/timer_bench")
from scen2 import SCEN

CEIL = 10800


# ---------- 1) CURRENT (ships today) ----------
def d_current(s):
    stall, base = 600, s["base"]
    cap = min(CEIL, max(max(stall * 4, 2400), int(3 * base)))
    alive = (s["cpu_hist"][-1] > 0) or (s["txt_hist"][-1] > 0)
    if not alive and s["elapsed"] > stall:
        return "kill", "FROZEN>600s"
    if s["since_commit"] > cap:
        return "kill", f"UNPRODUCTIVE>{cap}s"
    return "wait", f"within {cap}s"


# ---------- 2) AGENT ----------
P = """You are a watchdog for a model-optimization run. Decide: KEEP WAITING or KILL.

model/pipeline: {model}
baseline profile duration: {base}s   (0 = no history, cold start)
round elapsed: {elapsed}s ; since last commit/attempt: {since_commit}s
tool in flight: {tool} (running {tool_elapsed}s)
device CPU per window (oldest->newest, 5 windows): {cpu_hist}
agent transcript bytes per window (oldest->newest): {txt_hist}
absolute ceiling for any round: 10800s

Guidance:
- KILL only if genuinely stuck: nothing real is progressing.
- WAIT for legitimately slow work however long, unless the ceiling is breached.
- Host-side work (kernel compile, weight load from disk, thermal cooldown, device reset, git op, API backoff) consumes NO device CPU and may emit NO log, yet is HEALTHY.
- A small constant CPU trickle with zero transcript growth can be a zombie, not progress.
- Log output alone, with no commit/attempt for a very long time, can be a spin loop.
- Device spinning with a totally dead log for a very long time can be a hung op.

Reply with ONLY: {{"decision":"wait"|"kill","reason":"<=12 words"}}"""


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
        d = json.loads(o[i : j + 1])
        return d.get("decision", "?"), d.get("reason", "")[:52]
    except Exception as e:
        return "ERR", type(e).__name__


# ---------- 3) AGENT + GUARDRAILS ----------
def d_hybrid(agent_decision, s):
    grace = max(120.0, 8 * (s["base"] or 30))  # never kill before a grace period
    if agent_decision == "kill" and s["tool_elapsed"] < grace and (s["cpu_hist"][-1] > 0 or s["txt_hist"][-1] > 0):
        return "wait", f"guard: grace {grace:.0f}s, still active"
    if agent_decision == "wait" and s["since_commit"] > CEIL:
        return "kill", "guard: ceiling breached"
    if (
        agent_decision == "wait"
        and all(c == 0 for c in s["cpu_hist"])
        and all(t == 0 for t in s["txt_hist"])
        and s["tool"] in (None, "measure_candidate", "check_pcc", "profile_model")
        and s["since_commit"] > max(600, 30 * (s["base"] or 20))
    ):
        return "kill", "guard: all-flat well past bound"
    return agent_decision, "agent"


REP = 3


def work(s):
    votes = []
    for _ in range(REP):
        votes.append(d_agent(s))
    c = collections.Counter(d for d, _ in votes)
    return s["id"], c.most_common(1)[0][0], votes[0][1], len(c) == 1


t0 = time.time()
with ThreadPoolExecutor(max_workers=6) as ex:
    agent_res = {i: (d, w, st) for i, d, w, st in ex.map(work, SCEN)}
elapsed = time.time() - t0

rows = []
for s in SCEN:
    cur, curw = d_current(s)
    ag, agw, stable = agent_res[s["id"]]
    hy, hyw = d_hybrid(ag, s)
    rows.append(
        dict(id=s["id"], truth=s["truth"], note=s["note"], cur=cur, ag=ag, hy=hy, stable=stable, agw=agw, hyw=hyw)
    )


def sc(k):
    ok = sum(1 for r in rows if r[k] == r["truth"])
    fk = sum(1 for r in rows if r["truth"] == "wait" and r[k] == "kill")
    fw = sum(1 for r in rows if r["truth"] == "kill" and r[k] == "wait")
    return ok, fk, fw


print(f"{'scenario':<30}{'truth':<6}{'current':<9}{'agent':<8}{'hybrid':<8}{'stable'}")
print("-" * 74)
for r in rows:
    m = lambda v: v if v == r["truth"] else v.upper() + "*"
    print(f"{r['id']:<30}{r['truth']:<6}{m(r['cur']):<9}{m(r['ag']):<8}{m(r['hy']):<8}{'Y' if r['stable'] else 'N'}")
print("\n" + "=" * 74)
print(f"{'decider':<26}{'correct':>10}{'false KILL':>12}{'false WAIT':>12}")
print("-" * 74)
for n, k in (("current arithmetic", "cur"), ("claude code agent", "ag"), ("agent + guardrails", "hy")):
    ok, fk, fw = sc(k)
    print(f"{n:<26}{ok:>7}/{len(rows)}{fk:>12}{fw:>12}")
print("-" * 74)
flap = [r["id"] for r in rows if not r["stable"]]
print(
    f"agent determinism: {len(rows)-len(flap)}/{len(rows)} identical across {REP} repeats"
    + (f" | flapped: {flap}" if flap else "")
)
print(f"wall clock: {elapsed:.0f}s for {len(SCEN)*REP} agent calls ({elapsed/(len(SCEN)*REP):.1f}s/call, 6 parallel)")
print("\nMISSES by decider:")
for n, k in (("current", "cur"), ("agent", "ag"), ("hybrid", "hy")):
    for r in rows:
        if r[k] != r["truth"]:
            print(f"  {n:<8}{'FALSE-KILL' if r['truth']=='wait' else 'FALSE-WAIT'} {r['id']:<30} {r['note'][:46]}")
json.dump(rows, open("/tmp/timer_bench/results2.json", "w"), indent=1)
