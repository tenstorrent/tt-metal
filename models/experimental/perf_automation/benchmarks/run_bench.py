import sys
import json
import collections

sys.path.insert(0, "/tmp/timer_bench")
from scenarios import SCENARIOS
from deciders import decide_current, decide_improved, decide_agent

REPEATS = 3
rows = []
agent_calls = 0
agent_time = 0.0
for s in SCENARIOS:
    cur, cur_why = decide_current(s)
    imp, imp_why = decide_improved(s)
    ag = []
    for _ in range(REPEATS):
        d, why, lat = decide_agent(s)
        ag.append((d, why))
        agent_calls += 1
        agent_time += lat
    votes = collections.Counter(d for d, _ in ag)
    ag_maj = votes.most_common(1)[0][0]
    stable = len(votes) == 1
    rows.append(
        dict(
            id=s["id"],
            truth=s["truth"],
            note=s["note"],
            cur=cur,
            cur_why=cur_why,
            imp=imp,
            imp_why=imp_why,
            ag=ag_maj,
            ag_why=ag[0][1],
            ag_stable=stable,
            ag_votes=dict(votes),
        )
    )
    print(
        f"  {s['id']:<26} truth={s['truth']:<4} cur={cur:<4} imp={imp:<4} agent={ag_maj:<4} stable={'Y' if stable else 'N'}",
        flush=True,
    )


def score(key):
    ok = sum(1 for r in rows if r[key] == r["truth"])
    fk = sum(1 for r in rows if r["truth"] == "wait" and r[key] == "kill")  # false kill (worst)
    fw = sum(1 for r in rows if r["truth"] == "kill" and r[key] == "wait")  # false wait
    return ok, fk, fw


print("\n" + "=" * 78)
print(f"{'decider':<24}{'correct':>9}{'false KILL':>12}{'false WAIT':>12}{'deterministic':>15}")
print("-" * 78)
for name, key, det in (
    ("current arithmetic", "cur", "yes"),
    ("improved arithmetic", "imp", "yes"),
    ("claude code agent", "ag", None),
):
    ok, fk, fw = score(key)
    d = (
        "yes"
        if det
        else (
            "yes" if all(r["ag_stable"] for r in rows) else f"NO ({sum(1 for r in rows if not r['ag_stable'])} flapped)"
        )
    )
    print(f"{name:<24}{ok:>6}/{len(rows)}{fk:>12}{fw:>12}{d:>15}")
print("-" * 78)
print(f"agent cost: {agent_calls} calls, {agent_time:.0f}s total, {agent_time/agent_calls:.1f}s/call")
print("\nmisses:")
for r in rows:
    for name, key in (("current", "cur"), ("improved", "imp"), ("agent", "ag")):
        if r[key] != r["truth"]:
            kind = "FALSE-KILL" if r["truth"] == "wait" else "FALSE-WAIT"
            print(f"  {name:<9} {kind} on {r['id']:<26} ({r['note'][:52]})")
json.dump(rows, open("/tmp/timer_bench/results.json", "w"), indent=1)
