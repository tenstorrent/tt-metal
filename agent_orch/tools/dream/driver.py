"""The campaign driver: a deterministic loop that runs rounds, steps, dreaming and the final result.

It replaces the old orchestrator agent. Everything it decides comes from the policy (where to start,
when to stop) or from the budget (hard limits). It is resumable: all progress lives in git refs and the
ledger, so `dream resume` picks up an interrupted step, round or dreaming phase where it stopped.

State per round in the ledger (rounds/r<tt>/):
    manifest.json      round planned
    decisions.jsonl    one line per step (+ lost/override records)
    summary.md         round finished (written by the round summary session)
    dreamed.json       dreaming after this round finished (or skipped)
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from . import agents, report
from .campaign import Campaign
from .evaluate import drift_check, load_baseline
from .gitops import finalize_best, ledger_commit, now, prepare_worker, read_jsonl, verify_node
from .history import write_md
from .replay import replay_policy
from .steps import active_version, defaults, next_batch, pending_items, plan_round
from .tree import load_round, recorded_rounds


class Stop(Exception):
    """Raised to end the run: finished, budget, blocked."""

    def __init__(self, state: str, message: str):
        super().__init__(message)
        self.state, self.message = state, message


# ---------------------------------------------------------------- status + budget
def write_status(c: Campaign, state: str, message: str = "", **kw) -> None:
    c.home.mkdir(parents=True, exist_ok=True)
    p = c.home / "status.json"
    st = json.loads(p.read_text()) if p.exists() else {}
    st.update(state=state, message=message, updated=now(), pid=os.getpid(), **kw)
    p.write_text(json.dumps(st, indent=2) + "\n")


def read_status(c: Campaign) -> dict:
    p = c.home / "status.json"
    return json.loads(p.read_text()) if p.exists() else {"state": "not started"}


def attempts_used(c: Campaign) -> int:
    n = 0
    for r in recorded_rounds(c):
        for d in read_jsonl(c.ledger / "rounds" / f"r{r:02d}" / "decisions.jsonl"):
            if d.get("type") == "decision":
                n += len(d.get("batch", []))
    return n


def usd_spent(c: Campaign) -> float:
    return round(sum(float(x.get("usd") or 0) for x in read_jsonl(c.ledger / "costs.jsonl")), 2)


class Clock:
    """Active campaign time, persisted in the ledger so pauses (stop/resume) don't count."""

    def __init__(self, c: Campaign):
        self.c, self.path = c, c.ledger / "budget.json"
        d = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.base = float(d.get("active_seconds", 0.0))
        self.t0 = time.time()

    def seconds(self) -> float:
        return self.base + time.time() - self.t0

    def save(self) -> None:
        self.path.write_text(json.dumps({"active_seconds": round(self.seconds())}, indent=2) + "\n")


def budget(c: Campaign, clock: Clock) -> dict:
    b = c.cfg["budget"]
    return {
        "attempts": attempts_used(c),
        "max_attempts": int(b["max_attempts"]),
        "hours": round(clock.seconds() / 3600, 2),
        "max_hours": float(b["max_hours"]),
        "usd": usd_spent(c),
        "max_usd": float(b["max_usd"]),
    }


def budget_left(c: Campaign, clock: Clock) -> tuple[int, str]:
    """(attempts we may still start, why the limit binds)."""
    s = budget(c, clock)
    if s["hours"] >= s["max_hours"]:
        return 0, f"time budget used ({s['hours']:.2f} of {s['max_hours']} h)"
    if s["usd"] >= s["max_usd"]:
        return 0, f"cost budget used (${s['usd']:.2f} of ${s['max_usd']:.0f})"
    left = s["max_attempts"] - s["attempts"]
    why = f"attempt budget ({s['attempts']} of {s['max_attempts']} used)"
    costs = [x for x in read_jsonl(c.ledger / "costs.jsonl") if x.get("kind") == "worker"]
    if costs:
        per = sum(float(x.get("usd") or 0) for x in costs) / len(costs)
        if per > 0:
            by_usd = int((s["max_usd"] - s["usd"]) // per)
            if by_usd < left:
                left, why = by_usd, f"cost budget (${s['usd']:.2f} of ${s['max_usd']:.0f}, ~${per:.2f}/attempt)"
    return max(0, left), why


# ---------------------------------------------------------------- one step
def refresh(c: Campaign, clock: Clock, msg: str) -> None:
    write_md(c, c.home / "history.md")
    clock.save()
    ledger_commit(c, msg)
    try:
        report.build(c)
    except Exception as e:  # the report must never stop the campaign
        print(f"[driver] report failed: {e}", flush=True)


def run_items(c: Campaign, rnd: int, items: list[dict], clock: Clock) -> None:
    b = budget(c, clock)
    minutes_left = max(5.0, (b["max_hours"] - b["hours"]) * 60)
    timeout = min(float(c.cfg["worker"]["timeout_min"]), minutes_left)
    wts = {}
    for it in items:
        wts[it["node"]] = prepare_worker(c, it["node"], it["parent"])
    print(f"[driver] r{rnd:02d}: running {[i['node'] for i in items]} (timeout {timeout:.0f} min)", flush=True)
    write_status(c, "running", f"round {rnd}: workers running", round=rnd, nodes=[i["node"] for i in items])
    with ThreadPoolExecutor(max_workers=len(items)) as ex:
        futs = {
            it["node"]: ex.submit(agents.run_worker, c, it["node"], it["parent"], wts[it["node"]], timeout)
            for it in items
        }
        for node, f in futs.items():
            info = f.result()
            print(
                f"[driver] {node}: rc={info['rc']} committed={info['committed']} ${info['usd']:.2f} "
                f"{info['minutes']} min",
                flush=True,
            )
    for it in items:
        v = verify_node(c, it["node"], record=True)
        if not v["ok"]:
            print(f"[driver] {it['node']}: {v['issues']}", flush=True)
    audit_items(c, rnd, items)
    handle_lost(c, rnd, items)


def audit_items(c: Campaign, rnd: int, items: list[dict]) -> None:
    """Isolation tripwire: flag (or invalidate) attempts whose worker reached outside the campaign."""
    from .audit import audit_transcript
    from .gitops import append_jsonl

    if not c.isolated:
        return
    path = c.ledger / "rounds" / f"r{rnd:02d}" / "decisions.jsonl"
    for it in items:
        node = it["node"]
        if not c.ref_exists(c.ref_node(node)):
            continue
        flags = audit_transcript(c, c.logs / f"worker_{node}.jsonl")
        if not flags:
            continue
        print(f"[driver] {node}: isolation flags {flags[:3]}", flush=True)
        append_jsonl(path, {"type": "audit", "node": node, "flags": flags, "time": now()})
        if c.cfg["isolation"].get("on_violation") == "invalidate":
            append_jsonl(
                path,
                {
                    "type": "override",
                    "node": node,
                    "fail_class": "isolation",
                    "why": "isolation violation: " + "; ".join(flags[:3]),
                    "time": now(),
                },
            )


LOST_RETRIES = 1  # a node whose worker never commits is retried this many times


def handle_lost(c: Campaign, rnd: int, items: list[dict]) -> None:
    """Retry a lost node once; then close its branch (or abandon the slot of a lost first attempt).
    Two consecutive steps in which every worker was lost mean something systemic is broken: block."""
    from .campaign import parse_node_id
    from .gitops import append_jsonl

    path = c.ledger / "rounds" / f"r{rnd:02d}" / "decisions.jsonl"
    recs = read_jsonl(path)
    lost_n: dict[str, int] = {}
    for d in recs:
        if d.get("type") == "lost":
            lost_n[d["node"]] = lost_n.get(d["node"], 0) + 1
    for it in items:
        n = lost_n.get(it["node"], 0)
        if n > LOST_RETRIES:
            _, b, a = parse_node_id(it["node"])
            why = f"worker lost {n} times on {it['node']}"
            append_jsonl(path, {"type": "close" if a > 1 else "abandon", "branch": b, "why": why, "time": now()})
            print(f"[driver] {'closing' if a > 1 else 'abandoning'} branch b{b:02d}: {why}", flush=True)
    # index of each step's decision; a node counts as lost in a step if a lost record follows that decision
    # and precedes the next one
    idx = [i for i, d in enumerate(recs) if d.get("type") == "decision" and d.get("batch")]

    def lost_in(k: int) -> bool:
        lo, hi = idx[k], idx[k + 1] if k + 1 < len(idx) else len(recs)
        lost = {d["node"] for d in recs[lo:hi] if d.get("type") == "lost"}
        return all(it["node"] in lost for it in recs[lo]["batch"])

    distinct = {it["node"] for k in idx[-2:] for it in recs[k]["batch"]}
    if len(idx) >= 2 and lost_in(len(idx) - 1) and lost_in(len(idx) - 2) and len(distinct) >= 2:
        raise Stop(
            "blocked",
            "workers in the last two steps all failed to commit; check the worker logs in "
            f"{c.logs} (model access, permissions, a broken worktree)",
        )


def run_round(c: Campaign, rnd: int, clock: Clock) -> str:
    """Run the steps of round `rnd` until the policy stops. Returns 'stop' or 'budget'."""
    while True:
        pending = pending_items(c, rnd)
        if pending:
            print(f"[driver] r{rnd:02d}: resuming unfinished step ({[i['node'] for i in pending]})", flush=True)
            run_items(c, rnd, pending, clock)
            refresh(c, clock, f"r{rnd:02d} step (resumed)")
            continue
        left, why = budget_left(c, clock)
        if left <= 0:
            return "budget"
        res = next_batch(c, rnd, max_items=left, budget_note=why)
        if res["stop"]:
            refresh(c, clock, f"r{rnd:02d} stop: {res.get('why', '')}")
            return "budget" if res.get("why", "").startswith("budget") else "stop"
        if res["closed"]:
            print(f"[driver] r{rnd:02d}: closed {res['closed']}", flush=True)
        run_items(c, rnd, res["items"], clock)
        refresh(c, clock, f"r{rnd:02d} step {res['step']}")


# ---------------------------------------------------------------- rounds
def global_best(c: Campaign):
    best = None
    for r in recorded_rounds(c):
        for n in load_round(c, r).nodes:
            if n.ok and (best is None or n.score > best.score):
                best = n
    return best


def finish_round(c: Campaign, rnd: int, clock: Clock) -> None:
    rdir = c.ledger / "rounds" / f"r{rnd:02d}"
    write_md(c, c.home / "history.md")
    if not (rdir / "summary.md").exists():
        write_status(c, "running", f"round {rnd}: writing the round summary", round=rnd)
        info = agents.run_round_summary(c, rnd)
        if not (rdir / "summary.md").exists():
            (rdir / "summary.md").write_text(f"# r{rnd:02d}\n\n(summary session failed: rc={info['rc']})\n")
    (c.ledger / "snapshots").mkdir(exist_ok=True)
    (c.ledger / "snapshots" / f"history_r{rnd:02d}.md").write_text((c.home / "history.md").read_text())
    refresh(c, clock, f"r{rnd:02d} finished")


def dream(c: Campaign, rnd: int, clock: Clock) -> None:
    rdir = c.ledger / "rounds" / f"r{rnd:02d}"
    current = active_version(c)
    rounds = [r for r in recorded_rounds(c) if (c.ledger / "rounds" / f"r{r:02d}" / "summary.md").exists()]
    out = {"after_round": rnd, "current": current, "rounds": rounds, "time": now()}
    if not c.cfg["dreaming"]["enabled"]:
        out.update(skipped="dreaming disabled", winner=current)
    else:
        write_status(c, "running", f"dreaming after round {rnd} (policy {current})", round=rnd)
        info = agents.run_policy_dev(c, current, rounds)
        out["policy_dev"] = {k: info[k] for k in ("rc", "usd", "minutes", "timed_out")}
        out["reply"] = info["text"][-2000:]
        nxt = f"v{int(current[1:]) + 1}"
        cand = c.ledger / "policies" / nxt / "policy.py"
        out["winner"] = current
        if cand.exists():
            # verify the agent's claim ourselves: the replay is deterministic
            rr = [load_round(c, r) for r in rounds]
            noise = load_baseline(c)["noise_pct"]
            d = defaults(c)
            v_cur = replay_policy(c.ledger / "policies" / current / "policy.py", rr, noise, d, c.cfg["dreaming"])
            v_new = replay_policy(cand, rr, noise, d, c.cfg["dreaming"])
            (c.ledger / "policies" / nxt / "replay.json").write_text(json.dumps(v_new, indent=2) + "\n")
            out.update(current_V=v_cur["objective_V"], candidate=nxt, candidate_V=v_new["objective_V"])
            if v_new["objective_V"] >= v_cur["objective_V"] and not any("error" in e for e in v_new["episodes"]):
                (c.ledger / "policies" / "ACTIVE").write_text(nxt + "\n")
                out["winner"] = nxt
            else:
                out["rejected"] = "replay V below the current policy's, or illegal batches"
    (rdir / "dreamed.json").write_text(json.dumps(out, indent=2) + "\n")
    refresh(c, clock, f"dreaming after r{rnd:02d}: active {out['winner']}")


def start_round(c: Campaign, rnd: int, clock: Clock) -> None:
    best = global_best(c)
    if c.cfg["search"].get("round_root", "origin") == "best" and best and rnd > 1:
        root_ref = c.ref_node(best.node_id)  # compound: build on the best attempt so far
    else:
        root_ref = c.ref_root()  # as in the paper: a fresh tree from the campaign root; earlier rounds are history
    extra = {}
    if c.cfg["eval"].get("drift_check") and load_baseline(c):
        write_status(c, "running", f"round {rnd}: re-measuring the campaign root for drift", round=rnd)
        drifted, rows = drift_check(c, f"r{rnd:02d}_root")
        extra["root_remeasure"] = [
            {"case": r[0], "value": r[1], "baseline": r[2], "pct": round(r[3], 2), "drift": r[4]} for r in rows
        ]
        if drifted:  # no manifest: `dream resume` re-measures and continues once the machine is fixed
            (c.ledger / f"drift_r{rnd:02d}.json").write_text(json.dumps(extra, indent=2) + "\n")
            refresh(c, clock, f"r{rnd:02d}: baseline drift")
            raise Stop(
                "blocked",
                f"round {rnd}: the campaign root measures outside the noise band; the machine "
                f"changed and scores are not comparable. Fix it (or re-baseline), then resume",
            )
    m = plan_round(c, rnd, root_ref, extra)
    print(f"[driver] r{rnd:02d}: planned with {m['policy']} W={m['W']} R={m['R']} root={root_ref}", flush=True)
    refresh(c, clock, f"r{rnd:02d} planned")


def finalize(c: Campaign) -> dict | None:
    best = global_best(c)
    if not best:
        return None
    sha = finalize_best(c, best.node_id, best.score)
    return {
        "node": best.node_id,
        "score": best.score,
        "branch": c.ref_best().removeprefix("refs/heads/"),
        "commit": sha,
    }


def run(c: Campaign) -> None:
    if not load_baseline(c):
        raise SystemExit("no baseline; run `dream check` first")
    clock = Clock(c)
    write_status(c, "running", "driver started", started=now())
    max_rounds = int(c.cfg["search"]["max_rounds"])
    try:
        while True:
            rounds = recorded_rounds(c)
            rnd = rounds[-1] if rounds else 0
            rdir = c.ledger / "rounds" / f"r{rnd:02d}"
            if rnd and not (rdir / "summary.md").exists():
                outcome = run_round(c, rnd, clock)
                finish_round(c, rnd, clock)
                if outcome == "budget":
                    raise Stop("finished", budget_left(c, clock)[1])
                continue
            if rnd >= max_rounds:
                raise Stop("finished", f"max_rounds ({max_rounds}) reached")
            if rnd and not (rdir / "dreamed.json").exists():
                if budget_left(c, clock)[0] <= 0:
                    raise Stop("finished", budget_left(c, clock)[1])
                dream(c, rnd, clock)
                continue
            if budget_left(c, clock)[0] <= 0:
                raise Stop("finished", budget_left(c, clock)[1])
            start_round(c, rnd + 1, clock)
    except Stop as s:
        result = finalize(c) if s.state == "finished" else None
        clock.save()
        ledger_commit(c, f"campaign {s.state}: {s.message}")
        write_status(c, s.state, s.message, result=result)
        report.build(c)
        print(f"[driver] {s.state}: {s.message}" + (f"; best {result}" if result else ""), flush=True)
    except Exception as e:
        clock.save()
        ledger_commit(c, f"driver error: {e}")
        write_status(c, "error", f"{type(e).__name__}: {e}")
        try:
            report.build(c)
        except Exception:
            pass
        raise
