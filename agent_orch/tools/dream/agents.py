"""Launch headless Claude Code sessions (workers, the policy developer, round summaries) and account for them.

Every session runs `claude -p ... --output-format stream-json` in its own process group, with its
transcript in $DREAM_HOME/<c>/logs/. Its cost is appended to the ledger's costs.jsonl, which the
budget reads. $DREAM_CLAUDE overrides the claude binary (tests use a stub).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable

from .campaign import Campaign, ORCH_DIR
from .gitops import append_jsonl, now


def claude_bin() -> str:
    for cand in (os.environ.get("DREAM_CLAUDE"), shutil.which("claude"), str(Path.home() / ".local/bin/claude")):
        if cand and Path(cand).exists():
            return cand
    raise FileNotFoundError("claude CLI not found (set DREAM_CLAUDE or put claude on PATH)")


def _kill(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
        time.sleep(5)
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def parse_transcript(log: Path) -> dict:
    """Cost, turns and final text from a stream-json transcript (the session may emit several result events)."""
    usd, turns, text = 0.0, 0, ""
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("type") == "result":
                usd = max(usd, float(d.get("total_cost_usd") or 0.0))
                turns = max(turns, int(d.get("num_turns") or 0))
                text = d.get("result") or text
    return {"usd": round(usd, 4), "turns": turns, "text": text}


def track(pid: int, sessions: Path | None):
    """Context: register a process group in $DREAM_HOME/<c>/sessions so `dream stop` can kill it."""
    import contextlib

    @contextlib.contextmanager
    def cm():
        f = sessions / f"{pid}.pid" if sessions else None
        if f:
            sessions.mkdir(parents=True, exist_ok=True)
            f.write_text(str(pid))
        try:
            yield
        finally:
            if f:
                f.unlink(missing_ok=True)

    return cm()


def run_session(
    prompt: str,
    cwd: Path,
    log: Path,
    model: str | None,
    timeout_min: float,
    done: Callable[[], bool] | None = None,
    resume: str | None = None,
    sessions: Path | None = None,
) -> dict:
    """Run one headless session to completion (or until `done()` holds and a 2 min grace passes)."""
    log.parent.mkdir(parents=True, exist_ok=True)
    args = [claude_bin(), "-p"]
    args += ["--resume", resume, prompt] if resume else [prompt]
    args += ["--permission-mode", "bypassPermissions", "--output-format", "stream-json", "--verbose"]
    if model:
        args += ["--model", model]
    t0 = time.time()
    with open(log, "a") as out:
        p = subprocess.Popen(
            args,
            cwd=cwd,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "DREAM_SESSION": "1"},
        )
    deadline = t0 + timeout_min * 60
    timed_out = False
    with track(p.pid, sessions):
        _wait(p, deadline, done)
        timed_out = time.time() > deadline
    rc = p.wait()
    info = parse_transcript(log)
    return {**info, "rc": rc, "timed_out": timed_out, "minutes": round((time.time() - t0) / 60, 1)}


def _wait(p: subprocess.Popen, deadline: float, done: Callable[[], bool] | None) -> None:
    while p.poll() is None:
        if done and done():
            for _ in range(24):  # a session can keep a background shell alive after it has committed
                if p.poll() is not None:
                    break
                time.sleep(5)
            if p.poll() is None:
                _kill(p.pid)
            break
        if time.time() > deadline:
            _kill(p.pid)
            break
        time.sleep(10)


def record_cost(c: Campaign, kind: str, label: str, info: dict) -> None:
    append_jsonl(
        c.ledger / "costs.jsonl",
        {
            "time": now(),
            "kind": kind,
            "label": label,
            "usd": info.get("usd", 0.0),
            "turns": info.get("turns"),
            "minutes": info.get("minutes"),
            "rc": info.get("rc"),
            "timed_out": info.get("timed_out"),
        },
    )


def worker_prompt(c: Campaign, node: str, parent: str, wt: Path) -> str:
    brief = wt / c.campaign_rel / "brief.md"
    rules = c.cfg.get("rules") or []
    p = f"""You are a worker in a Dream-RSI discovery campaign that optimizes code in a tt-metal checkout.
Read {wt}/agent_orch/WORKER.md and follow it exactly. It defines your whole job.

Inputs:
- CAMPAIGN={c.name}
- NODE_ID={node}
- PARENT={parent}
- WORKTREE={wt}
- HISTORY={c.home / 'history.md'}
- SPEC={wt / c.campaign_rel / 'dream.yaml'}
- BRIEF={brief if brief.exists() else '(none)'}

Work only inside WORKTREE (your current directory). Run the tools from it:
agent_orch/bin/dream eval --campaign {c.name} --node {node}
agent_orch/bin/dream commit --campaign {c.name} --node {node}
Set "worker" in node.json to your model id.
Never push, never edit files outside WORKTREE, never kill processes you did not start, never reset the device.
Your final message must be only the JSON line printed by `dream commit`."""
    if rules:
        p += "\n\nCampaign rules (an attempt that breaks one is marked invalid):\n" + "\n".join(f"- {r}" for r in rules)
    if c.isolated:
        p += (
            "\n\nIsolation: use only this campaign's own material: your worktree, the refs under "
            f"refs/dream/{c.name}/, HISTORY and the brief. Do not read other checkouts or repositories on this machine "
            f"(e.g. {c.main_repo}, except its python_env), other campaigns under {c.dream_home}, other git branches or "
            "remotes, and do not fetch, clone or search the web for earlier optimizations of this code. Every "
            "transcript is audited; attempts that reach outside are flagged"
            + (" and invalidated." if c.cfg["isolation"].get("on_violation") == "invalidate" else ".")
        )
    return p


def run_worker(c: Campaign, node: str, parent: str, wt: Path, timeout_min: float) -> dict:
    log = c.logs / f"worker_{node}.jsonl"

    def committed() -> bool:
        return c.ref_exists(c.ref_node(node))

    info = run_session(
        worker_prompt(c, node, parent, wt),
        wt,
        log,
        c.cfg["models"].get("worker"),
        timeout_min,
        done=committed,
        sessions=c.home / "sessions",
    )
    info["committed"] = committed()
    record_cost(c, "worker", node, info)
    return info


def run_policy_dev(c: Campaign, current: str, rounds: list[int], timeout_min: float = 90) -> dict:
    nxt = f"v{int(current[1:]) + 1}"
    prompt = f"""You are the policy-development agent of a Dream-RSI campaign (the "dreaming" phase).
Read {ORCH_DIR}/POLICY_DEV.md and follow it exactly.

Inputs:
- CAMPAIGN={c.name}
- LEDGER={c.ledger}
- CURRENT={current}
- NEXT={nxt}
- ROUNDS={','.join(map(str, rounds))}
- M={c.cfg['dreaming']['revisions']}
- REPLAY: {ORCH_DIR}/bin/dream replay --campaign {c.name} --rounds {','.join(map(str, rounds))} --policy <path>/policy.py --out <path>/replay.json --traces <path>/traces.jsonl

Write only under {c.ledger}/policies/. Never edit ACTIVE. Your final message must be only the JSON reply described in POLICY_DEV.md §6."""
    log = c.logs / f"policy_dev_{current}.jsonl"
    info = run_session(
        prompt, c.ledger, log, c.cfg["models"].get("policy_dev"), timeout_min, sessions=c.home / "sessions"
    )
    record_cost(c, "policy_dev", current, info)
    return info


def run_round_summary(c: Campaign, rnd: int, timeout_min: float = 20) -> dict:
    rdir = c.ledger / "rounds" / f"r{rnd:02d}"
    prompt = f"""Summarize round r{rnd:02d} of the Dream-RSI campaign '{c.name}' for the people following it.
Read {c.home / 'history.md'} (all rounds so far). For details, read nodes with the git commands at its end.

Write two files:
1. {rdir / 'summary.md'}: under 200 words. Best node of this round and of the campaign so far, what worked,
   what failed and why, what the policy closed, what the next round should build on.
2. {rdir / 'insights.json'}: {{"worked": [...], "dead_ends": [...], "open_leads": [...]}}, covering the whole campaign so
   far, 3-8 short items each (one sentence, name node ids in parentheses). Dead ends must be things measured not to
   help, not untried ideas.

Plain, specific language. Do not edit anything else. Reply with only "done"."""
    log = c.logs / f"summary_r{rnd:02d}.jsonl"
    info = run_session(prompt, c.ledger, log, c.cfg["models"].get("summary"), timeout_min, sessions=c.home / "sessions")
    record_cost(c, "summary", f"r{rnd:02d}", info)
    return info
