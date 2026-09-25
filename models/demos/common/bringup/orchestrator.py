# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The orchestrator: runs every step of a bring-up in dependency order and stops only when a person is needed.

    python -m models.demos.common.bringup.orchestrator run --spec S [--until ID] [--only ID] [--model M]
    python -m models.demos.common.bringup.orchestrator resume --spec S      # STOPPED tasks become runnable again

Per task:
  scripted step (goldens, box, integrate, perf; no role)  run the gate; on failure hand the log to a ``fix`` agent
  agent step (reference, plan, contract, implement)       write a brief, start a fresh ``claude -p`` with the
                                                           bringup-engineer definition, check what it did, run the gate
  tests declared and not frozen                            render the template, ``test`` role reviews it, freeze
                                                           (reference passes, zero stub fails), then implement
  three failed attempts                                    WIP commit, then ``ttnn-expert-debugger`` with the WIP sha,
                                                           the failure logs and the triage report, three attempts
  still failing                                            the task becomes STOPPED and the run stops (exit 1)
  approval needed (plan) or opportunity list written       the run stops for a person (exit 3)

After every agent step: the tree diff must stay inside the paths the brief allowed, no command may reach the device
except through the safe runners, and the knowledge files must keep their format. A violation fails the attempt.
Each agent run is recorded in state.json under the task: role, attempt, session id, model, agent-definition hashes.

BRINGUP_AGENT_CMD replaces the ``claude`` executable (tests use a mock agent). Exit codes: 0 done, 1 stopped on a
failure, 3 waiting for a person.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from string import Template

import yaml

from models.demos.common.bringup.core import freeze as F
from models.demos.common.bringup.core.gate import (
    _is_direct_pytest,
    _program,
    _segments,
    git_commit,
    log_dir,
    run_gate,
    run_name,
)
from models.demos.common.bringup.core.ledger import Ledger
from models.demos.common.bringup.core.runs import FreezeError, agent_hashes, freeze_task
from models.demos.common.bringup.core.spec import CODE_ROOT, Spec
from models.demos.common.bringup.knowledge import check as kcheck
from models.demos.common.bringup.plan import approvals

HERE = Path(__file__).resolve().parent
AGENT_DEF = HERE / "agents" / "bringup-engineer.md"
DEBUGGER = "ttnn-expert-debugger"
DEBUGGER_DEF = CODE_ROOT / ".claude" / "agents" / f"{DEBUGGER}.md"
ROLE_OF_STEP = {"reference": "reference", "plan": "plan", "implement": "implement", "contract": "contract"}
IGNORED = (
    r"/dashboard/index\.html$",
    r"(^|/)__pycache__/",
    r"\.pyc$",
    r"^generated/",
    r"/probes/probe_\d+\.py$",
    r"(^|/)\.lock$",
    r"\.tmp$",
)
DONE, STOPPED, HUMAN = 0, 1, 3


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def rel(spec: Spec, p) -> str:
    p = Path(p)
    return str(p.resolve().relative_to(spec.repo)) if p.is_absolute() else str(p)


# ---------------------------------------------------------------- agent definitions and command
def agents_json(run_dir: Path) -> Path:
    """The --agents file for claude -p, built from the markdown definition (frontmatter + body)."""
    text = AGENT_DEF.read_text()
    _, fm, body = text.split("---", 2)
    meta = yaml.safe_load(fm)
    out = run_dir / "agents.json"
    out.write_text(
        json.dumps(
            {
                meta["name"]: {
                    "description": meta["description"],
                    "prompt": body.strip(),
                    "tools": [t.strip() for t in meta["tools"].split(",")],
                }
            },
            indent=1,
        )
    )
    return out


def agent_command(spec: Spec, run_dir: Path, agent: str, prompt: str, model: str | None) -> list[str]:
    exe = shlex.split(os.environ.get("BRINGUP_AGENT_CMD", "claude"))
    cmd = exe + ["-p", "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions"]
    if agent == "bringup-engineer":
        cmd += ["--agents", str(agents_json(run_dir))]
    cmd += ["--agent", agent]
    if model:
        cmd += ["--model", model]
    return cmd + [prompt]


def parse_stream(log: Path) -> dict:
    """session id, model, final result and every Bash command from a stream-json log."""
    info = {"session_id": None, "model": None, "commands": [], "result": None, "is_error": None}
    for line in log.read_text(errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            info["session_id"], info["model"] = ev.get("session_id"), ev.get("model")
        elif ev.get("type") == "assistant":
            for c in (ev.get("message") or {}).get("content") or []:
                if c.get("type") == "tool_use" and c.get("name") == "Bash":
                    info["commands"].append((c.get("input") or {}).get("command", ""))
        elif ev.get("type") == "result":
            info["result"], info["is_error"] = ev.get("result"), ev.get("is_error")
            info["session_id"] = info["session_id"] or ev.get("session_id")
    return info


def command_violations(commands: list[str], repo: Path) -> list[str]:
    """Commands that reach the device without the safe runners: any direct pytest, or python on code that imports ttnn."""
    out = []
    for cmd in commands:
        try:
            segs = _segments(cmd)
        except ValueError:
            segs = [cmd.split()]
        for words in segs:
            w = _program(words)
            if not w:
                continue
            if _is_direct_pytest(w):
                out.append(f"direct pytest: {cmd[:120]!r}")
            elif Path(w[0]).name.startswith("python"):
                target = next((x for x in w[1:] if x.endswith(".py")), None)
                inline = "-c" in w and "ttnn" in " ".join(w)
                heredoc = "<<" in cmd and "import ttnn" in cmd
                imports_ttnn = False
                if target and (repo / target).exists():
                    imports_ttnn = bool(
                        re.search(r"^\s*(import ttnn|from ttnn)", (repo / target).read_text(errors="replace"), re.M)
                    )
                if inline or heredoc or imports_ttnn:
                    out.append(f"python on device code without a safe runner: {cmd[:120]!r}")
    return sorted(set(out))


# ---------------------------------------------------------------- tree snapshots
def dirty(repo: Path) -> dict[str, str]:
    """path -> content hash for every modified or untracked file (deleted files hash to '')."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "-uall", "-z"], cwd=repo, capture_output=True, text=True
    ).stdout
    res = {}
    for entry in out.split("\0"):
        if len(entry) < 4:
            continue
        p = entry[3:]
        if any(re.search(pat, p) for pat in IGNORED):
            continue
        f = repo / p
        res[p] = F.sha256_file(f) if f.is_file() else ""
    return res


def changed_since(before: dict, after: dict) -> list[str]:
    """Files that are modified after the step and were not modified the same way before it. A file that was dirty
    before and is clean after was committed or reverted meanwhile (agents never commit), so it is not the agent's."""
    return sorted(p for p, h in after.items() if before.get(p) != h)


def allowed(path: str, patterns: list[str]) -> bool:
    return any(path == p or path.startswith(p.rstrip("/") + "/") for p in patterns)


# ---------------------------------------------------------------- the orchestrator
class Orchestrator:
    def __init__(
        self,
        spec: Spec,
        model: str | None = None,
        max_attempts: int = 3,
        debugger_attempts: int = 3,
        timeout_s: int | None = None,
        echo=print,
    ):
        self.spec, self.led = spec, Ledger(spec.bringup_dir)
        self.model = model or spec.get("agents.model")
        self.max_attempts, self.debugger_attempts = max_attempts, debugger_attempts
        self.timeout_s = timeout_s or int(spec.get("agents.timeout_s", 4 * 3600))
        self.run_dir = spec.run_dir(run_name(self.led))
        (self.run_dir / "briefs").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "agents").mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self.roles = yaml.safe_load((HERE / "briefs" / "roles.yaml").read_text())

    # ---- policy
    def common_paths(self) -> list[str]:
        b = rel(self.spec, self.spec.bringup_dir)
        return [
            "models/demos/common/bringup/knowledge/known_issues.md",
            "models/demos/common/bringup/knowledge/repo_map.md",
            f"{b}/BREADCRUMBS.md",
            f"{b}/findings.yaml",
        ]

    def allowed_paths(self, task: dict, role: str) -> list[str]:
        b = rel(self.spec, self.spec.bringup_dir)
        if role == "test":
            return list(task.get("tests") or []) + self.common_paths()
        extra = [f"{b}/plan.yaml", f"{b}/plan.md", f"{b}/components.yaml", f"{b}/tasks.yaml"] if role == "plan" else []
        return list(task.get("paths") or []) + extra + self.common_paths()

    # ---- briefs
    def brief(self, task: dict, role: str, attempt: int, previous: str = "", extra_read: list[str] = ()) -> Path:
        s, b = self.spec, self.spec.bringup_dir
        brief = task.get("brief") or {}
        comp = self._component_entry(brief)
        desc = (
            f"`{brief.get('step')}` ({brief.get('kind')}) of block type `{brief.get('block_type')}`, layer {brief.get('layer')}"
            if "step" in brief
            else f"the `{brief.get('block_type')}` block with {brief.get('swapped')} on device"
            if "swapped" in brief
            else task["title"]
        )
        vals = {
            "tid": task["id"],
            "role": role,
            "attempt": attempt,
            "model": s.model,
            "spec_path": rel(s, s.path) if s.path else "",
            "run": run_name(self.led),
            "now": now(),
            "title": task["title"],
            "model_dir": rel(s, s.model_dir),
            "bringup_dir": rel(s, b),
            "hf_dir": str(s.hf_dir),
            "mesh": "x".join(map(str, s.mesh)),
            "chips": s.mesh[0] * s.mesh[1],
            "dram": s.get("box.chip_dram_gb", 32),
            "tests": ", ".join(task.get("tests") or []),
            "component_desc": desc,
            "component_entry": yaml.safe_dump(comp).strip() if comp else "(none in components.yaml)",
            "gate_cmd": task["gate"]["cmd"],
            "repo": str(s.repo),
            "breadcrumbs": f"{rel(s, b)}/BREADCRUMBS.md",
        }
        vals["role_text"] = Template(self.roles[role]).safe_substitute(vals)
        # spec agents.read.<role>: files every brief of that role lists (e.g. the HF modeling code for the reference role)
        reads = list(task.get("tests") or []) + list(extra_read) + list(brief.get("read") or [])
        reads += [r for r in (s.get(f"agents.read.{role}") or []) if r not in reads]
        vals["read_list"] = "\n".join(f"- `{r}`" for r in reads)
        vals["allowed"] = "\n".join(f"- `{p}`" for p in self.allowed_paths(task, role))
        vals["thresholds"] = (
            "\n".join(f"- `{k}` {v}" for k, v in (task["gate"].get("metrics") or {}).items()) or "- (exit code only)"
        )
        vals["previous"] = f"## Previous attempt failed\n```\n{previous[-6000:]}\n```\n" if previous else ""
        text = Template((HERE / "briefs" / "brief.md").read_text()).safe_substitute(vals)
        p = self.run_dir / "briefs" / f"{task['id']}.{role}.{attempt}.md"
        p.write_text(text)
        return p

    def _component_entry(self, brief: dict) -> dict | None:
        from models.demos.common.bringup.plan.components import load

        for c in load(self.spec).get("components") or []:
            if c.get("block_type") == brief.get("block_type") and c.get("step") == brief.get("step"):
                return c
        return None

    # ---- one agent run
    def run_agent(self, task: dict, role: str, attempt: int, brief: Path, agent: str = "bringup-engineer") -> list[str]:
        tid = task["id"]
        before = dirty(self.spec.repo)
        log = self.run_dir / "agents" / f"{tid}.{role}.{attempt}.jsonl"
        cmd = agent_command(self.spec, self.run_dir, agent, f"Read the brief at {brief} and carry it out.", self.model)
        env = dict(os.environ, PYTHONPATH=str(CODE_ROOT), BRINGUP_TASK=tid)
        if self.spec.path:
            env["BRINGUP_SPEC"] = str(self.spec.path)
        self.echo(f"  [{tid}] {role} attempt {attempt}: {agent} (brief {brief.name})")
        t0 = time.time()
        with open(log, "w") as f:
            try:
                rc = subprocess.run(
                    cmd,
                    cwd=self.spec.repo,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    timeout=self.timeout_s,
                ).returncode
            except subprocess.TimeoutExpired:
                rc = -9
        info = parse_stream(log)
        problems = []
        if rc != 0 or info["is_error"]:
            problems.append(f"agent exited rc={rc} is_error={info['is_error']}")
        problems += command_violations(info["commands"], self.spec.repo)
        pats = self.allowed_paths(task, role)
        outside = [p for p in changed_since(before, dirty(self.spec.repo)) if not allowed(p, pats)]
        if outside:
            problems.append(f"changed files outside the allowed paths (revert them): {outside}")
        ki, _ = kcheck.check_known_issues(kcheck.HERE / "known_issues.md")
        problems += [f"known_issues.md: {e}" for e in ki]
        # Agent definitions and brief templates live in the framework checkout (which may not be the model's repo).
        defs = (
            [AGENT_DEF, HERE / "briefs" / "brief.md", HERE / "briefs" / "roles.yaml"]
            if agent != DEBUGGER
            else [DEBUGGER_DEF]
        )
        defs = [str(d.relative_to(CODE_ROOT)) for d in defs]
        run = {
            "role": role,
            "attempt": attempt,
            "agent": agent,
            "session_id": info["session_id"],
            "model": info["model"] or self.model,
            "rc": rc,
            "seconds": round(time.time() - t0, 1),
            "brief": str(brief),
            "log": str(log),
            "problems": problems,
            "defs": agent_hashes(CODE_ROOT, defs),
            "t": now(),
        }
        entry = self.led.state().get(tid, {})
        self.led.update(
            tid,
            agent_runs=entry.get("agent_runs", []) + [run],
            agent={"session_id": run["session_id"], "model": run["model"], "defs": run["defs"]},
        )
        for p in problems:
            self.echo(f"  [{tid}] problem: {p}")
        return problems

    # ---- steps
    def gate(self, tid: str):
        res = run_gate(self.spec, self.led, tid, commit=True)
        self.echo(f"  [{tid}] gate {res.verdict}" + (f" -> {res.commit}" if res.commit else ""))
        return res

    def failure_text(self, res) -> str:
        tail = res.log.read_text(errors="replace")[-4000:] if res.log and res.log.exists() else ""
        return res.summary() + "\n--- log tail ---\n" + tail

    def freeze_tests(self, task: dict) -> str | None:
        """Render, review (test role), freeze. None on success, else the reason to stop."""
        from models.demos.common.bringup.testing.templates import render_component_test, render_swap_test

        b = task.get("brief") or {}
        if "swapped" in b:
            render_swap_test(self.spec, b["block_type"], b["swapped"])
        elif "step" in b:
            render_component_test(self.spec, b["block_type"], b["step"])
        previous = ""
        for attempt in range(1, self.max_attempts + 1):
            problems = self.run_agent(task, "test", attempt, self.brief(task, "test", attempt, previous))
            if problems:
                previous = "\n".join(problems)
                continue
            try:
                rec = freeze_task(self.spec, self.led, task["id"])
                self.echo(f"  [{task['id']}] frozen: reference={rec.get('reference')} stub={rec.get('stub')}")
                return None
            except FreezeError as e:
                previous = str(e)
                self.echo(f"  [{task['id']}] freeze failed: {str(e).splitlines()[0]}")
        return f"tests could not be frozen after {self.max_attempts} attempts: {previous.splitlines()[0] if previous else ''}"

    def needs_human(self, task: dict) -> str | None:
        if task.get("step") == "plan" and not approvals.is_approved(self.spec, "plan"):
            return (
                f"approve the plan in {rel(self.spec, self.spec.bringup_dir)} (plan.yaml, plan.md, components.yaml, tasks.yaml), "
                f"then: python -m models.demos.common.bringup approve plan --spec {self.spec.path}"
            )
        return None

    def attempt_loop(self, task: dict, role: str, first_failure: str = "") -> bool:
        previous = first_failure
        for attempt in range(1, self.max_attempts + 1):
            problems = self.run_agent(task, role, attempt, self.brief(task, role, attempt, previous))
            if task.get("step") == "plan" and not problems and self.needs_human(task):
                return True  # the plan exists; the gate needs the approval next
            res = self.gate(task["id"])
            if res.verdict == "PASS" and not problems:
                return True
            previous = "\n".join(problems + [self.failure_text(res)])
        return self.debug(task, previous)

    def debug(self, task: dict, previous: str) -> bool:
        tid = task["id"]
        paths = [p for p in task.get("paths") or [] if (self.spec.repo / p).exists()]
        wip = None
        if paths:
            wip = git_commit(
                self.spec,
                paths,
                f"[{self.spec.tag}][{tid}][wip] {task['title']}",
                f"Work in progress after {self.max_attempts} failed attempts; handed to {DEBUGGER}.",
            )
        triage = log_dir(self.spec, self.led) / f"{tid}.triage.txt"
        extra = [str(triage)] if triage.exists() else []
        for attempt in range(1, self.debugger_attempts + 1):
            text = (
                f"WIP commit: {wip or '(no changes to commit)'}\nThe implementer failed {self.max_attempts} times; "
                f"the logs are in {self.run_dir / 'agents'} and {log_dir(self.spec, self.led)}.\n{previous}"
            )
            brief = self.brief(
                task, "fix" if task.get("step") not in ("implement",) else "implement", 100 + attempt, text, extra
            )
            problems = self.run_agent(task, "debug", attempt, brief, agent=DEBUGGER)
            entry = self.led.state().get(tid, {})
            self.led.update(tid, debugger_attempts=entry.get("debugger_attempts", 0) + 1)
            res = self.gate(tid)
            if res.verdict == "PASS" and not problems:
                return True
            previous = "\n".join(problems + [self.failure_text(res)])
        self.led.update(
            tid,
            status="STOPPED",
            reason=[f"stopped after {self.max_attempts} attempts and " f"{self.debugger_attempts} debugger attempts"],
            history_add={"t": now(), "status": "STOPPED"},
        )
        return False

    def step(self, tid: str) -> int:
        task = self.led.task(tid)
        role = task.get("role") or ROLE_OF_STEP.get(task.get("step"))
        self.echo(f"[{tid}] {task['title']} (step {task.get('step')}, role {role or 'script'})")
        if task.get("tests") and not task.get("frozen"):
            reason = self.freeze_tests(task)
            if reason:
                self.led.update(tid, status="STOPPED", reason=[reason], history_add={"t": now(), "status": "STOPPED"})
                return STOPPED
            task = self.led.task(tid)
        plan_written = task.get("step") == "plan" and (self.spec.bringup_dir / "plan.yaml").exists()
        if role is None or plan_written:
            # A scripted step, or a plan written by an earlier run: gate it; an agent only sees it after a failure.
            if plan_written and self.needs_human(task):
                return self._human(task, self.needs_human(task))
            res = self.gate(tid)
            if res.verdict != "PASS" and not self.attempt_loop(task, role or "fix", self.failure_text(res)):
                return STOPPED
        else:
            # The work may already exist (e.g. R.3 once R.2's reference is written, a swap test once its components
            # pass): try the gate before starting an agent.
            existing = any((self.spec.repo / p).exists() for p in task.get("paths") or [])
            pre = self.gate(tid) if existing and task.get("precheck", True) else None
            if (pre is None or pre.verdict != "PASS") and not self.attempt_loop(
                task, role, self.failure_text(pre) if pre else ""
            ):
                return STOPPED
        if self.needs_human(task):
            return self._human(task, self.needs_human(task))
        return self._after(task) if self.led.status(tid) == "PASS" else STOPPED

    def _after(self, task: dict) -> int:
        if task["id"] == "X.2" or task.get("stop_after"):
            return self._human(
                task,
                f"pick opportunities in {rel(self.spec, self.spec.bringup_dir)}/opportunities.md; "
                "each pick becomes a perf task in tasks.yaml",
            )
        return DONE

    def _human(self, task: dict, why: str) -> int:
        self.echo(f"[{task['id']}] WAITING FOR A PERSON: {why}")
        self.led.update(task["id"], waiting=why)
        return HUMAN

    def run(self, until: str | None = None, only: str | None = None) -> int:
        stop_at = set(self.led.downstream(until)[1:]) if until else set()
        while True:
            state = self.led.state()
            runnable = [
                t for t in self.led.runnable() if state.get(t, {}).get("status") != "STOPPED" and t not in stop_at
            ]
            if only:
                runnable = [t for t in runnable if t == only]
            if not runnable:
                stopped = [t for t in self.led.tasks() if state.get(t, {}).get("status") == "STOPPED"]
                if stopped:
                    self.echo(f"stopped tasks: {stopped} (fix, then `resume`)")
                    return STOPPED
                self.echo("nothing left to run" + (f" before {until}" if until else ""))
                return DONE
            tid = runnable[0]
            rc = self.step(tid)
            if rc != DONE:
                return rc
            if only:
                return DONE


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "resume"])
    ap.add_argument("--spec", default=os.environ.get("BRINGUP_SPEC"))
    ap.add_argument("--until", help="stop after this task")
    ap.add_argument("--only", help="run just this task (if runnable)")
    ap.add_argument("--model")
    ap.add_argument("--attempts", type=int, default=3)
    a = ap.parse_args(argv)
    spec = Spec.load(a.spec)
    orch = Orchestrator(spec, model=a.model, max_attempts=a.attempts, debugger_attempts=a.attempts)
    if a.command == "resume":
        led = orch.led
        for tid, e in led.state().items():
            if isinstance(e, dict) and e.get("status") in ("STOPPED", "RUNNING"):  # RUNNING = an interrupted run
                led.update(tid, status="TODO", history_add={"t": now(), "status": "RESUMED"})
    return orch.run(until=a.until, only=a.only)


if __name__ == "__main__":
    sys.exit(main())
