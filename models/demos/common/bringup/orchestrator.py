# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The orchestrator: runs every step of a bring-up in dependency order and stops only when a person is needed.

    python -m models.demos.common.bringup.orchestrator run --spec S [--until ID] [--only ID] [--model M]
                                                          --until: that task and what it needs, then stop
    python -m models.demos.common.bringup.orchestrator resume --spec S      # STOPPED tasks become runnable again
    python -m models.demos.common.bringup.orchestrator decide --task P.3 --accept|--reject|--rerun --spec S
    python -m models.demos.common.bringup.watch --spec S       # the supervisor's wait: returns on the next event

Per task:
  scripted step (goldens, box, integrate, perf; no role)  run the gate; on failure hand the log to a ``fix`` agent
  agent step (reference, plan, contract, implement)       write a brief, start a fresh ``claude -p`` with the
                                                           bringup-engineer definition, check what it did, run the gate
  tests declared and not frozen                            render the template, ``test`` role reviews it, freeze
                                                           (reference passes, zero stub fails), then implement.
                                                           A swap test is frozen without the review (F49: it gates
                                                           every swapped step itself) unless agents.swap_review
                                                           names its block type; a failed freeze starts the review.
                                                           A component test (CHECKS="auto", F56) likewise when
                                                           agents.component_review leaves its block type out
                                                           (default all: reviewed) and its mistake sweep passes
  failed attempts (DEFAULT_POLICY)                         implement / device fix: WIP commit, then ``ttnn-expert-debugger``
                                                           (TTNN only) with the WIP sha, logs and triage; other roles
                                                           (reference, plan, contract, test): STOPPED for a person
  still failing                                            the task becomes STOPPED and the run stops (exit 1)
  deferral (F46; component tasks, implement role only)     an implement attempt whose gate fails but which wrote a
                                                           valid op request (plan/op_request.py check) makes the task
                                                           DEFERRED: the request and the bridge wiring are committed,
                                                           dependents run, the step stays on the CPU bridge. A rejected
                                                           request is a failed attempt. After the debugger's attempts,
                                                           one last implement attempt may still defer instead of STOPPED
  approval needed (plan) or opportunity list written       the run stops for a person (exit 3)
  gate passed, commit refused (F149)                       COMMIT_PENDING: the run stops (exit 1), no agent starts;
                                                           on resume the commit alone is retried, nothing re-runs
  perf pick (step perf with a role, F57)                   no gate before or after the agent. The agent makes the
                                                           change behind its switch and runs the frozen tests once
                                                           (accuracy first, roles.yaml). Then the orchestrator runs
                                                           the A/B report: per config, off (tasks.yaml ``ab.env``)
                                                           and on, every frozen test of the gate, the ladder at the
                                                           ``ab_rungs`` (default: the gate's rung and the spec's last
                                                           rung) and one plain profile, each into
                                                           runs/<run>/ab/<task>/<old|new>/<name>/, and writes
                                                           table.md there and the record to state.json (``ab``). The
                                                           task waits for the owner (exit 3): ``decide --accept`` runs
                                                           the gate on resume and commits on PASS (dropping a failing
                                                           frozen test from the gate is the owner's edit);
                                                           ``--reject`` reverts the change, the task becomes REJECTED
                                                           (done: dependents run); ``--rerun`` measures again. A new
                                                           brief or switch voids the report (the agent runs again)

After every agent step: the tree diff must stay inside the paths the brief allowed, no command may reach the device
except through the safe runners, and the knowledge files must keep their format. A violation fails the attempt.
Each agent run is recorded in state.json under the task: role, attempt, session id, model, agent-definition hashes.

BRINGUP_AGENT_CMD replaces the ``claude`` executable (tests use a mock agent). Exit codes: 0 done (possibly "complete
with N deferred": those steps run on the CPU until op-gen delivers their ops; ``op-ready`` brings them in), 1 stopped on a
failure (including a box that fails at device open: reset it, then resume), 3 waiting for a person, 4 paused
(``pause`` asks a running orchestrator to stop before its next task; edit the framework only while paused).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from string import Template

import yaml

from models.demos.common.bringup import overseer as OV
from models.demos.common.bringup import timing as TL
from models.demos.common.bringup import watch as W
from models.demos.common.bringup.core import checkout_env, defaults
from models.demos.common.bringup.core import freeze as F
from models.demos.common.bringup.core import metrics as M
from models.demos.common.bringup.core import stamps
from models.demos.common.bringup.core.gate import (
    SAFE_HANG_RC,
    _is_direct_pytest,
    _program,
    _segments,
    commit_only,
    format_paths,
    gate_command,
    gate_env,
    gate_outputs,
    gate_units,
    git_commit,
    log_dir,
    run_gate,
    run_name,
    stage_paths,
    unit_base,
)
from models.demos.common.bringup.core.ledger import Ledger, satisfied
from models.demos.common.bringup.core.runs import FreezeError, agent_hashes, freeze_task
from models.demos.common.bringup.core.spec import CODE_ROOT, Spec
from models.demos.common.bringup.knowledge import check as kcheck
from models.demos.common.bringup.knowledge import usage as kusage
from models.demos.common.bringup.plan import approvals
from models.demos.common.bringup.plan import op_request as OR
from models.demos.common.bringup.testing import accuracy_guard, cards, model_precision, ops_lint
from models.demos.common.bringup.testing import serving as SV
from models.demos.common.bringup.testing import settings_lint
from models.demos.common.bringup.testing.noise_floor import TAPS_FILE

HERE = Path(__file__).resolve().parent
AGENT_DEF = HERE / "agents" / "bringup-engineer.md"
# Roles run by their own agent definition in agents/ (default: bringup-engineer).
AGENT_OF_ROLE = {"serving": "serving-contract", "settings": "settings-audit"}
DEBUGGER = "ttnn-expert-debugger"
DEBUGGER_DEF = CODE_ROOT / ".claude" / "agents" / f"{DEBUGGER}.md"
# Retry budget and escalation per role. ttnn-expert-debugger is specialized for TTNN ops (hangs, CB sync, kernel
# numerics), so only roles whose code is TTNN device code escalate to it: implement, and fix after a device gate.
# Everything else (reference, plan, contract, test, fix after a CPU gate) stops for a person with the logs.
# The spec overrides per role: agents.policy.<role>: {attempts, escalate: debugger | stop, debugger_attempts}.
# F56: component review default; a spec sets agents.component_review: all to review every component test
COMPONENT_REVIEW_DEFAULT = defaults.get("agents.component_review")  # defaults.yaml
DEFAULT_POLICY = defaults.get("agents.policy")  # defaults.yaml: per role attempts, escalate, debugger_attempts
ROLE_OF_STEP = {
    "reference": "reference",
    "plan": "plan",
    "implement": "implement",
    "assemble": "assemble",
    "contract": "contract",
    "serving": "serving",
    "settings": "settings",
    "scale": "scale",
}
IGNORED = (
    r"/dashboard/[^/]+\.(html|yaml)$",  # the dashboards and their links (F71)
    r"(^|/)bringup/(ledgers/[^/]+/)?(overseer\.yaml|supervision\.md)$",  # the overseer's own files (F69; a seeded run's ledger too, F189)
    r"(^|/)bringup/(ledgers/[^/]+/)?history_state\.json$",  # F180: the attempt / A/B histories beside state.json (uncommitted)
    r"(^|/)__pycache__/",
    r"\.pyc$",
    r"^generated/",
    r"/probes/probe_\d+\.py$",
    r"(^|/)\.lock$",
    r"\.tmp$",
)
DONE, STOPPED, HUMAN, PAUSED = 0, 1, 3, 4
CRASH_RETRIES, CRASH_WAIT_S = 2, 60  # F170: an agent that died without changing anything is retried, not charged
# A gate that fails at device open is a box problem (a board reset is needed), not the task's: retrying it only burns
# attempts. These signatures stop the run at once with a reason for the person.
INFRA_FAILURES = re.compile(
    r"Timed out while waiting for active ethernet core|Try resetting the board|No Tenstorrent devices|"
    r"Failed to open device|fabric router .* timed out|Device \d+: .*timed out waiting for .*firmware",
    re.I,
)
# The prefill engine the contract step plugs a model into; its agent may change it (owner, F29).
CONTRACT_SHARED = "models/demos/common/prefill"
# A spec's ``prior`` (an earlier bring-up of the same checkpoint on another mesh or configuration): per role, the prior's
# files each brief lists (relative to the prior's model dir; a folder means its contents), and what to do with them.
PRIOR_READS = {
    "reference": ["bringup/hooks.py", "reference/"],
    "plan": [
        "bringup/plan.md",
        "bringup/plan.yaml",
        "bringup/components.yaml",
        "bringup/findings.yaml",
        "bringup/known_issues.md",
        "bringup/repo_map.md",
    ],
    "test": ["tests/bringup/"],
    "implement": ["tt/", "bringup/BREADCRUMBS.md", "bringup/known_issues.md", "bringup/repo_map.md"],
    "assemble": ["tt/model.py", "bringup/hooks.py"],
    "fix": ["tt/", "bringup/BREADCRUMBS.md", "bringup/known_issues.md"],
    "contract": ["tt/runners/"],
    "perf": ["bringup/opportunities.md", "bringup/tasks.yaml", "bringup/results/X.3_profile.json"],
    "optests": ["bringup/results/fork_calls.json"],
}
PRIOR_TEXT = {
    "reference": "The CPU reference is the prior's: this model's hooks call the prior's `reference` (and tokenizer / "
    "HF hooks). Do not copy it; change it there only for a bug, which then applies to both.",
    "plan": "Start from the prior's plan and re-plan for this mesh ({mesh}, the prior ran {prior_mesh}). For each "
    "component say what changes (sharding, collectives and their axes, MoE dispatch groups, memory per chip) and why, "
    "and what carries over unchanged. A collective or op the new layout needs that TTNN lacks or cannot do goes through "
    "agent rule 6 (a fork or a new op in ttnn/ttnn/bringup), never an edit of an existing op.",
    "test": "The prior's frozen test for the same component shows the comparison mode and the structural checks that "
    "mattered; its numeric limits are not reused (limits come from this run's noise floor).",
    "implement": "Start from the prior's module for the same component (in its `tt/`): copy it into this model's "
    "`tt/` and change what this plan changes. Do not import from the prior's `tt/`: each bring-up's device code "
    "stands alone.",
    "assemble": "The prior's `tt/model.py` shows how its validated modules were assembled; do the same here.",
    "fix": "The prior's module for the component that broke, and its breadcrumbs, show what worked on {prior_mesh}; "
    "the difference is usually the new layout.",
    "contract": "The prior's runners show the adapter and KV layout for {prior_mesh}; the address table and layout "
    "change with the mesh.",
    "perf": "The prior's opportunities, perf picks (P.* in its tasks.yaml) and final profile show what paid off on "
    "{prior_mesh}; measure before assuming the same here.",
    "optests": "The prior's fork_calls.json shows which forks it used; this model may need different ones, or none.",
}

# Forked TTNN ops of bring-ups (agent rule 6): every step that writes code may fork or extend one there.
BRINGUP_OPS = "ttnn/ttnn/bringup"
# A test killed by pytest-timeout ran out of time, it did not fail a check: an agent cannot fix that from the log.
TEST_TIMEOUT = re.compile(r"Timeout \(>[\d.]+s\) from pytest-timeout")
# F207: a measuring process that died of a signal (not the change: the run never reached a result) is run once more
PROCESS_CRASH = re.compile(
    r"Fatal Python error: (Bus error|Segmentation fault|Aborted)|\b(Bus error|Segmentation fault) \(core dumped\)"
)
# Perf picks (F57): the A/B report's lines. A component / swap check that fails prints "FAIL <metric>: <values> (...)".
FAILED_CHECK = re.compile(r"^FAIL\s+\S.*$", re.M)
LADDER_TEST = "models/demos/common/bringup/tests/test_ladder.py"
PROFILE_TEST = "models/demos/common/bringup/tests/test_profile.py"
AB_ROWS = (  # (row, ladder metric); pcc_layer_* is the minimum over the layers
    ("min layer PCC", "pcc_layer_*"),
    ("final hidden PCC", "pcc_final_hidden"),
    ("logits PCC (tail)", "pcc_logits_tail"),
    ("top1", "top1_match"),
    ("top5", "top5_overlap"),
)


def ab_limits(got: dict) -> dict:
    """F131: {metric: [value, limit]} of every metric the run recorded a limit for."""
    return {k[: -len("_limit")]: [got.get(k[: -len("_limit")]), v] for k, v in got.items() if k.endswith("_limit")}


def margin_used(metric: str, v, lim) -> float | None:
    """The share of the margin a value uses (> 1 fails): (1 - v) / (1 - limit) for a score, v / limit for an error."""
    if v is None or lim is None:
        return None
    if metric.startswith("err_"):
        return v / lim if lim else float("inf") if v else 0.0
    return (1 - v) / (1 - lim) if lim < 1 else float("inf") if v < 1 else 0.0


def limits_table(runs: list[tuple[str, dict]]) -> list[str]:
    """F131: value / limit (share of the margin used) per metric, for each (column, result) that recorded limits;
    worst first."""
    runs = [(c, r.get("limits") or {}) for c, r in runs]
    keys = list(dict.fromkeys(k for _, lim in runs for k in lim))
    if not keys:
        return []
    worst = lambda k: max((margin_used(k, *lim[k]) or 0) for _, lim in runs if k in lim)  # noqa: E731
    out = ["", "## Accuracy vs the recorded limits (value / limit, share of the margin used; > 1 fails)", ""]
    out += ["| metric | " + " | ".join(c for c, _ in runs) + " |", "|---|" + "---|" * len(runs)]
    for k in sorted(keys, key=worst, reverse=True):
        cells = []
        for _, lim in runs:
            v, lm = lim.get(k, (None, None))
            u = margin_used(k, v, lm)
            cells.append("-" if u is None else f"{v:.5f} / {lm:.5f} ({u:.2f})")
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    return out


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def rel(spec: Spec, p) -> str:
    p = Path(p)
    return str(p.resolve().relative_to(spec.repo)) if p.is_absolute() else str(p)


# ---------------------------------------------------------------- agent definitions and command
def agents_json(run_dir: Path, agent: str = "bringup-engineer") -> Path:
    """The --agents file for claude -p, built from the markdown definition (frontmatter + body) in agents/."""
    text = (AGENT_DEF.parent / f"{agent}.md").read_text()
    _, fm, body = text.split("---", 2)
    meta = yaml.safe_load(fm)
    out = run_dir / ("agents.json" if agent == "bringup-engineer" else f"agents.{agent}.json")
    out.write_text(
        json.dumps(
            {
                meta["name"]: {
                    "description": meta["description"],
                    "prompt": body.strip(),
                    "tools": [t.strip() for t in meta["tools"].split(",")],
                    **({"model": meta["model"]} if meta.get("model") not in (None, "inherit") else {}),
                }
            },
            indent=1,
        )
    )
    return out


def _killpg(pgid: int, sig) -> None:
    from models.demos.common.bringup.core.hangdump import wait_triage

    wait_triage()  # F204: an agent's device run that hung is writing its triage: never cut it off
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def agent_command(spec: Spec, run_dir: Path, agent: str, prompt: str, model: str | None) -> list[str]:
    exe = shlex.split(os.environ.get("BRINGUP_AGENT_CMD", "claude"))
    cmd = exe + ["-p", "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions"]
    if (AGENT_DEF.parent / f"{agent}.md").exists():  # the framework's own agents (bringup-engineer, serving-contract)
        cmd += ["--agents", str(agents_json(run_dir, agent))]
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


# Opening a device is what needs the safe runner's lock; importing ttnn does not grab the box.
DEVICE_OPEN = re.compile(
    r"open_mesh_device|open_device\s*\(|CreateDevice|ttnn\.MeshDevice\s*\(|ttnn\.open_|synchronize_device|"
    r"get_device\s*\(|mesh_device\s*=\s*ttnn"
)


def code_opens_device(code: str, repo: Path) -> bool:
    """True if Python source (a heredoc, -c code or a script) opens a Tenstorrent device itself. Only real code counts:
    a script that edits a source file carries device calls inside string literals, and that is not device access."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return bool(DEVICE_OPEN.search(code))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and DEVICE_OPEN.search(ast.unparse(node.func) + "("):
            return True
    return False


_HEREDOC = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?[^\n]*\n(.*?)\n\1\b", re.S)


def command_violations(commands: list[str], repo: Path) -> list[str]:
    """Commands that reach the device without the safe runners: any direct pytest, or python whose own code (a script
    file, -c code, a -m module or a heredoc) opens a device. Importing ttnn or reading goldens is not device access."""
    out = []
    for cmd in commands:
        heredocs = [m.group(2) for m in _HEREDOC.finditer(cmd)]
        first_line = cmd.split("\n", 1)[0]
        try:
            segs = _segments(first_line)
        except ValueError:
            segs = [first_line.split()]
        for words in segs:
            w = _program(words)
            if not w:
                continue
            if _is_direct_pytest(w):
                out.append(f"direct pytest: {cmd[:120]!r}")
            elif Path(w[0]).name.startswith("python"):
                if "-c" in w:
                    code = w[w.index("-c") + 1] if w.index("-c") + 1 < len(w) else ""
                elif "-m" in w:
                    mod = w[w.index("-m") + 1] if w.index("-m") + 1 < len(w) else ""
                    base = repo / Path(*mod.split("."))
                    f = base.with_suffix(".py") if base.with_suffix(".py").exists() else base / "__main__.py"
                    code = f.read_text(errors="replace") if f.exists() else ""
                elif any(x.endswith(".py") for x in w[1:]):
                    f = repo / next(x for x in w[1:] if x.endswith(".py"))
                    code = f.read_text(errors="replace") if f.exists() else ""
                else:  # python - <<EOF
                    code = "\n".join(heredocs)
                touches = code_opens_device(code, repo)
                if touches:
                    out.append(f"python on device code without a safe runner: {cmd[:120]!r}")
    return sorted(set(out))


def ab_switch_text(task: dict) -> str:
    """A perf pick's switch as the brief names it: tasks.yaml ab.change (the change) and ab.env (the old path)."""
    ab = task.get("ab") or {}
    on, off = ab.get("change") or {}, ab.get("env") or {}
    if not on and not off:
        return "a named setting or an env var"
    fmt = lambda e: " ".join(f"{k}={v}" for k, v in e.items())  # noqa: E731
    return f"change on: {fmt(on) or '(the default)'}; off (the old path): {fmt(off)}"


def ab_switch_missing(task: dict, repo: Path, changed: list[str], names: set | None = None) -> list[str]:
    """Switch names of ab.env / ab.change that no changed file (or the task's paths) mentions: the switch would do nothing.
    ``names``: only these (F103: a model setting's name resolves to its variable instead, model_precision.switch_env).
    """
    if names is None:
        names = set((task.get("ab") or {}).get("env") or {}) | set((task.get("ab") or {}).get("change") or {})
    files = [repo / p for p in changed if (repo / p).is_file()]
    for pat in task.get("paths") or []:
        base = repo / pat
        files += [f for f in (base.rglob("*.py") if base.is_dir() else [base]) if f.is_file()]
    text = "".join(
        f.read_text(errors="replace") for f in dict.fromkeys(files) if f.suffix in (".py", ".cpp", ".hpp", ".h")
    )
    return sorted(n for n in names if n not in text)


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
class FrameworkStop(Exception):
    """F170 (audit item 4): a gate failed inside the framework's own code (a programming error, e.g. a renamed key):
    an agent cannot change framework files, so the run stops for the overseer instead of starting attempts."""

    def __init__(self, tid: str, where: str):
        super().__init__(f"{tid}: {where}")
        self.tid, self.where = tid, where


FRAMEWORK_CODE = re.compile(r"(^|/)models/demos/common/bringup/(core|testing|tests|reference)/[\w/]+\.py$")
PROGRAMMING_ERRORS = (
    "KeyError",
    "AttributeError",
    "TypeError",
    "NameError",
    "ImportError",
    "ModuleNotFoundError",
    "SyntaxError",
    "IndentationError",
    "UnboundLocalError",
)
PYTEST_FRAME = re.compile(r"^([\w./-]+\.py):(\d+): (\w+)\s*$", re.M)  # pytest: the innermost entry's "path:line: Exc"
PY_FRAME = re.compile(r'File "([^"]+\.py)", line (\d+)')
PY_EXC = re.compile(r"^(\w+(?:Error|Exception)):", re.M)


def framework_error(text: str) -> str | None:
    """F170: 'file:line: Exc' when a log's innermost traceback frame is the framework's code and the exception is a
    programming error (not a failed check: an AssertionError or pytest Failed is the model's)."""
    m = list(PYTEST_FRAME.finditer(text))
    if m:
        path, line, exc = m[-1].groups()
    else:
        frames, excs = list(PY_FRAME.finditer(text)), list(PY_EXC.finditer(text))
        if not frames or not excs:
            return None
        (path, line), exc = frames[-1].groups(), excs[-1].group(1)
    if exc in PROGRAMMING_ERRORS and FRAMEWORK_CODE.search(path):
        return f"{path}:{line}: {exc}"
    return None


class InfraStop(Exception):
    def __init__(self, tid: str, sig: str):
        super().__init__(sig)
        self.tid, self.sig = tid, sig


class CommitPending(Exception):
    """F149: a gate passed and git refused its commit: the run stops, no agent starts, nothing re-runs."""

    def __init__(self, tid: str, text: str):
        super().__init__(text)
        self.tid, self.text = tid, text


def ab_finished(result: dict) -> bool:
    """F228: a kept A/B run is reused only if it finished: it passed, or it recorded a measurement or a failed check.
    A run killed before any test ran (DeepSeek-V4.1 P.16, 2026-10-30: the 'on' side SIGKILLed at model load, rc 1, no
    metric) is run again by `decide --rerun`, never reused as 'same code and switch'."""
    if result.get("rc") == 0 or result.get("checks"):
        return True
    return any(v is not None for k, v in result.items() if k not in ("rc", "limits", "checks"))


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
        self.spec, self.led = spec, Ledger(spec.ledger_dir)
        self.model = model or spec.get("agents.model")
        self.max_attempts, self.debugger_attempts = max_attempts, debugger_attempts
        self.attempts_override = None
        self.timeout_s = timeout_s or int(spec.get("agents.timeout_s"))
        self.run_dir = spec.run_dir(run_name(self.led))
        self.shown: dict[str, list] = {}  # brief name -> the overseer notes it showed (F69)
        self.aborted: dict | None = None  # the overseer's abort of the last agent run (F69)
        (self.run_dir / "briefs").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "agents").mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self.roles = yaml.safe_load((HERE / "briefs" / "roles.yaml").read_text())
        self.pause_file = self.run_dir / "PAUSE"
        self.last_changed: list[str] = []  # the files the last agent run changed
        self.crashed = False  # F170: the last agent run died (rc != 0 / is_error), not by the overseer's abort
        self.last_gate: dict = {}  # F170: {task: (code key, GateResult)} of its last gate, to not re-measure a FAIL

    # ---- policy
    def common_paths(self) -> list[str]:
        b = rel(self.spec, self.spec.bringup_dir)
        # The shared knowledge files are read-only for agents: lessons go to the model's own files, and the overseer
        # promotes the durable ones when the run ends.
        return [
            f"{b}/known_issues.md",
            f"{b}/known_issues_detail.md",
            f"{b}/repo_map.md",
            f"{b}/BREADCRUMBS.md",
            f"{b}/findings.yaml",
            f"{rel(self.spec, self.spec.ledger_dir)}/{kusage.REPORT_DIR}",  # the step's knowledge-use report (the brief names it)
        ]

    def place_fixes(self) -> bool:
        return _place_fixes(self)

    def allowed_paths(self, task: dict, role: str) -> list[str]:
        b = rel(self.spec, self.spec.bringup_dir)
        if role == "test":
            return list(task.get("tests") or []) + self.common_paths()
        extra = (
            [f"{b}/plan.yaml", f"{b}/plan.md", f"{b}/components.yaml", f"{b}/tasks.yaml", f"{b}/scaling.yaml"]
            if role == "plan"
            else []
        )
        if role == "settings":  # the settings audit moves switches; it may touch the model's code and the forks
            md = rel(self.spec, self.spec.model_dir)
            return [f"{md}/tt", f"{md}/bringup/hooks.py", f"{b}/BREADCRUMBS.md", BRINGUP_OPS] + self.common_paths()
        if role == "serving":  # the how-to, its test list and the tests; never model code
            return [
                f"{b}/serving_contract.md",
                f"{b}/contract_tests.yaml",
                f"{rel(self.spec, self.spec.model_dir)}/tests/bringup/contract",
            ] + self.common_paths()
        if task.get("step") == "contract":
            extra.append(CONTRACT_SHARED)  # the engine's producer and registry learn each new model's layout
            extra.append(f"{b}/hooks.py")  # contract_state_pcc (fixed-size state read-back) is a hook
        extra.append(BRINGUP_OPS)
        if role in ("reference", OV.FIX_ROLE):  # F80: the official dtypes of every step's inputs; it sets the floors
            extra.append(f"{b}/{TAPS_FILE}")
        if role == "implement" and OR.deferrable(task):
            extra.append(rel(self.spec, OR.root(self.spec)))  # an op request, if the agent defers the step (F46)
        # F55: the files this task's gate writes are the agent's to change too (one list: core.gate.gate_outputs), so
        # an agent that runs its own gate command is never charged for them; the gate deletes and rewrites them.
        extra += [rel(self.spec, p) for p in gate_outputs(self.led, task)]
        return list(task.get("paths") or []) + extra + self.common_paths()

    # ---- briefs
    def brief(
        self, task: dict, role: str, attempt: int, previous: str = "", extra_read: list[str] = (), defer: bool = True
    ) -> Path:
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
            "ledger_dir": rel(s, s.ledger_dir),
            "hf_dir": str(s.hf_dir),
            "mesh": "x".join(map(str, s.mesh)),
            "chips": s.mesh[0] * s.mesh[1],
            "dram": s.get("box.chip_dram_gb"),
            "tests": ", ".join(task.get("tests") or []),
            "component_desc": desc,
            "component_entry": yaml.safe_dump(comp).strip() if comp else "(none in components.yaml)",
            "gate_cmd": task["gate"]["cmd"],
            "gate_check": (  # F170: the agent runs its gate through the framework, so every unit it passes is kept
                f"PYTHONPATH={s.repo} python -m models.demos.common.bringup gate {task['id']} "
                f"--spec {rel(s, s.path) if s.path else '<spec>'} --check"
            ),
            "gate_units": "\n".join(f"- `{u}`: `{c}`" for u, c in gate_units(task, self.led.tasks())),
            "details": brief.get("details", ""),
            "gate_tasks": ", ".join(brief.get("gate_tasks") or []),
            "task_id": task["id"],
            "repo": str(s.repo),
            "breadcrumbs": f"{rel(s, b)}/BREADCRUMBS.md",
            "knowledge_report": rel(s, kusage.report_path(b, task["id"], role, attempt)),
            "ab_switch": ab_switch_text(task),
            "scaled_hf": str(s.scaled_hf_dir) if s.has_deploy else "",
        }
        can_defer = defer and role == "implement" and OR.deferrable(task)  # never the debugger (it cannot defer)
        vals["defer"] = self.defer_text(task, comp, vals) if can_defer else ""
        vals["deferred"] = self.deferred_text()
        vals["serving"] = self.serving_text(task, role)
        vals["boxes"] = self.boxes_text(role)
        vals["role_text"] = Template(self.roles[role]).safe_substitute(vals)
        # spec agents.read.<role>: files every brief of that role lists (e.g. the HF modeling code for the reference role)
        reads = list(task.get("tests") or []) + list(extra_read) + list(brief.get("read") or [])
        reads += [r for r in (s.get(f"agents.read.{role}") or []) if r not in reads]
        vals["prior"] = ""
        if s.prior:
            ps, pdir = s.prior_spec(), rel(s, s.prior)
            reads += [f"{pdir}/{r}" for r in PRIOR_READS.get(role, []) if (s.prior / r).exists()]
            text = PRIOR_TEXT.get(role, "").format(
                mesh="x".join(map(str, s.mesh)), prior_mesh="x".join(map(str, ps.mesh))
            )
            vals["prior"] = (
                f"## Prior bring-up\nThis checkpoint was brought up before as `{pdir}` (mesh "
                f"{'x'.join(map(str, ps.mesh))}). Its goldens and CPU reference are shared with this one. {text}\n"
            )
        if s.seed:  # F184: a run grown from a seed (same model dir, same tt/)
            ss = s.seed_spec()
            vals |= {"seed_spec": rel(s, ss.path), "seed_mesh": "x".join(map(str, ss.mesh))}
            vals["prior"] += (
                f"## Seed\nThis run grows the bring-up `{rel(s, ss.path)}` (mesh {vals['seed_mesh']}, layers "
                f"{ss.layers()}) to {vals['mesh']}: the same `{vals['model_dir']}/tt`, hooks, CPU reference, plan, serving "
                "contract and forks. Its reference, contract and plan tasks passed there and are not re-run. Change the "
                "seed's code in place for both boxes; never copy it.\n"
            )
            vals["role_text"] = Template(self.roles[role]).safe_substitute(vals)
        vals["read_list"] = "\n".join(f"- `{r}`" for r in reads)
        rules = s.get("agents.rules") or []
        vals["rules"] = (
            ("## Rules from the owner (must follow)\n" + "\n".join(f"- {r}" for r in rules) + "\n") if rules else ""
        )
        vals["allowed"] = "\n".join(f"- `{p}`" for p in self.allowed_paths(task, role))
        vals["thresholds"] = (
            "\n".join(f"- `{k}` {v}" for k, v in (task["gate"].get("metrics") or {}).items()) or "- (exit code only)"
        )
        vals["previous"] = f"## Previous attempt failed\n```\n{previous[-6000:]}\n```\n" if previous else ""
        notes = OV.for_task(s.ledger_dir, task["id"], role)
        vals["overseer"] = OV.brief_text(notes)

        text = Template((HERE / "briefs" / "brief.md").read_text()).safe_substitute(vals)
        p = self.run_dir / "briefs" / f"{task['id']}.{role}.{attempt}.md"
        p.write_text(text)
        self.shown[p.name] = notes
        return p

    def defer_text(self, task: dict, comp: dict | None, vals: dict) -> str:
        """How an implement agent defers a component step to op-gen (F46); the plan's OPGEN tag makes it the task."""
        head = (
            "The plan tagged this step OPGEN (components entry above): TTNN has no proper op for it. Do not implement "
            "it on the device; defer it from the start, as below. The plan's `searched` is where your evidence starts."
            if comp and comp.get("tag") == "OPGEN"
            else "If TTNN has no proper op for this step (no fork of an existing op fits and no composition of TTNN ops "
            "works), you may defer it instead of implementing it, on any attempt. Deferring to skip a hard step is "
            "cheating: the overseer reviews every deferral like a gate commit and reverts a thin one."
        )
        cmd = "python -m models.demos.common.bringup.plan.op_request"
        return (
            f"## Deferring this step to op-gen\n{head}\n\n"
            f"1. `{cmd} new <op> --task {task['id']} --spec {vals['spec_path']}` writes "
            f"`{vals['ledger_dir']}/{OR.REQUESTS}/<op>/` (one snake_case op name): request.yaml, op_prompt.txt, "
            "feature_spec.py, reference.py, bind.py, with the shapes, tolerance and acceptance test filled in.\n"
            f"2. Replace every `{OR.MARK}` marker. request.yaml `evidence`: `searched` (the repo map rows and greps you "
            "checked), `tried` (each op, composition or fork you tried, with the gate or test outcome quoted), "
            '`why_not_fork`. op_prompt.txt: the math, the signature, and `## Rules` lines ("When X: MUST / MUST NOT '
            '..."). reference.py: `pytorch_<op>`, the step as standalone pure torch (no model imports). bind.py: the '
            "layer's weights and scalars it takes beyond the step's inputs (list them in request.yaml too). Set each "
            f"tensor's mesh `placement` in request.yaml, then `{cmd} refresh <dir> --spec {vals['spec_path']}`.\n"
            f"3. `{cmd} check <dir> --spec {vals['spec_path']}` must print `valid`.\n"
            "4. The step stays on the CPU. The swap tests run a deferred step on the reference by themselves; a device "
            "model calls it through `CpuBridge` (`models/demos/common/bringup/testing/cpu_bridge.py`), never through "
            "host code of its own.\n"
            "When this attempt's gate fails and the check passes, the orchestrator marks the task DEFERRED and commits "
            "the request; a request the check rejects counts as a failed attempt. Launching op-gen is the owner's call.\n"
        )

    def boxes_text(self, role: str) -> str:
        """F64: the deployment box this run stands in for, and for the plan role how to plan for it ('' without one)."""
        s = self.spec
        if not s.has_deploy:
            return ""
        mesh = lambda b: "x".join(map(str, s.boxes[b]["mesh"]))  # noqa: E731
        dep, run = s.deploy_box, s.run_box
        rows = "\n".join(
            f"- `{b}`: mesh {mesh(b)}, {s.boxes[b]['chip_dram_gb']} GB per chip"
            + (" (deployment)" if b == dep else "")
            + (" (this run)" if b == run else "")
            for b in s.boxes
        )
        text = (
            f"## Boxes\nThis run uses `{run}` (mesh {mesh(run)}) in {s.mode} mode. The model is planned for the "
            f"deployment box `{dep}` (mesh {mesh(dep)}) at {s.deploy_seq:,} tokens in {s.deploy_chunk:,}-token chunks. "
            "The code must work on every box below, not only this one (agent rule 15).\n" + rows + "\n"
        )
        if s.scaled:  # F75: on a smaller box the whole run is the scaled model
            text += (
                f"\nThis box is smaller than the deployment box, so this run IS the scaled model: the checkpoint at "
                f"`{s.scaled_hf_dir}` (heads and experts cut so each chip holds what a deployment chip holds), its "
                f"goldens, and {s.data['target']['chunk']:,}-token chunks. Read every dimension from that checkpoint's "
                "config; never assume the real model's counts.\n"
            )
        if role == "plan":
            text += (
                "\nPlan for the deployment box, then fit the plan to every box (schema and checks in "
                "`models/demos/common/bringup/plan/boxes.py`). In `plan.yaml`: `roles` (which mesh axes split the "
                "sequence, the heads, the experts, ...); a placement for every tensor of every layer (no `skip` for "
                "layers outside this run's subset: the check applies the subset); state entries with total `heads` and "
                "the roles they are split over (`split_heads`, `split_seq`), not per-chip numbers; and in "
                "`conditions` every per-chip count that decides a code path (an op's limit on experts or heads per "
                "chip, tile alignment of a chip's share of a chunk), with the path taken otherwise in `else`. "
                "Per-box dtype or activation differences go under `boxes`. The gate prints the per-box table (scale, "
                "scaled context, full-model fit and longest context, the run's layers' memory, each "
                "condition's path) and writes `results/plan_boxes.json`. `plan.md` needs a `## Boxes` section with "
                "that table and, for every path that differs from the deployment box, either the change that makes "
                "the box take the deployment's path, or why it stays different and what goes untested until the "
                "deployment box (list those as deployment-only checks).\n"
            )
            from models.demos.common.bringup.plan.boxes import scaling_applies

            if scaling_applies(s):
                text += (
                    f"\nThis run's box has fewer chips than the deployment box, so the whole run is the scaled model "
                    "(F75): heads and experts cut so each chip holds what a deployment chip holds, the deployment chunk "
                    "times the sequence factor; the goldens and every step after the plan use it (task SG.1, "
                    "`models/demos/common/bringup/plan/scale.py`). Write `scaling.yaml` next to the plan: which "
                    "config counts are scaled (by the factor of which role), and for every tensor whose size follows "
                    "one of them, the dimension to cut or the index to keep (per-expert tensors). The gate checks the "
                    "rules against the checkpoint and prints the kept counts.\n"
                )
        return text

    def serving_text(self, task: dict, role: str) -> str:
        """The serving contract this step builds to (testing/serving.py): the plan reads all of it; a component step
        gets its own how-to sections and contract tests; the contract step gets the runner (adapter) part."""
        md, _ = SV.paths(self.spec)
        if not md.exists() or role in ("serving", "test", "reference"):
            return ""
        if role == "plan":
            return (
                f"## Serving contract\n\nBuild to `{rel(self.spec, md)}` (how tt-d-gen drives the model, read from its"
                " code). `plan.md` must have a `## Serving contract` section saying, for each of its sections, how the"
                " planned layout meets it. Do not plan anything it rules out.\n"
            )
        gate = SV.ADAPTER if task.get("step") == "contract" else (task.get("brief") or {}).get("step")
        return SV.brief_text(self.spec, gate) if gate else ""

    def deferred_text(self) -> str:
        state, tasks = self.led.state(), self.led.tasks()
        rows = []
        for tid in self.led.deferred():
            b, d = tasks[tid].get("brief") or {}, state[tid].get("deferred") or {}
            rows.append(f"- {tid}: `{b.get('block_type')}.{b.get('step')}`, op request `{d.get('request')}`")
        if not rows:
            return ""
        return (
            "## Steps deferred to op-gen (on the CPU bridge)\nTTNN has no op for these yet; they stay on the CPU "
            "reference until op-gen delivers one. Do not implement them yourself. A device model calls each through "
            "`CpuBridge` (`models/demos/common/bringup/testing/cpu_bridge.py`): inputs to the host per their mesh "
            "placement, the reference step, the output back with the placement the next step expects. Its transfers are "
            "not counted in `host_transfers_per_layer`; its host time is `deferred_cpu_ms`.\n" + "\n".join(rows) + "\n"
        )

    def try_defer(self, task: dict) -> tuple[bool, str]:
        """(deferred, why not). A valid op request naming this task defers it: DEFERRED, request and bridge committed."""
        tid = task["id"]
        dirs = OR.for_task(self.spec, tid)
        if not dirs:
            return False, ""
        if len(dirs) > 1:
            return False, f"more than one op request names {tid}: {[rel(self.spec, d) for d in dirs]}"
        d = dirs[0]
        format_paths(
            self.spec.repo, [rel(self.spec, d)] + [p for p in task.get("paths") or [] if (self.spec.repo / p).exists()]
        )
        errs = OR.check(self.spec, d)
        if errs:
            self.echo(f"  [{tid}] deferral rejected ({len(errs)} problems): {errs[0]}")
            self.led.update(tid, history_add={"t": now(), "status": "DEFER_REJECTED", "why": errs})
            return False, f"op request {rel(self.spec, d)} rejected by the checker:\n" + "\n".join(
                f"- {e}" for e in errs
            )
        req = OR.load(d)
        why = f"deferred to op-gen: {req['op']} (request {rel(self.spec, d)}); {OR.evidence_summary(req)}"
        self.led.update(
            tid,
            status="DEFERRED",
            deferred={"op": req["op"], "request": rel(self.spec, d), "at": now()},
            reason=[why],
            waiting=None,
            history_add={"t": now(), "status": "DEFERRED"},
        )
        with self.led.locked():
            sha = git_commit(
                self.spec,
                stage_paths(self.spec, self.led, task) + [rel(self.spec, d)],
                f"[{self.spec.tag}][{tid}] {task['title']} (deferred to op-gen: {req['op']})",
                f"Deferred: the step stays on the CPU bridge until op-gen delivers {req['op']}.\n{why}",
            )
        self.echo(f"  [{tid}] DEFERRED to op-gen: {req['op']} ({rel(self.spec, d)})" + (f" -> {sha}" if sha else ""))
        return True, ""

    def _component_entry(self, brief: dict) -> dict | None:
        from models.demos.common.bringup.plan.components import load

        for c in load(self.spec).get("components") or []:
            if c.get("block_type") == brief.get("block_type") and c.get("step") == brief.get("step"):
                return c
        return None

    # ---- one agent run
    def run_agent(
        self,
        task: dict,
        role: str,
        attempt: int,
        brief: Path,
        agent: str = "bringup-engineer",
        report: tuple[str, int] | None = None,
    ) -> list[str]:
        """report: (role, attempt) the brief named the knowledge-use report under, when it differs from this run's
        (F170, audit item 21: the debugger's brief names <tid>.fix.10n.json; it was read as <tid>.debug.n.json)."""
        tid = task["id"]
        before = dirty(self.spec.repo)
        knowledge_before = set(self.knowledge_errors())
        log = self.run_dir / "agents" / f"{tid}.{role}.{attempt}.jsonl"
        cmd = agent_command(self.spec, self.run_dir, agent, f"Read the brief at {brief} and carry it out.", self.model)
        env = dict(os.environ, PYTHONPATH=str(CODE_ROOT), BRINGUP_TASK=tid)
        if self.spec.path:
            env["BRINGUP_SPEC"] = str(self.spec.path)
        self.echo(f"  [{tid}] {role} attempt {attempt}: {agent} (brief {brief.name})")
        t0 = time.time()
        with open(log, "w") as f:
            proc = subprocess.Popen(
                cmd,
                cwd=self.spec.repo,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            OV.agent_started(self.run_dir, proc.pid, tid, role, attempt)
            try:
                rc = proc.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                _killpg(proc.pid, signal.SIGKILL)
                rc = proc.wait()
                rc = -9
            except BaseException:  # the orchestrator is stopped: its agent goes with it
                _killpg(proc.pid, signal.SIGTERM)
                raise
            finally:
                OV.agent_finished(self.run_dir)
        info = parse_stream(log)
        problems = []
        self.aborted = OV.take_abort(self.run_dir)
        # F77: the tree diff before any ledger write of ours (an abort's history entry is not the agent's change)
        self.last_changed = changed_since(before, dirty(self.spec.repo))
        if self.aborted:
            why = self.aborted.get("reason", "")
            problems.append(f"aborted by the overseer: {why}")
            self.led.update(tid, history_add={"t": now(), "status": "ABORTED", "why": why})
            self.echo(f"  [{tid}] ABORTED by the overseer ({role} attempt {attempt}): {why}")
        self.crashed = (rc != 0 or bool(info["is_error"])) and not self.aborted
        if self.crashed:
            problems.append(f"agent exited rc={rc} is_error={info['is_error']}")
        problems += command_violations(info["commands"], self.spec.repo)
        pats = self.allowed_paths(task, role)
        outside = [p for p in self.last_changed if not allowed(p, pats)]
        if outside:
            problems.append(f"changed files outside the allowed paths (revert them): {outside}")
        # one place for a model's switches: no environment read outside tt/settings.py, none in a fork (diagnostics aside)
        problems += [
            f"settings: {v}"
            for v in settings_lint.violations(
                self.spec.repo,
                rel(self.spec, self.spec.model_dir),
                [p for p in self.last_changed if p.endswith((".py", ".cpp", ".hpp", ".h"))],
            )
        ]
        if f"{rel(self.spec, self.spec.model_dir)}/tt/settings.py" in self.last_changed:  # F137: no emulation default
            sets = model_precision.model_settings(self.spec, fresh=True)
            problems += [f"settings: {v}" for v in settings_lint.emulation_violations(sets)]
        # F113: bring-up ops before stock TTNN (a listed stock op needs a BREADCRUMBS reason); a changed op folder keeps
        # its README.md and INDEX.md entry
        problems += [
            f"bring-up ops: {v}"
            for v in ops_lint.violations(
                self.spec.repo,
                rel(self.spec, self.spec.model_dir),
                list(self.last_changed),
                self.spec.bringup_dir / "BREADCRUMBS.md",
            )
        ]
        # F170 (audit item 10): only the knowledge-file errors this attempt introduced are its problems (a pre-existing
        # format error in any model's file failed every attempt and made the debugger reject passes)
        problems += [e for e in self.knowledge_errors() if e not in knowledge_before]
        # The knowledge-use report: validated and cross-checked against what the step touched; a missing or malformed
        # one is recorded (the dashboard shows it), not a failed attempt.
        report = kusage.report_path(self.spec.ledger_dir, tid, *(report or (role, attempt)))
        touched, _, visible = kusage.transcript_touches(log, self.spec.repo)
        knowledge_report = kusage.finalize(
            report,
            {
                "model": self.spec.model_dir.name,
                "run": run_name(self.led),
                "task": tid,
                "role": role,
                "attempt": attempt,
                "agent": agent,
                "t": now(),
            },
            sorted(set(touched) | set(self.last_changed)),
            info["commands"],
            CODE_ROOT,
            visible,
        )
        for w in knowledge_report:
            self.echo(f"  [{tid}] knowledge report: {w}")
        # An entry an agent found wrong is the owner's to know now, not at the run-end tally.
        contradicted = json.loads(report.read_text()).get("contradicted") or []
        for c in contradicted:
            self.echo(f"  [{tid}] KNOWLEDGE CONTRADICTED {c['id']}: {c.get('evidence', '')}")
        if not self.aborted:
            for line in OV.answers(json.loads(report.read_text()), self.shown.get(brief.name, [])):
                self.echo(f"  [{tid}] {line}")
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
            "knowledge_report": knowledge_report or "ok",
            "knowledge_contradicted": [c["id"] for c in contradicted],
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

    def knowledge_errors(self) -> list[str]:
        """The shared and the model's known-issues files' format errors and the knowledge id clashes, as problem lines."""
        ki, _ = kcheck.check_known_issues(kcheck.HERE / "known_issues.md")
        out = [f"known_issues.md: {e}" for e in ki]
        mki, _ = kcheck.check_model_known_issues(self.spec.bringup_dir / "known_issues.md")
        out += [f"{rel(self.spec, self.spec.bringup_dir)}/known_issues.md: {e}" for e in mki]
        ids, _ = kcheck.check_ids(kcheck.knowledge_files(CODE_ROOT))
        return out + [f"knowledge ids: {e}" for e in ids]

    # ---- steps
    def gate(self, tid: str, reuse: dict | None = None):
        def on_unit(k, n, unit, verdict, secs):  # F169: the gate's progress per unit, as it goes (the watch reads it)
            if verdict == "KEPT":
                self.echo(f"  [{tid}] gate unit {k}/{n} {unit} kept (passed before on the same code under test)")
            else:
                self.echo(f"  [{tid}] gate unit {k}/{n} {unit} {verdict} ({secs:.0f} s)")

        key = stamps.code_key(self.spec.repo, self.spec)
        res = run_gate(self.spec, self.led, tid, commit=True, reuse=reuse, on_unit=on_unit)
        sig = self.infra_failure(res) if res.verdict in ("FAIL", "HANG") else None
        if sig and reuse is None:  # F170 (audit items 7, 17): one automatic retry of the stopped unit, not a run stop
            if not TEST_TIMEOUT.search(sig) and not self.box_ok(tid):
                raise InfraStop(tid, sig)
            self.echo(f"  [{tid}] {sig!r}: retrying the stopped unit once (the units that passed are kept)")
            res = run_gate(self.spec, self.led, tid, commit=True, on_unit=on_unit)
        for ln in res.lines:
            if ln.startswith(("  units passed:", "  units remaining:")):
                self.echo(f"  [{tid}]{ln[1:]}")
        if res.verdict == "PASS":
            OV.clear_classification(self.spec.ledger_dir, tid)
        if res.verdict == "COMMIT_PENDING":
            raise CommitPending(tid, res.summary())
        if res.verdict in ("FAIL", "HANG"):
            sig = self.infra_failure(res)
            if sig:
                raise InfraStop(tid, sig)  # still failing after the automatic retry
            fw = self.framework_failure(tid, res)
            if fw:
                raise FrameworkStop(tid, fw)
        self.last_gate[tid] = (key, res)
        if res.reused:  # F150
            self.echo(f"  [{tid}] reused: same code, not run: {', '.join(res.reused)}")
        self.echo(f"  [{tid}] gate {res.verdict}" + (f" -> {res.commit}" if res.commit else ""))
        self.refresh_dashboards()
        return res

    BOX_CHECK = "scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_box.py"

    def gate_unless_unchanged(self, tid: str):
        """F170 (audit item 4): after an agent attempt, the gate runs only if something a gate reads changed since the
        task's last failing gate; else that failure stands (an agent that changed nothing is not measured again)."""
        same = self.last_gate.get(tid)
        if same and same[1].verdict != "PASS" and same[0] == stamps.code_key(self.spec.repo, self.spec):
            self.echo(f"  [{tid}] the tree did not change since the failing gate: no gate re-run, its failure stands")
            return same[1]
        return self.gate(tid)

    def framework_failure(self, tid: str, res) -> str | None:
        """F170: a gate failure inside the framework's code (framework_error) on a task that has passed before (it is
        not the model's missing work: a first-time check before an implementation fails in the framework too)."""
        text = res.log.read_text(errors="replace") if res and res.log and res.log.exists() else ""
        where = framework_error(text)
        if not where:
            return None
        task, state = self.led.task(tid), self.led.state()
        seen = [tid] + list(dict.fromkeys(unit_base(u) for u, _ in gate_units(task, self.led.tasks())))
        passed = any(any(h.get("status") == "PASS" for h in (state.get(t) or {}).get("history") or []) for t in seen)
        return where if passed else None

    def stop_for_framework(self, tid: str, where: str) -> int:
        why = (
            f"the gate failed in the framework's own code ({where}); no agent attempt (it cannot change framework "
            f"files): fix the framework, `gate {tid} --unit <failed unit>`, then resume (F170)"
        )
        self.led.update(tid, status="STOPPED", reason=[why], history_add={"t": now(), "status": "STOPPED"})
        self.echo(f"  [{tid}] STOPPED (framework): {why}")
        return STOPPED

    def box_ok(self, tid: str) -> bool:
        """F170: after a box error, one quick box check through the safe runner (it resets the cards after the run;
        never tt-smi -r): the mesh opens and its collectives work. True: retry the unit; False: a person resets."""
        log = log_dir(self.spec, self.led) / f"{tid}.boxcheck.log"
        self.echo(f"  [{tid}] box error: quick box check ({log.name})")
        return self.run_cmd(self.BOX_CHECK, gate_env(self.spec, self.led, tid), log) == 0

    def refresh_dashboards(self) -> None:
        """Re-export the dashboards after every gate, in the background (BRINGUP_DASHBOARD=0 turns it off)."""
        if os.environ.get("BRINGUP_DASHBOARD", "1") != "1" or not self.spec.path:
            return
        cmd = [sys.executable, "-m", "models.demos.common.bringup.dashboard.export", "--spec", str(self.spec.path)]
        env = dict(os.environ, PYTHONPATH=str(CODE_ROOT))
        subprocess.Popen(
            cmd, cwd=CODE_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
        )

    def failure_text(self, res) -> str:
        tail = res.log.read_text(errors="replace")[-4000:] if res.log and res.log.exists() else ""
        return res.summary() + "\n--- log tail ---\n" + tail

    def swap_review(self, block_type: str) -> bool:
        """Spec ``agents.swap_review``: all | [block types] (default none). F49: the swap template gates every swapped
        step itself, so a swap test is frozen without a test-role review unless the spec names its block type."""
        v = self.spec.get("agents.swap_review")
        if v in (None, False, "none", []):
            return False
        return v in ("all", True) or block_type in (v if isinstance(v, list) else [v])

    def component_review(self, block_type: str) -> bool:
        """Spec ``agents.component_review``: all | none | [block types] (default none, COMPONENT_REVIEW_DEFAULT). F56: a
        component test with the built-in checks is frozen without a test-role review when this leaves its block type
        out, provided its mistake sweep (BRINGUP_IMPL=mutations) passes."""
        v = self.spec.get("agents.component_review")
        if v in (None, False, "none", []):
            return False
        return v in ("all", True) or block_type in (v if isinstance(v, list) else [v])

    def freeze_tests(self, task: dict) -> str | None:
        """Render, review (test role), freeze. None on success, else the reason to stop.

        A swap task skips the review (F49, unless agents.swap_review names its block type) and freezes at once; the
        freeze still requires PASS with the reference and FAIL with the stub. If that freeze fails, the test role is
        started with the failure, as for a reviewed test."""
        from models.demos.common.bringup.testing.templates import render_component_test, render_swap_test

        b = task.get("brief") or {}

        # F95: a test frozen without review IS the template: render it afresh, so the edits of an agent that was
        # aborted or cut off (a crash) never reach the freeze
        fresh = ("step" in b and "swapped" not in b and not self.component_review(b["block_type"])) or (
            "swapped" in b and not self.swap_review(b["block_type"])
        )

        def render():
            if "swapped" in b:
                render_swap_test(self.spec, b["block_type"], b["swapped"], overwrite=fresh)
            elif "step" in b:
                render_component_test(self.spec, b["block_type"], b["step"], overwrite=fresh)

        render()
        previous, unreviewed = "", None  # the kwargs of the freeze without review, when the task has one
        if "step" in b and "swapped" not in b and not self.component_review(b["block_type"]):
            unreviewed = {"mutations": True}
            try:
                rec = freeze_task(self.spec, self.led, task["id"], mutations=True)
                self.echo(
                    f"  [{task['id']}] component test frozen without review (F56): reference={rec.get('reference')} "
                    f"stub={rec.get('stub')} mutations={rec.get('mutations')}"
                )
                return None
            except FreezeError as e:
                previous = str(e)
                self.echo(
                    f"  [{task['id']}] freeze without review failed, starting the test role: {previous.splitlines()[0]}"
                )
        if "swapped" in b and not self.swap_review(b["block_type"]):
            unreviewed = {}
            try:
                rec = freeze_task(self.spec, self.led, task["id"])
                self.echo(
                    f"  [{task['id']}] swap test frozen without review (F49): "
                    f"reference={rec.get('reference')} stub={rec.get('stub')}"
                )
                return None
            except FreezeError as e:
                previous = str(e)
                self.echo(
                    f"  [{task['id']}] freeze without review failed, starting the test role: {previous.splitlines()[0]}"
                )
        budget = self.policy(task, "test")["attempts"]
        attempt, cut = self.attempts_so_far(task["id"], "test")  # F95
        used = aborts = 0
        if cut is not None:
            previous = (
                f"A previous run was interrupted during test attempt {cut} (a crash or power loss); "
                "the test was rendered afresh from its template.\n" + previous
            )
        while used < budget:
            self.pause_point(task["id"], f"test attempt {attempt + 1}")
            attempt += 1
            problems = self.run_agent(task, "test", attempt, self.brief(task, "test", attempt, previous))
            if self.aborted:  # F69
                aborts += 1
                previous = "\n".join(problems)
                if aborts > OV.MAX_ABORTS:
                    return f"aborted by the overseer {aborts} times"
                if unreviewed is not None:  # F94: the overseer may have fixed the check that sent it to an agent
                    try:
                        render()  # the aborted agent's half-edited test is replaced by the template's
                        rec = freeze_task(self.spec, self.led, task["id"], **unreviewed)
                        self.echo(f"  [{task['id']}] frozen without review after the abort (F94)")
                        return None
                    except FreezeError as e:
                        previous = str(e)
                        self.echo(
                            f"  [{task['id']}] freeze without review after the abort failed: {previous.splitlines()[0]}"
                        )
                continue
            used += 1
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
        return f"tests could not be frozen after {budget} attempts: {previous.splitlines()[0] if previous else ''}"

    def needs_human(self, task: dict) -> str | None:
        # A task that declares an approval point (PL.1: plan) waits for a person until that approval is recorded.
        if task.get("approval") == "plan" and not approvals.is_approved(self.spec, "plan"):
            return (
                f"approve the plan in {rel(self.spec, self.spec.bringup_dir)} (plan.yaml, plan.md, components.yaml, tasks.yaml), "
                f"then: python -m models.demos.common.bringup approve plan --spec {self.spec.path}"
            )
        return None

    def policy(self, task: dict, role: str) -> dict:
        p = dict(DEFAULT_POLICY.get(role, {"attempts": 3, "escalate": "stop"}))
        p.update(self.spec.get(f"agents.policy.{role}") or {})
        if self.attempts_override:
            p["attempts"] = p["debugger_attempts"] = self.attempts_override
        if role == "fix" and not task.get("device"):
            p["escalate"] = "stop"  # a failed CPU gate (goldens, plan check) is not a TTNN problem
        return p

    def attempts_so_far(self, tid: str, role: str) -> tuple[int, int | None]:
        """F95: the last attempt number of ``role`` on this task (finished or interrupted), and the attempt a crash
        interrupted (None if the task was not resumed from RUNNING in that role). New attempts continue the numbering,
        so a resumed task never overwrites an earlier attempt's brief or transcript."""
        e = self.led.state().get(tid, {})
        seen = [r["attempt"] for r in e.get("agent_runs", []) if r.get("role") == role and r.get("attempt", 0) < 100]
        resumes = [h for h in e.get("history") or [] if h.get("status") == "RESUMED"]
        last = resumes[-1] if resumes else {}
        after = [r for r in e.get("agent_runs", []) if r.get("role") == role and r.get("t", "") > last.get("t", "")]
        cut = None
        if last.get("from") == "RUNNING" and last.get("role") == role and not after:  # no agent ran since the resume
            cut = last.get("attempt") if isinstance(last.get("attempt"), int) else None
            seen += [cut] if cut is not None and cut < 100 else []
        return max(seen, default=0), cut

    def attempt_loop(self, task: dict, role: str, first_failure: str = "") -> bool:
        previous = first_failure
        pol = self.policy(task, role)
        attempt, cut = self.attempts_so_far(task["id"], role)
        used = aborts = crashes = 0
        if cut is not None:
            previous = (
                f"A previous run was interrupted during {role} attempt {cut} of this task (a crash or power loss). "
                "Its uncommitted changes are still in the tree: check them (git diff), keep what is right, revert "
                "the rest.\n" + previous
            )
        while used < pol["attempts"]:
            self.pause_point(task["id"], f"{role} attempt {attempt + 1}")
            attempt += 1
            # the dashboard shows the agent at work, not the check that ran before it started
            self.led.update(task["id"], status="RUNNING", attempt=attempt, role=role, waiting=None)
            agent = AGENT_OF_ROLE.get(role, "bringup-engineer")
            problems = self.run_agent(task, role, attempt, self.brief(task, role, attempt, previous), agent=agent)
            if self.aborted:  # F69: the overseer stopped it; the next attempt reads its note
                aborts += 1
                previous = "\n".join(problems)
                if aborts > OV.MAX_ABORTS:
                    return self.stop_task(task, f"aborted by the overseer {aborts} times; waiting for a person")
                if self.pause_file.exists():  # F147: abort + pause stops here; no next attempt starts
                    self.echo(f"  [{task['id']}] paused after the overseer's abort; `resume` starts the next attempt")
                    raise SystemExit(PAUSED)
                continue
            if self.crashed and not self.last_changed and crashes < CRASH_RETRIES:
                # F170 (audit item 4): an agent that died at once (an API outage) changed nothing: not an attempt, no gate
                crashes += 1
                attempt_note = f"agent crashed without changes ({crashes}/{CRASH_RETRIES} free retries)"
                self.echo(f"  [{task['id']}] {attempt_note}; retrying in {CRASH_WAIT_S} s, no gate")
                time.sleep(CRASH_WAIT_S)
                continue
            used += 1
            if task.get("approval") and not problems and self.needs_human(task):
                return True  # the plan exists; the gate needs the approval next
            res = self.gate_unless_unchanged(task["id"])
            outside = [p for p in problems if p.startswith("changed files outside")]
            if res.verdict == "PASS" and not outside:
                # A passing gate is not redone for a command problem; the overseer reviews it (state: review).
                if problems:
                    self.led.update(task["id"], review=problems)
                    self.echo(f"  [{task['id']}] passed with problems for review: {len(problems)}")
                return True
            if res.verdict == "PASS":
                return self.stop_passed_out_of_scope(task, problems)
            rejected = ""
            if role == "implement" and OR.deferrable(task) and not outside:
                deferred, rejected = self.try_defer(task)
                if deferred:
                    return True
            previous = "\n".join(problems + ([rejected] if rejected else []) + [self.failure_text(res)])
        if pol["escalate"] == "debugger":
            return self.debug(task, previous, pol["attempts"], pol.get("debugger_attempts", 3))
        why = f"{role} failed {pol['attempts']} attempts; waiting for a person (logs in {self.run_dir / 'agents'})"
        self.led.update(task["id"], status="STOPPED", reason=[why], history_add={"t": now(), "status": "STOPPED"})
        self.echo(f"  [{task['id']}] STOPPED: {why}")
        return False

    def pause_point(self, tid: str, before: str) -> None:
        """F170 (audit item 6): a pause stops the run before the next agent attempt of every loop (implement / fix,
        debugger, last-chance deferral, perf), not only after an abort: the overseer pauses to fix by hand, and no
        agent starts on the failure meanwhile."""
        if self.pause_file.exists():
            self.echo(f"  [{tid}] paused before {before}; `resume` continues the task")
            raise SystemExit(PAUSED)

    def stop_passed_out_of_scope(self, task: dict, problems: list[str]) -> bool:
        """F190: every unit passed and the gate committed, but the attempt changed files outside its paths. No new
        attempt (one cannot do better than a pass; W.1 attempt 2 deleted results/ and edited state.json): the task
        stops for the overseer, who reverts or keeps those files; `resume` then reuses the passed units."""
        self.led.update(task["id"], review=problems)
        return self.stop_task(
            task,
            "gate passed and committed, but the attempt changed files outside its paths: revert or keep them, then "
            "resume (the passed units are reused, no agent starts). " + "; ".join(problems),
        )

    def stop_task(self, task: dict, why: str) -> bool:
        self.led.update(task["id"], status="STOPPED", reason=[why], history_add={"t": now(), "status": "STOPPED"})
        self.echo(f"  [{task['id']}] STOPPED: {why}")
        return False

    def debug(self, task: dict, previous: str, attempts: int, debugger_attempts: int) -> bool:
        tid = task["id"]
        paths = [p for p in task.get("paths") or [] if (self.spec.repo / p).exists()]
        wip = None
        if paths:
            wip = git_commit(
                self.spec,
                paths,
                f"[{self.spec.tag}][{tid}][wip] {task['title']}",
                f"Work in progress after {attempts} failed attempts; handed to {DEBUGGER}.",
            )
        triage = log_dir(self.spec, self.led) / f"{tid}.triage.txt"
        extra = [str(triage)] if triage.exists() else []
        attempt = aborts = 0
        while attempt < debugger_attempts:
            self.pause_point(tid, f"debugger attempt {attempt + 1}")
            attempt += 1
            text = (
                f"WIP commit: {wip or '(no changes to commit)'}\nThe implementer failed {attempts} times; "
                f"the logs are in {self.run_dir / 'agents'} and {log_dir(self.spec, self.led)}.\n{previous}"
            )
            brief_role = "fix" if task.get("step") not in ("implement",) else "implement"
            brief = self.brief(task, brief_role, 100 + attempt, text, extra, False)
            problems = self.run_agent(task, "debug", attempt, brief, agent=DEBUGGER, report=(brief_role, 100 + attempt))
            if self.aborted:  # F170: an abort is the overseer's stop, as in attempt_loop: no gate, not counted
                aborts += 1
                attempt -= 1
                previous = "\n".join(problems)
                if aborts > OV.MAX_ABORTS:
                    return self.stop_task(task, f"aborted by the overseer {aborts} times; waiting for a person")
                continue
            entry = self.led.state().get(tid, {})
            self.led.update(tid, debugger_attempts=entry.get("debugger_attempts", 0) + 1)
            res = self.gate_unless_unchanged(tid)
            outside = [p for p in problems if p.startswith("changed files outside")]
            if res.verdict == "PASS" and not outside:  # F170 (audit item 9): as attempt_loop: a pass is a pass
                if problems:
                    self.led.update(tid, review=problems)
                    self.echo(f"  [{tid}] passed with problems for review: {len(problems)}")
                return True
            if res.verdict == "PASS":
                return self.stop_passed_out_of_scope(task, problems)
            previous = "\n".join(problems + [self.failure_text(res)])
        if self.last_chance_defer(task, previous, attempts, debugger_attempts):
            return True
        self.led.update(
            tid,
            status="STOPPED",
            reason=[f"stopped after {attempts} attempts and " f"{debugger_attempts} debugger attempts"],
            history_add={"t": now(), "status": "STOPPED"},
        )
        return False

    def last_chance_defer(self, task: dict, previous: str, attempts: int, debugger_attempts: int) -> bool:
        """After the debugger: one implement attempt that may defer the step (F46) instead of stopping the run."""
        if not OR.deferrable(task) or not self.policy(task, "implement").get("defer_after_debugger", True):
            return False
        text = (
            f"The implementer failed {attempts} attempts and {DEBUGGER} {debugger_attempts}. This attempt is only for a "
            "deferral: if TTNN has no proper op for this step, defer it (section 'Deferring this step to op-gen'), "
            "with the evidence from those attempts. If it does have one, change nothing: the task then stops for a "
            f"person.\n{previous}"
        )
        self.pause_point(task["id"], "the last-chance deferral attempt")
        problems = self.run_agent(task, "implement", 201, self.brief(task, "implement", 201, text))
        if self.aborted or [p for p in problems if p.startswith("changed files outside")]:
            return False
        deferred, why = self.try_defer(task)
        if why:
            self.led.update(task["id"], reason=[why])
        return deferred

    def overseer_owned(self, tid: str) -> str | None:
        """F169: the overseer's classification of this task's last gate failure, when it is not the model's code
        (framework, waiver, infra): the orchestrator runs only the gate's remaining units and starts no agent."""
        c = OV.classification(self.spec.ledger_dir, tid)
        return c["kind"] if c and c.get("kind") != "model" else None

    def stop_for_overseer(self, tid: str) -> int:
        kind = self.overseer_owned(tid)
        why = (
            f"the gate still fails after the overseer's {kind} fix; no agent attempt (F169): fix it, "
            f"`gate {tid} --unit <failed unit>`, then resume (the remaining units run)"
        )
        self.led.update(tid, status="STOPPED", reason=[why], history_add={"t": now(), "status": "STOPPED"})
        self.echo(f"  [{tid}] STOPPED (overseer-classified {kind}): {why}")
        return STOPPED

    def infra_failure(self, res) -> str | None:
        text = res.log.read_text(errors="replace") if res and res.log and res.log.exists() else ""
        m = INFRA_FAILURES.search(text) or TEST_TIMEOUT.search(text)
        return m.group(0) if m else None

    def stop_for_infra(self, tid: str, sig: str) -> int:
        if TEST_TIMEOUT.search(sig):
            why = (
                f"the test ran past its pytest timeout ({sig!r}) twice (one automatic retry, F170): check the machine's "
                "load (CPU jobs next to the gate) and the log for a slow path, or raise box.test_timeout_s; then "
                "resume (only the stopped unit and the ones after it run)"
            )
        else:
            why = (
                f"the box failed ({sig!r}) and the automatic box check failed too; reset the board, then resume "
                "(only the stopped unit and the ones after it run)"
            )
        self.led.update(tid, status="STOPPED", reason=[why], history_add={"t": now(), "status": "STOPPED"})
        self.echo(f"  [{tid}] STOPPED (infrastructure): {why}")
        return STOPPED

    # ---- perf picks (F57): the agent makes the change; the orchestrator measures it on vs off; the owner decides
    def perf_pick(self, task: dict) -> bool:
        return task.get("step") == "perf" and bool(task.get("role"))

    def ab_key(self, task: dict) -> str:
        """What a report and a decision are about: the pick's brief and switch. An edit of the gate (the owner dropping
        a frozen test) keeps them; a new brief is a new change."""
        import hashlib

        keep = {k: task.get(k) for k in ("brief", "ab", "ab_rungs")}
        return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def ab_code_key(self, task: dict) -> str:
        """The code a side measures: F170 (audit item 14) the same key the gate stamps use (stamps.code_key: every file
        a measurement can read, committed, changed or untracked, without the run's records), so an agent's new,
        untracked module that changed after a measurement is a code change (it was not: the accept could commit code
        that was never measured) and an unrelated commit of the same content is not (F105)."""
        return stamps.code_key(self.spec.repo, self.spec)

    def changed_under(self, task: dict) -> list[str]:
        """Uncommitted files under the task's paths (the change a report measures)."""
        return sorted(p for p in dirty(self.spec.repo) if allowed(p, list(task.get("paths") or [])))

    def ab_record(self, task: dict) -> dict | None:
        rec = self.led.state().get(task["id"], {}).get("ab")
        return rec if isinstance(rec, dict) and rec.get("key") == self.ab_key(task) else None

    def perf_step(self, task: dict) -> int:
        """A perf pick: agent (the change behind its switch, the frozen tests once, no profile), then the A/B report,
        then WAITING FOR A PERSON until ``decide``; accept runs the gate and commits, reject reverts (REJECTED)."""
        tid = task["id"]
        rec = self.ab_record(task)
        if rec and rec.get("decision") == "accept":
            return self.perf_accept(task, rec)
        if rec and rec.get("decision") == "reject":
            return self.perf_reject(task, rec)
        if rec and rec.get("decision") == "rerun":  # measure the same change again (e.g. after a board reset)
            rec = self.ab_report(task, sorted(set(rec.get("changed") or []) | set(self.changed_under(task))))
            self.led.update(tid, ab=rec, status="TODO", reason=[rec["why"]])
            return self._human(task, rec["why"])
        if rec:
            return self._human(task, rec["why"])
        changed = self.perf_agent(task)
        if changed is None:
            return STOPPED
        # F108: the change is everything uncommitted under the task's paths, an earlier attempt's work included
        rec = self.ab_report(task, sorted(set(changed) | set(self.changed_under(task))))
        self.led.update(tid, ab=rec, status="TODO", reason=[rec["why"]])
        return self._human(task, rec["why"])

    def perf_agent(self, task: dict) -> list[str] | None:
        """The perf role until one attempt ends without problems; the files it changed, or None (STOPPED)."""
        tid, pol, previous, changed = task["id"], self.policy(task, "perf"), "", set()
        done, cut = self.attempts_so_far(tid, "perf")  # F107: a resumed pick continues the numbering (F95)
        if cut is not None:
            previous = (
                f"A previous run was interrupted during perf attempt {cut} of this task (a crash, power loss or an "
                "overseer stop). Its changes may be in the tree: check them before redoing anything."
            )
        attempt, used, aborts = done, 0, 0
        while used < pol["attempts"]:
            self.pause_point(tid, f"perf attempt {attempt + 1}")
            attempt += 1
            self.led.update(tid, status="RUNNING", attempt=attempt, role="perf", waiting=None)
            accuracy_guard.clear(self.spec, tid)  # the marker is per attempt
            problems = self.run_agent(task, "perf", attempt, self.brief(task, "perf", attempt, previous))
            changed |= set(self.last_changed)
            previous = "\n".join(problems)
            if self.aborted:  # F170: as attempt_loop: an overseer's abort is not a used attempt
                aborts += 1
                if aborts > OV.MAX_ABORTS:
                    self.stop_task(task, f"aborted by the overseer {aborts} times; waiting for a person")
                    return None
                continue
            used += 1
            if not problems:
                return sorted(changed)
        why = f"perf failed {pol['attempts']} attempts; waiting for a person (logs in {self.run_dir / 'agents'})"
        self.led.update(tid, status="STOPPED", reason=[why], history_add={"t": now(), "status": "STOPPED"})
        self.echo(f"  [{tid}] STOPPED: {why}")
        return None

    def run_cmd(self, cmd: str, env: dict, log: Path) -> int:
        """One device command of an A/B report, run like a gate command (shell, repo root, through the safe runner)."""
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            f.write(f"$ {cmd}\n")
            f.flush()
            return subprocess.run(
                cmd, shell=True, cwd=self.spec.repo, env=env, stdout=f, stderr=subprocess.STDOUT
            ).returncode

    def ab_rungs(self, task: dict) -> list[str]:
        """tasks.yaml ``ab_rungs``, else the rung the gate runs (or ``last``) and the spec's last (full-target) rung."""
        if task.get("ab_rungs"):
            return list(task["ab_rungs"])
        names = [r["name"] for r in self.spec.data.get("ladder") or []]
        m = re.search(r"BRINGUP_RUNG=(\S+)", task["gate"]["cmd"])
        return list(dict.fromkeys(r for r in (m.group(1) if m else "last", names[-1] if names else None) if r in names))

    def ab_report(self, task: dict, changed: list[str]) -> dict:
        """Per config (off = ``ab.env``, the old path; on = ``ab.change``, the change, both set explicitly: the agent may
        leave either default in the tree), each into its own results dir runs/<run>/ab/<task>/<old|new>/: every frozen
        test of the gate (all of them, not stopping at the first failure), the ladder at every A/B rung, and one plain
        profile. A side already measured for the same code and switch (side.json) is reused, not run again. Then
        table.md and the ledger record."""
        tid, ab = task["id"], task.get("ab") or {}
        d = self.run_dir / "ab" / tid
        old, plain_old = model_precision.switch_env(self.spec, ab.get("env") or {})  # F103: setting -> its variable
        on, plain_on = model_precision.switch_env(self.spec, ab.get("change") or {})
        if old and old == on:
            raise ValueError(f"{tid}: ab.env and ab.change are the same configuration")
        rec = {
            "key": self.ab_key(task),
            "at": now(),
            "changed": changed,
            "env": old,
            "on": on,
            "table": str(d / "table.md"),
        }
        rec.update(rungs=self.ab_rungs(task), tests={}, ladder={}, profile={}, quick={}, note="")
        owned = [p for p in changed if allowed(p, list(task.get("paths") or []))]
        base = self.ab_base(old, on, tid)
        sides = ([("old", old)] if old else []) + [("new", on)]
        if not owned:
            rec["note"] = "the agent changed nothing under the task's paths (its summary and BREADCRUMBS say why)"
            sides = []
        elif not old:
            rec["note"] = "no `ab` switch in tasks.yaml: the change is measured on only"
        elif not on:
            rec[
                "note"
            ] = "no `ab.change` in tasks.yaml: 'on' is the tree's default, which the agent may have left on the old path"
        missing = ab_switch_missing(task, self.spec.repo, owned, set(plain_old) | set(plain_on)) if sides else []
        if missing:
            rec[
                "note"
            ] = f"INVALID: switch {missing} (tasks.yaml ab) appears in none of the task's code, so it would change nothing"
            sides = []
        code = self.ab_code_key(task)
        kinds = {n: k for n, k, _, _ in self.ab_items(task)}
        try:
            for side, env in sides:  # F137: every run at the shipped settings, the switch as each side sets it
                for n, r in self.ab_measure(task, side, env, base, code).items():
                    if kinds[n] == "quick":
                        rec["quick"][side] = r
                    elif kinds[n] == "profile":
                        rec["profile"][side] = r
                    else:
                        rec["tests" if kinds[n] == "test" else "ladder"].setdefault(n, {})[side] = r
        except InfraStop as e:
            rec["note"] = f"stopped early: {e.sig} (reset the board; `decide --task {tid} --rerun` measures again)"
        same = [
            n
            for n, r in list(rec["tests"].items()) + list(rec["ladder"].items())
            if "old" in r
            and "new" in r
            and r["old"].get("checks", r["old"]) == r["new"].get("checks", r["new"])
            and self.ab_metrics(tid, d, "old", n) == self.ab_metrics(tid, d, "new", n)
        ]
        measured = any(self.ab_metrics(tid, d, "old", n) for n in same)
        if measured and len(same) == len(rec["tests"]) + len(rec["ladder"]):
            rec["note"] = (
                "INVALID: off and on gave bit-identical metrics on every run, so the switch changed nothing (check"
                " ab.env / ab.change in tasks.yaml against the code's switch)"
            )
        rec["gate_on"] = self.gate_on(task, rec, d)
        (d / "table.md").parent.mkdir(parents=True, exist_ok=True)
        (d / "table.md").write_text(self.ab_table(task, rec))
        verdict = "PASS" if rec["gate_on"]["ok"] else "FAIL"
        rec["why"] = (
            f"apply {tid} or not? A/B table at {rec['table']}; the gate with the change on would {verdict}"
            + (f" ({rec['note']})" if rec["note"] else "")
            + f"; then: python -m models.demos.common.bringup.orchestrator decide --task {tid} --accept|--reject"
            + f" --spec {self.spec.path}, and resume"
        )
        self.echo(f"  [{tid}] A/B report: {rec['table']}")
        return rec

    def ab_base(self, old: dict, on: dict, tid: str) -> dict:
        """The A/B runs' environment without either side's switch (each run adds its side's)."""
        base = gate_env(self.spec, self.led, tid)
        base[accuracy_guard.AB_ENV] = "1"  # the profile guard lets the orchestrator's own profile through
        for k in list(old) + list(on) + ["BRINGUP_PROFILE_OPS"]:
            base.pop(k, None)
        return base

    def ab_quick(self, task: dict) -> bool:
        """F122: the pick's A/B mode: tasks.yaml ab_mode, else the spec's perf.ab_mode (defaults.yaml: quick)."""
        return (task.get("ab_mode") or self.spec.get("perf.ab_mode") or "full") == "quick"

    def ab_items(self, task: dict) -> list[tuple[str, str, str, dict]]:
        """The A/B report's runs as (name, kind, command, extra env): each frozen test command of the gate (as the gate
        runs it, gate_command), the ladder at every A/B rung, and the plain profile (with F114's clean wall run)."""
        if self.ab_quick(task):  # F122: one process a side
            quick = f"BRINGUP_QUICK=1 BRINGUP_PROFILE_WALL=1 scripts/run_safe_pytest.sh --no-precompile --run-all {PROFILE_TEST}"
            return [("quick", "quick", quick, {})]
        segs = [x.strip() for x in gate_command(task).split("&&") if x.strip()]
        tests = [x for x in segs if "test_profile.py" not in x and "test_ladder.py" not in x]
        items = [
            (Path(next((w for w in c.split() if w.endswith(".py")), f"test{i}")).stem, "test", c, {})
            for i, c in enumerate(tests)
        ]
        ladder = f"scripts/run_safe_pytest.sh --no-precompile --run-all {LADDER_TEST}"
        items += [(r, "ladder", ladder, {"BRINGUP_RUNG": r}) for r in self.ab_rungs(task)]
        prof = " && ".join(re.sub(r"\bBRINGUP_PROFILE_OPS=\S+\s*", "", x) for x in segs if "test_profile.py" in x)
        return items + ([("profile", "profile", prof, {})] if prof else [])

    def ab_measure(self, task: dict, side: str, env: dict, base: dict, code: str, only=None) -> dict:
        """F116: one side's runs (all, or the names / kinds in ``only``). Each run is kept in side.json with its own
        stamp (code key, switch, command); a run whose stamp matches is reused, not run again: the agent's `measure`
        runs of the 'on' side are the report's. {name: result}; raises InfraStop on a board problem."""
        tid = task["id"]
        d = self.run_dir / "ab" / tid / side
        f = d / "side.json"
        kept = json.loads(f.read_text()) if f.exists() else {}
        items = kept.get("items", {}) if kept.get("version") == 2 else {}
        out = {}
        for name, kind, cmd, extra in self.ab_items(task):
            if only is not None and name not in only and kind not in only:
                continue
            stamp = {"code": code, "env": env, "cmd": cmd, "extra": extra}
            label = f"  [{tid}] A/B {'on' if side == 'new' else 'off'}: {name}"
            if (items.get(name) or {}).get("stamp") == stamp and ab_finished(items[name]["result"]):
                out[name] = items[name]["result"]
                self.echo(f"{label} reused (same code and switch)")
                continue
            res = d / name
            M.reset(tid, res)
            self.echo(label)
            rc = self.run_cmd(cmd, dict(base, **{M.RESULTS_ENV: str(res)}, **env, **extra), res / f"{name}.log")
            text = (res / f"{name}.log").read_text(errors="replace")
            crash = PROCESS_CRASH.search(text) if rc != 0 else None
            if (
                crash
            ):  # F207 (DeepSeek-V4.1 P.2, 2026-10-28: a bus error while loading cached weights lost the off side)
                self.echo(f"{label}: the process died ({crash.group(0)}); running it once more (F207)")
                (res / f"{name}.log").rename(res / f"{name}.crashed.log")
                M.reset(tid, res)
                rc = self.run_cmd(cmd, dict(base, **{M.RESULTS_ENV: str(res)}, **env, **extra), res / f"{name}.log")
                text = (res / f"{name}.log").read_text(errors="replace")
            sig = INFRA_FAILURES.search(text) or TEST_TIMEOUT.search(text)
            if sig or rc == SAFE_HANG_RC:
                raise InfraStop(tid, sig.group(0) if sig else f"{name} hung (rc {rc})")
            got = {k: v["value"] for k, v in M.load(tid, res).items()}
            if kind == "test":
                result = {"rc": rc, "checks": [m.strip() for m in FAILED_CHECK.findall(text)][:12]}
            elif kind == "ladder":
                result = dict(self.ab_summary(got), rc=rc, limits=ab_limits(got))
            elif kind == "quick":
                lay = {k: v for k, v in got.items() if k.startswith("pcc_layer_L")}
                worst = min(lay, key=lay.get) if lay else None
                result = {
                    "min layer PCC": lay.get(worst),
                    "min layer": worst and worst[len("pcc_layer_") :],
                    "chunk_wall_ms": got.get("chunk_wall_ms"),
                    "chunk_wall_ms_spread": got.get("chunk_wall_ms_spread"),
                    "chunk_wall_gap_ms": got.get("chunk_wall_gap_ms"),
                    "pcc_chunk_out": got.get("pcc_chunk_out"),
                    "host_transfers_per_layer": got.get("host_transfers_per_layer"),
                    "rc": rc,
                    "limits": ab_limits(got),
                }
            else:
                keys = ["device_ms_total", "chunk_wall_ms", "profiled_programs", "pcc_chunk_out"] + [
                    k for k in task["gate"].get("metrics") or {} if k.startswith("device_ms_") and "*" not in k
                ]
                result = dict({k: got.get(k) for k in dict.fromkeys(keys)}, rc=rc)
            out[name] = result
            items[name] = {"kind": kind, "stamp": stamp, "result": result, "t": now()}
            d.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({"version": 2, "items": items}, indent=1))
        return out

    def ab_metrics(self, tid: str, d: Path, side: str, name: str) -> dict:
        return {k: v["value"] for k, v in M.load(tid, d / side / name).items()}

    def gate_on(self, task: dict, rec: dict, d: Path) -> dict:
        """Would the gate pass with the change on: every command exited 0 and every threshold holds on its metrics."""
        from models.demos.common.bringup.core.gate import check_metrics

        if rec.get("quick"):  # F122: the thresholds the quick run measured; the full gate runs before the final task
            from models.demos.common.bringup.core.gate import measured_only

            r = rec["quick"].get("new")
            got = M.load(task["id"], d / "new" / "quick")
            ok, lines = check_metrics(measured_only(task["gate"].get("metrics") or {}, got), got)
            failing = ["quick: not run" if r is None else f"quick: exit code {r['rc']}"] if not r or r["rc"] else []
            failing += [x.strip() for x in lines if "FAIL" in x or "MISSING" in x]
            return {"ok": ok and not failing, "failing": failing, "partial": True}
        m = re.search(r"BRINGUP_RUNG=(\S+)", task["gate"]["cmd"])
        runs = {n: r.get("new") for n, r in rec["tests"].items()}  # what the gate itself runs, with the change on
        runs.update({m.group(1): rec["ladder"].get(m.group(1), {}).get("new")} if m and rec["ladder"] else {})
        runs.update({"profile": rec["profile"].get("new")} if rec["profile"] else {})
        got = {}
        for n in runs:
            got.update(M.load(task["id"], d / "new" / n))
        ok, lines = check_metrics(task["gate"].get("metrics") or {}, got)
        failing = [
            f"{n}: not run" if r is None else f"{n}: exit code {r['rc']}" for n, r in runs.items() if not r or r["rc"]
        ]
        failing += [x.strip() for x in lines if "FAIL" in x or "MISSING" in x]
        return {"ok": ok and bool(runs) and not failing, "failing": failing}

    @staticmethod
    def ab_summary(got: dict) -> dict:
        out = {}
        for label, key in AB_ROWS:
            if key.endswith("*"):
                vals = {k: v for k, v in got.items() if k.startswith(key[:-1]) and v is not None}
                if vals:
                    k = min(vals, key=vals.get)
                    out[label], out["min layer"] = vals[k], k[len(key) - 1 :]
            elif got.get(key) is not None:
                out[label] = got[key]
        return out

    def ab_table(self, task: dict, rec: dict) -> str:
        fmt = lambda v: "-" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
        sides = [("new", "on"), ("old", "off")]
        old = " ".join(f"{k}={v}" for k, v in rec["env"].items()) or "(no ab switch)"
        out = [
            f"# {task['id']}: {task['title']} -- apply or not?",
            "",
            f"on = the tree as the agent left it; off = `{old}` (the old path). {rec['at']}.",
        ]
        if rec["note"]:
            out += ["", f"Note: {rec['note']}"]
        if rec.get("quick"):  # F122
            out += ["", "## Quick A/B: the last chunk after the golden prefix, one process a side", ""]
            out += ["| | on | off |", "|---|---|---|"]
            for k in (
                "min layer PCC",
                "min layer",
                "pcc_chunk_out",
                "chunk_wall_ms",
                "chunk_wall_ms_spread",
                "chunk_wall_gap_ms",
                "host_transfers_per_layer",
                "rc",
            ):
                vals = [rec["quick"].get(s, {}).get(k) for s, _ in sides]
                out.append(f"| {'exit code' if k == 'rc' else k} | " + " | ".join(fmt(v) for v in vals) + " |")
            out += limits_table([(lbl, rec["quick"].get(s) or {}) for s, lbl in sides])
            out += [
                "",
                "Frozen tests, every ladder rung and the per-section profile run once, before the final task "
                "(the pick's full gate, merged with the other quick picks').",
            ]
        if rec["tests"]:
            out += ["", "## Frozen accuracy tests of the gate (component metric vs its limit)", ""]
            out += ["| test | on | off |", "|---|---|---|"]
        for name, r in rec["tests"].items():
            cell = lambda s: "-" if s not in r else ("pass" if r[s]["rc"] == 0 else f"FAIL (rc {r[s]['rc']})")  # noqa
            out.append(f"| {name} | {cell('new')} | {cell('old')} |")
        for name, r in rec["tests"].items():
            for side, label in sides:
                out += [f"- {name} {label}: {c}" for c in (r.get(side) or {}).get("checks", [])]
        rungs = rec["rungs"] if rec["ladder"] else []
        out += ["", "## End to end (ladder)", ""] if rungs else []
        out += ["| | " + " | ".join(f"{r} {lbl}" for r in rungs for _, lbl in sides) + " |"] if rungs else []
        out += ["|---|" + "---|" * (2 * len(rungs))] if rungs else []
        for label, _ in AB_ROWS + (("exit code", "rc"),) if rungs else ():
            cells = []
            for r in rungs:
                for side, _ in sides:
                    row = rec["ladder"].get(r, {}).get(side)
                    v = None if row is None else row.get("rc" if label == "exit code" else label)
                    ok = label == "min layer PCC" and v is not None
                    cells.append(f"{v:.4f} ({row.get('min layer')})" if ok else fmt(v))
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out += limits_table([(f"{r} {lbl}", rec["ladder"].get(r, {}).get(s) or {}) for r in rungs for s, lbl in sides])
        if rec["profile"]:
            gate = task["gate"].get("metrics") or {}
            out += ["", "## Device time, one plain profile per config (ms per chunk)", ""]
            out += ["| | on | off |", "|---|---|---|"]
            keys = [k for k in rec["profile"].get("new", rec["profile"].get("old", {})) if k != "rc"]
            for k in keys + ["rc"]:
                lbl = {"device_ms_total": "device_ms_total (e2e chunk)", "rc": "exit code"}.get(k, k)
                lbl += f" (gate {gate[k]})" if k in gate else ""
                vals = [rec["profile"].get(s, {}).get(k) for s, _ in sides]
                ms = k.startswith("device_ms_")
                out.append(
                    f"| {lbl} | " + " | ".join(f"{v:.1f}" if ms and v is not None else fmt(v) for v in vals) + " |"
                )
        g = rec["gate_on"]
        part = " (the thresholds the quick run measured)" if g.get("partial") else ""
        out += ["", f"## Gate with the change on{part}: {'PASS' if g['ok'] else 'FAIL'}", ""]
        out += [f"- {x}" for x in g["failing"]]
        out += ["", f"Logs: {self.run_dir / 'ab' / task['id']}", ""]
        return "\n".join(out)

    def perf_accept(self, task: dict, rec: dict) -> int:
        """The owner applies the change: the gate (as tasks.yaml now has it) runs and commits the change on a PASS.
        F104: a change behind a model setting (ab.change) ships: its default becomes the accepted value first, so the
        gate, the commit and every later task run it (an opt-in switch left at the old default measured the old path).
        """
        tid = task["id"]
        reuse = self.ab_on_reuse(task, rec)  # F105: before set_defaults (it changes the tree the key covers)
        shipped = model_precision.set_defaults(self.spec, (task.get("ab") or {}).get("change") or {})
        if shipped:
            self.echo(f"  [{tid}] accepted: default {'; '.join(shipped)} (tt/settings.py)")
        if reuse is not None:
            self.echo(f"  [{tid}] accept gate: the A/B 'on' side's measurement, not run again ({reuse['source']})")
        res = self.gate(tid, reuse=reuse)
        if res.verdict == "PASS":
            if reuse is not None and reuse.get("partial"):
                self.led.update(tid, full_pending=True)  # F122: its full gate runs before the final task
            return self._after(task)
        why = (
            f"accepted, but the gate fails with the change on:\n{res.summary()}\nThe change misses a recorded limit: "
            f"reject it (decide --task {tid} --reject), or fix it and measure again; limits are never edited"
        )
        return self._human(task, why)

    def ab_on_reuse(self, task: dict, rec: dict) -> dict | None:
        """F105: the accept gate's metrics from the A/B report's 'on' side when that side measured exactly the gate's
        commands on the tree as it is now (same code key), every command exited 0, and the gate's ladder rung was one
        of its rungs; else None (the gate runs). The A/B ran the change explicitly on, which F104's new default makes
        the tree's behaviour."""
        tid = task["id"]
        side = self.run_dir / "ab" / tid / "new"
        stamp_file = side / "side.json"
        if not stamp_file.exists() or not rec.get("on"):
            return None
        if self.ab_quick(task):  # F122: accept on the quick run; the full gate is pending (run before the final task)
            it = (json.loads(stamp_file.read_text()).get("items") or {}).get("quick") or {}
            if it.get("stamp", {}).get("code") != self.ab_code_key(task) or (it.get("result") or {}).get("rc") != 0:
                return None
            got = {k: v["value"] for k, v in M.load(tid, side / "quick").items()}
            return {"metrics": got, "source": f"quick A/B {side / 'quick'}; full gate pending", "partial": True}
        kept = json.loads(stamp_file.read_text())
        code = self.ab_code_key(task)
        if kept.get("version") == 2:  # F116: per-run stamps; every run of the report's 'on' side, on this code
            items = kept.get("items", {})
            want = [n for n, _, _, _ in self.ab_items(task)]
            if any((items.get(n) or {}).get("stamp", {}).get("code") != code for n in want):
                return None
            kept = {
                k: {n: items[n]["result"] for n in want if items[n]["kind"] == kind}
                for k, kind in (("tests", "test"), ("ladder", "ladder"))
            }
            kept["profile"] = next((items[n]["result"] for n in want if items[n]["kind"] == "profile"), None)
        elif (kept.get("stamp") or {}).get("code") != code:
            return None
        m = re.search(r"BRINGUP_RUNG=(\S+)", task["gate"]["cmd"])
        rung = m.group(1) if m else None
        rcs = [r.get("rc") for r in kept.get("tests", {}).values()] + [
            r.get("rc") for r in kept.get("ladder", {}).values()
        ]
        if (kept.get("profile") or {}).get("rc") is not None:
            rcs.append(kept["profile"]["rc"])
        if any(rc != 0 for rc in rcs) or (rung and rung not in kept.get("ladder", {})):
            return None
        metrics = {}
        names = [
            d.name
            for d in side.iterdir()
            if d.is_dir() and (d.name in kept["tests"] or d.name in kept["ladder"] or d.name == "profile")
        ]
        ladder = set(kept.get("ladder", {}))
        for name in sorted(names, key=lambda n: (n in ladder, n == "profile")):  # tests, then the gate's rung, profile
            if name in ladder and name != rung:
                continue
            metrics.update({k: v["value"] for k, v in M.load(tid, side / name).items()})
        return {"metrics": metrics, "source": str(side)}

    def perf_reject(self, task: dict, rec: dict) -> int:
        """The owner declines the change: the agent's files under the task's paths go back to HEAD, the task is
        REJECTED (a done status: its dependents run), and the decision is committed."""
        tid, repo = task["id"], self.spec.repo
        owned = [p for p in rec.get("changed") or [] if allowed(p, list(task.get("paths") or []))]
        tracked = [p for p in owned if subprocess.run(["git", "cat-file", "-e", f"HEAD:{p}"], cwd=repo).returncode == 0]
        if tracked:
            subprocess.run(["git", "checkout", "HEAD", "--", *tracked], cwd=repo, check=True)
        for p in set(owned) - set(tracked):
            (repo / p).unlink(missing_ok=True)
        why = f"rejected by the owner on the A/B report {rec['table']}; reverted {len(owned)} files (old path)"
        self.led.update(
            tid, status="REJECTED", reason=[why], waiting=None, history_add={"t": now(), "status": "REJECTED"}
        )
        accuracy_guard.clear(self.spec, tid)
        with self.led.locked():
            sha = git_commit(
                self.spec,
                [rel(self.spec, self.led.state_path)],
                f"[{self.spec.tag}][{tid}] {task['title']} (rejected by the owner)",
                why,
            )
        self.echo(f"  [{tid}] REJECTED: {why}" + (f" -> {sha}" if sha else ""))
        return DONE

    def step(self, tid: str) -> int:
        task = self.led.task(tid)
        role = task.get("role") or ROLE_OF_STEP.get(task.get("step"))
        self.echo(f"[{tid}] {task['title']} (step {task.get('step')}, role {role or 'script'})")
        if self.led.state().get(tid, {}).get("waiting"):
            self.led.update(tid, waiting=None)  # a person answered (approval, reset, resume): the task runs again
        if self.led.status(tid) == "COMMIT_PENDING":  # F149: its gate passed; only the commit is retried
            res = commit_only(self.spec, self.led, tid)
            if res.verdict != "PASS":
                raise CommitPending(tid, res.summary())
            self.echo(f"  [{tid}] committed without a re-run -> {res.commit}")
            return self._human(task, self.needs_human(task)) if self.needs_human(task) else self._after(task)
        if task.get("tests") and not task.get("frozen"):
            reason = self.freeze_tests(task)
            if reason:
                self.led.update(tid, status="STOPPED", reason=[reason], history_add={"t": now(), "status": "STOPPED"})
                return STOPPED
            task = self.led.task(tid)
        if task.get("step") == "perf" or PROFILE_TEST in gate_command(task):  # F194: a profile needs every chip
            why = iommu_problem()
            if why:
                self.stop_task(task, f"not started: {why}")
                return STOPPED
        why = engine_missing(self.spec, gate_command(task, self.led.tasks()))
        if why:  # F203: no gate and no agent for a runner test without its engine; independent work goes on (F109)
            return self._human(task, why)
        if self.perf_pick(task):  # F57: no gate before or after the agent; the owner decides on the A/B report
            return self.perf_step(task)
        if task.get("step") == "perf" and role is None:  # F122: the final task: the quick picks' full gates first
            stop = self.pending_full_gates()
            if stop is not None:
                return stop
        plan_written = task.get("approval") == "plan" and (self.spec.bringup_dir / "plan.yaml").exists()
        if role is None or plan_written:
            # A scripted step, or a plan written by an earlier run: gate it; an agent only sees it after a failure.
            if plan_written and self.needs_human(task):
                return self._human(task, self.needs_human(task))
            res = self.gate(tid)
            if res.verdict != "PASS" and self.overseer_owned(tid):
                return self.stop_for_overseer(tid)
            if res.verdict != "PASS" and not self.attempt_loop(task, role or "fix", self.failure_text(res)):
                return STOPPED
        else:
            # The work may already exist (e.g. R.3 once R.2's reference is written, a swap test once its components
            # pass): try the gate before starting an agent.
            existing = any((self.spec.repo / p).exists() for p in task.get("paths") or [])
            # F87: a task resumed after a stop (a fix job included) gates first: its work may be done meanwhile
            hist = self.led.state().get(tid, {}).get("history") or []
            resumed = bool(hist) and hist[-1].get("status") == "RESUMED"
            # F193: a task without a precheck (a fix job: its gate passes before the change it asks for) gates first
            # only when resumed after a STOP; one whose agent was cut off (RUNNING) starts its agent again: a pass of
            # tests that passed before says nothing (DeepSeek-V4.1 FIX.3 passed with no agent after a kill + resume)
            resumed = resumed and (task.get("precheck", True) or hist[-1].get("from") == "STOPPED")
            owned = self.overseer_owned(tid)  # F169: a failure the overseer fixed: the delta runs, never an agent
            pre = self.gate(tid) if owned or (existing and (task.get("precheck", True) or resumed)) else None
            if owned and pre.verdict != "PASS":
                return self.stop_for_overseer(tid)
            if (pre is None or pre.verdict != "PASS") and not self.attempt_loop(
                task, role, self.failure_text(pre) if pre else ""
            ):
                return STOPPED
        if self.needs_human(task):
            return self._human(task, self.needs_human(task))
        return self._after(task) if satisfied(self.led.status(tid)) else STOPPED

    def ran_final(self) -> bool:
        """F188: the final measurement (a perf task without a role) has passed, so it will not run the quick-accepted
        picks' full gates any more: the end of the run does."""
        tasks = self.led.tasks()
        return any(
            t.get("step") == "perf" and not t.get("role") and self.led.status(i) == "PASS" for i, t in tasks.items()
        )

    def pending_full_gates(self) -> int | None:
        """F122: every quick-accepted pick's full gate, once, on the tree as it is now; picks with the same gate
        command share one run. None when all pass (cleared), else the stop."""
        pending = [tid for tid, e in self.led.state().items() if isinstance(e, dict) and e.get("full_pending")]
        order = list(self.led.tasks())  # F175: the newest pick first: its own gate units already passed on this code
        pending.sort(key=lambda t: -order.index(t) if t in order else 0)  # (its agent's gate --check), so the shared
        done = {}  # gate runs only the delta; the older picks with the same command take its result
        for tid in pending:
            cmd = gate_command(self.led.task(tid))
            if cmd in done:
                src, metrics = done[cmd]
                res = self.gate(tid, reuse={"metrics": metrics, "source": f"the same gate as {src}, run just now"})
            else:
                self.echo(f"  [{tid}] full gate (quick-accepted pick)")
                res = self.gate(tid)
                done[cmd] = (tid, {k: v["value"] for k, v in M.load(tid, self.led.results_dir).items()})
            if res.verdict != "PASS":
                return self._human(
                    self.led.task(tid),
                    f"{tid} was accepted on its quick A/B; its full gate fails now:\n{res.summary()}",
                )
            self.led.update(tid, full_pending=False)
        return None

    def _after(self, task: dict) -> int:
        picked = any(  # a pick is an agent's perf task after X.2 (the final X.3 is scripted)
            t.get("step") == "perf" and t.get("role") and "X.2" in (t.get("deps") or [])
            for t in self.led.tasks().values()
        )
        if (task["id"] == "X.2" and not picked) or task.get("stop_after"):
            return self._human(
                task,
                f"pick opportunities in {rel(self.spec, self.spec.ledger_dir)}/opportunities.md; "
                "each pick becomes a perf task in tasks.yaml",
            )
        return DONE

    def _human(self, task: dict, why: str) -> int:
        self.echo(f"[{task['id']}] WAITING FOR A PERSON: {why}")
        self.led.update(task["id"], waiting=why)
        return HUMAN

    def run(self, until: str | None = None, only: str | None = None) -> int:
        """until (F152): run that task and what it needs, then stop; only: just that task."""
        while True:
            if self.place_fixes():  # F77 / F143: queued fixes are placed first, so `--only FIX.<n>` can name one
                continue
            if until and until not in self.led.tasks():
                self.echo(f"no task {until} (--until)")
                return STOPPED
            if until and satisfied(self.led.status(until)):
                self.echo(f"reached {until}; stopping (--until)")
                return DONE
            scope = set(self.led.upstream(until)) if until else set(self.led.tasks())  # deps may change (F126)
            state = self.led.state()
            runnable = [t for t in self.led.runnable() if state.get(t, {}).get("status") != "STOPPED" and t in scope]
            if only:
                runnable = [t for t in runnable if t == only]
            parked = self.parked_waiting(runnable, state)
            if parked and len(parked) < len(runnable):  # F109: a person's decision does not hold up independent work
                runnable = [t for t in runnable if t not in parked]
            if not runnable:
                stopped = [t for t in self.led.tasks() if t in scope and state.get(t, {}).get("status") == "STOPPED"]
                if stopped:
                    self.echo(f"stopped tasks: {stopped} (fix, then `resume`)")
                    return STOPPED
                if not until and not only and self.ran_final():  # F188: a pick accepted after the final task passed
                    stop = self.pending_full_gates()
                    if stop is not None:
                        return stop
                self.echo("nothing left to run" + (f" before {until}" if until else ""))
                deferred = self.led.deferred()
                if deferred:
                    ops = [f"{t} ({(state.get(t, {}).get('deferred') or {}).get('op')})" for t in deferred]
                    self.echo(
                        f"complete with {len(deferred)} deferred: {', '.join(ops)}; these steps run on the CPU bridge "
                        f"until op-gen delivers their ops (op requests in {rel(self.spec, OR.root(self.spec))})"
                    )
                return DONE
            if self.pause_file.exists():
                self.echo(f"paused before {runnable[0]} ({self.pause_file}); `resume` continues")
                return PAUSED
            stopped_fix = [
                t
                for t, x in self.led.tasks().items()
                if x.get("step") == OV.FIX_STEP and state.get(t, {}).get("status") == "STOPPED" and t in scope
            ]
            if stopped_fix:
                self.echo(f"stopped fix tasks: {stopped_fix}; the run waits for them (fix, then `resume`)")
                return STOPPED
            tid = runnable[0]
            try:
                rc = self.step(tid)
            except InfraStop as e:
                rc = self.stop_for_infra(e.tid, e.sig)
            except FrameworkStop as e:
                rc = self.stop_for_framework(e.tid, e.where)
            except CommitPending as e:
                self.echo(f"  [{e.tid}] COMMIT PENDING (the run stops; nothing re-runs):\n{e.text}")
                rc = STOPPED
            if rc == HUMAN and not only:
                state = self.led.state()
                rest = [t for t in self.led.runnable() if t != tid and state.get(t, {}).get("status") != "STOPPED"]
                rest = [t for t in rest if t in scope and t not in self.parked_waiting(rest, state)]
                if rest and tid in self.parked_waiting([tid], state):
                    self.echo(f"  [{tid}] waits for a person; the run goes on with {rest[0]} (F109)")
                    continue
            if rc != DONE:
                return rc
            if only:
                return DONE

    def parked_waiting(self, tids: list[str], state: dict) -> set[str]:
        """F109: tasks waiting for a person that the run may pass: not done themselves (so their dependents stay
        blocked; a done task that stops for a decision, e.g. the opportunity list's picks, still stops the run) and
        nothing of theirs uncommitted in the tree (another task's commit would take it along)."""
        out = set()
        for t in tids:
            if not state.get(t, {}).get("waiting") or satisfied(state.get(t, {}).get("status")):
                continue
            if str(state[t]["waiting"]).startswith(ENGINE_WAIT) and not engine_missing(
                self.spec, gate_command(self.led.task(t), self.led.tasks())
            ):
                continue  # F203: its build has appeared; it runs at its turn
            dirty_here = set(self.changed_under(self.led.task(t)))
            rec = state.get(t, {}).get("ab")
            if (
                isinstance(rec, dict) and rec.get("changed") is not None
            ):  # its own change, not a neighbour's (paths overlap)
                dirty_here &= set(rec["changed"])
            if not dirty_here:
                out.add(t)
        return out


def _live_orchestrator(pid_file: Path) -> int | None:
    """F147: the pid in the run's pid file when that process is alive and is not this one."""
    try:
        pid = int(pid_file.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    if pid == os.getpid():
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def _first_open(orch: Orchestrator) -> str | None:
    """The first unpassed task a fix may go before. F213: never a perf pick whose A/B is measured (waiting for the
    decision or decided): the tree holds its uncommitted change, and a fix placed first would work on it and sweep it
    into its own commit (DeepSeek-V4.1 FIX.9 before the accepted P.6). Such a pick keeps its place."""
    state = orch.led.state()
    for t in orch.led.topo_order():
        e = state.get(t, {})
        if satisfied(e.get("status")):
            continue
        if orch.led.task(t).get("step") == "perf" and (e.get("waiting") or e.get("ab")):
            continue
        return t
    return None


def _place_fixes(orch: Orchestrator) -> bool:
    """F77: every queued overseer fix becomes a ledger task placed just before the first unpassed task."""
    jobs = OV.take_fixes(orch.spec.ledger_dir)
    if not jobs:
        return False
    spec = yaml.safe_load(orch.led.tasks_path.read_text())
    tasks = {t["id"]: t for t in spec["tasks"]}
    for job in jobs:
        first = _first_open(orch)
        ids = [t["id"] for t in spec["tasks"]]
        at = ids.index(first) if first in ids else len(ids)
        fix = OV.fix_task(job, tasks)
        # F126: a gate task not passed yet runs after the fix (the fix takes its deps), never before it
        open_gates = [g for g in job["gate"] if not satisfied(orch.led.status(g))]
        if open_gates:
            done = [g for g in job["gate"] if g not in open_gates]
            fix["deps"] = list(dict.fromkeys([d for g in open_gates for d in tasks[g].get("deps") or []] + done))
            for g in open_gates:
                tasks[g]["deps"] = list(tasks[g].get("deps") or []) + [job["id"]]
            at = min([at] + [ids.index(g) for g in open_gates])
        spec["tasks"].insert(at, fix)
        tasks[job["id"]] = spec["tasks"][at]
        # F206: the task it is placed before depends on it, so it is in --until's scope and runs first (with every
        # gate task passed nothing else depended on it: DeepSeek-V4.1 FIX.6 was skipped and X.1 ran first)
        if first in tasks and first != job["id"]:
            tasks[first]["deps"] = list(dict.fromkeys(list(tasks[first].get("deps") or []) + [job["id"]]))
        orch.echo(
            f"[{job['id']}] placed by the overseer before {first}: {job['title']} (gate: {', '.join(job['gate'])})"
        )
    orch.led.write_tasks(spec)
    return True


def measure(orch: Orchestrator, tid: str, parts: list[str] | None) -> int:
    """F116: a perf agent's checks with the change on, run as the A/B report's 'on' side (its commands, the switch of
    tasks.yaml ab.change, its results dirs): each run is stamped with the code key, and the report reuses it while the
    code under the task's paths is unchanged. 0 when every run exited 0."""
    task = orch.led.task(tid)
    ab = task.get("ab") or {}
    old, _ = model_precision.switch_env(orch.spec, ab.get("env") or {})
    on, _ = model_precision.switch_env(orch.spec, ab.get("change") or {})
    try:
        got = orch.ab_measure(task, "new", on, orch.ab_base(old, on, tid), orch.ab_code_key(task), only=parts)
    except InfraStop as e:
        print(f"measure {tid}: stopped: {e.sig}")
        return 2
    for n, r in got.items():
        print(f"{n}: rc {r.get('rc')} " + json.dumps({k: v for k, v in r.items() if k != "rc"})[:400])
    if not got:
        print(f"measure {tid}: no run matches {parts}; runs: {[n for n, _, _, _ in orch.ab_items(task)]}")
    return 0 if got and all(r.get("rc") == 0 for r in got.values()) else 1


def decide(orch: Orchestrator, tid: str, decision: str, note: str = "") -> int:
    """The owner's answer to a perf pick's A/B report; ``resume`` then applies it (perf_step)."""
    import getpass

    task = orch.led.task(tid)
    rec = orch.ab_record(task)
    if not rec and decision == "rerun" and task.get("step") == "perf":
        # e.g. a report cut short: measure the change the tree holds, without another agent attempt
        rec = {"key": orch.ab_key(task), "changed": orch.changed_under(task), "why": "", "at": now()}
    if not rec:
        print(f"{tid}: no A/B report for the current brief and switch (tasks.yaml); nothing to decide")
        return 1
    orch.led.update(tid, ab=dict(rec, decision=decision, decided=now(), by=getpass.getuser(), note=note))
    then = {"accept": "the gate runs and commits the change", "reject": "the change is reverted, the task REJECTED"}
    print(f"{tid}: {decision} recorded; `resume`: {then.get(decision, 'the A/B report is measured again')}")
    return 0


SYSFS_TT = "/sys/class/tenstorrent"  # F194 (a selftest points it at a fake tree)
IOMMU_SERVICES = ("/etc/systemd/system/tt-iommu-domains.service", "/lib/systemd/system/tt-iommu-domains.service")


def iommu_mismatch(root: str | None = None) -> dict[int, tuple[str, str]]:
    """F194: {card index: (PCI address, IOMMU domain type)} for the cards whose domain is translated (DMA / DMA-FQ)
    differently from the majority of the box. Metal keeps ONE IOMMU flag for the whole cluster (that of the first MMIO
    chip), so such a card takes the wrong pinned-memory path: on the 8x4 Galaxy two cards re-enumerated after a board
    reset came back `identity` among `DMA-FQ` ones and lost the real-time profiler (DeepSeek-V4.1 FIX.3, DS41-120).
    Empty when the box is uniform or has no IOMMU groups."""
    import glob

    kinds = {}
    for cls in glob.glob(os.path.join(root or SYSFS_TT, "tenstorrent!*")):
        try:
            dev = os.path.realpath(os.path.join(cls, "device"))
            kinds[int(cls.rsplit("!", 1)[1])] = (
                os.path.basename(dev),
                Path(dev, "iommu_group", "type").read_text().strip(),
            )
        except (OSError, ValueError):
            continue
    translated = {i for i, (_, k) in kinds.items() if k.startswith("DMA")}
    odd = translated if len(translated) * 2 < len(kinds) else set(kinds) - translated
    return {i: kinds[i] for i in sorted(odd)}


def iommu_problem(root: str | None = None) -> str | None:
    """F194: one line for a mixed-IOMMU box (the cause and the root-only fix), or None."""
    bad = iommu_mismatch(root)
    if not bad:
        return None
    cards = ", ".join(f"/dev/tenstorrent/{i} ({bdf}: {kind})" for i, (bdf, kind) in bad.items())
    svc = next((s for s in IOMMU_SERVICES if os.path.exists(s)), None)
    fix = (
        f"`sudo systemctl restart {Path(svc).stem}` with no mesh open"
        if svc
        else "re-apply the host's IOMMU domain setup for those cards with no mesh open"
    )
    return (
        f"IOMMU domains differ across the box's cards: {cards}; Metal uses one IOMMU mode for all, so these cards' "
        f"pinned host buffers fail (the real-time profiler loses them: no device time from those chips). Fix (root, a "
        f"person): {fix}"
    )


ENGINE_WAIT = "tt-d-gen build missing"
ENGINE_MARKS = ("dgen_engine", "use_engine")  # the runner tests that tt-d-gen's engine drives (testing/dgen_engine.py)


def engine_tests(repo, cmd: str) -> list[str]:
    """F203: the gate's test files that tt-d-gen's engine drives (they name dgen_engine or use_engine)."""
    out = []
    for word in shlex.split(cmd.replace("&&", " ")):
        p = Path(repo) / word.split("::")[0]
        if p.suffix == ".py" and p.is_file() and any(m in p.read_text(errors="replace") for m in ENGINE_MARKS):
            out.append(word)
    return out


def engine_missing(spec, cmd: str) -> str | None:
    """F203: why a gate that runs a runner test cannot pass (serving.require_engine and no tt-d-gen build), else None.
    The gate would fail on the missing build and start an agent attempt that cannot succeed (DeepSeek-V4.1 8x4 K.1,
    2026-10-27: the engine was built in parallel with the run)."""
    from models.demos.common.bringup.testing import dgen_engine

    tests = engine_tests(spec.repo, cmd)
    if not tests or not dgen_engine.engine_required(spec):
        return None
    build, why = dgen_engine.find_build(spec)
    if build is not None:
        return None
    return (
        f"{ENGINE_WAIT}: {why}; {', '.join(Path(t).name for t in tests)} need it (serving.require_engine). Build it "
        f"(dgen-build agent, REPO={dgen_engine.server_repo(spec)}); the run goes on with independent tasks and "
        "starts this one once the build is found"
    )


def commit_preflight(repo) -> list[str]:
    """F191: what would make a gate's commit fail, checked when the run starts (CK.1 on the 8x4 node: a passed gate's
    commit was refused for a missing git identity, found only after the gate). Empty when a commit can go through.
    (Here, not in core/gate.py: a measured file's edit would re-run every passed unit, core/stamps.py.)"""
    problems = []
    for ident in ("GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT"):
        r = subprocess.run(["git", "var", ident], cwd=repo, capture_output=True, text=True)
        if r.returncode != 0:
            problems.append(
                f"git cannot commit: no {ident.split('_')[1].lower()} identity ({r.stderr.strip().splitlines()[-1:]}); "
                "set user.name / user.email for this repo"
            )
    gdir = Path(subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"], cwd=repo, text=True).strip())
    for name, what in (("MERGE_HEAD", "a merge"), ("rebase-merge", "a rebase"), ("rebase-apply", "a rebase / am")):
        if (gdir / name).exists():
            problems.append(f"git is in the middle of {what} ({gdir / name}); finish or abort it")
    if (gdir / "index.lock").exists():
        problems.append(f"git index is locked ({gdir / 'index.lock'}); another git process, or a stale lock")
    return problems


def cached_cards(shape) -> str | None:
    """F192: the card subset already found for `shape` on this box (testing/cards.py cache; no probing), or None."""
    rows, cols = int(shape[0]), int(shape[1])
    n = cards.num_cards()
    if n == 0 or rows * cols >= n:
        return None
    return cards._load().get(cards._key(), {}).get(f"{rows}x{cols}")


def run_mesh_cards(shape, probe_fn=None) -> tuple[str | None, str | None]:
    """F192: the cards a run on a mesh smaller than this box shows to its gates and agents, at launch:
    (TT_VISIBLE_DEVICES value or None for every card, the problem when no subset forms the mesh). A miss is cached
    (probing up to cards.MAX_PROBES subsets on every resume is too slow); delete "<RxC>:none" from the cache to
    re-probe. (DeepSeek-V4.1: a 2x2 run on a Galaxy opened all 32 chips; no 2x2 subset exists there.)"""
    rows, cols = int(shape[0]), int(shape[1])
    n, k = cards.num_cards(), rows * cols
    if n == 0 or k >= n:
        return None, None
    name = f"{rows}x{cols}"
    miss = (
        f"no subset of this box's {n} cards forms a {name} mesh (probed); a {name} run here needs a mesh-graph "
        f'descriptor or a submesh of the whole box. To re-probe, delete "{name}:none" under "{cards._key()}" in '
        f"{cards.CACHE}"
    )
    if cards._load().get(cards._key(), {}).get(f"{name}:none"):
        return None, miss
    found = cards.visible_devices((rows, cols), probe_fn or cards.probe)
    if found is None:
        cache = cards._load()
        cache.setdefault(cards._key(), {})[f"{name}:none"] = True
        cards._save(cache)
        return None, miss
    return found, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "resume", "pause", "decide", "measure"])
    ap.add_argument("--spec", default=os.environ.get("BRINGUP_SPEC"))
    ap.add_argument("--until", help="run this task and the tasks it needs, then stop (F152); nothing else runs")
    ap.add_argument("--only", help="run just this task (if runnable)")
    ap.add_argument("--model")
    ap.add_argument("--attempts", type=int, default=None, help="override every role's attempt budget")
    ap.add_argument("--task", help="decide: the perf pick")
    how = ap.add_mutually_exclusive_group()
    for d in ("accept", "reject", "rerun"):
        how.add_argument(f"--{d}", dest="decision", action="store_const", const=d, help=f"decide: {d}")
    ap.add_argument("--note", default="", help="decide: stored with the decision")
    ap.add_argument(
        "--part", action="append", help="measure: a run (test file stem, rung, profile) or a kind (test, ladder)"
    )
    a = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if a.command in ("run", "resume", "measure"):
        checkout_env.pin()  # F68: gates, agents and tests use this checkout's cache, venv and paths
    spec = Spec.load(a.spec)
    orch = Orchestrator(spec, model=a.model)
    if a.command == "decide":
        if not a.task or not a.decision:
            ap.error("decide needs --task and one of --accept / --reject / --rerun")
        return decide(orch, a.task, a.decision, a.note)
    if a.command == "measure":
        if not a.task:
            ap.error("measure needs --task")
        return measure(orch, a.task, a.part)
    if a.command == "pause":
        orch.pause_file.write_text(now() + "\n")
        print(f"pause requested: the running orchestrator stops before its next task ({orch.pause_file})")
        return 0
    if a.command in ("run", "resume"):  # F191: a gate's commit must not be the first to find git unable to commit
        bad = commit_preflight(spec.repo)
        if not os.environ.get("TT_VISIBLE_DEVICES"):  # F192: a run mesh smaller than the box sees only its cards
            vis, why = run_mesh_cards(spec.mesh)
            if vis:
                os.environ["TT_VISIBLE_DEVICES"] = vis
                print(f"{now()} cards: mesh {spec.mesh} on this box: TT_VISIBLE_DEVICES={vis}", flush=True)
            bad += [why] if why else []
        for b in bad:
            print(f"{now()} PREFLIGHT: {b}", flush=True)
        if bad:
            print(f"{now()} STOPPED: the run did not start: fix the preflight problems above, then {a.command}")
            return STOPPED
        warn = iommu_problem()  # F194: only a profile needs every chip; the profiling tasks stop on it (run_task)
        if warn:
            print(f"{now()} PREFLIGHT (warning): {warn}", flush=True)
    orch.pause_file.unlink(missing_ok=True)
    orch.attempts_override = a.attempts
    if a.command == "resume":
        led = orch.led
        for tid, e in led.state().items():
            if isinstance(e, dict) and e.get("status") in ("STOPPED", "RUNNING"):  # RUNNING = an interrupted run
                h = {
                    "t": now(),
                    "status": "RESUMED",
                    "from": e["status"],
                    "role": e.get("role"),
                    "attempt": e.get("attempt"),
                }
                led.update(tid, status="TODO", history_add=h)
    OV.clear_status(orch.run_dir)  # F84: the overseer's "stopped because" line ends when the run starts again
    OV.agent_finished(orch.run_dir)  # F95: an agent record left by a crash names a dead (or reused) process
    (orch.run_dir / OV.ABORT_FILE).unlink(missing_ok=True)
    pid_file = orch.run_dir / W.PID_FILE  # the supervisor's watch (watch.py) tells a stopped run from a quiet one
    other = _live_orchestrator(pid_file)
    if other:  # F147: two orchestrators on one run raced (DeepSeek-V4.1 FIX.9: a second agent attempt beside a gate)
        print(f"another orchestrator (pid {other}) runs this run; stop it first (pause, then kill)", file=sys.stderr)
        return STOPPED
    pid_file.write_text(f"{os.getpid()} {now()}\n")
    # F70: the run's timeline (start / exit with its code) and every device run's lock wait and run time
    os.environ.setdefault("TT_DEVICE_TIMING_LOG", str(orch.run_dir / TL.DEVICE_LOG))
    TL.record(orch.run_dir, "start", command=a.command)
    rc = None
    try:
        rc = orch.run(until=a.until, only=a.only)
        return rc
    finally:
        TL.record(orch.run_dir, "exit", code=rc)
        pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
