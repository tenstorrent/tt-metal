# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Run one task's gate and record the verdict. The runner owns the verdict; tests only record numbers.

A gate passes only if, in this order:
  1. every dep is PASS or DEFERRED (else BLOCKED, nothing runs)
  2. every frozen file matches its hash (else FAIL, nothing runs)
  3. a device task's command goes through scripts/run_safe_pytest.sh or scripts/tt-probe.sh (else FAIL)
  4. the command exits 0 (exit 2 from the safe runner = HANG; the triage report is copied next to the log)
  5. every metric glob matches >= 1 recorded metric and every match meets its threshold
  6. every declared artifact exists
"""

from __future__ import annotations

import fnmatch
import operator
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from models.demos.common.bringup.core import freeze
from models.demos.common.bringup.core import metrics as M
from models.demos.common.bringup.core.ledger import Ledger, satisfied
from models.demos.common.bringup.core.spec import CODE_ROOT, Spec

OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "==": operator.eq}
SAFE_RUNNERS = ("scripts/run_safe_pytest.sh", "scripts/tt-probe.sh")
SAFE_HANG_RC = 2
TRIAGE = "generated/tt-triage/triage.txt"
DEFAULT_TRAILER = "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"


@dataclass
class GateResult:
    tid: str
    verdict: str  # PASS FAIL HANG BLOCKED
    rc: int | None = None
    lines: list[str] = field(default_factory=list)
    log: Path | None = None
    commit: str | None = None
    duration_s: float = 0.0

    @property
    def exit_code(self) -> int:
        return {"PASS": 0, "BLOCKED": 2}.get(self.verdict, 1)

    def summary(self) -> str:
        return "\n".join([f"{self.verdict} {self.tid}"] + self.lines)


def check_metrics(spec: dict[str, str], got: dict) -> tuple[bool, list[str]]:
    ok, lines = True, []
    for glob, cond in (spec or {}).items():
        op_s, val_s = str(cond).split()
        want = float(val_s)
        matched = {k: v["value"] for k, v in got.items() if fnmatch.fnmatchcase(k, glob)}
        if not matched:
            ok = False
            lines.append(f"  MISSING  {glob} (need {cond})")
            continue
        for k, v in sorted(matched.items()):
            try:
                good = v is not None and OPS[op_s](float(v), want)
            except (TypeError, ValueError):
                good = False
            ok &= good
            lines.append(f"  {'ok  ' if good else 'FAIL'}     {k} = {v} (need {cond})")
    return ok, lines


def _segments(cmd: str) -> list[list[str]]:
    """Split a shell command into simple commands (quote-aware), each as its word list."""
    lex = shlex.shlex(cmd, posix=True, punctuation_chars=";&|<>()")
    lex.whitespace_split = True
    segs, cur = [], []
    for tok in lex:
        if tok and set(tok) <= set(";&|()"):
            segs.append(cur)
            cur = []
        else:
            cur.append(tok)
    segs.append(cur)
    return [s for s in segs if s]


def _program(words: list[str]) -> list[str]:
    """Words of a simple command after leading VAR=value assignments."""
    i = 0
    while i < len(words) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[i]):
        i += 1
    return words[i:]


def _is_direct_pytest(words: list[str]) -> bool:
    if not words:
        return False
    if Path(words[0]).name == "pytest":
        return True
    return Path(words[0]).name.startswith("python") and words[1:3] == ["-m", "pytest"]


def device_policy_errors(task: dict) -> list[str]:
    """A device gate must enter through a safe runner, and no command may call pytest directly."""
    progs = [_program(w) for w in _segments(task["gate"]["cmd"])]
    progs = [p for p in progs if p]
    errs = [f"gate calls pytest directly: {' '.join(p)!r}" for p in progs if _is_direct_pytest(p)]
    if task.get("device"):
        for p in progs:
            if Path(p[0]).name.startswith("python") and not _is_direct_pytest(p) and _python_touches_device(p):
                errs.append(
                    f"device gate runs '{' '.join(p[:3])}' on device code directly; use {' or '.join(SAFE_RUNNERS)}"
                )
        if not any(p[0] in SAFE_RUNNERS for p in progs):
            errs.append(f"device gate does not use {' or '.join(SAFE_RUNNERS)}")
    return errs


def _python_touches_device(words: list[str]) -> bool:
    """True if a python command runs code that imports ttnn (or cannot be resolved, to stay safe)."""
    if "-c" in words:
        return "ttnn" in " ".join(words)
    if "-m" in words:
        mod = words[words.index("-m") + 1] if words.index("-m") + 1 < len(words) else ""
        base = CODE_ROOT / Path(*mod.split("."))
        f = base.with_suffix(".py") if base.with_suffix(".py").exists() else base / "__main__.py"
    else:
        f = next((CODE_ROOT / w for w in words[1:] if w.endswith(".py")), None)
    if f is None or not f.exists():
        return True
    return bool(re.search(r"^\s*(import ttnn|from ttnn)", f.read_text(errors="replace"), re.M))


def _uses_safe_runner(cmd: str) -> bool:
    return any(_program(w)[:1] and _program(w)[0] in SAFE_RUNNERS for w in _segments(cmd))


def run_name(ledger: Ledger) -> str:
    return ledger.state().get("_run", {}).get("name", "default")


def log_dir(spec: Spec, ledger: Ledger) -> Path:
    d = spec.run_dir(run_name(ledger)) / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def gate_env(spec: Spec, ledger: Ledger, tid: str) -> dict:
    # Pin imports to this checkout: an inherited PYTHONPATH pointing at another tt-metal tree once mixed
    # modules from two checkouts. The model spec rides along so generic tests know what they test.
    env = dict(os.environ, PYTHONPATH=str(CODE_ROOT))
    env[M.TASK_ENV] = tid
    env[M.RESULTS_ENV] = str(ledger.results_dir)
    if spec.path:
        env["BRINGUP_SPEC"] = str(spec.path)
    return env


FULL_MODEL_STEPS = ("integrate", "assemble", "contract", "perf")


def gate_command(task: dict) -> str:
    """The gate's shell command as run. Full-model steps skip run_safe_pytest's precompile pass: it runs the whole
    test once more with stubbed checks, which doubles a model-sized gate for kernels that are almost all cached."""
    cmd = task["gate"]["cmd"]
    if task.get("step") in FULL_MODEL_STEPS and "precompile" not in cmd:
        cmd = re.sub(r"(run_safe_pytest\.sh)(?=\s)", r"\1 --no-precompile", cmd)
    return cmd


def stage_paths(spec: Spec, ledger: Ledger, task: dict) -> list[str]:
    repo = spec.repo
    rel = lambda p: str(Path(p).resolve().relative_to(repo)) if Path(p).is_absolute() else p  # noqa: E731
    paths = [rel(ledger.state_path), rel(ledger.results_dir / f"{task['id']}.json"), rel(ledger.tasks_path)]
    paths.append(rel(ledger.dir / ".gitignore"))
    for extra in (
        "approvals.yaml",
        "findings.yaml",
        "__init__.py",
        "results/block_graphs.json",
        "spec.yaml",
        "supervision.md",
    ):
        if (ledger.dir / extra).exists():
            paths.append(rel(ledger.dir / extra))
    if ledger.breadcrumbs.exists():
        paths.append(rel(ledger.breadcrumbs))
    # The task's other result files (<task>_profile.json, <task>_ops_profile.json, ...) and the record of the model's
    # ttnn.bringup calls the derived-op test task writes.
    paths += [rel(p) for p in sorted(ledger.results_dir.glob(f"{task['id']}_*.json"))]
    if task.get("step") == "optests" and (ledger.results_dir / "fork_calls.json").exists():
        paths.append(rel(ledger.results_dir / "fork_calls.json"))
    paths += [p for p in task.get("paths", []) if _files_under(repo, [p])]  # an empty folder is no pathspec for git
    if task.get("step") == "contract":
        paths.append("models/demos/common/prefill")  # orchestrator.CONTRACT_SHARED: the contract agent may change it
    return [p for p in paths if (repo / p).exists() or _tracked(repo, p)]


def _tracked(repo: Path, p: str) -> bool:
    return subprocess.run(["git", "ls-files", "--error-unmatch", p], cwd=repo, capture_output=True).returncode == 0


def _files_under(repo: Path, paths: list[str]) -> list[str]:
    out = []
    for p in paths:
        full = repo / p
        if full.is_dir():
            out += [
                str(f.relative_to(repo))
                for f in sorted(full.rglob("*"))
                if f.is_file() and "__pycache__" not in f.parts
            ]
        elif full.is_file():
            out.append(p)
    return out


def format_paths(repo: Path, paths: list[str]) -> None:
    """Run the repo's pre-commit hooks (black, EOF fixer, ...) on the files now, so that what a gate tests and
    what freeze hashes is byte-identical to what the commit stores. No-op without a pre-commit config."""
    files = _files_under(repo, paths)
    if not files or not (repo / ".pre-commit-config.yaml").exists() or not shutil.which("pre-commit"):
        return
    for _ in range(2):  # hooks that modify files exit 1; the second pass settles
        if subprocess.run(["pre-commit", "run", "--files", *files], cwd=repo, capture_output=True).returncode == 0:
            return


def git_commit(spec: Spec, paths: list[str], subject: str, body: str) -> str | None:
    """Commit only the given paths, so work in progress elsewhere in the tree never leaks into a gate commit."""
    repo = spec.repo
    subprocess.run(["git", "add", "-A", "--", *paths], cwd=repo, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--", *paths], cwd=repo).returncode == 0:
        return None
    trailer = spec.get("commit_trailer", DEFAULT_TRAILER)
    msg = f"{subject}\n\n{body}\n" + (f"\n{trailer}\n" if trailer else "")
    commit = ["git", "commit", "-q", "-m", msg, "--", *paths]
    # pre-commit hooks may still rewrite files (black, EOF fixer): re-stage and retry once.
    if subprocess.run(commit, cwd=repo, capture_output=True).returncode != 0:
        subprocess.run(["git", "add", "-A", "--", *paths], cwd=repo, check=True)
        r = subprocess.run(commit, cwd=repo, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"git commit failed:\n{r.stdout}\n{r.stderr}")
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo, text=True).strip()


def commit_subject(spec: Spec, task: dict) -> str:
    return f"[{spec.tag}][{task['id']}] {task['title']}"


def task_commit(spec: Spec, tid: str) -> str:
    return subprocess.run(
        ["git", "log", "-1", "--format=%h", "--fixed-strings", f"--grep=[{spec.tag}][{tid}] "],
        cwd=spec.repo,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_gate(
    spec: Spec,
    ledger: Ledger,
    tid: str,
    commit: bool = False,
    force: bool = False,
    extra_env: dict | None = None,
    record: bool = True,
) -> GateResult:
    """Run the gate. record=False runs it without touching state.json (used by freeze's stub check)."""
    task = ledger.task(tid)
    state = ledger.state()
    blocked = [d for d in task.get("deps", []) if not satisfied(state.get(d, {}).get("status"))]
    if blocked and not force:
        return GateResult(tid, "BLOCKED", lines=[f"  deps not PASS or DEFERRED: {blocked}"])

    now = lambda: time.strftime("%Y-%m-%dT%H:%M:%S")  # noqa: E731
    unfrozen = []
    if record and task.get("tests") and not task.get("frozen"):  # freeze's own validation runs use record=False
        unfrozen = ["tests declared but not frozen (run: freeze <id>)"]
    pre = [f"  {e}" for e in unfrozen + freeze.verify(spec.repo, task) + device_policy_errors(task)]
    if pre:
        res = GateResult(tid, "FAIL", lines=pre)
        if record:
            ledger.update(tid, status="FAIL", last_run=now(), reason=pre, history_add={"t": now(), "status": "FAIL"})
        return res

    if commit:
        format_paths(spec.repo, task.get("paths", []))
    M.reset(tid, ledger.results_dir)
    if record:
        ledger.update(tid, status="RUNNING", started=now())
    log = log_dir(spec, ledger) / (f"{tid}.log" if record else f"{tid}.check.log")
    env = gate_env(spec, ledger, tid)
    env.update(extra_env or {})
    cmd = gate_command(task)
    t0 = time.time()
    with open(log, "w") as f:
        f.write(f"$ {cmd}\n")
        f.flush()
        rc = subprocess.run(cmd, shell=True, cwd=spec.repo, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    dur = time.time() - t0

    got = M.load(tid, ledger.results_dir)
    m_ok, lines = check_metrics(task["gate"].get("metrics", {}), got)
    missing = [a for a in task.get("artifacts", []) if not (spec.repo / a).exists()]
    hang = rc == SAFE_HANG_RC and _uses_safe_runner(task["gate"]["cmd"])
    verdict = "PASS" if (rc == 0 and m_ok and not missing) else ("HANG" if hang else "FAIL")
    head = [f"  cmd rc={rc} ({dur:.0f}s), log: {log}"]
    if hang and (spec.repo / TRIAGE).exists():
        shutil.copy(spec.repo / TRIAGE, log.with_suffix(".triage.txt"))
        head.append(f"  triage: {log.with_suffix('.triage.txt')}")
    res = GateResult(tid, verdict, rc, head + lines + [f"  MISSING artifact {a}" for a in missing], log, None, dur)
    if not record:
        return res

    entry = ledger.state().get(tid, {})
    ledger.update(
        tid,
        status=verdict,
        last_run=now(),
        duration_s=round(dur, 1),
        rc=rc,
        attempts=entry.get("attempts", 0) + (0 if verdict == "PASS" else 1),
        metrics={k: v["value"] for k, v in got.items()},
        log=str(log),
        reason=[] if verdict == "PASS" else res.lines,
        history_add={"t": now(), "status": verdict},
    )
    if commit and verdict == "PASS":
        with ledger.locked():
            res.commit = git_commit(
                spec,
                stage_paths(spec, ledger, task),
                commit_subject(spec, task),
                f"Gate: PASS\n" + "\n".join(res.lines),
            )
    return res
