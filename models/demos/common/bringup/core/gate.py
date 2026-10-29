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
A gate that passes but whose commit git refuses is COMMIT_PENDING (F149): `commit_only` commits it later, nothing re-runs.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import operator
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from models.demos.common.bringup.core import freeze, hangdump
from models.demos.common.bringup.core import metrics as M
from models.demos.common.bringup.core import stamps
from models.demos.common.bringup.core.ledger import Ledger, fixed_accuracy, satisfied
from models.demos.common.bringup.core.spec import CODE_ROOT, Spec

OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "==": operator.eq}
SAFE_RUNNERS = ("scripts/run_safe_pytest.sh", "scripts/tt-probe.sh")
SAFE_HANG_RC = 2
TRIAGE = "generated/tt-triage/triage.txt"
DEFAULT_TRAILER = "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"


@dataclass
class GateResult:
    tid: str
    verdict: str  # PASS FAIL HANG BLOCKED COMMIT_PENDING
    rc: int | None = None
    lines: list[str] = field(default_factory=list)
    log: Path | None = None
    commit: str | None = None
    duration_s: float = 0.0
    reused: list[str] = field(default_factory=list)  # F150: units restored from a PASS on the same stamp

    @property
    def exit_code(self) -> int:
        return {"PASS": 0, "BLOCKED": 2, "COMMIT_PENDING": 3}.get(self.verdict, 1)

    def summary(self) -> str:
        return "\n".join([f"{self.verdict} {self.tid}"] + self.lines)


# F129: state.json keeps only the gated metrics (and their limits); results/<task>.json keeps them all. The repo's
# 500 KB large-file hook refused DeepSeek-V4.1's state.json (502 KB) at FIX.6's commit.


def state_metrics(spec: dict[str, str], got: dict) -> dict:
    """The metrics state.json records for a gate run: the gated ones and their limits (F129)."""
    vals = {k: (v["value"] if isinstance(v, dict) else v) for k, v in got.items()}
    if not spec:  # a task without gated metrics keeps what it recorded
        return vals
    globs = list(spec or {})
    globs += [f"{g}_limit" for g in globs]  # F76: a '>= limit' gate's limit, for the dashboard
    return {k: v for k, v in vals.items() if any(fnmatch.fnmatchcase(k, g) for g in globs)}


def measured_only(spec: dict[str, str], got: dict) -> dict[str, str]:
    """F122: the thresholds whose metric a partial (quick) measurement recorded; the rest wait for the full gate."""
    return {g: c for g, c in (spec or {}).items() if any(fnmatch.fnmatchcase(k, g) for k in got)}


def check_metrics(spec: dict[str, str], got: dict) -> tuple[bool, list[str]]:
    ok, lines = True, []
    for glob, cond in (spec or {}).items():
        op_s, val_s = str(cond).split()
        # F76: "limit" = the <metric>_limit the test recorded (its noise floor; testing/noise_floor.py)
        from_test = val_s == "limit"
        matched = {
            k: v["value"]
            for k, v in got.items()
            if fnmatch.fnmatchcase(k, glob) and not (from_test and k.endswith("_limit"))
        }
        if not matched:
            ok = False
            lines.append(f"  MISSING  {glob} (need {cond})")
            continue
        for k, v in sorted(matched.items()):
            if (got.get(f"{k}_waived") or {}).get("value"):  # F85: the owner waived this step's noise-floor checks
                lines.append(f"  WAIVED   {k} = {v} (need {cond}; owner waiver, see the log)")
                continue
            want = (got.get(f"{k}_limit") or {}).get("value") if from_test else float(val_s)
            try:
                good = v is not None and want is not None and OPS[op_s](float(v), float(want))
            except (TypeError, ValueError):
                good = False
            ok &= good
            need = f"{op_s} {want} (its noise floor)" if from_test else cond
            if from_test and want is None:
                need = f"{cond}: no {k}_limit recorded"
            lines.append(f"  {'ok  ' if good else 'FAIL'}     {k} = {v} (need {need})")
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


COLLECT_ONLY = ("--collect-only", "--co")  # F115: lists the tests, runs none (no fixture, no device)


def _is_direct_pytest(words: list[str]) -> bool:
    if not words or any(w in COLLECT_ONLY for w in words):
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
    env[M.RECORD_ENV] = tid  # F79: a fix job's segments override BRINGUP_TASK only
    env[M.RESULTS_ENV] = str(ledger.results_dir)
    if spec.path:
        env["BRINGUP_SPEC"] = str(spec.path)
    return env


FULL_MODEL_STEPS = ("integrate", "assemble", "contract", "perf")


def step_command(step: str | None, cmd: str) -> str:
    """A gate command as run. Full-model steps skip run_safe_pytest's precompile pass: it runs the whole test once
    more with stubbed checks, which doubles a model-sized gate for kernels that are almost all cached."""
    if step in FULL_MODEL_STEPS and "precompile" not in cmd:
        cmd = re.sub(r"(run_safe_pytest\.sh)(?=\s)", r"\1 --no-precompile", cmd)
    return with_clean_wall(cmd)


UNIT_TASK = re.compile(r"^BRINGUP_TASK=(\S+)\s+(.*)$", re.S)
# F166: `for t in a b c; do <one command with $t / ${t}> [|| exit 1]; done`, a literal list over one template
FOR_LOOP = re.compile(r"^for\s+(\w+)\s+in\s+([\w.\- ]+?)\s*;\s*do\s+(.+?)\s*(?:\|\|\s*exit\s+1\s*)?;\s*done$", re.S)
SAFE_RUN = re.compile(r"^((?:\w+=\S*\s+)*)(scripts/run_safe_pytest\.sh)\s+(.+)$", re.S)
NO_VALUE_FLAGS = {"--run-all", "--no-precompile", "--dev", "-s", "-x", "-q", "-v"}  # flags that take no value


def unit_base(unit: str) -> str:
    """F166: the task a unit belongs to (a split unit is <task>#<label>)."""
    return unit.split("#", 1)[0]


def split_segment(seg: str) -> list[tuple[str, str]]:
    """F166: one gate segment as its separately stamped parts [(label, command)]: a for loop over a literal list of one
    command template, one part per value; a run_safe_pytest call over several test files (flags without values only),
    one part per file; else [("", seg)]. Each part fails or passes alone, so a resumed gate reuses the parts that
    passed (DeepSeek-V4.1 FIX.10: a stop at the 7th of 8 looped tests re-ran all 8, ~25 min)."""
    m = FOR_LOOP.match(seg.strip())
    if m:
        var, values, body = m.group(1), m.group(2).split(), m.group(3).strip()
        if values and any(t in body for t in (f"${var}", f"${{{var}}}")) and not re.search(r"[;&|]", body):
            return [(v, body.replace(f"${{{var}}}", v).replace(f"${var}", v)) for v in values]
        return [("", seg)]
    m = SAFE_RUN.match(seg.strip())
    if m:
        words = m.group(3).split()
        flags = [w for w in words if w.startswith("-")]
        files = [w for w in words if not w.startswith("-")]
        if len(files) > 1 and set(flags) <= NO_VALUE_FLAGS and all(".py" in f for f in files):
            pre = f"{m.group(1)}{m.group(2)}" + "".join(f" {x}" for x in flags)
            return [(Path(f.split("::")[0]).stem, f"{pre} {f}") for f in files]
    return [("", seg)]


def segment_label(seg: str, n: int) -> str:
    """F170: a readable, stable unit label for one gate segment: its test file's stem, else its python module, else
    its position among its task's segments (the same in the task's own gate and in a fix job that runs it)."""
    m = re.search(r"([\w./-]+)\.py\b", seg)
    if m:
        return Path(m.group(1)).name
    m = re.search(r"-m\s+([\w.]+)", seg)
    return m.group(1).rsplit(".", 1)[-1] if m else f"seg{n}"


TEST_RUN = re.compile(r"run_safe_pytest\.sh|\bpytest\b|tt-probe\.sh")


def gate_units(task: dict, tasks: dict | None = None) -> list[tuple[str, str]]:
    """F150: the gate as units (unit, command): a fix job's segments of one source task each (F79's BRINGUP_TASK=<id>
    prefix, which the runner sets as the unit's BRINGUP_TASK instead), else the whole command.
    F151: a unit runs as its source task's gate does (that task's step: --no-precompile for a full-model step).
    F166: a segment that is a loop over test files, or one call over several, is split into one unit per test.
    F170: every test-running `&&` segment is its own unit (<task>#<label>), so a resumed gate never re-runs a segment
    that passed (audit item 5: X.3 stopping in test_positions re-ran the full-target ladder and both profile runs). A
    segment that runs no test (a setup or a capture step) travels with the test segment after it (a trailing one with
    the last), so a kept unit never skips a step its test needs. A gate of one unit keeps the name <task>."""
    tid, cmd, tasks = task.get("id"), task["gate"]["cmd"], tasks or {}
    groups: list[tuple[str, list[str]]] = []  # (source task, segments): one test segment with its setup steps
    pending: list[str] = []
    pending_src = tid

    def flush():
        if not pending:
            return
        if groups and groups[-1][0] == pending_src:
            groups[-1][1].extend(pending)  # steps after a source's last test (e.g. writing its profile file)
        else:
            groups.append((pending_src, list(pending)))
        pending.clear()

    for seg in (x.strip() for x in cmd.split("&&")):
        m = UNIT_TASK.match(seg)
        src, body = (m.group(1), m.group(2).strip()) if m else (tid, seg)
        if pending and src != pending_src:
            flush()
        if TEST_RUN.search(body) or FOR_LOOP.match(body):
            groups.append((src, pending + [body]))
            pending.clear()
        else:
            pending.append(body)
            pending_src = src
    flush()
    per_src: dict[str, int] = {}
    for src, _ in groups:
        per_src[src] = per_src.get(src, 0) + 1
    units: list[tuple[str, str]] = []
    seen: set[str] = set()
    counter: dict[str, int] = {}
    for src, bodies in groups:
        counter[src] = counter.get(src, 0) + 1
        body = " && ".join(bodies)
        parts = split_segment(bodies[-1]) if len(bodies) == 1 else [("", body)]
        if len(parts) > 1:
            names = [(f"{src}#{label}", c) for label, c in parts]
        elif per_src[src] == 1:
            names = [(src, body)]
        else:
            names = [(f"{src}#{segment_label(bodies[-1], counter[src])}", body)]
        for i, (key, c) in enumerate(names):
            if key in seen:
                key = f"{key}.{counter[src]}.{i}"
            seen.add(key)
            units.append((key, c))
    if [u for u, _ in units] == [tid]:
        return [(tid, step_command(task.get("step"), cmd))]
    return [(u, step_command((tasks.get(unit_base(u)) or task).get("step"), c)) for u, c in units]


def gate_command(task: dict, tasks: dict | None = None) -> str:
    """The gate's shell command as run (a fix job's units with their BRINGUP_TASK prefix on every segment)."""
    tid = task.get("id")
    units = gate_units(task, tasks)
    if [u for u, _ in units] == [tid]:
        return units[0][1]
    if all(unit_base(u) == tid for u, _ in units):  # F166: one task's split units: the same command, unrolled
        return " && ".join(c for _, c in units)
    return " && ".join(f"BRINGUP_TASK={unit_base(u)} {seg.strip()}" for u, c in units for seg in c.split("&&"))


PROFILER_VARS = re.compile(
    r"\b(TT_METAL_(DEVICE_PROFILER|PROFILER_\w+)|BRINGUP_PROFILE_OPS|BRINGUP_FULL_PREFILL|BRINGUP_PROFILER)=\S+\s*"
)


DEVICE_PROFILER_VARS = re.compile(r"\bTT_METAL_(DEVICE_PROFILER|PROFILER_\w+)=\S+\s*")


def with_clean_wall(cmd: str) -> str:
    """F114: a profiled test_profile.py run is preceded by the same run with the device profiler off
    (BRINGUP_PROFILE_WALL=1: warm wall time only), so chunk_wall_ms never includes the profiler's instrumentation or
    buffer dumps; the profiled run then reports its own wall as chunk_wall_ms_profiled (BRINGUP_WALL_CLEAN=1).
    F155: the profiled run measures with the real-time profiler, so the device profiler's variables are dropped from
    it too (ledgers written before F155 carry them), unless the segment asks for the device profiler
    (BRINGUP_PROFILER=device, perf.profiler: device in ledger_gen)."""
    out = []
    for seg in cmd.split("&&"):
        s = seg.strip()
        if "test_profile.py" in s and "BRINGUP_WALL_CLEAN" not in s and "BRINGUP_PROFILE_WALL" not in s:
            prof = s if "BRINGUP_PROFILER=device" in s else DEVICE_PROFILER_VARS.sub("", s)
            out += [f"BRINGUP_PROFILE_WALL=1 {PROFILER_VARS.sub('', s)}", f"BRINGUP_WALL_CLEAN=1 {prof}"]
        else:
            out.append(s)
    return " && ".join(out) if len(out) != len(cmd.split("&&")) else cmd


def gate_outputs(ledger: Ledger, task: dict) -> list[Path]:
    """F55: the files a task's gate writes itself: results/<task>.json, results/<task>_*.json and, for the derived-op
    test task, results/fork_calls.json. One list, used three ways: the gate deletes them before it runs (so nothing an
    agent left there survives), the gate commits them, and the agent may change them (orchestrator.allowed_paths), since
    an agent that runs its own gate command rewrites them."""
    return outputs_in(ledger.results_dir, task)


def outputs_in(results: Path, task: dict) -> list[Path]:
    tid = task["id"]
    out = [results / f"{tid}.json", *sorted(results.glob(f"{tid}_*.json"))]
    out += sorted(results.glob(f"{tid}_*.json.gz"))  # F148: a profile's gzipped per-op rows
    if task.get("step") == "optests":
        out.append(results / "fork_calls.json")
        # F64 / F201: every other capture file the gate writes (the scaled run's, a settings variant's)
        import re

        for name in sorted(set(re.findall(r"\b(fork_calls_[\w-]+\.json)", (task.get("gate") or {}).get("cmd", "")))):
            out.append(results / name)
    return out


def stage_paths(spec: Spec, ledger: Ledger, task: dict) -> list[str]:
    repo = spec.repo
    rel = lambda p: str(Path(p).resolve().relative_to(repo)) if Path(p).is_absolute() else p  # noqa: E731
    paths = [rel(ledger.state_path), rel(ledger.results_dir / f"{task['id']}.json"), rel(ledger.tasks_path)]
    paths.append(rel(ledger.dir / ".gitignore"))
    # F183: the run's own files are in its ledger; the model's shared files in bringup_dir (the same dir unless seeded)
    for d, extras in (
        (ledger.dir, ("approvals.yaml", "results/block_graphs.json", "spec.yaml", "supervision.md")),
        (ledger.dir, ("results/knowledge_use",)),  # the agents' knowledge-use reports (knowledge/usage.py)
        (
            spec.bringup_dir,
            (
                "findings.yaml",
                "known_issues.md",  # the model's own lessons and repo pointers (agents write these, not the shared files)
                "known_issues_detail.md",
                "repo_map.md",
                "__init__.py",
                "BREADCRUMBS.md",
                "serving_contract.md",  # F58: the serving contract (SC.1) and its test list
                "contract_tests.yaml",
                "official_taps.yaml",  # F80: the official code mapped onto the steps (testing/noise_floor.py)
            ),
        ),
    ):
        for extra in extras:
            if (d / extra).exists() and rel(d / extra) not in paths:
                paths.append(rel(d / extra))
    # The files the task's own gate writes (gate_outputs): <task>_profile.json, <task>_ops_profile.json, ... and the
    # record of the model's ttnn.bringup calls the derived-op test task writes.
    paths += [rel(p) for p in gate_outputs(ledger, task) if p.exists() and p.name != f"{task['id']}.json"]
    paths += [p for p in task.get("paths", []) if _files_under(repo, [p])]  # an empty folder is no pathspec for git
    if (
        task.get("step") == "settings"
    ):  # Z.1 moves switches in the model's code and the forks (agents/settings-audit.md)
        md = Path(spec.model_dir)
        paths += [rel(p) for p in (md / "tt", md / "bringup" / "hooks.py") if p.exists()]
        paths.append("ttnn/ttnn/bringup")
    if task.get("step") == "serving":  # SC.1's frozen contract tests (agents/serving-contract.md)
        tests = Path(spec.model_dir) / "tests" / "bringup" / "contract"
        if tests.is_dir():
            paths.append(rel(tests))
    if task.get("step") == "contract":
        paths.append("models/demos/common/prefill")  # orchestrator.CONTRACT_SHARED: the contract agent may change it
        paths.append(rel(spec.bringup_dir / "hooks.py"))  # and the model's hooks (contract_state_pcc)
    # F48: every agent may write ttnn/ttnn/bringup (a fork, or a change to one behind an option); one agent runs at a
    # time, so what changed there is this task's. Only changed files, so the gate's formatting pass does not touch the
    # other forks. The shared knowledge files are no longer agent-writable (the overseer curates them at run end).
    paths += _dirty_under(repo, [BRINGUP_OPS])
    return list(dict.fromkeys(p for p in paths if (repo / p).exists() or _tracked(repo, p)))


BRINGUP_OPS = "ttnn/ttnn/bringup"


def _dirty_under(repo: Path, prefixes: list[str]) -> list[str]:
    """Modified or untracked files (not deleted, no __pycache__) at or under any prefix."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "-uall", "-z", "--", *prefixes], cwd=repo, capture_output=True, text=True
    ).stdout
    files = []
    for entry in out.split("\0"):
        if len(entry) < 4:
            continue
        p = entry[3:]
        if "__pycache__" not in p and (repo / p).is_file():
            files.append(p)
    return sorted(files)


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


def _git_retry(cmd: list[str], repo, check: bool = True, tries: int = 20, wait: float = 3.0):
    """Run a git command, waiting out another process's index.lock (a person's or the overseer's git command) instead
    of crashing the run (F74). Other failures are returned (check=False) or raised (check=True) as before."""
    for i in range(tries):
        r = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        if r.returncode == 0 or "index.lock" not in (r.stderr + r.stdout) or i == tries - 1:
            break
        time.sleep(wait)
    if check and r.returncode != 0:
        raise subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)
    return r


# F210: the generated dashboard pages (re-exported after every gate, published as artifacts; ~0.9 MB each by
# DeepSeek-V4.1 FIX.8) never go into a gate commit: the repo's 500 KB hook refused FIX.8's passed gate over them.
GENERATED = [":(exclude,glob)**/dashboard/*.html"]


def git_commit(spec: Spec, paths: list[str], subject: str, body: str) -> str | None:
    """Commit only the given paths, so work in progress elsewhere in the tree never leaks into a gate commit."""
    repo = spec.repo
    _git_retry(["git", "add", "-A", "--", *paths, *GENERATED], repo)
    _git_retry(["git", "reset", "-q", "--", *[f.split(")", 1)[1] for f in GENERATED]], repo, check=False)
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--", *paths, *GENERATED], cwd=repo).returncode == 0:
        return None
    trailer = spec.get("commit_trailer", DEFAULT_TRAILER)
    msg = f"{subject}\n\n{body}\n" + (f"\n{trailer}\n" if trailer else "")
    commit = ["git", "commit", "-q", "-m", msg, "--", *paths, *GENERATED]
    # pre-commit hooks may still rewrite files (black, EOF fixer): re-stage and retry once.
    if _git_retry(commit, repo, check=False).returncode != 0:
        _git_retry(["git", "add", "-A", "--", *paths, *GENERATED], repo)
        r = _git_retry(commit, repo, check=False)
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


def _merge_metrics(results: Path, tid: str, metrics: dict) -> None:
    p = M._path(tid, results)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(p.read_text()) if p.exists() else {"task": tid, "metrics": {}}
    data["metrics"].update(metrics)
    p.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def unit_stamps(spec: Spec, task: dict, env: dict, tasks: dict | None = None, keys: tuple | None = None) -> list:
    """F150: (unit, command, stamp, stamp id) for every unit of the gate, on the tree as it is now (keys: the
    (code_key, goldens_key) pair when the caller already has it)."""
    tasks = tasks or {}
    code, goldens = keys or (stamps.code_key(spec.repo, spec), stamps.goldens_key(spec))
    out = []
    for u, cmd in gate_units(task, tasks):
        frozen, uenv = (tasks.get(unit_base(u)) or task).get("frozen"), dict(env, **{M.TASK_ENV: unit_base(u)})
        st = stamps.stamp(cmd, frozen, uenv, code, goldens, stamps.waiver_key(spec, cmd))  # F162
        out.append((u, cmd, st, stamps.stamp_id(st)))
    return out


def unit_ok(limits: dict, got: dict, whole: bool) -> bool:
    """A unit's metrics within the gate's limits: all of them for a one-unit gate, else the ones it recorded."""
    return check_metrics(limits if whole else measured_only(limits, got), got)[0]


def reusable(rec: dict | None, limits: dict, whole: bool) -> bool:
    """F150: a kept unit run counts as this gate's only when it passed and its metrics are within today's limits."""
    return bool(rec) and rec.get("verdict") == "PASS" and unit_ok(limits, rec["metrics"], whole)


INFRA_STOP = re.compile(  # F169: a unit that did not finish for the box's or the clock's sake: STOPPED, not FAILED
    r"Timeout \(>[\d.]+s\) from pytest-timeout|Timed out while waiting for active ethernet core|"
    r"Try resetting the board|No Tenstorrent devices|Failed to open device|fabric router .* timed out",
    re.I,
)


def under_test_paths(task: dict, tasks: dict | None = None) -> list[str]:
    """F169: the paths the task may change, and for a fix job those of the gate tasks its units run (their code)."""
    tasks = tasks or {}
    out = list(task.get("paths") or [])
    for u, _ in gate_units(task, tasks):
        out += list((tasks.get(unit_base(u)) or {}).get("paths") or [])
    return sorted(set(out))


def _kept(d: dict | None, under: str, cmd: str, uenv: dict, limits: dict, whole: bool) -> bool:
    """F169: a unit that passed in this task's gate on the same code under test, command and environment."""
    return (
        bool(d)
        and d.get("verdict") == "PASS"
        and d.get("under_test") in ((under,) if isinstance(under, str) else tuple(under))  # F175: (semantic, bytes)
        and d.get("cmd") == stamps.sha16(cmd)
        and d.get("env") == stamps.env_key(uenv)
        and unit_ok(limits, d.get("metrics") or {}, whole)
    )


def run_units(
    spec: Spec,
    task: dict,
    env: dict,
    log: Path,
    results: Path,
    store: stamps.Store | None,
    fresh: bool = False,
    tasks: dict | None = None,
    delta: stamps.Delta | None = None,
    only: set | None = None,
    rerun: set | frozenset = frozenset(),
    on_unit=None,
) -> tuple[int, list[str], dict[str, str]]:
    """F150: the gate's units in order (&&: the first failure stops), each with its own BRINGUP_TASK. With a store,
    every run unit is kept under its stamp, and a unit whose stamp has a PASS (its metrics still within the gate's
    limits) is restored instead of run. Returns (rc, reused units, {unit: stamp id}).
    F169: with ``delta`` (the task's own record of its units), a unit that passed in this task's gate on the same code
    under test is kept, whatever the overseer edited since (the spec, waivers, the framework): a gate never restarts,
    it runs the delta. ``only``: run just these units (gate --unit); ``rerun``: run these even if kept (gate --rerun).
    """
    tid, tasks = task["id"], tasks or {}
    limits = task["gate"].get("metrics") or {}
    plan = (
        unit_stamps(spec, task, env, tasks)
        if store is not None
        else [(u, c, None, None) for u, c in gate_units(task, tasks)]
    )
    unders = (  # F175: (semantic, bytes): a record written before F173 carries the byte key of an identical tree
        stamps.under_test_key(spec.repo, spec, under_test_paths(task, tasks), with_bytes=True)
        if delta is not None
        else ("", "")
    )
    under = unders[0]
    rc, reused, stop = 0, [], False
    sids = {u: sid for u, _, _, sid in plan if sid}
    n = len(plan)
    note = on_unit or (lambda *a: None)
    with open(log, "a" if only else "w") as f:
        f.write(f"$ {gate_command(task, tasks)}\n")
        for k, (u, cmd, st, sid) in enumerate(plan, 1):
            if only is not None and u not in only:
                continue
            uenv = dict(env, **{M.TASK_ENV: unit_base(u)})  # F166: a split unit runs as its task
            d = delta.get(u) if delta is not None and not fresh and u not in rerun else None
            if only is None and _kept(d, unders, cmd, uenv, limits, len(plan) == 1):
                _merge_metrics(results, tid, d["metrics"])
                srec = store.get(u, d.get("sid")) if store is not None and d.get("sid") else None
                if srec:
                    store.restore(u, d["sid"], srec, results, tid)
                f.write(f"# {u}: reused: same code under test, passed in this gate at {d.get('t')} (F169), not run\n")
                reused.append(u)
                note(k, n, u, "KEPT", 0.0)
                continue
            rec = None if store is None or fresh or u in rerun else store.get(u, sid)
            if reusable(rec, limits, len(plan) == 1) and only is None:
                _merge_metrics(results, tid, rec["metrics"])
                store.restore(u, sid, rec, results, tid)
                f.write(f"# {u}: reused: same code (PASS at {rec.get('t')} in the {rec.get('by')} gate), not run\n")
                if delta is not None:
                    delta.put(u, _delta_rec("PASS", under, cmd, uenv, sid, rec["metrics"], 0.0))
                reused.append(u)
                note(k, n, u, "KEPT", 0.0)
                continue
            if k > 1 and (store is not None or delta is not None):
                # F170 (audit item 15): the unit's stamp as the tree is when it starts, not as it was at the gate's start
                # (an edit mid-gate filed later units under the old key: re-runs, or a reuse that was never measured)
                if store is not None:
                    st, sid = next((x[2], x[3]) for x in unit_stamps(spec, task, env, tasks) if x[0] == u)
                if delta is not None:
                    unders = stamps.under_test_key(spec.repo, spec, under_test_paths(task, tasks), with_bytes=True)
                    under = unders[0]
            if len(plan) > 1:
                f.write(f"# {u}\n$ {cmd}\n")
            f.flush()
            start = f.tell()
            before = M.load(tid, results)
            seen = {p: p.stat().st_mtime_ns for p in outputs_in(results, task) if p.exists()}
            t0 = time.time()
            rc = subprocess.run(cmd, shell=True, cwd=spec.repo, env=uenv, stdout=f, stderr=subprocess.STDOUT).returncode
            if rc == SAFE_HANG_RC and _uses_safe_runner(cmd):  # F204: before the next unit's mesh open wipes it
                kept = hangdump.keep(spec.repo, hangdump.dest_for(log, u))
                f.write(f"\n# {u}: HANG; triage + inspector kept in {kept}\n")
                f.flush()
            if store is not None:
                got = {k: v for k, v in M.load(tid, results).items() if before.get(k) != v}
                ok = unit_ok(limits, got, len(plan) == 1)
                hang = rc == SAFE_HANG_RC and _uses_safe_runner(cmd)
                outs = [
                    p
                    for p in outputs_in(results, task)
                    if p.exists() and p.name != f"{tid}.json" and seen.get(p) != p.stat().st_mtime_ns
                ]
                verdict = "PASS" if rc == 0 and ok else ("HANG" if hang else "FAIL")
                rec = {"stamp": st, "verdict": verdict, "rc": rc, "s": round(time.time() - t0, 1), "metrics": got}
                store.put(u, sid, dict(rec, by=tid, t=time.strftime("%Y-%m-%dT%H:%M:%S")), outs, tid)
                sids[u] = sid
                if delta is not None:
                    f.flush()
                    with open(log, errors="replace") as lf:
                        lf.seek(start)
                        infra = rc != 0 and bool(INFRA_STOP.search(lf.read()))
                    dv = "STOPPED" if infra else ("PASS" if verdict == "PASS" else "FAIL")
                    delta.put(u, _delta_rec(dv, under, cmd, uenv, sid, got, round(time.time() - t0, 1)))
                    stop = verdict != "PASS"  # F169: a unit whose metrics missed stops the gate here too
                    note(k, n, u, dv, round(time.time() - t0, 1))
            if rc != 0 or stop:
                break
    return rc, reused, sids


def _delta_rec(verdict: str, under: str, cmd: str, uenv: dict, sid, metrics: dict, secs: float) -> dict:
    return {
        "verdict": verdict,
        "under_test": under,
        "cmd": stamps.sha16(cmd),
        "env": stamps.env_key(uenv),
        "sid": sid,
        "metrics": metrics,
        "s": secs,
    }


def unit_states(spec: Spec, ledger: Ledger, tid: str) -> dict[str, str]:
    """F169: {unit: its last verdict in this task's gate (PASS, FAIL, STOPPED) or 'not run'}, in gate order."""
    tasks = ledger.tasks()
    delta = stamps.Delta(spec.run_dir(run_name(ledger)), tid).all()
    return {u: (delta.get(u) or {}).get("verdict", "not run") for u, _ in gate_units(tasks[tid], tasks)}


def progress_lines(spec: Spec, ledger: Ledger, tid: str) -> list[str]:
    st = unit_states(spec, ledger, tid)
    done = [u for u, v in st.items() if v == "PASS"]
    left = [u for u, v in st.items() if v != "PASS"]
    return [
        f"  units passed: {', '.join(done) or '-'}",
        f"  units remaining: {', '.join(left) or '-'} (gate {tid} --continue runs only these)",
    ]


def delta_status(spec: Spec, ledger: Ledger, tid: str) -> list[tuple[str, str]]:
    """F169: (unit, state) for every unit of the task's gate: kept (passed on the current code under test), failed,
    stopped, stale (passed on other code under test: runs again) or not run."""
    tasks = ledger.tasks()
    task = tasks[tid]
    delta = stamps.Delta(spec.run_dir(run_name(ledger)), tid)
    under = stamps.under_test_key(spec.repo, spec, under_test_paths(task, tasks), with_bytes=True)  # F175
    limits = task["gate"].get("metrics") or {}
    env = gate_env(spec, ledger, tid)
    units = gate_units(task, tasks)
    out = []
    for u, cmd in units:
        d = delta.get(u)
        uenv = dict(env, **{M.TASK_ENV: unit_base(u)})
        if _kept(d, under, cmd, uenv, limits, len(units) == 1):
            out.append((u, f"kept (PASS at {d.get('t')})"))
        elif not d:
            out.append((u, "not run"))
        elif d.get("verdict") == "PASS":
            out.append((u, "stale (passed on other code under test or command: runs again)"))
        else:
            out.append((u, f"{d.get('verdict')} at {d.get('t')}"))
    return out


def run_gate(
    spec: Spec,
    ledger: Ledger,
    tid: str,
    commit: bool = False,
    force: bool = False,
    extra_env: dict | None = None,
    record: bool = True,
    reuse: dict | None = None,
    fresh: bool = False,
    units: list[str] | None = None,
    rerun: list[str] | None = None,
    on_unit=None,
    check: bool = False,
) -> GateResult:
    """Run the gate. record=False runs it without touching state.json (used by freeze's stub check).
    check (F170, `gate <task> --check`, the agent's way to run its gate): like record=False (no state, no commit), but
    every unit is kept in the stamp store and the task's delta, so the orchestrator's gate after the agent runs only
    what the agent did not pass on the same code (audit item 2: every attempt cost one extra full gate).
    reuse (F105): {"metrics": {...}, "source": text} measured by the same commands on the same code (a perf pick's A/B
    'on' side): recorded, checked and committed like a run of the command, without running it again.
    F150: a unit whose stamp has a recorded PASS is restored, not run (fresh=True runs every unit).
    F169: a gate never restarts: a unit that passed in this task's gate on the same code under test is kept (the
    delta runs); units (gate --unit) runs only those units and records them, without a gate verdict; rerun (gate
    --rerun) runs the named units again even if kept. fresh (owner only) runs every unit."""
    if units:
        return run_units_only(spec, ledger, tid, units)
    if check:
        record, commit = False, False
    tasks = ledger.tasks()
    if tid not in tasks:
        ledger.task(tid)  # raises
    task = tasks[tid]
    state = ledger.state()
    if record and reuse is None and (state.get(tid) or {}).get("status") == "COMMIT_PENDING":  # F149: never re-run
        if commit:
            return commit_only(spec, ledger, tid)
        return GateResult(tid, "COMMIT_PENDING", lines=[f"  passed, commit pending: `gate {tid} --commit-only`"])
    blocked = [d for d in task.get("deps", []) if not satisfied(state.get(d, {}).get("status"))]
    if blocked and not force:
        return GateResult(tid, "BLOCKED", lines=[f"  deps not PASS or DEFERRED: {blocked}"])

    now = lambda: time.strftime("%Y-%m-%dT%H:%M:%S")  # noqa: E731
    unfrozen = []
    if record and task.get("tests") and not task.get("frozen"):  # freeze's own validation runs use record=False
        unfrozen = ["tests declared but not frozen (run: freeze <id>)"]
    pre = [
        f"  {e}" for e in unfrozen + freeze.verify(spec.repo, task) + device_policy_errors(task) + fixed_accuracy(task)
    ]
    if pre:
        res = GateResult(tid, "FAIL", lines=pre)
        if record:
            ledger.update(tid, status="FAIL", last_run=now(), reason=pre, history_add={"t": now(), "status": "FAIL"})
        return res

    if commit:  # F170 (audit item 22): only the task's changed files; none changed (a reused gate): no pre-commit run
        changed = _dirty_under(spec.repo, list(task.get("paths") or [])) if task.get("paths") else []
        if changed:
            format_paths(spec.repo, changed)
    # F172: a validation run (freeze's reference / stub / mutation checks: record=False, not --check) writes its metrics
    # to a scratch dir: the zero stub's rel 1.0 overwrote the task's real results (DeepSeek-V4.1 C.swa.shared_expert)
    results_dir = ledger.results_dir if (record or check) else spec.run_dir(run_name(ledger)) / "validation" / tid
    M.reset(tid, results_dir)
    for out in (
        gate_outputs(ledger, task) if results_dir == ledger.results_dir else ()
    ):  # F55: only what this gate writes now counts
        if out.name != f"{tid}.json":
            out.unlink(missing_ok=True)
    if record:
        ledger.update(tid, status="RUNNING", started=now())
    log = log_dir(spec, ledger) / (f"{tid}.log" if record else f"{tid}.check.log")
    env = gate_env(spec, ledger, tid)
    env.update(extra_env or {})
    env[M.RESULTS_ENV] = str(results_dir)  # F172
    cmd = gate_command(task, tasks)
    t0 = time.time()
    reused, sids = [], {}
    if reuse is not None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        out = M._path(tid, results_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {"task": tid, "metrics": {k: {"value": v, "t": stamp} for k, v in (reuse.get("metrics") or {}).items()}}
        out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        log.write_text(f"$ {cmd}\nnot run again (F105): {reuse.get('source', '')}\n")
        rc = 0
    else:
        store = stamps.Store(spec.run_dir(run_name(ledger))) if record or check else None
        fresh = fresh or any(not (spec.repo / a).exists() for a in task.get("artifacts", []))  # made only by a run
        delta = stamps.Delta(spec.run_dir(run_name(ledger)), tid) if record or check else None
        rc, reused, sids = run_units(
            spec, task, env, log, results_dir, store, fresh, tasks, delta, None, frozenset(rerun or ()), on_unit
        )
    dur = time.time() - t0

    got = M.load(tid, results_dir)
    thresholds = task["gate"].get("metrics", {})
    if reuse is not None and reuse.get("partial"):  # F122: a quick A/B; the full gate runs before the final task
        thresholds = measured_only(thresholds, got)
    m_ok, lines = check_metrics(thresholds, got)
    missing = [a for a in task.get("artifacts", []) if not (spec.repo / a).exists()]
    hang = rc == SAFE_HANG_RC and _uses_safe_runner(task["gate"]["cmd"])
    verdict = "PASS" if (rc == 0 and m_ok and not missing) else ("HANG" if hang else "FAIL")
    head = [f"  cmd rc={rc} ({dur:.0f}s), log: {log}"]
    if reused:
        head.append(f"  reused: same code, not run (F150): {', '.join(reused)}")
    if hang and (spec.repo / TRIAGE).exists():
        shutil.copy(spec.repo / TRIAGE, log.with_suffix(".triage.txt"))
        head.append(f"  triage: {log.with_suffix('.triage.txt')}")
    kept = sorted(log.parent.glob(f"{log.stem}.hang-*"), key=lambda p: p.stat().st_mtime) if hang else []
    if hang and not kept:  # F204: a single-command gate (no unit loop) keeps its dump here
        kept = [hangdump.keep(spec.repo, hangdump.dest_for(log, tid))]
    if kept and kept[-1]:
        head.append(f"  hang dump (triage + inspector, F204): {kept[-1]}")
    if record and reuse is None and verdict != "PASS":  # F169: where the gate stopped, for --continue and the overseer
        head += progress_lines(spec, ledger, tid)
    res = GateResult(tid, verdict, rc, head + lines + [f"  MISSING artifact {a}" for a in missing], log, None, dur)
    res.reused = reused
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
        metrics=state_metrics(task["gate"].get("metrics", {}), got),
        log=str(log),
        reason=[] if verdict == "PASS" else res.lines,
        stamp={"units": sids, "reused": reused} if sids else None,  # F150
        units=unit_states(spec, ledger, tid) if reuse is None else None,  # F169: {unit: PASS|FAIL|STOPPED|not run}
        history_add={"t": now(), "status": verdict, "s": round(dur, 1), **({"reused": reused} if reused else {})},
    )
    if verdict == "PASS" and task.get("step") == "goldens":  # F134: a drift changes only through its G task
        from models.demos.common.bringup.reference.drift import pin

        pinned = pin(spec, ledger)
        res.lines.append(f"  drift files pinned in {len(pinned)} end-to-end gates: {', '.join(pinned) or '-'}")
    if commit and verdict == "PASS":
        try:
            with ledger.locked():
                res.commit = git_commit(
                    spec,
                    stage_paths(spec, ledger, task),
                    commit_subject(spec, task),
                    f"Gate: PASS\n" + "\n".join(res.lines),
                )
        except RuntimeError as e:  # F149: the checks passed; only the commit waits (F110 made it a FAIL and a re-run)
            return _refused(spec, ledger, task, res, e)
    return res


def run_units_only(spec: Spec, ledger: Ledger, tid: str, names: list[str]) -> GateResult:
    """F169 (gate --unit): run only the named units of the task's gate (after a fix, to see the failed one pass), record
    them in the task's delta; the task's status is not changed (gate --continue then runs the rest and commits)."""
    tasks = ledger.tasks()
    task = tasks[tid]
    known = [u for u, _ in gate_units(task, tasks)]
    bad = [n for n in names if n not in known]
    if bad:
        return GateResult(tid, "FAIL", lines=[f"  no unit {bad} in {tid}'s gate; units: {', '.join(known)}"])
    log = log_dir(spec, ledger) / f"{tid}.units.log"
    env = gate_env(spec, ledger, tid)
    store = stamps.Store(spec.run_dir(run_name(ledger)))
    delta = stamps.Delta(spec.run_dir(run_name(ledger)), tid)
    t0 = time.time()
    rc, _, _ = run_units(spec, task, env, log, ledger.results_dir, store, True, tasks, delta, set(names))
    states = {u: (delta.get(u) or {}).get("verdict", "not run") for u in names}
    verdict = "PASS" if rc == 0 and all(v == "PASS" for v in states.values()) else "FAIL"
    lines = [f"  units only (F169), rc={rc} ({time.time() - t0:.0f}s), log: {log}"]
    lines += [f"  {u}: {v}" for u, v in states.items()]
    lines.append(f"  next: `gate {tid} --continue` runs the units not yet passed and commits when all pass")
    return GateResult(tid, verdict, rc, lines, log, None, time.time() - t0)


TOO_LARGE = re.compile(r"(\S+) \((\d+) KB\) exceeds")  # scripts/check_file_size.py, the repo's large-file hook


def task_files_map(repo: Path, task: dict) -> dict:
    """{file: sha256} of the task's own files (its paths), without the records the framework itself writes
    (stamps.SKIP: state.json, results/, dashboards, overseer.yaml, ...). F211: FIX.8's paths held its ledger, so the
    refusal's own state.json write made commit-only see "the task's files changed" and refuse for good."""
    files = sorted(f for f in set(_files_under(repo, list(task.get("paths") or []))) if not stamps._SKIP.match(f))
    return {f: freeze.sha256_file(repo / f) for f in files}


def task_files_key(repo: Path, task: dict) -> str:
    """The content of the task's own files (its paths and frozen files): what a pending commit must still match."""
    text = "\n".join(f"{f}\0{h}" for f, h in task_files_map(repo, task).items())
    return hashlib.sha256((text + json.dumps(task.get("frozen"), sort_keys=True)).encode()).hexdigest()[:16]


def _hook_fixed(pend: dict, now: dict) -> tuple[list, list]:
    """F154: the files changed since the gate passed, split into those the refusing hook named (a hook fix: commit
    without a re-run) and the rest (need the gate)."""
    was = {f: h for f, h in (pend.get("files") or {}).items() if not stamps._SKIP.match(f)}  # F211: records never count
    changed = sorted(f for f in set(was) | set(now) if was.get(f) != now.get(f))
    named = [f for f in changed if any(f in ln for ln in pend.get("hook") or [])]
    return named, [f for f in changed if f not in named]


def _refused(spec: Spec, ledger: Ledger, task: dict, res: GateResult, err: Exception, pend: dict | None = None):
    """F149: a passed gate whose commit git refused stays COMMIT_PENDING with the hook's output; its outputs stay."""
    tid = task["id"]
    hook = [ln for ln in str(err).splitlines() if ln.strip() and "Skipped" not in ln and "Passed" not in ln][-25:]
    big = [m.group(1) for ln in hook for m in [TOO_LARGE.search(ln)] if m]
    why = [
        "  the gate passed; git refused its commit. Nothing re-runs: fix what the hook names, then "
        f"`python -m models.demos.common.bringup gate {tid} --commit-only --spec <spec>` (or `resume`)"
    ]
    if big:
        why.append(f"  too large for the repo: {', '.join(big)} (a framework fix keeps it out of the commit)")
    why += [f"    {ln}" for ln in hook]
    pend = dict(pend or {"lines": res.lines, "task_files": task_files_key(spec.repo, task)})
    pend.setdefault("files", task_files_map(spec.repo, task))  # F154: per file, so a hook fix can be told apart
    pend.update(t=time.strftime("%Y-%m-%dT%H:%M:%S"), hook=hook, too_large=big)
    ledger.update(
        tid,
        status="COMMIT_PENDING",
        commit_pending=pend,
        reason=why,
        history_add={"t": pend["t"], "status": "COMMIT_PENDING"},
    )
    res.verdict, res.commit, res.lines = "COMMIT_PENDING", None, res.lines + why
    return res


def commit_only(spec: Spec, ledger: Ledger, tid: str) -> GateResult:
    """F149: commit a passed gate whose commit was refused, from the outputs it left; nothing runs. Refused again: it
    stays pending and names the files too large."""
    task, entry = ledger.task(tid), ledger.state().get(tid, {})
    pend = entry.get("commit_pending")
    if entry.get("status") != "COMMIT_PENDING" or not pend:
        return GateResult(tid, "FAIL", lines=[f"  no pending commit (status {entry.get('status', 'TODO')})"])
    res = GateResult(tid, "PASS", 0, list(pend.get("lines") or []))
    fixed = []
    if pend.get("task_files") and pend["task_files"] != task_files_key(spec.repo, task):
        fixed, other = _hook_fixed(pend, task_files_map(spec.repo, task)) if pend.get("files") else ([], ["?"])
        if other:  # F154: only the files the hook named may change after the gate (the fix it asked for)
            res.verdict = "COMMIT_PENDING"
            res.lines.append(
                f"  the task's files changed since its gate passed: not committed (`rerun --from {tid}`)"
                + (f": {', '.join(other)} (not named by the hook)" if other != ["?"] else "")
            )
            return res
        res.lines.append(f"  changed after the gate, as the hook asked (F154): {', '.join(fixed)}")
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    ledger.update(tid, status="PASS", commit_pending=None, reason=[], history_add={"t": now, "status": "COMMITTED"})
    try:
        with ledger.locked():
            res.commit = git_commit(
                spec,
                stage_paths(spec, ledger, task),
                commit_subject(spec, task),
                "Gate: PASS (committed without a re-run, F149)\n" + "\n".join(res.lines),
            )
    except RuntimeError as e:
        return _refused(spec, ledger, task, res, e, pend)
    res.lines.append(f"  committed without a re-run (the gate passed at {pend.get('t')})")
    return res
