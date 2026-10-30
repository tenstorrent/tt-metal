# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F150: what a gate result was measured on (its stamp) and the results kept per stamp, so a gate never runs twice
on the same work.

A gate is one or more units: a fix job's segments of one source task each (F79), else the whole command. A unit's
stamp is the hash of its command as run, its source task's frozen block, the gate environment's BRINGUP_* / TT_*
variables, the content of every file a measurement can depend on (code_key: the tree as it is, committed or not, by
content, without the run's own records and the files that cannot change a measurement) and the goldens (goldens_key).
Every unit a gate runs is kept under runs/<run>/stamps/<unit>.json by stamp: verdict, metrics, side outputs. A unit
whose stamp has a PASS (rc 0, its metrics within the gate's limits) is not run again: its metrics and outputs are
restored ("reused: same code"). A FAIL or HANG is never reused; a changed stamp always runs.

F162: the code key is the tree a unit can depend on, not the whole tree: without the records and the orchestration
(SKIP), without another model's bring-up files and frozen tests (its bringup/ non-Python files, its tests/bringup/),
and with the spec counted without its owner waivers. A waiver changes only the units that read it (waiver_key):
the component test of the waived step, every swap test (a waiver reaches the block of the step, F158), a test
directory run whole; ladder, contract and profile units never read a waiver (only testing/component.py does).
"""

from __future__ import annotations

import fnmatch
import functools
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

# F120: files that cannot change what a run measures (the orchestration, its docs and selftests, any markdown)
NO_MEASURE = tuple(
    f"models/demos/common/bringup/{p}"
    for p in (
        "orchestrator.py",
        "watch.py",
        "testing/settings_lint.py",  # F234: checks an agent's diff; no test or measurement reads it
        "testing/ops_lint.py",
        "overseer.py",
        "agents/*",
        "briefs/*",
        "skill/*",
        "selftest/*",
        "dev/*",
        "dashboard/*",
        "knowledge/*",
    )
) + ("*.md",)
RECORDS = (  # F105; F234: a seeded run's ledger (bringup/ledgers/<box>-<mode>/tasks.yaml) is a record too
    "*state.json",
    "*/results/*",
    "*BREADCRUMBS.md",
    "*supervision.md",
    "*bringup/tasks.yaml",
    "*bringup/ledgers/*/tasks.yaml",
)
SKIP = (
    RECORDS
    + NO_MEASURE
    + (
        "*bringup/overseer.yaml",  # notes and the fix queue
        "*bringup/approvals.yaml",  # F162: who approved what; no test reads it
        "*bringup/dashboard/*",
        "*bringup/ledgers/*/overseer.yaml",  # F189: the same files in a seeded run's ledger (F183)
        "*bringup/ledgers/*/approvals.yaml",
        "*bringup/ledgers/*/dashboard/*",
        "models/demos/common/bringup/core/stamps.py",
        "models/demos/common/bringup/core/verify.py",
        "models/demos/common/bringup/__main__.py",
        "models/demos/common/bringup/timing.py",
        "*__pycache__*",
    )
)
_SKIP = re.compile("|".join(fnmatch.translate(g) for g in SKIP))
ENV_PREFIXES = ("BRINGUP_", "TT_")
ENV_IGNORED = {  # where a gate files its results, not what it measures
    "BRINGUP_TASK",
    "BRINGUP_RECORD_TASK",
    "BRINGUP_RESULTS_DIR",
    "BRINGUP_SPEC",
    "BRINGUP_DASHBOARD",
    "BRINGUP_AGENT_CMD",
    "TT_DEVICE_TIMING_LOG",
    "BRINGUP_CPU_THREADS",  # F170 (audit item 12): how many CPU threads a job takes, not what it measures
}
ENV_IGNORED_PREFIXES = ("TT_METAL_JIT_SERVER",)  # F170: where kernels compile (a JIT server or locally), not the result
KEEP = 8  # stamps kept per unit


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class GitError(RuntimeError):
    """F170: git failed while a stamp was taken; a gate must not go on with an empty file map (one constant key would
    reuse a PASS that never ran on this code)."""


def _git(repo: Path, *a, inp: str | None = None) -> str:
    r = subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, input=inp)
    if r.returncode != 0:  # F170: never an empty answer for a failed command (audit item 16)
        raise GitError(f"git {' '.join(a[:3])} failed in {repo} (rc {r.returncode}): {r.stderr.strip()[:300]}")
    return r.stdout


_MODEL_FILES = re.compile(r"models/demos/([^/]+)/(bringup|tests/bringup)/")


def _other_model(path: str, own: str | None) -> bool:
    """F162: another model's bring-up records and frozen tests (never imported by this model's units); its Python
    under bringup/ (hooks a prior-based bring-up may import) still counts."""
    m = _MODEL_FILES.match(path)
    if not m or own is None or m.group(1) == own:
        return False
    return m.group(2) == "tests/bringup" or not path.endswith(".py")


def spec_text(spec) -> str:
    """F162: the spec as its units read it, without the owner waivers (waiver_key stamps those per unit)."""
    import copy

    import yaml

    d = copy.deepcopy(spec.data)
    (d.get("tests") or {}).pop("waivers", None)
    (d.get("tests") or {}).pop("unit_waivers", None)  # F225: stamped per unit, like the step waivers
    if d.get("tests") == {}:  # a spec whose tests section held only waivers reads as one without it
        d.pop("tests")
    return yaml.safe_dump(d, sort_keys=True)


def unit_waivers(spec, cmd: str) -> list[dict]:
    """F225: the owner's waivers of named failures of a unit's test (spec ``tests.unit_waivers``: {<test file stem>:
    {match: [regex, ...], why: "..."}}), for a frozen test with no limit of its own to waive (a contract test's
    check). [] when none names this unit's test."""
    w = (spec.get("tests.unit_waivers") or {}) if spec is not None else {}
    return [dict(v, test=k) for k, v in w.items() if re.search(rf"(^|[/\s]){re.escape(str(k))}(\.py|::|\s|$)", cmd)]


def waiver_key(spec, cmd: str) -> str:
    """F162: the owner waivers a unit's tests can read (testing/component.py only): a component test reads its step's
    waiver, a swap test or a whole test directory every waiver; other units none. '' when it reads none."""
    w = (spec.get("tests.waivers") or {}) if spec is not None else {}
    uw = unit_waivers(spec, cmd)  # F225: a unit waiver re-runs only its own unit
    if not w and not uw:
        return ""
    if "test_swap_" in cmd or re.search(r"tests/bringup/?(\s|$|['\"])", cmd):
        keep = w
    elif "test_c_" in cmd:
        keep = {k: v for k, v in w.items() if str(k).split(".")[-1] in cmd}
    else:
        keep = {}
    if uw:
        keep = dict(keep, _unit_waivers=uw)
    return sha16(json.dumps(keep, sort_keys=True, default=str)) if keep else ""


def _file_map(repo: Path, keep) -> dict:
    """{path: content hash} of the tree as it is (committed, changed and untracked files alike, a commit of the same
    content keeps the hash) for the paths ``keep(path)`` accepts."""
    files = {}
    for entry in _git(repo, "ls-tree", "-r", "-z", "HEAD").split("\0"):
        if "\t" in entry:
            meta, path = entry.split("\t", 1)
            if keep(path):
                files[path] = meta.split()[2]
    changed = _git(repo, "diff", "HEAD", "--name-only", "-z").split("\0")
    changed += _git(repo, "ls-files", "-o", "--exclude-standard", "-z").split("\0")
    changed = sorted({p for p in changed if p and keep(p)})
    plain = [p for p in changed if (repo / p).is_file()]
    shas = _git(repo, "hash-object", "--stdin-paths", inp="\n".join(plain) + "\n").split() if plain else []
    files.update(zip(plain, shas))
    for p in set(changed) - set(plain):  # a deleted file, or a submodule at another commit
        files[p] = _git(repo / p, "rev-parse", "HEAD").strip() if (repo / p / ".git").exists() else "-"
    return files


def code_key(repo: Path, spec=None) -> str:
    """The content of the tree a gate runs on: committed, changed and untracked files alike (a commit of the same
    content keeps the key), without SKIP; with the spec (F162) also without another model's bring-up files and with
    the spec file counted without its waivers."""
    files = _file_map(repo, lambda p: not _SKIP.match(p))
    if spec is not None:
        own = Path(spec.model_dir).name
        files = {p: h for p, h in files.items() if not _other_model(p, own)}
        try:
            sp = str(Path(spec.path).resolve().relative_to(Path(repo).resolve()))
        except (TypeError, ValueError):  # a spec outside the repo: hashed whole
            sp = None
        if sp in files:
            files[sp] = sha16(spec_text(spec))
        elif spec.path is not None:
            files["<spec>"] = sha16(spec_text(spec))
    return sha16("\n".join(f"{p}\0{h}" for p, h in sorted(files.items())))


# F169: the code under test: what an agent's change can alter in a unit's verdict. The overseer's edits (the spec,
# its waivers and timeouts, approvals, the framework, records, docs, knowledge, the skill) are not in it.
UNDER_TEST_ROOTS = ("ttnn/", "tt_metal/")
_NOT_CODE = re.compile(r".*(__pycache__/|\.md$)")


def semantic_py_hash(text: str) -> str:
    """F173: a Python file's hash without what cannot change behaviour: comments, formatting, docstrings, and a
    model Setting's explanation (`why`, the 3rd argument; core/model_settings.Setting). A file that does not parse is
    hashed as text (never a false reuse)."""
    import ast

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "text:" + sha16(text)

    class _Strip(ast.NodeTransformer):
        def _body(self, node):
            self.generic_visit(node)
            b = getattr(node, "body", None)
            if (
                b
                and isinstance(b[0], ast.Expr)
                and isinstance(getattr(b[0], "value", None), ast.Constant)
                and isinstance(b[0].value.value, str)
            ):
                node.body = b[1:] or [ast.Pass()]
            return node

        visit_Module = visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _body

        def visit_Call(self, node):
            self.generic_visit(node)
            f = node.func
            if (isinstance(f, ast.Name) and f.id == "Setting") or (
                isinstance(f, ast.Attribute) and f.attr == "Setting"
            ):
                if len(node.args) >= 3:
                    node.args[2] = ast.Constant("")
                node.keywords = [k for k in node.keywords if k.arg != "why"]
            return node

    return "ast:" + sha16(ast.dump(_Strip().visit(tree), annotate_fields=False, include_attributes=False))


def under_test_key(repo: Path, spec, paths=(), with_bytes: bool = False):
    """F169: the content of the model's own code and tests (models/demos/<model>/ without its bring-up records: of
    bringup/ only the Python, e.g. hooks.py), the forks and ttnn / tt_metal (ttnn/, tt_metal/), the paths the task
    (and a fix job's gate tasks) may change, and the goldens. A passed unit stays passed for its task while this key is
    unchanged (gate --continue). with_bytes (F175): also the byte-for-byte key of the same files, which records made
    before F173 carry (an identical tree is the same code: they stay valid)."""
    try:
        own = str(Path(spec.model_dir).resolve().relative_to(Path(repo).resolve())).rstrip("/") + "/"
    except ValueError:  # a model outside the repo
        own = f"models/demos/{Path(spec.model_dir).name}/"
    extra = list(paths) + list(spec.get("paths.own") or [])  # the task's paths; the spec's own code dirs
    roots = UNDER_TEST_ROOTS + tuple(str(p).rstrip("/") for p in extra if p)

    def keep(p: str) -> bool:
        if _NOT_CODE.match(p) or _SKIP.match(p):
            return False
        if p.startswith(own + "bringup/"):
            return p.endswith(".py")
        return p.startswith(own) or any(p == r or p.startswith(r if r.endswith("/") else r + "/") for r in roots)

    files = _file_map(repo, keep)
    gk = goldens_key(spec)
    raw = sha16("\n".join(f"{p}\0{h}" for p, h in sorted(files.items())) + "\n" + gk)
    # F173: the model's own Python by behaviour, not bytes (a docstring / Setting-text edit re-ran 18 units)
    sem = {
        p: semantic_py_hash((Path(repo) / p).read_text(errors="replace"))
        if p.endswith(".py") and p.startswith(own) and (Path(repo) / p).is_file()
        else h
        for p, h in files.items()
    }
    key = sha16("\n".join(f"{p}\0{h}" for p, h in sorted(sem.items())) + "\n" + gk)
    return (key, raw) if with_bytes else key


def goldens_key(spec) -> str:
    """The goldens' own records (manifest with its content_hash, drift and floor files) under the spec's golden root."""
    try:
        root = spec.golden_root
    except Exception:  # a spec without golden paths
        return ""
    if not root.is_dir():
        return ""
    parts = [
        f"{f.relative_to(root)}\0{hashlib.sha256(f.read_bytes()).hexdigest()}" for f in sorted(root.glob("*/*.json"))
    ]
    return sha16("\n".join(parts)) if parts else ""


@functools.lru_cache(maxsize=1)
def settings_prefixes() -> tuple:
    """F220: every model's switch prefix (``Settings("DS41_", ...)`` in models/demos/<model>/tt/settings.py): a model
    switch set in the environment changes what a unit measures, so it is in the stamp like BRINGUP_ / TT_."""
    root = Path(__file__).resolve().parents[5]
    found = set()
    for f in root.glob("models/demos/*/tt/settings.py"):
        found.update(re.findall(r"""Settings\(\s*["']([A-Z0-9_]+)["']""", f.read_text(errors="ignore")))
    return tuple(sorted(found))


def env_key(env: dict) -> str:
    prefixes = ENV_PREFIXES + settings_prefixes()
    keep = {
        k: v
        for k, v in env.items()
        if k.startswith(prefixes) and k not in ENV_IGNORED and not k.startswith(ENV_IGNORED_PREFIXES)
    }
    return sha16(json.dumps(keep, sort_keys=True))


def stamp(cmd: str, frozen, env: dict, code: str, goldens: str, waivers: str = "") -> dict:
    return {
        "cmd": sha16(cmd),
        "frozen": sha16(json.dumps(frozen, sort_keys=True)),
        "env": env_key(env),
        "code": code,
        "goldens": goldens,
        "waivers": waivers,  # F162: only the owner waivers this unit's tests read
    }


def stamp_id(st: dict) -> str:
    return sha16(json.dumps(st, sort_keys=True))


def output_key(name: str, record_tid: str) -> str:
    """A gate output's name without its record task (FIX.5_profile.json -> _profile.json), so another gate restores it."""
    return name[len(record_tid) :] if name.startswith(f"{record_tid}_") else name


class Store:
    """runs/<run>/stamps/<unit>.json: {stamp id: record}; side outputs in runs/<run>/stamps/<unit>/<stamp id>/."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir) / "stamps"

    def _index(self, unit: str) -> Path:
        return self.dir / f"{unit}.json"

    def records(self, unit: str) -> dict:
        p = self._index(unit)
        try:
            return json.loads(p.read_text()) if p.exists() else {}
        except ValueError:
            return {}

    def get(self, unit: str, sid: str) -> dict | None:
        """The unit's record for this stamp; F177: else the same test's record filed by another task (a unit is
        `<task>#<test>`: the identical command on the identical code, e.g. four picks sharing one frozen test set)."""
        rec = self.records(unit).get(sid)
        if rec is not None or "#" not in unit:
            return rec
        label = unit.split("#", 1)[1]
        for idx in sorted(self.dir.glob(f"*#{label}.json")) if self.dir.is_dir() else []:
            other = idx.stem
            if other != unit and sid in self.records(other):
                return dict(self.records(other)[sid], _from=other)
        return None

    def put(self, unit: str, sid: str, rec: dict, outputs: list[Path], record_tid: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        keep = self.dir / unit / sid
        shutil.rmtree(keep, ignore_errors=True)
        names = []
        for f in outputs:
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, keep / output_key(f.name, record_tid))
            names.append(output_key(f.name, record_tid))
        recs = self.records(unit)
        recs[sid] = dict(rec, outputs=names, t=rec.get("t") or time.strftime("%Y-%m-%dT%H:%M:%S"))
        for old in sorted(recs, key=lambda s: recs[s].get("t", ""))[:-KEEP]:
            recs.pop(old)
            shutil.rmtree(self.dir / unit / old, ignore_errors=True)
        tmp = self._index(unit).with_suffix(".tmp")
        tmp.write_text(json.dumps(recs, indent=1, sort_keys=True) + "\n")
        tmp.replace(self._index(unit))

    def restore(self, unit: str, sid: str, rec: dict, results_dir: Path, record_tid: str) -> None:
        for name in rec.get("outputs") or []:
            src = self.dir / rec.get("_from", unit) / sid / name
            if src.exists():
                shutil.copy2(src, results_dir / (f"{record_tid}{name}" if name.startswith("_") else name))


class Delta:
    """F169: a task's gate as units: runs/<run>/delta/<task>.json {unit: {verdict, under_test, cmd, env, sid, metrics,
    s, t}}. A unit with a PASS on the current code under test is kept (not run) when its task's gate runs again; a
    FAIL, a STOPPED (infrastructure: a timeout, a box error) or a never-run unit runs. A gate never restarts."""

    def __init__(self, run_dir: Path, tid: str):
        self.path = Path(run_dir) / "delta" / f"{tid}.json"

    def all(self) -> dict:
        try:
            return json.loads(self.path.read_text()) if self.path.exists() else {}
        except ValueError:
            return {}

    def get(self, unit: str) -> dict | None:
        return self.all().get(unit)

    def put(self, unit: str, rec: dict) -> None:
        data = self.all()
        data[unit] = dict(rec, t=rec.get("t") or time.strftime("%Y-%m-%dT%H:%M:%S"))
        self._write(data)

    def drop(self, units) -> list[str]:
        data = self.all()
        gone = [u for u in units if data.pop(u, None) is not None]
        self._write(data)
        return gone

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        tmp.replace(self.path)
