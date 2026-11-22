# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F259 (owner 2026-11-06: "this is taking insanely too much time"): several device-test runs in ONE pytest process,
the model loaded once. CPU side: which safe-runner commands may share a process, in what order, the plan file the
device side reads (testing/batch.py, tests/test_batch.py) and the results it writes back.

A Galaxy 8x4 DeepSeek-V4.1 process spends ~4 min uploading weights and ~1.5-2.5 min on its first chunk for a
measurement of seconds; a perf pick's A/B ran ~5-10 such processes a side. Shared here: the mesh (opened once), the
loaded device model (testing/batch.device_model), the program cache, a model's process-level tables (the resident
engram table). Not shared: everything a run records (its own env, results dir, log, metric suffix) and its state
(KV caches: freed between runs).

A run may join a batch when it is one ``scripts/run_safe_pytest.sh --no-precompile`` call (flags: --no-precompile,
--run-all) over tests that take the model from the batch (``shareable``: the framework's SHARED_TESTS, or a test file
that declares ``BATCH_SHARED = True``: it loads the model only through testing/batch.device_model or a model loader
that goes through testing/batch.shared_model, e.g. DeepSeek-V4.1's serving adapter, tt/runners/adapter.load_model)
with no device profiler variable (TT_METAL_*: read at process start). The runs of one batch must agree on every
variable but the per-run ones (ITEM_VARS): a model's switches are read at its load (DeepSeek-V4.1 HOST_GLUE freezes
them), so different switch values are different processes (off / on); a switch a test sets itself (monkeypatch) and
the model reads at load makes testing/batch.shared_model load again in the process (a warning, counted in
model_loads); one the model reads per call (hooks.batch_call_settings) does not.
FIX.29 (OWNER 2026-11-13, "LOAD THE MODEL ONLY ONCE"): the contract tests and the model's own tests share it too, and
a gate runs all its units that may share a process in ONE batch process (``gate_groups``, core/gate.run_units): one
model load per gate and model configuration (``gate_model_loads``, counted from each process's ``model_loads``).
F269 (owner 2026-11-18): a process holds one model. A run that needs another one (the device side finds out at its
load: testing/batch.shared_model) ends its process there, it and the runs after it left to a fresh process
(``deferred``): ``drive`` runs a plan's processes until every run has run, each run's result recorded once.
F281 (owner 2026-11-19, iteration time): while a resident runner serves the spec (resident.py), ``drive`` submits the
plan to it instead of starting a process (a plan with the model determinism check as a fresh-process request).
F283 (owner 2026-11-19): a gate loads the model at most once (spec tests.one_load): ``one_load_errors`` refuses a gate
whose shared-model runs need more than one process group (the cold determinism run alone aside), and a run that would
need another model in its process fails instead of going to a fresh one (``ONE_LOAD_ENV``). A switch the model reads
per run (hooks.batch_call_settings, e.g. DeepSeek-V4.1's DECODER_SKIP but replay128) is not its configuration
(``call_vars``).
"""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path

PLAN_ENV = "BRINGUP_BATCH"  # the plan file (json) of a batch process
CMD_ENV = "BRINGUP_BATCH_CMD"  # or: the original command, planned by the batch process itself (a gate unit)
KEY_ENV = "BRINGUP_BATCH_KEY"  # set per run by the batch process: the model key the plan gave it
LM_HEAD_ENV = "BRINGUP_BATCH_LM_HEAD"  # load the shared model with its LM head (a ladder run in the batch needs it)
MAX_SEQ_ENV = "BRINGUP_BATCH_MAX_SEQ"  # load it with target.seq at least this (the furthest BRINGUP_POS + chunk)
BATCH_TEST = "models/demos/common/bringup/tests/test_batch.py"
KEEP_GOING_ENV = "BRINGUP_BATCH_KEEP_GOING"  # 1: a hand-run batch (BRINGUP_BATCH_CMD) runs every run past a failure
DRIVEN_ENV = "BRINGUP_BATCH_DRIVEN"  # F269: 1: a driver (drive) runs the runs this process leaves to a fresh one
ONE_LOAD_ENV = "BRINGUP_ONE_LOAD"  # F283: 1 (or a plan's "one_load"): a second model fails the run, no fresh process
INJECT_ENV = "BRINGUP_INJECT"  # diagnostic (FIX.29): a mistake injected into the batch's model (testing/batch.inject)
TESTS_DIR = "models/demos/common/bringup/tests/"
SHARED_TESTS = (
    "test_ladder.py",
    "test_profile.py",
    "test_positions.py",
    "test_smoke.py",
    "test_determinism.py",
    # FIX.29: the serving contract through the engine's adapter (its runtime takes the batch's model: the model's
    # loader goes through testing/batch.shared_model), and its host-only ack-timing check
    "test_contract.py",
    "test_contract_acks.py",
)
# FIX.29: a model test file that takes the batch's model declares it with this line (module level)
SHARED_MARK = re.compile(r"^BATCH_SHARED\s*=\s*True\b", re.M)
# F285 (OWNER 2026-11-19: "accuracy loads the whole model; performance does not"): a shared-model test is an accuracy
# check ("full": every layer, in the accept gate's one fresh process with the determinism check) or a performance
# measurement ("executed": the layers it executes, testing/batch.run_layers, in the resident runner). A model test file
# declares ``BATCH_LAYERS = "executed"``; the framework's own: EXECUTED_TESTS; anything else is "full".
LAYERS_MARK = re.compile(r"^BATCH_LAYERS\s*=\s*[\"'](full|executed)[\"']", re.M)
EXECUTED_TESTS = ("test_profile.py", "test_positions.py")
LOADS_METRIC = "model_loads"  # each process: the model loads it made (testing/batch.shared_model)
GATE_LOADS_METRIC = "gate_model_loads"  # a gate run: the model loads of all its processes (core/gate.run_units)
_REPO = Path(__file__).resolve().parents[5]
# FIX.25: the model-level determinism check (testing/model_determinism.py) every model-level gate runs (core/gate.py)
DETERMINISM_TEST = TESTS_DIR + "test_determinism.py"
DETERMINISM_CMD = f"scripts/run_safe_pytest.sh --run-all --no-precompile {DETERMINISM_TEST}"
FLAGS = {"--no-precompile", "--run-all"}
# per-run variables: what a run measures and where it records it; anything else is the model's configuration
ITEM_VARS = {
    "BRINGUP_TASK",
    "BRINGUP_RECORD_TASK",
    "BRINGUP_RESULTS_DIR",
    "BRINGUP_METRIC_SUFFIX",
    "BRINGUP_POS",
    "BRINGUP_PROFILE_WALL",
    "BRINGUP_WALL_CLEAN",
    "BRINGUP_PROFILE_HOST",
    "BRINGUP_QUICK",
    "BRINGUP_RUNG",
    "BRINGUP_CHUNK_LAST",
    "BRINGUP_FULL_PREFILL",
    "BRINGUP_PROFILE_OPS",
    "BRINGUP_PROFILE_LAYERS",
    "BRINGUP_AB",
    "BRINGUP_POS_TOKENS",  # F279: the text a position run reads (F276); the model is the same for every text
}
# F279 (OWNER 2026-11-19, "load once per gate"): a shareable model test names its own per-run variables (the
# positions / text / chunk count it measures; the model does not read them) with a module-level line
# ``BATCH_ITEM_VARS = ("SKIP_POS", ...)``; runs differing only in those share one process. Collected by ``parse``.
ITEM_MARK = re.compile(r"^BATCH_ITEM_VARS\s*=\s*\(([^)]*)\)", re.M)
# F288 (P.84: a process loaded for 770048 then restarted for 1024000): a position test names the variables holding
# the chunk starts it measures (comma-separated) with ``BATCH_POS_VARS = ("SKIP_POS",)``; a plan's model is loaded
# for the furthest of them (``hints``), never past deploy.seq (where such a test stops itself)
POS_MARK = re.compile(r"^BATCH_POS_VARS\s*=\s*\(([^)]*)\)", re.M)
# PB.T (2026-11-21, DeepSeek-V4.1: the serving runtime's warm-up, runtime.compile, needs slots of >= 112k positions and
# the accuracy load's tables reached 64512): a model test file names the positions its own load-time tables must reach
# with ``BATCH_MAX_SEQ = <int>``; a plan's model is loaded for at least that (``hints``), never past deploy.seq
SEQ_MARK = re.compile(r"^BATCH_MAX_SEQ\s*=\s*(\d+)\b", re.M)
# F297 (2026-11-21, DeepSeek-V4.1 PB.9C: test_no_rebuild FAILED with 153 / 16 builds in a fresh process, then PASSED in 18 s
# in the held resident process on the same code: the earlier run's programs were still cached): a test that counts
# program builds after its own warm-up declares ``BATCH_COLD = True``; it runs in a fresh process (a cold program cache)
# and first in its batch after the determinism check, so no other run's traffic warms programs it must find cold
COLD_MARK = re.compile(r"^BATCH_COLD\s*=\s*True\b", re.M)
DECLARED_ITEM_VARS: set = set()
SAFE_RUN = re.compile(r"^((?:\w+=\S*\s+)*)scripts/run_safe_pytest\.sh\s+(.+)$", re.S)
ASSIGN = re.compile(r"^(\w+)=(\S*)$")


def shareable(test: str, repo: Path | None = None) -> bool:
    """A test file (or ::nodeid) that may run in a batch: a framework SHARED_TESTS file, or a test file under models/
    that declares ``BATCH_SHARED = True`` (FIX.29)."""
    path = test.split("::")[0]
    if path.startswith(TESTS_DIR) and path.rsplit("/", 1)[-1] in SHARED_TESTS:
        return True
    if not (path.startswith("models/") and path.endswith(".py") and "/tests/" in path):
        return False
    f = (repo or _REPO) / path
    try:
        return bool(SHARED_MARK.search(f.read_text(errors="replace")))
    except OSError:
        return False


def layer_class(test: str, repo: Path | None = None) -> str:
    """F285: "executed" (a performance measurement) or "full" (an accuracy check) of a test file (or ::nodeid)."""
    path = test.split("::")[0]
    if path.startswith(TESTS_DIR):
        return "executed" if path.rsplit("/", 1)[-1] in EXECUTED_TESTS else "full"
    try:
        m = LAYERS_MARK.search(((repo or _REPO) / path).read_text(errors="replace"))
    except OSError:
        return "full"
    return m.group(1) if m else "full"


def unit_class(runs: list[dict] | None) -> str | None:
    """F285: a unit's layer class: "executed" when every run is a performance measurement, else "full"; None for a
    unit that does not take the shared model."""
    if not runs:
        return None
    return "executed" if all(layer_class(t) == "executed" for r in runs for t in r["args"]) else "full"


def declared_pos_vars(test: str, repo: Path | None = None) -> set:
    """F288: the variables a model test file names as its chunk starts (``BATCH_POS_VARS = (...)``)."""
    try:
        m = POS_MARK.search(((repo or _REPO) / test.split("::")[0]).read_text(errors="replace"))
    except OSError:
        return set()
    return set(re.findall(r"[\"']([A-Z0-9_]+)[\"']", m.group(1))) if m else set()


def declared_max_seq(test: str, repo: Path | None = None) -> int:
    """PB.T: the positions a model test file needs its load to cover (``BATCH_MAX_SEQ = <int>``; 0 = none declared)."""
    try:
        m = SEQ_MARK.search(((repo or _REPO) / test.split("::")[0]).read_text(errors="replace"))
    except OSError:
        return 0
    return int(m.group(1)) if m else 0


def declared_item_vars(test: str, repo: Path | None = None) -> set:
    """F279: the per-run variables a model test file declares (``BATCH_ITEM_VARS = (...)``)."""
    try:
        m = ITEM_MARK.search(((repo or _REPO) / test.split("::")[0]).read_text(errors="replace"))
    except OSError:
        return set()
    return set(re.findall(r"[\"']([A-Z0-9_]+)[\"']", m.group(1))) if m else set()


def parse(seg: str) -> dict | None:
    """One `&&` segment as a batch run {env, args, run_all}, or None when it cannot share a process."""
    m = SAFE_RUN.match(seg.strip())
    if not m:
        return None
    env = dict(ASSIGN.match(w).groups() for w in m.group(1).split())
    words = shlex.split(m.group(2))
    flags = [w for w in words if w.startswith("-")]
    tests = [w for w in words if not w.startswith("-")]
    if not tests or "--no-precompile" not in flags or not set(flags) <= FLAGS:
        return None
    if not all(shareable(t) for t in tests):
        return None
    for t in tests:  # F279
        DECLARED_ITEM_VARS.update(declared_item_vars(t))
    if any(k.startswith("TT_METAL_") for k in env) or env.get("BRINGUP_PROFILER") == "device":
        return None
    return {"env": env, "args": tests, "run_all": "--run-all" in flags}


def parse_all(cmd: str) -> list[dict] | None:
    """Every segment of ``cmd`` as a batch run, or None if any cannot share a process."""
    runs = [parse(s) for s in cmd.split("&&") if s.strip()]
    return runs if runs and all(runs) else None


def cold(run: dict, repo: Path | None = None) -> bool:
    """F297: a run of a test file that declares ``BATCH_COLD = True`` (it needs a cold program cache)."""
    for t in run["args"]:
        try:
            if COLD_MARK.search(((repo or _REPO) / t.split("::")[0]).read_text(errors="replace")):
                return True
        except OSError:
            pass
    return False


def phase(run: dict) -> int:
    """The order inside a batch: the determinism check first (FIX.25: its first sequence is the cold one, the first
    after the load), then ladders (their first chunk and the load time as close to a process of their own as it
    gets), then clean walls (F114: before any real-time-profiled run), then the other runs, the profiled runs last."""
    test = run["args"][0].split("::")[0]
    env = run["env"]
    if test.endswith("test_determinism.py"):
        return -1
    if test.endswith("test_ladder.py"):
        return 0
    if test.endswith("test_profile.py") and env.get("BRINGUP_PROFILE_WALL") == "1":
        return 1
    if test.endswith("test_profile.py"):
        return 3
    return 2


def hints(runs: list[dict], spec) -> dict:
    """What the shared model is loaded for, so no run needs another load: the LM head when a ladder or the smoke runs,
    target.seq up to the furthest position a profile (BRINGUP_POS) or test_positions times. FIX.29: at least the
    longest sequence the spec's target and ladders reach, whatever loads first (the serving adapter's runtime asks
    for its rung's max seq only; a model's load-time tables of more positions serve fewer: DeepSeek-V4.1's RoPE)."""
    out = {}
    tests = [r["args"][0].split("::")[0] for r in runs]
    if any(t.endswith(("test_ladder.py", "test_smoke.py")) for t in tests):
        out[LM_HEAD_ENV] = "1"
    pos = [int(r["env"]["BRINGUP_POS"]) for r in runs if r["env"].get("BRINGUP_POS")]
    deploy = int(spec.get("deploy.seq") or 0) or None
    starts = [  # F288: the chunk starts a position test declares (BATCH_POS_VARS), as far as the deployment reaches
        int(p)
        for r in runs
        for v in sorted(set().union(*(declared_pos_vars(t) for t in r["args"])))
        for p in str(r["env"].get(v) or "").split(",")
        if p.strip().isdigit()
    ]
    need = pos or starts or any(t.endswith("test_positions.py") for t in tests)
    chunk = int(spec.get("target.chunk")) if need else 0  # (only then: FIX.25's determinism run needs no hint)
    seqs = [p + chunk for p in pos] + [p + chunk for p in starts if deploy is None or p + chunk <= deploy]
    # PB.T: what a test declares its load must cover (BATCH_MAX_SEQ), at most the deployment's length
    seqs += [n for r in runs for t in r["args"] for n in [declared_max_seq(t)] if n and (deploy is None or n <= deploy)]
    if len(runs) > 1:  # (a process of one run loads what it needs)
        seqs += [int(spec.get("target.seq", 0) or 0)]
        seqs += [int(r.get("seq", 0) or 0) for key in ("ladder", "scaled.ladder") for r in (spec.get(key) or [])]
    if any(t.endswith("test_positions.py") for t in tests):
        from models.demos.common.bringup.testing.positions import positions

        seqs.append(max(positions(spec)) + chunk)
    if seqs:
        out[MAX_SEQ_ENV] = str(max(seqs))
    return out


def command(plan: Path) -> str:
    """The batch process: one safe-runner call over tests/test_batch.py reading ``plan``."""
    return f"{PLAN_ENV}={shlex.quote(str(plan))} scripts/run_safe_pytest.sh --no-precompile --run-all {BATCH_TEST}"


def unit_command(cmd: str, spec=None) -> str | None:
    """F259: a gate unit of two or more runs that may share a process (the clean wall and the profiled run, F114) as
    one batch process, its runs in the unit's order (&&: the first failure stops), metrics into the unit's results as
    before; else None (the unit runs as written)."""
    runs = parse_all(cmd)
    if not runs or len(runs) < 2:
        return None
    names = {k for r in runs for k in r["env"]}
    if len({model_key(r["env"], names) for r in runs}) != 1:
        return None
    extra = " ".join(f"{k}={v}" for k, v in hints(runs, spec).items()) if spec is not None else ""
    pre = f"{extra} " if extra else ""
    return (
        f"{pre}{CMD_ENV}={shlex.quote(cmd)} scripts/run_safe_pytest.sh --no-precompile --run-all {BATCH_TEST}"
    ).strip()


LAUNCH_FLAGS = ("--precompile", "--no-precompile")  # FIX.31: where kernels compile, not what a run measures


def launch_form(cmd: str) -> tuple[str, list[str]]:
    """FIX.31: a command whose runs ask for the safe runner's precompile pass (no --no-precompile flag) in the form a
    shared process runs it, and a note per such run. A batch process runs its tests with kernels compiled on demand:
    a precompile pass inside it would run every test body under NO_DISPATCH first, and the batch's model cache
    (testing/batch.shared_model) would keep the model that pass built from shape-only tensors. Model-level runs find
    their kernels in the cache (F151: full-model steps already skip the pass); the first chunk compiles what is not."""
    out, notes = [], []
    for seg in cmd.split("&&"):
        m = SAFE_RUN.match(seg.strip())
        if not m:
            out.append(seg.strip())
            continue
        words = m.group(2).split()
        if "--no-precompile" not in words:
            words = ["--no-precompile"] + [w for w in words if w != "--precompile"]
            notes.append(
                f"{' '.join(w for w in words if not w.startswith('-'))}: asked for the precompile pass; in the shared "
                "process its kernels compile on demand (FIX.31, core/batch.launch_form)"
            )
        out.append(f"{m.group(1)}scripts/run_safe_pytest.sh {' '.join(words)}".strip())
    return " && ".join(out), notes


def why_not(cmd: str) -> str | None:
    """FIX.31: why a unit's command cannot run in a shared process (the gate log says it), or None when it can."""
    cmd = launch_form(cmd)[0]
    for seg in (x.strip() for x in cmd.split("&&") if x.strip()):
        m = SAFE_RUN.match(seg)
        if not m:
            return f"`{seg[:80]}` is not one scripts/run_safe_pytest.sh call"
        words = shlex.split(m.group(2))
        flags = [w for w in words if w.startswith("-")]
        tests = [w for w in words if not w.startswith("-")]
        env = dict(ASSIGN.match(w).groups() for w in m.group(1).split())
        if not tests:
            return f"`{seg[:80]}` names no test"
        bad = sorted(set(flags) - FLAGS)
        if bad:
            return f"flags {bad} (a shared process takes only {sorted(FLAGS)})"
        no = [t for t in tests if not shareable(t)]
        if no:
            return f"{no} cannot share the model (not a framework shared test, no BATCH_SHARED = True)"
        if any(k.startswith("TT_METAL_") for k in env) or env.get("BRINGUP_PROFILER") == "device":
            return "a device-profiler variable (read at process start)"
    return None


_TEST_VALUES: dict = {}


def test_value(name: str):
    """FIX.31: (found, value) of a model switch variable ``name`` (e.g. DS41_ENGRAM_TABLE) as a test process reads it
    unset: its Setting's ``test`` value when it has one (F222), else its default (core/model_settings.Setting)."""
    import importlib

    if name in _TEST_VALUES:
        return _TEST_VALUES[name]
    _TEST_VALUES[name] = (False, None)
    for f in sorted(_REPO.glob("models/demos/*/tt/settings.py")):
        mod = f"models.demos.{f.parent.parent.name}.tt.settings"
        try:
            m = importlib.import_module(mod)
        except Exception:  # noqa: BLE001 -- a model whose settings do not import: its switches stay as written
            continue
        for v in vars(m).values():
            prefix = getattr(v, "prefix", None)
            table = getattr(v, "table", None)
            if isinstance(prefix, str) and isinstance(table, dict) and name.startswith(prefix):
                st = table.get(name[len(prefix) :])
                if st is not None:
                    _TEST_VALUES[name] = (True, st.test if st.test is not None else st.default)
    return _TEST_VALUES[name]


def normalized(env: dict) -> dict:
    """FIX.31 (owner 2026-11-13): a model switch set to the value a test process reads when it is unset is the same
    model as unset (P.51's units ran as 5 processes over one such variable). Compared as text with the value's type
    (bools as 1/true/yes). DS41_ENGRAM_TABLE=resident is NOT such a switch: its test value is mmap (F222), and a
    timed gate sets resident on purpose (X.3, owner 2026-11-06)."""
    out = {}
    for k, v in env.items():
        found, val = test_value(k) if k.isupper() and "_" in k and not k.startswith(("BRINGUP_", "TT_")) else (0, 0)
        if found:
            same = (str(v).lower() in ("1", "true", "yes")) == val if isinstance(val, bool) else str(v) == str(val)
            if same:
                continue
        out[k] = v
    return out


_CALL_HOOKS: list = []  # F283: [(settings prefix, hooks.batch_call_settings)] of the models in the repo


def _call_hooks() -> list:
    import importlib

    if _CALL_HOOKS:
        return _CALL_HOOKS[0]
    out = []
    for f in sorted(_REPO.glob("models/demos/*/bringup/hooks.py")):
        if "def batch_call_settings" not in f.read_text(errors="replace"):
            continue
        name = f.parent.parent.name
        try:
            hooks = importlib.import_module(f"models.demos.{name}.bringup.hooks")
            st = importlib.import_module(f"models.demos.{name}.tt.settings")
        except Exception:  # noqa: BLE001 -- a model whose hooks do not import: its switches stay configuration
            continue
        prefix = next((v.prefix for v in vars(st).values() if isinstance(getattr(v, "prefix", None), str)), None)
        if prefix:
            out.append((prefix, hooks.batch_call_settings))
    _CALL_HOOKS.append(out)
    return out


def call_vars(env: dict) -> set:
    """F283: the variables (set in ``env`` or not) that are switches a model reads per run at ``env``'s values, not at
    its load
    (hooks.batch_call_settings(env): the batch's model is put to the run's value; DeepSeek-V4.1: DECODER_SKIP off /
    exact / decode, FIX.28 use_decoder_skip): they never split a process."""
    names = set()
    for prefix, fn in _call_hooks():
        try:
            got = fn(env)
        except TypeError:
            got = fn()
        names |= {prefix + n for n in got}
    return names


def model_key(env: dict, names: set) -> str:
    """The model configuration of a run: its effective value of every non-per-run variable any run sets (FIX.31: a
    switch at the value a test reads unset counts as unset). Only which runs share a process: the device side loads
    again whenever a load-time switch value differs (testing/batch.shared_model). F283: a switch the model reads per
    run is not configuration (``call_vars``)."""
    item = ITEM_VARS | DECLARED_ITEM_VARS | call_vars(env)  # F279, F283
    env = normalized({k: v for k, v in env.items() if k in names - item})
    return json.dumps({k: env.get(k) for k in sorted(names - item)}, sort_keys=True)


def gate_groups(units: list[tuple[str, str]], env: dict) -> list[list[tuple[str, str]]]:
    """FIX.29: the gate units to run (unit, command), in gate order, as process groups: units whose every
    run may share a process at one model configuration (F283: also when other units stand between them: the group
    runs at its first unit's turn, the units between keep their own processes; one model load per configuration) (``model_key`` over the gate's environment and the runs'
    prefixes) form one group (one batch process, one model load); any other unit is a group of its own (its command
    as written, or core/batch.unit_command). A single unit stays its own group.
    FIX.31: a unit whose runs ask for the precompile pass joins too, in its launch form (``launch_form``: the command a
    group of two or more carries)."""
    out: list[list[tuple[str, str]]] = []
    keys: list[str | None] = []
    forms = {u: launch_form(c)[0] for u, c in units}
    parsed = {u: parse_all(forms[u]) for u, _ in units}
    names = {k for runs in parsed.values() if runs for r in runs for k in r["env"]}
    for u, c in units:
        runs = parsed[u]
        key = None
        if runs:
            ks = {model_key({**env, **_with_config(r, units, parsed)["env"]}, names) for r in runs}
            # F285: accuracy units ("full") and performance units ("executed") are never one process
            key = f"{unit_class(runs)}:{ks.pop()}" if len(ks) == 1 else None
        if key is not None and key in keys:  # F283: the gate's group of that model, wherever it started
            out[keys.index(key)].append((u, forms[u]))
        else:
            out.append([(u, forms[u] if key is not None else c)])
            keys.append(key)
    return [g if len(g) > 1 else [(g[0][0], dict(units)[g[0][0]])] for g in out]


def _config_of(units: list[tuple[str, str]], parsed: dict | None = None) -> dict:
    """F285: the model configuration variables (not per-run) of the gate's first accuracy unit that is not the
    determinism check: the configuration the determinism check runs at (it checks the model the gate tests)."""
    for u, c in units:
        runs = parsed[u] if parsed is not None else parse_all(launch_form(c)[0])
        if runs and not _determinism_only(runs) and unit_class(runs) == "full":
            # F296: only the unit's OWN tests' declared per-run variables: another test's declaration (test_no_rebuild's
            # BRINGUP_CHUNK) is that test's business; for this unit it is the model's geometry (an 8192 gate's
            # determinism check otherwise loaded the 5120 model and F283 refused the 8192 oracles: PB.MB, PB.9X)
            own = set().union(*(declared_item_vars(t) for r in runs for t in r["args"]))
            item = ITEM_VARS | own | call_vars(runs[0]["env"])
            return {k: v for k, v in runs[0]["env"].items() if k not in item}
    return {}


def _with_config(run: dict, units, parsed=None) -> dict:
    """The run, a determinism run with no switches of its own taking the gate's accuracy configuration (F285)."""
    if phase(run) != -1 or run["env"].keys() - ITEM_VARS:
        return run
    return dict(run, env={**_config_of(units, parsed), **run["env"]})


def gate_entries(group: list[tuple[str, str]], env: dict, unit_env: dict, logs: dict) -> list[dict]:
    """FIX.29: the batch entries of one process group of gate units: each unit's runs (its `&&` segments) in a group
    of their own (a failed run skips the rest of its unit), the unit's own log and per-run variables (``unit_env``:
    {unit: {BRINGUP_TASK: ...}}), the determinism runs first (FIX.25: the first sequence after the load is the cold
    one), else the gate's order; the plan stops at the first failure (the gate stops there too). ``logs``: {unit:
    log path}."""
    parsed = [(u, [_with_config(r, group) for r in parse_all(c)]) for u, c in group]  # F285
    names = {k for _, runs in parsed for r in runs for k in r["env"]}
    entries = []
    for u, runs in parsed:
        for j, r in enumerate(runs):
            entries.append(
                dict(
                    r,
                    env={**(unit_env.get(u) or {}), **r["env"]},
                    name=f"{u}.{j}",
                    group=u,
                    unit=u,
                    log=str(logs[u]),
                    key=model_key({**env, **r["env"]}, names),
                )
            )
    # (stable: the gate's order otherwise; F297: a cold run right after the determinism check)
    entries.sort(key=lambda e: -2 if phase(e) == -1 else (-1 if cold(e) else 0))
    return entries


def _determinism_only(runs: list[dict]) -> bool:
    return bool(runs) and all(phase(r) == -1 for r in runs)


def group_class(group: list[tuple[str, str]]) -> str | None:
    """F285: the layer class of a process group (gate_groups never mixes them), None when not a shared-model one."""
    runs = [parse_all(launch_form(c)[0]) for _, c in group]
    if not all(runs):
        return None
    return unit_class([r for rs in runs for r in rs])


def one_load_errors(units: list[tuple[str, str]], env: dict) -> list[str]:
    """F283 / F285 (OWNER 2026-11-19: "a task may load the model at most once per gate"; "accuracy loads the whole
    model; performance does not"): the gate's units (unit, command) as gate_groups puts them in processes. Allowed:
    one group of accuracy units ("full": the fresh full-model process with the determinism check) and one group of
    performance units ("executed": the resident runner's model of the executed layers, no load when it holds it).
    More: why, naming the groups' units and the variables they differ in."""
    groups = gate_groups(units, env)
    loads, by = [], {}
    for g in groups:
        cls = group_class(g)
        if cls is None:
            continue  # not a shared-model process (a component test, a CPU check): no model of the gate's
        runs = [parse_all(launch_form(c)[0]) for _, c in g]
        loads.append((g, [r for rs in runs for r in rs]))
        by.setdefault(cls, []).append(loads[-1])
    over = [cls for cls, gs in by.items() if len(gs) > 1]
    if not over:
        return []
    loads = by[over[0]]
    names = {k for _, flat in loads for r in flat for k in r["env"]}
    keys = [json.loads(model_key({**env, **flat[0]["env"]}, names)) for _, flat in loads]
    diff = sorted(k for k in set().union(*keys) if len({json.dumps(x.get(k)) for x in keys}) > 1)
    parts = [f"[{', '.join(u for u, _ in g)}]" for g, _ in loads]
    why = f"they differ in {', '.join(diff)}" if diff else "their runs differ within a unit"
    return [
        f"the gate needs {len(loads)} {over[0]}-layer model loads, one per process group {' '.join(parts)}: {why}. "
        "Owner rule (2026-11-19, spec tests.one_load): one full-model load for the accuracy units (with the "
        "determinism check) and one executed-layers load for the performance units per gate; make the units of a "
        "class share one model configuration"
    ]


def write_plan(path: Path, entries: list[dict], stop_on_fail: bool = False, one_load: bool = False) -> Path:
    """entries: {name, group, env, args, run_all, log, key}. ``group``: runs that stand or fall together (an A/B item:
    its clean wall and its profiled run); a failed run skips the rest of its group, or of the plan with stop_on_fail."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": 1, "stop_on_fail": stop_on_fail, "entries": entries}
    if one_load:  # F283: a run needing another model fails here (testing/batch.run_plan), no fresh process
        data["one_load"] = True
    path.write_text(json.dumps(data, indent=1) + "\n")
    results_path(path).unlink(missing_ok=True)
    return path


def unit_plan(cmd: str, path: Path, spec=None, one_load: bool = False) -> Path:
    """F269: a gate unit's command (unit_command) as a plan file (plan_from_cmd: its hints too), for ``drive``."""
    p = plan_from_cmd(cmd, spec)
    if one_load:  # F283
        p["one_load"] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(p, indent=1) + "\n")
    results_path(path).unlink(missing_ok=True)
    return path


def _drive_resident(d: Path, plan: Path, env: dict, echo) -> int:
    """F281: the plan on the resident runner: a determinism check makes it a fresh-process request; runs without a log
    of their own get one next to the plan, its text echoed after (as the batch process's output was). The reply's
    rc: 0, 1, 2 (a hang); the plan's results file as a batch process writes it."""
    from models.demos.common.bringup import resident

    data = json.loads(plan.read_text())
    own = []
    for e in data["entries"]:
        if not e.get("log"):
            e["log"] = f"{plan}.{e['name']}.log"
            own.append(e["log"])
    plan.write_text(json.dumps(data, indent=1) + "\n")
    fresh = any(phase(e) == -1 or cold(e) for e in data["entries"])  # F297: a cold run too
    echo(
        f"# F281: the plan runs on the resident runner ({d}){' in a fresh process (determinism / cold, F297)' if fresh else ''}\n"
    )
    r = resident.submit_plan(d, plan, {**env, **(data.get("hints") or {})}, fresh=fresh)
    for log in own:
        if Path(log).exists():
            echo(Path(log).read_text(errors="replace"))
    echo(resident.describe(r) + "\n")
    if r.get("error"):
        echo(f"# F281: {r['error']}\n")
        if not results_path(plan).exists():
            results_path(plan).write_text(json.dumps({"running": None, "runs": {}, "error": r["error"]}) + "\n")
    rc = int(r.get("rc") or 0)
    return 2 if rc == 2 else (1 if rc else 0)


def drive(plan: Path, run, env: dict | None = None, echo=None) -> int:
    """F269: the batch processes of ``plan`` until every run has run: ``run(cmd, k)`` runs the k-th process's command
    (k = 0: the plan's own; its exit code back). A process that ends at a run needing another model than it holds
    lists that run and the rest in its results' ``deferred``; they go to a fresh process (plan <plan>.fresh<k>.json:
    those entries only, the rest of the plan as it was), whose results join the plan's own: callers read
    load_results(plan) as from one process (a run ``running`` in the last process: it hung or died there). No new
    process after a hang or a death (``running`` set) or when a fresh process ran none of its runs. The exit code:
    the last process's, else an earlier failing one's.
    F281: ``env`` (the process's environment, over this one's): when a resident runner serves its spec
    (resident.for_env), the plan goes there instead (``echo(text)``: where the process's output would have gone)."""
    full = dict(os.environ, **(env or {}))
    from models.demos.common.bringup import resident

    say = echo or (lambda t: print(t, end="", flush=True))
    try:
        d = resident.for_env(full, echo=lambda t: say(t + "\n"))
    except resident.Refused as e:  # F286: never a fresh process while the runner is meant to serve
        say(f"# F286: {e}\n")
        results_path(Path(plan)).write_text(json.dumps({"running": None, "runs": {}, "error": str(e)}) + "\n")
        return 1
    if d is not None:
        return _drive_resident(d, Path(plan), full, say)
    rcs = [run(f"{DRIVEN_ENV}=1 {command(plan)}", 0)]
    got = load_results(plan)
    cur, k = plan, 0
    while got.get("running") is None and got.get("deferred"):
        data = json.loads(cur.read_text())
        rest = [e for e in data["entries"] if e["name"] in set(got["deferred"])]
        if not rest or len(rest) == len(data["entries"]):  # (none ran: a new process would do the same)
            break
        k += 1
        cur = write_plan(Path(f"{plan}.fresh{k}.json"), rest, stop_on_fail=bool(data.get("stop_on_fail")))
        if data.get("hints"):
            cur.write_text(json.dumps(dict(json.loads(cur.read_text()), hints=data["hints"]), indent=1) + "\n")
        rcs.append(run(f"{DRIVEN_ENV}=1 {command(cur)}", k))
        sub = load_results(cur)
        got["runs"].update(sub.get("runs") or {})
        got["running"] = sub.get("running")
        got["deferred"] = sub.get("deferred") or []
        got["processes"] = k + 1
        tmp = results_path(plan).with_suffix(".tmp")
        tmp.write_text(json.dumps(got, indent=1) + "\n")
        os.replace(tmp, results_path(plan))
    return rcs[-1] or next((r for r in reversed(rcs) if r), 0)


def results_path(plan: Path) -> Path:
    return Path(str(plan) + ".results.json")


def load_results(plan: Path) -> dict:
    """{"running": name or None, "runs": {name: {rc, s, skipped?}}} as the batch process left it."""
    p = results_path(plan)
    return json.loads(p.read_text()) if p.exists() else {"running": None, "runs": {}}


def plan_from_cmd(cmd: str, spec=None) -> dict:
    """The plan of a gate unit's command (unit_command): its runs in order, output to the unit's log (no file of
    their own), &&: stop at the first failure; with ``spec``, the load hints (a hand-run batch needs no others)."""
    runs = parse_all(cmd)
    if not runs:
        raise ValueError(f"{CMD_ENV}: not a command of runs that may share a process: {cmd!r}")
    names = {k for r in runs for k in r["env"]}
    stop = os.environ.get(KEEP_GOING_ENV) != "1"
    entries = [  # (one group: a failure skips the rest, as &&; KEEP_GOING: each run on its own)
        dict(
            r,
            name=f"run{i}",
            group="unit" if stop else f"run{i}",
            log=None,
            key=model_key(dict(os.environ, **r["env"]), names),
        )  # fmt: skip
        for i, r in enumerate(runs)
    ]
    return {"version": 1, "stop_on_fail": stop, "entries": entries, "hints": hints(runs, spec) if spec else {}}
