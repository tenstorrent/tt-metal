# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F259: the device side of a batch (core/batch.py): one pytest process runs several device-test runs, each as its own
in-process pytest session with its own environment, log and results, on one mesh and one loaded model.

``device_model(s, mesh, layers, lm_head)`` is what the framework's tests call instead of hooks.device_model: outside a
batch it is hooks.device_model; inside, the first run loads the model (with the LM head and the longest target.seq any
run of the plan needs: core/batch.hints) and the next runs reuse it. Between runs the model's active state is freed
(``model.state``, DeepSeek-V4.1; a model whose states are not on the model frees them with the run's locals) and the
process environment restored; the program cache stays warm (a timed run is warm-only anyway: F196 refuses misses).
F269 (owner 2026-11-18): a run that needs a model the shared one cannot serve (other layers, LM head, seq, config,
load-time switches) is NEVER loaded again in this process: a released model's device memory does not come back (its
states, a runtime's buffers, per-mesh load-time buffers: DeepSeek-V4.1's pinned outputs, L1 reservations), and the
reload ran out of device DRAM (2026-11-17, ab/P.72/old/test_full_prefill). ``shared_model`` raises
``NeedsFreshProcess``; ``run_plan`` ends the process at that run, the run and every later one ``deferred`` (not run,
not failed, no result), and the driver (core/batch.drive) runs them in a fresh process. A run that itself needs a
second model configuration fails (no process can hold both).

``run_plan(mesh, plan)`` runs the plan: per run, ``pytest.main`` over its test files with its env, stdout / stderr (the
C++ log too: file descriptors 1 and 2) into its log, and the outer mesh handed to its mesh_device fixture
(``SharedMesh``). The test modules are imported again for each run (they read their environment at import: the
ladder's rungs, a spec's target). Results are written after every run, ``running`` before it (a hang names its run).

FIX.29 (OWNER 2026-11-13, "LOAD THE MODEL ONLY ONCE"): ``shared_model`` is the one cache of the process; the
framework's ``device_model`` and a model's own loader (DeepSeek-V4.1: tt/runners/adapter.load_model, which both its
bring-up hook and its serving runtime call, so the contract tests' adapter path takes the batch's model) go through
it. Every real load is counted (metric ``model_loads``: this process's loads; the gate sums them as
gate_model_loads). The model's switch values at its load (hooks.batch_settings, read from the environment) are kept;
a later run whose values differ in a switch the model reads at load (all but hooks.batch_call_settings) gets a fresh
load with a warning, never a model of another configuration. After each run every model Settings snapshot is
re-read (a run's monkeypatch is undone by then) and the run's metrics and result files are noted in the results
(``metrics``, ``files``: what a gate keeps per unit).

F282 (owner 2026-11-19, "load only the layers a measurement executes"): ``run_layers(s, layers, last_chunk)`` is the
layers a measurement runs, as the model's optional hook ``run_layers(spec, layers, last_chunk)`` says (DeepSeek-V4.1:
under DECODER_SKIP=decode, a measurement whose chunks are never a request's last runs layers 0-20 only); the test loads
and loops those. A model already held with more layers (a superset within the full list) serves such a subset run;
BRINGUP_RUN_LAYERS=all turns the subset off (an A/B of the full load against the subset).
F281: inside the resident runner (testing/resident.py) a run's timing is noted (``TIMING``: the model ready, its first
chunk: the first ``begin_chunk`` / ``embed`` call), the model build goes through the device-weight registry
(testing/weights.py), and ``run_plan(.., keep=True)`` keeps the model past the plan.
"""

from __future__ import annotations

import copy
import gc
import json
import os
import sys
import time
from pathlib import Path

import pytest

from models.demos.common.bringup.core import batch as B

_MODELS: dict = {}  # "model": the shared model, "key", "layers", "lm_head", "seq", "by" (the run that loaded it)
_RUN = {"name": None}
_LOADS = {"n": 0}  # FIX.29: real model loads in this process (metric model_loads)
_LOADING = {"on": False}  # a load is running (a nested loader call is part of it)
_FRESH: dict = {}  # F269: "why" (the run needs another model than this process holds), "by" (the run that loaded it)
_SUBSETS: dict = {}  # F282: (a measurement's layers) -> (the full layer list it was taken from): run_layers
TIMING: dict = {}  # F281: this run's times (time.time()): "model_ready", "first_chunk"; "loaded": the run loaded it
RUN_LAYERS_ENV = "BRINGUP_RUN_LAYERS"  # F282: "all" = every layer of the test's list, never the model's subset
RESIDENT_ENV = "BRINGUP_RESIDENT_DIR"  # F281: set in the resident runner's process (its state directory)
REQUEST: dict = {}  # F281: the resident runner's request being served: "received" (time.time())


class NeedsFreshProcess(RuntimeError):
    """F269: this run needs another model than the one this process holds; never loaded again in place."""


def active() -> bool:
    return bool(os.environ.get(B.KEY_ENV))


def _target_seq(s) -> int:
    return int(s.data["target"]["seq"])


def release() -> None:
    """Drop the shared model (a run that needs another one, or the end of the batch)."""
    m = _MODELS.pop("model", None)
    _MODELS.clear()
    if m is not None:
        free_state(m)
        del m
        gc.collect()


def free_state(model) -> None:
    """Free the model's active state (its KV caches) between runs."""
    st = getattr(model, "state", None)
    if st is not None:
        from models.demos.common.bringup.testing.positions import _free

        try:
            model.sync()
        except Exception:
            pass
        _free(st)
        if getattr(model, "state", None) is st:
            model.state = None
    gc.collect()


def inject(m):
    """Diagnostic (FIX.29, never in a gate): BRINGUP_INJECT=rowshift2 makes the batch's model wrong on purpose, to
    show a test that takes the shared model still fails when what it checks breaks: the run's 2nd ``embed(ids,
    start)`` call (and only it) gets its token ids rolled by one position (an off-by-one in the sequence position on
    one chunk; once, so not repeatable: a determinism check sees it too). Undone after the run (``uninject``)."""
    kind = os.environ.get(B.INJECT_ENV)
    if not kind or "embed" in vars(m):
        return m
    assert kind == "rowshift2", f"{B.INJECT_ENV}={kind!r}: only rowshift2"
    import torch

    orig, calls = m.embed, {"n": 0}

    def embed(ids, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            ids = torch.roll(ids.reshape(-1), 1).reshape(ids.shape)
        return orig(ids, *a, **kw)

    m.embed = embed
    print(f"BATCH (F259) DIAGNOSTIC: {B.INJECT_ENV}={kind}: the run's 2nd embed call's ids rolled by one", flush=True)
    return m


def uninject(m) -> None:
    if m is not None and "embed" in vars(m):
        del m.embed


def _hook(s, name: str, *args):
    """A model hook that may be absent (older models): its value or None."""
    try:
        fn = getattr(s.hooks(), name, None)
    except Exception:
        return None
    return fn(*args) if callable(fn) else None


def run_layers(s, layers, last_chunk: bool | None = None) -> list:
    """F282: the layers a measurement loads and runs: the model hook ``run_layers(spec, layers, last_chunk)`` (a
    subset, or None for all), ``last_chunk``: whether any chunk the measurement runs is a request's last (None: as
    BRINGUP_CHUNK_LAST says, unset = last). BRINGUP_RUN_LAYERS=all: every layer of ``layers``."""
    layers = list(layers)
    if last_chunk is None:
        last_chunk = os.environ.get("BRINGUP_CHUNK_LAST") != "0"
    if os.environ.get(RUN_LAYERS_ENV) == "all":
        return layers
    sub = _hook(s, "run_layers", s, layers, bool(last_chunk))
    if not sub:
        return layers
    keep = {int(i) for i in sub}
    out = [i for i in layers if i in keep]
    if out != layers:
        _SUBSETS[tuple(out)] = tuple(layers)
        print(
            f"BATCH (F282): this measurement runs layers {out[0]}-{out[-1]} ({len(out)} of {len(layers)}): only those "
            f"are loaded ({RUN_LAYERS_ENV}=all loads every one)",
            flush=True,
        )
    return out


def _serves_layers(held: list, want: list) -> bool:
    """F282: the held model's layers serve ``want``: the same list, or ``want`` is a run_layers subset and the held
    model's layers lie between it and the full list it came from. F285: in the resident runner a performance
    measurement never runs on a full model an earlier plan loaded (an accuracy load): it gets its own."""
    if held == want:
        return True
    if os.environ.get(RESIDENT_ENV) and _MODELS.get("plan") is not _RUN.get("plan"):
        return False
    full = _SUBSETS.get(tuple(want))
    return full is not None and set(want) <= set(held) <= set(full)


def _first_call_timed(m):
    """F281 (resident runner only): the first ``begin_chunk`` / ``embed`` call of the run notes TIMING["first_chunk"]
    (the measurement's first chunk starts). An instance attribute, removed after the run (``_untimed``)."""
    if not os.environ.get(RESIDENT_ENV) or m is None or not hasattr(m, "__dict__"):
        return m
    for name in ("begin_chunk", "embed"):
        if getattr(vars(m).get(name), "_bringup_timed", False):
            continue
        orig = getattr(m, name, None)
        if not callable(orig):
            continue

        def timed(*a, _orig=orig, **kw):
            TIMING.setdefault("first_chunk", time.time())
            return _orig(*a, **kw)

        timed._bringup_timed = True
        setattr(m, name, timed)
    return m


def _untimed(m) -> None:
    for name in ("begin_chunk", "embed"):
        if m is not None and hasattr(m, "__dict__") and getattr(vars(m).get(name), "_bringup_timed", False):
            delattr(m, name)


def _load_settings(s) -> dict:
    """The model's switch values a load would read now (hooks.batch_settings: from the environment), minus the ones
    it reads per call (hooks.batch_call_settings)."""
    got = dict(_hook(s, "batch_settings") or {})
    for k in _hook(s, "batch_call_settings") or ():
        got.pop(k, None)
    return got


def _count_load(by: str | None) -> int:
    from models.demos.common.bringup.core import metrics

    _LOADS["n"] += 1
    # value: this process's loads; total: every load recorded in this results file, across processes (a gate takes
    # its difference over a unit, however many processes the unit runs); pid / ns: a new record even within a second
    prev = metrics.load(metrics.record_task(), metrics.results_dir()).get(B.LOADS_METRIC) or {}
    total = int(prev.get("total") or 0) + 1
    metrics.record(B.LOADS_METRIC, _LOADS["n"], by=by or "", total=total, pid=os.getpid(), ns=time.time_ns())
    return _LOADS["n"]


def shared_model(mesh, layers, *, seq: int, lm_head: bool, load, config=None, spec=None):
    """FIX.29: the model of ``layers`` loaded once per process and model configuration. ``load(seq, lm_head)`` builds
    it (the model's one load path); ``seq``: the positions the caller needs; ``config``: anything else the load
    depends on (e.g. the chunk, the weight cache). Outside a batch: ``load`` as asked (counted). Inside: the batch's
    model when it serves the call (same plan key, mesh, layers, config, switch values read at load, seq >= the
    call's, its LM head if asked), else a fresh load of max(seq, the plan's hint) with the plan's LM head hint. A
    call made while a load runs (a loader that calls another loader) is that load: no cache, no count."""
    from models.demos.common.bringup.core import metrics

    want = list(layers)
    if _LOADING["on"]:
        return load(seq, lm_head)
    if not active():
        _LOADING["on"] = True
        try:
            m = load(seq, lm_head)
        finally:
            _LOADING["on"] = False
        _count_load(_RUN["name"])
        return m
    key = os.environ[B.KEY_ENV]
    sw = _load_settings(spec) if spec is not None else {}
    m = _MODELS.get("model")
    if (
        m is not None
        and _MODELS["key"] == key
        and _MODELS["mesh"] is mesh
        and _serves_layers(_MODELS["layers"], want)  # F282: or a held superset of a run_layers subset
        and _MODELS["config"] == config
        and _MODELS["settings"] == sw
        and (_MODELS["lm_head"] or not lm_head)
        and _MODELS["seq"] >= seq
    ):
        free_state(m)
        if spec is not None:  # F283: the held model put to this run's per-run switch values (hooks.batch_call_apply)
            _hook(spec, "batch_call_apply", m)
        metrics.record("model_shared", 1)
        print(f"BATCH (F259): model reused (loaded by {_MODELS['by']} in {_MODELS['load_s']:.0f} s)", flush=True)
        TIMING.setdefault("model_ready", time.time())
        return _first_call_timed(inject(m))
    if m is not None:  # F269: never loaded again in place (the released model's device memory does not come back)
        diff = sorted(k for k in set(sw) | set(_MODELS["settings"]) if sw.get(k) != _MODELS["settings"].get(k))
        why = (
            f"{_RUN['name']} needs another model (layers / LM head / seq / config / switches {diff} differ from "
            f"{_MODELS['by']}'s)"
        )
        _FRESH.update(why=why, by=_MODELS["by"], same_plan=_MODELS.get("plan") == _RUN.get("plan"))
        print(f"BATCH (F269): {why}: not loaded again in this process; the run goes to a fresh process", flush=True)
        raise NeedsFreshProcess(why)
    seq = max(int(seq), int(os.environ.get(B.MAX_SEQ_ENV) or 0))
    head = lm_head or os.environ.get(B.LM_HEAD_ENV) == "1"
    t0 = time.time()
    _LOADING["on"] = True
    from models.demos.common.bringup.testing import weights

    if weights.active():  # F281: the resident runner's build takes the device weights an earlier build loaded
        weights.begin_build()
    built = None
    try:
        m = load(seq, head)
    finally:
        _LOADING["on"] = False
        if weights.active():
            built = weights.end_build()
            print(f"BATCH (F281): {weights.describe(built)}", flush=True)
    load_s = getattr(m, "load_seconds", time.time() - t0)
    _MODELS.update(
        model=m, key=key, mesh=mesh, layers=want, config=config, settings=sw, lm_head=head, seq=seq,
        by=_RUN["name"], load_s=load_s, weights=built, plan=_RUN.get("plan"),
    )  # fmt: skip
    n = _count_load(_RUN["name"])
    metrics.record("model_shared", 0)
    print(
        f"BATCH (F259): model loaded for the batch ({len(want)} layers, lm_head {head}, seq {seq}; load {n} of this "
        f"process, {load_s:.0f} s)",
        flush=True,
    )
    TIMING["loaded"] = True
    TIMING.setdefault("model_ready", time.time())
    return _first_call_timed(inject(m))


def device_model(s, mesh, layers, lm_head: bool = True):
    """The device model of ``layers`` (hooks.device_model), shared by the runs of a batch (``shared_model``). A model
    whose own loader goes through ``shared_model`` (hooks.SHARED_LOADER = True: DeepSeek-V4.1's adapter.load_model)
    is asked directly, so its two load paths (bring-up hook, serving runtime) share one cache entry."""
    try:
        own = bool(getattr(s.hooks(), "SHARED_LOADER", False))
    except Exception:
        own = False
    if own:
        return s.hooks().device_model(mesh, s, list(layers), lm_head=lm_head)

    def load(seq, head):
        sb = copy.copy(s)  # the load's spec: the sequence and the LM head the batch needs
        sb.data = copy.deepcopy(s.data)
        sb.data["target"]["seq"] = max(int(seq), _target_seq(s))
        return s.hooks().device_model(mesh, sb, list(layers), lm_head=head)

    return shared_model(mesh, layers, seq=_target_seq(s), lm_head=lm_head, load=load, config=None, spec=s)


class SharedMesh:
    """pytest plugin of a run inside a batch: its mesh_device fixture is the batch's mesh (never opened or closed
    again). A test asking for another mesh shape fails: it cannot share this process."""

    def __init__(self, mesh, shape):
        self.mesh, self.shape = mesh, tuple(shape)

    @pytest.hookimpl(tryfirst=True)
    def pytest_fixture_setup(self, fixturedef, request):
        if fixturedef.argname != "mesh_device":
            return None
        for name in fixturedef.argnames:
            request.getfixturevalue(name)
        want = getattr(request, "param", None)
        if want is not None and tuple(want) != self.shape:
            raise RuntimeError(f"batch (F259): the test wants mesh {want}, the batch holds {self.shape}")
        fixturedef.cached_result = (self.mesh, fixturedef.cache_key(request), None)
        return self.mesh


def _drop_test_modules() -> None:
    """Test modules read their environment at import (rungs, engine variables): each run imports them again."""
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None) or ""
        if Path(f).name.startswith("test_") and "/tests/" in f and Path(f).name != "test_batch.py":
            sys.modules.pop(name, None)


def _redirected(log: str | None, fn):
    """fn() with file descriptors 1 and 2 (Python and the C++ log) appended to ``log``; as is without a log."""
    if not log:
        return fn()
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    streams = sys.stdout, sys.stderr  # (under an outer capture these are not file descriptors 1 and 2)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as f:
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        sys.stdout = open(1, "w", buffering=1, closefd=False, errors="replace")
        sys.stderr = open(2, "w", buffering=1, closefd=False, errors="replace")
        try:
            return fn()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = streams
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def _refresh_settings() -> None:
    """FIX.29: every model Settings snapshot re-read from the environment as it is now (a run's monkeypatch was
    undone; a snapshot taken under it would carry its values into the next run)."""
    from models.demos.common.bringup.core.model_settings import Settings

    for name, mod in list(sys.modules.items()):
        if not name.endswith(".settings") or mod is None:
            continue
        for v in list(vars(mod).values()):
            if isinstance(v, Settings):
                v.refresh()


def _results_view() -> tuple[str, Path, dict, dict]:
    """(record task, results dir, its metrics, {result file: mtime}) as the current environment points them."""
    from models.demos.common.bringup.core import metrics

    task, root = metrics.record_task(), metrics.results_dir()
    files = {f.name: f.stat().st_mtime_ns for f in root.glob("*.json")} if root.is_dir() else {}
    return task, root, metrics.load(task, root), files


def load_plan() -> tuple[dict, Path | None]:
    if os.environ.get(B.PLAN_ENV):
        p = Path(os.environ[B.PLAN_ENV])
        return json.loads(p.read_text()), p
    from models.demos.common.bringup.testing.harness import spec

    return B.plan_from_cmd(os.environ[B.CMD_ENV], spec() if os.environ.get("BRINGUP_SPEC") else None), None


DEFERRED: list[str] = []  # F269: the runs the last run_plan left for a fresh process (in plan order)


def _record_timing(t0: float) -> None:
    """F281: the resident runner's per-run timing as the run's metrics (seconds): resident_model_s (run start ->
    model ready), resident_first_chunk_s (run start -> its first chunk), resident_request_first_chunk_s (the
    request's arrival -> this run's first chunk), resident_model_loaded (1: this run loaded the model)."""
    if not os.environ.get(RESIDENT_ENV) or not TIMING:
        return
    from models.demos.common.bringup.core import metrics

    try:
        if TIMING.get("model_ready"):
            metrics.record("resident_model_s", round(TIMING["model_ready"] - t0, 2))
        if TIMING.get("first_chunk"):
            metrics.record("resident_first_chunk_s", round(TIMING["first_chunk"] - t0, 2))
            if REQUEST.get("received"):
                metrics.record("resident_request_first_chunk_s", round(TIMING["first_chunk"] - REQUEST["received"], 2))
        metrics.record("resident_model_loaded", int(bool(TIMING.get("loaded"))))
    except Exception as e:  # noqa: BLE001 -- a record never fails a run
        print(f"BATCH (F281): timing not recorded: {e}", flush=True)


def run_plan(
    mesh, shape, plan: dict, plan_path: Path | None = None, main=None, keep: bool = False, after_run=None
) -> list[str]:
    """Run every entry of ``plan`` in this process; the names of the runs that failed. ``main``: pytest.main (tests).
    F269: a run that needs another model than this process holds, and every run after it (but a failed group's: those
    are skipped), get no result here: they are listed in the results' ``deferred`` (and ``DEFERRED``), for a fresh
    process (core/batch.drive). F281 (the resident runner): ``keep`` keeps the model past the plan; ``after_run(name,
    rec)`` sees each run's record before it is saved and stops the plan when it returns True (a device hang: rc 2)."""
    main = main or pytest.main
    out = {"running": None, "runs": {}}
    res = B.results_path(plan_path) if plan_path else None

    def save():
        if res is not None:
            tmp = res.with_suffix(".tmp")
            tmp.write_text(json.dumps(out, indent=1) + "\n")
            os.replace(tmp, res)

    failed, dead_groups, stop = [], set(), False
    entries = plan["entries"]
    _RUN["plan"] = object()  # F283: this plan (a model it loads carries it)
    _FRESH.clear()
    DEFERRED.clear()
    for k, v in (plan.get("hints") or {}).items():  # the load's LM head / sequence, unless the caller set them
        os.environ.setdefault(k, str(v))
    for k, e in enumerate(entries, 1):
        name = e["name"]
        group = e.get("group") or name
        if DEFERRED and group not in dead_groups:  # F269: the rest of the plan runs in a fresh process
            DEFERRED.append(name)
            continue
        if stop or group in dead_groups:
            out["runs"][name] = {"rc": None, "skipped": True}
            save()
            continue
        out["running"] = name
        save()
        _RUN["name"] = name
        saved = dict(os.environ)
        os.environ.update({k2: str(v) for k2, v in e["env"].items()})
        os.environ[B.KEY_ENV] = e.get("key") or "default"
        args = list(e["args"]) + ([] if e.get("run_all") else ["-x"])
        print(f"BATCH (F259) run {k}/{len(entries)} {name}: {' '.join(f'{a}={b}' for a, b in e['env'].items())} {args}")
        _drop_test_modules()
        _refresh_settings()
        task, root, m0, f0 = _results_view()
        loads0 = _LOADS["n"]
        t0 = time.time()
        TIMING.clear()
        try:
            rc = int(_redirected(e.get("log"), lambda: main(args, plugins=[SharedMesh(mesh, shape)])))
        finally:
            m = _MODELS.get("model")
            if m is not None:
                free_state(m)
                _untimed(m)
                uninject(m)
            gc.collect()  # (the run's own device tensors: an adapter's KV caches, its runtime's slot states)
            _record_timing(t0)
            _, _, m1, f1 = _results_view()
            os.environ.clear()
            os.environ.update(saved)
            _refresh_settings()
        rc = 0 if rc == 0 else 1  # a test failure; 2 stays the safe runner's HANG
        fresh = dict(_FRESH)
        _FRESH.clear()
        # F283: a second load for this plan (a gate's units) is refused; a model an earlier plan left (the resident
        # runner's previous request) is not this gate's load: that run goes to a new process as before
        one = plan.get("one_load") or os.environ.get(B.ONE_LOAD_ENV) == "1"
        # F287 (P.84 A/B, 2026-11-19): a model another plan loaded (the resident runner's previous request, whose run
        # may carry this run's name: both A/B sides' runs are "test_skip_positions.0") is never this run's own second
        # configuration: the run goes to a new process
        other = bool(fresh) and (fresh["by"] != name or not fresh.get("same_plan", True))
        if fresh and other and fresh.get("same_plan") and one:
            note = (
                f"\n# F283: {fresh['why']}: refused (owner rule 2026-11-19, spec tests.one_load: one model load per "
                "gate); the run fails instead of going to a fresh process\n"
            )
            if e.get("log"):
                with open(e["log"], "a") as f:
                    f.write(note)
            print(note.strip(), flush=True)
            rc, fresh = 1, {}
        if fresh and other:  # F269: this run and the rest of the plan in a fresh process
            release()
            DEFERRED.append(name)
            out["running"] = None
            out["deferred"] = DEFERRED  # (the same list: the later runs join it)
            save()
            note = f"\n# F269: {fresh['why']}: not run here; it runs in a fresh process (core/batch.drive)\n"
            if e.get("log"):
                with open(e["log"], "a") as f:
                    f.write(note)
            print(note.strip(), flush=True)
            continue
        if fresh:  # the run itself asked for two model configurations: no process holds both
            rc = 1
            print(f"BATCH (F269) run {name}: needs two model configurations itself: {fresh['why']}", flush=True)
        out["runs"][name] = {
            "rc": rc,
            "s": round(time.time() - t0, 1),
            "loads": _LOADS["n"] - loads0,
            # FIX.29: what this run recorded (a gate keeps it per unit) and the result files it wrote
            "metrics": {k: v for k, v in m1.items() if m0.get(k) != v},
            "files": sorted(k for k, t in f1.items() if f0.get(k) != t),
        }
        if TIMING:  # F281: seconds from the run's start to its model and to its first chunk
            tm = {k: round(TIMING[k] - t0, 2) for k in ("model_ready", "first_chunk") if TIMING.get(k)}
            out["runs"][name]["timing"] = dict(tm, loaded=bool(TIMING.get("loaded")), t0=t0)
        if after_run is not None and after_run(name, out["runs"][name]):  # F281: a hang; nothing more runs here
            out["runs"][name]["rc"] = rc = 2
            stop = True
        out["running"] = None
        save()
        print(f"BATCH (F259) run {name}: rc {rc} in {time.time() - t0:.0f} s", flush=True)
        if rc:
            failed.append(name)
            dead_groups.add(group)
            stop = bool(plan.get("stop_on_fail"))
    if DEFERRED:
        out["deferred"] = DEFERRED
        save()
        print(f"BATCH (F269): runs for a fresh process: {', '.join(DEFERRED)}", flush=True)
    if not keep:
        release()
    return failed
