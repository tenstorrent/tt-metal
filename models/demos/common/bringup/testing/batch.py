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
A run that needs a model the shared one cannot serve (other layers) gets a fresh load after the old one is released,
with a warning: a model may keep per-mesh load-time buffers (DeepSeek-V4.1's pinned outputs, L1 reservations), so the
planner never puts two model configurations in one process.

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
        and _MODELS["layers"] == want
        and _MODELS["config"] == config
        and _MODELS["settings"] == sw
        and (_MODELS["lm_head"] or not lm_head)
        and _MODELS["seq"] >= seq
    ):
        free_state(m)
        metrics.record("model_shared", 1)
        print(f"BATCH (F259): model reused (loaded by {_MODELS['by']} in {_MODELS['load_s']:.0f} s)", flush=True)
        return inject(m)
    if m is not None:
        diff = sorted(k for k in set(sw) | set(_MODELS["settings"]) if sw.get(k) != _MODELS["settings"].get(k))
        print(
            f"BATCH (F259) WARNING: {_RUN['name']} needs another model (layers / LM head / seq / config / switches "
            f"{diff} differ from {_MODELS['by']}'s): releasing it and loading again in this process",
            flush=True,
        )
        release()
    seq = max(int(seq), int(os.environ.get(B.MAX_SEQ_ENV) or 0))
    head = lm_head or os.environ.get(B.LM_HEAD_ENV) == "1"
    t0 = time.time()
    _LOADING["on"] = True
    try:
        m = load(seq, head)
    finally:
        _LOADING["on"] = False
    load_s = getattr(m, "load_seconds", time.time() - t0)
    _MODELS.update(
        model=m, key=key, mesh=mesh, layers=want, config=config, settings=sw, lm_head=head, seq=seq,
        by=_RUN["name"], load_s=load_s,
    )  # fmt: skip
    n = _count_load(_RUN["name"])
    metrics.record("model_shared", 0)
    print(
        f"BATCH (F259): model loaded for the batch ({len(want)} layers, lm_head {head}, seq {seq}; load {n} of this "
        f"process, {load_s:.0f} s)",
        flush=True,
    )
    return inject(m)


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


def run_plan(mesh, shape, plan: dict, plan_path: Path | None = None, main=None) -> list[str]:
    """Run every entry of ``plan`` in this process; the names of the runs that failed. ``main``: pytest.main (tests)."""
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
    for k, v in (plan.get("hints") or {}).items():  # the load's LM head / sequence, unless the caller set them
        os.environ.setdefault(k, str(v))
    for k, e in enumerate(entries, 1):
        name = e["name"]
        group = e.get("group") or name
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
        try:
            rc = int(_redirected(e.get("log"), lambda: main(args, plugins=[SharedMesh(mesh, shape)])))
        finally:
            m = _MODELS.get("model")
            if m is not None:
                free_state(m)
                uninject(m)
            gc.collect()  # (the run's own device tensors: an adapter's KV caches, its runtime's slot states)
            _, _, m1, f1 = _results_view()
            os.environ.clear()
            os.environ.update(saved)
            _refresh_settings()
        rc = 0 if rc == 0 else 1  # a test failure; 2 stays the safe runner's HANG
        out["runs"][name] = {
            "rc": rc,
            "s": round(time.time() - t0, 1),
            "loads": _LOADS["n"] - loads0,
            # FIX.29: what this run recorded (a gate keeps it per unit) and the result files it wrote
            "metrics": {k: v for k, v in m1.items() if m0.get(k) != v},
            "files": sorted(k for k, t in f1.items() if f0.get(k) != t),
        }
        out["running"] = None
        save()
        print(f"BATCH (F259) run {name}: rc {rc} in {time.time() - t0:.0f} s", flush=True)
        if rc:
            failed.append(name)
            dead_groups.add(group)
            stop = bool(plan.get("stop_on_fail"))
    release()
    return failed
