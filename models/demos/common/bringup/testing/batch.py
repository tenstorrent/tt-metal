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


def device_model(s, mesh, layers, lm_head: bool = True):
    """The device model of ``layers`` (hooks.device_model), shared by the runs of a batch."""
    from models.demos.common.bringup.core import metrics

    if not active():
        return s.hooks().device_model(mesh, s, layers, lm_head=lm_head)
    key, want = os.environ[B.KEY_ENV], list(layers)
    m = _MODELS.get("model")
    need_seq = _target_seq(s)
    if (
        m is not None
        and _MODELS["key"] == key
        and _MODELS["mesh"] is mesh
        and _MODELS["layers"] == want
        and (_MODELS["lm_head"] or not lm_head)
        and _MODELS["seq"] >= need_seq
    ):
        free_state(m)
        metrics.record("model_shared", 1)
        print(f"BATCH (F259): model reused (loaded by {_MODELS['by']} in {_MODELS['load_s']:.0f} s)", flush=True)
        return m
    if m is not None:
        print(
            f"BATCH (F259) WARNING: {_RUN['name']} needs another model (layers / LM head / seq / switches differ from "
            f"{_MODELS['by']}'s): releasing it and loading again in this process",
            flush=True,
        )
        release()
    sb = copy.copy(s)  # the load's spec: the plan's longest sequence and the LM head if any run needs them
    sb.data = copy.deepcopy(s.data)
    seq = max(need_seq, int(os.environ.get(B.MAX_SEQ_ENV) or 0))
    sb.data["target"]["seq"] = seq
    head = lm_head or os.environ.get(B.LM_HEAD_ENV) == "1"
    t0 = time.time()
    m = s.hooks().device_model(mesh, sb, want, lm_head=head)
    load_s = getattr(m, "load_seconds", time.time() - t0)
    _MODELS.update(model=m, key=key, mesh=mesh, layers=want, lm_head=head, seq=seq, by=_RUN["name"], load_s=load_s)
    metrics.record("model_shared", 0)
    print(f"BATCH (F259): model loaded for the batch ({len(want)} layers, lm_head {head}, seq {seq})", flush=True)
    return m


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
        t0 = time.time()
        try:
            rc = int(_redirected(e.get("log"), lambda: main(args, plugins=[SharedMesh(mesh, shape)])))
        finally:
            m = _MODELS.get("model")
            if m is not None:
                free_state(m)
            os.environ.clear()
            os.environ.update(saved)
        rc = 0 if rc == 0 else 1  # a test failure; 2 stays the safe runner's HANG
        out["runs"][name] = {"rc": rc, "s": round(time.time() - t0, 1)}
        out["running"] = None
        save()
        print(f"BATCH (F259) run {name}: rc {rc} in {time.time() - t0:.0f} s", flush=True)
        if rc:
            failed.append(name)
            dead_groups.add(group)
            stop = bool(plan.get("stop_on_fail"))
    release()
    return failed
