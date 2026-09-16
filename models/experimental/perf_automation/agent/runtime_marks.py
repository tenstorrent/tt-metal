# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fire the marked pass from the model's own construction, not from the test file's text.

inject_stage_marks (stage_marks.py) finds a bare call to bracket by reading the test's source: where
to insert, whether it already ran, whether the pipeline existed yet at that point. Three separate bugs
lived in exactly that placement logic, because there are as many test shapes as there are generated
tests. This reuses the same shape predicate (looks_like_a_pipeline) but changes WHEN the pass fires:
on the model's own construction and on its own input-preparer call, both discovered by wrapping
objects the model already creates, not by editing the file that creates them.

TWO SHAPES, BOTH REAL. voxtral's marked pass needs its own batch bound first -- a module-level
preparer assigns <stage>_trace_inputs from a real batch, because the pipeline's own hooks would
otherwise load captured tensors this tree does not ship. Firing at construction, before that
preparer runs, would trade today's shape-detection bug for a missing-data one on the model that
currently works. nemotron's marked pass needs nothing extra: no preparer exists, and it marks the
instant the pipeline is built. So: wrap the preparer when the test has one, and fire after it runs;
wrap __init__ on pipeline-shaped classes when it doesn't, and fire the instant one is built, since
there is nothing further to wait for.
"""
from __future__ import annotations

import inspect
import sys

from . import stage_marks as _sm


def _model_root(module) -> str:
    """The top-level package a test's own model lives under, e.g. 'models.demos.foo_bar'.

    Restricting the class scan to this prefix is what weight_census.py already does for the same
    reason: PIPELINE_STAGES lives inside the model's own tree, and scanning all of sys.modules would
    wrap classes that have nothing to do with this test."""
    parts = (getattr(module, "__name__", "") or "").split(".")
    return ".".join(parts[:3]) if len(parts) >= 3 else (module.__name__ or "")


def _pipeline_classes_under(root: str):
    """Every class, in a module under `root`, that carries the stage surface -- by shape, not name."""
    if not root:
        return
    seen = set()
    for name, mod in list(sys.modules.items()):
        if mod is None or not (name == root or name.startswith(root + ".")):
            continue
        for obj in list(vars(mod).values()):
            if isinstance(obj, type) and obj not in seen and _sm.looks_like_a_pipeline(obj):
                seen.add(obj)
                yield obj


def install(module):
    """Wrap whatever in `module` will make its pipeline's inputs ready, so the marked pass fires the
    moment they are -- however the test is shaped. Returns a restore callable; always safe to call,
    including when nothing in `module` was found to wrap (a no-op restore).

    Fires at most once: the first pipeline built, or the first preparer call, wins. A test that builds
    more than one pipeline (warmup plus the real run) must not be marked twice.
    """
    restores = []

    def restore_all():
        for fn in reversed(restores):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass

    fired = {"done": False}

    def fire(pipe, device=None):
        if fired["done"]:
            return
        fired["done"] = True
        dev = device if device is not None else getattr(pipe, "device", None)
        if dev is None:
            _sm.no_marks("%r has no .device and none was supplied" % (pipe,))
            return
        _sm.mark_stages_for(pipe, dev)

    try:
        text = inspect.getsource(module)
    except (OSError, TypeError):
        text = ""
    prep_name = _sm.find_input_preparer(text) if text else ""
    prep_fn = getattr(module, prep_name, None) if prep_name else None

    if callable(prep_fn):
        original = prep_fn

        def wrapped_prep(pipe, *a, **kw):
            result = original(pipe, *a, **kw)
            fire(pipe)
            return result

        setattr(module, prep_name, wrapped_prep)
        restores.append(lambda: setattr(module, prep_name, original))
        return restore_all

    # No preparer in this test: nothing to wait for. Mark the instant a pipeline-shaped object exists.
    for cls in _pipeline_classes_under(_model_root(module)):
        real_init = cls.__init__

        def wrapped_init(self, *a, _real=real_init, **kw):
            _real(self, *a, **kw)
            fire(self)

        cls.__init__ = wrapped_init
        restores.append(lambda c=cls, r=real_init: setattr(c, "__init__", r))

    return restore_all
