# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pytest plugin: make "all layers" survive a perf test that fills in its own depth default.

THE PROBLEM
    The tool asks for the whole model by REMOVING TT_PERF_LAYERS -- absence is the one value no
    builder can misread (a literal 0 was read as "build zero layers"). But a perf test can fill the
    gap back in at import time:

        os.environ.setdefault("TT_PERF_LAYERS", "2")     # models/demos/xtts_v2/.../test_tts_perf.py

    `setdefault` fires precisely BECAUSE the key is missing, so it converts "all layers" into a
    2-layer build. The full-pipeline gate then measures 2 blocks and reports the number as
    whole-model latency -- no crash, no marker, just a wrong number. Every perf test the tool
    generated before 2026-07-26 carries that line, so this is the tool's own past output, and those
    model files cannot be edited.

WHY A PLUGIN, AND WHY THIS HOOK
    The default is applied at MODULE IMPORT; the model is built inside the TEST BODY. Deleting the
    variable between those two points restores absence no matter what the module did at import --
    without monkeypatching os.environ, and without needing to know the model's depth, its config, or
    its all-layers sentinel.

    `pytest_runtest_setup` is exactly that seam: it runs after collection/import and before the test
    function. Registered via `-p` so it loads before the test module is imported.

SCOPE
    Active ONLY when PERF_MCP_FORCE_ALL_LAYERS=1, which the callers that want the whole model set
    (the full-pipeline gate, the correctness gate, the op-signature probes). The tracy run must NOT
    set it: that run legitimately wants a capped window, and its cap is an explicit positive number
    the plugin leaves alone.
"""

from __future__ import annotations

import os

ENV = "TT_PERF_LAYERS"
FORCE_ALL = "PERF_MCP_FORCE_ALL_LAYERS"
DEPTH_VARS = "PERF_MCP_DEPTH_VARS"


def _forcing_all_layers() -> bool:
    return os.environ.get(FORCE_ALL) == "1"


def depth_vars() -> list:
    """Which variable names cap this model's depth.

    TT_PERF_LAYERS is the tool's own convention and covers every perf test the tool generated, but an
    EXISTING demo can read anything (MAX_LAYERS, TT_NUM_LAYERS, ...). run.py discovers the real names
    via _llm_depth_env -- an agent reading the model's tt/*.py -- and passes them here as a
    comma-separated PERF_MCP_DEPTH_VARS, so the guard drops the right key instead of a guessed one.

    Deliberately a NAMED list, never "drop whatever the module added at import": llama's own perf test
    sets os.environ["HF_MODEL"] at import time (test_main_perf.py:23), so a blanket sweep would break
    the model it is trying to measure.
    """
    raw = os.environ.get(DEPTH_VARS) or ""
    names = [n.strip() for n in raw.split(",") if n.strip()]
    if ENV not in names:
        names.append(ENV)
    return names


def pytest_runtest_setup(item):  # noqa: ARG001 - pytest hook signature
    """Drop a depth cap the test module filled in for itself at import time.

    Only fires when the caller asked for all layers. A cap the CALLER set is a positive number it
    wants honoured -- but the caller expresses "all layers" as absence, so any depth variable present
    at this point, with FORCE_ALL set, was put there by the module.
    """
    if not _forcing_all_layers():
        return
    for name in depth_vars():
        os.environ.pop(name, None)


def _install_factory_tagger() -> None:
    """Wrap every model factory AS IT IS IMPORTED, so the op-sig probe can walk the built model.

    THE TRIGGER IS THE PROBLEM, NOT THE FINDER. find_all_stacks reads "any same-typed object with
    __dict__" and can see any model -- but it was only ever reached from a torch.nn.Module.__call__
    or LightweightModule.__call__ hook, so a model whose blocks subclass neither is never walked.
    Two models have paid for that: llama3_1_8b_p150 (torch-only hook, full_blocks=0, four extra
    ladder probes to recover depths the markers give free) and Voxtral-Mini-3B (`class
    TtEncoderLayer:` with no base, a discoverable 32-element list nothing walked, ONE depth sized
    for a three-section model). Each fix added a class to the whitelist and left the next shape to be
    found later.

    ON IMPORT, not at collection. The first attempt wrapped factories found in sys.modules from
    pytest_collection_finish and did nothing at all, because generated perf tests import the factory
    INSIDE the function that builds:

        def _build_for_perf(dev):
            from ...tt.pipeline import build_pipeline

    At collection time that module is not imported yet, so there was nothing to wrap and the run
    still reported full_blocks=0. A post-import hook fires whenever the module actually loads, which
    is the only moment that works for both eager and lazy imports.

    Best-effort: no probe, no factory, an import that raises -- all leave the run untouched. This
    runs inside a profiling probe, and a probe that can break the test it measures is worse than one
    that measures less.
    """
    import builtins

    if getattr(builtins, "_perf_factory_tagger", False):
        return
    _orig_import = builtins.__import__

    def _tag(mod):
        fn = getattr(mod, "build_pipeline", None)
        if fn is None or not callable(fn) or getattr(fn, "_perf_tagged", False):
            return
        # Read the tagger from builtins: the probe runs as __main__, so importing it by name
        # yields a different module object without the tagger attached.
        tagger = getattr(builtins, "_perf_tag_built_model", None)
        if tagger is None:
            return  # probe not installed for this run

        def _wrapped(*a, __fn=fn, __t=tagger, **k):
            out = __fn(*a, **k)
            try:
                __t(out)
            except Exception:  # noqa: BLE001
                pass
            return out

        _wrapped._perf_tagged = True
        try:
            setattr(mod, "build_pipeline", _wrapped)
        except Exception:  # noqa: BLE001
            pass

    def _import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        mod = _orig_import(name, globals, locals, fromlist, level)
        try:
            _tag(mod)
            # `from X import build_pipeline` returns the PACKAGE for a dotted name, so the
            # submodule that actually defines the factory has to be reached through fromlist.
            for sub in fromlist or ():
                _tag(getattr(mod, sub, None)) if hasattr(mod, sub) else None
            import sys as _s

            if fromlist:
                _tag(_s.modules.get(name))
        except Exception:  # noqa: BLE001
            pass
        return mod

    builtins.__import__ = _import
    builtins._perf_factory_tagger = True


_install_factory_tagger()
