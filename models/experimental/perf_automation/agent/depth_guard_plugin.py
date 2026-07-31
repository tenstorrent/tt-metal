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
