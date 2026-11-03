# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""pytest plugin: dump every raw LLK profiler readout (absolute wall-clock zones) to CSV.

The perf tests only keep per-tile aggregates; aligning a waveform with the
TILE_LOOP zone needs the absolute ZONE_START / ZONE_END timestamps. Load this
plugin without touching tt-llk test code:

    LLK_PROFILER_RAW_DUMP=/path/to/dir \
    PYTHONPATH=<tt-llk>/codegen/scripts/pytest_plugins:$PYTHONPATH \
    PYTEST_ADDOPTS="-p llk_profiler_dump" \
    .claude/scripts/run_test.sh simulate ...

Each readout becomes ``<NNN>_<RUN_TYPE>_run<i>.csv`` (call order, run type and
run index read from the caller's frame in helpers/perf/core.py).
"""

from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

_ENV = "LLK_PROFILER_RAW_DUMP"
_MARK = "_llk_profiler_dump_wrapped"


def install(profiler_module, out_dir) -> None:
    """Wrap ``profiler_module.Profiler.get_data`` so each result is also written to ``out_dir``."""

    profiler = profiler_module.Profiler
    original = profiler.get_data
    if getattr(original, _MARK, False):
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counter = {"n": 0}

    @functools.wraps(original)
    def get_data(*args, **kwargs):
        result = original(*args, **kwargs)
        caller = sys._getframe(1).f_locals
        run_type = caller.get("run_type")
        run_type = getattr(run_type, "name", run_type) or "unknown"
        run_index = caller.get("run_index", 0)
        df = getattr(result, "df", result)
        df.to_csv(
            out / f"{counter['n']:03d}_{run_type}_run{run_index}.csv", index=False
        )
        counter["n"] += 1
        return result

    setattr(get_data, _MARK, True)
    profiler.get_data = (
        staticmethod(get_data)
        if isinstance(profiler.__dict__.get("get_data"), staticmethod)
        else get_data
    )


def pytest_sessionstart(session) -> None:
    out_dir = os.environ.get(_ENV)
    if not out_dir:
        return
    from helpers import (
        profiler as profiler_module,  # tt-llk tests/python_tests on sys.path
    )

    install(profiler_module, out_dir)
