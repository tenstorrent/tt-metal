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

Each readout becomes ``<NNN>_<RUN_TYPE>_run<i>.csv`` (call order, run type, run
index), handed over by helpers/perf/core.py's ``PROFILER_READOUT_HOOKS``.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV = "LLK_PROFILER_RAW_DUMP"


class _Dumper:
    def __init__(self, out_dir) -> None:
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.count = 0

    def __call__(self, run_type, run_index, profiler_data) -> None:
        name = getattr(run_type, "name", run_type)
        df = getattr(profiler_data, "df", profiler_data)
        df.to_csv(self.out / f"{self.count:03d}_{name}_run{run_index}.csv", index=False)
        self.count += 1


def install(hooks: list, out_dir) -> None:
    """Append a dumper writing to ``out_dir`` to ``hooks``; a second install is a no-op."""

    if not any(isinstance(h, _Dumper) for h in hooks):
        hooks.append(_Dumper(out_dir))


def pytest_sessionstart(session) -> None:
    out_dir = os.environ.get(_ENV)
    if not out_dir:
        return
    from helpers.perf import core  # tt-llk tests/python_tests on sys.path

    hooks = getattr(core, "PROFILER_READOUT_HOOKS", None)
    if hooks is None:
        raise RuntimeError(
            f"{_ENV} is set but {core.__file__} has no PROFILER_READOUT_HOOKS; "
            "the worktree under test predates the profiler readout hook"
        )
    install(hooks, out_dir)
