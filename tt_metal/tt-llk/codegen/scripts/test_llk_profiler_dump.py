# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the llk_profiler_dump pytest plugin (raw profiler zones -> CSV)."""

import enum
import importlib.util
from pathlib import Path

_PATH = Path(__file__).parent / "pytest_plugins" / "llk_profiler_dump.py"
_SPEC = importlib.util.spec_from_file_location("llk_profiler_dump", _PATH)
plugin = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(plugin)


class _Df:
    def __init__(self, text):
        self.text = text

    def to_csv(self, path, index=False):
        Path(path).write_text(self.text)


class _Data:
    def __init__(self, text):
        self.df = _Df(text)


class RunType(enum.Enum):
    MATH_ISOLATE = 3
    L1_TO_L1 = 1


def test_each_readout_is_dumped_with_its_run_type_and_index(tmp_path):
    hooks = []
    plugin.install(hooks, tmp_path)
    for hook in hooks:
        hook(RunType.MATH_ISOLATE, 0, _Data("t,v\n"))
        hook(RunType.L1_TO_L1, 1, _Data("u,w\n"))
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["000_MATH_ISOLATE_run0.csv", "001_L1_TO_L1_run1.csv"]
    assert (tmp_path / files[0]).read_text() == "t,v\n"


def test_install_is_idempotent(tmp_path):
    hooks = []
    plugin.install(hooks, tmp_path)
    plugin.install(hooks, tmp_path)
    # a second install must not add a second dumper (both would write 000_...)
    assert len(hooks) == 1


def test_the_hook_is_called_by_the_perf_harness():
    """helpers/perf/core.py calls PROFILER_READOUT_HOOKS right after Profiler.get_data."""
    core = (
        Path(__file__).resolve().parents[2] / "tests/python_tests/helpers/perf/core.py"
    ).read_text()
    assert "PROFILER_READOUT_HOOKS: list[" in core
    assert "hook(run_type, run_index, profiler_data)" in core
