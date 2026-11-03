# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the llk_profiler_dump pytest plugin (raw profiler zones -> CSV)."""

import enum
import importlib.util
import types
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


def _fake_profiler_module():
    module = types.ModuleType("fake_profiler")

    class Profiler:
        @staticmethod
        def get_data(test_name, variant_id, location="0,0"):
            return _Data(f"{test_name},{variant_id}\n")

    module.Profiler = Profiler
    return module


def test_wrapped_get_data_dumps_with_caller_run_type(tmp_path):
    module = _fake_profiler_module()
    plugin.install(module, tmp_path)

    def caller():
        run_type = (
            RunType.MATH_ISOLATE
        )  # noqa: F841 - read from the frame by the plugin
        run_index = 0  # noqa: F841
        return module.Profiler.get_data("t", "v")

    result = caller()
    assert isinstance(result, _Data)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["000_MATH_ISOLATE_run0.csv"]
    assert (tmp_path / files[0]).read_text() == "t,v\n"


def test_install_is_idempotent_and_numbers_calls(tmp_path):
    module = _fake_profiler_module()
    plugin.install(module, tmp_path)
    plugin.install(module, tmp_path)
    module.Profiler.get_data("t", "v")
    module.Profiler.get_data("t", "v")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "000_unknown_run0.csv",
        "001_unknown_run0.csv",
    ]
