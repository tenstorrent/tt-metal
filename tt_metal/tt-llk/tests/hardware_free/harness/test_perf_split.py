# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import llk_pytest_plugin as plugin
from helpers.test_config import BuildMode, TestConfig


def _items(*nodeids):
    return [SimpleNamespace(nodeid=n, path=Path(n.split("::")[0])) for n in nodeids]


def _cut(items, splits, group):
    options = {"--perf-splits": splits, "--perf-group": group}
    config = SimpleNamespace(
        getoption=options.get,
        hook=SimpleNamespace(pytest_deselected=lambda items: None),
    )
    items = list(items)
    plugin._split_by_cost(config, items)
    return [item.nodeid for item in items]


TESTS = ["perf_a.py::t[1]", "perf_a.py::t[2]", "perf_a.py::t[3]", "perf_b.py::t[1]"]


def test_durations_set_the_weights(tmp_path, monkeypatch):
    path = tmp_path / "blackhole.json"
    path.write_text(
        json.dumps({TESTS[0]: 9.0, TESTS[1]: 1.0, TESTS[2]: 1.0, TESTS[3]: 7.0})
    )
    monkeypatch.setenv("LLK_SPLIT_DURATIONS", str(path))
    assert _cut(_items(*TESTS), 2, 1) == [TESTS[0]]
    assert _cut(_items(*TESTS), 2, 2) == TESTS[1:]


def test_a_test_without_a_duration_weighs_the_mean(tmp_path, monkeypatch):
    path = tmp_path / "blackhole.json"
    path.write_text(json.dumps({TESTS[0]: 1.0, TESTS[1]: 1.0, TESTS[2]: 4.0}))
    monkeypatch.setenv("LLK_SPLIT_DURATIONS", str(path))
    assert _cut(_items(*TESTS), 2, 2) == TESTS[2:]


@pytest.mark.parametrize("env", [None, "missing.json"])
def test_without_durations_the_module_table_is_used(env, monkeypatch):
    if env:
        monkeypatch.setenv("LLK_SPLIT_DURATIONS", env)
    else:
        monkeypatch.delenv("LLK_SPLIT_DURATIONS", raising=False)
    monkeypatch.setattr(
        TestConfig, "CHIP_ARCH", SimpleNamespace(value="blackhole"), raising=False
    )
    monkeypatch.setattr(TestConfig, "SPEED_OF_LIGHT", False, raising=False)
    assert _cut(_items(*TESTS), 2, 1) == TESTS[:3]
    assert _cut(_items(*TESTS), 2, 2) == TESTS[3:]


def _session(tmp_path, workers, reports):
    stats = {
        "passed": [
            pytest.TestReport(n, (n, 0, n), {}, "passed", None, "call", duration=d)
            for n, d in reports
        ]
    }
    options = {"--perf-splits": 5, "numprocesses": workers}
    reporter = SimpleNamespace(stats=stats)
    config = SimpleNamespace(
        getoption=lambda name, default=None: options.get(name, default),
        pluginmanager=SimpleNamespace(get_plugin=lambda name: reporter),
    )
    return SimpleNamespace(config=config)


def test_the_run_pass_adds_to_the_compile_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(TestConfig, "BUILD_MODE", BuildMode.PRODUCE, raising=False)
    plugin._store_split_durations(_session(tmp_path, 10, [("a::t", 10.0)]))
    monkeypatch.setattr(TestConfig, "BUILD_MODE", BuildMode.CONSUME, raising=False)
    plugin._store_split_durations(
        _session(tmp_path, 15, [("a::t", 3.0), ("a::t", 1.5), ("b::t", 15.0)])
    )
    assert json.loads(Path(".test_durations").read_text()) == pytest.approx(
        {"a::t": 1.3, "b::t": 1.0}
    )


def test_a_compile_pass_starts_a_new_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path(".test_durations").write_text(json.dumps({"old::t": 5.0}))
    monkeypatch.setattr(TestConfig, "BUILD_MODE", BuildMode.PRODUCE, raising=False)
    plugin._store_split_durations(_session(tmp_path, 1, [("a::t", 2.0)]))
    assert json.loads(Path(".test_durations").read_text()) == {"a::t": 2.0}
