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
    # The mean of 1 s cuts in the middle; a weight of 0 would put only TESTS[0]
    # in the first half.
    path = tmp_path / "blackhole.json"
    path.write_text(json.dumps({TESTS[0]: 1.0, TESTS[1]: 1.0, TESTS[3]: 1.0}))
    monkeypatch.setenv("LLK_SPLIT_DURATIONS", str(path))
    assert _cut(_items(*TESTS), 2, 1) == TESTS[:2]
    assert _cut(_items(*TESTS), 2, 2) == TESTS[2:]


@pytest.fixture
def table(monkeypatch):
    """Weigh both modules at 3 s, whatever the committed table says.

    perf_a spreads its 3 s over three tests, perf_b has one, so the cut is 3:1.
    """
    monkeypatch.setattr(
        plugin,
        "_module_costs",
        lambda: ("perf_split_costs.json", {"perf_a": 3.0, "perf_b": 3.0}),
    )


@pytest.mark.parametrize(
    "contents",
    ["{truncated", json.dumps([1, 2, 3])],
    ids=["truncated", "not-an-object"],
)
def test_a_bad_durations_file_falls_back_to_the_table(
    contents, tmp_path, monkeypatch, table
):
    path = tmp_path / "blackhole.json"
    path.write_text(contents)
    monkeypatch.setenv("LLK_SPLIT_DURATIONS", str(path))
    assert _cut(_items(*TESTS), 2, 1) == TESTS[:3]


@pytest.mark.parametrize("env", [None, "missing.json"])
def test_without_durations_the_module_table_is_used(env, monkeypatch, table):
    if env:
        monkeypatch.setenv("LLK_SPLIT_DURATIONS", env)
    else:
        monkeypatch.delenv("LLK_SPLIT_DURATIONS", raising=False)
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
    assert json.loads(Path(".perf_split_durations").read_text()) == pytest.approx(
        {"a::t": 10.0 / 10 + (3.0 + 1.5) / 15, "b::t": 15.0 / 15}
    )


def test_no_terminal_reporter_stores_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session = _session(tmp_path, 1, [("a::t", 1.0)])
    session.config.pluginmanager.get_plugin = lambda name: None
    plugin._store_split_durations(session)
    assert not Path(".perf_split_durations").exists()


def test_a_compile_pass_starts_a_new_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path(".perf_split_durations").write_text(json.dumps({"old::t": 5.0}))
    monkeypatch.setattr(TestConfig, "BUILD_MODE", BuildMode.PRODUCE, raising=False)
    plugin._store_split_durations(_session(tmp_path, 1, [("a::t", 2.0)]))
    assert json.loads(Path(".perf_split_durations").read_text()) == {"a::t": 2.0}
