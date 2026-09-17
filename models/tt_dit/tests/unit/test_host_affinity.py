# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Pure tests for models.tt_dit.utils.host_affinity (fake sysfs trees, no device)."""

import os

from models.tt_dit.utils import host_affinity as ha


def _fake_sysfs(tmp_path, siblings_by_cpu):
    root = tmp_path / "cpu"
    for cpu, siblings in siblings_by_cpu.items():
        d = root / f"cpu{cpu}" / "topology"
        d.mkdir(parents=True)
        (d / "thread_siblings_list").write_text(siblings + "\n")
    return str(root)


def test_parse_cpu_list_ranges_and_singles():
    assert ha._parse_cpu_list("0,32") == [0, 32]
    assert ha._parse_cpu_list("0-3,8-11") == [0, 1, 2, 3, 8, 9, 10, 11]
    assert ha._parse_cpu_list("5\n") == [5]
    assert ha._parse_cpu_list("") == []


def test_smt_host_keeps_one_sibling_per_core(tmp_path):
    # 4 cores x 2 threads, siblings paired as (c, c+4) like the galaxy host's 0-31/32-63 layout.
    sysfs = _fake_sysfs(tmp_path, {c: f"{c % 4},{c % 4 + 4}" for c in range(8)})
    assert ha.one_thread_per_core(sysfs) == [0, 1, 2, 3]


def test_no_smt_returns_none(tmp_path):
    sysfs = _fake_sysfs(tmp_path, {c: str(c) for c in range(4)})
    assert ha.one_thread_per_core(sysfs) is None


def test_missing_topology_returns_none(tmp_path):
    assert ha.one_thread_per_core(str(tmp_path / "nope")) is None


def test_allowed_mask_is_respected(tmp_path):
    # Operator already restricted the process to CPUs 4-7 (the second siblings): pick those, never widen.
    sysfs = _fake_sysfs(tmp_path, {c: f"{c % 4},{c % 4 + 4}" for c in range(8)})
    assert ha.one_thread_per_core(sysfs, allowed={4, 5, 6, 7}) == [4, 5, 6, 7]
    assert ha.one_thread_per_core(sysfs, allowed={9}) is None


def test_pin_env_disable(monkeypatch):
    monkeypatch.setenv("LTX_PIN_CORES", "0")
    monkeypatch.setattr(ha, "_applied", None)
    assert ha.pin_one_thread_per_core("test") is None


def test_pin_is_idempotent_and_only_narrows(monkeypatch, tmp_path):
    if not hasattr(os, "sched_getaffinity"):
        return  # non-Linux host: nothing to pin
    monkeypatch.delenv("LTX_PIN_CORES", raising=False)
    monkeypatch.setattr(ha, "_applied", None)
    sysfs = _fake_sysfs(tmp_path, {c: f"{c % 4},{c % 4 + 4}" for c in range(8)})
    calls = []
    monkeypatch.setattr(ha, "_SYSFS_CPU", sysfs)
    monkeypatch.setattr(ha.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3, 4, 5, 6, 7})
    monkeypatch.setattr(ha.os, "sched_setaffinity", lambda pid, cpus: calls.append(sorted(cpus)))
    assert ha.pin_one_thread_per_core("test") == [0, 1, 2, 3]
    assert calls == [[0, 1, 2, 3]]
    # second call: no new sched_setaffinity, same answer
    assert ha.pin_one_thread_per_core("test") == [0, 1, 2, 3]
    assert calls == [[0, 1, 2, 3]]


def test_pin_is_noop_when_mask_already_one_per_core(monkeypatch, tmp_path):
    if not hasattr(os, "sched_getaffinity"):
        return
    monkeypatch.delenv("LTX_PIN_CORES", raising=False)
    monkeypatch.setattr(ha, "_applied", None)
    sysfs = _fake_sysfs(tmp_path, {c: f"{c % 4},{c % 4 + 4}" for c in range(8)})
    calls = []
    monkeypatch.setattr(ha, "_SYSFS_CPU", sysfs)
    monkeypatch.setattr(ha.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3})
    monkeypatch.setattr(ha.os, "sched_setaffinity", lambda pid, cpus: calls.append(sorted(cpus)))
    assert ha.pin_one_thread_per_core("test") is None
    assert calls == []
