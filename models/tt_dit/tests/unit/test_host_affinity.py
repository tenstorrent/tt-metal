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


def _reset(monkeypatch):
    monkeypatch.setattr(ha, "_applied", None)
    monkeypatch.setattr(ha, "_full", None)


def test_pin_env_disable(monkeypatch):
    monkeypatch.setenv("LTX_PIN_CORES", "0")
    _reset(monkeypatch)
    assert ha.pin_one_thread_per_core("test") is None


def test_pin_narrows_only_full_mask_threads(monkeypatch, tmp_path):
    if not hasattr(os, "sched_getaffinity"):
        return  # non-Linux host: nothing to pin
    monkeypatch.delenv("LTX_PIN_CORES", raising=False)
    _reset(monkeypatch)
    sysfs = _fake_sysfs(tmp_path, {c: f"{c % 4},{c % 4 + 4}" for c in range(8)})
    monkeypatch.setattr(ha, "_SYSFS_CPU", sysfs)
    full = {0, 1, 2, 3, 4, 5, 6, 7}
    masks = {
        100: set(full),
        101: {6},
        102: set(full),
        103: {0, 1, 2, 3},
    }  # 101: tt-metal's own pin; 103: already narrowed
    monkeypatch.setattr(ha, "_thread_ids", lambda: sorted(masks))
    monkeypatch.setattr(ha.os, "sched_getaffinity", lambda tid: set(masks[tid]) if tid else set(full))
    calls = []
    monkeypatch.setattr(ha.os, "sched_setaffinity", lambda tid, cpus: calls.append((tid, sorted(cpus))))
    monkeypatch.setattr(ha, "_cap_torch_threads", lambda n: calls.append(("torch", n)))
    assert ha.pin_one_thread_per_core("test") == [0, 1, 2, 3]
    assert calls == [(100, [0, 1, 2, 3]), (102, [0, 1, 2, 3]), ("torch", 4)]
    # a thread born later with the full mask gets narrowed on the next call; the pinned one is still left alone
    masks[100] = {0, 1, 2, 3}
    masks[102] = {0, 1, 2, 3}
    masks[104] = set(full)
    assert ha.pin_one_thread_per_core("test") == [0, 1, 2, 3]
    assert calls[-1] == (104, [0, 1, 2, 3])


def test_pin_is_noop_when_mask_already_one_per_core(monkeypatch, tmp_path):
    if not hasattr(os, "sched_getaffinity"):
        return
    monkeypatch.delenv("LTX_PIN_CORES", raising=False)
    _reset(monkeypatch)
    sysfs = _fake_sysfs(tmp_path, {c: f"{c % 4},{c % 4 + 4}" for c in range(8)})
    monkeypatch.setattr(ha, "_SYSFS_CPU", sysfs)
    monkeypatch.setattr(ha, "_thread_ids", lambda: [100])
    monkeypatch.setattr(ha.os, "sched_getaffinity", lambda tid: {0, 1, 2, 3})
    calls = []
    monkeypatch.setattr(ha.os, "sched_setaffinity", lambda tid, cpus: calls.append((tid, sorted(cpus))))
    assert ha.pin_one_thread_per_core("test") is None
    assert calls == []
