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


# --- reexec_pinned_before_torch: pure no-op-condition tests (os.execv always mocked) ---


def _arm_reexec(monkeypatch, current, chosen, execv_calls, setaff_calls=None):
    """Wire host_affinity so reexec_pinned_before_torch computes ``chosen`` from ``current``.

    os.execv is always mocked so the test process never actually re-execs.
    """
    monkeypatch.setattr(ha.os, "sched_getaffinity", lambda pid: set(current))
    monkeypatch.setattr(ha, "_chosen_cores", lambda: (set(chosen) if chosen is not None else None))
    if setaff_calls is not None:
        monkeypatch.setattr(ha.os, "sched_setaffinity", lambda pid, cpus: setaff_calls.append((pid, sorted(cpus))))
    else:
        monkeypatch.setattr(ha.os, "sched_setaffinity", lambda pid, cpus: None)
    monkeypatch.setattr(ha.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(ha.sys, "argv", ["prog", "arg"])
    monkeypatch.setattr(ha.os, "execv", lambda exe, argv: execv_calls.append((exe, list(argv))))


def test_reexec_noop_when_pin_cores_disabled(monkeypatch):
    monkeypatch.setenv("LTX_PIN_CORES", "0")
    monkeypatch.delenv("_LTX_REEXECED", raising=False)
    calls = []
    _arm_reexec(monkeypatch, current={0, 1, 2, 3, 4, 5, 6, 7}, chosen={0, 1, 2, 3}, execv_calls=calls)
    ha.reexec_pinned_before_torch("test")
    assert calls == []


def test_reexec_noop_when_already_reexeced(monkeypatch):
    monkeypatch.delenv("LTX_PIN_CORES", raising=False)
    monkeypatch.setenv("_LTX_REEXECED", "1")
    calls = []
    _arm_reexec(monkeypatch, current={0, 1, 2, 3, 4, 5, 6, 7}, chosen={0, 1, 2, 3}, execv_calls=calls)
    ha.reexec_pinned_before_torch("test")
    assert calls == []


def test_reexec_noop_when_mask_already_chosen(monkeypatch):
    monkeypatch.delenv("LTX_PIN_CORES", raising=False)
    monkeypatch.delenv("_LTX_REEXECED", raising=False)
    calls = []
    # _chosen_cores returns None when the mask already equals the chosen set.
    _arm_reexec(monkeypatch, current={0, 1, 2, 3}, chosen=None, execv_calls=calls)
    ha.reexec_pinned_before_torch("test")
    assert calls == []


def test_reexec_noop_when_no_topology(monkeypatch):
    monkeypatch.delenv("LTX_PIN_CORES", raising=False)
    monkeypatch.delenv("_LTX_REEXECED", raising=False)
    calls = []
    _arm_reexec(monkeypatch, current={0, 1, 2, 3, 4, 5, 6, 7}, chosen=None, execv_calls=calls)
    ha.reexec_pinned_before_torch("test")
    assert calls == []


def test_reexec_pins_and_execs_when_armed(monkeypatch):
    if not hasattr(os, "sched_getaffinity"):
        return
    monkeypatch.delenv("LTX_PIN_CORES", raising=False)
    monkeypatch.delenv("_LTX_REEXECED", raising=False)
    execv_calls = []
    setaff_calls = []
    _arm_reexec(
        monkeypatch,
        current={0, 1, 2, 3, 4, 5, 6, 7},
        chosen={0, 1, 2, 3},
        execv_calls=execv_calls,
        setaff_calls=setaff_calls,
    )
    ha.reexec_pinned_before_torch("test")
    # It set the mask to the chosen cores, marked the sentinel, and re-execed the same argv.
    assert setaff_calls == [(0, [0, 1, 2, 3])]
    assert os.environ.get("_LTX_REEXECED") == "1"
    assert execv_calls == [("/usr/bin/python3", ["/usr/bin/python3", "prog", "arg"])]
