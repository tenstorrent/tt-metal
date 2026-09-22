# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only ownership tests; all signalled processes are test-created dummies."""

import ast
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

GUARD_PATH = Path(__file__).with_name("vllm_process_guard.py")
spec = importlib.util.spec_from_file_location("vllm_process_guard", GUARD_PATH)
guard = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = guard
spec.loader.exec_module(guard)

DUMMY_CHILD = """
import json, os, signal, sys, time
from pathlib import Path
if sys.argv[2] == 'ignore':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(json.dumps({
    'pid': os.getpid(), 'marker': os.environ['QWEN_VLLM_LAUNCH_ID']}))
time.sleep(30)
"""

DUMMY_PARENT = """
import subprocess, sys, time
from pathlib import Path
subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2], sys.argv[3]])
while not Path(sys.argv[2]).exists():
    time.sleep(0.01)
if sys.argv[4] == 'hold':
    time.sleep(30)
else:
    sys.exit(int(sys.argv[4]))
"""


def wait_record(path):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.01)
    raise AssertionError(f"Dummy process did not record its identity at {path}")


@pytest.fixture
def dummies():
    records, handles = [], []
    yield records, handles
    # A TERM-ignoring dummy deliberately survives the guard's bounded timeout.
    # Test cleanup may kill that known dummy; production guard never SIGKILLs.
    for path in records:
        if path.exists():
            record = json.loads(path.read_text())
            process = guard.read_owned(record["pid"], record["marker"])
            if process is not None:
                guard.signal_owned(process, record["marker"], signal.SIGKILL)
    for process in handles:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


def test_packaged_shutdown_reproduces_orphan(tmp_path, dummies):
    """Execute the installed shutdown body against an API-style dummy parent."""
    run_root = Path(__file__).resolve().parents[5]
    packaged = (
        run_root
        / "codex-home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.4"
        / "runtime/readiness_check/run_vllm_server.py"
    )
    source = ast.parse(packaged.read_text())
    shutdown = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_shutdown")
    namespace = {"subprocess": subprocess, "Path": Path}
    module = ast.fix_missing_locations(ast.Module(body=[shutdown], type_ignores=[]))
    exec(compile(module, str(packaged), "exec"), namespace)
    records, handles = dummies
    record_path = tmp_path / "original_child.json"
    records.append(record_path)
    marker = uuid.uuid4().hex
    parent = subprocess.Popen(
        [sys.executable, "-c", DUMMY_PARENT, DUMMY_CHILD, str(record_path), "normal", "hold"],
        env={**os.environ, guard.LAUNCH_MARKER: marker},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    handles.append(parent)
    record = wait_record(record_path)

    namespace["_shutdown"](parent, tmp_path / "no_server_log")

    assert parent.poll() is not None
    assert guard.read_owned(record["pid"], marker) is not None


@pytest.mark.parametrize("cancel_signal", [None, signal.SIGINT, signal.SIGTERM])
def test_owned_orphan_cleanup_preserves_other_launch(tmp_path, dummies, cancel_signal):
    records, handles = dummies
    sentinel_path = tmp_path / "sentinel.json"
    child_path = tmp_path / "owned.json"
    records.extend([sentinel_path, child_path])
    previous_marker = uuid.uuid4().hex
    sentinel = subprocess.Popen(
        [sys.executable, "-c", DUMMY_CHILD, str(sentinel_path), "normal"],
        env={**os.environ, guard.LAUNCH_MARKER: previous_marker},
    )
    handles.append(sentinel)
    wait_record(sentinel_path)
    process = subprocess.Popen(
        [
            sys.executable,
            str(GUARD_PATH),
            "--runner-grace",
            "0.2",
            "--shutdown-timeout",
            "1",
            "--",
            sys.executable,
            "-c",
            DUMMY_PARENT,
            DUMMY_CHILD,
            str(child_path),
            "normal",
            "hold" if cancel_signal is not None else "7",
        ],
        env={**os.environ, guard.LAUNCH_MARKER: previous_marker},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    handles.append(process)
    child = wait_record(child_path)
    assert child["marker"] != previous_marker
    if cancel_signal is not None:
        process.send_signal(cancel_signal)
    output, _ = process.communicate(timeout=8)

    assert process.returncode == (7 if cancel_signal is None else 128 + cancel_signal), output
    assert "VLLM_STAGE_CLEANUP complete" in output
    assert guard.read_owned(child["pid"], child["marker"]) is None
    assert not Path(f"/proc/{child['pid']}").exists()
    assert sentinel.poll() is None


def test_term_ignoring_child_is_reported_without_kill(tmp_path, dummies):
    records, handles = dummies
    child_path = tmp_path / "ignores_term.json"
    records.append(child_path)
    process = subprocess.Popen(
        [
            sys.executable,
            str(GUARD_PATH),
            "--shutdown-timeout",
            "0.3",
            "--",
            sys.executable,
            "-c",
            DUMMY_PARENT,
            DUMMY_CHILD,
            str(child_path),
            "ignore",
            "0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    handles.append(process)
    child = wait_record(child_path)
    # The dummy inherits stdout. Do not communicate() while it deliberately
    # remains alive; wait for the guard, then read the three bounded log records.
    process.wait(timeout=5)
    output = "".join(process.stdout.readline() for _ in range(3))

    assert process.returncode == 1, output
    assert f"remaining_pids={child['pid']}" in output
    assert guard.read_owned(child["pid"], child["marker"]) is not None


def test_marker_and_process_identity_are_exact(tmp_path):
    pid = 99999999
    path = tmp_path / str(pid)
    path.mkdir()
    fields = ["S"] + ["0"] * 19
    fields[19] = "12345"
    (path / "stat").write_text(f"{pid} (name with spaces) {' '.join(fields)}")
    (path / "environ").write_bytes(b"QWEN_VLLM_LAUNCH_ID=abcd-extra\0")
    assert guard.owned_processes("abcd", tmp_path) == []
    (path / "environ").write_bytes(b"OTHER=x\0QWEN_VLLM_LAUNCH_ID=abcd\0")
    assert guard.owned_processes("abcd", tmp_path) == [guard.OwnedProcess(pid, 12345)]
    fields[0] = "Z"
    (path / "stat").write_text(f"{pid} (zombie) {' '.join(fields)}")
    assert guard.owned_processes("abcd", tmp_path) == []


def test_identity_is_rechecked_after_pidfd_open(monkeypatch):
    sent, closed = [], []
    process = guard.OwnedProcess(1234, 100)
    monkeypatch.setattr(guard, "pidfd_open", lambda pid: 42)
    monkeypatch.setattr(guard.os, "close", closed.append)
    monkeypatch.setattr(guard, "pidfd_send_signal", lambda fd, sig: sent.append((fd, sig)))
    monkeypatch.setattr(guard, "read_owned", lambda pid, marker: guard.OwnedProcess(pid, 101))

    assert not guard.signal_owned(process, "abcd", signal.SIGTERM)
    assert sent == []
    assert closed == [42]

    monkeypatch.setattr(guard, "read_owned", lambda pid, marker: process)
    assert guard.signal_owned(process, "abcd", signal.SIGTERM)
    assert sent == [(42, signal.SIGTERM)]


def test_unmarked_adopted_child_is_reaped(tmp_path):
    """An engine can lose its marker through exec/setproctitle before API exit."""
    child_path = tmp_path / "unmarked.json"
    child = "import os,time; print(os.getpid(), flush=True); time.sleep(30)"
    parent = (
        "import subprocess,sys; from pathlib import Path; "
        "p=subprocess.Popen([sys.executable,'-c',sys.argv[1]],env={},stdout=subprocess.PIPE,text=True); "
        "Path(sys.argv[2]).write_text(p.stdout.readline()); sys.exit(7)"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            str(GUARD_PATH),
            "--shutdown-timeout",
            "2",
            "--",
            sys.executable,
            "-c",
            parent,
            child,
            str(child_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        output, _ = process.communicate(timeout=8)
        assert process.returncode == 7, output
        assert "VLLM_STAGE_CLEANUP complete" in output
        assert not Path(f"/proc/{int(child_path.read_text())}").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_adopted_identity_and_parent_are_rechecked(monkeypatch):
    process = guard.OwnedProcess(1234, 100)
    sent = []
    monkeypatch.setattr(guard, "pidfd_open", lambda pid: 42)
    monkeypatch.setattr(guard.os, "close", lambda fd: None)
    monkeypatch.setattr(guard, "pidfd_send_signal", lambda fd, sig: sent.append(sig))
    monkeypatch.setattr(guard, "read_owned", lambda pid, marker: None)
    monkeypatch.setattr(guard, "read_adopted", lambda pid: guard.OwnedProcess(pid, 101))
    assert not guard.signal_owned(process, "abcd", signal.SIGTERM, allow_adopted=True)
    monkeypatch.setattr(guard, "read_adopted", lambda pid: process)
    assert not guard.signal_owned(process, "abcd", signal.SIGTERM)
    assert guard.signal_owned(process, "abcd", signal.SIGTERM, allow_adopted=True)
    assert sent == [signal.SIGTERM]


def test_idle_preflight_rejects_unknown_owner_and_missing_devices(tmp_path, expect_error):
    for index in range(4):
        directory = tmp_path / str(index)
        directory.mkdir()
        (directory / "pids").write_text("")
    guard.require_idle_tt(4, tmp_path)
    (tmp_path / "0/pids").write_text("0\n0\n")
    with expect_error(RuntimeError, "idle devices"):
        guard.require_idle_tt(4, tmp_path)
    (tmp_path / "0/pids").unlink()
    with expect_error(RuntimeError, "idle devices"):
        guard.require_idle_tt(4, tmp_path)
