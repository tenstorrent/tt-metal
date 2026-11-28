# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Where scan_all_chips puts each chip's log and kernel cache. No device: the per-chip pytest is faked."""

import pytest

import tests.didt.test_mm_mcast_write_ack_failure as scan

# chip id (BusDeviceFunction rank) -> (bdf, /dev/tenstorrent minor); minors follow tray/slot, not the rank
IDENTITY = {0: ("0000:01:00.0", 0), 1: ("0000:41:00.0", 8), 2: ("0000:c1:00.0", 16)}


@pytest.fixture
def children(monkeypatch):
    started = []

    class FakePopen:
        def __init__(self, cmd, env, stdout, stderr):
            started.append(env)
            stdout.write("1 passed\n")
            stdout.flush()

        def poll(self):
            return 0

    monkeypatch.setattr(scan, "chip_identity", lambda: IDENTITY)
    monkeypatch.setattr(scan.subprocess, "Popen", FakePopen)
    monkeypatch.delenv("MCAST_ACK_CACHE_DIR", raising=False)
    return started


def test_cache_defaults_to_work_dir_by_kmd(tmp_path, children):
    work = tmp_path / "work"
    assert scan.scan_all_chips(work_dir=work) == []
    assert [e["TT_METAL_CACHE"] for e in children] == [str(work / f"kmd{m}") for m in (0, 8, 16)]
    assert [e["OUT_DIR"] for e in children] == [str(work / f"chip{c}") for c in (0, 1, 2)]
    assert all((work / f"chip{c}" / "pytest.log").is_file() for c in (0, 1, 2))


def test_cache_dir_env_moves_only_the_cache(tmp_path, children, monkeypatch):
    work, cache = tmp_path / "work", tmp_path / "local" / "cache"
    (cache / "kmd8").mkdir(parents=True)
    (cache / "kmd8" / "stale").write_text("warm")
    monkeypatch.setenv("MCAST_ACK_CACHE_DIR", str(cache))
    assert scan.scan_all_chips(work_dir=work) == []
    assert [e["TT_METAL_CACHE"] for e in children] == [str(cache / f"kmd{m}") for m in (0, 8, 16)]
    # always a cold-cache run, wherever the cache lives
    assert not (cache / "kmd8" / "stale").exists()
    assert all((work / f"chip{c}" / "pytest.log").is_file() for c in (0, 1, 2))
