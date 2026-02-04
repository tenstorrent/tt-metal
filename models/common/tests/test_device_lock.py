# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tt_device_lock cross-process coordination."""

import multiprocessing
import os
import time

import pytest

from models.common.tests.conftest import DeviceLockTimeout, tt_device_lock


class TestDeviceLock:
    """Tests for tt_device_lock context manager."""

    def test_lock_acquires_and_releases(self, tmp_path):
        """Basic acquire/release works."""
        lock_path = str(tmp_path / "test.lock")

        with tt_device_lock(lock_path=lock_path, timeout=5):
            assert os.path.exists(lock_path)
            with open(lock_path) as f:
                content = f.read().strip()
            assert content == str(os.getpid())

    def test_lock_blocks_concurrent_access(self, tmp_path):
        """Second process waits while first holds lock."""
        lock_path = str(tmp_path / "test.lock")
        result_file = str(tmp_path / "results.txt")

        def worker(worker_id, hold_time):
            """Acquire lock, record timestamp, hold for hold_time, release."""
            with tt_device_lock(lock_path=lock_path, timeout=30):
                with open(result_file, "a") as f:
                    f.write(f"{worker_id}:acquired:{time.time()}\n")
                time.sleep(hold_time)
                with open(result_file, "a") as f:
                    f.write(f"{worker_id}:released:{time.time()}\n")

        # Start worker 1, let it acquire lock
        p1 = multiprocessing.Process(target=worker, args=(1, 2))
        p1.start()
        time.sleep(0.3)  # Give p1 time to acquire

        # Start worker 2, should block
        p2 = multiprocessing.Process(target=worker, args=(2, 0.1))
        p2.start()

        p1.join()
        p2.join()

        # Parse results
        with open(result_file) as f:
            lines = f.read().strip().split("\n")

        events = []
        for line in lines:
            parts = line.split(":")
            events.append((int(parts[0]), parts[1], float(parts[2])))

        # Worker 1 should acquire before worker 2
        next(e for e in events if e[0] == 1 and e[1] == "acquired")
        w1_release = next(e for e in events if e[0] == 1 and e[1] == "released")
        w2_acquire = next(e for e in events if e[0] == 2 and e[1] == "acquired")

        # Worker 2 should only acquire AFTER worker 1 releases
        assert (
            w2_acquire[2] >= w1_release[2]
        ), f"Worker 2 acquired at {w2_acquire[2]} but worker 1 released at {w1_release[2]}"

    def test_lock_timeout_raises(self, tmp_path):
        """Timeout raises DeviceLockTimeout."""
        lock_path = str(tmp_path / "test.lock")

        def holder():
            """Hold lock for a long time."""
            with tt_device_lock(lock_path=lock_path, timeout=60):
                time.sleep(10)

        # Start holder
        p = multiprocessing.Process(target=holder)
        p.start()
        time.sleep(0.3)  # Let it acquire

        # Try to acquire with short timeout - should raise
        try:
            with pytest.raises(DeviceLockTimeout):
                with tt_device_lock(lock_path=lock_path, timeout=1):
                    pass
        finally:
            p.terminate()
            p.join()

    def test_lock_file_created_if_missing(self, tmp_path):
        """Lock file is created if it doesn't exist."""
        lock_path = str(tmp_path / "subdir" / "test.lock")
        assert not os.path.exists(lock_path)

        with tt_device_lock(lock_path=lock_path, timeout=5):
            assert os.path.exists(lock_path)

    def test_sequential_locks_work(self, tmp_path):
        """Multiple sequential lock/unlock cycles work."""
        lock_path = str(tmp_path / "test.lock")

        for i in range(3):
            with tt_device_lock(lock_path=lock_path, timeout=5):
                with open(lock_path) as f:
                    assert f.read().strip() == str(os.getpid())
