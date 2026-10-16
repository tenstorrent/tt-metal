# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F71: the dashboard links and the DASHBOARD STALE wake."""

import os
import time

from models.demos.common.bringup import watch as W
from models.demos.common.bringup.dashboard import links as LK


def test_stale_only_with_links_and_newer_files(tmp_path):
    d = tmp_path / "dashboard"
    d.mkdir()
    (d / "index.html").write_text("x")
    assert LK.stale(tmp_path) is None  # never published: no links, nothing to keep current
    LK.save(tmp_path, {"links": {"standard": "https://claude.ai/artifact/A"}, "published_at": time.time()})
    assert LK.stale(tmp_path) is None  # just published
    old = time.time() - 3600
    LK.save(tmp_path, {"links": {"standard": "https://claude.ai/artifact/A"}, "published_at": old})
    os.utime(d / "index.html", (time.time(), time.time()))
    line = LK.stale(tmp_path)
    assert line.startswith("DASHBOARD STALE") and "standard -> https://claude.ai/artifact/A" in line
    os.utime(d / "index.html", (old - 10, old - 10))  # files older than the last publish: current
    assert LK.stale(tmp_path) is None


def test_watch_wakes_on_a_stale_dashboard(tmp_path):
    (tmp_path / "orchestrator.log").write_text("")
    (tmp_path / W.PID_FILE).write_text(f"{os.getpid()} t\n")
    out = []
    assert W.watch(tmp_path, timeout=5, poll=0.01, out=out.append, stale=lambda: "DASHBOARD STALE: x") == W.EVENT
    assert out == ["DASHBOARD STALE: x"]


def test_publish_time_lives_in_the_run_dir(tmp_path):
    """F74: a commit's pre-commit stash hid the uncommitted publish time in links.yaml (false DASHBOARD STALE)."""
    bdir, rdir = tmp_path / "b", tmp_path / "r"
    (bdir / "dashboard").mkdir(parents=True)
    rdir.mkdir()
    (bdir / "dashboard" / "index.html").write_text("x")
    LK.save(bdir, {"links": {"standard": "https://claude.ai/artifact/A"}})
    assert LK.stale(bdir, run_dir=rdir, now=time.time() + 3600).startswith("DASHBOARD STALE")
    LK.mark_published(rdir)
    assert LK.stale(bdir, run_dir=rdir) is None and "published_at" not in LK.load(bdir)


def test_git_commit_waits_out_an_index_lock(tmp_path):
    import subprocess
    import threading

    from models.demos.common.bringup.core import gate as G

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "f").write_text("x")
    lock = tmp_path / ".git" / "index.lock"
    lock.write_text("")
    threading.Timer(1.0, lock.unlink).start()  # another process releases its lock a second later
    r = G._git_retry(["git", "add", "f"], tmp_path, tries=10, wait=0.3)
    assert r.returncode == 0


def test_a_red_export_is_reported_before_publishing(tmp_path):
    """F88: the export's own check (CHECK.txt: passed tasks the page shows red) wakes the supervisor's watch."""
    d = tmp_path / "dashboard"
    d.mkdir()
    (d / "index.html").write_text("x")
    LK.save(tmp_path, {"links": {"standard": "https://claude.ai/artifact/A"}, "published_at": time.time()})
    (d / "CHECK.txt").write_text("C.1: err_x_L* '<= limit' is not a number\n")
    line = LK.stale(tmp_path)
    assert line.startswith("DASHBOARD RED: 1 passed tasks") and "C.1" in line
    (d / "CHECK.txt").unlink()
    assert LK.stale(tmp_path) is None
