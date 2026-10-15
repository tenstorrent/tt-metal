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
