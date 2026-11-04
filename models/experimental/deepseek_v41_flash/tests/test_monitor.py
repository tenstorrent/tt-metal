# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The web monitor (demo/monitor.py) against scripted feed snapshots and counters. Host only, no device."""

import json
import mmap
import os
import struct
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from models.experimental.deepseek_v41_flash.demo import monitor


def _write(path, data, stamp):
    with open(path + ".tmp", "w") as f:
        json.dump(data, f)
    os.replace(path + ".tmp", path)
    os.utime(path, ns=(stamp, stamp))  # distinct mtimes, however fast the writes


def _counters(path, **values):
    words = {name: 0 for name in monitor.COUNTER_FIELDS}
    words.update(magic=monitor.COUNTERS_MAGIC, version=monitor.COUNTERS_VERSION, pid=4242, **values)
    data = struct.pack(f"<{len(words)}Q", *(words[name] for name in monitor.COUNTER_FIELDS))
    if os.path.exists(path):  # in place, as the runtime updates it
        with open(path, "r+b") as f, mmap.mmap(f.fileno(), len(data)) as m:
            m[:] = data
    else:
        with open(path, "wb") as f:
            f.write(data)


def _turn(serial, user, slot, phase, prompt, prefilled, generated):
    return {
        "serial": serial,
        "user": user,
        "slot": slot,
        "phase": phase,
        "prompt_tokens": prompt,
        "cached_tokens": 0,
        "prefilled": prefilled,
        "generated": generated,
        "max_tokens": 1000,
        "age_s": 1.0,
        "cancelled": False,
    }


def _snapshot(t, turns, ended=(), log=(), started=1000):
    return {
        "t": t,
        "model_id": "deepseek-v4.1-flash",
        "started": started,
        "max_seq": 4096,
        "slots": 4,
        "slot_states": {"running": len(turns), "queued": 0, "parked": 1, "empty": 1, "max_running": 8},
        "users": [],
        "turns": list(turns),
        "inflight": 3,
        "pool": {"compress": [10, 100], "sliding": [30, 60]},
        "tokens_left": 5000,
        "host": {"users": 0, "bytes": 0, "max_users": 0, "max_bytes": 0},
        "broken": None,
        "ended_turns": list(ended),
        "log": [list(line) for line in log],
    }


def test_monitor_rates_totals_and_history(tmp_path):
    prefix = str(tmp_path / "feed")
    path, cpath = monitor.snapshot_path(prefix), monitor.counters_path(prefix)
    mon = monitor.Monitor(prefix, keep_seconds=3600)

    _counters(cpath, outputs_read=0, dispatch_count=10, dispatch_ns=10_000)
    _write(
        path,
        _snapshot(
            100.0,
            [_turn(1, "alice", 0, "decode", 50, 50, 10), _turn(2, "bob", 1, "prefill", 1000, 100, 0)],
            log=[(1, 100.0, "INFO", "hello")],
        ),
        1,
    )
    mon.poll()
    _counters(cpath, outputs_read=50, dispatch_count=20, dispatch_ns=30_000)
    _write(
        path,
        _snapshot(
            101.0,
            [_turn(1, "alice", 0, "decode", 50, 50, 30), _turn(2, "bob", 1, "prefill", 1000, 600, 0)],
            log=[(1, 100.0, "INFO", "hello"), (2, 101.0, "DEBUG", "round")],
        ),
        2,
    )
    mon.poll()
    point = mon.series[-1]
    assert point["decode"] == pytest.approx(20.0)
    assert point["prefill"] == pytest.approx(500.0)
    assert point["users"] == {"alice": pytest.approx(20.0)}
    assert point["timers"]["dispatch"] == pytest.approx(2.0)
    assert point["steps"] > 0
    assert point["kv"] == {"compress": pytest.approx(0.1), "sliding": pytest.approx(0.5)}

    # alice ends between the snapshots, and this one caught her both ended and still among the active turns.
    record = {
        "serial": 1, "user": "alice", "slot": 0, "outcome": "stop", "ended": 101.9, "queued_s": 0.1,
        "prompt_tokens": 50, "cached_tokens": 0, "prefilled": 50, "prefilled_on_device": 0, "generated": 40,
        "prefill_s": 0.5, "decode_s": 2.0, "total_turns": 1, "total_prefilled": 50, "total_generated": 40,
    }  # fmt: skip
    _write(
        path,
        _snapshot(
            102.0,
            [_turn(1, "alice", 0, "decode", 50, 50, 40), _turn(2, "bob", 1, "decode", 1000, 1000, 5)],
            ended=[record],
            log=[(2, 101.0, "DEBUG", "round"), (3, 102.0, "INFO", "done")],
        ),
        3,
    )
    mon.poll()
    point = mon.series[-1]
    assert point["decode"] == pytest.approx(15.0)  # (40 + 5) - 30
    assert point["prefill"] == pytest.approx(400.0)  # (50 + 1000) - (50 + 600)
    assert point["users"] == {"alice": pytest.approx(10.0), "bob": pytest.approx(5.0)}
    view = mon.view(0, 0, 0, 0)
    assert view["state"]["totals"] == {"generated": 45, "prefilled": 1050, "turns": 1}
    assert [turn["serial"] for turn in view["state"]["turns"]] == [2]
    assert [line[0] for line in view["log"]] == [1, 2, 3]
    (ended,) = view["history"]
    assert ended["decode_tps"] == pytest.approx(20.0) and ended["prefill_tps"] == pytest.approx(100.0)
    assert ended["ttft_s"] == pytest.approx(0.6)
    assert len(mon.view(101.5, 2, 101.95, 0)["series"]) == 1

    # Images: the server keeps only its last few done, the monitor all it has seen.
    images = {"rows_free": 900, "rows_total": 1024, "held": 2, "encoded": 1, "files": {"count": 0, "bytes": 0},
              "live": [{"label": "abc", "stage": "vit", "rows": 300, "seconds": 0.2}],
              "done": [{"label": "d1", "at": 102.5, "rows": 100, "decode": 0.1, "vit": 0.3, "embedding": 0.05}]}  # fmt: skip
    for t, stamp, done in (
        (103.0, 4, images["done"]),
        (104.0, 5, images["done"] + [{**images["done"][0], "at": 103.5}]),
    ):
        _write(
            path,
            {**_snapshot(t, [_turn(2, "bob", 1, "decode", 1000, 1000, 5)]), "images": {**images, "done": done}},
            stamp,
        )
        mon.poll()
    assert [e["at"] for e in mon.view(0, 0, 0, 0)["images_done"]] == [102.5, 103.5]
    assert [e["at"] for e in mon.view(0, 0, 0, 103.0)["images_done"]] == [103.5]
    assert mon.series[-1]["images_live"] == 1 and mon.series[-1]["image_rows_free"] == 900

    # A restarted server starts the monitor over.
    _write(path, _snapshot(200.0, [], started=2000), 4)
    mon.poll()
    assert len(mon.series) == 1 and not mon.history and not mon.log and not mon.images_done


def test_feed_writes_snapshots_and_the_page_serves(tmp_path):
    prefix = str(tmp_path / "feed")
    feed = monitor.MonitorFeed(lambda: _snapshot(100.0, []), monitor.snapshot_path(prefix), period=0.01)
    feed.start()
    try:
        for _ in range(500):
            if os.path.exists(feed.path):
                break
            threading.Event().wait(0.01)
    finally:
        feed.stop()
    mon = monitor.Monitor(prefix, keep_seconds=60)
    mon.poll()
    assert mon.state["model_id"] == "deepseek-v4.1-flash" and mon.state["counters"] is None

    monitor._Handler.monitor = mon
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), monitor._Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "/api/state" in page
        view = json.loads(urllib.request.urlopen(base + "/api/state?since=0").read())
        assert view["state"]["tokens_left"] == 5000 and len(view["series"]) == 1
    finally:
        httpd.shutdown()
