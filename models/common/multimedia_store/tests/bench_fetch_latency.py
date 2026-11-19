# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What fetch_batch costs for one MAX_PIXELS image, against a real master.

Reports latency rather than asserting it, so it is kept out of pytest's default discovery: the file is
bench_*, not test_*, and CI never collects it. Run it explicitly:

    pytest -s models/common/multimedia_store/tests/bench_fetch_latency.py

It reuses the integration test's cluster, so the same master, segments and writer apply. Both branches
of fetch_batch are timed, because only one of them is a view:

- this host's segment, returned in place (_view_at) — the case the prefill runner hopes for;
- another host's segment, which copies through the read buffer (_read_remote).

Both segments are loopback addresses on one machine, so the remote figure is a floor for a real
cross-host read, not an estimate of one.
"""

import statistics
import time

import numpy as np

from models.common.multimedia_store.reader import FetchRequest
from models.common.multimedia_store.tests.test_mooncake_store_integration import IMAGE_BYTES, cluster  # noqa: F401

REPS = 50


def _time_fetches(reader, media_key, reps=REPS):
    """Milliseconds per fetch_batch of one whole value, warm. The first is dropped: it pays the
    master's first lookup for this key."""
    reader.fetch_batch([FetchRequest(media_key)])
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        got = reader.fetch_batch([FetchRequest(media_key)])
        out.append((time.perf_counter() - t0) * 1e3)
        assert got[0].nbytes == IMAGE_BYTES
    return out


def test_fetch_batch_latency(cluster):  # noqa: F811
    rows = []
    for where, segment in (("this host", cluster.store.segment), ("another host", cluster.other.segment)):
        media_key, value = cluster.put(segment)
        assert cluster.placed_on(media_key) == segment
        ms = _time_fetches(cluster.store, media_key)
        got = cluster.store.fetch_batch([FetchRequest(media_key)])[0]
        assert bytes(got) == value.tobytes(), where
        in_place = np.frombuffer(got, np.uint8).ctypes.data == cluster.address(media_key)
        rows.append((where, in_place, ms))

    print(f"\n  one {IMAGE_BYTES / 1e6:.1f} MB image, {REPS} warm fetches each")
    for where, in_place, ms in rows:
        print(
            f"  {where:13s} {'in place' if in_place else 'copied  '}  "
            f"median {statistics.median(ms):7.3f} ms   min {min(ms):7.3f}   max {max(ms):7.3f}"
        )
