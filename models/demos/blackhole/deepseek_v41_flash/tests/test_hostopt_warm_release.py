# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU check: threads created while the round thread is pinned by DSV41_HOSTOPT_WARM inherit the one-core mask; ``hostopt._release_pinned_threads`` (called by ``warm_stop`` at the start of every
prefill call) gives them the full mask back and leaves the calling thread and threads with another mask alone."""

import os
import threading

import pytest

from models.demos.blackhole.deepseek_v41_flash.tt import hostopt


@pytest.mark.skipif(len(os.sched_getaffinity(0)) < 3, reason="needs at least 3 allowed cpus")
def test_release_pinned_threads():
    full = os.sched_getaffinity(0)
    cpu = sorted(full)[0]
    other = sorted(full)[1]
    saved = dict(hostopt._WARM)
    started, release, done = threading.Event(), threading.Event(), threading.Event()
    result = {}

    def child(name, own=None):
        if own is not None:
            os.sched_setaffinity(0, own)
        result[name] = threading.get_native_id()
        started.set()
        release.wait(30)

    try:
        hostopt._WARM.update(cpu=cpu, mask=set(full))
        os.sched_setaffinity(0, {cpu})  # the pinned round thread ...
        t1 = threading.Thread(target=child, args=("inherited",))  # ... a thread created now inherits the pin
        t1.start()
        started.wait(10)
        started.clear()
        t2 = threading.Thread(target=child, args=("own", {other}))  # a thread with its own mask
        t2.start()
        started.wait(10)
        assert os.sched_getaffinity(result["inherited"]) == {cpu}
        os.sched_setaffinity(0, set(full))  # (what warm_stop does for the calling thread)
        hostopt._release_pinned_threads()
        assert os.sched_getaffinity(result["inherited"]) == full
        assert os.sched_getaffinity(result["own"]) == {other}
    finally:
        release.set()
        os.sched_setaffinity(0, set(full))
        hostopt._WARM.clear()
        hostopt._WARM.update(saved)
