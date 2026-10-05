# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host->device copies that can be RECORDED in a worker thread: the per-chunk host work of the traced prefill (building the host tensors) runs ahead
in the prefetch thread, the main thread then only replays the recorded (host, device) copies. Outside ``recording()`` ``h2d`` copies directly.
"""

import threading
from contextlib import contextmanager

import ttnn

_tl = threading.local()


def h2d(host, dev):
    rec = getattr(_tl, "ops", None)
    if rec is None:
        ttnn.copy_host_to_device_tensor(host, dev)
    else:
        rec.append((host, dev))


@contextmanager
def recording():
    ops = []
    _tl.ops = ops
    try:
        yield ops
    finally:
        _tl.ops = None


def replay(ops):
    for host, dev in ops:
        ttnn.copy_host_to_device_tensor(host, dev)
