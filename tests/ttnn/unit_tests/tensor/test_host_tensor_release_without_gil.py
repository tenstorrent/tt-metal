# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Dropping the last reference to a large host ``ttnn.Tensor`` frees its memory without holding the GIL.

Freeing hundreds of MB of host memory takes tens of milliseconds. ttnn.Tensor's deallocator releases the GIL while a
large host tensor is destroyed, so a caller can drop such a tensor on a worker thread while the main thread keeps
running. Buffers borrowed from Python objects still drop those objects with the GIL.
"""

import gc
import threading
import time
import weakref

import numpy as np
import torch

import ttnn


def _drop_on_worker(holder):
    """Drop holder[0] on a worker thread. Returns (worst main-thread gap, worker drop time), in seconds."""
    drop_s = []

    def worker():
        tensor = holder.pop()
        start = time.perf_counter()
        del tensor
        drop_s.append(time.perf_counter() - start)

    thread = threading.Thread(target=worker)
    worst_gap = 0.0
    prev = time.perf_counter()
    thread.start()
    while thread.is_alive():
        now = time.perf_counter()
        worst_gap = max(worst_gap, now - prev)
        prev = now
    worst_gap = max(worst_gap, time.perf_counter() - prev)
    thread.join()
    return worst_gap, drop_s[0]


def test_dropping_a_large_host_tensor_on_a_worker_does_not_block_the_main_thread():
    # About 272 MB of bfloat8_b tiles: the tensor owns its converted buffer.
    host = ttnn.from_torch(
        torch.randn(1, 1, 2048, 128 * 1024, dtype=torch.bfloat16), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
    )
    holder = [host]
    del host
    gc.collect()
    worst_gap, drop = _drop_on_worker(holder)
    # The drop is measured on the worker; while it ran, the main thread must have kept the GIL turning over.
    assert worst_gap < max(
        0.005, drop / 4
    ), f"main thread stalled {worst_gap * 1e3:.2f} ms during a {drop * 1e3:.2f} ms drop"


def test_dropping_a_large_borrowed_host_tensor_on_a_worker_releases_its_python_owner():
    elements = 4 * 1024 * 1024  # 8 MB of bfloat16, above the size released without the GIL
    raw = torch.randn(elements).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint8)
    raw_ref = weakref.ref(raw)
    spec = ttnn.TensorSpec(
        ttnn.Shape([1, 1, 1024, 4096]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG
    )
    holder = [ttnn.host_tensor_from_bytes(raw, spec)]
    del raw
    gc.collect()
    assert raw_ref() is not None, "the tensor pins the buffer"

    _drop_on_worker(holder)
    gc.collect()
    assert raw_ref() is None, "the pinned buffer is released with the tensor"


def test_a_large_host_tensor_shared_by_a_copy_stays_alive():
    source = torch.randn(1, 1, 1024, 1024, dtype=torch.float32)
    tensor = ttnn.from_torch(source, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    alias = ttnn.reshape(tensor, tensor.shape)  # shares the host buffer
    expected = ttnn.to_torch(tensor)
    holder = [tensor]
    del tensor
    _drop_on_worker(holder)
    torch.testing.assert_close(ttnn.to_torch(alias), expected, rtol=0, atol=0)
