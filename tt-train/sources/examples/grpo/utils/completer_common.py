# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the GRPO completers (Llama and Qwen3).

Only byte-identical, completer-agnostic utilities live here. Per-completer
knobs (e.g. the ``CHUNK`` decode-readback cadence) intentionally stay in each
completer module so they can diverge independently.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any, List, Sequence, Tuple

import numpy as np
import ttnn

import ttml

SAMPLE_PROFILE_ENV = "GRPO_SAMPLE_PROFILE"


def sample_profiling_enabled() -> bool:
    return os.environ.get(SAMPLE_PROFILE_ENV, "0") == "1"


@contextlib.contextmanager
def profile_sample(mesh_device: Any, label: str):
    """Time one ``sample_op`` call and capture the device memory it peaks at.

    OFF unless ``GRPO_SAMPLE_PROFILE=1``, because measuring costs more than the op does:

      * Dispatch is ASYNCHRONOUS. Without a synchronize on both sides the timer measures how
        long it took to enqueue the program, which is roughly constant and says nothing about
        the kernel. The two syncs make the number real, and simultaneously serialize a decode
        loop that is otherwise pipelined -- so the surrounding tok/s figures degrade while
        this is on. Read the per-call number, not the throughput, when profiling.
      * The graph capture behind the memory numbers traces every allocation in the region.

    ``peak_dram`` is the high-water mark of DRAM allocated during the call, and ``peak_l1``
    the per-core L1 total (CBs plus program-scope buffers) -- for the fused sampler that is
    the handful of streamed tiles, not anything proportional to the vocabulary.
    """
    if not sample_profiling_enabled():
        yield
        return

    tracker = ttml.core.utils.MemoryUsageTracker
    ttnn.synchronize_device(mesh_device)
    guard = tracker.begin_capture()
    started = time.perf_counter()
    try:
        yield
    finally:
        # Sync BEFORE stopping the clock: the op is only enqueued when the body returns.
        ttnn.synchronize_device(mesh_device)
        elapsed_ms = (time.perf_counter() - started) * 1e3
        tracker.end_capture(label)
        try:
            dram = tracker.get_dram_usage(label)
            l1 = tracker.get_l1_usage(label)
            # print, not logging: logging writes to stderr, which the sbatch scripts route to the
            # .err file, while stdout goes to .out alongside the rest of the run. flush because a
            # redirected stdout is block-buffered, so without it these lines land late (or not at
            # all, if the job is killed).
            print(
                f"[sample] {label:<16} {elapsed_ms:8.3f} ms"
                f" | peak DRAM {dram.peak / 2**20:9.3f} MB"
                f" (alloc {dram.total_allocations / 2**20:.3f} MB,"
                f" free {dram.total_deallocations / 2**20:.3f} MB)"
                f" | peak L1/core {l1.peak_total / 2**10:8.3f} KB"
                f" (cb {l1.peak_cb / 2**10:.3f} KB)",
                flush=True,
            )
        finally:
            tracker.clear()
            guard.release()


def positions_to_tensor(positions: Sequence[int], B: int, tokens: int, dp_mapper: Any) -> ttml.autograd.Tensor:
    """Per-row sample positions for ``sample_op`` as [B, 1, 1, 1] UINT32.

    ``dp_mapper`` must be the SAME mapper the batch was sharded with (``None`` when the batch is
    replicated). That is what makes the shard landing on each device BE that device's rows -- true by
    construction, rather than by two separately-written mapper configs happening to agree.

    The op cannot range-check these: they live in device memory, and reading them back would mean a
    blocking sync on the dispatch path. So the loud check lives here, once, against the global list
    while the values are still on the host. The kernel clamps the tile row as a backstop, but a
    clamped position is silently the wrong row rather than an error -- which is why this assert
    matters more than it looks.
    """
    positions = [int(p) for p in positions]
    assert len(positions) == B, f"expected {B} positions, got {len(positions)}"
    bad = [(b, p) for b, p in enumerate(positions) if not 0 <= p < tokens]
    assert not bad, f"positions outside [0, {tokens}): {bad[:8]}"
    return ttml.autograd.Tensor.from_numpy(
        np.asarray(positions, dtype=np.uint32).reshape(B, 1, 1, 1),
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.UINT32,
        dp_mapper,
    )


def deallocate_tensors(tensors: Any) -> None:
    if tensors is None:
        return
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    for t in tensors:
        if t is None:
            continue
        if isinstance(t, ttml.autograd.Tensor):
            ttnn.deallocate(t.get_value(), force=True)
        elif isinstance(t, ttnn.Tensor):
            ttnn.deallocate(t, force=True)


def async_read_to_host(tensors: List[Any], mesh_device: Any) -> Tuple[List[Any], Any]:
    """Issue non-blocking d2h reads for ``tensors`` on the single command queue.

    Returns ``(host_tensors, event)``. The caller must call
    ``event_synchronize(event)`` before consuming ``host_tensors``; deallocating
    the source ``tensors`` before then races with the in-flight DMA.
    """
    hosts = [t.cpu(blocking=False) for t in tensors]
    done = ttnn.record_event(mesh_device=mesh_device, cq_id=0)
    return hosts, done
