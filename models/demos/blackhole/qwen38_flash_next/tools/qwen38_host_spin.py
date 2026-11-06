"""Busy-wait for a decode step's token read instead of a blocking wait (QWEN38_BATCH_SPIN, default "0").

A decode loop that sleeps in a blocking wait for ~40 ms per step lets the host CPU drop to its minimum clock under the
schedutil governor, and the host Python after the wait (PLE lookup, token upload, readback, bookkeeping) then runs ~2.4x
slower.  A spinning wait keeps the clock up.  The ttnn build has no event query (``ttnn.events`` offers record_event,
wait_for_event and event_synchronize only: ttnn/cpp/ttnn-nanobind/events.cpp:38,60,75), so the spin polls a non-blocking
device-to-host copy of the token row into a host tensor that holds NaN (the id is a finite integer), the way
``Qwen38TracedChain._read_token_row_spin`` does.  The copy sits behind the step on the same command queue, so the first
element turns finite only after the step finished on the polled device.  The caller then runs the blocking wait it ran
before, which now returns at once (and covers the other devices of the mesh).

With the flag off nothing here makes a ttnn call: the callers' calls are the ones they made before.
"""

from __future__ import annotations

import os
import time

import torch

import ttnn

SPIN_VARIABLE = "QWEN38_BATCH_SPIN"
# read once: a host tensor made as a spin read is polled only while the flag is on
SPIN = os.environ.get(SPIN_VARIABLE, "1") == "1"
SPIN_TIMEOUT_S = 60.0
TOKEN_ROW_SHAPE = (1, 1, 1, 32)

_sentinel: torch.Tensor | None = None


def enabled() -> bool:
    return SPIN


def _sentinel_row() -> torch.Tensor:
    global _sentinel
    if _sentinel is None:
        _sentinel = torch.full(TOKEN_ROW_SHAPE, float("nan"))
    return _sentinel


def start_read(shard):
    """A non-blocking copy of the device tensor ``shard`` (an FP32 TILE token row) into a fresh host tensor that holds
    NaN; poll it with :func:`wait_read`."""

    host = ttnn.from_torch(_sentinel_row(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    ttnn.copy_device_to_host_tensor(shard, host, blocking=False)
    return host


def wait_read(host, timeout_s: float = SPIN_TIMEOUT_S) -> int:
    """Spin until the first element of ``host`` (from :func:`start_read`) is no longer NaN.  A no-op with the flag off.
    Returns the poll count (0 when off)."""

    if not SPIN:
        return 0
    polls = 0
    deadline = time.perf_counter() + timeout_s
    while True:
        value = host.to_numpy().item(0)
        polls += 1
        if value == value:
            return polls
        if polls % 4096 == 0 and time.perf_counter() > deadline:
            raise TimeoutError(f"token row spin read: no value after {timeout_s:g} s")


def spin_until_done(mesh, shard, cq_id: int = 0) -> int:
    """Wait for everything enqueued on ``cq_id`` before this call, spinning: the non-blocking token-row read, the poll,
    then a full-mesh event synchronize that returns at once (the other devices of the mesh).  Returns the poll count."""

    host = start_read(shard)
    polls = wait_read(host)
    ttnn.event_synchronize(ttnn.record_event(mesh, cq_id=cq_id))
    return polls
