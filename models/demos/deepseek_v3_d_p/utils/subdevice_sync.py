# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Helpers for in-trace cross-sub-device ordering (GLM prefill pipeline E2).

Ordering facts (tt-metal dispatch, verified on BH Galaxy, see notes/pipeline/e2.md):
  * One CQ: go signals are released in ISSUE ORDER and a go for sub-device X waits until every earlier program on X
    has completed. Hence a program C issued after any program F on sub-device X can start only after everything on
    X issued before F is done. A cross-sub-device edge P (SD_p) -> C (SD_c) therefore holds iff some SD_p program is
    issued after P and before C. `fence()` issues such a program (one core) when the stream has none.
  * The wait half (`HandOff.wait`) adds no ordering inside one CQ, and it deadlocks unless the matching signal is
    issued before the next program on the waiter's sub-device after the wait. It is load-bearing across CQs
    (each CQ has its own dispatcher, e.g. CQ0 issues only SD0 programs and CQ1 only SD1 programs).
"""

from dataclasses import dataclass

import ttnn


def _crs(core):
    return ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})


class SignalPool:
    """n GlobalSemaphores on one consumer core, created once (before any trace capture) and reset to 0."""

    def __init__(self, mesh_device, consumer_core, n, initial_value=0):
        self.mesh_device = mesh_device
        self.consumer_core = consumer_core
        self.initial_value = initial_value
        self.sems = [ttnn.create_global_semaphore(mesh_device, _crs(consumer_core), initial_value) for _ in range(n)]

    def __len__(self):
        return len(self.sems)

    def __getitem__(self, i):
        return self.sems[i]

    def reset(self):
        """Host-side reset (never inside a capture). Needed only after an aborted / unbalanced run."""
        for s in self.sems:
            ttnn.reset_global_semaphore_value(s, self.initial_value)


@dataclass
class HandOff:
    """One cross-sub-device edge: signal() on the producer SD after the producer, wait() on the consumer SD before
    the consumer. Balanced pairs leave the semaphore at its initial value, so every replay starts clean."""

    sem: object
    producer_sd: object
    producer_core: object
    consumer_sd: object
    consumer_core: object
    count: int = 1

    def signal(self):
        ttnn.experimental.subdevice_signal(
            self.sem,
            subdevice_id=self.producer_sd,
            core=self.producer_core,
            consumer_core=self.consumer_core,
            increment=self.count,
        )

    def wait(self, timeout_cycles=0):
        ttnn.experimental.subdevice_wait(
            self.sem, subdevice_id=self.consumer_sd, core=self.consumer_core, value=self.count, timeout_cycles=timeout_cycles
        )


class Fence:
    """A one-core program on `subdevice_id` that does nothing but occupy its slot in the issue order: every program
    issued after it (on ANY sub-device of the same CQ) starts only after all earlier programs on `subdevice_id` are
    done. Implemented as a subdevice_signal to a private never-waited semaphore on the fence core itself."""

    def __init__(self, mesh_device, subdevice_id, core):
        self.subdevice_id = subdevice_id
        self.core = core
        self.sem = ttnn.create_global_semaphore(mesh_device, _crs(core), 0)

    def __call__(self):
        ttnn.experimental.subdevice_signal(self.sem, subdevice_id=self.subdevice_id, core=self.core, consumer_core=self.core)
