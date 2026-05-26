# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Worker-side companion to ``helpers.barrier_scheduler``.

The master scheduler hands each xdist worker the *complete* scope-ordered
list of its tests in one shot (plus a ``SHUTDOWN`` sentinel). All
synchronisation between files happens here on the workers, using the shared
filesystem as the coordination medium:

1. Every worker computes the ordered list of scopes (file paths) from its
   collection during ``pytest_collection_modifyitems``.
2. Before a test runs (``pytest_runtest_setup``), the worker checks whether
   its scope has advanced past the one we are currently "in". If so, it
   crosses the FS barriers for every intermediate scope, in order.
3. At session finish, the worker crosses every barrier it has not yet
   crossed. This is the safety net for workers whose round-robin slice
   happened to end before the last scope -- without it, peers that are still
   marching through later scopes would wait at those scopes' barriers
   forever.

Crossing a single barrier:

* The worker drops an ``arrived.<worker_id>`` sentinel into
  ``<barrier_dir>/<barrier_name>/``.
* It then polls the directory until ``num_workers`` arrival sentinels exist
  (or a timeout fires, in which case the run is aborted -- better than
  silently hanging).
* All workers race for ``<barrier_name>.reset.lock`` (``filelock.FileLock``).
  Whoever wins checks ``<barrier_name>.cleared`` -- if it doesn't exist yet,
  it runs ``HardwareController().reset_card()`` and creates it. Everyone
  else just observes the cleared sentinel and moves on.
* Finally, each worker drops the process-local TestConfig caches that mirror
  on-card state (``BRISC_ELF_LOADED`` / ``LAST_LOADED_ELFS``); after a card
  reset those caches are stale and would cause the next test to hang trying
  to use an ELF that is no longer on the device.

Why setup-time instead of teardown-time:

Doing the barrier in ``pytest_runtest_setup`` of the *first* test of a new
scope (rather than ``pytest_runtest_teardown`` of the *last* test of the old
scope) ensures the previous test has already reported back to the master.
That keeps master's view of "test N completed" perfectly accurate even
though workers may then idle for seconds at the barrier.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import pytest
from filelock import FileLock

# Maximum time a worker will wait at a single barrier before declaring the
# run hung. 30 minutes is generous; in practice barriers should clear in
# seconds (the slowest worker's last test + one tt-smi -r).
_BARRIER_TIMEOUT_SECONDS = 30 * 60

# How long to sleep between polls of the barrier directory. Small enough to
# keep the latency tight, large enough that 8 workers spinning on the same
# directory don't burn a meaningful amount of CPU.
_BARRIER_POLL_INTERVAL_SECONDS = 0.1


class WorkerBarrierPlugin:
    """xdist-worker plugin enforcing a per-scope barrier between test files."""

    def __init__(
        self,
        worker_id: str,
        num_workers: int,
        barrier_dir: Path,
        on_barrier: Callable[[str], None],
    ) -> None:
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.barrier_dir = Path(barrier_dir)
        self.barrier_dir.mkdir(parents=True, exist_ok=True)
        self._on_barrier = on_barrier
        self._ordered_scopes: Optional[list[str]] = None
        # Index of the scope we are currently "inside". Starts at 0 (the
        # scope of the very first test in the worker's collection).
        self._current_scope_idx = 0

    @staticmethod
    def _scope_of(nodeid: str) -> str:
        return nodeid.split("::", 1)[0]

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        # All workers see the same items in the same order (xdist enforces
        # this by comparing collections). Walking that order, in-position,
        # gives us an unambiguous "scope N comes before scope N+1" ranking
        # to drive barrier crossings.
        seen: list[str] = []
        for item in items:
            scope = self._scope_of(item.nodeid)
            if not seen or seen[-1] != scope:
                seen.append(scope)
        self._ordered_scopes = seen

    def pytest_runtest_setup(self, item):
        if not self._ordered_scopes:
            return
        scope = self._scope_of(item.nodeid)
        try:
            target_idx = self._ordered_scopes.index(scope)
        except ValueError:
            # Defensive: an item with a scope nobody else collected. Skip
            # the barrier handling for it rather than crashing the run.
            return
        while self._current_scope_idx < target_idx:
            self._cross_barrier(self._current_scope_idx)
            self._current_scope_idx += 1

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        if not self._ordered_scopes:
            return
        # If this worker's last test landed before the final scope, cross
        # every remaining barrier so peers that are still running tests in
        # those later scopes don't stall on a missing arrival sentinel.
        # The "len - 1" is intentional: there is no barrier *after* the
        # final scope, only between consecutive scopes.
        while self._current_scope_idx < len(self._ordered_scopes) - 1:
            self._cross_barrier(self._current_scope_idx)
            self._current_scope_idx += 1

    def _cross_barrier(self, scope_idx: int) -> None:
        """Block until every worker has arrived; one of us runs the reset."""
        # Use the scope index in the barrier name -- not the file path --
        # so the keys are short and never collide with filesystem-unsafe
        # characters. The whole barrier directory is namespaced per run, so
        # there is no risk of collisions across runs either.
        barrier_dir = self.barrier_dir / f"barrier_{scope_idx:04d}"
        barrier_dir.mkdir(parents=True, exist_ok=True)

        arrived_file = barrier_dir / f"arrived.{self.worker_id}"
        cleared_file = barrier_dir / "cleared"
        lock_file = barrier_dir / "reset.lock"

        arrived_file.touch()

        # Phase 1: wait for everyone to arrive. We deliberately poll for the
        # count of arrival sentinels (not the cleared sentinel) so that the
        # tt-smi -r runs only after every worker is provably idle at this
        # barrier. Doing the reset earlier would race with a worker still
        # touching the card.
        start = time.time()
        while True:
            arrived_count = sum(1 for _ in barrier_dir.glob("arrived.*"))
            if arrived_count >= self.num_workers:
                break
            if time.time() - start > _BARRIER_TIMEOUT_SECONDS:
                raise RuntimeError(
                    f"[{self.worker_id}] Timeout ({_BARRIER_TIMEOUT_SECONDS}s) "
                    f"waiting at barrier {scope_idx}: only {arrived_count}/"
                    f"{self.num_workers} workers arrived. Files: "
                    f"{sorted(p.name for p in barrier_dir.glob('arrived.*'))}"
                )
            time.sleep(_BARRIER_POLL_INTERVAL_SECONDS)

        # Phase 2: serialize on the lock. The first holder runs the reset
        # and creates the cleared sentinel; later holders see it exists and
        # skip. Cheap and idempotent.
        with FileLock(str(lock_file)):
            if not cleared_file.exists():
                try:
                    self._on_barrier(
                        self._ordered_scopes[scope_idx]
                        if self._ordered_scopes
                        else f"scope_{scope_idx}"
                    )
                finally:
                    # Always mark cleared, even if reset_card raised, so we
                    # don't deadlock other workers on a transient failure.
                    # The exception still propagates up to fail the test.
                    cleared_file.touch()

        # Phase 3: drop per-process state that mirrors on-card state. After
        # tt-smi -r nothing is loaded on the card anymore, AND the Python-side
        # BRISC command counter must be re-synced to BRISC's local counter:
        # BRISC firmware (tests/helpers/src/brisc.cpp) keeps its own local
        # counter that restarts at 0 every time the firmware boots and writes
        # it into the brisc_counter L1 mailbox after each command. Meanwhile
        # device.commit_brisc_command marches a process-global common_counter
        # that grows for every test the worker ever ran. After a reset, the
        # next test reloads brisc.elf and BRISC begins writing 1, 2, 3, ...
        # If we leave common_counter at e.g. 209, commit_brisc_command waits
        # for the mailbox to reach 210 and never sees it -- the "Polling
        # brisc command timed out | Python counter: N | Brisc Counter: 1..N"
        # failures are exactly this race.
        from helpers import device as device_module
        from helpers.test_config import TestConfig

        TestConfig.BRISC_ELF_LOADED = False
        TestConfig.LAST_LOADED_ELFS = Path()
        device_module.common_counter = 0
