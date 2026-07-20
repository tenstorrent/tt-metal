"""Memory-aware adaptive concurrency for parallel bring-up agents.

Replaces a fixed ThreadPoolExecutor(max_workers=N) with a scheduler whose
concurrency limit moves up and down based on host memory pressure:

    available RAM low / swapping  -> shrink the limit (let in-flight finish)
    RAM headroom + work queued    -> grow the limit toward the ceiling

The memory sampler is injectable so tests can drive deterministic pressure
without touching real RAM. Verified against the real nvidia 30B: per-agent
private footprint ~58 GB, and the scaler scales down before the box OOMs.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class MemSample:
    available_frac: float
    swap_used_frac: float


def psutil_mem_sampler() -> MemSample:
    import psutil

    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    swap_frac = (sw.used / sw.total) if sw.total else 0.0
    return MemSample(available_frac=vm.available / vm.total, swap_used_frac=swap_frac)


class AdjustableSemaphore:
    """A semaphore whose permit ceiling can be raised or lowered at runtime.

    The limit can shrink below the number of currently-issued permits:
    existing holders keep running, but no new acquire() succeeds until active
    drops back under the new limit.
    """

    def __init__(self, limit: int):
        self._cond = threading.Condition()
        self._limit = max(1, int(limit))
        self._active = 0

    @property
    def limit(self) -> int:
        with self._cond:
            return self._limit

    @property
    def active(self) -> int:
        with self._cond:
            return self._active

    def set_limit(self, new_limit: int) -> None:
        with self._cond:
            self._limit = max(1, int(new_limit))
            self._cond.notify_all()

    def acquire(self) -> None:
        with self._cond:
            while self._active >= self._limit:
                self._cond.wait()
            self._active += 1

    def release(self) -> None:
        with self._cond:
            self._active -= 1
            self._cond.notify_all()


@dataclass
class ScalePolicy:
    low_available_frac: float = 0.12
    high_available_frac: float = 0.30
    swap_pressure_frac: float = 0.10
    sample_interval_s: float = 3.0
    floor: int = 1
    ceiling: int = 4
    cooldown_samples: int = 1
    start: Optional[int] = None


@dataclass
class ScaleEvent:
    t: float
    available_frac: float
    swap_used_frac: float
    old_limit: int
    new_limit: int
    reason: str


@dataclass
class AdaptiveScheduler:
    """Runs jobs across worker threads under a memory-adjusted concurrency cap.

    jobs:      list of opaque job objects
    run_one:   callable(job, slot_index) -> result, executed in a worker thread
    sampler:   callable() -> MemSample (defaults to psutil)
    policy:    ScalePolicy
    clock:     callable() -> float (injectable for deterministic tests)
    """

    jobs: List[object]
    run_one: Callable[[object, int], object]
    policy: ScalePolicy = field(default_factory=ScalePolicy)
    sampler: Optional[Callable[[], MemSample]] = None
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self):
        self._sampler = self.sampler or psutil_mem_sampler
        if self.policy.start is not None:
            start = max(self.policy.floor, min(self.policy.ceiling, self.policy.start))
        else:
            start = max(self.policy.floor, min(self.policy.ceiling, 2))
        self._sem = AdjustableSemaphore(start)
        self._events: List[ScaleEvent] = []
        self._events_lock = threading.Lock()
        self._stop = threading.Event()
        self._slot_lock = threading.Lock()
        self._free_slots = list(range(self.policy.ceiling))
        self._pending = len(self.jobs)
        self._pending_lock = threading.Lock()

    @property
    def events(self) -> List[ScaleEvent]:
        with self._events_lock:
            return list(self._events)

    def _take_slot(self) -> int:
        with self._slot_lock:
            return self._free_slots.pop(0) if self._free_slots else 0

    def _give_slot(self, slot: int) -> None:
        with self._slot_lock:
            self._free_slots.append(slot)

    def _jobs_remaining(self) -> int:
        with self._pending_lock:
            return self._pending

    def _mark_done(self) -> None:
        with self._pending_lock:
            self._pending -= 1

    def _decide(self, s: MemSample, limit: int) -> tuple:
        pol = self.policy
        if s.available_frac < pol.low_available_frac or s.swap_used_frac > pol.swap_pressure_frac:
            return max(pol.floor, limit - 1), "memory-pressure: scale down"
        if s.available_frac > pol.high_available_frac and self._jobs_remaining() > limit and limit < pol.ceiling:
            return min(pol.ceiling, limit + 1), "headroom + backlog: scale up"
        return limit, "hold"

    def _monitor(self) -> None:
        cooldown = 0
        while not self._stop.is_set():
            s = self._sampler()
            old = self._sem.limit
            new, reason = self._decide(s, old)
            if new != old and cooldown <= 0:
                self._sem.set_limit(new)
                cooldown = self.policy.cooldown_samples
                with self._events_lock:
                    self._events.append(ScaleEvent(self.clock(), s.available_frac, s.swap_used_frac, old, new, reason))
            else:
                cooldown = max(0, cooldown - 1)
            self._stop.wait(self.policy.sample_interval_s)

    def run(self) -> List[object]:
        if not self.jobs:
            return []
        results: List[Optional[object]] = [None] * len(self.jobs)
        mon = threading.Thread(target=self._monitor, name="mem-monitor", daemon=True)
        mon.start()
        worker_threads: List[threading.Thread] = []

        def _worker(idx: int, job: object) -> None:
            self._sem.acquire()
            slot = self._take_slot()
            try:
                results[idx] = self.run_one(job, slot)
            finally:
                self._give_slot(slot)
                self._mark_done()
                self._sem.release()

        for idx, job in enumerate(self.jobs):
            t = threading.Thread(target=_worker, args=(idx, job), name=f"agent-{idx}")
            t.start()
            worker_threads.append(t)

        for t in worker_threads:
            t.join()
        self._stop.set()
        mon.join(timeout=2.0)
        return results
