# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Progress files the supervisor watches, the done-log it resumes from, and the
per-case result log the junit report is built out of.

A wedged worker is blocked inside a device read. It cannot raise, log, or run a
signal handler, so there is nothing we could ask it to tell us. The only usable
signal is one it stops producing: every worker rewrites a small file before each
variant, and `supervise.py` calls a wedge when every live worker's file has
stopped moving.

Writes are atomic (temp + rename) so a reader never sees half a record. Both
sides no-op when TTNOP_STATE_DIR is unset, which is how focus.sh and a bare
ci.sh run stay unaffected.
"""

import json
import os
import time
from pathlib import Path

STATE_DIR_ENV = "TTNOP_STATE_DIR"
# Mid-case, so a stalled clock means the device stopped answering this worker.
ALIVE = "alive"
# Between cases, holding nothing on the device. Distinct from ALIVE because the
# supervisor calls a wedge on one silent worker now: at the tail of a sweep the
# queue drains unevenly and a worker can sit here for a long time with nothing
# wrong, which under a plain age check would trip a wedge on every single run.
IDLE = "idle"
DONE = "done"

_HEARTBEAT_PREFIX = "hb."
_DONE_PREFIX = "done."
_RESULTS_PREFIX = "results."

# Enough of a failure to identify it in a report without carrying a whole tensor
# dump per case into the junit XML.
_MESSAGE_LIMIT = 2000


def state_dir():
    """Where to keep progress files, or None when nothing is supervising us."""
    root = os.environ.get(STATE_DIR_ENV, "").strip()
    return Path(root) if root else None


def worker_id() -> str:
    """xdist worker name, or "master" when running without -n."""
    return os.environ.get("PYTEST_XDIST_WORKER", "master")


# -- writer side (the pytest workers) --------------------------------------


class Writer:
    """One worker's view of its own progress files."""

    def __init__(self):
        self.root = state_dir()
        self.worker = worker_id()
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
        self._path = (
            None
            if self.root is None
            else self.root / f"{_HEARTBEAT_PREFIX}{self.worker}"
        )
        self._done = (
            None if self.root is None else self.root / f"{_DONE_PREFIX}{self.worker}"
        )
        self._results = (
            None if self.root is None else self.root / f"{_RESULTS_PREFIX}{self.worker}"
        )

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def beat(self, case: str = "", variant: dict = None, status: str = ALIVE) -> None:
        """Publish what this worker is about to do. Cheap enough to call per variant."""
        if self._path is None:
            return
        payload = {
            "worker": self.worker,
            "status": status,
            "ts": time.time(),
            "pid": os.getpid(),
            "case": case,
            "variant": variant or {},
        }
        # Rename onto the real name so the supervisor only ever sees whole records.
        temp = self._path.parent / f"{self._path.name}.{os.getpid()}.tmp"
        with open(temp, "w") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        os.replace(temp, self._path)

    def idle(self) -> None:
        """Declare nothing in flight, so the silence that follows is not a wedge."""
        self.beat(status=IDLE)

    def finish(self) -> None:
        """Stop counting as live, so a finished worker is not mistaken for a wedged one."""
        self.beat(status=DONE)

    def mark_done(self, nodeid: str) -> None:
        """Record a case we never need to run again, so a reset can resume past it."""
        if self._done is None:
            return
        with open(self._done, "a") as handle:
            handle.write(nodeid + "\n")

    def record_result(
        self, nodeid: str, outcome: str, duration: float = 0.0, message: str = ""
    ) -> None:
        """Log how a case ended, one line at a time, for the junit report.

        Appended as each case finishes rather than left to pytest's own --junit-xml,
        which is only written at session end — a session the supervisor kills outright
        when a core wedges, taking every result in it down. See ttnop/junit.py.
        """
        if self._results is None:
            return
        payload = {
            "nodeid": nodeid,
            "outcome": outcome,
            "duration": round(float(duration or 0.0), 3),
            "message": (message or "")[:_MESSAGE_LIMIT],
        }
        with open(self._results, "a") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


# -- reader side (the supervisor) ------------------------------------------


def completed(root: Path) -> set:
    """Every case any worker finished, across all attempts so far."""
    done = set()
    for path in root.glob(f"{_DONE_PREFIX}*"):
        with open(path) as handle:
            done.update(line.strip() for line in handle if line.strip())
    return done


def results(root: Path) -> dict:
    """Every case outcome any worker logged, keyed by node id, across all attempts.

    Tolerates a truncated final line: the process that was appending may have been
    killed mid-write, which is precisely the run this log exists to survive.
    """
    found = {}
    for path in root.glob(f"{_RESULTS_PREFIX}*"):
        try:
            with open(path) as handle:
                lines = handle.readlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            nodeid = record.get("nodeid")
            if nodeid:
                found[nodeid] = record
    return found


def _read(path: Path):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _float(value) -> float:
    # A record is only ever read back while the process that wrote it may be
    # dying, and a bad timestamp here would crash the watchdog rather than the
    # run it is meant to be watching.
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def live_workers(root: Path) -> list:
    """Heartbeats of workers that have not declared themselves finished."""
    records = []
    for path in root.glob(f"{_HEARTBEAT_PREFIX}*"):
        if path.suffix == ".tmp":
            continue
        record = _read(path)
        if record is None or record.get("status") == DONE:
            continue
        record["ts"] = _float(record.get("ts"))
        record["age"] = max(0.0, time.time() - record["ts"])
        records.append(record)
    return records


def stalled(workers, timeout: float) -> list:
    """Workers that went silent mid-case. One of these is enough to call a wedge.

    Only ALIVE counts: an IDLE worker between cases holds nothing on the device,
    so its silence says nothing about whether the card is answering.
    """
    return [
        worker
        for worker in workers
        if worker.get("status") == ALIVE and worker["age"] >= timeout
    ]


def newest_beat(workers) -> float:
    """Most recent publish time across workers — the only proof the run is moving."""
    return max((worker["ts"] for worker in workers), default=0.0)


def record_skipped(root: Path, nodeids) -> None:
    """Treat cases as covered without having run them, so a resume steps over them."""
    wanted = [nodeid for nodeid in nodeids if nodeid]
    if not wanted:
        return
    with open(root / f"{_DONE_PREFIX}skipped", "a") as handle:
        handle.writelines(nodeid + "\n" for nodeid in wanted)


def clear_heartbeats(root: Path) -> None:
    """Drop stale heartbeats between attempts; the done-log is deliberately kept."""
    for path in root.glob(f"{_HEARTBEAT_PREFIX}*"):
        try:
            path.unlink()
        except OSError:
            pass
