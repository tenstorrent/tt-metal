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
# One file, not one per worker: the supervisor only needs to be told once that a
# worker is stuck on a core it can no longer use, and a second worker hanging
# before it gets there is the same request.
_RECOVERY_REQUEST = "recovery-requested"

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

    def request_recovery(
        self, case: str, variant: str, skip_family: bool = True
    ) -> None:
        """Ask the supervisor to take this worker off the core it just hung.

        Eight workers share one card, so resetting from in here would take the
        other seven down mid-case; and it would not even fix this process, since
        the reset is only safe once the run it interrupts is dead. So the worker
        states the problem and the supervisor, which owns both the card and the
        run, decides how to act on it — normally by killing this worker so xdist
        replaces it on one of the card's spare cores, and only resetting the card
        when there are none of those left.

        skip_family is True for a hang: the rest of that test's params hit the
        same site and would cost another core each. A mismatch race dirties dest
        and semaphores the same way a hang does, but the siblings are a different
        window — pass False so they run on the spare instead of being skipped.
        """
        if self.root is None:
            return
        payload = {
            "worker": self.worker,
            "case": case,
            "variant": variant,
            "skip_family": skip_family,
            # Carried because killing this process is how the core is given up,
            # and the supervisor cannot look the pid up from a heartbeat: parking
            # publishes DONE, and a DONE worker is deliberately not in the live set.
            "pid": os.getpid(),
            "ts": time.time(),
        }
        # Same temp-and-rename as a beat: the supervisor polls this file and must
        # never catch a half-written record.
        temp = self.root / f"{_RECOVERY_REQUEST}.{os.getpid()}.tmp"
        with open(temp, "w") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        os.replace(temp, self.root / _RECOVERY_REQUEST)

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


def family_key(nodeid: str) -> str:
    """The test, without `[params]`. Sibling formats of one hang share this."""
    return nodeid.split("[", 1)[0]


def unrun_family(root: Path, hung: str, all_ids: list) -> list:
    """Unfinished cases of the same test as `hung`.

    One hang is the race. The other params hit the same site and need another
    `tt-smi -r` to clear. Other tests stay on the queue.
    """
    if not hung:
        return []
    key = family_key(hung)
    done = completed(root)
    return [
        nodeid for nodeid in all_ids if nodeid not in done and family_key(nodeid) == key
    ]


def record_skipped(root: Path, nodeids, reason: str = "") -> None:
    """Treat cases as covered without having run them, so a resume steps over them.

    `reason` writes a skipped result line so the cases show up in junit instead
    of vanishing. The silent-wedge path leaves it empty: those cases are
    reported as wedges, not skips.
    """
    wanted = [nodeid for nodeid in nodeids if nodeid]
    if not wanted:
        return
    with open(root / f"{_DONE_PREFIX}skipped", "a") as handle:
        handle.writelines(nodeid + "\n" for nodeid in wanted)
    if not reason:
        return
    with open(root / f"{_RESULTS_PREFIX}skipped", "a") as handle:
        for nodeid in wanted:
            handle.write(
                json.dumps(
                    {
                        "nodeid": nodeid,
                        "outcome": "skipped",
                        "duration": 0.0,
                        "message": reason[:_MESSAGE_LIMIT],
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )


def recovery_request(root: Path):
    """The worker asking to be taken off a hung core, or None if nobody has hung."""
    return _read(root / _RECOVERY_REQUEST)


def clear_recovery_request(root: Path) -> None:
    """Drop a request once it has been acted on, so one hang buys one recovery."""
    try:
        (root / _RECOVERY_REQUEST).unlink()
    except OSError:
        pass


def clear_heartbeats(root: Path) -> None:
    """Drop stale heartbeats between attempts; the done-log is deliberately kept."""
    for path in root.glob(f"{_HEARTBEAT_PREFIX}*"):
        try:
            path.unlink()
        except OSError:
            pass
