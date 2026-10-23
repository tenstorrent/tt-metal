# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Watch a TTNOP sweep, recover wedged workers, and resume unfinished cases.

Workers publish heartbeats before each variant. A silent worker is recorded and
evicted so xdist can use a spare core. If eviction or spare capacity fails, the
supervisor resets the card and resumes from the done log.

    python3 supervise.py IDS_FILE [pytest args...]
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from enum import IntEnum
from pathlib import Path

import heartbeat
import junit
import report
import sweep as sweep_module
from _pytest_runner import run as run_pytest

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "_pytest_runner.py"

# Includes the 60-second addr2line timeout on failing variants.
WEDGE_TIMEOUT = float(os.environ.get("TTNOP_WEDGE_TIMEOUT", "120"))
QUIET_TIMEOUT = float(os.environ.get("TTNOP_QUIET_TIMEOUT", "900"))
MAX_RESETS = int(os.environ.get("TTNOP_MAX_RESETS", "5"))
POLL_SECONDS = float(os.environ.get("TTNOP_POLL_SECONDS", "5"))
PROGRESS_SECONDS = float(os.environ.get("TTNOP_PROGRESS_SECONDS", "300"))
# xdist replacements consume successive functional cores.
MAX_EVICTIONS = int(os.environ.get("TTNOP_MAX_EVICTIONS", "8"))
MIN_WORKERS = int(os.environ.get("TTNOP_MIN_WORKERS", "5"))
REAP_TIMEOUT = float(os.environ.get("TTNOP_REAP_TIMEOUT", "120"))
EVICT_GRACE = float(os.environ.get("TTNOP_EVICT_GRACE", "30"))
RESET_TIMEOUT = float(os.environ.get("TTNOP_RESET_TIMEOUT", "180"))
RESET_COST = float(os.environ.get("TTNOP_RESET_COST", "240"))


class ExitStatus(IntEnum):
    CLEAN = 0
    TESTS_FAILED = 1
    USAGE_ERROR = 4
    WEDGED = 75
    INCOMPLETE = 76

    @classmethod
    def from_code(cls, code) -> "ExitStatus":
        try:
            return cls(code)
        except (TypeError, ValueError):
            return cls.INCOMPLETE


def log(message: str) -> None:
    print(f"\n>> supervisor: {message}", flush=True)


def read_ids(path) -> list:
    with open(path) as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


# -- reacting to a wedge ---------------------------------------------------


def skip_hang_family(root: Path, hung: str, all_ids: list, report_dir: Path) -> None:
    """Skip unfinished parameters of a test that just hung."""
    siblings = heartbeat.unrun_family(root, hung, all_ids)
    if not siblings:
        return
    heartbeat.record_skipped(
        root,
        siblings,
        reason=f"skipped: same hang family as {hung}",
    )
    report.append_skips(report_dir, hung, siblings)
    log(f"skipping {len(siblings)} sibling(s) of {heartbeat.family_key(hung)}")


def record_wedge(config, workers) -> None:
    """Write the variants the workers were sitting on when everything stopped."""
    for worker in workers:
        variant = worker.get("variant") or {}
        if not variant:
            continue
        report.append(
            config.report_dir,
            {
                "case": worker["case"],
                "arch": config.arch,
                "site_mode": config.site_mode,
                "thread": variant["thread"],
                "site_index": variant["site_index"],
                "addr": variant["addr"],
                "op": variant["op"],
                "filler": variant["filler"],
                "filler_word": variant["filler_word"],
                "delay": variant["delay"],
                "runs": 1,
                "fails": 1,
                "tag": "wedge",
                "error": (
                    f"card stopped answering for {worker['age']:.0f}s on "
                    f"{variant['label']}"
                ),
                # Resolve before the next attempt rebuilds this ELF.
                "chain": list(report.source_chain(variant["elf"], variant["addr"])),
            },
        )
        log(f"recorded wedge on {worker['case']}: {variant['label']}")


def terminate(child) -> None:
    """Stop the run and everything it spawned."""
    try:
        group = os.getpgid(child.pid)
    except OSError:
        return

    for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 20)):
        try:
            os.killpg(group, sig)
        except OSError:
            return
        try:
            child.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue
    log("run did not die on SIGKILL; continuing to the reset anyway")


def _pid_alive(pid: int) -> bool:
    """Whether a Linux process is running, counting a zombie as gone."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1][:1]
        return state != "Z"
    except (OSError, IndexError):
        return False


def evict(root: Path, worker) -> bool:
    """Kill a wedged worker; return whether it exited."""
    pid = worker["pid"]
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        heartbeat.clear_heartbeat(root, worker)
        return True
    except OSError as err:
        log(f"could not signal pid {pid}: {err}")
        return False

    deadline = time.time() + EVICT_GRACE
    while time.time() < deadline:
        if not _pid_alive(pid):
            heartbeat.clear_heartbeat(root, worker)
            return True
        time.sleep(POLL_SECONDS)
    return False


def core_pool() -> int:
    """Return the functional Tensix count, or 0 when unavailable."""
    try:
        from helpers.device import get_functional_tensix_locations

        return len(get_functional_tensix_locations())
    except Exception as err:
        log(f"could not size the core pool: {type(err).__name__}: {err}")
        return 0


def should_reset_now(root: Path, total: int, started, baseline, lost, healthy) -> bool:
    """Whether recovering lost cores is cheaper than finishing short-handed."""
    if healthy <= 0:
        return True
    done = len(heartbeat.completed(root))
    progressed = done - baseline
    elapsed = time.time() - started
    if progressed <= 0 or elapsed <= 0:
        return True
    remaining = total - done
    if remaining <= 0:
        return False
    return (remaining * elapsed / progressed) * (lost / (lost + healthy)) > RESET_COST


def reap(child, timeout: float) -> bool:
    """Wait until a terminated run's process group is gone."""
    deadline = time.time() + timeout
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    while True:
        try:
            os.killpg(child.pid, 0)
        except OSError:
            return True
        if time.time() >= deadline:
            return False
        time.sleep(POLL_SECONDS)


def reset_card() -> bool:
    try:
        subprocess.run(["tt-smi", "-r"], check=True, timeout=RESET_TIMEOUT)
        return True
    except Exception as err:
        log(f"card reset FAILED: {type(err).__name__}: {err}")
        return False


# -- watching one attempt --------------------------------------------------


def watch(child, root: Path, total: int, config, pool: int, all_ids: list):
    """Return (outcome, returncode, wedged workers) for one attempt."""
    started = quiet_since = last_progress = time.time()
    baseline = len(heartbeat.completed(root))
    newest = 0.0
    seen = set()
    records = []
    lost = 0
    evicted = 0
    # Keep peak width because workers leave normally as the queue drains.
    width = 0
    budget = MAX_EVICTIONS
    while True:
        if child.poll() is not None:
            return "exited", child.returncode, records

        now = time.time()
        if now - last_progress >= PROGRESS_SECONDS:
            last_progress = now
            log(f"{len(heartbeat.completed(root))}/{total} case(s) done")

        workers = [
            worker
            for worker in heartbeat.live_workers(root)
            if _pid_alive(worker["pid"])
        ]
        if len(workers) > width:
            width = len(workers)
            budget = max(0, pool - width) if pool else MAX_EVICTIONS

        for request_path, request in heartbeat.recovery_requests(root):
            heartbeat.clear_recovery_request(request_path)
            name = request.get("worker", "?")
            label = request.get("variant") or "an unknown variant"
            skip_hang_family(root, request.get("case", ""), all_ids, config.report_dir)
            if evicted < budget and evict(root, request):
                evicted += 1
                log(f"{name} hung on {label}; killed, xdist replaces it on a spare")
            else:
                log(f"{name} hung on {label}; no spare core left, resetting the card")
                return "wedged", None, records

        for worker in heartbeat.stalled(workers, WEDGE_TIMEOUT):
            name = worker.get("worker", "?")
            if name in seen:
                continue
            seen.add(name)
            records.append(worker)
            log(f"{name} silent mid-case for >{WEDGE_TIMEOUT:.0f}s")
            record_wedge(config, [worker])
            heartbeat.record_skipped(root, [worker.get("case", "")])
            skip_hang_family(root, worker.get("case", ""), all_ids, config.report_dir)

            if evicted >= budget:
                lost += 1
                log(f"{name} wedged and the card has no spare core left")
            elif evict(root, worker):
                evicted += 1
                log(f"{name} killed; xdist replaces it on a spare core")
            else:
                lost += 1
                log(f"{name} would not die; only a card reset frees that core")

        if lost:
            if width - lost < MIN_WORKERS:
                log(
                    f"resetting: {width - lost} working core(s), floor is {MIN_WORKERS}"
                )
                return "wedged", None, records
            if should_reset_now(root, total, started, baseline, lost, width - lost):
                log(f"resetting: {lost} lost core(s) cost more than a reset would")
                return "wedged", None, records

        beat = heartbeat.newest_beat(workers)
        if beat > newest:
            newest, quiet_since = beat, now
        elif now - quiet_since > QUIET_TIMEOUT:
            log(f"no worker progress for >{QUIET_TIMEOUT:.0f}s — calling it a wedge")
            return "wedged", None, records

        alive = any(worker.get("status") == heartbeat.ALIVE for worker in workers)
        left = total - len(heartbeat.completed(root))
        if left > 0 and not alive and now - quiet_since > WEDGE_TIMEOUT:
            log(
                f"{left} case(s) left but no worker mid-case for >{WEDGE_TIMEOUT:.0f}s; "
                "xdist spawn looks stuck"
            )
            return "wedged", None, records

        time.sleep(POLL_SECONDS)


# -- entry point -----------------------------------------------------------


def final_status(
    status: ExitStatus, all_ids, done, results, wedges, aborted: bool
) -> ExitStatus:
    """Preserve findings and make every partial or aborted run non-zero."""
    if aborted or set(all_ids) - set(done):
        return ExitStatus.INCOMPLETE
    if wedges:
        return ExitStatus.WEDGED
    if status == ExitStatus.CLEAN and any(
        record.get("outcome") == "failed" for record in results.values()
    ):
        return ExitStatus.TESTS_FAILED
    return status


def main(argv) -> ExitStatus:
    if len(argv) < 2:
        print("usage: supervise.py IDS_FILE [pytest args...]", file=sys.stderr)
        return ExitStatus.USAGE_ERROR

    pytest_args = argv[2:]
    all_ids = read_ids(argv[1])
    if not all_ids:
        print("ttnop: nothing collected")
        return ExitStatus.CLEAN

    root = heartbeat.state_dir()
    if root is None:
        log(f"{heartbeat.STATE_DIR_ENV} unset; running unsupervised")
        return ExitStatus.from_code(run_pytest(argv[1], pytest_args))
    root.mkdir(parents=True, exist_ok=True)

    config = sweep_module.Config.from_env()
    remaining_path = root / "remaining.txt"
    pytest_args_path = root / "pytest-args.json"
    pytest_args_path.write_text(json.dumps(pytest_args))
    status = ExitStatus.CLEAN
    wedged = []
    aborted = False
    # Query before any wedge can make device discovery hang.
    pool = core_pool()
    log(f"card has {pool} functional Tensix" if pool else "core pool unknown")

    for reset_count in range(MAX_RESETS + 1):
        done = heartbeat.completed(root)
        remaining = [nodeid for nodeid in all_ids if nodeid not in done]
        if not remaining:
            log("nothing left to run")
            break
        if done:
            log(f"{len(remaining)} case(s) left of {len(all_ids)}")
        remaining_path.write_text("\n".join(remaining) + "\n")

        heartbeat.clear_heartbeats(root)
        heartbeat.clear_recovery_requests(root)
        child = subprocess.Popen(
            [sys.executable, str(RUNNER), str(remaining_path), str(pytest_args_path)],
            start_new_session=True,
            shell=False,
        )
        log(f"run started (pid {child.pid})")
        outcome, code, payload = watch(child, root, len(all_ids), config, pool, all_ids)
        wedged.extend(payload)

        if outcome == "exited":
            status = ExitStatus.from_code(code)
            break

        terminate(child)

        if reset_count >= MAX_RESETS:
            log(f"hit the {MAX_RESETS}-reset cap; giving up on this shard")
            aborted = True
            break

        log(f"resetting the card (reset {reset_count + 1} of {MAX_RESETS})")
        if not reset_card():
            aborted = True
            break
        if not reap(child, REAP_TIMEOUT):
            log(
                "previous run outlived the reset; stopping rather than putting a "
                "second sweep on the same cores"
            )
            aborted = True
            break
    path = report.write_markdown(
        config.report_dir,
        report.environment(config.arch, config.site_mode, config.filler, config.drift),
    )
    if path:
        log(f"findings -> {path}")

    results = heartbeat.results(root)
    junit_path = junit.render(results, wedged, Path(config.report_dir) / "junit.xml")
    log(f"{len(results)} case result(s) + {len(wedged)} wedge(s) -> {junit_path}")

    done = heartbeat.completed(root)
    status = final_status(status, all_ids, done, results, wedged, aborted)
    if status == ExitStatus.INCOMPLETE:
        missing = len(set(all_ids) - done)
        log(
            f"sweep incomplete ({missing} case(s) unfinished); "
            f"exiting {status.value}"
        )
    elif status == ExitStatus.WEDGED:
        log(f"{len(wedged)} wedge(s) recorded; exiting {status.value}")
    shutil.rmtree(root, ignore_errors=True)
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
