# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Watchdog around a sweep: reset the card when it wedges, resume on what is left.

There are two ways a sweep loses a core, and they arrive here differently.

The first is a device that still answers: the mailbox poll gives up after a
couple of seconds and raises TimeoutError, so the worker can record the hang as
the finding it is, take the case off the resume list, and ask for a reset
(heartbeat.request_reset). Nothing else clears a core holding a kernel that
never finished — a soft reset from inside the worker reboots BRISC mid-session
and leaves it failing every case after that — so the request is honoured
immediately: kill the run, reset the card, resume at the next case. It costs the
other workers their in-flight cases, which come back on the resume list, and it
is only worth doing because a hang is rare and is itself the finding.

The second is a read that never returns at all, because the poll's deadline is
only checked between reads — the worker is blocked inside one, where it can
neither raise nor log nor run a signal handler, and so cannot ask for anything.

How far that spreads varies. A card wedged below the NoC takes every worker with
it; a NoC path wedged under one core strands that worker alone while the others
carry on. Both are fatal to the worker involved, so one silent worker is enough
to act on: waiting for the rest to agree would defer recovery until they had
drained the whole queue, which is hours on a full shard. Left alone this costs
the shard outright — the run that prompted this burned six to eight hours per
shard that way and produced no report at all.

So the detector lives outside the pytest process. Workers publish what they are
about to attempt (heartbeat.py), and when one goes quiet mid-case for longer
than any real variant could take, we:

  1. record the variant it was sitting on — the one that wedged the core is
     precisely the race the sweep exists to find, and it is the one finding the
     in-process recorder can never write, because the call never returns;
  2. kill that worker alone. xdist replaces a crashed worker under a new gateway
     id, and gwN maps to the Nth functional core, so the replacement comes up on
     a core the sweep was not using. The card has many more Tensix than a sweep
     occupies, so a lost core costs a spare rather than any throughput, and the
     wedged one is simply never addressed again;
  3. only if that worker will not die — the usual outcome when it is parked in an
     uninterruptible driver call — fall back to killing the run and resetting the
     card, then resume on the cases nobody finished yet. That reset costs every
     healthy worker's in-flight case, so it waits until it would actually pay for
     itself (see should_reset_now).

The wedging cases are marked covered as soon as they are detected, on the grounds
that re-running a case that just wedged a core mostly wedges the next one.
Sibling params of the same test are marked covered too: one hang is the race,
and the rest of that family cooks the card the same way.

    python3 supervise.py IDS_FILE [pytest args...]
"""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import heartbeat
import junit
import report
import sweep as sweep_module

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "_pytest_runner.py"

# Beats are per variant, not per case, so this is not sized by how long a case
# takes — a slow case is still many fast variants. Nor is it sized by the tests,
# which are quick (run 31714492632 implies 0.13-1.65s per variant).
#
# What sets the floor is the reporting path: a *failing* variant resolves its
# inline chain before the next beat, and report.source_chain allows addr2line up
# to 60s, so the worst legitimate silence is one variant plus that. 120 leaves
# roughly double. Anyone tempted to tighten this toward the 2s the tests
# actually take has to bound addr2line first, and should weigh that a false
# wedge costs a card reset, a skipped case and a red shard, while detecting a
# real one slowly costs seconds.
#
# That sizing was always about a single worker, which is now also how it is
# applied: one silent worker calls a wedge. So the exposure is every worker's
# chance of a slow beat rather than all of them coinciding, and the guard
# against the obvious false positive — a worker idle between cases — is the
# ALIVE/IDLE split in heartbeat.py rather than this number.
WEDGE_TIMEOUT = float(os.environ.get("TTNOP_WEDGE_TIMEOUT", "120"))
# Collection of a full suite is minutes, so this only fires on a real stall
# before the first beat or after the last one.
QUIET_TIMEOUT = float(os.environ.get("TTNOP_QUIET_TIMEOUT", "900"))
MAX_RESETS = int(os.environ.get("TTNOP_MAX_RESETS", "5"))
POLL_SECONDS = float(os.environ.get("TTNOP_POLL_SECONDS", "5"))
# The sweep itself reports nothing per case (ci.sh turns the per-test reporters
# off; at this scale they cost more than they tell anyone), so this is the only
# sign of life in a log that otherwise sits silent for hours. Cheap enough to be
# frequent -- it counts files once per interval -- but kept coarse so it stays
# skimmable over an eight-hour shard.
PROGRESS_SECONDS = float(os.environ.get("TTNOP_PROGRESS_SECONDS", "300"))
# Killing only the wedged worker is worth trying before resetting the whole card.
# xdist replaces a crashed worker with a *new* gateway id (dsession._clone_node
# clears spec.id and reallocates), and gwN maps to the Nth functional core, so the
# replacement comes up on a core the sweep was not using — the card has far more
# Tensix than a sweep occupies. The wedged core is then simply never addressed
# again, which is why this needs no blacklist.
#
# An eviction costs a spare core, not width, so the real bound is how many spare
# cores the card has; this is only the fallback for when we cannot ask it. Running
# the pool dry would be worse than a reset: the replacement worker would raise on
# a core that does not exist, crash, be replaced, and crash again until xdist gave
# up on the session.
MAX_EVICTIONS = int(os.environ.get("TTNOP_MAX_EVICTIONS", "8"))
# The floor on working cores. Only unkillable workers count against it — an
# evicted one is replaced and costs no width — so this is how many losses we
# absorb before the card has to be reset to get those cores back.
MIN_WORKERS = int(os.environ.get("TTNOP_MIN_WORKERS", "5"))
# How long to wait after the reset for a run that outlived SIGKILL to finally go.
# Generous because it is waiting on a device call to return, not on scheduling.
REAP_TIMEOUT = float(os.environ.get("TTNOP_REAP_TIMEOUT", "120"))
# How long to give a SIGKILL before concluding the worker cannot be killed at all,
# which is the expected outcome when it is parked in an uninterruptible driver
# call — precisely the case only a card reset can clear.
EVICT_GRACE = float(os.environ.get("TTNOP_EVICT_GRACE", "30"))
# What a reset costs before the sweep is back at full speed: tt-smi -r, bounded at
# RESET_TIMEOUT_SECONDS=180 in hardware_controller, plus starting pytest again and
# re-collecting the resume list. Only ever compared against the cost of carrying
# on short-handed, so a rough figure is enough.
RESET_COST = float(os.environ.get("TTNOP_RESET_COST", "240"))

# Outside pytest's 0-5 so a caller can tell the two apart: a non-zero pytest
# status means the sweep found races, which is a result and the whole point,
# while this means the card stopped answering and had to be reset, which is a
# broken run whatever the sweep managed to collect around it.
EXIT_WEDGED = 75


def log(message: str) -> None:
    print(f"\n>> supervisor: {message}", flush=True)


def read_ids(path) -> list:
    with open(path) as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


# -- reacting to a wedge ---------------------------------------------------


def _int(value, fallback: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else fallback


def skip_hang_family(root: Path, hung: str, all_ids: list) -> None:
    """Step over the other params of a test that just hung.

    The hung case is already on the done-log. What used to happen next is the
    resume ran the next format of the same test, hit the same site, and paid
    another `tt-smi -r`. Other tests are a different race and stay queued.
    """
    siblings = heartbeat.unrun_family(root, hung, all_ids)
    if not siblings:
        return
    heartbeat.record_skipped(
        root,
        siblings,
        reason=f"skipped: same hang family as {hung}",
    )
    log(f"skipping {len(siblings)} sibling(s) of {heartbeat.family_key(hung)}")


def record_wedge(config, workers) -> None:
    """Write the variants the workers were sitting on when everything stopped."""
    for worker in workers:
        variant = worker.get("variant") or {}
        if not variant:
            continue
        elf = variant.get("elf") or ""
        addr = variant.get("addr")
        # Every field is coerced to the type the renderer formats it as. These
        # records are assembled from a file a dying process wrote, and one
        # malformed value here would take down the rendering of the whole
        # report -- losing every finding in it, not just this one.
        report.append(
            config.report_dir,
            {
                "case": worker.get("case", ""),
                "arch": config.arch,
                "site_mode": config.site_mode,
                "thread": variant.get("thread") or "unknown",
                "site_index": _int(variant.get("site_index"), -1),
                "addr": _int(addr, 0),
                "op": variant.get("op") or "unknown",
                "filler": variant.get("filler") or "unknown",
                "filler_word": _int(variant.get("filler_word"), 0),
                "delay": _int(variant.get("delay"), 0),
                "runs": 1,
                "fails": 1,
                "tag": "wedge",
                "error": (
                    f"card stopped answering for {worker['age']:.0f}s on "
                    f"{variant.get('label', 'unknown variant')}"
                ),
                # Resolved here because the ELF is still on disk; after the reset
                # and the next attempt's rebuild it may not be.
                "chain": (
                    list(report.source_chain(elf, addr))
                    if elf and isinstance(addr, int)
                    else []
                ),
            },
        )
        log(f"recorded wedge on {worker.get('case', '?')}: {variant.get('label', '?')}")


def terminate(child) -> bool:
    """Stop the run and everything it spawned. False if something outlived SIGKILL."""
    try:
        group = os.getpgid(child.pid)
    except OSError:
        return True

    for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 20)):
        try:
            os.killpg(group, sig)
        except OSError:
            return True
        try:
            child.wait(timeout=grace)
            return True
        except subprocess.TimeoutExpired:
            continue
    # A worker parked in an uninterruptible device call cannot be signalled at
    # all. Resetting the card is what usually lets that call return, so this is
    # reported rather than treated as fatal.
    log("run did not die on SIGKILL; continuing to the reset anyway")
    return False


def _pid_alive(pid: int) -> bool:
    """Whether the process is still running, counting a zombie as gone.

    A worker is a child of pytest, not of us, so a killed one lingers unreaped for
    a moment and still answers signals; treating that as alive would send us to a
    card reset we had just avoided needing. /proc settles it on Linux, which is
    what CI runs. Elsewhere the fallback cannot tell a zombie from a live process
    and will say alive, which costs an unnecessary reset rather than a missed one.
    """
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1][:1]
        return state != "Z"
    except (OSError, IndexError):
        pass
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def evict(worker) -> bool:
    """Kill one wedged worker so xdist can replace it on a spare core.

    False if it will not die, which is not a surprise: a thread inside a device
    read that never returns cannot take a signal until the read does.
    """
    pid = worker.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except OSError as err:
        log(f"could not signal pid {pid}: {err}")
        return False

    deadline = time.time() + EVICT_GRACE
    while time.time() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(POLL_SECONDS)
    return False


def core_pool() -> int:
    """How many Tensix the card has, or 0 if it could not be asked.

    Deliberately called once at startup rather than when a wedge happens. It opens
    a device context, and doing that while a core is wedged risks hanging the
    watchdog on the very fault it exists to recover from. The answer is a static
    property of the part (harvesting), so nothing is lost by asking early.
    """
    try:
        from helpers.device import get_functional_tensix_locations

        return len(get_functional_tensix_locations())
    except Exception as err:
        log(f"could not size the core pool: {type(err).__name__}: {err}")
        return 0


def should_reset_now(root: Path, total: int, started, baseline, lost, healthy) -> bool:
    """Whether resetting now beats finishing the run short-handed.

    Only asked about cores we could not evict, since those stay lost until the
    card is reset. Resetting buys their throughput back but costs every healthy
    worker's in-flight case, so it is worth it early in a long run and not worth
    it near the end.
    """
    if healthy <= 0:
        return True
    done = len(heartbeat.completed(root))
    progressed = done - baseline
    elapsed = time.time() - started
    if progressed <= 0 or elapsed <= 0:
        # Nothing to extrapolate from, which only happens early, when the whole
        # shard is still ahead and a reset always pays for itself.
        return True
    remaining = total - done
    if remaining <= 0:
        return False
    # Time to finish at the rate we are actually managing, times the share of the
    # capacity the wedge took away: what carrying on would forfeit.
    return (remaining * elapsed / progressed) * (lost / (lost + healthy)) > RESET_COST


def reap(child, timeout: float) -> bool:
    """Wait for a run we already signalled, and all it spawned, to really be gone.

    start_new_session gave the run its own process group, so its pid doubles as
    the group id. Worth confirming rather than assuming: a worker stuck in a
    device call only dies once that call returns, which is what the reset is for,
    and starting the next attempt while one still holds a Tensix would put two
    sweeps on the same core.
    """
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
        from helpers.hardware_controller import HardwareController

        HardwareController().reset_card()
        return True
    except Exception as err:
        log(f"card reset FAILED: {type(err).__name__}: {err}")
        return False


# -- watching one attempt --------------------------------------------------


def watch(child, root: Path, total: int, config, pool: int, all_ids: list):
    """Block until the run ends or the card has to be reset.

    Returns (outcome, returncode, wedged workers). Outcome is "exited" if the run
    finished on its own, which it still can after a worker was evicted since xdist
    carries on with a replacement, or "wedged" if a core we could not clear makes
    a reset the cheaper option. The wedge list comes back either way: a run that
    completed around an evicted worker is not a clean run.
    """
    started = quiet_since = last_progress = time.time()
    baseline = len(heartbeat.completed(root))
    newest = 0.0
    seen = set()
    records = []
    lost = 0
    evicted = 0
    # Peak rather than current: at the tail workers legitimately finish and drop
    # out, and measuring the floor against a shrinking live count would call a
    # healthy end-of-run degraded.
    width = 0
    budget = MAX_EVICTIONS
    while True:
        if child.poll() is not None:
            return "exited", child.returncode, records

        now = time.time()
        if now - last_progress >= PROGRESS_SECONDS:
            last_progress = now
            log(f"{len(heartbeat.completed(root))}/{total} case(s) done")

        # A worker that hung a core says so and then parks. It has already
        # recorded the finding and taken its case off the resume list, so there
        # is nothing to work out here: clear the core the only way that works and
        # let the resume pick up at the next case.
        request = heartbeat.reset_request(root)
        if request:
            skip_hang_family(root, request.get("case", ""), all_ids)
            heartbeat.clear_reset_request(root)
            log(
                f"{request.get('worker', '?')} hung on "
                f"{request.get('variant') or 'an unknown variant'}; resetting the card"
            )
            return "wedged", None, records

        workers = heartbeat.live_workers(root)
        if len(workers) > width:
            width = len(workers)
            # Every replacement takes the next core in the enumeration, so this is
            # exactly how many we can lose before asking for one that is not there.
            budget = max(0, pool - width) if pool else MAX_EVICTIONS
        # One silent worker is enough to act on. Workers sit on separate cores, so
        # a NoC path wedged under one of them strands that worker alone while the
        # rest carry on, and waiting for the others to agree would defer this
        # until they had drained the queue — hours on a full shard.
        for worker in heartbeat.stalled(workers, WEDGE_TIMEOUT):
            name = worker.get("worker", "?")
            if name in seen:
                continue
            seen.add(name)
            records.append(worker)
            log(f"{name} silent mid-case for >{WEDGE_TIMEOUT:.0f}s")
            # Both done here rather than after the run stops: the ELF that
            # resolves the chain is still on disk, and the case never finished,
            # so without this a later resume would retry it and wedge another core.
            record_wedge(config, [worker])
            heartbeat.record_skipped(root, [worker.get("case", "")])
            skip_hang_family(root, worker.get("case", ""), all_ids)

            # Only evictions spend the budget: a worker we fail to kill is never
            # replaced, so it takes no new core from the pool.
            if evicted >= budget:
                # Killing it now would buy a replacement that has nowhere to run,
                # and that worker would crash on startup, be replaced, and crash
                # again until xdist gave up on the session. Better to hold the
                # width we have and let the reset below hand the cores back.
                lost += 1
                log(f"{name} wedged and the card has no spare core left")
            elif evict(worker):
                evicted += 1
                log(f"{name} killed; xdist replaces it on a spare core")
            else:
                lost += 1
                log(f"{name} would not die; only a card reset frees that core")

        if lost:
            # The floor first, as the thing we refuse to run below whatever the
            # arithmetic says. Then the economics, because sitting at 7 of 8 for
            # the rest of a long shard costs far more than the reset would.
            if width - lost < MIN_WORKERS:
                log(
                    f"resetting: {width - lost} working core(s), floor is {MIN_WORKERS}"
                )
                return "wedged", None, records
            if should_reset_now(root, total, started, baseline, lost, width - lost):
                log(f"resetting: {lost} lost core(s) cost more than a reset would")
                return "wedged", None, records

        # Progress means a beat newer than any we have seen, not merely the
        # existence of workers: at the tail they are all idle but present, and a
        # master that stalls there would otherwise never be noticed.
        beat = heartbeat.newest_beat(workers)
        if beat > newest:
            newest, quiet_since = beat, now
        elif now - quiet_since > QUIET_TIMEOUT:
            log(f"no worker progress for >{QUIET_TIMEOUT:.0f}s — calling it a wedge")
            return "wedged", None, records

        time.sleep(POLL_SECONDS)


def attempt(
    ids_path: Path,
    pytest_args,
    root: Path,
    total: int,
    config,
    pool: int,
    all_ids: list,
):
    child = subprocess.Popen(
        [sys.executable, str(RUNNER), str(ids_path), *pytest_args],
        # Its own process group, so a wedge can be cleared with one killpg
        # instead of hunting xdist workers individually.
        start_new_session=True,
    )
    log(f"run started (pid {child.pid})")
    return child, watch(child, root, total, config, pool, all_ids)


# -- entry point -----------------------------------------------------------


def main(argv) -> int:
    if len(argv) < 2:
        print("usage: supervise.py IDS_FILE [pytest args...]", file=sys.stderr)
        return 4

    pytest_args = argv[2:]
    all_ids = read_ids(argv[1])
    if not all_ids:
        print("ttnop: nothing collected")
        return 0

    root = heartbeat.state_dir()
    if root is None:
        log(f"{heartbeat.STATE_DIR_ENV} unset; running unsupervised")
        return subprocess.call([sys.executable, str(RUNNER), argv[1], *pytest_args])
    root.mkdir(parents=True, exist_ok=True)

    config = sweep_module.Config.from_env()
    remaining_path = root / "remaining.txt"
    status = 0
    wedged = []
    # Asked now, while the card is known good — the workflow resets it just before
    # this — because the answer is only ever needed once a core has wedged, and
    # that is the moment we least want to be talking to the device.
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
        child, (outcome, code, payload) = attempt(
            remaining_path, pytest_args, root, len(all_ids), config, pool, all_ids
        )
        wedged.extend(payload)

        if outcome == "exited":
            # Reached even after an eviction: xdist replaced the worker and the
            # run finished around it. The wedges still count, so the shard is red.
            status = code
            break

        terminate(child)

        if reset_count >= MAX_RESETS:
            log(f"hit the {MAX_RESETS}-reset cap; giving up on this shard")
            break

        log(f"resetting the card (reset {reset_count + 1} of {MAX_RESETS})")
        if not reset_card():
            break
        # The reset is what lets a stuck device call return, so this is where a
        # worker we could not kill finally dies. Confirm it before handing the
        # cores to a new run.
        if not reap(child, REAP_TIMEOUT):
            log(
                "previous run outlived the reset; stopping rather than putting a "
                "second sweep on the same cores"
            )
            break
        # The hung case and the rest of its test are already on the done-log, so
        # the resume starts at a different test; nothing to do here but go again.

    path = report.write_markdown(
        config.report_dir,
        report.environment(config.arch, config.site_mode, config.filler),
    )
    if path:
        log(f"findings -> {path}")

    # Assembled here rather than by pytest because only this process has seen the
    # whole run: the result log spans every attempt, including ones whose session
    # was killed before it could write anything, and the wedges are cases that no
    # pytest worker survived to report.
    results = heartbeat.results(root)
    junit_path = junit.render(results, wedged, Path(config.report_dir) / "junit.xml")
    log(f"{len(results)} case result(s) + {len(wedged)} wedge(s) -> {junit_path}")

    if wedged:
        # Overrides whatever the last attempt exited with, including a clean 0: the
        # wedges are recorded in the junit file above, but nothing carries them into
        # pytest's exit code, and the attempt that mopped up the remainder can easily
        # look perfectly healthy on its own.
        log(f"{len(wedged)} wedge(s) recorded; exiting {EXIT_WEDGED}")
        status = EXIT_WEDGED
    # Resume files (hb.gwN, done.*, remaining.txt) only exist so this process can
    # pick up after a reset. The report dir should hold what a human reads:
    # report.md, failures.jsonl, junit.xml.
    shutil.rmtree(root, ignore_errors=True)
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
