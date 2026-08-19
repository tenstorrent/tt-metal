#!/usr/bin/env python3
"""corpus_watch.py — deterministic completion waiter for lane legs.

laneBU (pin-cycle infrastructure): the 2026-08-18 session lost ~3h to dead
completion waiters — ad-hoc `while sleep` loops that waited forever on a
producer that had already crashed (8 separate failures).  This watcher makes
the wait DETERMINISTIC: it polls an explicit condition AND watches the
producer's log for liveness, so a dead producer is a distinct, machine-
readable exit code the lane can react to (relaunch) instead of an unbounded
hang.

This is the REQUIRED wait mechanism for lane legs (see corpus/README.md,
"Waiting on another leg"): never `sleep N && hope`, never a bare tail-loop.

Conditions (all given conditions must hold simultaneously):
  --exists PATH          file or directory exists (repeatable)
  --grep REGEX --grep-file FILE
                         FILE exists and some line matches REGEX (one pair)

Liveness (the reason this tool exists):
  --producer-log PATH --max-age-min N
                         if the condition is unmet AND PATH's mtime is older
                         than N minutes (or PATH never appears for N minutes
                         after the watch starts), the producer is declared
                         DEAD: exit 3.  A producer that is alive keeps its
                         log advancing; a lane that gets exit 3 relaunches
                         the producer deterministically instead of waiting.
                         PATH may be a directory: the age is then measured
                         from the newest mtime found by a recursive walk
                         with a liveness early-exit (watch a producer's
                         BUILD/RUN DIR when its output lands in nested
                         per-row files rather than one streaming log —
                         e.g. a corpus leg's build dir).

Exit codes:
  0  condition met
  2  --timeout-min elapsed with the condition unmet (producer still alive)
  3  producer dead (log stale/missing beyond --max-age-min, condition unmet)
  4  usage error

Typical lane use:
  python3 corpus_watch.py \
    --exists ~/sfpi-uplift/corpus-legs/<cc1>/<flags>/leg.json \
    --producer-log ~/sfpi-uplift/laneXX/base-leg.log --max-age-min 20 \
    --interval 30 --timeout-min 90 \
  || case $? in 3) echo "producer dead — relaunching"; relaunch_leg ;; esac
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys
import time


def condition_met(exists, grep, grep_file):
    """True when every given condition holds (pure; selftest-covered)."""
    for p in exists:
        if not pathlib.Path(p).exists():
            return False
    if grep is not None:
        f = pathlib.Path(grep_file)
        if not f.is_file():
            return False
        try:
            text = f.read_text(errors="replace")
        except OSError:
            return False
        if not re.search(grep, text, re.M):
            return False
    return True


def producer_age_s(log, watch_start, fresh_within=None):
    """Seconds since the producer log last advanced (mtime).

    `log` may be a FILE (mtime) or a DIRECTORY — for a directory the age is
    measured from the NEWEST mtime found by a RECURSIVE walk, because many
    producers are quiet on any single file while steadily writing per-row
    artifacts in nested trees (e.g. a corpus leg writes run/collect-*.log
    and run/compile.log under its build dir — a file-only or shallow watch
    there false-positives 'dead'; both found dogfooding the base-leg
    store).  When `fresh_within` (seconds) is given, the walk EARLY-EXITS
    at the first path fresh enough to prove liveness, so polling a large
    build tree stays cheap.  A missing path is aged from the watch start: a
    producer that never wrote within the allowance is as dead as one that
    stopped writing."""
    log = pathlib.Path(log)
    now = time.time()
    try:
        newest = log.stat().st_mtime
    except OSError:
        return now - watch_start
    if log.is_dir():
        for root, dirs, files in os.walk(log):
            for name in dirs + files:
                try:
                    mt = os.stat(os.path.join(root, name)).st_mtime
                except OSError:
                    continue
                if mt > newest:
                    newest = mt
                    if fresh_within is not None and now - newest <= fresh_within:
                        return now - newest
    return now - newest


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--exists",
        action="append",
        default=[],
        metavar="PATH",
        help="condition: file/dir exists (repeatable; all must hold)",
    )
    ap.add_argument("--grep", help="condition: regex that must match a line")
    ap.add_argument("--grep-file", help="file the --grep regex is applied to")
    ap.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="poll interval seconds (default 30)",
    )
    ap.add_argument(
        "--producer-log",
        help="the producer's log; staleness beyond --max-age-min = dead",
    )
    ap.add_argument(
        "--max-age-min",
        type=float,
        help="minutes the producer log may go without advancing before the "
        "producer is declared dead (exit 3); requires --producer-log",
    )
    ap.add_argument(
        "--timeout-min",
        type=float,
        help="overall wall-clock bound; exit 2 when it elapses unmet",
    )
    ap.add_argument("--quiet", action="store_true", help="suppress heartbeats")
    a = ap.parse_args(argv)

    if not a.exists and a.grep is None:
        print(
            "corpus-watch: usage error — no condition given "
            "(--exists and/or --grep/--grep-file)",
            file=sys.stderr,
        )
        return 4
    if (a.grep is None) != (a.grep_file is None):
        print(
            "corpus-watch: usage error — --grep and --grep-file "
            "must be given together",
            file=sys.stderr,
        )
        return 4
    if (a.max_age_min is None) != (a.producer_log is None):
        print(
            "corpus-watch: usage error — --producer-log and --max-age-min "
            "must be given together (a liveness check needs both the log "
            "and the allowance)",
            file=sys.stderr,
        )
        return 4
    if a.interval <= 0:
        print("corpus-watch: usage error — --interval must be > 0", file=sys.stderr)
        return 4

    start = time.time()
    n = 0
    while True:
        if condition_met(a.exists, a.grep, a.grep_file):
            if not a.quiet:
                print(f"corpus-watch: condition MET after {time.time()-start:.0f}s")
            return 0
        if a.producer_log is not None:
            age = producer_age_s(a.producer_log, start, a.max_age_min * 60.0)
            if age > a.max_age_min * 60.0:
                print(
                    f"corpus-watch: PRODUCER DEAD — log {a.producer_log} has "
                    f"not advanced for {age:.0f}s (> {a.max_age_min} min) and "
                    "the condition is unmet; relaunch the producer "
                    "deterministically instead of waiting (exit 3)",
                    file=sys.stderr,
                )
                return 3
        if a.timeout_min is not None and time.time() - start > a.timeout_min * 60.0:
            print(
                f"corpus-watch: TIMEOUT — condition unmet after "
                f"{a.timeout_min} min (exit 2)",
                file=sys.stderr,
            )
            return 2
        n += 1
        if not a.quiet and n % 10 == 0:
            print(
                f"corpus-watch: waiting ({time.time()-start:.0f}s elapsed, "
                f"poll #{n})",
                flush=True,
            )
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
