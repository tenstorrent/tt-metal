#!/usr/bin/env python3
"""Self-test for corpus_watch.py (deterministic completion waiter).

Drives the REAL CLI with fast intervals and fixture files:

  1. condition already met -> exit 0 immediately;
  2. condition met on a later poll (file appears while waiting) -> exit 0;
  3. PRODUCER DEAD: producer log mtime older than --max-age-min and the
     condition unmet -> exit 3 (the dead-waiter class that cost ~3h on
     2026-08-18: the watcher must exit, never wait forever);
  4. producer log MISSING beyond --max-age-min -> exit 3 (a producer that
     never wrote its log is as dead as one that stopped);
  5. a LIVE producer (log advancing) is NOT declared dead — the watch
     rides through to the condition;
  6. --timeout-min elapsing with a live producer -> exit 2;
  7. --grep condition: line appears in the log -> exit 0;
  8. usage errors (no condition; --grep without --grep-file; --max-age-min
     without --producer-log) -> exit 4.

Exit 0 all green.
"""
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
WATCH = HERE / "corpus_watch.py"
FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"SELFTEST PASS: {name}")
    else:
        print(f"SELFTEST FAIL: {name} {detail}")
        FAILS.append(name)


def run_watch(args, timeout=60):
    return subprocess.run(
        [sys.executable, str(WATCH), "--quiet", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)

    # 1. already met
    done = td / "done.marker"
    done.write_text("done\n")
    r = run_watch(["--exists", str(done), "--interval", "0.05"])
    check("condition already met -> exit 0", r.returncode == 0, r.stderr)

    # 2. met on a later poll
    late = td / "late.marker"

    def create_later(p, delay):
        time.sleep(delay)
        p.write_text("late\n")

    t = threading.Thread(target=create_later, args=(late, 0.4))
    t.start()
    r = run_watch(["--exists", str(late), "--interval", "0.05"])
    t.join()
    check("condition met on a later poll -> exit 0", r.returncode == 0, r.stderr)

    # 3. producer dead: stale log mtime, condition unmet
    log = td / "producer.log"
    log.write_text("producer output\n")
    old = time.time() - 3600
    os.utime(log, (old, old))
    r = run_watch(
        [
            "--exists",
            str(td / "never.marker"),
            "--interval",
            "0.05",
            "--producer-log",
            str(log),
            "--max-age-min",
            "1",
        ]
    )
    check(
        "stale producer log + unmet condition -> exit 3 PRODUCER DEAD",
        r.returncode == 3 and "PRODUCER DEAD" in r.stderr,
        f"rc={r.returncode} {r.stderr}",
    )

    # 4. missing producer log beyond the allowance -> exit 3
    r = run_watch(
        [
            "--exists",
            str(td / "never2.marker"),
            "--interval",
            "0.05",
            "--producer-log",
            str(td / "no-such.log"),
            "--max-age-min",
            "0.003",  # ~0.2s allowance from watch start
        ]
    )
    check(
        "missing producer log beyond allowance -> exit 3",
        r.returncode == 3,
        f"rc={r.returncode} {r.stderr}",
    )

    # 5. live producer is not declared dead; condition lands -> exit 0
    live_log = td / "live.log"
    live_log.write_text("start\n")
    stop = threading.Event()

    def keep_alive():
        while not stop.is_set():
            with open(live_log, "a") as f:
                f.write("tick\n")
            time.sleep(0.05)

    late2 = td / "late2.marker"
    ka = threading.Thread(target=keep_alive)
    ka.start()
    t = threading.Thread(target=create_later, args=(late2, 0.6))
    t.start()
    r = run_watch(
        [
            "--exists",
            str(late2),
            "--interval",
            "0.05",
            "--producer-log",
            str(live_log),
            "--max-age-min",
            "0.005",  # 0.3s — shorter than the 0.6s wait, but the log advances
        ]
    )
    t.join()
    stop.set()
    ka.join()
    check(
        "live producer rides through to the condition -> exit 0",
        r.returncode == 0,
        f"rc={r.returncode} {r.stderr}",
    )

    # 5b. DIRECTORY producer log: per-row files keep appearing (the corpus
    # collection-phase pattern) while no single file streams -> alive.
    run_dir = td / "run"
    run_dir.mkdir()
    (run_dir / "collect-000.log").write_text("row\n")
    old = time.time() - 3600
    os.utime(run_dir / "collect-000.log", (old, old))
    os.utime(run_dir, (old, old))
    stop2 = threading.Event()

    deep = run_dir / "run/nested"
    deep.mkdir(parents=True)
    for p in (deep, deep.parent):
        os.utime(p, (old, old))

    def keep_creating():
        # NESTED per-row files (the corpus build-dir shape): liveness must
        # come from the recursive walk, not the top-level dir mtime.
        i = 0
        while not stop2.is_set():
            (deep / f"collect-{i:03d}.log").write_text("row\n")
            i += 1
            time.sleep(0.05)

    late3 = td / "late3.marker"
    kc = threading.Thread(target=keep_creating)
    kc.start()
    t = threading.Thread(target=create_later, args=(late3, 0.6))
    t.start()
    r = run_watch(
        [
            "--exists",
            str(late3),
            "--interval",
            "0.05",
            "--producer-log",
            str(run_dir),
            "--max-age-min",
            "0.005",
        ]
    )
    t.join()
    stop2.set()
    kc.join()
    check(
        "directory producer log: fresh per-row files count as alive -> exit 0",
        r.returncode == 0,
        f"rc={r.returncode} {r.stderr}",
    )

    # 5c. directory producer log gone stale (recursively) -> exit 3
    for root, dirs, files in os.walk(run_dir):
        for name in dirs + files:
            os.utime(os.path.join(root, name), (old, old))
    os.utime(run_dir, (old, old))
    r = run_watch(
        [
            "--exists",
            str(td / "never4.marker"),
            "--interval",
            "0.05",
            "--producer-log",
            str(run_dir),
            "--max-age-min",
            "1",
        ]
    )
    check(
        "stale directory producer log -> exit 3",
        r.returncode == 3,
        f"rc={r.returncode} {r.stderr}",
    )

    # 6. timeout with a live-enough setup -> exit 2
    r = run_watch(
        [
            "--exists",
            str(td / "never3.marker"),
            "--interval",
            "0.05",
            "--timeout-min",
            "0.005",
        ]
    )
    check(
        "timeout with condition unmet -> exit 2",
        r.returncode == 2 and "TIMEOUT" in r.stderr,
        f"rc={r.returncode} {r.stderr}",
    )

    # 7. --grep condition
    glog = td / "leg.log"
    glog.write_text("compiling...\n")

    def append_done():
        time.sleep(0.4)
        with open(glog, "a") as f:
            f.write("LEG PUBLISHED rc=0\n")

    t = threading.Thread(target=append_done)
    t.start()
    r = run_watch(
        [
            "--grep",
            r"LEG PUBLISHED rc=0",
            "--grep-file",
            str(glog),
            "--interval",
            "0.05",
        ]
    )
    t.join()
    check("--grep condition met -> exit 0", r.returncode == 0, r.stderr)

    # 8. usage errors
    r = run_watch(["--interval", "0.05"])
    check("no condition -> exit 4", r.returncode == 4, r.returncode)
    r = run_watch(["--grep", "x", "--interval", "0.05"])
    check("--grep without --grep-file -> exit 4", r.returncode == 4, r.returncode)
    r = run_watch(["--exists", str(done), "--max-age-min", "1", "--interval", "0.05"])
    check(
        "--max-age-min without --producer-log -> exit 4",
        r.returncode == 4,
        r.returncode,
    )

if FAILS:
    print(f"corpus-watch self-test: FAILED ({len(FAILS)}: {', '.join(FAILS)})")
    sys.exit(1)
print(
    "corpus-watch self-test: ALL GREEN (met/late-met -> 0, stale/missing "
    "producer -> 3, live producer rides through, dir-log alive/stale, timeout -> 2, grep, usage -> 4)"
)
