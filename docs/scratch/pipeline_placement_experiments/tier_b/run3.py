# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Time split: simple checks, full search on passing cases, full search on gap cases."""

import collections
import random
import signal
import time

import run
from tierb import full_check, slim_check


def _alarm(*_):
    raise TimeoutError()


def pct(a, q):
    a = sorted(a)
    return a[min(len(a) - 1, int(q * len(a)))] if a else float("nan")


def main():
    signal.signal(signal.SIGALRM, _alarm)
    random.seed(2)
    t = collections.defaultdict(list)
    for path, fab in run.fabrics():
        big = len(fab.meshes) > 8
        total = sum(m.nslices for m in fab.meshes)
        for _ in range(run.PER_MGD):
            p, _ = run.gen_pipeline(fab, "p0", total)
            t0 = time.perf_counter()
            s_ok, _ = slim_check(fab, [p])
            t["slim" + ("_big" if big else "")].append((time.perf_counter() - t0) * 1e3)
            if not s_ok:
                continue
            signal.alarm(run.CASE_TIMEOUT)
            t0 = time.perf_counter()
            try:
                v, _, _ = full_check(fab, [p], 200_000)
            except TimeoutError:
                v = "TIMEOUT"
            signal.alarm(0)
            ms = (time.perf_counter() - t0) * 1e3
            key = {"PASS": "pass", "REJECT": "gap", "UNDECIDED": "budget", "TIMEOUT": "timeout"}[v]
            t[key + ("_big" if big else "")].append(ms)
    print(f"{'bucket':14s} {'n':>5s} {'p50 ms':>9s} {'p90 ms':>9s} {'p99 ms':>9s} {'max ms':>9s} {'sum s':>8s}")
    for k in [
        "slim",
        "pass",
        "gap",
        "budget",
        "timeout",
        "slim_big",
        "pass_big",
        "gap_big",
        "budget_big",
        "timeout_big",
    ]:
        a = t.get(k, [])
        if a:
            print(
                f"{k:14s} {len(a):5d} {pct(a,.5):9.3f} {pct(a,.9):9.3f} {pct(a,.99):9.2f} {max(a):9.1f} {sum(a)/1e3:8.2f}"
            )


if __name__ == "__main__":
    main()
