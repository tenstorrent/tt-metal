#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host memory log of a run.sh run: which stage leaves memory held in the kernel.

  memlog.py sample <rows file> <label> [--meminfo F]   append one row, print a one-line text of it
  memlog.py version [--module-dir D]                   the tt-kmd version (/sys/module/tenstorrent/version) or 'unknown'
  memlog.py summary <rows file> <stages.tsv>           the lines for the final report (nothing without two rows)

A row is epoch<TAB>kind<TAB>label<TAB>MemAvailable kB<TAB>held kB. kind: stage (a stage boundary, label = the stage code the
run enters; the row before it is the end of the stage before), reset (after a glx_reset), start, end.
Held memory is what no process, cache or kernel counter in /proc/meminfo accounts for:
MemTotal - MemFree - Buffers - Cached - Slab - AnonPages - PageTables - VmallocUsed - KernelStack - Hugetlb - Percpu.
Logging only: a growth over 1 GB is a WARN line, never a FAIL.
"""
import os
import sys
import time

SUB = (
    "MemFree",
    "Buffers",
    "Cached",
    "Slab",
    "AnonPages",
    "PageTables",
    "VmallocUsed",
    "KernelStack",
    "Hugetlb",
    "Percpu",
)
WARN_KB = 1024 * 1024
KMD = "/sys/module/tenstorrent/version"


def parse(text):
    """(MemAvailable kB, held kB) from /proc/meminfo text; a field the kernel lacks counts 0, (None, None) with no MemTotal"""
    f = {}
    for ln in text.splitlines():
        k, _, v = ln.partition(":")
        w = v.split()
        if w and w[0].isdigit():
            f[k.strip()] = int(w[0])
    if "MemTotal" not in f:
        return None, None
    held = f["MemTotal"] - sum(f.get(k, 0) for k in SUB)
    return f.get("MemAvailable", 0), held


def gb(kb):
    return kb / 1048576.0


def sample(path, label, meminfo="/proc/meminfo", now=None):
    try:
        with open(meminfo) as f:
            avail, held = parse(f.read())
    except OSError:
        return ""
    if avail is None:
        return ""
    kind = (
        "reset" if label.startswith("reset:") else "start" if label == "start" else "end" if label == "end" else "stage"
    )
    t = int(now if now is not None else time.time())
    try:
        with open(path, "a") as f:
            f.write(f"{t}\t{kind}\t{label}\t{avail}\t{held}\n")
    except OSError:
        return ""
    return f"host memory at {label}: MemAvailable {gb(avail):.1f} GB, held {gb(held):.2f} GB"


def version(d="/sys/module/tenstorrent"):
    try:
        with open(os.path.join(d, "version")) as f:
            return f.read().strip() or "unknown"
    except OSError:
        return "unknown"


def rows(path):
    out = []
    try:
        with open(path) as f:
            for ln in f:
                p = ln.rstrip("\n").split("\t")
                if len(p) == 5 and p[3].lstrip("-").isdigit() and p[4].lstrip("-").isdigit():
                    out.append((int(p[0]), p[1], p[2], int(p[3]), int(p[4])))
    except OSError:
        pass
    return out


def names(tsv):
    """stage code -> 'stage N name' from fleet/stages.tsv"""
    n = {}
    try:
        with open(tsv) as f:
            for ln in f:
                p = ln.rstrip("\n").split("\t")
                if not ln.startswith("#") and len(p) >= 3:
                    n[p[1]] = f"stage {p[0]} {p[2]}"
    except OSError:
        pass
    return n


def retained(grow):
    """growth still held at the end: a later stage that gives memory back (a reset frees what the load pinned)
    takes it off the latest earlier growth first, so the stage named is the one whose memory stays held"""
    kept = [[n, g] for n, g in grow]
    for i, (_, g) in enumerate(grow):
        back = -g
        for j in range(i - 1, -1, -1):
            if back <= 0:
                break
            if kept[j][1] > 0:
                t = min(back, kept[j][1])
                kept[j][1] -= t
                back -= t
    return [(n, g) for n, g in kept]


def summary(path, tsv):
    """report lines: per-stage held growth, the total with the stage whose growth stays held (largest after give-backs), a WARN over 1 GB"""
    r = rows(path)
    if len(r) < 2:
        return []
    nm = names(tsv)
    bounds = [x for x in r if x[1] in ("stage", "end")]
    grow = []  # (stage name, held growth kB) over each stage's span, boundary to the next boundary
    for a, b in zip(bounds, bounds[1:]):
        if a[1] == "end":
            break
        grow.append((nm.get(a[2], a[2]), b[4] - a[4]))
    total = r[-1][4] - r[0][4]
    out = []
    rs = [
        (b[2][6:], b[4] - a[4]) for a, b in zip(r, r[1:]) if b[1] == "reset"
    ]  # held growth from the row before a reset to it
    if grow:
        out.append("Host memory held per stage: " + ", ".join(f"{n} {g / 1048576.0:+.2f} GB" for n, g in grow))
    if rs:
        out.append(
            "Host memory held at each glx_reset (since the row before it): "
            + ", ".join(f"{n} {g / 1048576.0:+.2f} GB" for n, g in rs)
        )
    kept = retained(grow)
    big = max(kept, key=lambda x: x[1])[0] if kept and max(g for _, g in kept) > 0 else "none"
    out.append(f"host memory held: {gb(total):+.1f} GB this run (largest: {big})")
    if total > WARN_KB:
        out.append(
            f"MEM WARN host memory held grew {gb(total):.1f} GB during the run (largest: {big}); logging only, "
            f"not a fault of a chip or tray (tt-kmd {version()})"
        )
    return out


def main(argv):
    if len(argv) >= 3 and argv[0] == "sample":
        mi = argv[argv.index("--meminfo") + 1] if "--meminfo" in argv else "/proc/meminfo"
        s = sample(argv[1], argv[2], mi)
        if s:
            print(s)
        return 0
    if argv[:1] == ["version"]:
        print(version(argv[argv.index("--module-dir") + 1] if "--module-dir" in argv else "/sys/module/tenstorrent"))
        return 0
    if len(argv) == 3 and argv[0] == "summary":
        for ln in summary(argv[1], argv[2]):
            print(ln)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
