# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Summarize a tt-triage --llm-output dump of a mesh hang.

    python -I handoff/wan-bin6/analyze_triage.py triage.txt [triage_callstacks_2.txt]

Prints:
  * the running ops and which devices are in each (the straggler is in the earliest op)
  * for each straggler device: cores grouped by callstack signature, with the rarest
    signatures (the likely culprit cores) listed by location
  * with a second callstack snapshot: which straggler cores have identical PCs in both
    (frozen, not polling)
"""

import collections
import csv
import io
import re
import sys


def section(lines, name):
    start = lines.index(name) + 1
    end = next((i for i in range(start, len(lines)) if re.fullmatch(r"[a-z_]+\.py:", lines[i])), len(lines))
    return "\n".join(lines[start:end])


def load(path):
    return open(path).read().split("\n")


def callstacks(lines):
    return list(csv.DictReader(io.StringIO(section(lines, "dump_callstacks.py:"))))


def site(cs):
    # Group by the op's own kernel-source frames (ttnn/cpp/...); innermost helper frames such as
    # reg_read or invalidate_l1_cache vary with polling timing and would split identical waits.
    frames = []
    for f in cs.split("\\n"):
        # Frames look like "#0 0x0000B2F0 in func () at ./path 12:3" or "#2 func<...> () at ./path 12:3".
        m = re.match(r"\s*#\d+ (?:0x[0-9A-Fa-f]+ in )?([\w:~]*).*? at \./(\S+) (\d+)", f)
        if m:
            frames.append((m.group(2), f"{m.group(1) or '?'}@{m.group(2).split('/')[-1]}:{m.group(3)}"))
    kernel = [label for path, label in frames if path.startswith("ttnn/cpp")]
    return " <- ".join((kernel or [label for _, label in frames])[:2]) or "-"


def main():
    lines = load(sys.argv[1])
    running = list(csv.DictReader(io.StringIO(section(lines, "dump_op_window.py:"))))
    running = [r for r in running if r.get("Status") == "RUNNING"]
    if not running:
        print("No RUNNING ops: every device idle -> host-side hang (see host_stacks.txt)")
        return
    print("RUNNING ops:")
    for r in running:
        print(f"  op {r['Op Id']:>7} {r['Op Name']:45s} devices: {r['Devices']}")
    earliest = min(running, key=lambda r: int(r["Op Id"]))
    stragglers = [d.strip() for d in earliest["Devices"].split(",")]
    later = [r for r in running if r is not earliest]
    # Also flag devices in an intermediate op (waiting on the straggler).
    print(f"\nStraggler device(s): {stragglers} in op {earliest['Op Id']} {earliest['Op Name']}")

    rows = callstacks(lines)
    snap2 = None
    if len(sys.argv) > 2:
        snap2 = {(r["Dev"], r["Loc"], r["RiscV"]): r["PC"] for r in callstacks(load(sys.argv[2]))}

    for dev in stragglers:
        dev_rows = [r for r in rows if r["Dev"] == dev and not r["Kernel Name"].startswith(("cq_", "fabric_"))]
        by_core = collections.defaultdict(dict)
        kernels = collections.Counter()
        for r in dev_rows:
            by_core[r["Loc"]][r["RiscV"]] = site(r["Kernel Callstack"])
            kernels[r["Kernel Name"]] += 1
        print(f"\n=== Device {dev}: kernels {dict(kernels)}")
        sigs = collections.Counter(tuple(sorted(d.items())) for d in by_core.values())
        for sig, n in sorted(sigs.items(), key=lambda kv: kv[1]):
            if n > 3 and len(sigs) > 3:
                continue
            locs = [loc for loc, d in by_core.items() if tuple(sorted(d.items())) == sig]
            print(f"  {n} core(s) {locs if n <= 3 else ''}")
            for rv, st in sig:
                print(f"      {rv:7s} {st[:150]}")
        common = sigs.most_common(1)[0]
        print(f"  most common signature: {common[1]} cores")
        if snap2 is not None:
            frozen = collections.defaultdict(list)
            for r in dev_rows:
                k = (r["Dev"], r["Loc"], r["RiscV"])
                if r["RiscV"].startswith("trisc") and snap2.get(k) == r["PC"]:
                    frozen[r["Loc"]].append(r["RiscV"])
            all3 = [loc for loc, rvs in frozen.items() if len(rvs) == 3]
            print(f"  cores with all 3 TRISC PCs identical across snapshots (frozen compute): {all3}")


if __name__ == "__main__":
    main()
