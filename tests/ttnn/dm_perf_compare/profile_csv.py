# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device time and device op names from a tracy ops-perf CSV, read between the worker's signposts."""

import csv
from glob import glob

KERNEL_NS = "DEVICE KERNEL DURATION [ns]"


def tag(impl, index, rep):
    return f"{impl}~{index}#{rep}"


def parse_tag(text):
    impl, rest = text.split("~", 1)
    index, rep = rest.rsplit("#", 1)
    return impl, int(index), int(rep)


def find_csv(root):
    matches = sorted(glob(root + "/**/ops_perf_results*.csv", recursive=True))
    return matches[-1] if matches else None


def parse_csv(path):
    """`{(impl, call index): {"ns": best rep, "ops": device op names of that rep}}`.

    A call can dispatch several device ops (composite implementations); a rep's time is their
    kernel durations summed. The best rep is kept, the same reduction the host pass applies.
    """
    reps = {}
    active = None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            op_code = (row.get("OP CODE") or "").strip()
            if op_code.startswith("BEGIN "):
                active = {"ns": 0.0, "ops": []}
                continue
            if op_code.startswith("END "):
                if active is not None:
                    impl, index, _ = parse_tag(op_code[4:])
                    reps.setdefault((impl, index), []).append(active)
                active = None
                continue
            if active is not None and op_code:
                active["ns"] += float(row.get(KERNEL_NS) or 0)
                active["ops"].append(op_code)
    return {key: min(samples, key=lambda s: s["ns"]) for key, samples in reps.items() if samples}
