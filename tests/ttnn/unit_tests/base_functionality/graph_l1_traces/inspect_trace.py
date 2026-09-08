#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Render the views of a captured trace JSON used in README.md.  No device needed.

    python3 inspect_trace.py conv2d.json                # all views
    python3 inspect_trace.py conv2d.json --view l1      # just the L1-moving nodes
    python3 inspect_trace.py matmul_l1_out.json --view raw --elide 60

Views:
    census    node_type histogram
    skeleton  function_start / function_end tree, with durations
    l1        every L1-moving node, params verbatim, with the enclosing op resolved
    raw       the whole trace, pretty-printed, long reflection strings elided
"""

import argparse
import json
from collections import Counter

L1_NODES = {
    "buffer_allocate",
    "buffer_deallocate",
    "circular_buffer_allocate",
    "dataflow_buffer_allocate",
    "scratchpad_allocate",
    "circular_buffer_deallocate_all",
}
BUFFER_TYPE = {0: "DRAM", 1: "L1", 2: "SYSTEM_MEMORY", 3: "L1_SMALL", 4: "TRACE"}


def census(trace):
    print(f"{len(trace)} nodes\n")
    for name, count in Counter(n["node_type"] for n in trace).most_common():
        print(f"  {count:>4}  {name}")


def skeleton(trace):
    for n in trace:
        if n["node_type"] == "function_start":
            pad = "  " * max(0, n["stacking_level"] - 1)
            print(f"  {n['counter']:>4} {pad}> {n['params']['name']}   (stacking_level={n['stacking_level']})")
        elif n["node_type"] == "function_end":
            pad = "  " * max(0, n["stacking_level"] - 1)
            print(f"  {n['counter']:>4} {pad}< {n['params']['name']}   ({n.get('duration_ns', '-')} ns)")


def l1(trace):
    stack = []
    for n in trace:
        nt = n["node_type"]
        if nt == "function_start":
            stack.append(n["params"]["name"])
            continue
        if nt == "function_end":
            if stack:
                stack.pop()
            continue
        if nt not in L1_NODES:
            continue
        params = dict(n["params"])
        if "buffer_type" in params:
            params["buffer_type"] = f"{params['buffer_type']} ({BUFFER_TYPE.get(params['buffer_type'], '?')})"
        print(f"{n['counter']:>4} [{stack[-1] if stack else '-'}]")
        print(f"       {nt}  {json.dumps(params, sort_keys=True)}")


def raw(trace, elide):
    out = json.loads(json.dumps(trace))
    for n in out:
        if n.get("arguments"):
            n["arguments"] = [a[:elide] + " ...ELIDED" if len(a) > elide else a for a in n["arguments"]]
        params = n.get("params") or {}
        for key in ("memory_config",):
            if key in params and len(params[key]) > elide:
                params[key] = params[key][:elide] + " ...ELIDED"
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--view", choices=["census", "skeleton", "l1", "raw", "all"], default="all")
    ap.add_argument("--elide", type=int, default=60, help="truncate reflection strings in the raw view")
    args = ap.parse_args()

    with open(args.trace) as f:
        trace = json.load(f)

    if args.view in ("census", "all"):
        print("=== census ===")
        census(trace)
    if args.view in ("skeleton", "all"):
        print("\n=== skeleton ===")
        skeleton(trace)
    if args.view in ("l1", "all"):
        print("\n=== L1-moving nodes ===")
        l1(trace)
    if args.view == "raw":
        raw(trace, args.elide)


if __name__ == "__main__":
    main()
