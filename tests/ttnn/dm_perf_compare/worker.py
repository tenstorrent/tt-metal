# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""One measurement process: opens the device once, runs one pass over the call list, writes JSON.

Two passes, kept in separate processes because the device profiler inflates host time:
  time    profiler off; per call and implementation, best-of-N time per call (host launch plus
          device, synced) and a digest of the output for the output-match guard.
  device  run under `python -m tracy`; every timed rep sits between signposts and the orchestrator
          reads the device kernel time and device op names from the profiler CSV afterwards.

The orchestrator runs this file once per side per round, with `PYTHONPATH` set to that side's build.
"""

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calls as calls_mod  # noqa: E402
import profile_csv  # noqa: E402
from ops import CONTROL, IMPLS, OpAdapter, Refused  # noqa: E402

MAX_ERROR_CHARS = 160


def first_line(exc):
    lines = str(exc).strip().splitlines()
    return (lines[0] if lines else type(exc).__name__)[:MAX_ERROR_CHARS]


def digest(ttnn, torch, out):
    host = ttnn.to_torch(out).contiguous()
    return hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def failure(impl, exc):
    """A refused forced entry is a skip; anything else is an error."""
    kind = "refused" if impl not in ("default", CONTROL) and isinstance(exc, (Refused, RuntimeError)) else "error"
    return {"status": kind, "error": first_line(exc)}


def bench(ttnn, device, fn, warmup, reps):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        best = min(best, (time.perf_counter() - start) * 1e6)
    return best


def time_call(adapter, ttnn, torch, device, call, impls, warmup, reps, rotation):
    results = {}
    try:
        x, kwargs = adapter.prepare(torch, device, call)
    except Exception as exc:  # noqa: BLE001
        return {impl: {"status": "error", "error": "input: " + first_line(exc)} for impl in impls}
    order = impls[rotation % len(impls) :] + impls[: rotation % len(impls)]
    for impl in order:
        try:
            fn = adapter.bind(impl, x, kwargs)
            start = time.perf_counter()
            out = fn()
            ttnn.synchronize_device(device)
            first_us = (time.perf_counter() - start) * 1e6
            out_digest = digest(ttnn, torch, out)
            del out
            results[impl] = {
                "status": "ok",
                "first_us": first_us,
                "best_us": bench(ttnn, device, fn, warmup, reps),
                "digest": out_digest,
            }
        except Exception as exc:  # noqa: BLE001
            results[impl] = failure(impl, exc)
    return results


def device_call(adapter, ttnn, torch, device, call, index, impls, reps, signpost):
    try:
        x, kwargs = adapter.prepare(torch, device, call)
    except Exception as exc:  # noqa: BLE001
        return {impl: {"status": "error", "error": "input: " + first_line(exc)} for impl in impls}
    results, bound = {}, {}
    for impl in impls:
        try:
            fn = adapter.bind(impl, x, kwargs)
            fn()
            ttnn.synchronize_device(device)
            bound[impl] = fn
            results[impl] = {"status": "ok"}
        except Exception as exc:  # noqa: BLE001
            results[impl] = failure(impl, exc)
    before = device.num_program_cache_entries()
    for rep in range(reps):
        for impl, fn in bound.items():
            signpost("BEGIN " + profile_csv.tag(impl, index, rep))
            fn()
            ttnn.synchronize_device(device)
            signpost("END " + profile_csv.tag(impl, index, rep))
    if device.num_program_cache_entries() != before:
        for impl in bound:
            results[impl] = {"status": "error", "error": "program cache grew during timed reps"}
    # The per-RISC marker buffer is finite and drops markers silently once full; draining between
    # calls, outside every signpost window, keeps a long call list measurable.
    ttnn.ReadDeviceProfiler(device)
    return results


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass", dest="mode", choices=("time", "device"), required=True)
    parser.add_argument("--calls", required=True)
    parser.add_argument("--impls", default=",".join(IMPLS))
    parser.add_argument("--select", default="", help="comma-separated call indices; default all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--rotation", type=int, default=0, help="shifts the implementation order")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    import torch
    import ttnn

    doc = calls_mod.load_calls(args.calls)
    impls = args.impls.split(",")
    indices = [int(i) for i in args.select.split(",")] if args.select else range(len(doc["calls"]))

    device = ttnn.open_device(device_id=0)
    if hasattr(device, "enable_program_cache"):
        device.enable_program_cache()
    out = {"mode": args.mode, "op": doc["op"], "impls": impls, "calls": {}}
    try:
        out["device_arch"] = str(device.arch())
        adapter = OpAdapter(doc["op"], ttnn)
        out["entries"] = {impl: adapter.available(impl) for impl in IMPLS}
        signpost = None
        if args.mode == "device":
            from tracy import signpost
        for position, index in enumerate(indices):
            call = doc["calls"][index]
            if args.mode == "time":
                result = time_call(
                    adapter, ttnn, torch, device, call, impls, args.warmup, args.reps, args.rotation + position
                )
            else:
                result = device_call(adapter, ttnn, torch, device, call, index, impls, args.reps, signpost)
            out["calls"][call["id"]] = {"index": index, "results": result}
    finally:
        ttnn.close_device(device)
    with open(args.out, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main(sys.argv[1:])
