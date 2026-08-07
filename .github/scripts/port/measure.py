#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the measurement legs for a codegen op port.

Three legs matter, and they answer different questions:

  native      the in-tree implementation the port has to beat
  generic_op  the tt-dm-codegen prototype, which already has the kernel win
  ported      the C++ port under test

The port ships the generator's kernels verbatim, so the device-kernel win already exists before
anyone writes C++. What the port has to prove is that productizing it -- moving from a per-call
program-descriptor rebuild to a cached DeviceOperation -- actually recovers that win at wall clock.
That is why wall time is measured against native and device time against both native and the
prototype: a port that wins on device but loses on wall has not delivered anything.

Bands:
  correctness  bit-exact vs a torch golden for in-scope cases; routing + flat program cache for
               out-of-scope cases, which must fall back to native under `auto`
  wall         min-of-N host wall clock with device sync, native vs ported
  device       run under `python3 -m tracy -p -r`; emits a dispatch-order sidecar that gate.py
               joins against the profiler CSV

This script never decides pass or fail. It reports numbers; gate.py applies the thresholds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

import ttnn

USE_TRACY = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"

try:  # only present under a tracy-enabled run
    from tracy import signpost as _signpost
except Exception:  # noqa: BLE001 - tracy is genuinely optional
    _signpost = None

_TTNN_DTYPE = {
    "bfloat16": ttnn.bfloat16,
    "float32": ttnn.float32,
    "int32": ttnn.int32,
    "uint32": ttnn.uint32,
    "uint16": ttnn.uint16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
}
_TTNN_LAYOUT = {"row_major": ttnn.ROW_MAJOR_LAYOUT, "tile": ttnn.TILE_LAYOUT}


def signpost(message: str) -> None:
    if _signpost:
        _signpost(message)


# --------------------------------------------------------------------------------------------
# Inputs and goldens
# --------------------------------------------------------------------------------------------


def random_torch_tensor(dtype: str, shape):
    """Mirror the generator sweep's input distribution so legs are comparable to its own results."""
    if dtype == "uint16":
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == "int32":
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == "uint32":
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


def _golden_pad(torch_input, kwargs):
    """ttnn gives (front, back) per trailing dim; torch wants a flat list, last dim first."""
    padding = kwargs["padding"]
    flat = []
    for i in range(len(padding) - 1, -1, -1):
        flat.extend(padding[i])
    return torch.nn.functional.pad(torch_input, flat, mode="constant", value=kwargs.get("value", 0))


# Per-op torch references. The correctness band prefers a host golden over native output, because
# native is sometimes the thing that is wrong (see the manifest's known_bad_golden entries).
GOLDENS = {"pad": _golden_pad}


# --------------------------------------------------------------------------------------------
# Call adapters
# --------------------------------------------------------------------------------------------


def call_ttnn(op: str, tensor, kwargs: dict, implementation: str | None):
    """Invoke `ttnn.<op>`, tolerating a tree where the `implementation` kwarg does not exist yet.

    The pre-port baseline is measured against clean main, where the selector has not been added,
    so asking for it there must degrade to a plain native call rather than fail the run.
    """
    fn = getattr(ttnn, op)
    if implementation is None:
        return fn(tensor, **kwargs)
    try:
        return fn(tensor, **kwargs, implementation=implementation)
    except TypeError as exc:
        if "implementation" not in str(exc):
            raise
        if implementation in ("codegen",):
            raise RuntimeError(
                f"ttnn.{op} has no `implementation` kwarg in this build -- the port is not wired up"
            ) from exc
        return fn(tensor, **kwargs)


def resolve_generic(op: str, device):
    """Instantiate the tt-dm-codegen prototype class, e.g. `ops.pad.PadCodegen`."""
    module = __import__(f"ops.{op}", fromlist=["*"])
    names = [n for n in dir(module) if n.endswith("Codegen")]
    if not names:
        return None
    return getattr(module, names[0])(device)


def call_generic(generic, op: str, tensor, kwargs: dict):
    return getattr(generic, op)(tensor, **kwargs)


# --------------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------------


def bench_sync(device, fn, warmup: int = 5, iters: int = 30) -> float:
    """Minimum wall-clock microseconds over `iters`, each synchronised to the device.

    Minimum rather than mean: the quantity of interest is the achievable dispatch cost, and the
    upper tail is host scheduling noise that says nothing about either implementation. Under the
    profiler the sample count drops because each iteration carries collection overhead.
    """
    if USE_TRACY:
        warmup, iters = 1, 5
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        times.append((time.perf_counter() - start) * 1e6)
    return min(times)


def program_cache_entries(device) -> int:
    for attr in ("num_program_cache_entries",):
        fn = getattr(device, attr, None)
        if callable(fn):
            return int(fn())
    return -1


# --------------------------------------------------------------------------------------------
# Bands
# --------------------------------------------------------------------------------------------


def make_input(case, device):
    torch_input = random_torch_tensor(case["dtype"], case["shape"])
    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        layout=_TTNN_LAYOUT[case["layout"]],
        dtype=_TTNN_DTYPE[case["dtype"]],
    )
    return torch_input, tt_input


def run_correctness(op: str, cases: list[dict], device) -> list[dict]:
    golden_fn = GOLDENS.get(op)
    results = []
    for case in cases:
        record = {"case_id": case["case_id"], "scope": case["scope"]}
        try:
            torch_input, tt_input = make_input(case, device)
            kwargs = case["kwargs"]

            if case["scope"] == "in":
                got = ttnn.to_torch(call_ttnn(op, tt_input, kwargs, "codegen"))
                if golden_fn is not None:
                    want = golden_fn(torch_input, kwargs)
                    source = "torch"
                else:
                    want = ttnn.to_torch(call_ttnn(op, tt_input, kwargs, "native"))
                    source = "native"
                record.update(
                    golden=source,
                    equal=bool(torch.equal(want.to(got.dtype), got)),
                    max_abs_diff=float((want.to(torch.float32) - got.to(torch.float32)).abs().max()),
                )
            else:
                # Out of scope: `auto` must route to native, and routing must not compile a new
                # codegen program. A flat cache across the routed call is what proves it.
                before = program_cache_entries(device)
                routed = ttnn.to_torch(call_ttnn(op, tt_input, kwargs, "auto"))
                after = program_cache_entries(device)
                native = ttnn.to_torch(call_ttnn(op, tt_input, kwargs, "native"))
                record.update(
                    equal=bool(torch.equal(native, routed)),
                    cache_before=before,
                    cache_after=after,
                    routing_ok=(before < 0 or after >= before),
                )
            record["error"] = None
        except Exception as exc:  # noqa: BLE001 - a failing case is data, not a crash
            record.update(equal=False, error=f"{type(exc).__name__}: {exc}")
        results.append(record)
    return results


def run_wall(op: str, cases: list[dict], device, generic, iters: int) -> list[dict]:
    results = []
    for case in cases:
        record = {"case_id": case["case_id"]}
        try:
            _, tt_input = make_input(case, device)
            kwargs = case["kwargs"]
            record["native_us"] = bench_sync(device, lambda: call_ttnn(op, tt_input, kwargs, "native"), iters=iters)
            record["ported_us"] = bench_sync(device, lambda: call_ttnn(op, tt_input, kwargs, "codegen"), iters=iters)
            if generic is not None:
                record["generic_us"] = bench_sync(
                    device, lambda: call_generic(generic, op, tt_input, kwargs), iters=iters
                )
            record["error"] = None
        except Exception as exc:  # noqa: BLE001
            record["error"] = f"{type(exc).__name__}: {exc}"
        results.append(record)
    return results


def run_device(op: str, cases: list[dict], device, generic, reps: int) -> list[dict]:
    """Dispatch interleaved reps under the profiler and record the exact dispatch order.

    Reps are interleaved across legs rather than run leg-by-leg so that thermal or clock drift hits
    every leg equally and cannot masquerade as a ratio. The order sidecar is what lets gate.py
    attribute profiler rows without depending on signpost columns surviving post-processing.
    """
    order = []
    for case in cases:
        try:
            _, tt_input = make_input(case, device)
        except Exception as exc:  # noqa: BLE001
            order.append({"case_id": case["case_id"], "leg": "setup", "error": str(exc)})
            continue
        kwargs = case["kwargs"]

        legs = [
            ("native", lambda: call_ttnn(op, tt_input, kwargs, "native")),
            ("ported", lambda: call_ttnn(op, tt_input, kwargs, "codegen")),
        ]
        if generic is not None:
            legs.append(("generic", lambda: call_generic(generic, op, tt_input, kwargs)))

        # One untimed warmup per leg so first-call compilation is never measured. The profiler sees
        # these dispatches too, so they are recorded to keep the order sidecar aligned row-for-row
        # with the CSV; gate.py drops them from the samples but still consumes their rows.
        for name, fn in legs:
            try:
                fn()
                ttnn.synchronize_device(device)
                order.append({"case_id": case["case_id"], "leg": f"{name}:warmup", "rep": -1, "error": None})
            except Exception as exc:  # noqa: BLE001
                order.append({"case_id": case["case_id"], "leg": f"{name}:warmup", "rep": -1, "error": str(exc)})

        for rep in range(reps):
            for name, fn in legs:
                tag = f"{name}~{case['case_id']}#{rep}"
                signpost(f"BEGIN {tag}")
                try:
                    fn()
                    ttnn.synchronize_device(device)
                    order.append({"case_id": case["case_id"], "leg": name, "rep": rep, "error": None})
                except Exception as exc:  # noqa: BLE001
                    order.append({"case_id": case["case_id"], "leg": name, "rep": rep, "error": str(exc)})
                signpost(f"END {tag}")
    return order


# --------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op", required=True)
    ap.add_argument("--ledger", required=True, help="JSON from ledger.py")
    ap.add_argument("--band", required=True, choices=["correctness", "wall", "device"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap the number of cases (perf bands)")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--no-generic", action="store_true", help="skip the tt-dm-codegen prototype leg")
    args = ap.parse_args()

    ledger = json.loads(Path(args.ledger).read_text())
    cases = ledger["cases"]
    if args.band == "correctness":
        selected = [c for c in cases if c["scope"] in ("in", "out")]
    else:
        selected = [c for c in cases if c["scope"] == "in"]
    if args.limit:
        selected = selected[: args.limit]

    device = ttnn.open_device(device_id=0)
    generic = None
    try:
        if not args.no_generic and args.band in ("wall", "device"):
            try:
                generic = resolve_generic(args.op, device)
            except Exception as exc:  # noqa: BLE001 - the prototype leg is informative, not required
                print(f"measure: generic_op leg unavailable: {exc}", file=sys.stderr)

        if args.band == "correctness":
            payload = {"band": "correctness", "results": run_correctness(args.op, selected, device)}
        elif args.band == "wall":
            payload = {
                "band": "wall",
                "iters": args.iters,
                "results": run_wall(args.op, selected, device, generic, args.iters),
            }
        else:
            payload = {
                "band": "device",
                "reps": args.reps,
                "order": run_device(args.op, selected, device, generic, args.reps),
            }
    finally:
        ttnn.close_device(device)

    payload["op"] = args.op
    payload["case_count"] = len(selected)
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps({"band": payload["band"], "cases": len(selected), "out": args.out}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
