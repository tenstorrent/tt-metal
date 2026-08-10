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

The perf bands measure a subset, because each case costs seconds of card time, and `strata.py`
chooses it -- see there for why a flat prefix of the ledger was the reason a whole class of inputs
went unmeasured. The chosen plan is written into the payload so gate.py grades within the same
classes rather than re-deriving them.

This script never decides pass or fail. It reports numbers; gate.py applies the thresholds.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import torch

import ttnn

# The device band runs this file under `python3 -m tracy`, which execs it with a `sys.path[0]` of its
# own choosing rather than this directory. Tracy happens to insert the script's directory too, but
# depending on that would trade a two-line guard for a `blocked` verdict an hour into a run.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata  # noqa: E402 - must follow the path insertion above

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


def resolve_golden(op: str, manifest: dict | None):
    """Find a host reference for `op`, preferring the most specific source available.

    The correctness band wants a host golden rather than native output, because native is sometimes
    the thing that is wrong -- that is what the manifest's `known_bad_golden` entries record. What it
    must not have is a table of per-op goldens living in this file: one entry per ported op in the
    shared harness is how a pipeline grows a maintenance surface proportional to its own success.

    So op-specific knowledge is resolved from where op-specific knowledge belongs:

      1. the manifest's `golden_callable`, a dotted path, for an op whose reference is neither of the
         below;
      2. `ttnn.get_golden_function`, which returns the reference ttnn itself already registers for
         the op -- note it returns the *raw* golden, without the pre/postprocessing that
         `get_fallback_function` layers on, so it takes a torch tensor and the op's own kwargs;
      3. nothing, in which case the caller compares against native and says so.

    Returns `(callable(torch_input, **kwargs), source)`, or `(None, "native")`.
    """
    path = (manifest or {}).get("golden_callable")
    if path:
        module_name, _, attr = str(path).replace(":", ".").rpartition(".")
        return getattr(importlib.import_module(module_name), attr), f"manifest:{path}"

    try:
        return ttnn.get_golden_function(getattr(ttnn, op)), f"ttnn.get_golden_function(ttnn.{op})"
    except (AttributeError, RuntimeError) as exc:
        print(f"measure: no host golden for {op!r} ({exc}); comparing against native", file=sys.stderr)
        return None, "native"


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


def run_correctness(op: str, cases: list[dict], device, golden_fn, golden_source: str) -> list[dict]:
    results = []
    for case in cases:
        record = {"case_id": case["case_id"], "scope": case["scope"]}
        try:
            torch_input, tt_input = make_input(case, device)
            kwargs = case["kwargs"]

            if case["scope"] == "in":
                got = ttnn.to_torch(call_ttnn(op, tt_input, kwargs, "codegen"))
                if golden_fn is not None:
                    want = golden_fn(torch_input, **kwargs)
                    source = golden_source
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
                # codegen program. A flat cache across the routed call is what proves it, and both
                # halves of that sentence were wrong here for the life of the harness.
                #
                # The order first. Native runs *before* the snapshot so that its program is already
                # compiled and cached; only then does `auto` have something to reuse. Called the
                # other way round, `auto` is the first dispatch of this configuration either way --
                # a correct fallback compiles the native program, a mis-route compiles the codegen
                # one, both add exactly one entry, and the check cannot tell them apart.
                #
                # Then the comparison. This asked `after >= before`, and a program cache never
                # shrinks, so it was true unconditionally: the routing gate that is supposed to be
                # the whole check on `is_demoted()` and `supported_by_codegen()` falling back has
                # never once failed. `==` is the assertion that was meant, and it is the same one
                # the emitted routing test makes.
                native = ttnn.to_torch(call_ttnn(op, tt_input, kwargs, "native"))
                before = program_cache_entries(device)
                routed = ttnn.to_torch(call_ttnn(op, tt_input, kwargs, "auto"))
                after = program_cache_entries(device)
                record.update(
                    equal=bool(torch.equal(native, routed)),
                    cache_before=before,
                    cache_after=after,
                    # A build without the cache-entry query cannot answer this. Say so rather than
                    # reporting a pass, which is how the tautology above went unnoticed.
                    routing_probe=("unavailable" if before < 0 else "program_cache"),
                    routing_ok=(None if before < 0 else after == before),
                )
            record["error"] = None
        except Exception as exc:  # noqa: BLE001 - a failing case is data, not a crash
            record.update(equal=False, error=f"{type(exc).__name__}: {exc}")
        results.append(record)
    return results


def run_golden_check(op: str, cases: list[dict], device, golden_fn, golden_source: str) -> list[dict]:
    """Check the resolved golden against native, before there is a port to blame.

    Resolving the golden generically is only safe if something checks that what came back is actually
    this op's reference. Native is the right thing to check it against: these cases are in scope, so
    `known_bad_golden` slices have already been dropped from the ledger, and native and the golden
    disagreeing here means the harness would have graded the port against the wrong answer. Finding
    that out in the baseline costs seconds; finding it out afterwards looks like a broken port.
    """
    results = []
    for case in cases:
        record = {"case_id": case["case_id"], "golden": golden_source, "dtype": case["dtype"]}
        try:
            torch_input, tt_input = make_input(case, device)
            native = ttnn.to_torch(call_ttnn(op, tt_input, case["kwargs"], "native"))
            want = golden_fn(torch_input, **case["kwargs"])
            record.update(
                equal=bool(torch.equal(want.to(native.dtype), native)),
                max_abs_diff=float((want.to(torch.float32) - native.to(torch.float32)).abs().max()),
                error=None,
            )
        except Exception as exc:  # noqa: BLE001
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

            # Which way `auto` actually routes this case, observed rather than assumed. The native
            # timing above has already compiled and cached the native program, so a flat cache across
            # one `auto` call means `auto` reused it -- and for an in-scope case, the only thing that
            # sends an in-scope case to native is `is_demoted()` claiming it. Growth means codegen.
            #
            # This sits between the two timings on purpose. It has to happen before anything caches a
            # codegen program for this configuration, and it costs exactly one dispatch on cases the
            # band was going to run anyway. gate.py uses it to decide which cases the port has taken
            # responsibility for and which it has handed back to native.
            before = program_cache_entries(device)
            call_ttnn(op, tt_input, kwargs, "auto")
            after = program_cache_entries(device)
            record["routes_to"] = None if before < 0 else ("native" if after == before else "codegen")

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
    ap.add_argument("--ledger", default=None, help="JSON from ledger.py; only used without --manifest")
    ap.add_argument("--band", required=True, choices=["correctness", "wall", "device", "golden"])
    ap.add_argument(
        "--manifest",
        default=None,
        help="port manifest; preferred over --ledger because it keeps ttnn objects in kwargs live",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="cap the number of cases (perf bands)")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--no-generic", action="store_true", help="skip the tt-dm-codegen prototype leg")
    ap.add_argument(
        "--select",
        default="stratified",
        choices=["stratified", "prefix"],
        help="how the perf bands spend their case budget; `prefix` is the old flat slice",
    )
    args = ap.parse_args()
    if not args.manifest and not args.ledger:
        ap.error("one of --manifest or --ledger is required")

    manifest = None
    if args.manifest:
        import yaml

        manifest = yaml.safe_load(Path(args.manifest).read_text()) or {}

    if manifest is not None:
        # Expanded here rather than read from the ledger JSON, because a case's kwargs can hold live
        # ttnn objects and JSON cannot carry them. `ledger.py --out` serialises with `default=str`, so
        # `untilize`'s `memory_config=ttnn.DRAM_MEMORY_CONFIG` arrives as the *string*
        # "MemoryConfig(...)" and every `ttnn.untilize(x, memory_config=...)` call raises. pad never
        # showed this: its kwargs are a list and a number, which survive the round trip intact.
        #
        # `build_ledger` is a deterministic function of the manifest and the sweep module, so
        # expanding it again here yields the same cases in the same order, with the same ids, that
        # gate.py grades against. Re-expanding costs a module import; encoding every ttnn type into
        # JSON and back would cost a codec that has to keep up with ttnn.
        import ledger as ledger_module

        cases = ledger_module.build_ledger(manifest)
    else:
        cases = json.loads(Path(args.ledger).read_text())["cases"]

    selection: dict | None = None
    if args.band == "correctness":
        # Never capped. Correctness is cheap per case and the routing check only means something
        # over the whole out-of-scope set, so there is no budget to spend here.
        selected = [c for c in cases if c["scope"] in ("in", "out")]
    elif args.band == "golden":
        # Stratified like the perf bands, and for the same reason: a golden that is right for one
        # dtype and wrong for another is exactly the failure this band exists to catch, so a flat
        # prefix would defeat it.
        pool = [c for c in cases if c["scope"] == "in"]
        selection = strata.plan_selection(pool, args.limit)
        selected = selection["cases"]
        selection = {k: v for k, v in selection.items() if k != "cases"}
    else:
        pool = [c for c in cases if c["scope"] == "in"]
        if args.select == "prefix":
            selected = pool[: args.limit] if args.limit else pool
            selection = {"select": "prefix", "coverage_complete": False}
        else:
            selection = strata.plan_selection(pool, args.limit)
            selected = selection["cases"]
            # The plan travels to gate.py, which grades per stratum and reports coverage; the cases
            # themselves would just duplicate the results list.
            selection = {k: v for k, v in selection.items() if k != "cases"}

    device = ttnn.open_device(device_id=0)
    generic = None
    try:
        if not args.no_generic and args.band in ("wall", "device"):
            try:
                generic = resolve_generic(args.op, device)
            except Exception as exc:  # noqa: BLE001 - the prototype leg is informative, not required
                print(f"measure: generic_op leg unavailable: {exc}", file=sys.stderr)

        if args.band in ("correctness", "golden"):
            golden_fn, golden_source = resolve_golden(args.op, manifest)

        if args.band == "correctness":
            payload = {
                "band": "correctness",
                "golden": golden_source,
                "results": run_correctness(args.op, selected, device, golden_fn, golden_source),
            }
        elif args.band == "golden":
            if golden_fn is None:
                # Nothing to check. The correctness band will compare against native, which is a
                # weaker oracle but an honest one, and it records that it did so.
                payload = {"band": "golden", "golden": "native", "results": []}
            else:
                payload = {
                    "band": "golden",
                    "golden": golden_source,
                    "results": run_golden_check(args.op, selected, device, golden_fn, golden_source),
                }
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
    if selection is not None:
        payload["selection"] = selection
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps({"band": payload["band"], "cases": len(selected), "out": args.out}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
