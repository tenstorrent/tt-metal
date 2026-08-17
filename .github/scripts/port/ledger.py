#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Enumerate the cases a port is graded on, from the generator's own sweep.

`discover.py` names a `sweep_module` in tt-dm-codegen whose `parameters` grid and `invalidate_vector`
gate together define the op's supported surface. Expanding that grid and classifying each point is
what makes the gates falsifiable: the port is judged on the generator's declared coverage rather than
on cases anyone chose after seeing the results.

Every point lands in exactly one of three buckets:

  in       codegen accepts it -- run the ported path and compare against a torch golden
  out      codegen rejects it -- `auto` must fall back to native, so this is a routing check
  dropped  no reliable oracle, or not constructible at all -- excluded from grading

`dropped` exists because two situations are genuinely ungradeable rather than merely failing.
`ungradeable_reasons` covers gates that are really native limitations too, so there is no working
implementation to compare against. `bad_golden` covers slices where native itself is wrong, where
comparing to it would fail a correct port.

All three narrowing inputs come from `axes/<op>.yaml` in this repository. They used to come from a
manifest in tt-dm-codegen, which is a sibling porting pipeline's own state file -- see `discover.py`
for why reading it was a standing source of silent staleness.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ttnn_names  # noqa: E402

SCOPE_IN, SCOPE_OUT, SCOPE_DROPPED = "in", "out", "dropped"

# Block-float dtypes cannot back a ROW_MAJOR tensor at all -- ttnn TT_FATALs on construction, so
# such a point is invalid input rather than an op-level rejection and never reaches grading.
_BLOCK_FLOAT = {"bfloat8_b", "bfloat4_b"}


def _pluck(vector: dict, path: str) -> Any:
    """Resolve a dotted `vector_map` path, e.g. `pad_specs.shape`, against a test vector."""
    node: Any = vector
    for part in path.split("."):
        if node is None:
            return None
        node = node[part] if isinstance(node, dict) else getattr(node, part)
    return node


def _dtype_name(value: Any) -> str:
    """ttnn dtype objects stringify as `DataType.BFLOAT16`; a case wants `bfloat16`."""
    text = str(value)
    return text.rsplit(".", 1)[-1].lower()


def _layout_name(value: Any) -> str:
    text = str(value).rsplit(".", 1)[-1].lower()
    return "row_major" if text.startswith("row_major") else "tile"


def _matches_known_bad(entry: dict, dtype: str, layout: str, kwargs: dict) -> bool:
    if entry.get("dtype") and entry["dtype"] != dtype:
        return False
    if entry.get("layout") and entry["layout"] != layout:
        return False
    nonzero = entry.get("nonzero_kwarg")
    if nonzero and not kwargs.get(nonzero):
        return False
    return True


# A dim is tile-aligned when it is a whole number of tile faces.
TILE = 32


def _outside_port_scope(case: dict, port_scope: dict) -> str | None:
    """Why this case falls outside the ported builder entry point, if it does.

    A sweep point can be entirely valid for the op and still sit outside the *entry point* being
    transliterated. `untilize`'s port covers the interleaved tile path plus bfloat16 unpadding, while
    non-tile-aligned bfloat8_b only reaches the generic through a typecast step that lives in no
    ported builder. Those points are real and must route to native, so they belong in scope=out
    rather than being dropped: scope=out is exactly what the routing check and the emitted routing
    test are for, and dropping them would retire the evidence that the predicate rejects them.

    Absent from `axes/<op>.yaml` this narrows nothing, which is why `pad` and `tilize` are unaffected.

    Worded as the constraint rather than the mechanism, and that is a requirement rather than a
    preference: these strings are what the emitted routing test groups its cases under, and that
    file ships to public tt-metal.
    """
    if not port_scope:
        return None

    layouts = port_scope.get("layouts")
    if layouts and case["layout"] not in layouts:
        return f"{case['layout']} layout is not served by the codegen path"

    dtypes = port_scope.get("dtypes")
    if dtypes and case["dtype"] not in dtypes:
        return f"{case['dtype']} is not served by the codegen path"

    # Listed dtypes are in scope only on a whole number of tiles; others are in scope either way.
    shape = case.get("shape") or []
    if case["dtype"] in (port_scope.get("tile_aligned") or []) and len(shape) >= 2:
        if shape[-2] % TILE or shape[-1] % TILE:
            # Deliberately without the offending dims. The reason is what the routing test groups
            # its cases under, and one that names a shape is unique per case, which turns the
            # grouping into a per-case comment on every line. Each case carries its own shape.
            return f"{case['dtype']} is served by the codegen path only on tile-aligned shapes"
    return None


def _signature(case: dict) -> str:
    """Identity of a case as an *input*, for deduplicating the union of several suites."""
    return json.dumps(
        [case["shape"], case["dtype"], case["layout"], case["kwargs"]],
        sort_keys=True,
        default=str,
    )


def _seed_shape_sampling(op: str) -> None:
    """Make the grid the same grid every time it is expanded.

    Upstream sweep suites build their shape lists with
    `gen_shapes(start, end, interval, num_samples)`, which draws each shape with `random.randint` and
    no seed, at module import time. The count is stable -- the try budget is ten times the sample
    count -- but the shapes are not, and `case_id` is the point's *position* in the grid. So two
    expansions agree on every identifier and disagree about what the identifiers mean.

    That is not a theoretical hazard, it is the load-bearing assumption of three things:

    - The prototype pass set and the correctness band are separate `measure.py` processes in the same
      job, neither handed a ledger. Unseeded, `codegen_tilize[57]` in the pass set excuses a port
      failure on `codegen_tilize[57]` from a different draw -- a confident, wrong attribution, in the
      one place whose comment promises the only safe way to be wrong is too harsh.
    - The no-progress chain stop compares this attempt's failing count against the last one's, which
      would be two counts over two different case sets.
    - `measure.prototype_key`'s `ledger_sig` hashes the case ids, so it cannot notice any of this.

    Seeded per op rather than globally so two ops do not draw the same shapes, and derived from the
    name rather than a stored constant so nothing has to be added to a descriptor. `random.seed`
    before the import is what matters: the sampling happens while `parameters` is being evaluated, so
    seeding after it is seeding after the draw.
    """
    random.seed(f"port-harness/{op}")


def build_ledger(descriptor: dict, *, suite: str | None = None) -> list[dict]:
    """Expand the sweep grid and classify every point.

    `suites` may name one suite or several. Several is a union, deduplicated on the input signature:
    `untilize` draws tile-aligned bfloat8_b from `broaden_suite` and everything else from `nightly`,
    and the two grids overlap. Without the dedupe a configuration present in both would be measured
    twice and counted twice in every coverage figure.

    The suites are expanded separately rather than merged, because their grids do not have to share
    keys -- `broaden_suite` carries a `shard_strategy` axis that `nightly` does not -- so there is no
    single product to take.

    Naming no suite enumerates every suite the sweep defines, which is the honest default: the sweep
    is the authority on what it covers, so taking all of it needs no decision recorded anywhere.
    """
    sweep = descriptor.get("sweep") or {}
    module_name = sweep["module"]
    _seed_shape_sampling(descriptor["op"])
    module = importlib.import_module(module_name)

    requested = suite or sweep.get("suites") or sorted(module.parameters)
    suites = [requested] if isinstance(requested, str) else list(requested)

    invalidate = getattr(module, "invalidate_vector", None)

    axes = descriptor.get("axes") or {}
    kwarg_map = axes.get("kwargs") or {}
    dropped_reasons = [r.lower() for r in (axes.get("ungradeable_reasons") or [])]
    known_bad = axes.get("bad_golden") or []
    port_scope = axes.get("scope") or {}

    cases = []
    seen: set[str] = set()
    for suite_name in suites:
        grid = module.parameters[suite_name]
        keys = sorted(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            vector = dict(zip(keys, combo))

            dtype = _dtype_name(_pluck(vector, axes.get("dtype", "dtype")))
            layout = _layout_name(_pluck(vector, axes.get("layout", "layout")))
            shape = _pluck(vector, axes.get("shape", "shape"))
            kwargs = {name: _pluck(vector, path) for name, path in kwarg_map.items()}

            case = {
                "case_id": f"{module_name.rsplit('.', 1)[-1]}[{len(cases)}]",
                "suite": suite_name,
                "shape": list(shape) if shape is not None else None,
                "dtype": dtype,
                "layout": layout,
                "kwargs": kwargs,
                # kwargs again, named rather than repr'd, and JSON-safe. `kwargs` holds live ttnn
                # objects for calling with; this is what anything reading the ledger *about* a case
                # uses. `strata.py` partitions and labels from it so that measure.py, holding the
                # live object, and gate.py, holding only what survived the JSON, name a stratum
                # identically -- which is what lets the coverage table join to the graded results.
                "kwargs_named": ttnn_names.readable(kwargs),
            }
            signature = _signature(case)
            if signature in seen:
                continue
            seen.add(signature)

            if dtype in _BLOCK_FLOAT and layout == "row_major":
                case.update(scope=SCOPE_DROPPED, reason="not constructible: block-float requires TILE")
                cases.append(case)
                continue

            if any(_matches_known_bad(e, dtype, layout, kwargs) for e in known_bad):
                case.update(scope=SCOPE_DROPPED, reason="the oracle is wrong here, so there is nothing to grade against")
                cases.append(case)
                continue

            invalid, reason = (False, None)
            if invalidate is not None:
                invalid, reason = invalidate(dict(vector))

            if not invalid:
                # The declared scope is ANDed in here: valid for the op, but still possibly outside
                # the entry point this port transliterates.
                outside = _outside_port_scope(case, port_scope)
                case.update(
                    scope=SCOPE_OUT if outside else SCOPE_IN,
                    reason=outside,
                )
            elif any(frag in (reason or "").lower() for frag in dropped_reasons):
                case.update(scope=SCOPE_DROPPED, reason=f"no implementation to compare against: {reason}")
            else:
                case.update(scope=SCOPE_OUT, reason=reason)
            cases.append(case)
    return cases


def summarize(cases: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["scope"]] = counts.get(case["scope"], 0) + 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--descriptor", required=True, help="JSON written by discover.py")
    ap.add_argument("--suite", default=None)
    ap.add_argument("--scope", default=None, choices=[SCOPE_IN, SCOPE_OUT, SCOPE_DROPPED])
    ap.add_argument("--out", default=None, help="write JSON here instead of stdout")
    args = ap.parse_args()

    descriptor = json.loads(Path(args.descriptor).read_text())
    cases = build_ledger(descriptor, suite=args.suite)
    if args.scope:
        cases = [c for c in cases if c["scope"] == args.scope]

    payload = {"op": descriptor.get("op"), "counts": summarize(cases), "cases": cases}
    text = json.dumps(payload, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text)
        print(json.dumps({"op": descriptor.get("op"), "counts": payload["counts"], "out": args.out}, indent=2))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
