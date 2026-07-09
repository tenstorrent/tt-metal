#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hook compute_sweep_matrix.py output into the pipeline-reorg sweep matrix.

The pipeline-reorg matrix (from prepare_test_matrix.py) is one entry per
category x SKU and carries governance (sku, runs_on, timeout, owner) but does
NOT know which concrete sweep modules belong to a category. sweeps_runner.py's
--module-name needs an explicit comma-separated list of real module names; a
bare category prefix resolves to nothing and the batch runs zero tests.

This injects a per-entry `module_selector` from compute_sweep_matrix.py output,
which has ALREADY truncated the grouping suffixes (e.g.
`.hw_wormhole_tt_galaxy_wh_32c.mesh_4x8`) and routed each module to the lane its
mesh shape belongs to. We trust that output verbatim and never re-parse suffixes:

  * mesh / lead_models splits — the entry's `category` equals a compute
    `test_group_name` (e.g. 'lead-models-galaxy', 'wormhole-galaxy-sweeps'); use
    the union of that lane's already-routed modules. Because each lane maps to one
    SKU, this keeps modules on their appropriate SKU.
  * category split (nightly/comprehensive) — the `category` is an op family
    (e.g. 'eltwise.unary'); compute chunks by hardware not by op, so group its
    flat module set by dot-boundary prefix (== category or startswith category +
    '.') so 'eltwise.unary' does not swallow 'eltwise.unary_backward'. The same
    list is used for every SKU the YAML fans the category to; sweeps_runner's
    load_vectors filters per-runner hardware at run time.

Usage:
    inject_module_selectors.py <matrix_json> <compute_matrix_json>

    matrix_json          prepare_test_matrix.py output (JSON list of entries).
    compute_matrix_json  compute_sweep_matrix.py --write-to-file output, with
                         "include" (entries with test_group_name + module_selector)
                         and "ccl_batches".

Writes `matrix<<EOF...EOF` to $GITHUB_OUTPUT when set, else prints matrix=<json>.
Entries whose category matches no modules are dropped with a warning (an empty
--module-name would run nothing and only burn a runner).
"""

import json
import os
import sys


def _selector_by_prefix(category: str, base_modules: list[str]) -> list[str]:
    """Category split: op family -> every module under the dot-boundary prefix."""
    prefix = category + "."
    return sorted({m for m in base_modules if m == category or m.startswith(prefix)})


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    with open(sys.argv[1], encoding="utf-8") as f:
        matrix = json.load(f)
    with open(sys.argv[2], encoding="utf-8") as f:
        compute = json.load(f)

    # Everything comes from compute's include[].module_selector, which compute has
    # already suffix-truncated and mesh-routed.
    #   test_group_modules: lane (test_group_name) -> its modules  [mesh/lead_models]
    #   base_modules:       flat union of every lane               [category prefix path]
    test_group_modules: dict[str, set] = {}
    base_module_set: set = set()
    for e in compute.get("include", []) or []:
        mods = [m for m in str(e.get("module_selector", "")).split(",") if m]
        base_module_set.update(mods)
        tg = e.get("test_group_name")
        if tg:
            test_group_modules.setdefault(tg, set()).update(mods)
    # ccl modules are reported in a separate list for standard runs; fold in.
    for batch in compute.get("ccl_batches", []) or []:
        base_module_set.update(m for m in batch.split(",") if m)
    base_modules = sorted(base_module_set)

    enriched = []
    for entry in matrix:
        category = entry.get("category")
        if not category:
            # model_traced hardware / mesh-blackhole lane: no test_group to key on.
            # The runner auto-detects its own hardware at run time and filters vectors
            # to that lane, so hand it the full generated module set — dynamic and
            # manifest-derived (only modules that actually generated vectors), which
            # is what the old hardcoded --module-name model_traced meant to express.
            selector = base_modules
        elif category in test_group_modules:
            # mesh / lead_models lane: compute already mesh-routed + suffix-truncated.
            selector = sorted(test_group_modules[category])
        else:
            # op-category split: group the flat module set by dot-boundary prefix.
            selector = _selector_by_prefix(category, base_modules)
        if not selector:
            print(
                f"::warning::no generated modules matched category '{category}' "
                f"for '{entry.get('name')}'; dropping entry",
                file=sys.stderr,
            )
            continue
        enriched.append({**entry, "module_selector": ",".join(selector)})

    print(f"Injected module_selector into {len(enriched)}/{len(matrix)} matrix entries.", file=sys.stderr)
    for e in enriched:
        n = e["module_selector"].count(",") + 1
        print(f"  {e.get('name')} [{e.get('sku')}]: {n} modules", file=sys.stderr)

    compact = json.dumps(enriched, separators=(",", ":"))
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"matrix<<EOF\n{compact}\nEOF\n")
    else:
        print(f"matrix={compact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
