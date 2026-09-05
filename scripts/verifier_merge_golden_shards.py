# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Merge several `eval_test_runner.sh` output dirs into one, for verify_supported.

The rms_norm_ttnn golden cartesian collects 120 960 cells (35 INPUTS shapes x a
3456-cell axis cartesian, ~21% of which survive INVALID and actually run). A
single whole-directory invocation does not fit the 10-minute tool ceiling — the
up-front precompile pass alone spends >30 min on 15 620 unique programs — so the
run is sharded with `-k <shape ids>` and the shards are merged here.

Merging is exact, not approximate:

  test_results.json  is a LIST of per-test records keyed by `nodeid`  -> dict-merge
                     by nodeid, so a cell selected by two overlapping `-k`
                     selectors appears once. Later shards win, which matters only
                     if the same cell was run twice (it is the same cell).
  test_axes.json     {nodeid: axes}    -> dict update
  test_extras.json   {nodeid: extras}  -> dict update
  test_groups.json   {nodeid: group}   -> dict update

A `skipped` record is kept like any other: `verify_supported` needs the INVALID
skips to fill its `invalid_skipped` bucket.

Usage:
    python3 scripts/verifier_merge_golden_shards.py <out_dir> <shard_dir> [<shard_dir> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _read(path: Path, default):
    if not path.exists():
        return default
    with path.open() as fh:
        return json.load(fh)


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    out = Path(argv[1])
    shards = [Path(p) for p in argv[2:]]
    out.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    axes: dict[str, object] = {}
    extras: dict[str, object] = {}
    groups: dict[str, object] = {}

    for shard in shards:
        recs = _read(shard / "test_results.json", [])
        if isinstance(recs, dict):
            recs = recs.get("tests", []) or []
        for rec in recs:
            nodeid = rec.get("nodeid") or rec.get("test_name")
            if nodeid:
                results[nodeid] = rec
        axes.update(_read(shard / "test_axes.json", {}))
        extras.update(_read(shard / "test_extras.json", {}))
        groups.update(_read(shard / "test_groups.json", {}))
        print(f"  {shard}: +{len(recs)} records")

    (out / "test_results.json").write_text(json.dumps(list(results.values()), indent=1))
    (out / "test_axes.json").write_text(json.dumps(axes, indent=1))
    (out / "test_extras.json").write_text(json.dumps(extras, indent=1))
    (out / "test_groups.json").write_text(json.dumps(groups, indent=1))

    from collections import Counter

    tally = Counter(r.get("status") for r in results.values())
    print(f"\nmerged -> {out}")
    print(f"  {len(results)} unique tests, {len(axes)} axes rows")
    for status, count in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {status}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
