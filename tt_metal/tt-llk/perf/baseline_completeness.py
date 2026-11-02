# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Keep a run that lost a shard out of the baseline pipelines.

A nightly run that lost a shard still produced the rows that finished, and
publishing them is correct: partial coverage is real data.

A baseline is not data to look at. It is the number the next queue entry is
measured against, and the gate loads it by commit. A baseline that holds two
thirds of an architecture makes the next gate compare two thirds of the points
and report clean on the rest without saying so. So for a baseline pipeline this
drops the architectures that are short, and the next entry then finds no
baseline at that commit and says so out loud.
"""

import argparse
import json
import os
import re

# Every LLK perf test name carries its architecture, and so does every artifact.
ARCHES = ("blackhole", "wormhole", "quasar")

# perf-data-blackhole-3, and with an artifact suffix perf-data-blackhole-3-head.
_ARTIFACT = re.compile(
    r"^perf-data-(?P<arch>[a-z0-9]+)-(?P<split>\d+)(?:-[A-Za-z0-9._-]+)?$"
)

# Pipelines whose output becomes somebody else's comparison point.
DEFAULT_BASELINE_PIPELINES = ("merge_baseline",)


def arch_of(text):
    """The architecture a test name or an artifact name belongs to, or None."""
    for arch in ARCHES:
        if arch in text:
            return arch
    return None


def expected_shards(matrix):
    """``{arch: {split_group, ...}}`` the run was supposed to measure."""
    out = {}
    for entry in matrix or []:
        arch = arch_of(str(entry.get("name") or ""))
        split = entry.get("split_group")
        if arch and split is not None:
            out.setdefault(arch, set()).add(str(split))
    return out


def found_shards(in_dir):
    """``{arch: {split_group, ...}}`` that actually arrived as artifacts."""
    out = {}
    for name in sorted(os.listdir(in_dir)) if os.path.isdir(in_dir) else []:
        m = _ARTIFACT.match(name)
        if m:
            out.setdefault(m["arch"], set()).add(m["split"])
    return out


def short_arches(expected, found):
    """``{arch: (missing, expected_count)}`` for every architecture with a gap."""
    short = {}
    for arch, splits in expected.items():
        missing = splits - found.get(arch, set())
        if missing:
            short[arch] = (sorted(missing), len(splits))
    return short


def parquets_of(out_dir, arch):
    """The merged files this run wrote for one architecture."""
    if not os.path.isdir(out_dir):
        return []
    tail = f"-{arch}.parquet"
    return [n for n in sorted(os.listdir(out_dir)) if n.endswith(tail)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", required=True, help="the run's test matrix, as JSON")
    ap.add_argument("--in-dir", required=True, help="where the artifacts were unpacked")
    ap.add_argument("--out-dir", required=True, help="where the merge wrote its files")
    ap.add_argument("--pipeline", default="", help="the pipeline this run publishes as")
    ap.add_argument(
        "--baseline-pipelines",
        default=",".join(DEFAULT_BASELINE_PIPELINES),
        help="pipelines that must never publish a partial architecture",
    )
    a = ap.parse_args(argv)

    guarded = {p.strip() for p in a.baseline_pipelines.split(",") if p.strip()}
    try:
        matrix = json.loads(a.matrix or "[]")
    except json.JSONDecodeError as exc:
        print(f"::warning::The test matrix did not parse ({exc}); nothing is dropped.")
        return 0

    short = short_arches(expected_shards(matrix), found_shards(a.in_dir))
    if not short:
        print("Every expected shard arrived.")
        return 0

    for arch, (missing, total) in sorted(short.items()):
        gap = f"{len(missing)} of {total} shard(s) missing ({', '.join(missing)})"
        if a.pipeline not in guarded:
            print(f"::warning::{arch}: {gap}. Pipeline '{a.pipeline}' keeps them.")
            continue
        for name in parquets_of(a.out_dir, arch):
            os.remove(os.path.join(a.out_dir, name))
            print(
                f"::warning::{arch}: {gap}. Dropped {name}, because a partial "
                f"'{a.pipeline}' would silently shrink the next comparison."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
