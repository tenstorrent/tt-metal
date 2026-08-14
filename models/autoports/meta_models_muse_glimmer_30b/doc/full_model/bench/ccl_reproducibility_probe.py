# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Is the mesh all-gather bit-reproducible, and is it *correct*?

``embedding_gather_probe.py`` narrowed the sporadic prefill nondeterminism to the
all-gather that replicates the column-parallel embedding, and then past it: with a
host readback (a full synchronise) sitting between the embedding lookup and the
gather, and with the lookup itself proven stable across runs, the gather still
returns different data run to run.  Both the async op and the composite wrapper do
it; only a 32-row payload was clean in every arm.

So this drops the model entirely.  It gathers a **known constant** tensor, which
buys two things the model-level probes could not:

* **correctness, not just reproducibility** -- the expected result is the exact
  concatenation of the per-device shards, so this reports the error against truth
  rather than against another run that may itself be wrong;
* **a minimal repro** -- no weights, no layers, no embedding, no generator.  If
  this reproduces, nothing in this port is implicated.

It is also the fabric-health control: run it immediately after
``bench/tt_reset.py`` on an otherwise idle machine, because a mesh left in a bad
state by an earlier killed job would look exactly like a platform defect.

Usage::

    python doc/full_model/bench/tt_reset.py
    python doc/full_model/bench/ccl_reproducibility_probe.py [--rows 32,64,128,1024]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

import ttnn

ROOT = pathlib.Path(__file__).resolve().parents[3]  # models/autoports/<model>/
REPO = ROOT.parents[2]  # the tt-metal checkout
sys.path.insert(0, str(REPO))

from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (  # noqa: E402
    CCL_TOPOLOGY,
    close_multichip_mesh,
    open_multichip_mesh,
)

HIDDEN = 6656
DEVICES = 4


def say(*args) -> None:
    print(*args, flush=True)


def semaphores(mesh):
    grid = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})

    def sem():
        return ttnn.create_global_semaphore(mesh, crs, 0, ttnn.BufferType.L1_SMALL)

    return [sem(), sem()], sem()


def gather(mesh, tensor, impl: str, sems):
    if impl == "composite":
        return ttnn.all_gather(tensor, dim=3, topology=CCL_TOPOLOGY)
    ag, barrier = sems
    return ttnn.experimental.all_gather_async(
        tensor,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=ag,
        barrier_semaphore=barrier,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=CCL_TOPOLOGY,
    )


def run(mesh, rows: int, impl: str, repeats: int, sems) -> dict:
    """Gather a fixed constant ``rows x HIDDEN`` tensor ``repeats`` times."""
    torch.manual_seed(17)
    reference = torch.randn(1, 1, rows, HIDDEN, dtype=torch.bfloat16).float()
    expected = reference.clone()  # a full all-gather returns the whole tensor

    first = None
    diverged_at = None
    wrong_at = None
    worst_vs_first = 0.0
    worst_vs_expected = 0.0
    shard_diffs = [0.0] * DEVICES
    shard = HIDDEN // DEVICES

    for index in range(repeats):
        # Re-staged each run so the input is a fresh, fully written device tensor.
        staged = ttnn.from_torch(
            reference.to(torch.bfloat16),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        )
        ttnn.synchronize_device(mesh)
        out = gather(mesh, staged, impl, sems)
        host = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
        ttnn.deallocate(out)
        ttnn.deallocate(staged)

        error = float((host - expected).abs().max())
        worst_vs_expected = max(worst_vs_expected, error)
        if error > 0 and wrong_at is None:
            wrong_at = index
        if first is None:
            first = host
        elif not torch.equal(first, host):
            if diverged_at is None:
                diverged_at = index
                for device in range(DEVICES):
                    lo, hi = device * shard, (device + 1) * shard
                    shard_diffs[device] = float((first[..., lo:hi] - host[..., lo:hi]).abs().max())
            worst_vs_first = max(worst_vs_first, float((first - host).abs().max()))

    row = {
        "rows": rows,
        "impl": impl,
        "repeats": repeats,
        "reproducible": diverged_at is None,
        "first_divergent_run": diverged_at,
        "correct_every_run": wrong_at is None,
        "first_incorrect_run": wrong_at,
        "max_abs_diff_vs_first_run": worst_vs_first,
        "max_abs_error_vs_expected": worst_vs_expected,
        "per_device_shard_max_abs_diff": shard_diffs,
    }
    say(f"CCL {json.dumps(row)}")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="32,64,128,1024")
    parser.add_argument("--impls", default="async,composite")
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--out", default="ccl_reproducibility_probe.json")
    args = parser.parse_args()

    mesh = open_multichip_mesh()
    results = []
    try:
        sems = semaphores(mesh)
        for rows in [int(x) for x in args.rows.split(",")]:
            for impl in args.impls.split(","):
                try:
                    results.append(run(mesh, rows, impl, args.repeats, sems))
                except Exception as exc:  # noqa: BLE001
                    say(f"CCL rows={rows} impl={impl} FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
                    results.append({"rows": rows, "impl": impl, "error": str(exc)[:400]})
        out = ROOT / "doc/full_model" / args.out
        out.write_text(json.dumps(results, indent=2) + "\n")
        say(f"CCL wrote {out}")
        say("CCL_OK")
        return 0
    finally:
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    raise SystemExit(main())
