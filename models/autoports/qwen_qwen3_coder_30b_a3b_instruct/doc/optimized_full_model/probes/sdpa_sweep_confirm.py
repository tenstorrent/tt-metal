# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Confirmation leg for ``sdpa_sweep_probe.py``: the finalists, tighter timing.

The full 24-point grid is timed at 20 iterations of host wall clock, which at
ctx127 (where every leg is 23-36 us) is inside dispatch noise -- the grid has at
least one obvious outlier there (``k512/c32`` at 51.81 us against 25.11 for
``k512/c8``). This leg re-runs only the finalists, at **median of 5 blocks of 50
iterations**, over a finer position ladder, so the fixed-choice decision is made
on a number that repeats.

    python sdpa_sweep_confirm.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import ttnn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sdpa_sweep_probe import bench, make, pcc_of, reference, run  # noqa: E402

CUR_POS = [127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767]
FINALISTS = [(256, 16), (256, 8), (128, 16), (128, 32), (512, 16), (64, 32), (32, 32)]
DEPTH = 65536
BLOCKS = 5
ITERS = 50


def main():
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
    grid = mesh.compute_with_storage_grid_size()
    results = []
    try:
        for cur in CUR_POS:
            q, qt, caches, pt, pos = make(mesh, DEPTH, cur, seed=11)
            ref = reference(mesh, qt, caches[0], caches[1], cur)
            legs = [("None", None)] + [
                (
                    f"k{k}/c{c}",
                    ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=grid,
                        q_chunk_size=32,
                        k_chunk_size=k,
                        max_cores_per_head_batch=c,
                    ),
                )
                for k, c in FINALISTS
            ]
            base = None
            print(f"\n=== cur_pos {cur} ===", flush=True)
            for name, pc in legs:
                out = run(q, caches, pt, pos, pc)
                p, md = pcc_of(mesh, out, ref)
                ttnn.deallocate(out)
                samples = [bench(mesh, q, caches, pt, pos, pc, iters=ITERS) for _ in range(BLOCKS)]
                us = statistics.median(samples)
                if base is None:
                    base = us
                print(
                    f"  {name:<10} {us:9.2f} us (min {min(samples):8.2f} max {max(samples):8.2f})"
                    f"  {base/us:6.2f}x  PCC {p:.6f}",
                    flush=True,
                )
                results.append(
                    {
                        "cur_pos": cur,
                        "cfg": name,
                        "us": us,
                        "min_us": min(samples),
                        "max_us": max(samples),
                        "speedup": base / us,
                        "pcc": p,
                        "max_abs_diff": md,
                    }
                )
            for x in (q, pt, pos, *caches):
                ttnn.deallocate(x)
    finally:
        ttnn.close_mesh_device(mesh)
    (HERE / "sdpa_sweep_confirm_bf16.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
