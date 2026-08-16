# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prefill SDPA, part 2: where the crossover is, and does it accept any length?

``sdpa_prefill_probe.py`` established that prefill has the same missing-config
gap and that it is worth 6.3-6.8x at seq 4096-16384 -- but also that the best
chunk pair is **not** the same at every length, and that ``q512/k512`` fails
outright at *every* length including 128, so the failure is a resource limit and
not an alignment rule. Prefill is **not chunked** in this model
(``generator.prefill_forward`` feeds the whole logical prompt length straight
into ``attention_prefill``), so whatever is adopted must accept arbitrary S.

This leg walks the crossover finely and then hammers non-tile-aligned lengths.

    python sdpa_prefill_confirm.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import ttnn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sdpa_prefill_probe import bench, mk, ref_pcc  # noqa: E402

SEQS = [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192]
CHUNKS = [(128, 128), (256, 256), (128, 256), (256, 128)]
# every one of these is off a 32-tile, a 128-chunk, or both
NON_ALIGNED = [1, 3, 31, 33, 100, 129, 255, 257, 1000, 1023, 1025, 2049, 4095, 4097, 5000]


def main():
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
    grid = mesh.compute_with_storage_grid_size()
    results = []
    try:
        for seq in SEQS:
            q, k, v = mk(mesh, seq)
            legs = [("None", None)] + [
                (
                    f"q{a}/k{b}",
                    ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=a, k_chunk_size=b),
                )
                for a, b in CHUNKS
            ]
            base = None
            line = f"seq {seq:6d}  "
            for name, pc in legs:
                try:
                    us = bench(mesh, q, k, v, pc)
                except Exception:  # noqa: BLE001
                    line += f"{name}=FAIL  "
                    results.append({"seq": seq, "cfg": name, "error": "build failed"})
                    continue
                if base is None:
                    base = us
                line += f"{name}={us:9.2f}({base/us:5.2f}x)  "
                results.append({"seq": seq, "cfg": name, "us": us, "speedup": base / us})
            print(line, flush=True)
            for t in (q, k, v):
                ttnn.deallocate(t)

        print("\n=== arbitrary (non-aligned) sequence lengths, q128/k128 and q256/k256 ===", flush=True)
        for seq in NON_ALIGNED:
            row = f"  seq {seq:6d}  "
            q, k, v = mk(mesh, seq)
            for name, pc in (
                ("None", None),
                (
                    "q128/k128",
                    ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=128, k_chunk_size=128),
                ),
                (
                    "q256/k256",
                    ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=256, k_chunk_size=256),
                ),
            ):
                try:
                    p = ref_pcc(mesh, q, k, v, pc)
                    row += f"{name} PCC {p:.6f}   "
                    results.append({"seq": seq, "cfg": name, "aligned": False, "pcc": p})
                except Exception as exc:  # noqa: BLE001
                    row += f"{name} FAIL({str(exc).splitlines()[0][:40]})   "
                    results.append({"seq": seq, "cfg": name, "aligned": False, "error": str(exc).splitlines()[0][:200]})
            print(row, flush=True)
            for t in (q, k, v):
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh)
    (HERE / "sdpa_prefill_confirm.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
