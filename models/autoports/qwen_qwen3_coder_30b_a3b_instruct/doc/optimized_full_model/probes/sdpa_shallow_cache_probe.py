# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Why ``k512/c16`` returned garbage in-model but not standalone: shallow caches.

``test_multichip_decode_batch`` allocates a **128-position** paged cache and
decodes at ``cur_pos=32``. With ``k_chunk_size=512`` adopted it returned
PCC **-0.0686** against HF -- not a small accuracy trade, complete nonsense --
while every standalone leg of ``sdpa_sweep_probe.py``, all of which allocated
**65536**, read PCC 0.9997. The sweep varied ``cur_pos`` and held the allocated
depth fixed, so it structurally could not see this. Same shape of miss as the
stage-04 ``rotary_embedding_llama`` rejection.

This probe varies **allocated depth** against ``k_chunk_size`` and scores PCC
against the float32 reference, to find the actual rule.

    python sdpa_shallow_cache_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import ttnn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sdpa_sweep_probe import PAGE, make, pcc_of, reference, run  # noqa: E402

DEPTHS = [128, 256, 512, 1024, 2048, 4096]
K_CHUNKS = [32, 64, 128, 256, 512, 1024]


def main():
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
    grid = mesh.compute_with_storage_grid_size()
    results = []
    try:
        for depth in DEPTHS:
            cur = min(32, depth - 1)
            q, qt, caches, pt, pos = make(mesh, depth, cur, seed=5)
            ref = reference(mesh, qt, caches[0], caches[1], cur)
            row = f"depth {depth:6d} (cur_pos {cur:5d}, {depth//PAGE:5d} pages)  "
            for k in [None] + K_CHUNKS:
                pc = (
                    None
                    if k is None
                    else ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=grid,
                        q_chunk_size=32,
                        k_chunk_size=k,
                        max_cores_per_head_batch=16,
                    )
                )
                name = "None" if k is None else f"k{k}"
                try:
                    out = run(q, caches, pt, pos, pc)
                    p, _ = pcc_of(mesh, out, ref)
                    ttnn.deallocate(out)
                    row += f"{name}={p:9.6f} "
                    results.append({"depth": depth, "cur_pos": cur, "k_chunk": k, "pcc": p})
                except Exception as exc:  # noqa: BLE001
                    row += f"{name}=RAISE "
                    results.append(
                        {"depth": depth, "cur_pos": cur, "k_chunk": k, "error": str(exc).splitlines()[0][:200]}
                    )
            print(row, flush=True)
            for x in (q, pt, pos, *caches):
                ttnn.deallocate(x)

        # Second question: at a depth that *is* deep enough, does a cur_pos far
        # below k_chunk still work? (128-deep cache is the failing case above;
        # here the cache is 4096 and only the position is small.)
        print("\n=== deep cache (4096), small cur_pos ===", flush=True)
        for cur in (0, 1, 31, 32, 33, 127, 511, 513):
            q, qt, caches, pt, pos = make(mesh, 4096, cur, seed=5)
            ref = reference(mesh, qt, caches[0], caches[1], cur)
            row = f"  cur_pos {cur:5d}  "
            for k in (None, 128, 512):
                pc = (
                    None
                    if k is None
                    else ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=grid,
                        q_chunk_size=32,
                        k_chunk_size=k,
                        max_cores_per_head_batch=16,
                    )
                )
                try:
                    out = run(q, caches, pt, pos, pc)
                    p, _ = pcc_of(mesh, out, ref)
                    ttnn.deallocate(out)
                    row += f"{'None' if k is None else 'k'+str(k)}={p:9.6f} "
                    results.append({"depth": 4096, "cur_pos": cur, "k_chunk": k, "pcc": p})
                except Exception as exc:  # noqa: BLE001
                    row += f"{'None' if k is None else 'k'+str(k)}=RAISE "
                    results.append(
                        {"depth": 4096, "cur_pos": cur, "k_chunk": k, "error": str(exc).splitlines()[0][:200]}
                    )
            print(row, flush=True)
            for x in (q, pt, pos, *caches):
                ttnn.deallocate(x)
    finally:
        ttnn.close_mesh_device(mesh)
    (HERE / "sdpa_shallow_cache_probe.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
