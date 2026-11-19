# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic TP4 packing floor on saved accepted TTNN cache tensors from real text activations; no model timing."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    import torch
    import ttnn
    from .multichip_assertions import pcc

    torch.set_num_threads(4)
    ref = torch.load(a.reference, weights_only=False, map_location="cpu")
    report = {
        "status": "running",
        "mesh": [1, 4],
        "reference": str(a.reference),
        "reference_metadata": ref["metadata"],
        "pcc_threshold": 0.995,
        "scope": "Accepted single-chip TTNN BFP8 cache reference -> BF16 source -> actual device BFP8/BFP4 pack -> BF16. Isolates representation loss; no SDPA, cache-update, production forward or performance claim.",
        "watcher": "TT_METAL_WATCHER" in os.environ,
        "cases": [],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }

    def save():
        a.output.write_text(json.dumps(report, indent=2) + "\n")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        for phase in ("prefill_state", "decode_state"):
            for role, host in zip(("key", "value"), ref[phase]):
                assert host.ndim == 4 and host.shape[1:] == (4, 32, 256)
                source = ttnn.from_torch(
                    host.bfloat16(),
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
                )
                for dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
                    packed = ttnn.typecast(source, dtype)
                    restored = ttnn.typecast(packed, ttnn.bfloat16)
                    values = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(restored)]
                    ranks = []
                    for rank, value in enumerate(values):
                        oracle = host[:, rank : rank + 1].float()
                        bf16 = oracle.bfloat16().float()
                        nonzero = [page for page in range(len(oracle)) if oracle[page].abs().max() > 0]
                        ranks.append(
                            {
                                "rank": rank,
                                "finite": bool(torch.isfinite(value).all()),
                                "pcc_reference": pcc(value, oracle),
                                "pcc_bf16": pcc(value, bf16),
                                "nonzero_pages": len(nonzero),
                                "minimum_nonzero_page_pcc": min(pcc(value[i], bf16[i]) for i in nonzero),
                            }
                        )
                    report["cases"].append(
                        {
                            "phase": phase,
                            "role": role,
                            "dtype": str(packed.dtype),
                            "logical_shape": list(packed.shape),
                            "padded_shape": list(packed.padded_shape),
                            "ranks": ranks,
                            "passes_unchanged_cache_bar": all(r["pcc_reference"] >= 0.995 for r in ranks),
                        }
                    )
                    assert all(r["finite"] for r in ranks)
                    del restored, packed
                    save()
                del source
        report["status"] = "pass"
    finally:
        save()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
