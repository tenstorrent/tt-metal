# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate mesh output assembly, source lifetime, and cached offset rebinding."""
import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.chunk_output import write_token_chunk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
    results = []
    try:
        mapper = ttnn.ShardTensorToMesh(mesh, dim=3)
        composer = ttnn.ConcatMeshToTensor(mesh, dim=3)
        for length in (33, 65, 33):
            expected = (torch.arange(length * 10240).reshape(1, 1, length, 10240) / 8192).bfloat16()
            destination = ttnn.empty(
                (1, 1, length, 2560),
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for start in range(0, length, 32):
                source = ttnn.from_torch(
                    expected[:, :, start : start + 32, :].contiguous(),
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=mapper,
                )
                write_token_chunk(source, destination, start)
                del source
            actual = ttnn.to_torch(destination, mesh_composer=composer)
            results.append(
                dict(
                    length=length,
                    exact=torch.equal(actual, expected),
                    finite=bool(torch.isfinite(actual).all()),
                    max_absolute=float(actual.abs().max()),
                    bad_rows=(actual != expected).any(dim=-1).flatten().nonzero().flatten().tolist(),
                )
            )
            del destination
    finally:
        ttnn.close_mesh_device(mesh)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    assert all(row["exact"] for row in results), results


if __name__ == "__main__":
    main()
