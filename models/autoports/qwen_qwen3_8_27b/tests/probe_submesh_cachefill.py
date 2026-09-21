# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate multi-head cache fill from the TP2 model and collectives."""

import argparse
import faulthandler
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import configure_fabric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    faulthandler.dump_traceback_later(60, repeat=True)
    torch.set_num_threads(8)
    configure_fabric()
    root = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    results = []
    submeshes = []
    try:
        submeshes = root.create_submeshes(ttnn.MeshShape(1, 2))
        for mesh_name, mesh in [("root", root), ("submesh0", submeshes[0]), ("submesh1", submeshes[1])]:
            for heads in (1, 2):
                for dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
                    label = dict(mesh=mesh_name, heads=heads, dtype=str(dtype))
                    print("CACHEFILL_BEGIN", json.dumps(label), flush=True)
                    host = (torch.arange(heads * 4096 * 256).reshape(1, heads, 4096, 256) % 17).bfloat16()

                    def upload(x, dtype=dtype, layout=ttnn.TILE_LAYOUT):
                        return ttnn.from_torch(
                            x, dtype=dtype, layout=layout, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
                        )

                    cache = upload(torch.zeros(129 * 8, heads, 32, 256).bfloat16())
                    update = upload(host)
                    table = upload(torch.arange(128).reshape(1, 128).int(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                    ttnn.experimental.paged_fill_cache(cache, update, table, batch_idx=0)
                    ttnn.synchronize_device(mesh)
                    restored = ttnn.to_torch(cache, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
                    expected = torch.zeros(129 * 8, heads, 32, 256).bfloat16()
                    expected[:128] = host[0].reshape(heads, 128, 32, 256).permute(1, 0, 2, 3)
                    label["exact"] = all(torch.equal(chip, expected) for chip in restored.chunk(mesh.get_num_devices()))
                    results.append(label)
                    args.output.write_text(json.dumps(results, indent=2) + "\n")
                    print("CACHEFILL_END", json.dumps(label), flush=True)
                    del cache, update, table, restored
            mesh.quiesce_devices()
    finally:
        for mesh in submeshes:
            ttnn.close_mesh_device(mesh)
        ttnn.close_mesh_device(root)


if __name__ == "__main__":
    main()
