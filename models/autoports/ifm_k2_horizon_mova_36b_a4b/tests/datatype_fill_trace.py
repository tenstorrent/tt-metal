# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""BFP4 zero/nonzero device fill with persistent outputs and repeated trace replay."""

import argparse
import json
from pathlib import Path

import torch

import ttnn

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/datatype_sweep/debug/trace_capture"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-like", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    records = []
    traces = []
    try:
        prepared = []
        for shape, mem in [((1, 8, 32, 2560), ttnn.L1_MEMORY_CONFIG), ((1, 100, 32, 2560), ttnn.DRAM_MEMORY_CONFIG)]:
            source = ttnn.from_torch(
                torch.ones(shape),
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
            )
            for value in (0.0, 1.25, -2.5, 7.0):
                output = ttnn.empty_like(source)

                def run(source=source, value=value, output=output):
                    if args.full_like:
                        return ttnn.full_like(source, value, optional_tensor=output)
                    return ttnn.fill(source, value, output_tensor=output)

                run()
                prepared.append((shape, value, output, run))
        for shape, value, output, run in prepared:
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            run()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            traces.append(tid)
        for repeat in range(3):
            for (shape, value, output, _), tid in zip(prepared, traces):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                for rank, tensor in enumerate(ttnn.get_device_tensors(output)):
                    actual = ttnn.to_torch(tensor)
                    assert torch.all(actual == value), (shape, value, actual.unique())
                records.append(
                    dict(
                        repeat=repeat,
                        shape=shape,
                        value=value,
                        dtype=str(output.dtype),
                        trace_id=str(tid),
                        all_four_ranks_exact=True,
                    )
                )
        (DOC / ("full_like_trace.json" if args.full_like else "fill_trace.json")).write_text(
            json.dumps(dict(status="pass", rows=records), indent=2) + "\n"
        )
    finally:
        for tid in traces:
            ttnn.release_trace(mesh, tid)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
