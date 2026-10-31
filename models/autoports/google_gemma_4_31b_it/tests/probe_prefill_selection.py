# SPDX-License-Identifier: Apache-2.0
"""Synthetic TP4 last-row selection; no model/logit accuracy claim.

Exercises PreparedPrefill.last_row at every physical tail bucket and all row
residues in the first and last tiles. All buckets are warmed before capturing;
only one selection trace is retained at a time. No trace allocations are exempted.
"""

import argparse
import json
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.google_gemma_4_31b_it.tt.prefill import PreparedPrefill


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=list(range(32, 1025, 32)))
    args = parser.parse_args()
    assert all(32 <= s <= 1024 and s % 32 == 0 for s in args.sizes)
    torch.set_num_threads(16)
    torch.manual_seed(73)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=16 * 1024 * 1024)
    try:
        run(mesh, args.sizes)
    finally:
        ttnn.close_mesh_device(mesh)


def run(mesh, sizes):
    width = 5376

    def upload(value, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            value, device=mesh, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
        )

    # This stand-in supplies only constructor metadata/upload, never model math.
    model = SimpleNamespace(
        mesh=mesh, config=SimpleNamespace(hidden_size=width), upload=upload, rope_tables={}, layers=[]
    )
    prepared = PreparedPrefill(model, 1024, [])
    hidden, reference = {}, {}
    for size in sizes:
        value = torch.randn(1, 1, size, width, dtype=torch.bfloat16)
        # Exactly representable row identifiers detect wrong tiles even where
        # random comparisons might be insufficient or BF16 rounds large IDs.
        value[0, 0, :, 0] = torch.arange(size) // 32
        value[0, 0, :, 1] = torch.arange(size) % 32
        reference[size] = value
        hidden[size] = upload(value)

    def select(size, row):
        # Exercise public staging's index calculation, not a second copy of it.
        prepared.stage(torch.zeros(row + 1, dtype=torch.int32), slot=0, start=0, size=size, valid=row + 1)

    def verify(output, size, row):
        expected = reference[size][:, :, row : row + 1, :]
        for device_index, tensor in enumerate(ttnn.get_device_tensors(output)):
            actual = ttnn.to_torch(tensor)
            assert actual.shape == expected.shape, (size, row, actual.shape, expected.shape)
            assert torch.equal(actual, expected), (
                size,
                row,
                device_index,
                float((actual.float() - expected.float()).abs().max()),
            )

    # Compile every shape before retaining any trace, including size32's
    # potential identity-slice path and the resulting untilize/embedding path.
    for size in sizes:
        select(size, 0)
        output = prepared.last_row(hidden[size])
        verify(output, size, 0)
        del output
    prepared_count = mesh.num_program_cache_entries()
    total = 0
    for size in sizes:
        rows = sorted(
            set(range(32))
            | set(range(size - 32, size))
            | {r for r in (31, 32, 63, 64, size // 2, size - 1) if r < size}
        )
        for row in rows:
            select(size, row)
            output = prepared.last_row(hidden[size])
            verify(output, size, row)
            del output
            assert mesh.num_program_cache_entries() == prepared_count, ("eager signature growth", size, row)
        select(size, rows[0])
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = prepared.last_row(hidden[size])
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        try:
            # Returning to earlier runtime indices must not preserve stale rows.
            for row in rows + list(reversed(rows)):
                select(size, row)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                verify(output, size, row)
                assert mesh.num_program_cache_entries() == prepared_count, ("trace signature growth", size, row)
        finally:
            ttnn.release_trace(mesh, trace)
        del output
        total += 3 * len(rows)
        print(
            json.dumps(
                {
                    "size": size,
                    "rows": rows,
                    "exact_on_devices": 4,
                    "program_cache_entries": prepared_count,
                    "scope": "synthetic last-row selection only",
                }
            ),
            flush=True,
        )
    print(
        json.dumps(
            {"status": "pass", "buckets": len(sizes), "row_checks": total, "program_cache_entries": prepared_count}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
