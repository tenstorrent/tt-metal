# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bit-exact four-rank striped-read comparison; parent serializes bounded jobs.

--cpu-only does not import TTNN or touch devices. Device mode requires trace
allocation tracking including program-cache allocations. All variants and
buffers are prepared before the first capture; no exemptions are needed.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from unittest.mock import patch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_striped_gather import cpu_proof
from models.autoports.qwen_qwen3_8_flash_next.tt.pipelined_striped_gather import kernel_source


def scratch_proof(shapes, batches):
    cases = 0
    for h, w in shapes:
        tile_width = (w + 31) // 32
        visited = set()
        for tile in range(((h + 31) // 32) * tile_width):
            for batch in batches:
                selected = []
                for base in range(0, 1024, batch):
                    destinations = []
                    for slot in range(min(batch, 1024 - base)):
                        element = base + slot
                        face = element >> 8
                        row = (tile // tile_width) * 32 + ((face >> 1) << 4) + ((element & 255) >> 4)
                        col = (tile % tile_width) * 32 + ((face & 1) << 4) + (element & 15)
                        if row >= h or col >= w:
                            continue
                        pointer = 12288 + slot * 64
                        assert pointer % 64 == 0 and pointer + 64 <= 16384
                        destinations.append(pointer)
                        selected.append((row, col))
                    assert len(destinations) == len(set(destinations))
                assert len(selected) == len(set(selected))
                if batch == batches[0]:
                    visited.update(selected)
                cases += 1
        assert visited == {(row, col) for row in range(h) for col in range(w)}
    return dict(
        passed=True,
        tile_batch_cases=cases,
        batch_sizes=batches,
        scratch_bytes=16384,
        read_arena=[12288, 16384],
        limitation="Host geometry proof; not NoC execution evidence",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--columns", type=int, default=8192)
    parser.add_argument("--shapes", default="1x1,15x17,16x20,17x33,31x31,32x32,33x65,496x20,512x20")
    parser.add_argument("--dtypes", default="uint32,bfloat16,float32")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--replays", type=int, default=12)
    args = parser.parse_args()
    shapes = [tuple(map(int, entry.split("x"))) for entry in args.shapes.split(",")]
    if any(len(shape) != 2 or min(shape) < 1 for shape in shapes):
        parser.error("Shapes must be positive HxW")
    if args.columns < 64 or args.columns % 32 or args.replays < 4:
        parser.error("columns must be at least64 and divisible by32; replays at least4")
    for batch in args.batch_sizes:
        kernel_source(batch)
    report = dict(
        passed=False,
        argv=sys.argv,
        mesh=[1, 4],
        regime="serial",
        rows=[],
        cpu_address_proof=cpu_proof(),
        scratch_proof=scratch_proof(shapes, args.batch_sizes),
        lifetime="All buffers/program variants pre-exist first trace; all traces retained until teardown",
        timing_scope="Tracked warmed trace wall latency; input refresh and readback excluded",
        exemptions=[],
        seed=851,
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    if args.cpu_only:
        report["passed"] = True
        report["device_evidence"] = False
        save()
        print(json.dumps({"passed": True, "cpu_only": True, "output": str(args.output)}), flush=True)
        return
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        parser.error("Set TT_METAL_TRACE_ALLOC_TRACKING=1 before Python starts")
    if os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0":
        parser.error("Program-cache allocations must remain tracked")
    import torch
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt import striped_gather as original

    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
    report["trace_tracking"] = True
    report["program_cache_tracked"] = True
    torch.set_num_threads(8)
    torch.manual_seed(851)
    dtype_map = dict(uint32=ttnn.uint32, bfloat16=ttnn.bfloat16, float32=ttnn.float32)
    original_source = original.kernel_source
    mesh, traces = None, []
    prepared = []
    try:
        # There are no Ethernet collectives in this op. Fabric matches the
        # model target so the probe uses exactly the authorized complete mesh.
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=16 * 1024 * 1024)

        def convert(value, kind, dim, device=True):
            return ttnn.from_torch(
                value.contiguous(),
                dtype=dtype_map[kind],
                layout=ttnn.TILE_LAYOUT,
                device=mesh if device else None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
            )

        def bits(value, kind):
            if kind == "uint32":
                return value.to(torch.int64)
            return value.contiguous().view(torch.int16 if kind == "bfloat16" else torch.int32).to(torch.int64)

        def read(output, kind, shape):
            # Reshape CPU metadata to expose padding without allocating any
            # device-side tensor while traces are live.
            host = ttnn.from_device(output)
            physical = host.reshape(host.padded_shape)
            value = ttnn.to_torch(physical, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            h, w = shape
            ph, pw = (h + 31) // 32 * 32, (w + 31) // 32 * 32
            ranks = value.reshape(4, ph, pw)
            mask = torch.ones((ph, pw), dtype=torch.bool)
            mask[:h, :w] = False
            padding_zero = bool((bits(ranks, kind)[:, mask] == 0).all())
            return bits(ranks[:, :h, :w], kind), padding_zero

        for kind in args.dtypes.split(","):
            if kind not in dtype_map:
                raise ValueError(f"Unknown dtype {kind}")
            tables = []
            for rank in range(4):
                if kind == "uint32":
                    value = torch.randint(0, 2**32, (32, args.columns), dtype=torch.int64).to(torch.uint32)
                else:
                    value = (torch.randn(32, args.columns) + rank).to(
                        torch.bfloat16 if kind == "bfloat16" else torch.float32
                    )
                    value[0, 0], value[31, -1] = 0.0, -0.0
                tables.append(value)
            table = convert(torch.cat(tables, dim=1), kind, 1)
            for shape in shapes:
                variants = []
                goldens = []
                for change in range(2):
                    columns, stripes, expected = [], [], []
                    for rank in range(4):
                        c = torch.randint(0, args.columns, shape, dtype=torch.int64)
                        s = torch.randint(0, 32, shape, dtype=torch.int64)
                        boundary = [
                            0,
                            1,
                            15,
                            16,
                            31,
                            32,
                            63,
                            min(64, args.columns - 1),
                            args.columns - 2,
                            args.columns - 1,
                        ]
                        size = min(c.numel(), len(boundary))
                        c.flatten()[:size] = torch.tensor(boundary[:size])
                        s.flatten()[: min(32, s.numel())] = (
                            torch.arange(min(32, s.numel())) + 7 * rank + 13 * change
                        ) % 32
                        expected.append(bits(tables[rank], kind)[s, c])
                        columns.append(c)
                        stripes.append(s)
                    host_c, host_s = torch.cat(columns), torch.cat(stripes)
                    variants.append((convert(host_c, "uint32", 0, False), convert(host_s, "uint32", 0, False)))
                    goldens.append(torch.stack(expected))
                c = ttnn.to_device(variants[0][0], mesh)
                s = ttnn.to_device(variants[0][1], mesh)
                operations = {}
                for name, source in [
                    ("original", original_source),
                    *[(f"batch{b}", lambda b=b: kernel_source(b)) for b in args.batch_sizes],
                ]:
                    output = convert(torch.zeros((shape[0] * 4, shape[1]), dtype=tables[0].dtype), kind, 0)
                    with patch.object(original, "kernel_source", source):
                        op = original.StripedGather(table=table, columns=c, stripes=s, output=output)
                    op.prepare()
                    op.forward()
                    operations[name] = (op, output)
                # All changed-index copies are host-to-existing-buffer writes.
                prepared.append(
                    dict(
                        kind=kind,
                        shape=shape,
                        columns=c,
                        stripes=s,
                        host_variants=variants,
                        goldens=goldens,
                        operations=operations,
                    )
                )
        ttnn.synchronize_device(mesh)
        report["prepared_cases"] = len(prepared)
        save()
        for case in prepared:
            row = dict(dtype=case["kind"], shape=list(case["shape"]), columns=args.columns, modes={})
            for name, (op, output) in case["operations"].items():
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                op.forward()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                traces.append(trace)
                checks, samples = [], []
                for replay in range(args.replays):
                    changed = replay % 2
                    for host, target in zip(case["host_variants"][changed], (case["columns"], case["stripes"])):
                        ttnn.copy_host_to_device_tensor(host, target)
                    ttnn.synchronize_device(mesh)
                    start = time.perf_counter()
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    samples.append((time.perf_counter() - start) * 1000)
                    actual, padding_zero = read(output, case["kind"], case["shape"])
                    checks.append(
                        dict(
                            replay=replay,
                            input=changed,
                            bit_exact=torch.equal(actual, case["goldens"][changed]),
                            padding_zero=padding_zero,
                        )
                    )
                row["modes"][name] = dict(
                    trace_id=int(trace),
                    checks=checks,
                    samples_ms=samples,
                    median_ms=statistics.median(samples[2:]),
                    passed=all(c["bit_exact"] and c["padding_zero"] for c in checks),
                )
            original_ms = row["modes"]["original"]["median_ms"]
            for result in row["modes"].values():
                result["speedup_vs_original"] = original_ms / result["median_ms"]
            row["passed"] = all(mode["passed"] for mode in row["modes"].values())
            report["rows"].append(row)
            save()
            print(
                json.dumps(
                    dict(
                        dtype=row["dtype"],
                        shape=row["shape"],
                        passed=row["passed"],
                        median_ms={name: result["median_ms"] for name, result in row["modes"].items()},
                    )
                ),
                flush=True,
            )
            if not row["passed"]:
                raise AssertionError("Striped output bits or physical padding differ")
        report["passed"] = True
        report["captured_traces"] = len(traces)
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        save()
        if mesh is not None:
            for trace in traces:
                ttnn.release_trace(mesh, trace)
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
