# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare identical eager and traced batched-prefill work; not an HTTP benchmark."""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator, configure_fabric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--active", type=int)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    active = args.active or args.batch
    slots = list(range(args.batch - active, args.batch))
    torch.set_num_threads(8)
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=134217728)
    gen, trace = None, None
    report = dict(batch=args.batch, slots=slots, length=args.length, full=args.full, rows=[])
    try:
        gen = build_generator("models/autoports/qwen_qwen3_8_27b", mesh, layer_indices=None if args.full else [0, 3])
        cache = gen._ensure_cache(args.batch, ((args.length + 31) // 32) * 32)
        host = (torch.arange(args.length) % 256 + 100).repeat(active, 1)
        host += torch.arange(active)[:, None] * 13
        ids = gen.model.upload(host.int(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        positions = gen.model.upload(
            torch.arange(args.length, dtype=torch.int32).repeat(active, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        def forward():
            outputs = gen.model.prefill_batch(
                ids,
                cache=cache,
                page_table=gen.page_table,
                length=args.length,
                start_pos=0,
                slots=slots,
                positions=positions,
            )
            return ttnn.concat(outputs, dim=2)

        def read(output):
            return ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()

        gen.reset()
        output = forward()
        stable = ttnn.clone(output)
        ttnn.copy(output, stable)
        del output
        references = []
        for index in range(3):
            gen._copy((host + index * 19).int(), ids, "probe_input_copies")
            gen.reset()
            ttnn.synchronize_device(mesh)
            begin = time.perf_counter()
            output = forward()
            ttnn.copy(output, stable)
            ttnn.synchronize_device(mesh)
            seconds = time.perf_counter() - begin
            references.append(read(stable))
            report["rows"].append(dict(mode="eager", index=index, seconds=seconds))
            del output
        gen.reset()
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = forward()
        ttnn.copy(output, stable)
        del output
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        for index in range(3):
            gen._copy((host + index * 19).int(), ids, "probe_input_copies")
            gen.reset()
            ttnn.synchronize_device(mesh)
            begin = time.perf_counter()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            seconds = time.perf_counter() - begin
            actual = read(stable)
            expected = references[index]
            exact = torch.equal(actual, expected)
            pcc = torch.corrcoef(torch.stack([actual.flatten(), expected.flatten()]))[0, 1].item()
            top1_equal = torch.equal(actual.argmax(-1), expected.argmax(-1))
            report["rows"].append(
                dict(mode="trace", index=index, seconds=seconds, exact=exact, pcc=pcc, top1_equal=top1_equal)
            )
            if not top1_equal or pcc < 0.99999:
                raise AssertionError(f"Trace mismatch: {report['rows'][-1]}")
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("BATCHED_PREFILL_TRACE", json.dumps(report), flush=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
