# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure occupancy/bucket compute costs with isolated synthetic request slots.

Not a serving padding implementation: omitted dummy-state preservation would add
cost. Every row owns disjoint pages; all state is reset between measurements.
"""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator, configure_fabric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--counts", default="1,2,4,7,8,15,16")
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=134217728)
    gen = None
    report = dict(capacity=args.capacity, length=args.length, full=args.full, rows=[])
    references = {}
    try:
        gen = build_generator("models/autoports/qwen_qwen3_8_27b", mesh, layer_indices=None if args.full else [0, 3])
        gen.batched_prefill = True
        cache = gen._ensure_cache(args.capacity, ((args.length + 31) // 32) * 32)
        tokens = (torch.arange(args.length) % 256 + 100).repeat(args.capacity, 1)
        tokens += torch.arange(args.capacity)[:, None] * 13
        for count in map(int, args.counts.split(",")):
            for repeat in range(3):
                gen.reset()
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                outputs = gen.prefill_forward(
                    tokens[:count],
                    page_table=gen.page_table,
                    kv_cache=cache,
                    prompt_lens=[args.length] * count,
                    slots=list(range(count)),
                )
                ttnn.synchronize_device(mesh)
                row = dict(count=count, repeat=repeat, seconds=time.perf_counter() - begin)
                if repeat == 2:
                    references[count] = torch.cat([gen._host_logits(x) for x in outputs], dim=2)
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print("PREFILL_BUCKET", json.dumps(row), flush=True)
                del outputs
        largest = references[max(references)]
        report["correctness"] = []
        for count, actual in references.items():
            expected = largest[:, :, :count]
            pcc = torch.corrcoef(torch.stack([actual.flatten(), expected.flatten()]))[0, 1].item()
            report["correctness"].append(
                dict(
                    count=count,
                    exact=torch.equal(actual, expected),
                    pcc=pcc,
                    top1_equal=torch.equal(actual.argmax(-1), expected.argmax(-1)),
                )
            )
            if pcc < 0.99999:
                raise AssertionError(f"Bucket mismatch at {count}: {pcc}")
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("BUCKET_CORRECTNESS", json.dumps(report["correctness"]), flush=True)
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
