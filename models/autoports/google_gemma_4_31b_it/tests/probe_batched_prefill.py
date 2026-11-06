# SPDX-License-Identifier: Apache-2.0
"""Reduced-layer real-weight exactness probe; no serving performance claim."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import ttnn

from ..tt.batched_prefill import PreparedBatchedPrefill
from ..tt.model import Gemma4Model
from ..tt.prefill import PreparedPrefill


def host(x):
    return torch.stack([ttnn.to_torch(value) for value in ttnn.get_device_tensors(x)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lengths", default="128")
    parser.add_argument("--output")
    parser.add_argument("--max-group-rows", type=int, default=1024)
    parser.add_argument("--timing-repeats", type=int, default=0)
    args = parser.parse_args()
    if args.timing_repeats < 0 or args.timing_repeats == 1:
        raise ValueError("Timing requires zero or at least two repetitions")
    torch.set_num_threads(8)
    torch.manual_seed(729)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    try:
        model = Gemma4Model(mesh, layer_indices=(0, 5))
        caches, _ = model.allocate_cache(batch=args.batch, context=2048)
        solo = PreparedPrefill(model, 2048, model.host_page_tables)
        batched = PreparedBatchedPrefill(solo, args.batch, max_rows=args.max_group_rows)
        reports = []
        for length in map(int, args.lengths.split(",")):
            size = (length + 31) // 32 * 32
            if (args.batch, size) not in batched.inputs:
                raise ValueError(f"Direct batched probe has no prepared bucket {(args.batch, size)}")
            tokens = torch.randint(3, model.config.vocab_size, (args.batch, length), dtype=torch.int32)
            outputs = batched.run(tokens, [length] * args.batch, list(range(args.batch)), size=size, kv_cache=caches)
            actual_logits = [host(value).clone() for value in outputs]
            del outputs
            actual_cache = [[host(value).clone() for value in pair] for pair in caches]
            results = []
            for row in range(args.batch):
                solo.stage(tokens[row], slot=row, start=0, size=size, valid=length)
                hidden = solo.hidden(size=size, start=0, kv_cache=caches)
                reference = host(model.terminal(solo.last_row(hidden)))
                del hidden
                results.append(
                    {
                        "row": row,
                        "logits_equal_per_device": [torch.equal(a, b) for a, b in zip(reference, actual_logits[row])],
                        "max_logit_delta": (reference.float() - actual_logits[row].float()).abs().max().item(),
                    }
                )
            reference_cache = [[host(value) for value in pair] for pair in caches]
            for row, result in enumerate(results):
                result["cache_equal"] = []
                for index, pages in enumerate(model.host_page_tables):
                    ids = pages[row, : size // 32].long()
                    for kind in range(2):
                        a = actual_cache[index][kind][:, ids].permute(0, 2, 1, 3, 4).flatten(2, 3)[:, :, :length]
                        b = reference_cache[index][kind][:, ids].permute(0, 2, 1, 3, 4).flatten(2, 3)[:, :, :length]
                        result["cache_equal"].append(
                            {
                                "layer": model.layer_indices[index],
                                "kind": "KV"[kind],
                                "equal_per_device": [torch.equal(x, y) for x, y in zip(a, b)],
                            }
                        )
            report = {"batch": args.batch, "length": length, "max_group_rows": args.max_group_rows, "results": results}
            reports.append(report)
            print(json.dumps(report), flush=True)
            if args.output:
                Path(args.output).write_text(json.dumps(reports, indent=2) + "\n")
            assert all(
                all(r["logits_equal_per_device"]) and all(all(c["equal_per_device"]) for c in r["cache_equal"])
                for r in results
            )
            if args.timing_repeats:
                group = max(
                    count
                    for count, rows in batched.inputs
                    if rows == size and count <= args.batch and count * size <= 1024
                )
                if args.batch % group:
                    raise ValueError("Timing control must partition the candidate exactly")

                def execute(mode):
                    count = args.batch if mode == "candidate" else group
                    for start in range(0, args.batch, count):
                        outputs = batched.run(
                            tokens[start : start + count],
                            [length] * count,
                            list(range(start, start + count)),
                            size=size,
                            kv_cache=caches,
                        )
                        del outputs

                programs_before = mesh.num_program_cache_entries()
                for mode in ("control", "candidate"):
                    execute(mode)
                    ttnn.synchronize_device(mesh)
                programs_warmed = mesh.num_program_cache_entries()
                timings = {"control": [], "candidate": []}
                sequence = []
                for _ in range(args.timing_repeats):
                    for mode in ("control", "candidate", "candidate", "control"):
                        ttnn.synchronize_device(mesh)
                        begin = time.perf_counter()
                        execute(mode)
                        ttnn.synchronize_device(mesh)
                        elapsed_ms = (time.perf_counter() - begin) * 1000
                        timings[mode].append(elapsed_ms)
                        sequence.append({"mode": mode, "ms": elapsed_ms, "programs": mesh.num_program_cache_entries()})
                report["timing"] = {
                    "scope": "reduced real layers 0,5; not full-model or serving performance",
                    "boundary": "synchronized host wall: input refresh, prefill, last-row terminal and output slicing; no sampling",
                    "control_group_rows": group,
                    "candidate_group_rows": args.batch,
                    "logical_prompt_tokens": length,
                    "physical_prompt_tokens": size,
                    "samples_ms": timings,
                    "median_ms": {mode: statistics.median(values) for mode, values in timings.items()},
                    "sequence": sequence,
                    "programs_before_warmup": programs_before,
                    "programs_after_warmup": programs_warmed,
                    "programs_after_timing": mesh.num_program_cache_entries(),
                }
                if args.output:
                    Path(args.output).write_text(json.dumps(reports, indent=2) + "\n")
                print(
                    json.dumps(
                        {"batch": args.batch, "length": length, "timing_median_ms": report["timing"]["median_ms"]}
                    ),
                    flush=True,
                )
                assert all(item["programs"] == programs_warmed for item in sequence), "Late compilation during timing"

    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
