# SPDX-License-Identifier: Apache-2.0
"""Host-wall bucket calibration; not device timing or vLLM serving evidence."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="all", help="all or explicit real layer indices")
    parser.add_argument("--repeats", type=int, default=2, help="Number of ABBA rounds")
    parser.add_argument("--buckets", help="Optional comma-separated BxS subset, e.g. 8x128,32x32")
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--pool-tokens", type=int, default=1048576)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError("Calibration needs at least two ABBA rounds")
    torch.set_num_threads(8)
    torch.manual_seed(729)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "full60 real-layer calibration" if args.layers == "all" else "reduced real-layer calibration",
        "layers": args.layers,
        "context_tokens": args.context,
        "pool_tokens": args.pool_tokens,
        "configured_batch": 32,
        "timing_boundary": "synchronized host wall including input refresh, prefill hidden, terminal and output extraction; no sampler",
        "not_device_time": True,
        "not_serving_or_headline_performance": True,
        "exactness_source": "separate batched-prefill acceptance probes; no cache readback in calibration",
        "status": "running",
        "buckets": [],
    }

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n")

    try:
        model = Gemma4Model(
            mesh, layer_indices=None if args.layers == "all" else tuple(map(int, args.layers.split(",")))
        )
        caches, _ = model.allocate_cache(batch=32, context=args.context, pool_tokens=args.pool_tokens)
        solo = PreparedPrefill(model, args.context, model.host_page_tables)
        batched = PreparedBatchedPrefill(solo, 32)
        buckets = list(batched.inputs)
        if args.buckets:
            buckets = [tuple(map(int, item.split("x"))) for item in args.buckets.split(",")]
            if any(bucket not in batched.inputs for bucket in buckets):
                raise ValueError("Requested bucket is not prepared by the production cap1024 policy")
        report["prepared_buckets"] = list(batched.inputs)
        report["precision_config"] = model.precision_config
        for batch, size in buckets:
            tokens = torch.randint(3, model.config.vocab_size, (batch, size), dtype=torch.int32)

            def execute(mode):
                if mode == "batched":
                    outputs = batched.run(tokens, [size] * batch, list(range(batch)), size=size, kv_cache=caches)
                    del outputs
                else:
                    for row in range(batch):
                        solo.stage(tokens[row], slot=row, start=0, size=size, valid=size)
                        hidden = solo.hidden(size=size, start=0, kv_cache=caches)
                        last = solo.last_row(hidden)
                        logits = model.terminal(last)
                        del hidden, last, logits

            programs_before = mesh.num_program_cache_entries()
            for mode in ("solo", "batched"):
                execute(mode)
                ttnn.synchronize_device(mesh)
            programs_warmed = mesh.num_program_cache_entries()
            samples = {"solo": [], "batched": []}
            sequence = []
            for repeat in range(args.repeats):
                for mode in ("solo", "batched", "batched", "solo"):
                    ttnn.synchronize_device(mesh)
                    begin = time.perf_counter()
                    execute(mode)
                    ttnn.synchronize_device(mesh)
                    elapsed_ms = (time.perf_counter() - begin) * 1000
                    samples[mode].append(elapsed_ms)
                    sequence.append(
                        {
                            "round": repeat,
                            "mode": mode,
                            "host_wall_ms": elapsed_ms,
                            "program_count": mesh.num_program_cache_entries(),
                        }
                    )
            medians = {mode: statistics.median(values) for mode, values in samples.items()}
            stable = all(row["program_count"] == programs_warmed for row in sequence)
            result = {
                "batch": batch,
                "logical_prompt_tokens": size,
                "physical_prompt_tokens": size,
                "total_physical_rows": batch * size,
                "median_host_wall_ms": medians,
                "batched_over_solo_ratio": medians["batched"] / medians["solo"],
                "batched_faster": medians["batched"] < medians["solo"],
                "samples": sequence,
                "program_count_before_warmup": programs_before,
                "program_count_after_warmup": programs_warmed,
                "program_cache_stable": stable,
            }
            report["buckets"].append(result)
            save()
            print(
                json.dumps(
                    {
                        key: result[key]
                        for key in (
                            "batch",
                            "logical_prompt_tokens",
                            "median_host_wall_ms",
                            "batched_over_solo_ratio",
                            "program_cache_stable",
                        )
                    }
                ),
                flush=True,
            )
            if not stable:
                raise AssertionError("Program-cache growth during warmed bucket calibration")
        report["status"] = "complete"
        save()
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
