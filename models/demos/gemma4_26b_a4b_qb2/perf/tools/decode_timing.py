"""WS0 baseline: native traced decode timing for the Gemma 4 26B autoport on one QB2.

Batch 1 uses Gemma4Generator.generate (its own metrics). Batches > 1 use the scheduler path
(prefill_forward + decode_forward with device feedback), timed over N replays after warmup.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.demos.gemma4_26b_a4b_qb2.tt.generator import build_generator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batches", default="1,8,16,32")
    p.add_argument("--b1-prompts", default="128,4096")
    p.add_argument("--steps", type=int, default=32)
    args = p.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1_000_000_000)
    report = {"steps": args.steps, "results": []}
    gen = None
    try:
        gen = build_generator(None, mesh, max_seq_len=8192)
        report["precision"] = gen.model.precision_config.get("config_id")
        for b in [int(x) for x in args.batches.split(",")]:
            if b == 1:
                for plen in [int(x) for x in args.b1_prompts.split(",")]:
                    prompt = [2] + [100 + (i % 50) for i in range(plen - 1)]
                    runs = []
                    for _ in range(3):  # first run compiles/captures; keep the last two
                        gen.generate(prompt, 128, stop_on_eos=False)
                        runs.append(dict(gen.metrics))
                    row = {"batch": 1, "prompt": plen, "new_tokens": 128, "metrics_warm": runs[1:]}
                    print("WS0", json.dumps(row), flush=True)
                    report["results"].append(row)
                gen._release_trace()
                continue
            caches, table = gen.model.allocate_cache(slots=b, context=512)
            plen = 127
            prompts = torch.stack([torch.full((plen,), 100 + s, dtype=torch.int64) for s in range(b)])
            gen.prefill_forward(
                prompts, page_table=table, kv_cache=caches, prompt_lens=[plen] * b, slots=list(range(b))
            )
            pos = torch.full((b,), plen, dtype=torch.int32)
            gen.decode_forward(torch.full((b,), 100), pos, page_table=table, kv_cache=caches)  # capture
            pos = pos + 1
            for _ in range(3):  # warm replays
                gen.decode_forward(None, pos, page_table=table, kv_cache=caches, device_feedback=True)
                pos = pos + 1
            ttnn.synchronize_device(mesh)
            per_step = []
            t0 = time.perf_counter()
            for _ in range(args.steps):
                s = time.perf_counter()
                gen.decode_forward(None, pos, page_table=table, kv_cache=caches, device_feedback=True)
                ttnn.synchronize_device(mesh)
                per_step.append((time.perf_counter() - s) * 1000)
                pos = pos + 1
            total = (time.perf_counter() - t0) * 1000
            row = {
                "batch": b,
                "prompt": plen,
                "step_ms_median": statistics.median(per_step),
                "step_ms_mean": total / args.steps,
                "tok_s_user": 1000 / statistics.median(per_step),
                "tok_s_total": b * 1000 / statistics.median(per_step),
            }
            print("WS0", json.dumps(row), flush=True)
            report["results"].append(row)
            gen._release_trace()
            del caches
    finally:
        args.output.write_text(json.dumps(report, indent=1))
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
