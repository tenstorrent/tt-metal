#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Measure all requested input lengths for one chunk on one loaded prefill model.

Fixed 131072-token capacity and two allocated slots match the P/D owner.
No reference model, golden files, PCC checks or decode traffic are used.
"""
import argparse
import json
import math
import os
import resource
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

LENGTHS = [1024, 2048, 4096, 5120, 8192, 16384, 32768, 56320, 65536]


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk", type=int, required=True, choices=[1024, 4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    cases = []
    for count in LENGTHS:
        if count < args.chunk:
            continue
        prompt = json.loads((args.prompts / f"{count}.json").read_text())
        assert len(prompt["token_ids"]) == prompt["input_tokens"] == count
        out = args.output / f"{count//1024}k-c{args.chunk//1024}k"
        out.mkdir(parents=True, exist_ok=False)
        row = dict(
            status="TO DO",
            input_tokens=count,
            chunk=args.chunk,
            layers=36,
            slots=2,
            capacity=131072,
            active_slots=1,
            runs=[],
            prompt_sha256=prompt["sha256_token_ids"],
            numerical_validation="not run, as requested",
        )
        save(out / "case.json", row)
        cases.append((prompt, out, row))
    if os.getenv("GPT_OSS_WEIGHTS_FROM_CACHE") != "1":
        raise RuntimeError("sweep must reuse the verified weight cache")
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != hard:
        resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
    start = time.perf_counter()

    def event(name, **values):
        item = dict(
            event=name,
            utc=datetime.now(timezone.utc).isoformat(),
            epoch=time.time(),
            elapsed_s=time.perf_counter() - start,
            chunk=args.chunk,
            **values,
        )
        with args.events.open("a") as f:
            f.write(json.dumps(item) + "\n")
        print("PREFILL_SWEEP_EVENT " + json.dumps(item), flush=True)

    event("process_started")
    from models.demos.gpt_oss_d_p.tt.model_config import ModelArgs
    from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    import ttnn

    event("imports_finished")
    assert ttnn.get_num_devices() == 32
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(4, 8))
    current = None
    try:
        event("mesh_opened")
        model_args = ModelArgs(mesh_device=mesh)
        assert model_args.hf_config.num_hidden_layers == 36
        cache = model_args.weight_cache_path(ttnn.bfloat8_b)
        if not cache.is_dir():
            raise RuntimeError("verified weight cache missing")
        event("model_build_started", cache=str(cache))
        cfg = TtPrefillRuntimeConfig(
            num_layers=36,
            max_seq_len=131072,
            mesh_shape=(4, 8),
            default_chunk_size=args.chunk,
            num_users=2,
            expert_weight_dtype=ttnn.bfloat4_b,
            cache_dtype=ttnn.bfloat8_b,
            weight_cache_path=cache,
            owns_kv_cache=True,
            topology=ttnn.Topology.Linear,
            bounded_sliding_kv_cache=False,
        )
        runtime = TtPrefillRuntime(mesh, model_args.hf_config, {}, cfg)
        event("model_built")
        runtime.compile()
        ttnn.synchronize_device(mesh)
        event("compile_finished")
        for prompt, out, row in cases:
            current = (out, row)
            if time.time() + 120 >= args.deadline:
                raise TimeoutError("near serving deadline")
            count = prompt["input_tokens"]
            steps = math.ceil(count / args.chunk)
            processed = steps * args.chunk
            padded = prompt["token_ids"] + [0] * (processed - count)
            row.update(status="IN PROGRESS", utc=datetime.now(timezone.utc).isoformat(), processed_tokens=processed)
            save(out / "case.json", row)
            for index in range(4):
                row.update(current_run=index, current_run_is_warmup=index == 0)
                save(out / "case.json", row)
                t0 = time.perf_counter()
                epoch = time.time()
                for step in range(steps):
                    a = step * args.chunk
                    inp = runtime.make_chunk_input(padded[a : a + args.chunk])
                    runtime.prefill_chunk(inp, slot_id=0, actual_start=a, actual_end=min(a + args.chunk, count))
                ttnn.synchronize_device(mesh)
                elapsed = time.perf_counter() - t0
                result = dict(
                    index=index,
                    measured=index > 0,
                    status="DONE",
                    start_epoch=epoch,
                    prefill_span_s=elapsed,
                    prefill_active_s=elapsed,
                    total_s=elapsed,
                    prefill_tps=count / elapsed,
                    prefill_processed_tps=processed / elapsed,
                    input_tokens=count,
                    processed_tokens=processed,
                    ttft_s=None,
                    kv_payload_bytes=None,
                    transfer_sum_s=None,
                    transfer_span_s=None,
                    overlap_s=None,
                    nonoverlap_prefill_s=None,
                    nonoverlap_transfer_s=None,
                    initial_prefill_s=None,
                    final_transfer_tail_s=None,
                    decode_interval_s=None,
                    tsu=None,
                )
                row["runs"].append(result)
                save(out / "case.json", row)
                event(
                    "iteration_finished", input_tokens=count, **{k: v for k, v in result.items() if k != "input_tokens"}
                )
            measured = [x for x in row["runs"] if x["measured"]]
            keys = ["prefill_span_s", "prefill_active_s", "total_s", "prefill_tps", "prefill_processed_tps"]
            row.update(
                status="DONE",
                warmup_s=row["runs"][0]["total_s"],
                mean={k: statistics.mean(x[k] for x in measured) for k in keys},
                sample_variance={k: statistics.variance(x[k] for x in measured) for k in keys},
                completed_utc=datetime.now(timezone.utc).isoformat(),
                definitions="Host input conversion + chunk calls + one final device synchronization; all prompt tokens computed; no prefix reuse; fixed 131072 capacity / 2 slots",
            )
            save(out / "case.json", row)
    except Exception as error:
        if current:
            out, row = current
            row.update(status="FAILED", error=repr(error))
            save(out / "case.json", row)
        event("failed", error=repr(error))
        raise
    finally:
        ttnn.close_mesh_device(mesh)
        event("mesh_closed")


if __name__ == "__main__":
    main()
