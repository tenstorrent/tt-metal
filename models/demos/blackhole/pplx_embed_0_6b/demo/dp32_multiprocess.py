# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Multi-process DP=32 pplx-embed-v1-0.6B benchmark.

Each chip runs in its own subprocess with TT_VISIBLE_DEVICES isolation,
eliminating ttnn's internal dispatch serialization. Workers synchronize
via a multiprocessing.Barrier after warmup so build/compile phases don't
overlap with measurement. Each worker is pinned to dedicated CPU cores.

The benchmark loop runs the full Generator pipeline (prefill trace +
RMSNorm post-processing + D2H + host extraction) to measure end-to-end
latency including all post-processing overhead.

Supports all (batch_size, seq_len) combinations defined in WORKLOAD_CONFIGS
with their optimized env-var settings.

Usage:
    python models/demos/blackhole/pplx_embed_0_6b/demo/dp32_multiprocess.py
    python models/demos/blackhole/pplx_embed_0_6b/demo/dp32_multiprocess.py \\
        --batch-size 8 --seq-len 512 --num-devices 32
    python models/demos/blackhole/pplx_embed_0_6b/demo/dp32_multiprocess.py \\
        --batch-size 32 --seq-len 2048 --iterations 5
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import statistics
import sys
import time


def _worker(
    chip_id: int,
    batch_size: int,
    seq_len: int,
    iterations: int,
    warmup: int,
    barrier: mp.Barrier,
    result_dict: dict,
    cores_per_worker: int,
) -> None:
    """Single-chip: build model, warmup, barrier-sync, measure, report."""
    os.environ["TT_VISIBLE_DEVICES"] = str(chip_id)

    total_cores = os.cpu_count() or 64
    start_core = (chip_id * cores_per_worker) % total_cores
    affinity = set(range(start_core, min(start_core + cores_per_worker, total_cores)))
    try:
        os.sched_setaffinity(0, affinity)
    except (OSError, AttributeError):
        pass

    try:
        from models.demos.blackhole.pplx_embed_0_6b.demo._common import (
            apply_workload_env,
            build_single_device_model,
            generate_synthetic_inputs,
        )

        apply_workload_env(batch_size, seq_len)

        import ttnn

        wall0 = time.perf_counter()

        t_open0 = time.perf_counter()
        device = ttnn.open_device(
            device_id=0,
            l1_small_size=32768,
            trace_region_size=200_000_000,
            num_command_queues=1,
        )
        open_sec = time.perf_counter() - t_open0

        t_build0 = time.perf_counter()
        generator, model_args, kv_caches, page_table = build_single_device_model(
            device, batch_size=batch_size, seq_len=seq_len
        )
        build_sec = time.perf_counter() - t_build0

        input_ids, prompt_lens = generate_synthetic_inputs(model_args.tokenizer, batch_size, seq_len)

        t_compile0 = time.perf_counter()
        generator.prefill_forward_text(
            input_ids,
            page_table=page_table,
            kv_cache=kv_caches,
            prompt_lens=prompt_lens,
            enable_trace=True,
            return_hidden_states=True,
            warmup_prefill=True,
        )
        compile_sec = time.perf_counter() - t_compile0

        for _ in range(warmup):
            generator.prev_page_table = None
            generator.prefill_forward_text(
                input_ids,
                page_table=page_table,
                kv_cache=kv_caches,
                prompt_lens=prompt_lens,
                enable_trace=True,
                return_hidden_states=True,
                warmup_prefill=False,
            )
            ttnn.synchronize_device(device)

        # ---- All workers rendezvous here before measurement ----
        barrier.wait()

        device_secs: list[float] = []
        for _ in range(iterations):
            generator.prev_page_table = None
            t0 = time.perf_counter()
            generator.prefill_forward_text(
                input_ids,
                page_table=page_table,
                kv_cache=kv_caches,
                prompt_lens=prompt_lens,
                enable_trace=True,
                return_hidden_states=True,
                warmup_prefill=False,
            )
            ttnn.synchronize_device(device)
            device_secs.append(time.perf_counter() - t0)

        try:
            ttnn.synchronize_device(device)
        except Exception:
            pass
        try:
            ttnn.close_device(device)
        except Exception:
            pass

        result_dict[chip_id] = {
            "chip": chip_id,
            "ok": True,
            "open_sec": open_sec,
            "build_sec": build_sec,
            "compile_sec": compile_sec,
            "warmup_iters": warmup,
            "measured_iters": iterations,
            "device_sec": statistics.mean(device_secs),
            "device_sec_min": min(device_secs),
            "device_sec_max": max(device_secs),
            "device_sec_median": statistics.median(device_secs),
            "device_secs": device_secs,
            "total_sec": time.perf_counter() - wall0,
        }
    except Exception as exc:
        result_dict[chip_id] = {
            "chip": chip_id,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _orchestrate(args: argparse.Namespace) -> None:
    n = args.num_devices
    batch_size = args.batch_size
    seq_len = args.seq_len
    iterations = args.iterations
    warmup = args.warmup

    total_cores = os.cpu_count() or 64
    cores_per_worker = max(1, total_cores // n)

    print(
        f"Multi-process DP={n} pplx-embed-v1-0.6B (bs={batch_size}, ISL={seq_len})\n"
        f"  warmup={warmup}  measured_iters={iterations}\n"
        f"  CPU cores: {total_cores} total, {cores_per_worker}/worker\n"
        f"  Barrier sync: all workers rendezvous after warmup\n"
        f"  Full pipeline: trace + RMSNorm post-proc + D2H + host extraction\n"
        f"  Each worker: TT_VISIBLE_DEVICES=<chip>, open_device(0), "
        f"independent model + trace\n"
    )

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(n)
    manager = ctx.Manager()
    result_dict = manager.dict()

    wall0 = time.perf_counter()
    procs: list[mp.Process] = []
    for i in range(n):
        p = ctx.Process(
            target=_worker,
            args=(i, batch_size, seq_len, iterations, warmup, barrier, result_dict, cores_per_worker),
            name=f"chip-{i}",
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=600)
        if p.is_alive():
            print(f"  WARNING: {p.name} hung — terminating")
            p.terminate()
            p.join(timeout=10)

    wall_sec = time.perf_counter() - wall0

    results = []
    for i in range(n):
        if i in result_dict:
            results.append(dict(result_dict[i]))
        else:
            results.append({"chip": i, "ok": False, "error": "no result (crash?)"})

    results.sort(key=lambda r: r.get("chip", 0))

    for r in results:
        chip = r.get("chip", "?")
        if r.get("ok"):
            print(
                f"  chip {chip:>2}: mean={r['device_sec']*1000:.1f}ms "
                f"median={r['device_sec_median']*1000:.1f}ms "
                f"min={r['device_sec_min']*1000:.1f}ms "
                f"max={r['device_sec_max']*1000:.1f}ms"
            )
        else:
            print(f"  chip {chip:>2}: FAILED - {r.get('error', '')[:200]}")

    oks = [r for r in results if r.get("ok")]
    bads = [r for r in results if not r.get("ok")]

    if not oks:
        print(f"\nAll {n} workers failed!")
        for r in bads:
            print(f"  chip {r.get('chip')}: {r.get('error', r)}")
        sys.exit(1)

    def _mean(key: str) -> float:
        return sum(float(r[key]) for r in oks) / len(oks)

    global_batch = batch_size * len(oks)
    device_mean = _mean("device_sec")
    device_median = _mean("device_sec_median")
    device_min = min(r["device_sec_min"] for r in oks)
    device_max = max(r["device_sec_max"] for r in oks)
    slowest_mean = max(r["device_sec"] for r in oks)
    slowest_median = max(r["device_sec_median"] for r in oks)

    emb_per_sec_mean = global_batch / slowest_median
    emb_per_sec_best = global_batch / device_min
    tok_per_sec_mean = global_batch * seq_len / slowest_median
    tok_per_sec_best = global_batch * seq_len / device_min

    print()
    print("=" * 60)
    print(f"  pplx-embed-v1-0.6B Multi-Process DP={len(oks)}")
    print("=" * 60)
    print(f"  Batch size (per chip):  {batch_size}")
    print(f"  Chips active:           {len(oks)}/{n}")
    print(f"  Global batch size:      {global_batch}")
    print(f"  Input seq length:       {seq_len}")
    print(f"  Warmup iters:           {warmup}")
    print(f"  Measured iters:         {iterations}")
    print(f"  CPU pinning:            {cores_per_worker} cores/worker")
    print(f"  Mode:                   full pipeline (trace + post-proc + D2H)")
    print("-" * 60)
    print(f"  Avg open_device time:   {_mean('open_sec'):.1f}s")
    print(f"  Avg model build time:   {_mean('build_sec'):.1f}s")
    print(f"  Avg compile time:       {_mean('compile_sec'):.2f}s")
    print("-" * 60)
    print(f"  Per-chip mean:          {device_mean * 1000:.1f}ms")
    print(f"  Per-chip median:        {device_median * 1000:.1f}ms")
    print(f"  Per-chip min:           {device_min * 1000:.1f}ms")
    print(f"  Per-chip max:           {device_max * 1000:.1f}ms")
    print(f"  Slowest chip (mean):    {slowest_mean * 1000:.1f}ms")
    print(f"  Slowest chip (median):  {slowest_median * 1000:.1f}ms")
    print("-" * 60)
    print(f"  Throughput (median):    {emb_per_sec_mean:.0f} embeddings/s  " f"({tok_per_sec_mean:.0f} tokens/s)")
    print(f"  Throughput (best):      {emb_per_sec_best:.0f} embeddings/s  " f"({tok_per_sec_best:.0f} tokens/s)")
    print("-" * 60)
    print(f"  Wall time (all):        {wall_sec:.1f}s")
    print("=" * 60)

    if bads:
        print(f"\nFailed chips ({len(bads)}):")
        for r in bads:
            print(f"  chip {r.get('chip')}: {r.get('error', r)[:300]}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Multi-process DP pplx-embed-v1-0.6B benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Supported (batch-size, seq-len) combinations with optimized settings:
  (1, 512)    L1-backed activations
  (8, 512)    L1-backed activations (TT_BATCHED_L1_PREFILL)
  (32, 512)   DRAM-resident + big matmul grid
  (1, 1024)   DRAM-resident + big matmul grid
  (1, 2048)   DRAM-resident + big matmul grid
  (8, 1024)   DRAM-resident + big matmul grid
  (8, 2048)   DRAM-resident + big matmul grid
  (32, 1024)  DRAM-resident + big matmul grid
  (32, 2048)  DRAM-resident + big matmul grid
""",
    )
    p.add_argument("--num-devices", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1, help="Per-chip batch size (default: 1)")
    p.add_argument("--seq-len", type=int, default=512, help="Input sequence length (default: 512)")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()
    _orchestrate(args)


if __name__ == "__main__":
    main()
