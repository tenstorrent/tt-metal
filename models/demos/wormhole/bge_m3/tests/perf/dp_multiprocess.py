# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multi-process DP BGE-M3 benchmark (one process per chip).

Modeled on models/demos/blackhole/pplx_embed_0_6b/demo/dp32_multiprocess.py.

Each chip runs in its own subprocess with TT_VISIBLE_DEVICES isolation, so we
sidestep ttnn's single-process mesh-dispatch serialization (the FDMeshCommandQueue
bottleneck) AND the 32-chip teardown hang we hit with the mesh-based path.
Each worker:
  * pins itself to a dedicated CPU core range (os.sched_setaffinity)
  * open_device(0) -> a single 1x1 device
  * builds a single-chip BGE-M3 model (create_tt_model), captures a trace
  * warms up, then rendezvous on a multiprocessing.Barrier so build/compile
    of slow workers never overlaps the measurement window
  * measures `iterations` of the full H2D -> Forward(trace) -> D2H pipeline,
    using the optimized copy_device_to_torch D2H fast path
  * reports per-chip latency stats back via a shared dict

The orchestrator aggregates per-chip timings into global throughput. Global
batch = per_chip_batch * num_chips.

Usage:
    cd /home/tt-admin/gtobar && source local_env.sh && cd tt-metal
    python models/demos/wormhole/bge_m3/tests/perf/dp_multiprocess.py \
        --batch-size 1  --num-devices 32 --iterations 100
    python models/demos/wormhole/bge_m3/tests/perf/dp_multiprocess.py \
        --batch-size 32 --num-devices 32 --iterations 50
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import statistics
import sys
import time

SEQ_LEN = 512
HIDDEN = 1024


def _worker(
    chip_id: int,
    batch_size: int,
    seq_len: int,
    iterations: int,
    warmup: int,
    barrier,
    result_dict,
    cores_per_worker: int,
    pooling=None,
) -> None:
    """Single-chip: build model, warmup, barrier-sync, measure, report."""
    # Isolate this worker to one physical chip BEFORE importing ttnn.
    os.environ["TT_VISIBLE_DEVICES"] = str(chip_id)

    total_cores = os.cpu_count() or 64
    start_core = (chip_id * cores_per_worker) % total_cores
    affinity = set(range(start_core, min(start_core + cores_per_worker, total_cores)))
    try:
        os.sched_setaffinity(0, affinity)
    except (OSError, AttributeError):
        # CPU pinning is best-effort; platforms without sched_setaffinity (or
        # with a restricted cpuset) just run unpinned.
        pass

    try:
        import ttnn
        from models.demos.wormhole.bge_m3.tests.perf.perf import (
            _allocate_pooled_d2h_stack,
            _d2h_step_pooled,
            allocate_device_tensors,
            copy_inputs_to_device,
            prepare_inputs,
            to_host_tensors,
        )
        from models.demos.wormhole.bge_m3.tt.common import create_tt_model

        wall0 = time.perf_counter()
        dtype = ttnn.bfloat8_b
        mask_dtype = dtype if batch_size in (1, 32) else ttnn.bfloat16

        # ── Open one device (this process only sees chip `chip_id` as id 0) ──
        t_open0 = time.perf_counter()
        device = ttnn.open_device(
            device_id=0,
            l1_small_size=32768,
            trace_region_size=50_000_000,
            num_command_queues=1,
        )
        open_sec = time.perf_counter() - t_open0

        # ── Build single-chip BGE-M3 model ──────────────────────────────────
        t_build0 = time.perf_counter()
        model_args, model, _ = create_tt_model(
            mesh_device=device,
            max_batch_size=batch_size,
            max_seq_len=seq_len,
            dtype=dtype,
            pooling=pooling,
        )
        build_sec = time.perf_counter() - t_build0

        # ── Inputs + persistent device tensors ──────────────────────────────
        host_inputs = prepare_inputs(model_args.tokenizer, batch_size, seq_len, model_args.pad_token_id)
        host_tensors = to_host_tensors(host_inputs, mask_dtype)
        device_tensors = allocate_device_tensors(host_tensors, device)

        # ── Compile + capture trace ─────────────────────────────────────────
        t_compile0 = time.perf_counter()
        out = model.forward(**device_tensors)
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)
        trace_out = model.capture_trace(**device_tensors, mesh_device=device, cq_id=0)
        compile_sec = time.perf_counter() - t_compile0

        # ── Optimized D2H staging (untilize -> dram -> copy_device_to_torch) ─
        # Sized from the actual trace_out shape so it works for raw encoder
        # output [B,1,S,H] AND any pooling head (cls/mean -> [B,1,1,H],
        # colbert -> [B,1,S,H]).
        pooled_s = int(trace_out.shape[2])
        dram_staging, dest_torch = _allocate_pooled_d2h_stack(trace_out, batch_size, pooled_s, HIDDEN)

        # ── Warmup the full pipeline ────────────────────────────────────────
        for _ in range(warmup):
            copy_inputs_to_device(host_tensors, device_tensors)
            ttnn.synchronize_device(device)
            model.execute_trace(blocking=True)
            _d2h_step_pooled(trace_out, dram_staging, dest_torch)

        # ── All workers rendezvous before measurement ───────────────────────
        barrier.wait()

        h2d_keys = ("input_ids", "attention_mask", "token_type_ids", "position_ids")
        iter_secs, h2d_secs, fwd_secs, d2h_secs = [], [], [], []
        for _ in range(iterations):
            t0 = time.perf_counter()
            # H2D: refresh the device input slots, sync so the timing is real.
            for k in h2d_keys:
                ttnn.copy_host_to_device_tensor(host_tensors[k], device_tensors[k])
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            # Forward: blocking trace execute (= dispatch + run + sync).
            model.execute_trace(blocking=True)
            t2 = time.perf_counter()
            # D2H: untilize -> DRAM staging -> copy_device_to_torch.
            _d2h_step_pooled(trace_out, dram_staging, dest_torch)
            t3 = time.perf_counter()
            h2d_secs.append(t1 - t0)
            fwd_secs.append(t2 - t1)
            d2h_secs.append(t3 - t2)
            iter_secs.append(t3 - t0)

        model.release_trace()
        try:
            ttnn.synchronize_device(device)
            ttnn.close_device(device)
        except Exception as exc:  # noqa: BLE001
            # Teardown is best-effort -- a failed close shouldn't mask results.
            # Surface it on stderr for debugging without failing the worker.
            print(f"[chip {chip_id}] device teardown error (ignored): {exc}", file=sys.stderr)

        result_dict[chip_id] = {
            "chip": chip_id,
            "ok": True,
            "open_sec": open_sec,
            "build_sec": build_sec,
            "compile_sec": compile_sec,
            "warmup_iters": warmup,
            "measured_iters": iterations,
            "iter_sec": statistics.mean(iter_secs),
            "iter_sec_min": min(iter_secs),
            "iter_sec_max": max(iter_secs),
            "iter_sec_median": statistics.median(iter_secs),
            "h2d_sec": statistics.mean(h2d_secs),
            "fwd_sec": statistics.mean(fwd_secs),
            "d2h_sec": statistics.mean(d2h_secs),
            "h2d_sec_median": statistics.median(h2d_secs),
            "fwd_sec_median": statistics.median(fwd_secs),
            "d2h_sec_median": statistics.median(d2h_secs),
            "total_sec": time.perf_counter() - wall0,
        }
    except Exception as exc:  # noqa: BLE001
        import traceback

        result_dict[chip_id] = {
            "chip": chip_id,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-1500:],
        }


def _orchestrate(args: argparse.Namespace) -> None:
    n = args.num_devices
    batch_size = args.batch_size
    seq_len = args.seq_len
    iterations = args.iterations
    warmup = args.warmup
    pooling = None if args.pooling == "none" else args.pooling

    total_cores = os.cpu_count() or 64
    cores_per_worker = max(1, total_cores // n)

    print(
        f"Multi-process DP={n} BGE-M3 (bs={batch_size}, ISL={seq_len}, pooling={args.pooling})\n"
        f"  warmup={warmup}  measured_iters={iterations}\n"
        f"  CPU cores: {total_cores} total, {cores_per_worker}/worker\n"
        f"  Barrier sync: all workers rendezvous after warmup\n"
        f"  Full pipeline: H2D -> Forward(trace) -> D2H (copy_device_to_torch)\n"
        f"  Each worker: TT_VISIBLE_DEVICES=<chip>, open_device(0), independent model + trace\n"
    )

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(n)
    manager = ctx.Manager()
    result_dict = manager.dict()

    wall0 = time.perf_counter()
    procs = []
    for i in range(n):
        p = ctx.Process(
            target=_worker,
            args=(i, batch_size, seq_len, iterations, warmup, barrier, result_dict, cores_per_worker, pooling),
            name=f"chip-{i}",
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=900)
        if p.is_alive():
            print(f"  WARNING: {p.name} hung -- terminating")
            p.terminate()
            p.join(timeout=10)

    wall_sec = time.perf_counter() - wall0

    results = []
    for i in range(n):
        results.append(
            dict(result_dict[i]) if i in result_dict else {"chip": i, "ok": False, "error": "no result (crash?)"}
        )
    results.sort(key=lambda r: r.get("chip", 0))

    for r in results:
        chip = r.get("chip", "?")
        if r.get("ok"):
            print(
                f"  chip {chip:>2}: iter={r['iter_sec_median']*1000:.2f}ms "
                f"[H2D={r['h2d_sec_median']*1000:.2f} "
                f"Fwd={r['fwd_sec_median']*1000:.2f} "
                f"D2H={r['d2h_sec_median']*1000:.2f}]ms "
                f"min={r['iter_sec_min']*1000:.2f} max={r['iter_sec_max']*1000:.2f}"
            )
        else:
            print(f"  chip {chip:>2}: FAILED - {r.get('error', '')[:200]}")
            if r.get("trace"):
                print(r["trace"])

    oks = [r for r in results if r.get("ok")]
    if not oks:
        print(f"\nAll {n} workers failed!")
        sys.exit(1)

    def _mean(key):
        return sum(float(r[key]) for r in oks) / len(oks)

    global_batch = batch_size * len(oks)
    # Throughput is gated by the SLOWEST chip (all run in lockstep after barrier).
    slowest_median = max(r["iter_sec_median"] for r in oks)
    fastest_min = min(r["iter_sec_min"] for r in oks)

    emb_per_sec_median = global_batch / slowest_median
    emb_per_sec_best = global_batch / fastest_min
    tok_per_sec_median = global_batch * seq_len / slowest_median

    print()
    print("=" * 64)
    print(f"  BGE-M3 Multi-Process DP={len(oks)}  (S={seq_len})")
    print("=" * 64)
    print(f"  Batch per chip:         {batch_size}")
    print(f"  Chips active:           {len(oks)}/{n}")
    print(f"  Global batch:           {global_batch}")
    print(f"  Warmup / measured:      {warmup} / {iterations}")
    print(f"  CPU pinning:            {cores_per_worker} cores/worker")
    print("-" * 64)
    print(f"  Avg open_device:        {_mean('open_sec'):.1f}s")
    print(f"  Avg model build:        {_mean('build_sec'):.1f}s")
    print(f"  Avg compile+trace:      {_mean('compile_sec'):.2f}s")
    print("-" * 64)
    print(f"  Per-chip mean:          {_mean('iter_sec')*1000:.2f}ms")
    print(f"  Per-chip median:        {_mean('iter_sec_median')*1000:.2f}ms")
    print(f"  Slowest chip (median):  {slowest_median*1000:.2f}ms")
    print(f"  Fastest chip (min):     {fastest_min*1000:.2f}ms")
    print("-" * 64)
    # Per-stage breakdown (averaged across all chips, using each chip's median).
    h2d_ms = _mean("h2d_sec_median") * 1000
    fwd_ms = _mean("fwd_sec_median") * 1000
    d2h_ms = _mean("d2h_sec_median") * 1000
    tot_ms = h2d_ms + fwd_ms + d2h_ms
    pct = lambda x: (x / tot_ms * 100.0) if tot_ms > 0 else 0.0
    print(f"  Stage breakdown (avg of per-chip medians):")
    print(f"    H2D:                  {h2d_ms:6.3f} ms  ({pct(h2d_ms):4.1f}%)")
    print(f"    Forward:              {fwd_ms:6.3f} ms  ({pct(fwd_ms):4.1f}%)")
    print(f"    D2H:                  {d2h_ms:6.3f} ms  ({pct(d2h_ms):4.1f}%)")
    print(f"    Sum:                  {tot_ms:6.3f} ms")
    print("-" * 64)
    print(f"  Throughput (median):    {emb_per_sec_median:.1f} emb/s  ({tok_per_sec_median:.0f} tok/s)")
    print(f"  Throughput (best):      {emb_per_sec_best:.1f} emb/s")
    print(f"  Orchestration wall:     {wall_sec:.1f}s")
    print("=" * 64)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-size", type=int, default=1, help="per-chip batch size")
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    p.add_argument("--num-devices", type=int, default=32)
    p.add_argument(
        "--pooling",
        type=str,
        default="none",
        choices=["none", "cls", "mean", "colbert"],
        help="pooling head: none (raw encoder, default), cls, mean, or colbert",
    )
    # Defaults match perf.py: NUM_ITERATIONS=10 measured, 3 trace warmups.
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    _orchestrate(_parse_args())
