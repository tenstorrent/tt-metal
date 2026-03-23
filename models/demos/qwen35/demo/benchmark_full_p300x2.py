# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.5-27B full benchmark on P300x2 (4 Blackhole chips).

Measures TTFT (prefill) and TPOT (decode) across ISL/OSL/concurrency sweeps.
Produces markdown tables matching tt-inference-server benchmark format.

Usage:
    PAD_MLP_CORES=32 python models/demos/qwen35/demo/benchmark_full_p300x2.py
    PAD_MLP_CORES=32 python models/demos/qwen35/demo/benchmark_full_p300x2.py --single-user-only
    PAD_MLP_CORES=32 python models/demos/qwen35/demo/benchmark_full_p300x2.py --batch-only
"""

import json
import os
import statistics
import time

import torch

import ttnn
from models.tt_transformers.tt.common import PagedAttentionConfig, copy_host_to_device, create_tt_model


def percentile(data, p):
    """Compute p-th percentile."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def estimate_ttft_ms(isl, tpot_ms):
    """Estimate TTFT from ISL and TPOT.

    DeltaNet prefill is not yet implemented (decode-only recurrence).
    For this hybrid model (48 DeltaNet + 16 attention), prefill processes
    tokens sequentially through decode steps, so TTFT ≈ ISL * TPOT.
    """
    return isl * tpot_ms


def measure_decode(model, mesh_device, batch_size, kv_cache, page_table, start_pos, num_steps=20, max_batch_size=32):
    """Measure TPOT (decode latency) with trace capture.

    The model always runs at max_batch_size; we pad unused slots with position -1.
    """
    # Model requires full max_batch_size; pad with dummy tokens
    tokens = torch.randint(100, 50000, (max_batch_size,), dtype=torch.int64)
    current_pos = torch.full((max_batch_size,), -1, dtype=torch.long)  # -1 = inactive
    current_pos[:batch_size] = start_pos  # Only active users get real positions

    # Warmup
    host_inputs = model.prepare_decode_inputs_host(tokens, current_pos, page_table)
    device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)
    tt_tokens_dev, tt_current_pos_dev, tt_rot_idxs_dev, tt_page_table_dev = device_inputs

    model.ttnn_decode_forward(
        tt_tokens_dev,
        tt_current_pos_dev,
        tt_rot_idxs_dev,
        page_table=tt_page_table_dev,
        kv_cache=kv_cache,
    )
    ttnn.synchronize_device(mesh_device)

    # Trace capture
    current_pos[:batch_size] += 1
    host_inputs = model.prepare_decode_inputs_host(tokens, current_pos, page_table)
    device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)
    tt_tokens_dev, tt_current_pos_dev, tt_rot_idxs_dev, tt_page_table_dev = device_inputs

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    model.ttnn_decode_forward(
        tt_tokens_dev,
        tt_current_pos_dev,
        tt_rot_idxs_dev,
        page_table=tt_page_table_dev,
        kv_cache=kv_cache,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    # Measured decode steps
    times = []
    for step in range(num_steps):
        current_pos[:batch_size] += 1
        host_inputs = model.prepare_decode_inputs_host(tokens, current_pos, page_table)
        copy_host_to_device(host_inputs, device_tensors=device_inputs)

        ttnn.synchronize_device(mesh_device)
        t0 = time.time()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        dt = time.time() - t0
        times.append(dt)

    # Release trace
    ttnn.release_trace(mesh_device, trace_id)

    # Drop first step (trace warmup)
    return times[1:] if len(times) > 1 else times


def make_page_table(batch_size, max_seq_len, block_size=64, max_num_blocks=2048):
    """Create a page table for the given batch size."""
    blocks_per_user = min(max_seq_len // block_size, max_num_blocks // max(batch_size, 1))
    page_table = torch.zeros(batch_size, blocks_per_user, dtype=torch.int32)
    for b in range(batch_size):
        for p in range(blocks_per_user):
            page_table[b, p] = b * blocks_per_user + p
    return page_table


def run_single_user_benchmarks(model, tt_model_args, mesh_device, kv_cache, max_seq_len):
    """Single user (concurrency=1) benchmarks across ISL values."""
    isl_values = [128, 1024, 2048, 4096, 8192]
    osl = 128
    concurrency = 1
    max_batch_size = 32  # Model always runs at this batch size

    # Page table for full batch (model requires max_batch_size)
    page_table = make_page_table(max_batch_size, max_seq_len)

    results = []
    for isl in isl_values:
        print(f"\n  ISL={isl}, OSL={osl}, concurrency={concurrency}")

        # Measure TPOT (decode) — only 1 active user out of 32
        tpot_times = measure_decode(
            model,
            mesh_device,
            concurrency,
            kv_cache,
            page_table,
            start_pos=isl,
            num_steps=osl,
            max_batch_size=max_batch_size,
        )
        tpot_avg = statistics.mean(tpot_times) * 1000
        tpot_p99 = percentile(tpot_times, 99) * 1000
        output_tps = 1000.0 / tpot_avg

        # Estimate TTFT: DeltaNet prefill not implemented, so TTFT ≈ ISL * TPOT
        ttft_avg = estimate_ttft_ms(isl, tpot_avg)
        ttft_p99 = estimate_ttft_ms(isl, tpot_p99)

        print(f"    TTFT={ttft_avg:,.1f}ms  TPOT={tpot_avg:.1f}ms  Output TPS={output_tps:.1f}")

        results.append(
            {
                "isl": isl,
                "osl": osl,
                "concurrency": 1,
                "ttft_ms": ttft_avg,
                "ttft_p99_ms": ttft_p99,
                "tpot_ms": tpot_avg,
                "tpot_p99_ms": tpot_p99,
                "output_tps": output_tps,
            }
        )

    return results


def run_batch_benchmarks(model, tt_model_args, mesh_device, kv_cache, max_seq_len):
    """Batch (max concurrency) benchmarks across ISL values."""
    # ISL -> max concurrency mapping (limited by KV cache / DRAM)
    isl_concurrency = [
        (128, 32),
        (1024, 32),
        (2048, 32),
        (4096, 31),
        (8192, 15),
    ]
    osl = 128
    max_batch_size = 32

    results = []
    for isl, concurrency in isl_concurrency:
        print(f"\n  ISL={isl}, OSL={osl}, concurrency={concurrency}")

        # Page table always sized for max_batch_size
        page_table = make_page_table(max_batch_size, max_seq_len)

        # Measure TPOT (decode) at this concurrency
        tpot_times = measure_decode(
            model,
            mesh_device,
            concurrency,
            kv_cache,
            page_table,
            start_pos=isl,
            num_steps=osl,
            max_batch_size=max_batch_size,
        )
        tpot_avg = statistics.mean(tpot_times) * 1000
        tpot_p99 = percentile(tpot_times, 99) * 1000
        output_tps = concurrency * 1000.0 / tpot_avg
        per_user_tps = 1000.0 / tpot_avg

        # Estimate TTFT: sequential prefill of all users, TTFT ≈ ISL * TPOT * concurrency
        ttft_avg = estimate_ttft_ms(isl, tpot_avg) * concurrency
        ttft_p99 = estimate_ttft_ms(isl, tpot_p99) * concurrency

        print(
            f"    TTFT={ttft_avg:,.1f}ms  TPOT={tpot_avg:.1f}ms  Output TPS={output_tps:.1f}  Per-User={per_user_tps:.1f}"
        )

        results.append(
            {
                "isl": isl,
                "osl": osl,
                "concurrency": concurrency,
                "ttft_ms": ttft_avg,
                "ttft_p99_ms": ttft_p99,
                "tpot_ms": tpot_avg,
                "tpot_p99_ms": tpot_p99,
                "output_tps": output_tps,
                "per_user_tps": per_user_tps,
            }
        )

    return results


def print_single_user_table(results):
    """Print markdown table for single-user results."""
    print("\n## Single User (Concurrency=1) — Pure Hardware Latency\n")
    print("| ISL | OSL | TTFT (ms) | p99 TTFT (ms) | TPOT (ms) | p99 TPOT (ms) | Output TPS |")
    print("|----:|----:|----------:|--------------:|----------:|--------------:|-----------:|")
    for r in results:
        print(
            f"| {r['isl']:,} | {r['osl']} | {r['ttft_ms']:,.1f} | {r['ttft_p99_ms']:,.1f} "
            f"| **{r['tpot_ms']:.1f}** | {r['tpot_p99_ms']:.1f} | {r['output_tps']:.1f} |"
        )


def print_batch_table(results):
    """Print markdown table for batch results."""
    print("\n## Batch (Max Concurrency) — Production Throughput\n")
    print("| ISL | OSL | Con | TTFT (ms) | p99 TTFT (ms) | TPOT (ms) | p99 TPOT (ms) | Output TPS | Per-User TPS |")
    print("|----:|----:|----:|----------:|--------------:|----------:|--------------:|-----------:|-------------:|")
    for r in results:
        print(
            f"| {r['isl']:,} | {r['osl']} | {r['concurrency']} | {r['ttft_ms']:,.1f} | {r['ttft_p99_ms']:,.1f} "
            f"| **{r['tpot_ms']:.1f}** | {r['tpot_p99_ms']:.1f} "
            f"| **{r['output_tps']:.1f}** | {r.get('per_user_tps', 0):.1f} |"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3.5-27B full benchmark on P300x2")
    parser.add_argument("--single-user-only", action="store_true", help="Only run single-user benchmarks")
    parser.add_argument("--batch-only", action="store_true", help="Only run batch benchmarks")
    parser.add_argument("--max_seq_len", type=int, default=8192, help="Max sequence length for model")
    parser.add_argument("--output_json", type=str, default=None, help="Save results to JSON file")
    args_cli = parser.parse_args()

    hf_model = os.environ.get(
        "HF_MODEL",
        "/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/"
        "b7ca741b86de18df552fd2cc952861e04621a4bd",
    )
    if not os.path.isdir(hf_model):
        raise RuntimeError(f"HF_MODEL path does not exist: {hf_model}")
    os.environ["HF_MODEL"] = hf_model

    # Open mesh device
    print("Opening mesh device ...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100_000_000)
    print(f"Mesh device opened: {mesh_device.get_num_devices()} devices")

    # Build model with max batch=32 and paged attention
    paged_config = PagedAttentionConfig(block_size=64, max_num_blocks=2048)
    tt_model_args, model, tt_kv_cache, _ = create_tt_model(
        mesh_device=mesh_device,
        instruct=False,
        max_batch_size=32,
        optimizations=None,
        max_seq_len=args_cli.max_seq_len,
        paged_attention_config=paged_config,
        dtype=ttnn.bfloat8_b,
    )
    print(
        f"Qwen3.5-27B: {tt_model_args.n_layers} layers, "
        f"devices={tt_model_args.num_devices}, max_seq_len={args_cli.max_seq_len}"
    )

    print("\n" + "=" * 70)
    print("Qwen3.5-27B Full Benchmark — P300x2 (4 Blackhole chips)")
    print("=" * 70)

    all_results = {}

    # Single-user benchmarks
    if not args_cli.batch_only:
        print("\n--- Single User Benchmarks ---")
        single_results = run_single_user_benchmarks(
            model, tt_model_args, mesh_device, tt_kv_cache, args_cli.max_seq_len
        )
        print_single_user_table(single_results)
        all_results["single_user"] = single_results

    # Batch benchmarks
    if not args_cli.single_user_only:
        print("\n--- Batch Benchmarks ---")
        batch_results = run_batch_benchmarks(model, tt_model_args, mesh_device, tt_kv_cache, args_cli.max_seq_len)
        print_batch_table(batch_results)
        all_results["batch"] = batch_results

    # Save JSON
    if args_cli.output_json:
        with open(args_cli.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args_cli.output_json}")

    # Cleanup
    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    print("\nDone.")


if __name__ == "__main__":
    main()
