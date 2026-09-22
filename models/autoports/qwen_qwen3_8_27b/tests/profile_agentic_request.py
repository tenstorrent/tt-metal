# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Profile sequential B1 requests with agent-like changing prompt lengths.

This is deliberately a serving-boundary probe, not a kernel benchmark.  Each
shape is followed by an exact repeat so trace recapture can be separated from
prefill execution.  Synchronization between phases makes the phase totals
diagnostic; the separately reported end-to-end time retains the normal adapter
boundary.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator, configure_fabric
from models.autoports.qwen_qwen3_8_27b.tt.generator_vllm import Qwen38ForCausalLM


def timed(mesh, function):
    ttnn.synchronize_device(mesh)
    begin = time.perf_counter()
    result = function()
    ttnn.synchronize_device(mesh)
    return result, time.perf_counter() - begin


def power_chunks(length):
    """Partition a prompt into reusable power-of-two shapes, capped at 4096."""
    chunks = []
    while length:
        count = min(4096, 1 << (length.bit_length() - 1))
        chunks.append(count)
        length -= count
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="4096,4096,4512,4512,8192,8192,9350,9350")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--context", type=int, default=12288)
    parser.add_argument("--reduced", action="store_true", help="Use layers 0 and 3 for a harness smoke test")
    parser.add_argument(
        "--prewarm",
        action="store_true",
        help="Warm every unique >4096 shape before measuring cross-shape trace reuse",
    )
    parser.add_argument(
        "--power-chunks",
        action="store_true",
        help="Run exact prompts as a sequence of reusable power-of-two prefill shapes",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lengths = [int(value) for value in args.lengths.split(",")]
    if not lengths or min(lengths) < 1 or max(lengths) + args.decode_steps >= args.context:
        parser.error("Lengths and decode steps must fit strictly inside context")

    root = Path("models/autoports/qwen_qwen3_8_27b")
    # Shape fixture only: keep tokenization and quality outside this timing probe.
    raw = list(range(100, 356))
    prompts = [(raw * ((length + len(raw) - 1) // len(raw)))[:length] for length in lengths]
    params = SimpleNamespace(temperature=[0.0], top_k=[1], top_p=[0.0], seed=[17])

    torch.set_num_threads(8)
    configure_fabric(payload_bytes=8192)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=134217728)
    generator = adapter = None
    report = {
        "scope": "synchronized serving-phase diagnostic; not a throughput benchmark",
        "lengths": lengths,
        "decode_steps": args.decode_steps,
        "context": args.context,
        "reduced": args.reduced,
        "prewarm": args.prewarm,
        "power_chunks": args.power_chunks,
        "prewarm_rows": [],
        "rows": [],
    }
    try:
        generator = build_generator(root, mesh, layer_indices=[0, 3] if args.reduced else None)
        adapter = Qwen38ForCausalLM(generator, 1, args.context)
        pages = (args.context + 31) // 32
        cache = adapter.allocate_kv_cache((pages, 1, 32, 256), None, len(generator.model.layers))
        page_table = torch.arange(pages, dtype=torch.int32).reshape(1, pages)

        original_capture = generator._capture
        capture_times = []

        def measured_capture(*capture_args, **capture_kwargs):
            begin = time.perf_counter()
            result = original_capture(*capture_args, **capture_kwargs)
            capture_times.append(time.perf_counter() - begin)
            return result

        generator._capture = measured_capture

        def prefill_and_sample(tokens, length, table):
            if not args.power_chunks:
                return generator.serving_prefill_tokens(
                    tokens,
                    page_table=table,
                    kv_cache=cache,
                    prompt_lens=[length],
                    start_pos=[0],
                    slots=[0],
                )
            output = None
            start = 0
            for count in power_chunks(length):
                [output] = generator.prefill_forward(
                    tokens[:, start : start + count],
                    page_table=table,
                    kv_cache=cache,
                    prompt_lens=[count],
                    start_pos=[start],
                    slots=[0],
                )
                start += count
            generator._sampling_step(output)
            return generator.tokens

        if args.prewarm:
            warmups = dict(zip(lengths, prompts))
            if args.power_chunks:
                # 8191 exercises every power-of-two shape from 4096 through 1.
                length = min(8191, args.context - args.decode_steps - 1)
                warmups = {length: (raw * ((length + len(raw) - 1) // len(raw)))[:length]}
            for length, prompt in warmups.items():
                if length <= 4096:
                    continue
                before = generator.counters.copy()
                tokens = torch.tensor([prompt], dtype=torch.int64)
                _, reset_s = timed(mesh, lambda: generator.reset_recurrent_slots([0]))
                _, sampling_s = timed(mesh, lambda: adapter._sampling(params, reset=True, output_positions=[length]))
                table = adapter._table(page_table, [0])
                first_device, prefill_s = timed(
                    mesh,
                    lambda: prefill_and_sample(tokens, length, table),
                )
                _, read_first_s = timed(
                    mesh,
                    lambda: adapter.process_decode_output_host(
                        adapter.read_decode_output(first_device), is_tokens=True
                    ),
                )
                report["prewarm_rows"].append(
                    {
                        "prompt_length": length,
                        "reset_s": reset_s,
                        "sampling_setup_s": sampling_s,
                        "prefill_and_first_sample_s": prefill_s,
                        "first_token_read_s": read_first_s,
                        "counters": dict(Counter(generator.counters) - before),
                    }
                )
                print("AGENTIC_REQUEST_PREWARM", json.dumps(report["prewarm_rows"][-1], sort_keys=True), flush=True)
        for index, (length, prompt) in enumerate(zip(lengths, prompts)):
            before = generator.counters.copy()
            capture_begin = len(capture_times)
            tokens = torch.tensor([prompt], dtype=torch.int64)

            _, reset_s = timed(mesh, lambda: generator.reset_recurrent_slots([0]))
            _, sampling_s = timed(mesh, lambda: adapter._sampling(params, reset=True, output_positions=[length]))
            table = adapter._table(page_table, [0])
            first_device, prefill_s = timed(
                mesh,
                lambda: prefill_and_sample(tokens, length, table),
            )
            first_host, read_first_s = timed(
                mesh,
                lambda: adapter.process_decode_output_host(adapter.read_decode_output(first_device), is_tokens=True),
            )
            adapter._decode_bound = False
            decoded, first_decode_s = timed(
                mesh,
                lambda: adapter.decode_forward(
                    first_host,
                    torch.tensor([length]),
                    page_table,
                    cache,
                    sampling_params=params,
                    reset_batch=True,
                    read_from_device=True,
                ),
            )
            steady_times = []
            for step in range(1, args.decode_steps):
                decoded, elapsed = timed(
                    mesh,
                    lambda decoded=decoded, step=step: adapter.decode_forward(
                        decoded,
                        torch.tensor([length + step]),
                        page_table,
                        cache,
                        sampling_params=params,
                        reset_batch=False,
                        read_from_device=True,
                    ),
                )
                steady_times.append(elapsed)

            counters = Counter(generator.counters) - before
            row_captures = capture_times[capture_begin:]
            report["rows"].append(
                {
                    "index": index,
                    "prompt_length": length,
                    "same_shape_as_previous": index > 0 and lengths[index - 1] == length,
                    "full_4096_chunks": length // 4096,
                    "tail_tokens": length % 4096,
                    "reset_s": reset_s,
                    "sampling_setup_s": sampling_s,
                    "prefill_and_first_sample_s": prefill_s,
                    "first_token_read_s": read_first_s,
                    "first_decode_s": first_decode_s,
                    "capture_s": row_captures,
                    "steady_decode_step_s": steady_times,
                    "steady_decode_tokens_per_s": len(steady_times) / sum(steady_times),
                    "phase_total_s": reset_s
                    + sampling_s
                    + prefill_s
                    + read_first_s
                    + first_decode_s
                    + sum(steady_times),
                    "counters": dict(counters),
                }
            )
            print("AGENTIC_REQUEST_PROFILE", json.dumps(report["rows"][-1], sort_keys=True), flush=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if adapter is not None:
            adapter.close()
        elif generator is not None:
            generator.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
