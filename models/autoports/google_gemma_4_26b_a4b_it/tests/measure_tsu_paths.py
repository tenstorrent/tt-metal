# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare full-context generator paths; profiling is restricted to two layers."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import build_generator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-length", type=int, default=4096)
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cache-context", type=int, default=262144)
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--head-blocks", nargs="+", type=int, help="Unprofiled same-process full-generator head controls"
    )
    args = parser.parse_args()
    if (args.profile or os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "0") and not args.reduced:
        parser.error("Device profiling must use --reduced, never the full model")
    if args.head_blocks and (args.profile or os.environ.get("TT_METAL_DEVICE_PROFILER", "0") != "0"):
        parser.error("Head generator controls must run without device profiling")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_root = Path(__file__).resolve().parent.parent
    report = {
        "config": vars(args) | {"output": str(args.output)},
        "source_sha256": {
            str(path.relative_to(model_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__).resolve(), *sorted((model_root / "tt").glob("*.py"))]
        },
        "rows": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1000000000)
    gen = None
    try:
        kwargs = {"layer_indices": (0, 5)} if args.reduced else {}
        gen = build_generator(None, mesh, max_seq_len=args.cache_context, **kwargs)
        gen._standalone_cache(args.cache_context)
        prompt = [2] + [100] * (args.input_length - 1)
        if args.head_blocks:
            expected = None
            for block in args.head_blocks:
                gen._release_trace()
                gen.model.head_program.in0_block_w = block
                for repeat in range(args.repeats + 1):
                    tokens = gen.generate(prompt, args.output_length, stop_on_eos=False, buffer_tokens=True)
                    if expected is None:
                        expected = tokens
                    row = {
                        "path": "head_control_buffered_generator",
                        "head_k_block": block,
                        "head_program": str(gen.model.head_program),
                        "repeat": repeat,
                        "warmup": repeat == 0,
                        "tokens": tokens,
                        "tokens_match_control": tokens == expected,
                        "metrics": gen.metrics,
                    }
                    report["rows"].append(row)
                    save()
                    assert row["tokens_match_control"], row
                    print("TSU_HEAD_GENERATOR", json.dumps(row), flush=True)
            return
        if args.profile:
            from tracy import signpost

            gen.generate(prompt, 4, stop_on_eos=False)
            gen._copy(torch.zeros(1, dtype=torch.int32), gen.output_index, "request_output_refreshes")
            ttnn.synchronize_device(mesh)
            signpost("PERF_DECODE")
            gen._replay()
            ttnn.execute_trace(mesh, gen.output_trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            signpost("PERF_DECODE_END")
            ttnn.ReadDeviceProfiler(mesh)
            report["profile_scope"] = "Real layers0/5 plus terminal, sampler and token recorder; context262144"
            save()
            return

        expected = None
        for buffered in (True, False):
            for repeat in range(args.repeats + 1):
                started = time.perf_counter()
                tokens = gen.generate(prompt, args.output_length, stop_on_eos=False, buffer_tokens=buffered)
                row = {
                    "path": "buffered_generator" if buffered else "token_out_generator",
                    "repeat": repeat,
                    "warmup": repeat == 0,
                    "e2el_ms": (time.perf_counter() - started) * 1000,
                    "tokens": tokens,
                    "metrics": gen.metrics,
                }
                if expected is None:
                    expected = tokens
                row["tokens_match_control"] = tokens == expected
                assert row["tokens_match_control"], row
                report["rows"].append(row)
                save()
                print("TSU_GENERATOR", json.dumps(row), flush=True)

        for repeat in range(args.repeats + 1):
            # Rebuild prompt KV and first token before each timing interval.
            # Input refreshes are deliberately outside the device-queue interval.
            tokens = gen.generate(prompt, args.output_length, stop_on_eos=False)
            ids = torch.zeros((1, 1, 1, 32), dtype=torch.int32)
            ids.flatten()[0] = tokens[0]
            positions = torch.full((1, 32), -1, dtype=torch.int32)
            positions[0, 0] = args.input_length
            gen._copy(ids, gen.tokens, "token_refreshes")
            gen._copy(positions, gen.positions, "position_refreshes")
            gen._copy(
                torch.tensor([args.input_length], dtype=torch.int32), gen.cache_positions, "cache_position_refreshes"
            )
            ttnn.synchronize_device(mesh)
            started = time.perf_counter()
            for _ in range(args.output_length - 1):
                gen._replay()
            ttnn.synchronize_device(mesh)
            elapsed_ms = (time.perf_counter() - started) * 1000
            row = {
                "path": "queued_model_and_sampler_no_readback",
                "repeat": repeat,
                "warmup": repeat == 0,
                "elapsed_ms": elapsed_ms,
                "mean_step_ms": elapsed_ms / (args.output_length - 1),
                "tsu": (args.output_length - 1) * 1000 / elapsed_ms,
                "final_token": int(gen._read_tokens()[0]),
            }
            assert row["final_token"] == expected[-1], row
            report["rows"].append(row)
            save()
            print("TSU_QUEUED", json.dumps(row), flush=True)
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
