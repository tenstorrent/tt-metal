# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measure real device batches at the serving adapter, with repeated warm shapes."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator, configure_fabric
from models.autoports.qwen_qwen3_8_27b.tt.generator_vllm import Qwen38ForCausalLM
from models.autoports.qwen_qwen3_8_27b.tt.precision import decoder_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--lengths", default="128,4096,32768")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--context", type=int)
    parser.add_argument("--pool-tokens", type=int)
    parser.add_argument("--distinct-prompts", action="store_true")
    parser.add_argument("--compare-prefill", action="store_true", help="Compare serial and grouped prefill in one load")
    parser.add_argument("--compare-fused-mlp", action="store_true", help="Fuse prefill MLP only; decode unchanged")
    parser.add_argument(
        "--compare-packed-swiglu",
        action="store_true",
        help="Replace packed-MLP slices and multiply with the exact packed consumer during prefill",
    )
    parser.add_argument("--compare-single-step", action="store_true", help="Experimental B16 decode recurrence")
    parser.add_argument("--compare-compact-mlp", action="store_true", help="Compare compact decode MLP rows")
    parser.add_argument("--compare-decode-layouts", action="store_true", help="Compare cumulative decode layouts")
    parser.add_argument("--compare-sharded-prefill", action="store_true", help="Prefill-only sharded residual")
    parser.add_argument(
        "--compare-row-parallel-norm",
        action="store_true",
        help="Compare row-parallel norm against selected layout/head baseline",
    )
    parser.add_argument("--replicated-prefill-norm", action="store_true", help="Gather residual before unchanged norm")
    parser.add_argument(
        "--prefill-head-optimizations", action="store_true", help="Combine batched final head and skip unused heads"
    )
    parser.add_argument(
        "--compare-sdpa", action="store_true", help="Compare per-user and batched SDPA with grouped prefill"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        sum(
            (
                args.compare_prefill,
                args.compare_sdpa,
                args.compare_fused_mlp,
                args.compare_packed_swiglu,
                args.compare_single_step,
                args.compare_compact_mlp,
                args.compare_decode_layouts,
                args.compare_sharded_prefill,
                args.compare_row_parallel_norm,
            )
        )
        > 1
    ):
        parser.error("Select only one comparison per run")
    if args.compare_single_step and args.batch != 16:
        parser.error("The experimental one-step model path is restricted to batch 16")
    if (args.compare_compact_mlp or args.compare_decode_layouts) and args.batch < 2:
        parser.error("Compact MLP comparison requires a multi-request batch")
    if args.replicated_prefill_norm and not args.compare_sharded_prefill:
        parser.error("--replicated-prefill-norm requires --compare-sharded-prefill")
    if args.prefill_head_optimizations and not args.compare_sharded_prefill:
        parser.error("--prefill-head-optimizations requires --compare-sharded-prefill")
    if args.steps < 2:
        parser.error("At least two decode steps are needed to separate first-use and steady execution")
    lengths = list(map(int, args.lengths.split(",")))
    context = args.context or ((max(lengths) + args.steps + 31) // 32) * 32
    pool_tokens = args.pool_tokens or args.batch * context
    if context < max(lengths) + args.steps or pool_tokens // args.batch < max(lengths) + args.steps:
        parser.error("Each request must fit in both context and its disjoint share of the KV pool")
    torch.set_num_threads(8)
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=134217728)
    adapter = None
    report = dict(batch=args.batch, context=context, pool_tokens=pool_tokens, steps=args.steps, rows=[])
    try:

        def policy(precision, layer):
            return {
                **decoder_policy(precision, layer),
                "minimal_mlp": args.compare_fused_mlp,
                "prefill_sharded_residual": args.compare_sharded_prefill or args.compare_row_parallel_norm,
                "prefill_replicated_norm": args.replicated_prefill_norm or args.compare_row_parallel_norm,
            }

        with patch("models.autoports.qwen_qwen3_8_27b.tt.model.decoder_policy", side_effect=policy):
            generator = build_generator(Path("models/autoports/qwen_qwen3_8_27b"), mesh)
        if (args.compare_compact_mlp or args.compare_decode_layouts) and not generator.model.compact_decode_residual:
            raise ValueError("Compact MLP comparison requires QWEN_COMPACT_DECODE_RESIDUAL=1")
        for layer in generator.model.layers:
            layer.policy["minimal_mlp"] = False
        adapter = Qwen38ForCausalLM(generator, args.batch, context)
        pages = (context + 31) // 32
        # Match the vLLM worker's one-extra-block-per-user allocation. Dividing
        # a token budget among users before rounding pages can otherwise leave
        # the final decode positions mapped to an unused zero page-table entry.
        physical_pages = (pool_tokens + 31) // 32 + args.batch
        cache = adapter.allocate_kv_cache((physical_pages, 1, 32, 256), None, 64)
        per_user = min(pages, physical_pages // args.batch)
        if per_user * 32 < max(lengths) + args.steps:
            raise ValueError("The disjoint per-request page allocation must cover every decode position")
        table = torch.zeros(args.batch, pages, dtype=torch.int32)
        table[:, :per_user] = torch.arange(args.batch * per_user, dtype=torch.int32).reshape(args.batch, per_user)
        params = SimpleNamespace(
            temperature=[0.0] * args.batch,
            top_k=[1] * args.batch,
            top_p=[0.0] * args.batch,
            seed=[17] * args.batch,
        )
        for length in lengths:
            tokens = (torch.arange(length).remainder(256) + 100).repeat(args.batch, 1)
            if args.distinct_prompts:
                tokens += torch.arange(args.batch).reshape(-1, 1) * 13
            for trial in range(
                8
                if args.compare_decode_layouts
                else (
                    6
                    if args.compare_row_parallel_norm
                    else (
                        4
                        if args.compare_prefill
                        or args.compare_sdpa
                        or args.compare_fused_mlp
                        or args.compare_packed_swiglu
                        or args.compare_single_step
                        or args.compare_compact_mlp
                        or args.compare_sharded_prefill
                        or args.compare_row_parallel_norm
                        else 2
                    )
                )
            ):
                repeat = trial % 2
                if args.compare_decode_layouts and repeat == 0:
                    generator._release_traces()
                    for layer in generator.model.layers:
                        layer.policy["compact_decode_mlp"] = trial >= 2
                        layer.policy["batched_decode_rope"] = trial >= 4
                        layer.policy["compact_decode_attention"] = trial >= 6
                if args.compare_compact_mlp and repeat == 0:
                    generator._release_traces()
                    for layer in generator.model.layers:
                        layer.policy["compact_decode_mlp"] = trial >= 2
                if args.compare_row_parallel_norm and repeat == 0:
                    generator._release_traces()
                    generator.prefill_signatures.clear()
                    generator.batched_prefill = True
                    generator.model.prefill_sharded_residual = True
                    generator.model.prefill_batched_head = True
                    generator.skip_intermediate_prefill_head = True
                    for layer in generator.model.layers:
                        layer.policy["prefill_row_parallel_norm"] = 2 <= trial < 4
                if args.compare_sharded_prefill and repeat == 0:
                    generator._release_traces()
                    generator.prefill_signatures.clear()
                    generator.batched_prefill = True
                    generator.model.prefill_sharded_residual = trial >= 2
                    generator.model.prefill_batched_head = args.prefill_head_optimizations and trial >= 2
                    generator.skip_intermediate_prefill_head = args.prefill_head_optimizations and trial >= 2
                if args.compare_single_step and repeat == 0:
                    generator._release_traces()
                    generator.batched_prefill = True
                    for layer in generator.model.layers:
                        layer.policy["experimental_single_step_recurrence"] = trial >= 2
                if args.compare_prefill and repeat == 0:
                    generator._release_traces()
                    generator.prefill_signatures.clear()
                    generator.batched_prefill = trial >= 2
                if args.compare_sdpa and repeat == 0:
                    generator._release_traces()
                    generator.prefill_signatures.clear()
                    generator.batched_prefill = True
                    for layer in generator.model.layers:
                        layer.policy["batched_prefill_sdpa"] = trial >= 2
                if args.compare_fused_mlp:
                    generator._release_traces()
                    generator.batched_prefill = True
                    for layer in generator.model.layers:
                        layer.policy["minimal_mlp"] = trial >= 2
                if args.compare_packed_swiglu and repeat == 0:
                    generator._release_traces()
                    generator.prefill_signatures.clear()
                    generator.batched_prefill = True
                    for layer in generator.model.layers:
                        layer.policy["prefill_packed_swiglu"] = trial >= 2
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                decoded, _ = adapter.prefill_forward(
                    tokens,
                    table,
                    cache,
                    [length] * args.batch,
                    sampling_params=params,
                )
                ttnn.synchronize_device(mesh)
                prefill = time.perf_counter() - begin
                if args.compare_fused_mlp:
                    for layer in generator.model.layers:
                        layer.policy["minimal_mlp"] = False
                outputs = [decoded.reshape(-1).tolist()]
                times = []
                for step in range(args.steps):
                    begin = time.perf_counter()
                    decoded = adapter.decode_forward(
                        decoded,
                        torch.full((args.batch,), length + step),
                        table,
                        cache,
                        sampling_params=params,
                        reset_batch=step == 0,
                    )
                    ttnn.synchronize_device(mesh)
                    times.append(time.perf_counter() - begin)
                    outputs.append(decoded.reshape(-1).tolist())
                row = dict(
                    compact_decode_residual=generator.model.compact_decode_residual,
                    compact_decode_mlp=generator.model.layers[0].policy.get("compact_decode_mlp", False),
                    batched_decode_rope=generator.model.layers[0].policy.get("batched_decode_rope", False),
                    compact_decode_attention=generator.model.layers[0].policy.get("compact_decode_attention", False),
                    batched_prefill=generator.batched_prefill,
                    fused_prefill_mlp=args.compare_fused_mlp and trial >= 2,
                    packed_prefill_swiglu=args.compare_packed_swiglu and trial >= 2,
                    sharded_prefill=args.compare_row_parallel_norm or (args.compare_sharded_prefill and trial >= 2),
                    row_parallel_norm=args.compare_row_parallel_norm and 2 <= trial < 4,
                    selected_baseline=args.compare_row_parallel_norm,
                    trial=trial,
                    replicated_prefill_norm=args.replicated_prefill_norm or args.compare_row_parallel_norm,
                    prefill_head_optimizations=args.compare_row_parallel_norm
                    or (args.prefill_head_optimizations and trial >= 2),
                    single_step_recurrence=args.compare_single_step and trial >= 2,
                    batched_sdpa=generator.model.layers[0].policy.get("batched_prefill_sdpa", False),
                    length=length,
                    repeat=repeat,
                    prefill_s=prefill,
                    first_decode_s=times[0],
                    steady_step_s=sum(times[1:]) / len(times[1:]),
                    aggregate_decode_tps=args.batch * len(times[1:]) / sum(times[1:]),
                    tokens=outputs,
                    counters=dict(generator.counters),
                )
                report["rows"].append(row)
                if (args.compare_compact_mlp or args.compare_decode_layouts) and trial >= 1:
                    control = next(r for r in report["rows"] if r["length"] == length and r["trial"] == 0)
                    row["tokens_match_control"] = outputs == control["tokens"]
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                if (args.compare_compact_mlp or args.compare_decode_layouts) and trial >= 1:
                    assert row["tokens_match_control"], "Decode layout or repeated-run tokens differ from control"
                print("BATCH_PROFILE", json.dumps({k: v for k, v in row.items() if k != "tokens"}), flush=True)
    finally:
        if adapter is not None:
            adapter.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
