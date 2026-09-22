# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Collect in-memory TT device-profiler timings for one warm prefill.

This is a diagnostic, not a benchmark.  It drains the device profiler around
individual Python-visible TTNN operations, so its wall time is deliberately
perturbed.  The reported DEVICE KERNEL DURATION fields are produced by the
device profiler itself.  Context-only wrappers retain the decoder/layer name
without adding synchronization around whole composite Python methods.
"""

import argparse
import json
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator, configure_fabric
from models.autoports.qwen_qwen3_8_27b.tt.precision import decoder_policy

DEVICE_KERNEL = "DEVICE KERNEL DURATION [ns]"


def aggregate_perf(perf_by_chip):
    """Sum logical programs after taking the slowest of their chip replicas."""
    per_program = {}
    max_cores = 0
    for chip, programs in perf_by_chip.items():
        for program in programs:
            max_cores = max(max_cores, int(getattr(program, "core_count", 0) or 0))
            uid = program.program_execution_uid
            # Multi-chip runtime IDs encode the chip in the low ten bits:
            # logical program N appears as N*1024 + chip.  Normalize before
            # applying the standard max-across-chip aggregation.
            key = (uid.runtime_id // 1024, uid.trace_id, uid.trace_id_counter)
            slot = per_program.setdefault(key, {})
            for name, result in program.program_analyses_results.items():
                slot[name] = max(slot.get(name, 0), int(result.duration))
    analyses = defaultdict(int)
    for slot in per_program.values():
        for name, duration in slot.items():
            analyses[name] += duration
    return {
        "program_count": len(per_program),
        "max_core_count": max_cores,
        "device_kernel_sum_ns": analyses.get(DEVICE_KERNEL, 0),
        "analyses_ns": dict(analyses),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--layers", default="0,3")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--packed-swiglu", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    required_env = (
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_PROFILER_MID_RUN_DUMP",
        "TT_METAL_PROFILER_CPP_POST_PROCESS",
    )
    import os

    missing = [name for name in required_env if os.getenv(name) != "1"]
    if missing:
        raise RuntimeError("Set profiler environment variables to 1: " + ", ".join(missing))

    torch.set_num_threads(8)
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=134217728)
    gen = None
    report = {
        "batch": args.batch,
        "length": args.length,
        "layers": [int(value) for value in args.layers.split(",")],
        "warning": "Diagnostic: profiler drains perturb wall time; use device timing fields.",
        "warmups": [],
        "events": [],
        "categories": {},
    }
    context = []
    profiling_depth = 0
    event_index = 0

    def context_name():
        return "/".join(context) if context else "outer"

    def drain(label):
        nonlocal event_index
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        data = aggregate_perf(ttnn.get_latest_programs_perf_data())
        if data["program_count"]:
            data.update(index=event_index, label=label)
            report["events"].append(data)
            event_index += 1
        return data

    def enter_context(stack, owner, name, label, named=False):
        original = getattr(owner, name)

        def contextual(*call_args, **call_kwargs):
            suffix = ""
            if named:
                suffix = "." + str(call_args[1] if len(call_args) > 1 else call_kwargs["name"])
            context.append(label + suffix)
            try:
                return original(*call_args, **call_kwargs)
            finally:
                context.pop()

        stack.enter_context(patch.object(owner, name, contextual))

    def enter_profile(stack, owner, name, label=None):
        original = getattr(owner, name)
        op_label = label or name

        def profiled(*call_args, **call_kwargs):
            nonlocal profiling_depth
            if profiling_depth:
                return original(*call_args, **call_kwargs)
            # Any device work launched by a Python path we cannot patch (notably
            # Tensor.__getitem__) is kept out of the following named operation.
            drain("gap." + context_name())
            profiling_depth += 1
            begin = time.perf_counter_ns()
            try:
                result = original(*call_args, **call_kwargs)
                ttnn.synchronize_device(mesh)
                wall_ns = time.perf_counter_ns() - begin
            finally:
                profiling_depth -= 1
            data = drain(context_name() + ":" + op_label)
            if data["program_count"]:
                report["events"][-1]["wall_ns"] = wall_ns
            return result

        stack.enter_context(patch.object(owner, name, profiled))

    try:
        selected = {
            "prefill_sharded_residual": True,
            "prefill_replicated_norm": True,
            "prefill_row_parallel_norm": True,
            "prefill_packed_swiglu": args.packed_swiglu,
        }
        # The experiment branch contains the row-parallel implementation but
        # intentionally does not carry the release branch's env-policy wiring.
        # Force the exact selected release policy while the layers are built.
        with patch(
            "models.autoports.qwen_qwen3_8_27b.tt.model.decoder_policy",
            side_effect=lambda precision, layer: {**decoder_policy(precision, layer), **selected},
        ):
            gen = build_generator("models/autoports/qwen_qwen3_8_27b", mesh, layer_indices=report["layers"])
        gen.batched_prefill = True
        gen.model.prefill_sharded_residual = True
        gen.model.prefill_batched_head = True
        gen.skip_intermediate_prefill_head = True
        cache = gen._ensure_cache(args.batch, ((args.length + 31) // 32) * 32)
        tokens = (torch.arange(args.length) % 256 + 100).repeat(args.batch, 1)
        tokens += torch.arange(args.batch)[:, None] * 13
        reference = None

        def run_model_batch():
            # This is the exact inner call made by QwenGenerator's equal-length,
            # aligned batched branch.  Calling it directly makes the diagnostic
            # independent of generator routing and counters.
            ids = gen.model.upload(tokens.int(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            return gen.model.prefill_batch(
                ids,
                cache=cache,
                page_table=gen.page_table,
                length=args.length,
                start_pos=0,
                slots=list(range(args.batch)),
            )

        for warmup in range(args.warmups):
            gen.reset()
            begin = time.perf_counter()
            outputs = run_model_batch()
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - begin
            actual = torch.cat([gen._host_logits(output) for output in outputs], dim=2)
            if reference is None:
                reference = actual.clone()
            report["warmups"].append({"seconds": elapsed, "logits_exact": torch.equal(actual, reference)})
            drain("discard.warmup")
            del outputs, actual

        report["preconditions"] = {
            "generator_batched_prefill": gen.batched_prefill,
            "model_sharded_prefill": gen.model.prefill_sharded_residual,
            "model_batched_head": gen.model.prefill_batched_head,
            "direct_model_prefill_batch": True,
        }

        with ExitStack() as stack:
            for layer in gen.model.layers:
                enter_context(stack, layer, "prefill_forward", "layer." + layer.kind)
                enter_context(stack, layer, "_linear", "projection", named=True)
                enter_context(stack, layer, "_norm", "normalization", named=True)
            for name in ("embed", "rope", "logits"):
                enter_context(stack, gen.model, name, "model." + name)

            common_ops = (
                "add",
                "clone",
                "concat",
                "copy",
                "embedding",
                "generic_op",
                "linear",
                "mul",
                "pad",
                "permute",
                "repeat_interleave",
                "reshape",
                "rms_norm",
                "rms_norm_post_all_gather",
                "rms_norm_pre_all_gather",
                "sigmoid",
                "slice",
                "split",
                "to_layout",
                "to_memory_config",
                "typecast",
            )
            for name in common_ops:
                if hasattr(ttnn, name):
                    enter_profile(stack, ttnn, name, "ttnn." + name)

            experimental_ops = (
                "all_gather_async",
                "all_gather_minimal_matmul_async",
                "all_reduce_async",
                "all_to_all_async_generic",
                "minimal_matmul",
                "minimal_matmul_strided_reduce_scatter_async",
                "nlp_create_qkv_heads_decode",
                "paged_fill_cache",
                "paged_fused_update_cache",
                "reduce_scatter_minimal_async",
                "rotary_embedding",
            )
            for name in experimental_ops:
                if hasattr(ttnn.experimental, name):
                    enter_profile(stack, ttnn.experimental, name, "experimental." + name)

            transformer_ops = (
                "chunk_gated_delta_rule",
                "chunked_scaled_dot_product_attention",
                "concatenate_heads",
                "paged_scaled_dot_product_attention_decode",
                "split_query_key_value_and_split_heads",
            )
            for name in transformer_ops:
                if hasattr(ttnn.transformer, name):
                    enter_profile(stack, ttnn.transformer, name, "transformer." + name)
            if hasattr(ttnn.experimental, "kda") and hasattr(ttnn.experimental.kda, "qkv_causal_conv1d_silu"):
                enter_profile(
                    stack,
                    ttnn.experimental.kda,
                    "qkv_causal_conv1d_silu",
                    "kda.qkv_causal_conv1d_silu",
                )

            drain("discard.before_measured")
            gen.reset()
            begin = time.perf_counter()
            outputs = run_model_batch()
            ttnn.synchronize_device(mesh)
            report["instrumented_wall_seconds"] = time.perf_counter() - begin
            drain("tail." + context_name())

        actual = torch.cat([gen._host_logits(output) for output in outputs], dim=2)
        report["instrumented_logits_exact"] = torch.equal(actual, reference)

        categories = defaultdict(lambda: defaultdict(int))
        for event in report["events"]:
            slot = categories[event["label"]]
            slot["calls"] += 1
            for key in (
                "program_count",
                "device_kernel_sum_ns",
                "wall_ns",
            ):
                slot[key] += int(event.get(key, 0))
            slot["max_core_count"] = max(slot["max_core_count"], event["max_core_count"])
            for name, duration in event["analyses_ns"].items():
                slot["analyses_ns." + name] += int(duration)
        report["categories"] = {name: dict(values) for name, values in categories.items()}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        top = sorted(
            report["categories"].items(),
            key=lambda item: item[1]["device_kernel_sum_ns"],
            reverse=True,
        )[:30]
        print(
            "DEVICE_PREFILL_PROFILE",
            json.dumps(
                {
                    "instrumented_wall_seconds": report["instrumented_wall_seconds"],
                    "logits_exact": report["instrumented_logits_exact"],
                    "event_count": len(report["events"]),
                    "top": top,
                }
            ),
            flush=True,
        )
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
