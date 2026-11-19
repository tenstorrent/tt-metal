# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic precision experiments; never a replacement for the resume gate."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "baseline",
    "configured",
    "fp32_boundary",
    "no_sr",
    "fp32_boundary_no_sr",
    "grouped_singleton",
    "fp32_boundary_grouped_singleton",
    "no_sr_grouped_singleton",
    "fp32_boundary_no_sr_grouped_singleton",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--variants", nargs="+", choices=VARIANTS)
    selection.add_argument("--case-plan", type=Path, help="JSON object mapping variant names to nonempty start lists")
    parser.add_argument("--starts", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--precision-config",
        default="baseline",
        help="Explicit policy for configured diagnostics; original baseline parity remains mandatory",
    )
    args = parser.parse_args(argv)
    if args.case_plan is not None:
        if args.starts is not None:
            parser.error("--starts cannot be combined with --case-plan; put starts in the JSON plan")

        def unique_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate case-plan variant: {key}")
                result[key] = value
            return result

        try:
            plan = json.loads(args.case_plan.read_text(), object_pairs_hook=unique_keys)
        except (OSError, ValueError) as error:
            parser.error(f"Cannot load case plan: {error}")
    else:
        if len(set(args.variants)) != len(args.variants):
            parser.error("--variants must be unique")
        plan = {variant: args.starts if args.starts is not None else [2559] for variant in args.variants}
    if not isinstance(plan, dict) or not plan:
        parser.error("Case plan must be a nonempty JSON object")
    for variant, starts in plan.items():
        if variant not in VARIANTS:
            parser.error(f"Unknown variant in case plan: {variant}")
        if not isinstance(starts, list) or not starts or any(type(start) is not int for start in starts):
            parser.error(f"Starts for {variant} must be a nonempty list of integers")
        if len(set(starts)) != len(starts) or any(not 0 < start < 2560 for start in starts):
            parser.error(f"Starts for {variant} must be unique and strictly between 0 and 2560")
    args.resolved_case_plan = plan
    if args.precision_config != "baseline" and "baseline" in plan:
        parser.error("Use --variants configured for a non-baseline precision policy")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.json"
    assert not result_path.exists(), "Refusing to replace diagnostic evidence"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    import torch
    from readiness_check.run_prefill_resume_check import (
        DEFAULT_THRESHOLDS,
        compare_logits,
        evaluate_case,
        window_positions,
    )

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import build_generator

    torch.set_num_threads(8)
    diagnostic = json.loads((ROOT / "doc/datatype_sweep/baseline_resume_diagnostics/L2560_s2559.json").read_text())
    tokens = diagnostic["tokens"]
    assert len(tokens) == 2560 and diagnostic["gate_modified"] is False
    thresholds = {key: DEFAULT_THRESHOLDS[key] for key in ("pcc_mean_min", "pcc_min_min", "top1_min")}
    report = dict(
        status="running",
        command=sys.argv,
        started=time.time(),
        thresholds=thresholds,
        tokens=tokens,
        mesh=[1, 4],
        context=4096,
        precision_config=args.precision_config,
        precision_config_sha256=(
            hashlib.sha256(Path(args.precision_config).read_bytes()).hexdigest()
            if args.precision_config != "baseline"
            else None
        ),
        cases=[],
        case_plan=args.resolved_case_plan,
        case_plan_source=(
            dict(path=str(args.case_plan), sha256=hashlib.sha256(args.case_plan.read_bytes()).hexdigest())
            if args.case_plan is not None
            else None
        ),
        scope="Diagnostic uniform precision-policy experiments; no negative controls, no gate waiver, no performance claim",
        production_runtime_changed=False,
        grouped_singleton_scope=(
            "Uniformly pad logical1 MoE residual input to32 inside the prefill executor only, "
            "run the original executor MoE including norm/shared/reduction, and retain logical row0. "
            "This changes those shape-dependent paths together; no attention, KV, recurrent-state or decode padding."
        ),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *sorted((ROOT / "tt").glob("*.py"))]
        },
    )

    def save():
        result_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    mesh = gen = None
    save()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
        gen = build_generator(
            ROOT, mesh, max_seq_len=4096, max_batch_size=1, host_sampling=False, precision_config=args.precision_config
        )
        gen.prefill_logits([1] * 33)
        gen.model.host_logits(gen.sampler_logits)
        gen.prepare(prefill_cases=((32, 0),))
        report["runtime_precision_before_diagnostic_overrides"] = gen.model.precision_summary()
        for variant, starts in args.resolved_case_plan.items():
            counters = dict(boundary_overrides=0, grouped_sr_overrides=0, singleton_moe_overrides=0)
            public_calls = []
            original_prefill = gen.prefill_executor.prefill
            original_moe = gen.prefill_executor._moe

            def observe_prefill(token_ids, **kwargs):
                length = int(token_ids.numel()) if isinstance(token_ids, torch.Tensor) else len(token_ids)
                public_calls.append(dict(start=int(kwargs.get("start", 0)), length=length))
                return original_prefill(token_ids, **kwargs)

            with ExitStack() as stack:
                stack.enter_context(patch.object(gen.prefill_executor, "prefill", observe_prefill))
                if "grouped_singleton" in variant:

                    def moe(layer, x):
                        if x.shape[-2] != 1:
                            return original_moe(layer, x)
                        assert tuple(x.shape) == (1, 1, 2048) and x.dtype == ttnn.bfloat16
                        counters["singleton_moe_overrides"] += 1
                        padded = ttnn.pad(x, [(0, 0), (0, 31), (0, 0)], 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                        result = original_moe(layer, padded)
                        return result[:, :1, :]

                    stack.enter_context(patch.object(gen.prefill_executor, "_moe", moe))
                if "fp32_boundary" in variant:

                    def begin(layer, state):
                        if layer.kind == "linear_attention":
                            assert state["recurrent"].dtype == ttnn.float32
                            counters["boundary_overrides"] += 1
                        # Existing FP32 buffer remains unchanged at *every* public prefill boundary.

                    stack.enter_context(patch.object(gen.prefill_executor, "_begin_layer", begin))
                if "no_sr" in variant:
                    for layer in gen.model.layers:
                        original = layer._grouped_moe

                        def grouped(x, ids, scores, *, stochastic_rounding=True, _original=original):
                            counters["grouped_sr_overrides"] += 1
                            return _original(x, ids, scores, stochastic_rounding=False)

                        stack.enter_context(patch.object(layer, "_grouped_moe", grouped))
                for start in starts:
                    counters_before = dict(counters)
                    runtime_before = dict(gen.counters)
                    calls_before = len(public_calls)
                    positions = window_positions(start, len(tokens), 256)
                    print("VARIANT_CASE", variant, start, flush=True)
                    fresh = gen.prefill_logits_continuation(tokens, 0, positions=positions, prefill_prefix=True)
                    resumed = gen.prefill_logits_continuation(tokens, start, positions=positions, prefill_prefix=True)
                    metrics = compare_logits(fresh, resumed)
                    reasons = evaluate_case(metrics, thresholds)
                    tensor_path = args.output_dir / f"{variant}_s{start}.pt"
                    torch.save(dict(fresh_logits=fresh.cpu(), resumed_logits=resumed.cpu()), tensor_path)
                    delta = {key: value - counters_before[key] for key, value in counters.items()}
                    runtime_delta = {key: value - runtime_before[key] for key, value in gen.counters.items()}
                    calls = public_calls[calls_before:]
                    row = dict(
                        variant=variant,
                        start=start,
                        positions=positions,
                        metrics=metrics,
                        reasons=reasons,
                        diagnostic_thresholds_pass=not reasons,
                        counters=dict(counters),
                        case_counter_delta=delta,
                        runtime_counter_delta=runtime_delta,
                        public_prefill_calls=calls,
                        tensors=str(tensor_path),
                    )
                    if variant == "baseline":
                        baseline_path = ROOT / f"doc/datatype_sweep/baseline_resume_diagnostics/L2560_s{start}.json"
                        assert baseline_path.exists(), "Baseline parity requires the saved original diagnostic"
                        baseline_diagnostic = json.loads(baseline_path.read_text())
                        assert baseline_diagnostic["tokens"] == tokens and baseline_diagnostic["positions"] == positions
                        assert baseline_diagnostic["gate_modified"] is False
                        prior = torch.load(baseline_diagnostic["tensor_path"], map_location="cpu", weights_only=True)
                        row["baseline_vs_original"] = dict(
                            diagnostic=str(baseline_path),
                            tensors=baseline_diagnostic["tensor_path"],
                            fresh_bit_equal=torch.equal(fresh.cpu(), prior["fresh_logits"]),
                            resumed_bit_equal=torch.equal(resumed.cpu(), prior["resumed_logits"]),
                        )
                    report["cases"].append(row)
                    save()
                    assert calls == [
                        dict(start=0, length=len(tokens)),
                        dict(start=0, length=start),
                        dict(start=start, length=len(tokens) - start),
                    ], "Expected exactly fresh, prefix, suffix public prefill calls"
                    assert all(
                        runtime_delta[key] == 0 for key in ("model_replays", "sampling_replays", "captures")
                    ), "Diagnostic unexpectedly invoked or captured decode"
                    if "fp32_boundary" in variant:
                        expected = 3 * sum(layer.kind == "linear_attention" for layer in gen.model.layers)
                        assert delta["boundary_overrides"] == expected, "FP32 override missed a public boundary"
                    if "grouped_singleton" in variant:
                        expected = sum(
                            length % layer.policy.moe_chunk == 1
                            for layer in gen.model.layers
                            for length in (len(tokens), start, len(tokens) - start)
                        )
                        assert delta["singleton_moe_overrides"] == expected, "Singleton override missed a prefill chunk"
                    if "no_sr" in variant:
                        expected = sum(
                            length // layer.policy.moe_chunk
                            + int(length % layer.policy.moe_chunk > 1)
                            + int("grouped_singleton" in variant and length % layer.policy.moe_chunk == 1)
                            for layer in gen.model.layers
                            for length in (len(tokens), start, len(tokens) - start)
                        )
                        assert delta["grouped_sr_overrides"] == expected, "SR override missed a grouped prefill chunk"
                    if variant == "baseline":
                        assert (
                            row["baseline_vs_original"]["fresh_bit_equal"]
                            and row["baseline_vs_original"]["resumed_bit_equal"]
                        ), "Baseline diagnostic differs from saved original logits"
                    print("VARIANT_RESULT", json.dumps(row), flush=True)
            if "fp32_boundary" in variant:
                assert counters["boundary_overrides"] > 0
            if "no_sr" in variant:
                assert counters["grouped_sr_overrides"] > 0
        report["status"] = "complete"
    except BaseException:
        report["status"] = "error"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"] = time.time() - report["started"]
        save()
        if gen is not None and not gen.closed:
            gen.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        report["device_closed"] = True
        save()
    print("VARIANTS_COMPLETE", result_path, flush=True)


if __name__ == "__main__":
    main()
