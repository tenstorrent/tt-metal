# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serial full-model precision evaluation using the installed readiness scorers."""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-kwargs-file", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--trace-region-size", type=int, default=0)
    parser.add_argument("--trace-diagnostics", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    kwargs = json.loads(args.build_kwargs_file.read_text())
    root = Path(__file__).resolve().parents[1]
    report = dict(
        completed=False,
        passed=False,
        command=[sys.executable, *sys.argv],
        build_kwargs=kwargs,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        hardware="4 Blackhole p300c devices",
        mesh=[1, 4],
        measurement_regime="serial; traced teacher forcing; caller sampling/readback; tracking enabled",
        environment={k: v for k, v in os.environ.items() if k.startswith(("TT_", "READINESS_"))},
        thresholds=dict(top1=0.90, top5=0.98, top100=1.0),
        trace_region_size=args.trace_region_size,
        runs=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    mesh = generator = None
    start = time.monotonic()
    try:
        assert os.environ.get("TT_VISIBLE_DEVICES") == "0,1,2,3"
        assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
        assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
        import torch

        import ttnn
        from models.autoports.qwen_qwen3_8_flash_next.tests.run_full_model_accuracy import memory
        from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator

        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size, l1_small_size=32768
        )
        generator = build_generator(root, mesh, **kwargs)
        assert generator.model.layer_indices == tuple(range(48))
        report["capacity"] = generator.capacity
        report["memory_loaded"] = memory(mesh)
        if hasattr(generator.model, "precision_summary"):
            report["runtime_precision"] = generator.model.precision_summary()
        generator.prepare()
        if args.trace_diagnostics:
            import inspect

            original_record = generator._record_trace
            report["trace_capture_ledger"] = []

            def trace_record(body):
                signature = inspect.getclosurevars(body).nonlocals
                entry = {
                    "body": body.__qualname__,
                    "signature": {k: v for k, v in signature.items() if k in ("count", "slots")},
                    "memory_before": memory(mesh),
                }
                report["trace_capture_ledger"].append(entry)
                print(json.dumps({"trace_capture_start": entry}), flush=True)
                try:
                    trace = original_record(body)
                    entry["trace_id"] = str(trace)
                    return trace
                finally:
                    entry["memory_after"] = memory(mesh)
                    if args.trace_region_size:
                        view = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
                        entry["reserved_trace_allocated_bytes_per_rank"] = (
                            view.total_bytes_allocated_per_bank * view.num_banks
                        )
                    save()

            generator._record_trace = trace_record
        report["memory_prepared"] = memory(mesh)
        report["setup_seconds"] = time.monotonic() - start
        save()
        for repeat in range(args.repeats):
            for reference in args.reference:
                for name in ("run_prefill_check", "run_teacher_forcing"):
                    module = importlib.import_module("readiness_check." + name)
                    warmup = None
                    if name == "run_teacher_forcing":
                        from readiness_check.teacher_forcing import TokenAccuracy

                        acc = TokenAccuracy(reference)
                        warm_before = dict(generator.counters)
                        for entry in range(acc.num_entries):
                            generator.generate(
                                acc.get_prompt_token_ids(entry),
                                2,
                                enable_trace=True,
                                next_input=lambda step, predicted, entry=entry: int(acc.gt_gen_1d[entry][step]),
                            )
                        warmup = {k: v - warm_before[k] for k, v in generator.counters.items()}
                    before = dict(generator.counters)

                    def resident_builder(*, model_dir, mesh_device, **forwarded):
                        assert Path(model_dir).resolve() == root
                        assert mesh_device is mesh and forwarded == kwargs
                        return generator

                    # The installed runners own a temporary generator normally.
                    # This harness owns one resident generator across references;
                    # defer teardown until the outer finally, outside all rows.
                    with (
                        patch.object(module, "_import_build_generator", return_value=resident_builder),
                        patch.object(generator, "teardown", return_value=None),
                    ):
                        rows = getattr(module, name)(
                            model_dir=root, reference_path=reference, mesh_device=mesh, build_kwargs=kwargs
                        )
                    delta = {k: v - before[k] for k, v in generator.counters.items()}
                    traced = name != "run_teacher_forcing" or (
                        delta["model_replays"] >= sum(r["total"] - 1 for r in rows)
                        and delta["captures"] == delta["releases"] == 0
                    )
                    record = dict(
                        repeat=repeat,
                        reference=str(reference.resolve()),
                        runner=name,
                        rows=rows,
                        counter_delta=delta,
                        excluded_trace_warmup=warmup,
                        trace_verified=traced,
                        passed=bool(rows)
                        and traced
                        and all(r["top1"] >= 0.90 and r["top5"] >= 0.98 and r["top100"] == 1 for r in rows),
                    )
                    report["runs"].append(record)
                    save()
                    print(json.dumps(record), flush=True)
        if args.trace_diagnostics:
            from readiness_check.teacher_forcing import TokenAccuracy

            from models.common.sampling.generator import SamplingParams

            prompt = TokenAccuracy(args.reference[-1]).get_prompt_token_ids(0)
            report["trace_inventory_smoke"] = []
            for length in (31, 32, 33):
                before = dict(generator.counters)
                tokens = generator.generate(prompt[:length], 2, enable_trace=True)
                assert len(tokens) == 2 and all(0 <= t < generator.model.config.vocab_size for t in tokens)
                report["trace_inventory_smoke"].append(
                    {
                        "logical_prompt_length": length,
                        "tokens": tokens,
                        "counter_delta": {k: v - before[k] for k, v in generator.counters.items()},
                    }
                )
            generator.generate(
                prompt[:33],
                2,
                enable_trace=True,
                sampling_params=SamplingParams(temperature=1.0, top_k=20, top_p=0.95, seed=1000),
            )
            generator.set_sampling_params()
            generator.generate(prompt[:31], 2, enable_trace=True)
            assert set(generator.prefill_traces) == {1, 2, 4, 8, 16, 32}
            report["retained_prefill_buckets"] = sorted(generator.prefill_traces)
        report["memory_final"] = memory(mesh)
        report["completed"] = True
        report["passed"] = all(r["passed"] for r in report["runs"])
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - start
        save()
        if generator is not None:
            generator.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
