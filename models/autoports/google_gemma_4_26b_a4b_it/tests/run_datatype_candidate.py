# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate one explicit precision policy with all-layer traced readiness."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import build_generator
from models.common.readiness_check.run_prefill_check import _run_one_entry_prefill
from models.common.readiness_check.run_teacher_forcing import _run_one_entry
from models.common.readiness_check.schema import load_reference
from models.common.readiness_check.teacher_forcing import TokenAccuracy

ROOT = Path("models/autoports/google_gemma_4_26b_a4b_it")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    policy = json.loads(args.config.read_text())
    report = {
        "config_id": policy["config_id"],
        "precision_config_path": str(args.config),
        "dtype_policy": policy,
        "compute_fidelity_policy": {
            kind: {k: v for k, v in fields.items() if "fidelity" in k}
            for kind, fields in policy.get("layer_types", {}).items()
        },
        "environment": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_WATCHER",
                "TT_METAL_DEVICE_PROFILER",
            )
        },
        "performance_eligible": not any(
            os.environ.get(k, "0") != "0"
            for k in ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER")
        ),
        "hardware": "4 Blackhole P300c ASICs",
        "mesh": [1, 4],
        "measurement_regime": "full-model traced teacher-forcing; warmed repeated requests",
        "command": "OMP_NUM_THREADS=8 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_datatype_candidate "
        f"--config {args.config} --output {args.output} --repeats {args.repeats}",
        "runtime_source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "tt").glob("*.py"))
        },
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "thresholds": {"top1": 0.90, "top5": 0.98, "top100": 1.0},
        "status": "started",
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    try:
        gen = build_generator(None, mesh, max_seq_len=8192, precision_config=args.config)
        report["runtime_policy"] = gen.model.precision_summary()
        reference_path = ROOT / "readiness_aime24_chat.refpt"
        report["reference"] = str(reference_path)
        report["reference_sha256"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        reference = load_reference(reference_path)
        report["prefill"] = [
            _run_one_entry_prefill(generator=gen, entry=e, reference=reference) for e in reference.entries
        ]
        save()
        acc = TokenAccuracy(reference_path)
        report["decode_runs"] = []
        for repeat in range(args.repeats + 1):
            acc = TokenAccuracy(reference_path)
            for i in range(acc.num_entries):
                row = _run_one_entry(generator=gen, acc=acc, entry_idx=i)
                metrics = dict(gen.metrics)
                assert metrics["counters"]["model_replays"] == 99
                assert metrics["counters"]["sampling_replays"] == 99
                assert not metrics["reduced_probe"]
                assert metrics["source"] == "teacher_forcing"
                assert row["total"] == 100
                report["decode_runs"].append({"warmup": repeat == 0, "entry": i, **row, "metrics": metrics})
                save()
        import statistics

        measured = [r for r in report["decode_runs"] if not r["warmup"]]
        report["top1"] = min(r["top1"] for r in measured)
        report["top5"] = min(r["top5"] for r in measured)
        report["top100"] = min(r["top100"] for r in measured)
        report["ttft_ms"] = statistics.median(r["ttft_ms"] for r in measured)
        report["decode_tps"] = statistics.median(r["decode_t/s/u"] for r in measured)
        report["trace_verified"] = True
        report["token_count"] = 100
        report["input_tokens"] = measured[0]["metrics"]["input_tokens"]
        report["runtime_policy"] = gen.model.precision_summary()
        report["active_experts_per_token"] = gen.model.config.top_k_experts
        phases = report["prefill"] + measured
        report["status"] = (
            "pass" if all(all(r[k] >= v for k, v in report["thresholds"].items()) for r in phases) else "accuracy_fail"
        )
        save()
    except Exception as error:
        report["status"] = "runtime_error"
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
