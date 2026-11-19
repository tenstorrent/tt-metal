# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced wrapper reproducer for the installed resumed-prefill readiness gate.

Uses the installed runner unchanged: natural-book L257, block32, window128,
resume starts32/68/256, default thresholds, and both negative controls. A
transparent observer records position-level diagnostics from the runner's
existing start68 calls, without any additional model forward passes.

Run serially after lane-env and restoring the accepted mesh descriptor::

    timeout 1800 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_prefill_resume \
      --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/reduced_resume_0_1_3.json

For one-kind isolation use --layer-indices 0, --layer-indices 1, or
--layer-indices 3 with a distinct --output. These are diagnostic reduced-stack
results, never replacements for readiness/prefill_resume_check.json.
"""

import argparse
import importlib
import json
import os
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator


def layer_indices(value):
    result = tuple(int(part) for part in value.split(",") if part.strip())
    if not result or len(result) != len(set(result)) or any(index not in (0, 1, 3) for index in result):
        raise argparse.ArgumentTypeError("Choose a nonempty distinct subset of 0,1,3")
    return result


class ObserveResume:
    """Forward every generator operation unchanged; inspect returned host logits."""

    def __init__(self, generator, runner, record):
        self.generator = generator
        self.runner = runner
        self.record = record
        self.positions = runner.window_positions(68, 257, 128)
        self.reference = None

    def __getattr__(self, name):
        return getattr(self.generator, name)

    def prefill_logits_continuation(self, tokens, start, *, positions=None, prefill_prefix=True):
        output = self.generator.prefill_logits_continuation(
            tokens, start, positions=positions, prefill_prefix=prefill_prefix
        )
        requested = list(positions) if positions is not None else None
        if len(tokens) == 257 and requested == self.positions:
            selected = self.runner._select_positions(output, start, self.positions, 257)
            if start == 0 and prefill_prefix:
                # Host outputs are immutable independent of subsequent device
                # work. Retain only this one gate window until its comparisons.
                self.reference = selected.detach().clone()
                self.record["prompt_token_ids"] = list(tokens)
            elif start == 68:
                self.compare(selected, "continuation" if prefill_prefix else "no_history")
        elif len(tokens) == 257 - 68 and start == 0 and requested == [p - 68 for p in self.positions]:
            selected = self.runner._select_positions(output, 0, requested, len(tokens))
            self.compare(selected, "fresh_suffix")
        return output

    def compare(self, candidate, label):
        assert self.reference is not None, "Start68 diagnostic candidate preceded its full-prefill reference"
        rows = []
        for offset, absolute in enumerate(self.positions):
            reference = self.reference[0, offset].float()
            value = candidate[0, offset].float()
            top = torch.topk(reference, 2)
            candidate_top1 = int(value.argmax())
            delta = value - reference
            rows.append(
                {
                    "position": absolute,
                    "resume_offset": absolute - 68,
                    "full_chunk_start": absolute // 32 * 32,
                    "continuation_chunk_start": next(
                        begin for begin, end in self.generator.prefill_spans(68, 257) if begin <= absolute < end
                    ),
                    "pcc": self.runner._pcc(reference, value),
                    "reference_top1": int(top.indices[0]),
                    "candidate_top1": candidate_top1,
                    "top1_equal": int(top.indices[0]) == candidate_top1,
                    "reference_top1_margin": float(top.values[0] - top.values[1]),
                    "max_absolute_logit_difference": float(delta.abs().max()),
                    "rms_logit_difference": float(delta.square().mean().sqrt()),
                    "finite": bool(torch.isfinite(reference).all() and torch.isfinite(value).all()),
                }
            )
        self.record.setdefault("start68_position_diagnostics", {})[label] = {
            "comparison": "full-prefill window versus " + label,
            "stats": self.runner.compare_logits(self.reference, candidate),
            "positions": rows,
            "worst_pcc_position": min(rows, key=lambda row: row["pcc"])["position"],
            "positions_below_default_pcc_floor": [
                row["position"] for row in rows if row["pcc"] < self.runner.DEFAULT_THRESHOLDS["pcc_min_min"]
            ],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer-indices", type=layer_indices, default=(0, 1, 3))
    parser.add_argument("--capacity", type=int, default=262144)
    parser.add_argument("--checkpoint", default=os.environ.get("QWEN_CHECKPOINT"))
    parser.add_argument("--packed-cache", default=os.environ.get("QWEN_PLE_CACHE"))
    parser.add_argument("--trace-region-size", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--hf-smoke", action="store_true", help="Also check the exact reduced HF fixture before resume")
    args = parser.parse_args()
    model_dir = Path(__file__).resolve().parents[1]
    destination = args.output.resolve()
    if not destination.is_relative_to(model_dir / "doc" / "full_model"):
        parser.error("Diagnostic output must be under doc/full_model; acceptance readiness artifacts are protected")
    runner = importlib.import_module("readiness_check.run_prefill_resume_check")
    report = {
        "passed": False,
        "scope": "reduced_resume_diagnostic",
        "layer_indices": list(args.layer_indices),
        "capacity": args.capacity,
        "regime": "serial",
        "runner_source": str(Path(runner.__file__).resolve()),
        "observer_additional_model_forwards": 0,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "TT_MESH_GRAPH_DESC_PATH",
                "TT_VISIBLE_DEVICES",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_WATCHER",
                "TT_METAL_DEVICE_PROFILER",
            )
        },
    }
    mesh = generator = None
    started = time.monotonic()
    try:
        assert report["environment"][
            "TT_MESH_GRAPH_DESC_PATH"
        ], "Restore accepted mesh descriptor after sourcing lane-env"
        assert [p["start"] for p in runner.resume_points(257, block_sizes=[32], sliding_window=None)] == [32, 68, 256]
        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size, l1_small_size=32768
        )
        options = {"layer_indices": args.layer_indices, "max_seq_len": args.capacity, "max_batch_size": 1}
        if args.checkpoint:
            options["checkpoint"] = args.checkpoint
        if args.packed_cache:
            options["packed_cache"] = args.packed_cache
        generator = build_generator(model_dir, mesh, **options)
        generator.prepare()
        report["setup_seconds"] = time.monotonic() - started
        if args.hf_smoke:
            assert args.layer_indices == (0, 1, 3), "HF fixture is for the exact reduced0/1/3 stack"
            fixture = Path("/home/ttuser/qwen38-flash/reference/full_model_reduced.pt")
            reference = torch.load(fixture, weights_only=False)["steps"]
            forced = generator.generate(
                [785, 13458, 13], 4, next_input=lambda step, prediction: [785, 13458, 13, 0][step]
            )
            ranks = torch.cat(
                [reference[0]["top100_ids"].reshape(-1, 100)[-1:]]
                + [step["top100_ids"].reshape(-1, 100) for step in reference[1:]]
            )
            topk = {
                str(k): float((torch.tensor(forced)[:, None] == ranks[:, :k]).any(-1).float().mean())
                for k in (1, 5, 100)
            }
            report["hf_smoke"] = {"fixture": str(fixture), "teacher_predictions": forced, "topk": topk}
            assert topk["5"] >= 0.98 and topk["100"] == 1, report["hf_smoke"]
        observer = ObserveResume(generator, runner, report)
        # Only dependency injection changes; token source, cases, thresholds,
        # scoring, negative controls and summary remain the installed code.
        with patch.object(runner, "_import_build_generator", return_value=lambda **kwargs: observer):
            result = runner.run_prefill_resume_check(
                model_dir=model_dir,
                mesh_device=mesh,
                lengths=[257],
                block_sizes=[32],
                sliding_window=None,
                tile=32,
                window=128,
                controls=runner.CONTROL_NAMES,
                thresholds=dict(runner.DEFAULT_THRESHOLDS),
                vocab_size=generator.model.config.vocab_size,
                seed=0,
            )
        report["readiness_result"] = result
        report["counters"] = dict(generator.counters)
        assert set(report.get("start68_position_diagnostics", {})) == {
            "continuation",
            "no_history",
            "fresh_suffix",
        }, "Missing observed start68 candidate/control diagnostics"
        report["passed"] = result["status"] == "pass"
        assert report["passed"], result["reasons"]
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if generator is not None:
            generator.close()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        print(
            json.dumps({"passed": report["passed"], "output": str(args.output), "layers": list(args.layer_indices)}),
            flush=True,
        )


if __name__ == "__main__":
    main()
