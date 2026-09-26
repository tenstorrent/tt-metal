# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU cache/RoPE controls against saved fused and optimized stress outputs."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

from models.autoports.google_gemma_4_26b_a4b_it.tests.hf_long_decode_precision_controls import (
    compare,
    run,
    tensor_digest,
)
from models.autoports.google_gemma_4_26b_a4b_it.tests.hf_precision_controls import (
    MODEL,
    REVISION,
    ROOT,
    THRESHOLD,
    BF16RoundedCache,
    digest,
    load_real_layer,
    pcc,
)


def summarize(values, length):
    return {
        "minimum_decode_pcc": min(values),
        "failed_positions": [length + i for i, value in enumerate(values) if value < THRESHOLD],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, choices=(0, 5), required=True)
    parser.add_argument("--length", type=int, default=1025)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "doc/optimized_decoder")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert "ttnn" not in sys.modules
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(Path(__file__).parent, local_files_only=True).text_config
    config._attn_implementation = "eager"
    layer = load_real_layer(config, args.layer)
    x = torch.randn(1, args.length, config.hidden_size).bfloat16().float()
    first = torch.randn(1, 1, config.hidden_size).bfloat16().float()
    inputs = torch.cat([first] + [torch.randn_like(first).bfloat16().float() for _ in range(1, args.steps)], dim=1)
    layer_type = config.layer_types[args.layer]
    extent = (args.length + max(128, args.steps) + 1023) // 1024 * 1024
    cos, sin = Gemma4TextRotaryEmbedding(config)(x, torch.arange(extent)[None], layer_type=layer_type)
    fixtures, metadata = {}, {}
    for name in ("fused", "optimized"):
        stem = args.artifact_dir / f"stress_pair_{name}_layer{args.layer}"
        fixture = torch.load(stem.with_suffix(".pt"), map_location="cpu", weights_only=True)
        recorded = json.loads(stem.with_suffix(".json").read_text())
        assert recorded["length"] == args.length and recorded["decode"]["steps"] == args.steps
        assert recorded["real_weights"] and recorded["prefix_length"] == 0
        assert fixture["prefill"].shape == x.shape and len(fixture["decode"]) == args.steps
        has_inputs = "x" in fixture and "decode_inputs" in fixture
        if has_inputs:
            decode_inputs = fixture["decode_inputs"]
            if isinstance(decode_inputs, list):
                decode_inputs = torch.cat(decode_inputs, dim=1)
            assert torch.equal(fixture["x"].float(), x), "Saved prefill input differs from seeded reconstruction"
            assert torch.equal(decode_inputs.float(), inputs), "Saved decode inputs differ from seeded reconstruction"
        fixtures[name] = fixture
        metadata[name] = {
            "output_fixture": str(stem.with_suffix(".pt")),
            "output_fixture_sha256": digest(stem.with_suffix(".pt")),
            "recorded_report": str(stem.with_suffix(".json")),
            "recorded_report_sha256": digest(stem.with_suffix(".json")),
            "saved_input_tensors_present": has_inputs,
            "saved_inputs_match": True if has_inputs else None,
            "recorded": recorded,
        }
    output = args.output or args.artifact_dir / f"stress_cpu_precision_layer{args.layer}.json"
    report = {
        "model": MODEL,
        "revision": REVISION,
        "layer": args.layer,
        "layer_type": layer_type,
        "seed": 42,
        "length": args.length,
        "steps": args.steps,
        "extent": extent,
        "sliding_window": config.sliding_window,
        "cpu_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "pcc_threshold": THRESHOLD,
        "x_sha256": tensor_digest(x),
        "decode_stream_sha256": tensor_digest(inputs),
        "config_sha256": digest(Path(__file__).parent / "config.json"),
        "source_sha256": digest(Path(__file__)),
        "cpu_run_helper_sha256": digest(Path(__file__).parent / "hf_long_decode_precision_controls.py"),
        "weight_helper_sha256": digest(Path(__file__).parent / "hf_precision_controls.py"),
        "device_harness_sha256": digest(Path(__file__).parent / "run_decoder.py"),
        "command": " ".join(sys.argv),
        "reference": "Real checkpoint, eager HF FP32 computation, seeded BF16-roundtripped input stream",
        "limitations": (
            "These fixtures contain TT outputs; absent input tensors are reconstructed from the device harness RNG "
            "order. Agreement with recorded FP32-vs-TT PCCs validates reconstruction indirectly, not bitwise input "
            "identity. CPU controls isolate cache/table rounding and do not emulate TT arithmetic, kernel "
            "reduction order, or the common-normalized router. No acceptance threshold is changed."
        ),
        "fixtures": {},
        "controls": {},
    }
    started = time.monotonic()

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        report["ttnn_imported"] = "ttnn" in sys.modules
        assert not report["ttnn_imported"]
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    ref_pre, reference = run(layer, x, inputs, cos, sin, DynamicCache(), "FP32_ORACLE")
    reference_pccs = {}
    for name, fixture in fixtures.items():
        values = [pcc(row["output"], actual) for row, actual in zip(reference, fixture["decode"])]
        reference_pccs[name] = values
        recorded = metadata[name].pop("recorded")
        checks = {row["position"]: row["pcc"] for row in recorded["decode"]["checks"]}
        prefill_pcc = pcc(ref_pre, fixture["prefill"])
        maximum_delta = max(abs(value - checks[args.length + i]) for i, value in enumerate(values))
        report["fixtures"][name] = {
            **metadata[name],
            "fp32_prefill_pcc": prefill_pcc,
            "prefill_pcc_delta_from_recorded": abs(prefill_pcc - recorded["pcc"]),
            "maximum_decode_pcc_delta_from_recorded": maximum_delta,
            "recorded_pccs_reproduced_within_1e_5": maximum_delta < 1e-5 and abs(prefill_pcc - recorded["pcc"]) < 1e-5,
            **summarize(values, args.length),
        }
        print("FIXTURE", name, json.dumps(report["fixtures"][name]), flush=True)
    save()
    assert all(
        row["recorded_pccs_reproduced_within_1e_5"] for row in report["fixtures"].values()
    ), "Reconstructed CPU oracle does not reproduce recorded harness comparisons"
    for control in ("cache", "rope-cache"):
        control_cos, control_sin = (
            (cos, sin) if control == "cache" else (cos.bfloat16().float(), sin.bfloat16().float())
        )
        prefill, outputs = run(layer, x, inputs, control_cos, control_sin, BF16RoundedCache(), control)
        rows = [compare(ref, actual, args.length + i) for i, (ref, actual) in enumerate(zip(reference, outputs))]
        device = {}
        for name, fixture in fixtures.items():
            values = [pcc(row["output"], actual) for row, actual in zip(outputs, fixture["decode"])]
            device[name] = {"prefill_pcc": pcc(prefill, fixture["prefill"]), **summarize(values, args.length)}
            for i, (row, value) in enumerate(zip(rows, values)):
                row[f"tt_{name}_vs_control_pcc"] = value
                row[f"tt_{name}_vs_fp32_pcc"] = reference_pccs[name][i]
            fp32_failures = set(report["fixtures"][name]["failed_positions"])
            cpu_failures = {row["position"] for row in rows if not row["passed"]}
            device[name]["hf_failure_positions_also_failing_cpu_control"] = sorted(fp32_failures & cpu_failures)
            device[name]["hf_failure_positions_passing_cpu_control"] = sorted(fp32_failures - cpu_failures)
        details = [
            row
            for row in rows
            if not row["passed"]
            or row["route_intersection"] != config.top_k_experts
            or any(
                row[f"tt_{name}_vs_fp32_pcc"] < THRESHOLD or row[f"tt_{name}_vs_control_pcc"] < THRESHOLD
                for name in fixtures
            )
        ]
        result = {
            "description": "FP32 HF with BF16-roundtripped K/V cache"
            + (" and BF16-roundtripped RoPE cos/sin tables" if control == "rope-cache" else ""),
            "prefill_pcc_vs_fp32": pcc(ref_pre, prefill),
            **summarize([row["pcc"] for row in rows], args.length),
            "changed_route_positions": [row["position"] for row in rows if row["route_intersection"] != 8],
            "tt_vs_control": device,
            "decode_details": details,
        }
        report["controls"][control] = result
        save()
        print("CONTROL", control, json.dumps({k: v for k, v in result.items() if k != "decode_details"}), flush=True)


if __name__ == "__main__":
    main()
