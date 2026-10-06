# SPDX-License-Identifier: Apache-2.0
"""Reproduce Stage 7 performance accounting from retained JSON; no devices."""

import json
import os
from pathlib import Path
from statistics import median


def main():
    root = Path(__file__).resolve().parents[1]
    out = Path(os.environ.get("FULL_ARTIFACT_DIR", root / "doc/optimized_full_model"))

    def read(name):
        return json.loads((out / f"{name}.json").read_text())

    def primary(data):
        rows = data["primary_128_128"]
        result = {}
        for teacher, label in ((False, "greedy_caller_readback"), (True, "constant_token_teacher_input")):
            samples = [r for r in rows if r["teacher_forcing"] == teacher]
            ms = median(r["decode_ms_per_token"] for r in samples)
            result[label] = {
                "ttft_ms": median(r["ttft_ms"] for r in samples),
                "decode_ms_per_token": ms,
                "decode_tokens_per_second_per_user": 1000 / ms,
                "samples": len(samples),
            }
        result["device_replay_ms"] = data["device_traced_ms"]
        return result

    before, after, accounting = read("baseline"), read("final_performance"), read("decode_accounting")
    assert before["passed"] and after["passed"]
    baseline, final = primary(before), primary(after)
    prefill = []
    for length, physical in ((31, 32), (128, 128), (8193, 8192)):
        row = {"logical_prompt_length": length, "prefill_tokens": length - 1, "physical_bucket": physical, "chunks": 1}
        for traced, label in ((False, "eager_ttft_ms"), (True, "traced_ttft_ms")):
            samples = [
                r for r in after["prefill_comparison"] if r["length"] == length and r["traced_prefill"] == traced
            ]
            assert len(samples) == 3
            row[label] = median(r["ttft_ms"] for r in samples)
        prefill.append(row)
    split_ms = final["device_replay_ms"]["split"]
    terminal_ms = max(
        sum(r["components_us"][k] for k in ("entry", "terminal", "sampling")) / 1000
        for r in accounting["reduced_devices"]
    )
    stack_ms = accounting["optimized_layer_stack_ms"]
    summary = {
        "workload": {
            "profile": "single_user_decode",
            "prompt_len": 128,
            "gen_len": 128,
            "batch": 1,
            "capacity": 1048576,
        },
        "ttft_ms": final["greedy_caller_readback"]["ttft_ms"],
        "decode_ms_per_token_e2e": split_ms,
        "decode_ms_per_token_device": None,
        "device_time_scope": "Direct device timing is from the reduced same-capture profile below; no all-layer profile was collected.",
        "full_stack_kernel_ms_estimate": accounting["full_stack_profile_kernel_estimate_ms"],
        "roofline_ms_per_token_estimate": accounting["full_roofline_ms"],
        "token_out_tokens_per_second_per_user": 1000 / split_ms,
        "baseline": baseline,
        "final": final,
        "paired_prefill": prefill,
        "prefill_input": after["prefill_comparison_input"],
        "layer_stack_ms": stack_ms,
        "layer_stack_tokens_per_second_per_user": 1000 / stack_ms,
        "terminal_and_entry_kernel_ms": terminal_ms,
        "stack_plus_terminal_ms": stack_ms + terminal_ms,
        "full_model_overhead_above_stack_ms": split_ms - stack_ms,
        "excess_above_stack_plus_terminal_fraction": split_ms / (stack_ms + terminal_ms) - 1,
        "reduced_same_capture": {
            "layers": [0, 4],
            "host_ms_including_token_readback": accounting["reduced_same_capture_host_ms"],
            "slowest_device_span_ms": max(r["span_us"] for r in accounting["reduced_devices"]) / 1000,
            "slowest_kernel_sum_ms": max(r["kernel_us"] for r in accounting["reduced_devices"]) / 1000,
            "roofline_ms": accounting["reduced_roofline_ms"],
            "host_minus_device_ms": accounting["reduced_host_minus_slowest_device_ms"],
        },
        "device_loop_counters": after["device_traced_counters"],
        "timing_boundary": "128 nonblocking public decode_forward(read_from_device=False) calls, persistent device feedback; one completion fence after loop; no per-token host refresh, sync or readback.",
        "named_limitations": [
            "The layer-stack budget uses Stage5 capacity1024 timings; full-model capacity is1048576. It is an estimate, not an exact subtraction.",
            "The all-layer device-kernel figure is a40/10 extrapolation of two real layer kinds, not a measured all-layer device duration.",
            "Many small decoder kernels and collectives remain above the weight/KV byte roofline; inherited precision-locked topology/geometry rejections apply.",
            "Public generate() includes per-token readbacks; only the explicit low-level token-out loop has no per-token host boundary.",
            "Prefill tracing applies to256-aligned physical chunks ending at or below65536; other valid chunks retain their prepared eager precision variant.",
        ],
        "artifacts": [
            "baseline.json",
            "final_performance.json",
            "decode_accounting.json",
            "profile_host_timing.json",
            "profile/provenance.json",
        ],
    }
    (out / "perf_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
