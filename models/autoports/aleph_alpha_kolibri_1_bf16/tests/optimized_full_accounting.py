# SPDX-License-Identifier: Apache-2.0
"""Reconcile the reduced same-capture profile and the full-stack estimate."""

import csv
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "doc/optimized_full_model"
    old = json.loads((root / "doc/optimized_multichip_decoder/decode_accounting.json").read_text())
    weight_bytes = sum(old["rows"][0]["weight_bytes_per_device"].values())
    # Same default full-capacity SDPA configuration as the measured wrapper:
    # sliding K256 and full K512 at absolute position129; local head BFP8 KV.
    kv_bytes = {"sliding": 2 * 8 * 4 * 1088, "full": 2 * 16 * 4 * 1088}
    terminal_bytes = 2 * 2560 * 16128 * 4 + 2560 * 2 + 80 * 2048 + 2 * 128 * 2
    per_rank = dict(
        sliding_layer=weight_bytes + kv_bytes["sliding"],
        full_layer=weight_bytes + kv_bytes["full"],
        terminal=terminal_bytes,
    )
    reduced_bytes = sum(per_rank.values())
    full_bytes = 40 * per_rank["sliding_layer"] + 10 * per_rank["full_layer"] + terminal_bytes
    host = json.loads((out / "profile_host_timing.json").read_text())
    device = []
    for rank in range(4):
        rows = list(csv.DictReader((out / f"profile/decode_device{rank}_report.csv").open()))
        raw = list(csv.DictReader((out / f"profile/decode_device{rank}_ops.csv").open()))
        assert len(raw) == 158 and len(rows) == 158
        assert raw[133]["OP CODE"] == "LayerNormDeviceOperation"
        assert raw[149]["OP CODE"] == "TopKDeviceOperation"
        spans = {}
        for name, start, end in [
            ("entry", 0, 11),
            ("sliding", 11, 76),
            ("full", 76, 133),
            ("terminal", 133, 145),
            ("sampling", 145, 158),
        ]:
            spans[name] = sum(float(r["DEVICE KERNEL DURATION [ns]"] or 0) for r in raw[start:end]) / 1000
        kernel = sum(float(r["Device Time"] or 0) for r in rows)
        gaps = sum(float(r["Op-to-Op Gap"] or 0) for r in rows)
        device.append(
            dict(
                rank=rank,
                kernel_us=kernel,
                gap_us=gaps,
                span_us=kernel + gaps,
                components_us=spans,
                full_stack_kernel_estimate_us=40 * spans["sliding"]
                + 10 * spans["full"]
                + spans["entry"]
                + spans["terminal"]
                + spans["sampling"],
            )
        )
    result = dict(
        method="Reduced profile: one real layer of each kind, full-capacity cache, complete surrounding token-out path. Device kernel sums and gaps and host timing are from the same profiled replay. Full-stack device time is a 40/10 extrapolation, not a prohibited all-layer profile. Uninstrumented full-model timing is reported separately.",
        theoretical_bytes="One read of active stored weights plus chunk-rounded local BFP8 KV; excludes repeat reads and intermediate traffic.512GB/s per chip is the tt-perf-report ceiling.",
        bytes_per_rank=per_rank,
        reduced_roofline_ms=reduced_bytes / 512e9 * 1000,
        full_roofline_ms=full_bytes / 512e9 * 1000,
        aggregate_full_bytes=full_bytes * 4,
        aggregate_dram_bytes_per_second=512e9 * 4,
        reduced_same_capture_host_ms=host["decode_host_ms"],
        reduced_devices=device,
        reduced_host_minus_slowest_device_ms=host["decode_host_ms"] - max(v["span_us"] for v in device) / 1000,
        optimized_layer_stack_ms=40 * 0.309710 + 10 * 0.296745,
        layer_stack_caveat="Stage5 short-prompt layer timings used1024 cache tokens; this wrapper retains1M. The lower bound is an estimate, not an exact subtraction.",
        full_stack_profile_kernel_estimate_ms=max(v["full_stack_kernel_estimate_us"] for v in device) / 1000,
    )
    (out / "decode_accounting.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, (list, dict))}, indent=2))


if __name__ == "__main__":
    main()
