"""Summarize complete numerical receipts without inventing quality gates."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def aggregate(rows, name):
    ms = [v[name] for v in rows]
    finite = all(m["finite"] for m in ms)
    if not finite:
        return dict(finite=False)
    rms = [m["relative_rms"] for m in ms]
    return dict(
        finite=True,
        relative_rms_min=min(rms),
        relative_rms_median=statistics.median(rms),
        relative_rms_max=max(rms),
        pcc_min=min(m["pcc"] for m in ms),
        max_abs=max(m["max_abs"] for m in ms),
        worst_query_head_relative_rms=max(m["worst_query_head_relative_rms"]["relative_rms"] for m in ms),
        worst_query_head_max_abs=max(m["worst_query_head_max_abs"]["max_abs"] for m in ms),
    )


def summarize(directory):
    report = json.loads((directory / "long-v2.json").read_text())
    groups = defaultdict(list)
    compact = []
    for case in report["cases"]:
        for query in case["causal_queries"]:
            for variant in query["variants"]:
                if variant.get("state") != "completed":
                    continue
                if case["context"] != 1024:
                    groups[variant["name"]].append(variant)
                row = dict(
                    context=case["context"],
                    active_tokens=query["active_tokens"],
                    variant=variant["name"],
                    kernel_gate_passed=variant["kernel_gate_passed"],
                    total_rms_ratio_to_bfp8=variant["total_rms_ratio_to_bfp8"],
                )
                for kind in [
                    "quantization_only",
                    "execution_on_quantized",
                    "total",
                    "actual_vs_bfp8_baseline",
                    "quantized_reference_vs_bfp8_baseline",
                ]:
                    row[kind] = {k: v for k, v in variant[kind].items() if k not in ("per_user", "per_query_head")}
                compact.append(row)
    summary = dict(
        state=report["state"],
        completed_variants=len(compact),
        cleanup_completed=report["cleanup_completed"],
        simulator_sha256=report["simulator_sha256"],
        native_revision=report["native_revision"],
        probe_sha256=report["probe_sha256"],
        model_accuracy_qualified=False,
        physical_devices_accessed=False,
        compiler_arithmetic_fallback=report["compiler_arithmetic_fallback"],
        kernel_gate=report["kernel_gate"],
        quantization_gate=None,
        group_scope="Requested 32768, 131072, and 262016 contexts; two causal bounds each; smoke excluded",
        groups={},
        rows=compact,
    )
    for name, variants in groups.items():
        summary["groups"][name] = dict(
            cases=len(variants),
            kernel_gate_passed=all(v["kernel_gate_passed"] for v in variants),
            metrics={
                kind: aggregate(variants, kind)
                for kind in ["quantization_only", "execution_on_quantized", "total", "actual_vs_bfp8_baseline"]
            },
        )
    smoke = json.loads((directory / "smoke-v1.json").read_text())
    original = [v for c in smoke["cases"] for q in c["causal_queries"] for v in q["variants"]]
    fallback = [v for c in report["cases"] if c["context"] == 1024 for q in c["causal_queries"] for v in q["variants"]]
    summary["smoke_output_bit_identical_with_fallback"] = len(original) == len(fallback) == 6 and all(
        a["output_sha256"] == b["output_sha256"] for a, b in zip(original, fallback)
    )
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in compact:
        print(
            f"{row['context']:6} {row['active_tokens']:6} {row['variant']} "
            f"quant={100*row['quantization_only']['relative_rms']:.4f}% "
            f"exec={100*row['execution_on_quantized']['relative_rms']:.4f}% "
            f"total={100*row['total']['relative_rms']:.4f}% "
            f"exec_pcc={row['execution_on_quantized']['pcc']:.8f} "
            f"gate={row['kernel_gate_passed']}"
        )
    print(json.dumps({k: v for k, v in summary.items() if k not in ("groups", "rows")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    summarize(parser.parse_args().directory)
