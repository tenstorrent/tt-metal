# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact real-GR row-split control; no full decoder/capacity acceptance.

All four actual layer1 down/up projections run ordinary versus split A/B/A at
Q1..32. Changed-input traces interleave warmed independent eager calls. Only
the two down projections are selected by the separate capacity experiment.
"""

import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--trace-counts", default="17,27,29,30,32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace_counts = [int(value) for value in args.trace_counts.split(",")]
    if not trace_counts or any(value not in range(17, 33) for value in trace_counts):
        parser.error("Trace counts must exercise the split path17..32")
    import torch
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.capacity_gr_rows import SplitGRQueryRows
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
    from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader
    from models.autoports.qwen_qwen3_8_flash_next.tt.precise_residual_projection import PreciseResidualProjection

    if not trace_allocation_tracker.TRACE_ALLOC_TRACKING:
        raise ValueError("TT_METAL_TRACE_ALLOC_TRACKING=1 required")
    torch.set_num_threads(8)
    torch.manual_seed(1911)
    names = [
        f"{prefix}.input_mix_weight_{direction}.weight"
        for prefix in ("attn_hyper_connection", "mlp_hyper_connection")
        for direction in ("down", "up")
    ]
    weights = {}
    for name in names:
        weights.update(load_weights(args.snapshot, 1, name))
    report = {
        "scope": __doc__,
        "snapshot": str(args.snapshot),
        "eager": [],
        "traces": [],
        "product_geometry": {},
        "passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        loader = WeightLoader(weights, mesh)
        original = {name: PreciseResidualProjection(loader, name) for name in names}
        split = {name: SplitGRQueryRows(module, name) for name, module in original.items()}
        report["wrappers"] = {name: module.report for name, module in split.items()}
        addresses = {weight.buffer_address(): name for name, module in original.items() for weight in module.weights}
        native_multiply = ttnn.multiply
        phase = "setup"

        def observe_product(left, right, *positional, **keywords):
            result = native_multiply(left, right, *positional, **keywords)
            address = right.buffer_address() if hasattr(right, "buffer_address") else None
            if address in addresses:
                name = addresses[address]
                shape = tuple(result.padded_shape)
                key = f"{phase}:{name}:{shape[1]}"
                half_shape = (*shape[:-1], ((original[name].input_width // 2 + 31) // 32) * 32)
                report["product_geometry"][key] = {
                    "product_logical_shape": list(result.shape),
                    "product_padded_shape": list(shape),
                    "product_bytes_from_observed_padded_shape": math.prod(shape) * 4,
                    "first_k_half_padded_shape_source_derived": list(half_shape),
                    "first_k_half_bytes_source_derived": math.prod(half_shape) * 4,
                    "first_k_half_scope": "Source-derived BF16-independent FP32 slice geometry; not an allocation measurement",
                }
            return result

        def cpu(value):
            return ttnn.to_torch(value).clone()

        def checks(actual, expected):
            return {"exact": torch.equal(actual, expected), "finite": bool(torch.isfinite(actual).all())}

        inputs, hosts, references = {}, {}, {}
        with patch.object(ttnn, "multiply", observe_product):
            for name in names:
                width = original[name].input_width
                for count in range(1, 33):
                    values = [(torch.randn(1, 1, count, width) * scale).bfloat16() for scale in (0.03, 0.07)]
                    hosts[name, count] = [
                        ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for value in values
                    ]
                    source = ttnn.from_torch(values[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
                    inputs[name, count] = source
                    expected = []
                    phase = "ordinary"
                    for which in (0, 1):
                        ttnn.copy_host_to_device_tensor(hosts[name, count][which], source)
                        with runtime_audit():
                            output = original[name](source)
                        expected.append(cpu(output))
                        ttnn.deallocate(output)
                    references[name, count] = expected
                    entry = {"name": name, "count": count, "cases": []}
                    phase = "split"
                    for which in (0, 1, 0):
                        ttnn.copy_host_to_device_tensor(hosts[name, count][which], source)
                        with runtime_audit():
                            output = split[name](source)
                        entry["cases"].append({"input": which, **checks(cpu(output), expected[which])})
                        ttnn.deallocate(output)
                    entry["passed"] = all(case["exact"] and case["finite"] for case in entry["cases"])
                    report["eager"].append(entry)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    assert entry["passed"], f"GR split mismatch {name} Q{count}"

            report["all_eager_signatures_warmed_before_any_capture"] = True
            for name in names:
                for count in trace_counts:
                    source = inputs[name, count]
                    ttnn.copy_host_to_device_tensor(hosts[name, count][0], source)
                    phase = "capture"
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    with runtime_audit():
                        output = split[name](source)
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    trace_allocation_tracker.acknowledge_corruptible(output)
                    entry = {"name": name, "count": count, "trace_id": int(trace), "cases": []}
                    for which, eager_count in zip((0, 1, 0), (1, 16, 31)):
                        ttnn.copy_host_to_device_tensor(hosts[name, count][which], source)
                        with runtime_audit():
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                        before = cpu(output)
                        case = {"input": which, "before_interleave": checks(before, references[name, count][which])}
                        eager_source = inputs[name, eager_count]
                        assert eager_source.buffer_address() != source.buffer_address()
                        ttnn.copy_host_to_device_tensor(hosts[name, eager_count][1], eager_source)
                        phase = "live_eager"
                        with runtime_audit():
                            eager_output = split[name](eager_source)
                        case["eager_count"] = eager_count
                        case["eager"] = checks(cpu(eager_output), references[name, eager_count][1])
                        ttnn.deallocate(eager_output)
                        with runtime_audit():
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                        after = cpu(output)
                        case["after_interleave"] = checks(after, references[name, count][which])
                        case["repeat_exact"] = torch.equal(before, after)
                        entry["cases"].append(case)
                    entry["passed"] = all(
                        case["repeat_exact"]
                        and all(
                            all(case[field].values()) for field in ("before_interleave", "eager", "after_interleave")
                        )
                        for case in entry["cases"]
                    )
                    report["traces"].append(entry)
                    ttnn.release_trace(mesh, trace)
                    trace = None
                    ttnn.deallocate(output)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    assert entry["passed"], f"GR split trace mismatch {name} Q{count}"
        report.update(passed=True, runtime_audit=True, trace_allocation_tracking=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"passed": report["passed"], "eager_plans": len(report["eager"]), "trace_plans": len(report["traces"])}
        )
    )


if __name__ == "__main__":
    main()
