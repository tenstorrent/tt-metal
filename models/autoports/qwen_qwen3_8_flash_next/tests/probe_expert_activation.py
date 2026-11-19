# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact real SparseMoe private activation reuse control.

Ordinary versus guarded in-place SiLU/multiply at every Q1..32 with A/B/A.
Changed-input trace replay interleaves independent warmed eager requests.
This isolates exact scheduling equivalence, not HF/full decoder acceptance.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--trace-counts", default="1,17,27,29,30,32")
    parser.add_argument("--expert-dtype", choices=("bfloat16", "bfloat8_b"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace_counts = [int(value) for value in args.trace_counts.split(",")]
    if not trace_counts or any(value not in range(1, 33) for value in trace_counts):
        parser.error("Trace counts must be1..32")
    import torch
    from transformers import AutoConfig
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.capacity_expert_activation import InplaceExpertActivation
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
    from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import SparseMoe, WeightLoader

    if not trace_allocation_tracker.TRACE_ALLOC_TRACKING:
        raise ValueError("TT_METAL_TRACE_ALLOC_TRACKING=1 required")
    torch.set_num_threads(8)
    torch.manual_seed(1911)
    names = ["mlp"]
    weights = load_weights(args.snapshot, 1, "mlp.")
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    report = {
        "scope": __doc__,
        "snapshot": str(args.snapshot),
        "eager": [],
        "traces": [],
        "expert_dtype": args.expert_dtype,
        "passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        loader = WeightLoader(weights, mesh)
        original = {"mlp": SparseMoe(loader, config, expert_dtype=getattr(ttnn, args.expert_dtype))}
        split = {"mlp": InplaceExpertActivation(mesh, original["mlp"])}
        report["activation_policy"] = split["mlp"].report

        def cpu(value):
            return ttnn.to_torch(value).clone()

        def checks(actual, expected):
            return {"exact": torch.equal(actual, expected), "finite": bool(torch.isfinite(actual).all())}

        inputs, hosts, references = {}, {}, {}
        for name in names:
            width = original[name].width
            for count in range(1, 33):
                values = [(torch.randn(1, 1, count, width) * scale).bfloat16() for scale in (0.03, 0.07)]
                hosts[name, count] = [
                    ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for value in values
                ]
                source = ttnn.from_torch(values[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
                inputs[name, count] = source
                expected = []
                for which in (0, 1):
                    ttnn.copy_host_to_device_tensor(hosts[name, count][which], source)
                    with runtime_audit():
                        output = original[name](source)
                    expected.append(cpu(output))
                    ttnn.deallocate(output)
                references[name, count] = expected
                entry = {"name": name, "count": count, "cases": []}
                for which in (0, 1, 0):
                    ttnn.copy_host_to_device_tensor(hosts[name, count][which], source)
                    with runtime_audit():
                        output = split[name](source)
                    entry["cases"].append({"input": which, **checks(cpu(output), expected[which])})
                    ttnn.deallocate(output)
                entry["passed"] = all(case["exact"] and case["finite"] for case in entry["cases"])
                report["eager"].append(entry)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                assert entry["passed"], f"Expert activation reuse mismatch {name} Q{count}"

        report["all_eager_signatures_warmed_before_any_capture"] = True
        for name in names:
            for count in trace_counts:
                source = inputs[name, count]
                ttnn.copy_host_to_device_tensor(hosts[name, count][0], source)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                with runtime_audit():
                    output = split[name](source)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                trace_allocation_tracker.acknowledge_corruptible(output)
                entry = {"name": name, "count": count, "trace_id": int(trace), "cases": []}
                eager_counts = [q for q in (1, 16, 31, 7) if q != count][:3]
                for which, eager_count in zip((0, 1, 0), eager_counts):
                    ttnn.copy_host_to_device_tensor(hosts[name, count][which], source)
                    with runtime_audit():
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    before = cpu(output)
                    case = {"input": which, "before_interleave": checks(before, references[name, count][which])}
                    eager_source = inputs[name, eager_count]
                    assert eager_source.buffer_address() != source.buffer_address()
                    ttnn.copy_host_to_device_tensor(hosts[name, eager_count][1], eager_source)
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
                    and all(all(case[field].values()) for field in ("before_interleave", "eager", "after_interleave"))
                    for case in entry["cases"]
                )
                report["traces"].append(entry)
                ttnn.release_trace(mesh, trace)
                trace = None
                ttnn.deallocate(output)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                assert entry["passed"], f"Expert activation reuse trace mismatch {name} Q{count}"
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
