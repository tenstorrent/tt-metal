# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-shape private reduction: poisoned padding, exact native output and allocation graphs."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.capacity_private_reduction import (
        allocation_summary,
        sum_private_down,
    )
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit

    if not trace_allocation_tracker.TRACE_ALLOC_TRACKING:
        raise ValueError("TT_METAL_TRACE_ALLOC_TRACKING=1 required")
    torch.set_num_threads(8)
    torch.manual_seed(1931)
    report = {
        "scope": __doc__,
        "weights": "No weights; exact real512-expert/down2560 geometry and native BF16 operands",
        "eager": [],
        "traces": [],
        "passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    graph_dir = args.output.with_suffix(".graphs")
    graph_dir.mkdir(exist_ok=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        cpu_values = [(torch.randn(1, 512, 32, 2560) * scale).bfloat16() for scale in (0.01, 0.02)]
        source = ttnn.from_torch(cpu_values[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        eager_source = ttnn.from_torch(cpu_values[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        views = {q: ttnn.reshape(source, (1, 512, q, 2560), (1, 512, 32, 2560)) for q in range(1, 33)}
        eager_views = {q: ttnn.reshape(eager_source, (1, 512, q, 2560), (1, 512, 32, 2560)) for q in range(1, 33)}
        banks = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM).num_banks
        references = {}

        def hosts(q):
            result = []
            for which, value in enumerate(cpu_values):
                poisoned = value.clone()
                poisoned[..., q:, :] = float("nan") if which == 0 else float("inf")
                poisoned[..., q:, ::2] = -17.5
                result.append(ttnn.from_torch(poisoned, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
            return result

        def check(output, q, which, base=source):
            actual = ttnn.to_torch(output)
            mutated = ttnn.to_torch(base)
            return {
                "exact": torch.equal(actual, references[q][which]),
                "finite": bool(torch.isfinite(actual).all()),
                "valid_input_unchanged": torch.equal(mutated[..., :q, :], cpu_values[which][..., :q, :]),
                "padding_zero": bool(torch.count_nonzero(mutated[..., q:, :]).item() == 0),
            }

        def graph_call(q, candidate):
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            try:
                with runtime_audit():
                    output = sum_private_down(views[q]) if candidate else ttnn.sum(views[q], dim=1, keepdim=True)
            finally:
                graph = ttnn.graph.end_graph_capture()
            path = graph_dir / f"q{q}_{'private' if candidate else 'ordinary'}.json"
            path.write_text(json.dumps(graph) + "\n")
            return output, {"graph": str(path), **allocation_summary(graph, banks)}

        for q in range(1, 33):
            prepared = hosts(q)
            expected = []
            for which in (0, 1):
                ttnn.copy_host_to_device_tensor(prepared[which], source)
                with runtime_audit():
                    output = ttnn.sum(views[q], dim=1, keepdim=True)
                expected.append(ttnn.to_torch(output).clone())
                ttnn.deallocate(output)
            references[q] = expected
            entry = {"count": q, "cases": []}
            for which in (0, 1, 0):
                ttnn.copy_host_to_device_tensor(prepared[which], source)
                record = {}
                with runtime_audit():
                    output = sum_private_down(views[q], record=record)
                entry["cases"].append({"input": which, **check(output, q, which), "operation": record})
                ttnn.deallocate(output)
            entry["graphs"] = {}
            for candidate in (False, True):
                ttnn.copy_host_to_device_tensor(prepared[0], source)
                output, summary = graph_call(q, candidate)
                entry["graphs"]["private" if candidate else "ordinary"] = summary
                assert torch.equal(ttnn.to_torch(output), expected[0])
                ttnn.deallocate(output)
            entry["passed"] = all(
                all(case[key] for key in ("exact", "finite", "valid_input_unchanged", "padding_zero"))
                for case in entry["cases"]
            )
            if q < 32:
                entry["clone_avoided"] = (
                    entry["graphs"]["ordinary"]["contains_80mib_allocation"]
                    and not entry["graphs"]["private"]["contains_80mib_allocation"]
                )
                entry["peak_reduced"] = (
                    entry["graphs"]["private"]["additional_peak_allocated_bytes"]
                    < entry["graphs"]["ordinary"]["additional_peak_allocated_bytes"]
                )
                entry["warm_fill_and_view_no_allocation"] = all(
                    case["operation"]["fill_allocated_delta"] == 0 and case["operation"]["view_allocated_delta"] == 0
                    for case in entry["cases"][1:]
                )
                entry["passed"] &= (
                    entry["clone_avoided"] and entry["peak_reduced"] and entry["warm_fill_and_view_no_allocation"]
                )
            report["eager"].append(entry)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            assert entry["passed"], f"Private reduction Q{q} failed"

        for q in (1, 17, 27, 29, 30, 32):
            prepared = hosts(q)
            eager_counts = [v for v in (1, 16, 31, 7) if v != q][:3]
            eager_hosts = {v: hosts(v)[1] for v in eager_counts}
            ttnn.copy_host_to_device_tensor(prepared[0], source)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            with runtime_audit():
                output = sum_private_down(views[q])
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            trace_allocation_tracker.acknowledge_corruptible(output)
            entry = {"count": q, "cases": []}
            for which, eq in zip((0, 1, 0), eager_counts):
                ttnn.copy_host_to_device_tensor(prepared[which], source)
                with runtime_audit():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                case = {"input": which, "first": check(output, q, which)}
                ttnn.copy_host_to_device_tensor(eager_hosts[eq], eager_source)
                with runtime_audit():
                    eager_output = sum_private_down(eager_views[eq])
                case["live_eager"] = check(eager_output, eq, 1, eager_source)
                ttnn.deallocate(eager_output)
                with runtime_audit():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                case["repeat"] = check(output, q, which)
                entry["cases"].append(case)
            entry["passed"] = all(
                all(all(case[phase].values()) for phase in ("first", "live_eager", "repeat")) for case in entry["cases"]
            )
            report["traces"].append(entry)
            ttnn.release_trace(mesh, trace)
            trace = None
            ttnn.deallocate(output)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            assert entry["passed"], f"Private reduction trace Q{q} failed"
        report.update(passed=True, runtime_audit=True, allocation_tracking=True)
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
