# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-shape/real-weight ordinary versus reserved sparse-down control.

Exercises ranks4/6 at every logical count1..32 with changed expert masks/activations and
repeated original input. A fixed-input-address trace separately replays A/B/A.
This proves optional-output equivalence, not full decoder/HF acceptance or a
public capacity limit. No full PLE table is loaded in this isolated control.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--expert-dtype", choices=("bfloat16", "bfloat8_b"), required=True)
    parser.add_argument("--trace-counts", default="1,27,29,30,32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace_counts = [int(v) for v in args.trace_counts.split(",")]
    if not trace_counts or any(v < 1 or v > 32 for v in trace_counts):
        parser.error("Trace counts must be within1..32")
    import torch
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.capacity_sparse_workspace import ReservedSparseDownWorkspace
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
    from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader
    from models.demos.gpt_oss.tt.experts.config import ProgramConfig

    if not trace_allocation_tracker.TRACE_ALLOC_TRACKING:
        raise ValueError("TT_METAL_TRACE_ALLOC_TRACKING=1 required")
    torch.set_num_threads(8)
    torch.manual_seed(9917)
    weights = load_weights(args.snapshot, 1, "mlp.experts.down_proj")
    report = {
        "scope": __doc__,
        "snapshot": str(args.snapshot),
        "expert_dtype": args.expert_dtype,
        "logical_ranks": [4, 6],
        "rank6_scope": "Exact metadata view of real gate/up producer shape observed in reserved_shape_v7; no runtime rank normalization",
        "eager": [],
        "traces": [],
        "passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        loader = WeightLoader(weights, mesh)
        weight = loader.tensor(
            weights["mlp.experts.down_proj"].transpose(-1, -2).unsqueeze(0), getattr(ttnn, args.expert_dtype)
        )
        workspace = ReservedSparseDownWorkspace(mesh, weight)
        report["workspace"] = workspace.report
        program = ProgramConfig(decode_gate_up_cores=(5, 1), decode_down_cores=(8, 1))
        cpu_a = (torch.randn(1, 512, 32, 640) * 0.04).bfloat16()
        cpu_b = (torch.randn(1, 512, 32, 640) * 0.07).bfloat16()
        mask_a = torch.zeros(1, 1, 1, 512, dtype=torch.bfloat16)
        mask_b = mask_a.clone()
        mask_a[..., :10], mask_b[..., 10:20] = 1, 1

        def upload(value, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=layout, device=mesh)

        source = upload(cpu_a)
        mask = upload(mask_a, ttnn.ROW_MAJOR_LAYOUT)
        inputs = {
            (rank, count): ttnn.reshape(
                source, (1,) * (rank - 3) + (512, count, 640), (1,) * (rank - 3) + (512, 32, 640)
            )
            for rank in (4, 6)
            for count in range(1, 33)
        }
        if any(view.buffer_address() != source.buffer_address() for view in inputs.values()):
            raise ValueError("Input metadata views unexpectedly moved storage")
        host_inputs = [ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for v in (cpu_a, cpu_b)]
        host_masks = [ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for v in (mask_a, mask_b)]

        def set_input(which):
            ttnn.copy_host_to_device_tensor(host_inputs[which], source)
            ttnn.copy_host_to_device_tensor(host_masks[which], mask)

        def forward(rank, count, reserved):
            call = workspace.call if reserved else workspace.original
            return call(
                inputs[rank, count],
                weight,
                sparsity=mask,
                nnz=None,
                is_input_a_sparse=True,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                compute_kernel_config=loader.compute,
                program_config=program.get_decode_down_config(count, 2560, k=640),
            )

        def reference(rank, count, which):
            set_input(which)
            with runtime_audit():
                value = forward(rank, count, False)
            cpu = ttnn.to_torch(value).clone()
            metadata = {
                "logical_shape": list(value.shape),
                "padded_shape": list(value.padded_shape),
                "dtype": str(value.dtype),
                "layout": str(value.layout),
            }
            ttnn.deallocate(value)
            if metadata["logical_shape"] != [1, 512, count, 2560] or metadata["padded_shape"] != [1, 512, 32, 2560]:
                raise ValueError(f"Unexpected native ordinary output shape: {metadata}")
            return cpu, metadata

        def check(actual, expected, which):
            return {
                "exact": torch.equal(actual, expected),
                "finite": bool(torch.isfinite(actual).all()),
                "nonzero_active": bool(torch.count_nonzero(actual[..., which * 10 : (which + 1) * 10, :, :]).item()),
                "old_expert_slots_zero": bool(
                    torch.count_nonzero(actual[..., :10, :, :] if which else actual[..., 10:20, :, :]).item() == 0
                ),
            }

        for rank, count in ((rank, count) for rank in (4, 6) for count in range(1, 33)):
            references = [reference(rank, count, which) for which in (0, 1)]
            baseline = [value for value, _ in references]
            entry = {"rank": rank, "count": count, "ordinary_outputs": [meta for _, meta in references], "cases": []}
            for which in (0, 1, 0):
                set_input(which)
                with runtime_audit():
                    result = forward(rank, count, True)
                actual = ttnn.to_torch(result)
                entry["cases"].append(
                    {
                        "input": which,
                        **check(actual, baseline[which], which),
                        "returned_reserved_address": result.buffer_address() == workspace.base.buffer_address(),
                    }
                )
                del result, actual
            entry["passed"] = all(
                all(
                    case[key]
                    for key in (
                        "exact",
                        "finite",
                        "nonzero_active",
                        "old_expert_slots_zero",
                        "returned_reserved_address",
                    )
                )
                for case in entry["cases"]
            )
            report["eager"].append(entry)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            assert entry["passed"], f"Reserved eager mismatch at rank{rank} count{count}"
        for rank, count in ((rank, count) for rank in (4, 6) for count in trace_counts):
            references = [reference(rank, count, which) for which in (0, 1)]
            baseline = [value for value, _ in references]
            set_input(0)
            with runtime_audit():
                forward(rank, count, True)
            ttnn.synchronize_device(mesh)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            with runtime_audit():
                result = forward(rank, count, True)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            # Only the sole returned workspace/output is intentionally mutable.
            trace_allocation_tracker.acknowledge_corruptible(result)
            entry = {
                "rank": rank,
                "count": count,
                "trace_id": int(trace),
                "ordinary_outputs": [meta for _, meta in references],
                "cases": [],
            }
            for which in (0, 1, 0):
                set_input(which)
                with runtime_audit():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                actual = ttnn.to_torch(result)
                entry["cases"].append({"input": which, **check(actual, baseline[which], which)})
                del actual
            entry["passed"] = all(
                all(case[key] for key in ("exact", "finite", "nonzero_active", "old_expert_slots_zero"))
                for case in entry["cases"]
            )
            report["traces"].append(entry)
            ttnn.release_trace(mesh, trace)
            trace = None
            del result
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            assert entry["passed"], f"Reserved trace mismatch at rank{rank} count{count}"
        report.update(runtime_audit=True, allocation_tracking=True, passed=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "eager_counts": len(report["eager"]),
                "trace_counts": trace_counts,
                "workspace_bytes": report["workspace"]["storage_bytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
