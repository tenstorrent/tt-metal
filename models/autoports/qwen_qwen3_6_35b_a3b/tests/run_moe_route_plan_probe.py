# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact route-plan component discriminator; run serialized under an external timeout.

Requires the opt-in route-plan patch. This is not full-model or serving evidence.
No profiler is used. All reference/helper/readback operations warm before capture.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"

import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenModel
from models.autoports.qwen_qwen3_6_35b_a3b.tt.moe_route_mask import mask_inactive_routes

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("masked_ids", "counts", "offsets")


def cpu_plan(ids, slots, active, rank):
    """Independent scatter-to-one semantics, including duplicate inactive skip IDs."""
    rows = ids.to(torch.int64).reshape(32, 8)
    assert all(len(set(row.tolist())) == 8 for row in rows), "Live TopK IDs must be unique"
    assert int(rows.min()) >= 0 and int(rows.max()) < 256
    masked = rows.clone()
    for row in range(32):
        if row >= len(slots) or slots[row] not in active:
            masked[row] = 64 if rank == 0 else 0
    selected = torch.zeros((32, 256), dtype=torch.int64)
    selected.scatter_(1, masked, 1)
    counts = selected.sum(dim=0)
    local = (torch.arange(256) >= rank * 64) & (torch.arange(256) < (rank + 1) * 64)
    counts *= local
    aligned = ((counts + 31) // 32) * 32
    offsets = aligned.cumsum(0) - aligned
    # Independent occurrence count over live rows checks the CPU scatter model.
    occurrences = torch.zeros(256, dtype=torch.int64)
    for row, slot in enumerate(slots):
        if slot in active:
            for expert in rows[row].tolist():
                if rank * 64 <= expert < (rank + 1) * 64:
                    occurrences[expert] += 1
    assert torch.equal(counts, occurrences)
    assert int(counts.max()) <= 32 and int(aligned.sum()) <= 2048
    assert bool((offsets % 32 == 0).all())
    return masked.reshape(1, 1, 32, 8), counts.reshape(1, 256), offsets.reshape(1, 256)


def exact(actual, expected, label):
    assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
    if not torch.equal(actual, expected):
        unequal = actual != expected
        index = int(unequal.reshape(-1).nonzero()[0])
        raise AssertionError(
            f"{label}: differing={int(unequal.sum())}, first={index}, "
            f"actual={actual.reshape(-1)[index].item()}, expected={expected.reshape(-1)[index].item()}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--trace-region-size", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--output", type=Path, default=ROOT / "readiness_vllm/route_plan_probe/result.json")
    args = parser.parse_args()
    assert args.context >= 1024
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        status="RUNNING",
        command=sys.argv,
        mesh=[1, 4],
        layers=[0],
        context=args.context,
        tracking=True,
        skip_program_cache=False,
        trace_region_size=args.trace_region_size,
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                ROOT / "tt/moe_route_mask.py",
                ROOT / "tt/multichip_decoder.py",
                ROOT / "tt/model.py",
            )
        },
        scope="Exact three-output routing component; not expert arithmetic, full-model logits, or serving performance",
        comparison="Actual TTNN scatter/sum/local-mask/ceil/BF16-prefix-matmul/subtract/casts plus independent CPU integers",
        limitations=[
            "Logical output values are compared; raw physical tile padding bytes are not separately inspected."
        ],
        cases=[],
        replays=[],
        captures=0,
        releases=0,
        cleanup_errors=[],
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    mesh = model = trace = trace_outputs = None
    persistent = []
    failure = None
    try:
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.moe_route_mask import plan_inactive_routes

        torch.set_num_threads(8)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size)
        model = QwenModel(mesh, checkpoint=args.model_dir, layer_indices=[0], max_seq_len=args.context)
        layer = model.layers[0]
        assert layer.local_expert_dispatch

        def upload(value, dtype, layout):
            return ttnn.from_torch(
                value.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        def refresh(value, target):
            host = ttnn.from_torch(
                value.contiguous(),
                dtype=target.dtype,
                layout=target.layout,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            ttnn.copy_host_to_device_tensor(host, target)

        def activity(active):
            value = torch.zeros(32, dtype=torch.int32)
            value[list(active)] = 1
            return value

        broad = torch.arange(256, dtype=torch.int32).reshape(1, 1, 32, 8)
        maximum = torch.tensor([0, 1, 64, 65, 128, 129, 192, 193], dtype=torch.int32).repeat(32).reshape(1, 1, 32, 8)
        rotated = (broad + 73) % 256
        device_ids = upload(broad, ttnn.uint32, ttnn.TILE_LAYOUT)
        device_active = upload(activity(range(32)), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        persistent.extend((device_ids, device_active))

        def host_outputs(outputs):
            expected_shapes = ((1, 1, 32, 8), (1, 256), (1, 256))
            expected_layouts = (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT)
            hosts = []
            for name, tensor, shape, layout in zip(NAMES, outputs, expected_shapes, expected_layouts):
                assert tuple(tensor.shape) == shape and tensor.dtype == ttnn.uint32, name
                assert tensor.layout == layout and tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG, name
                locals_ = ttnn.get_device_tensors(tensor)
                assert len(locals_) == 4
                hosts.append([ttnn.to_torch(local).to(torch.int64).clone() for local in locals_])
            return hosts

        def reference(ids, slots):
            # Keep this graph explicit: it is the route-mask-only implementation,
            # not a call to a decoder that may enable the candidate internally.
            masked = mask_inactive_routes(ids, device_active, layer.ep_reduce_table, slots, mesh)
            zero, one = layer.mask_zero[:, :, :32, :], layer.mask_one[:, :, :32, :]
            selected = ttnn.to_layout(
                ttnn.scatter(zero, dim=-1, index=ttnn.to_layout(masked, ttnn.ROW_MAJOR_LAYOUT), src=one),
                ttnn.TILE_LAYOUT,
            )
            counts = ttnn.sum(ttnn.typecast(selected, ttnn.float32), dim=2, keepdim=True)
            counts = ttnn.multiply(counts, layer.local_expert_mask)
            aligned = ttnn.multiply(ttnn.ceil(ttnn.multiply(counts, 1 / 32)), 32)
            prefix = ttnn.matmul(
                ttnn.typecast(aligned, ttnn.bfloat16),
                layer.expert_prefix_sum,
                dtype=ttnn.float32,
                compute_kernel_config=layer.compute,
            )
            offsets = ttnn.subtract(prefix, aligned)

            def ints(tensor):
                return ttnn.reshape(ttnn.to_layout(ttnn.typecast(tensor, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), (1, 256))

            return masked, ints(counts), ints(offsets)

        def reference_host(ids, slots):
            outputs = reference(ids, slots)
            try:
                return host_outputs(outputs)
            finally:
                for tensor in outputs:
                    ttnn.deallocate(tensor)
                del outputs
                gc.collect()

        def compare(actual, reference_values, host_ids, slots, active, label):
            for rank in range(4):
                expected = cpu_plan(host_ids[rank], slots, active, rank)
                for index, name in enumerate(NAMES):
                    exact(reference_values[index][rank], expected[index], f"{label}.reference.{name}.rank{rank}")
                    exact(actual[index][rank], reference_values[index][rank], f"{label}.candidate.{name}.rank{rank}")
            return dict(exact_all_four_ranks=True, exact_cpu=True, exact_reference=True)

        def eager_case(label, ids, host_ids, slots, active):
            refresh(activity(active), device_active)
            outputs = plan_inactive_routes(ids, device_active, layer.ep_reduce_table, slots, mesh)
            try:
                actual = host_outputs(outputs)
            finally:
                for tensor in outputs:
                    ttnn.deallocate(tensor)
                del outputs
            ref = reference_host(ids, slots)
            check = compare(actual, ref, host_ids, slots, active, label)
            result["cases"].append(dict(name=label, slots=list(slots), active=list(active), **check))
            save()
            print(f"ROUTE_PLAN_CASE_PASS {label}", flush=True)

        # This phase warms every bucket mapping, the reference graph, refresh and
        # readback paths before the only capture. No helper output survives it.
        mappings = ((0, 31), (31, 0, 17, 7), tuple(reversed(range(32))))
        for slots in mappings:
            for label, active in (("none", ()), ("all_mapped", slots), ("high31", (31,))):
                eager_case(f"b{len(slots)}_{label}", device_ids, [broad] * 4, slots, active)
        eager_case("b2_outside_mapping", device_ids, [broad] * 4, (0, 31), (17,))
        full = tuple(range(32))
        refresh(maximum, device_ids)
        eager_case("maximum_count32", device_ids, [maximum] * 4, full, full)
        refresh(broad, device_ids)
        eager_case("broad64_local_experts_and_offrank_tails", device_ids, [broad] * 4, full, full)

        tied_logits = upload(torch.zeros((1, 1, 32, 256)), ttnn.float32, ttnn.TILE_LAYOUT)
        tied_values = tied_ids = None
        try:
            tied_values, tied_ids = ttnn.topk(tied_logits, k=8, dim=-1, sorted=True)
            tied_host = [ttnn.to_torch(local).to(torch.int64).clone() for local in ttnn.get_device_tensors(tied_ids)]
            assert len(tied_host) == 4
            eager_case("actual_tied_fp32_topk_unique_ids", tied_ids, tied_host, full, full)
        finally:
            for tensor in (tied_values, tied_ids, tied_logits):
                if tensor is not None:
                    ttnn.deallocate(tensor)
            del tied_values, tied_ids, tied_logits
        gc.collect()
        ttnn.synchronize_device(mesh)

        refresh(broad, device_ids)
        refresh(activity(full), device_active)
        input_addresses = [int(tensor.buffer_address()) for tensor in persistent]
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            trace_outputs = plan_inactive_routes(device_ids, device_active, layer.ep_reduce_table, full, mesh)
        finally:
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
        result["captures"] = 1
        result["trace_id"] = int(trace)
        result["lifetime"] = (
            "Exactly the three captured output buffers are acknowledged. One retained trace fully writes each output "
            "on every replay; outputs remain allocated through release and are read to CPU before another replay. "
            "Persistent inputs and layer tables predate capture and are never acknowledged. Eager reference outputs "
            "are read and deallocated, and intermediate references die before the next replay. No allocation scope "
            "or program-cache exemption is used. Diagnostic host reads synchronize after nonblocking execute."
        )
        result["acknowledged"] = []
        for name, tensor in zip(NAMES, trace_outputs):
            trace_allocation_tracker.acknowledge_corruptible(tensor)
            result["acknowledged"].append(dict(name=name, buffer_id=int(tensor.buffer_unique_id())))
        replay_cases = (
            ("initial", broad, full, True),
            ("changed_ids", maximum, full, True),
            ("changed_activity", maximum, (31,), True),
            ("changed_both", rotated, (0, 7, 17, 31), True),
            ("unchanged_inputs", rotated, (0, 7, 17, 31), False),
        )
        for label, ids_host, active, write_inputs in replay_cases:
            if write_inputs:
                refresh(ids_host, device_ids)
                refresh(activity(active), device_active)
            assert [int(tensor.buffer_address()) for tensor in persistent] == input_addresses
            gc.collect()
            unsafe = ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(mesh, trace)
            assert not unsafe, (label, list(unsafe))
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            actual = host_outputs(trace_outputs)
            ref = reference_host(device_ids, full)
            check = compare(actual, ref, [ids_host] * 4, full, active, label)
            result["replays"].append(
                dict(name=label, active=list(active), input_refresh=write_inputs, trace_id=int(trace), **check)
            )
            save()
            print(f"ROUTE_PLAN_REPLAY_PASS {label}", flush=True)
        result["status"] = "PASS"
    except BaseException as error:
        failure = error
        result.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
    finally:
        if mesh is not None:
            try:
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                    result["releases"] += 1
                if trace_outputs is not None:
                    for tensor in trace_outputs:
                        ttnn.deallocate(tensor)
                for tensor in persistent:
                    ttnn.deallocate(tensor)
            except BaseException as error:
                result["cleanup_errors"].append(repr(error))
            finally:
                try:
                    ttnn.close_mesh_device(mesh)
                except BaseException as error:
                    result["cleanup_errors"].append(repr(error))
        if result["cleanup_errors"]:
            result["status"] = "FAIL"
        save()
    print(json.dumps(dict(status=result["status"], cases=len(result["cases"]), replays=len(result["replays"]))))
    if failure is not None:
        raise failure
    if result["status"] != "PASS":
        raise RuntimeError(f"Route-plan probe cleanup failed: {result['cleanup_errors']}")


if __name__ == "__main__":
    main()
