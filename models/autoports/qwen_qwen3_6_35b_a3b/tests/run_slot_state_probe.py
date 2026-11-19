# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Focused slot-state device probe. Run serialized under an external timeout.

Example: TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0
 timeout 1800 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_slot_state_probe

This checks state primitives, not full-model correctness or performance.
"""

import argparse
import gc
import json
import os
from pathlib import Path

# Tracker configuration must be set before TTNN's first import.
os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"

import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt.slot_state import (
    advance_position,
    advance_seeds,
    copy_if,
    merge_rows,
    position_inputs,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "doc/full_model/slot_state_probe"
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=10_000_000)
    trace = None
    result = dict(
        status="RUNNING",
        mesh=[1, 4],
        tracker=True,
        skip_program_cache=False,
        scope="slot state primitives only",
        steps=[],
        captures=0,
        releases=0,
    )

    def upload(value, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(
            value,
            dtype=dtype,
            layout=layout,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def refresh(value, dest):
        host = ttnn.from_torch(
            value.contiguous(), dtype=dest.dtype, layout=dest.layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
        )
        ttnn.copy_host_to_device_tensor(host, dest)

    def assert_values(tensor, expected, label, bits=False):
        for chip, local in enumerate(ttnn.get_device_tensors(tensor)):
            actual = ttnn.to_torch(local)
            assert tuple(actual.shape) == tuple(expected.shape), (label, actual.shape, expected.shape)
            if bits:
                assert torch.equal(actual.contiguous().view(torch.int32), expected.contiguous().view(torch.int32)), (
                    label,
                    chip,
                )
            else:
                assert torch.equal(actual.to(torch.int64), expected.to(torch.int64)), (label, chip, actual, expected)

    try:
        scalar_activity = upload(torch.tensor([1], dtype=torch.int32), ttnn.int32)
        position = upload(torch.tensor([123], dtype=torch.int32), ttnn.int32)
        # Mantissas deliberately include bits a BF16 round trip would destroy.
        fp_bits = (torch.arange(8 * 128 * 128, dtype=torch.int32) % 65535 + 0x3F800001).reshape(1, 8, 128, 128)
        fp_source_host = fp_bits.view(torch.float32)
        fp_initial_host = (fp_bits + 0x00800000).view(torch.float32)
        fp_source = upload(fp_source_host, ttnn.float32, ttnn.TILE_LAYOUT)
        fp_destination = upload(fp_initial_host, ttnn.float32, ttnn.TILE_LAYOUT)
        zero_source_host = torch.arange(32 * 32).reshape(1, 1, 32, 32).to(torch.bfloat16)
        zero_source = upload(zero_source_host, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        zero_destination = upload(torch.ones_like(zero_source_host), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        vectors = []
        for dtype, unsigned in ((ttnn.int32, False), (ttnn.uint32, True)):
            host_dtype = torch.uint32 if unsigned else torch.int32
            staging_host = (torch.arange(32, dtype=torch.int64) * 100000007 - 2147483648).to(host_dtype)
            destination_host = (torch.arange(32, dtype=torch.int64) * 3 + 17).to(host_dtype)
            seeds_host = (torch.arange(32, dtype=torch.int64) + 4294967280).to(host_dtype)
            mask_host = (torch.arange(32) % 2).to(host_dtype)
            vectors.append(
                dict(
                    dtype=dtype,
                    host_dtype=host_dtype,
                    staging_host=staging_host,
                    destination_host=destination_host,
                    seeds_host=seeds_host,
                    staging=upload(staging_host, dtype),
                    destination=upload(destination_host, dtype),
                    seeds=upload(seeds_host, dtype),
                    mask=upload(mask_host, dtype),
                )
            )

        def workload():
            rope, current = position_inputs(position, scalar_activity, mesh)
            copy_if(fp_source, fp_destination, scalar_activity, mesh)
            copy_if(zero_source, zero_destination, scalar_activity, mesh, zero_if_inactive=True)
            advance_position(position, scalar_activity, mesh)
            for row in vectors:
                merge_rows(row["staging"], row["destination"], row["mask"], mesh)
                advance_seeds(row["seeds"], row["mask"], mesh)
            return rope, current

        warm = workload()
        ttnn.synchronize_device(mesh)
        del warm
        gc.collect()

        # Restore warmup state before capture; repeat after capture so the probe
        # does not depend on whether capture itself executes device operations.
        def restore():
            refresh(torch.tensor([123], dtype=torch.int32), position)
            refresh(fp_initial_host, fp_destination)
            refresh(torch.ones_like(zero_source_host), zero_destination)
            for row in vectors:
                refresh(row["destination_host"], row["destination"])
                refresh(row["seeds_host"], row["seeds"])

        restore()
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        outputs = workload()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        result["captures"] = 1
        # Exact captured outputs only: one trace overwrites these every replay,
        # and each is consumed before the next replay. No unrelated alias or
        # alternate trace consumes them. No allocation scope is exempted.
        for tensor in outputs:
            trace_allocation_tracker.acknowledge_corruptible(tensor)
        result["acknowledged"] = [
            dict(
                name=name,
                buffer_id=int(tensor.buffer_unique_id()),
                invariant="single trace fully writes output each replay; host consumes before next replay",
            )
            for name, tensor in zip(("rope_position", "attention_position"), outputs)
        ]
        restore()
        expected_position = 123
        expected_fp = fp_initial_host.clone()
        for row in vectors:
            row["expected_destination"] = row["destination_host"].clone()
            row["expected_seeds"] = row["seeds_host"].clone()

        def replay_and_check(step, active, mask):
            nonlocal expected_position, expected_fp
            refresh(torch.tensor([active], dtype=torch.int32), scalar_activity)
            for row in vectors:
                refresh(mask.to(row["host_dtype"]), row["mask"])
            before = expected_position
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            if active:
                expected_position += 1
                expected_fp = fp_source_host
            assert_values(outputs[0], torch.tensor([before if active else 0], dtype=torch.int32), "rope")
            assert_values(outputs[1], torch.tensor([before if active else -1], dtype=torch.int32), "attention")
            assert_values(position, torch.tensor([expected_position], dtype=torch.int32), "position")
            assert_values(fp_destination, expected_fp, "FP32 exact bits", bits=True)
            assert_values(
                zero_destination, zero_source_host if active else torch.zeros_like(zero_source_host), "zero_if_inactive"
            )
            for row in vectors:
                use = mask.bool()
                # torch.where on int64 keeps UINT32 test values exact.
                row["expected_destination"] = torch.where(
                    use, row["staging_host"].to(torch.int64), row["expected_destination"].to(torch.int64)
                ).to(row["host_dtype"])
                row["expected_seeds"] = (row["expected_seeds"].to(torch.int64) + use.to(torch.int64)).to(
                    row["host_dtype"]
                )
                assert_values(row["destination"], row["expected_destination"], "merge_rows")
                assert_values(row["seeds"], row["expected_seeds"], "advance_seeds")
            result["steps"].append(
                dict(
                    step=step,
                    active=active,
                    active_rows=mask.nonzero().flatten().tolist(),
                    position=expected_position,
                    trace_id=int(trace),
                )
            )
            print(f"SLOT_STATE_REPLAY_PASS step={step} active={active} position={expected_position}", flush=True)

        patterns = [
            (0, torch.zeros(32, dtype=torch.int32)),
            (1, (torch.arange(32) % 2).to(torch.int32)),
            (1, torch.tensor([int(i in (0, 7, 31)) for i in range(32)], dtype=torch.int32)),
            (0, torch.ones(32, dtype=torch.int32)),
        ]
        for step, (active, mask) in enumerate(patterns):
            replay_and_check(step, active, mask)

        unsafe = ttnn.allocate_tensor_on_device(ttnn.Shape((1, 1, 32, 32)), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh)
        unsafe_id = int(unsafe.buffer_unique_id())
        try:
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        except RuntimeError as error:
            message = str(error)
            assert "device buffer(s) still alive before" in message and f"Buffer {unsafe_id}" in message, message
            result["positive_control"] = dict(status="REJECTED_AS_EXPECTED", buffer_id=unsafe_id, message=message)
            print(f"SLOT_STATE_POSITIVE_CONTROL_PASS buffer={unsafe_id}", flush=True)
        else:
            raise AssertionError("Trace allocation tracker did not reject the live unsafe allocation")
        finally:
            ttnn.deallocate(unsafe)
            del unsafe
            gc.collect()
        # Finish with a replay after positive-control cleanup. This also exposes
        # any accidental late program/persistent allocation in earlier checks.
        replay_and_check(4, 1, torch.ones(32, dtype=torch.int32))
        result["status"] = "PASS"
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
            result["releases"] += 1
        ttnn.close_mesh_device(mesh)
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"SLOT_STATE_PROBE_PASS {args.output_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
