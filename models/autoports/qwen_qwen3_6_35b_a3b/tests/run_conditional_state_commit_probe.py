# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact conditional state-slab writer versus existing temporary-state commit.

Reuses the established real/adversarial recurrence fixtures. Run device mode
serialized under an external timeout after applying the archived candidate.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_stage7_state_slab_probe import fixtures

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "acceptance"), default="acceptance")
    parser.add_argument("--rounded", type=int, choices=(0, 1), default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "readiness_vllm/conditional_state_commit/result.json")
    args = parser.parse_args()
    assert 4 <= args.steps <= 128
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        status="RUNNING",
        command=sys.argv,
        mode=args.mode,
        rounded=bool(args.rounded),
        steps=args.steps,
        hardware_verified=False,
        mesh=[1, 4],
        tracking=True,
        skip_program_cache=False,
        scope="State-slab component only; no generator, logits, sampling, serving or performance claim",
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), ROOT / "tt/mesh_state_slab_recurrence.py", ROOT / "tt/slot_state.py")
        },
        cases=[],
        captures=0,
        releases=0,
        cleanup_errors=[],
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    mesh, traces, outputs = None, {}, {}
    failure = None
    try:
        torch.set_num_threads(4)
        cases, result["fixture_provenance"] = fixtures(ROOT)
        assert all(torch.isfinite(value).all() for _, values in cases for value in values.values())
        result["fixture_names"] = [name for name, _ in cases]
        if args.mode == "cpu":
            result.update(status="PASS", tracking="not_exercised", skip_program_cache="not_exercised")
            return

        os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
        os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
        assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
        assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
        from ttnn.tools import trace_allocation_tracker

        import ttnn as t
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.mesh_state_slab_recurrence import StateSlabRecurrence
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.slot_state import copy_if

        t.set_fabric_config(t.FabricConfig.FABRIC_1D)
        mesh = t.open_mesh_device(t.MeshShape(1, 4), trace_region_size=64 * 1024 * 1024)
        engine = StateSlabRecurrence(t, mesh)

        def upload(key, value, device=False):
            kwargs = dict(
                dtype=t.bfloat16 if key == "v" else t.float32,
                layout=t.TILE_LAYOUT,
                mesh_mapper=t.ShardTensorToMesh(mesh, dim=1 if key == "state" else 2),
            )
            if device:
                kwargs.update(device=mesh, memory_config=t.DRAM_MEMORY_CONFIG)
            return t.from_torch(value.contiguous(), **kwargs)

        inputs = {key: upload(key, value, True) for key, value in cases[0][1].items() if key != "state"}
        states = {name: upload("state", cases[0][1]["state"], True) for name in ("reference", "candidate")}

        def activity_host(value):
            return t.from_torch(
                torch.tensor([value], dtype=torch.int32),
                dtype=t.int32,
                layout=t.ROW_MAJOR_LAYOUT,
                mesh_mapper=t.ReplicateTensorToMesh(mesh),
            )

        activity = t.to_device(activity_host(0), mesh, memory_config=t.DRAM_MEMORY_CONFIG)

        def call(name):
            args_ = (inputs["q"], inputs["k"], inputs["v"], inputs["beta"], states[name], inputs["decay"])
            if name == "candidate":
                assert engine.supports_native(*args_, output_state=states[name], activity=activity)
                output, new = engine.from_native(
                    *args_, round_state=bool(args.rounded), output_state=states[name], activity=activity
                )
                assert new is states[name], "Candidate must write the original persistent state object"
            else:
                output, new = engine.from_native(*args_, round_state=bool(args.rounded))
                assert new.memory_config() == t.L1_MEMORY_CONFIG
                copy_if(t.to_memory_config(new, t.DRAM_MEMORY_CONFIG), states[name], activity, mesh)
            return output

        def host_bits(tensor):
            values = [t.to_torch(part).contiguous() for part in t.get_device_tensors(tensor)]
            assert len(values) == 4 and all(value.dtype == torch.float32 for value in values)
            return [value.view(torch.int32).clone() for value in values]

        def exact(label, actual, expected):
            assert len(actual) == len(expected) == 4
            for rank, (left, right) in enumerate(zip(actual, expected)):
                assert left.shape == right.shape
                differences = int((left != right).sum())
                assert differences == 0, (label, rank, differences)

        def reset(values):
            for key, value in values.items():
                if key == "state":
                    for target in states.values():
                        t.copy_host_to_device_tensor(upload(key, value), target)
                else:
                    t.copy_host_to_device_tensor(upload(key, value), inputs[key])

        # Complete both exact signatures, copies and readback paths before any
        # capture. Activity changes device data, never the prepared signature.
        for active in (0, 1):
            t.copy_host_to_device_tensor(activity_host(active), activity)
            for name in states:
                value = call(name)
                host_bits(value)
                host_bits(states[name])
                t.deallocate(value)
                del value
        reset(cases[0][1])
        t.copy_host_to_device_tensor(activity_host(0), activity)
        t.synchronize_device(mesh)
        gc.collect()
        result["input_addresses"] = {key: int(value.buffer_address()) for key, value in inputs.items()}
        result["state_addresses"] = {key: int(value.buffer_address()) for key, value in states.items()}
        result["activity_address"] = int(activity.buffer_address())
        for name in states:
            trace = t.begin_trace_capture(mesh, cq_id=0)
            traces[name] = trace
            try:
                outputs[name] = call(name)
            finally:
                t.end_trace_capture(mesh, trace, cq_id=0)
            # Only this trace's fully produced output is acknowledged. State,
            # input and predicate buffers predate both captures and remain safe.
            trace_allocation_tracker.acknowledge_corruptible(outputs[name])
            result["captures"] += 1
        entries = mesh.num_program_cache_entries()
        result["acknowledged"] = {
            name: dict(trace_id=int(traces[name]), output_buffer_id=int(value.buffer_unique_id()))
            for name, value in outputs.items()
        }
        result["lifetime"] = (
            "Only the two distinct captured attention-output buffers are acknowledged and retained through release. "
            "Each producer fully writes its output; host snapshots consume both before the next replay. Persistent "
            "state, operands and activity predate capture and are never acknowledged. Reference staging temporaries "
            "die inside its call. No broad allocation scope or program-cache exemption is used."
        )

        for fixture, values in cases:
            for mode in ("eager", "trace"):
                reset(values)
                previous = {name: host_bits(state) for name, state in states.items()}
                for step in range(args.steps):
                    # An inactive replay followed by active replay tests dynamic
                    # activity. Consecutive zero/one steps also test stable inputs.
                    active = (0, 1, 0, 0, 1, 1, 0, 1)[step % 8]
                    unchanged = step % 8 in (3, 5)
                    if not unchanged:
                        changed = cases[step % 2][1] if fixture.startswith("real") else values
                        for key, value in changed.items():
                            if key != "state":
                                t.copy_host_to_device_tensor(upload(key, value), inputs[key])
                        t.copy_host_to_device_tensor(activity_host(active), activity)
                    assert {key: int(value.buffer_address()) for key, value in inputs.items()} == result[
                        "input_addresses"
                    ]
                    assert {key: int(value.buffer_address()) for key, value in states.items()} == result[
                        "state_addresses"
                    ]
                    assert int(activity.buffer_address()) == result["activity_address"]
                    actual = {}
                    if mode == "eager":
                        for name in states:
                            value = call(name)
                            actual[name] = host_bits(value)
                            t.deallocate(value)
                            del value
                    else:
                        gc.collect()
                        for name, trace in traces.items():
                            unsafe = t._ttnn.operations.trace.get_unsafe_tracked_ids(mesh, trace)
                            assert not unsafe, (name, list(unsafe))
                            t.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        actual = {name: host_bits(value) for name, value in outputs.items()}
                    current = {name: host_bits(state) for name, state in states.items()}
                    exact(f"{fixture}/{mode}/{step}/output", actual["candidate"], actual["reference"])
                    exact(f"{fixture}/{mode}/{step}/state", current["candidate"], current["reference"])
                    if not active:
                        for name in states:
                            exact(f"{fixture}/{mode}/{step}/{name}/inactive", current[name], previous[name])
                    previous = current
                    assert mesh.num_program_cache_entries() == entries
                    result["cases"].append(
                        dict(
                            fixture=fixture,
                            mode=mode,
                            step=step,
                            active=active,
                            unchanged_inputs=unchanged,
                            exact_ranks=4,
                        )
                    )
                save()
                print(f"CONDITIONAL_STATE_COMMIT_PASS {fixture} {mode}", flush=True)
        result.update(status="PASS", hardware_verified=True, program_cache_entries=entries)
    except BaseException as error:
        failure = error
        result.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
    finally:
        if mesh is not None:
            for trace in traces.values():
                try:
                    t.release_trace(mesh, trace)
                    result["releases"] += 1
                except BaseException as error:
                    result["cleanup_errors"].append(repr(error))
            try:
                t.close_mesh_device(mesh)
            except BaseException as error:
                result["cleanup_errors"].append(repr(error))
        if result["cleanup_errors"]:
            result["status"] = "FAIL"
        save()
        print(
            json.dumps(
                dict(status=result["status"], hardware_verified=result["hardware_verified"], cases=len(result["cases"]))
            )
        )
    if failure is not None:
        raise failure
    if result["status"] != "PASS":
        raise RuntimeError(f"Conditional state-commit probe cleanup failed: {result['cleanup_errors']}")


if __name__ == "__main__":
    main()
