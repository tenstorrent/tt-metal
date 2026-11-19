# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Default-off native Q/K reader versus the unchanged transformed reader.

Component exactness only: no full-model, serving, quality or performance claim.
CPU mode does not import TTNN. Device mode must be run by the device owner.
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


def cpu_mapping():
    """Exhaustively check reader indices against logical head/element order."""
    words = torch.arange(4 * 128, dtype=torch.int32).reshape(4, 128) + 0x3F000000
    tiled = torch.full((4, 1024), -1234567, dtype=torch.int32)
    for head in range(4):
        for column in range(128):
            within = column % 32
            tiled[column // 32, head * 16 + (within // 16) * 256 + within % 16] = words[head, column]
    for value_head in range(8):
        head = value_head // 2
        destination = torch.zeros((4, 1024), dtype=torch.int32)
        for tile in range(4):
            # Exactly the two contiguous 64-byte reads in the candidate.
            narrow = torch.cat(
                (tiled[tile, head * 16 : head * 16 + 16], tiled[tile, 256 + head * 16 : 256 + head * 16 + 16])
            )
            for row in range(32):
                destination[tile, (row // 16) * 512 + (row % 16) * 16] = narrow[row]
        mapped = torch.stack(
            [destination[tile, (row // 16) * 512 + (row % 16) * 16] for tile in range(4) for row in range(32)]
        )
        assert torch.equal(mapped, words[head])
        assert int((destination != 0).sum()) == 128
    return dict(
        passed=True,
        value_heads=8,
        qk_heads=4,
        words_per_head=128,
        poisoned_source_padding=True,
        source_logical=[1, 1, 4, 128],
        source_padded=[1, 1, 32, 128],
        consumed_destination="four FP32 tile column0 vectors per worker",
        hardware_verified=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "acceptance"), default="acceptance")
    parser.add_argument("--rounded", type=int, choices=(0, 1), default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--state-output", choices=("persistent", "temporary"), default="persistent")
    parser.add_argument("--output", type=Path, default=ROOT / "readiness_vllm/native_qk_reader/result.json")
    args = parser.parse_args()
    assert 4 <= args.steps <= 128
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        status="RUNNING",
        command=sys.argv,
        mode=args.mode,
        rounded=bool(args.rounded),
        steps=args.steps,
        state_output=args.state_output,
        native_qk_reader={"reference": False, "candidate": True},
        output_scope="all logical output words; state has no tile padding; unused attention-output padding is unspecified",
        source_read_bytes_per_layer={"reference": 1048576, "candidate": 32768},
        candidate_scratch_bytes_per_worker=2048,
        source_padding=dict(value=-31337.75, device_verified=False, rows=[4, 32]),
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
        result["cpu_mapping"] = cpu_mapping()
        cases, result["fixture_provenance"] = fixtures(ROOT)
        finite = {key: value.clone() for key, value in cases[0][1].items()}
        for index, key in enumerate(("q", "k")):
            # Distinguish every head/column/rank; these are non-BF16 FP32 words.
            values = torch.arange(finite[key].numel(), dtype=torch.int32)
            finite[key].copy_((0x3E000001 + values * 257 + index * 65537).view(torch.float32).reshape_as(finite[key]))
        cases.append(("distinct_native_qk_fp32", finite))
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
            if key in ("q", "k"):
                kwargs["pad_value"] = -31337.75
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
            persistent = args.state_output == "persistent"
            assert engine.supports_native(
                *args_,
                output_state=states[name] if persistent else None,
                activity=activity if persistent else None,
            )
            output, new = engine.from_native(
                *args_,
                round_state=bool(args.rounded),
                output_state=states[name] if persistent else None,
                activity=activity if persistent else None,
                native_qk_reader=name == "candidate",
            )
            if persistent:
                assert new is states[name], "Both paths must write the original persistent state object"
            else:
                assert new.memory_config() == t.L1_MEMORY_CONFIG
                copy_if(t.to_memory_config(new, t.DRAM_MEMORY_CONFIG), states[name], activity, mesh)
            return output

        def host_bits(tensor):
            values = [t.to_torch(part).contiguous() for part in t.get_device_tensors(tensor)]
            assert len(values) == 4 and all(value.dtype == torch.float32 for value in values)
            return [value.view(torch.int32).clone() for value in values]

        def padded_qk_bits(tensor):
            values = [
                t.from_device(part).to_torch_with_padded_shape().contiguous() for part in t.get_device_tensors(tensor)
            ]
            assert len(values) == 4
            poison = torch.tensor(-31337.75, dtype=torch.float32).view(torch.int32).item()
            for rank, value in enumerate(values):
                assert tuple(value.shape) == (1, 1, 32, 128) and value.dtype == torch.float32
                assert bool((value.view(torch.int32)[:, :, 4:] == poison).all()), (
                    "device_padding_not_poisoned",
                    rank,
                )
            result["source_padding"]["device_verified"] = True
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

        # Observe the actual uploaded patterns, not assumed host transport.
        # This also tests that the old permute does not canonicalize words which
        # the proposed reader copies bitwise. It does not read internal CBs;
        # recurrence comparisons below exercise the actual candidate reader.
        observed = [set() for _ in range(4)]
        seed_upload_differences = 0
        result["transport"] = dict(
            generations=128,
            expected_distinct_per_rank=65536,
            mapping="uploaded native logical words versus actual old permute logical words",
            internal_cb_readback=False,
        )
        for generation in range(128):
            codes = torch.cat(
                [(torch.arange(512, dtype=torch.int64) + generation * 512 + rank * 16384) % 65536 for rank in range(4)]
            )
            seed = (codes << 16).to(torch.int32).view(torch.float32).reshape(1, 1, 16, 128)
            seeds = {"q": seed, "k": seed.flip(-1).contiguous()}
            for key, value in seeds.items():
                t.copy_host_to_device_tensor(upload(key, value), inputs[key])
                physical_source = padded_qk_bits(inputs[key])
                source = [value[:, :, :4].contiguous() for value in physical_source]
                old = t.reshape(t.permute(inputs[key], (0, 2, 3, 1)), (1, 4, 1, 128, 1))
                transformed = host_bits(old)
                for rank, (native, columns) in enumerate(zip(source, transformed)):
                    expected_seed = value[:, :, rank * 4 : (rank + 1) * 4].contiguous().view(torch.int32)
                    seed_upload_differences += int((native != expected_seed).sum())
                    if key == "q":
                        observed[rank].update(int(word) & 0xFFFFFFFF for word in native.flatten().tolist())
                    assert torch.equal(native.flatten(), columns.flatten()), (
                        "old_permute_changes_bits",
                        generation,
                        key,
                        rank,
                        int((native.flatten() != columns.flatten()).sum()),
                    )
                exact("transport_source_preserved_with_padding", padded_qk_bits(inputs[key]), physical_source)
                t.deallocate(old)
                del old
        result["transport"].update(
            observed_distinct_per_rank=[len(words) for words in observed],
            seed_to_uploaded_word_differences=seed_upload_differences,
            source_to_old_word_differences=0,
            observed_hashes=[
                hashlib.sha256(b"".join(word.to_bytes(4, "little") for word in sorted(words))).hexdigest()
                for words in observed
            ],
        )
        assert all(len(words) == 65536 for words in observed), result["transport"]
        # Arbitrary FP32 low mantissas and nonfinite payloads are transport-only:
        # do not send them through recurrence where NaN arithmetic could hide
        # or change the copied payload. The normal finite fixtures follow.
        edge_words = [
            0x00000000,
            0x80000000,
            0x00000001,
            0x80000001,
            0x00008001,
            0x80008001,
            0x007FFFFF,
            0x807FFFFF,
            0x00800000,
            0x80800000,
            0x00800001,
            0x80800001,
            0x3F800001,
            0xBF800001,
            0x3F807FFF,
            0xBF807FFF,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7F800001,
            0xFF800001,
            0x7FC00001,
            0xFFC00001,
            0x7FFFFFFF,
            0xFFFFFFFF,
        ]
        edge_tensor = torch.tensor(edge_words, dtype=torch.int64).to(torch.int32)
        edge_observed = [set() for _ in range(4)]
        edge_record = dict(
            requested_words=[f"0x{word:08x}" for word in edge_words],
            generations=4,
            seed_to_uploaded_word_differences=0,
            source_to_old_word_differences=0,
            transitions=[],
            scope="transport only, independent of recurrence arithmetic",
        )
        result["arbitrary_fp32_transport"] = edge_record
        for generation in range(4):
            indices = (torch.arange(2048) + generation * 7) % len(edge_words)
            seed = edge_tensor[indices].view(torch.float32).reshape(1, 1, 16, 128)
            for key, value in (("q", seed), ("k", seed.flip(-1).contiguous())):
                t.copy_host_to_device_tensor(upload(key, value), inputs[key])
                physical_source = padded_qk_bits(inputs[key])
                old = t.reshape(t.permute(inputs[key], (0, 2, 3, 1)), (1, 4, 1, 128, 1))
                transformed = host_bits(old)
                for rank, (physical, columns) in enumerate(zip(physical_source, transformed)):
                    native = physical[:, :, :4].contiguous().flatten()
                    expected = value[:, :, rank * 4 : (rank + 1) * 4].contiguous().view(torch.int32).flatten()
                    edge_record["seed_to_uploaded_word_differences"] += int((native != expected).sum())
                    edge_observed[rank].update(int(word) & 0xFFFFFFFF for word in native.tolist())
                    different = (native != columns.flatten()).nonzero().flatten()
                    edge_record["source_to_old_word_differences"] += different.numel()
                    for index in different[:8]:
                        edge_record["transitions"].append(
                            dict(
                                generation=generation,
                                key=key,
                                rank=rank,
                                index=int(index),
                                source=f"0x{int(native[index]) & 0xFFFFFFFF:08x}",
                                old=f"0x{int(columns.flatten()[index]) & 0xFFFFFFFF:08x}",
                            )
                        )
                    assert not different.numel(), ("arbitrary_fp32_old_permute_changes_bits", edge_record)
                exact("fp32_edge_source_preserved_with_padding", padded_qk_bits(inputs[key]), physical_source)
                t.deallocate(old)
                del old
        edge_record["observed_words_per_rank"] = [
            [f"0x{word:08x}" for word in sorted(words)] for words in edge_observed
        ]
        # Coverage reports what uploaded; losing a requested word is a failed
        # edge discriminator, never relabeled exhaustive raw-copy acceptance.
        assert all(set(edge_words) <= words for words in edge_observed), edge_record
        reset(cases[0][1])
        save()

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
            "state, operands and activity predate capture and are never acknowledged. Any staging temporaries "
            "die inside their call. No broad allocation scope or program-cache exemption is used."
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
                    sources_before = {key: padded_qk_bits(inputs[key]) for key in ("q", "k")}
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
                    for key, expected in sources_before.items():
                        exact(
                            f"{fixture}/{mode}/{step}/{key}/preserved_with_padding",
                            padded_qk_bits(inputs[key]),
                            expected,
                        )
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
                print(f"NATIVE_QK_READER_PASS {fixture} {mode}", flush=True)
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
        raise RuntimeError(f"Native Q/K reader probe cleanup failed: {result['cleanup_errors']}")


if __name__ == "__main__":
    main()
