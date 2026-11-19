# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Parent-serialized TP4 native recurrence family against BF16/FP32 custom controls.

Timing covers packed raw QKV -> preparation -> recurrence -> compact FP32 output,
with persistent state updates included. It does not claim whole-decoder timing.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 3, 9, 10, 17, 32])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--bank-size", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=("normal", "zero_qkv", "zero_state"), default=["normal"])
    parser.add_argument("--g-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument(
        "--queued", type=int, default=50, help="Changed-input traced steps without per-step sync;0for initial smoke"
    )
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument(
        "--custom-variant", choices=("baseline", "cumulative_exp_scalar"), default="cumulative_exp_scalar"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert all(1 <= b <= 32 for b in args.batches)
    assert min(args.steps, args.bank_size) >= 2 and min(args.warmup, args.samples, args.replays) > 0
    assert args.queued == 0 or args.queued >= 50
    assert not args.output.exists(), "Preserve earlier evidence"
    assert args.output.resolve().is_relative_to((ROOT / "doc/optimized_multichip_decoder").resolve())
    from models.autoports.qwen_qwen3_6_27b.tt.native_recurrent_candidate import (
        NativeRecurrentCandidate,
        memory_estimate,
        wave_plan,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.cpu_only:
        rows = []
        for batch in range(1, 33):
            waves = wave_plan(batch)
            assert [u for start, end in waves for u in range(start, end)] == list(range(batch))
            assert all(0 < (end - start) * 12 <= 110 for start, end in waves)
            rows.append({"batch": batch, "waves": waves, "memory": memory_estimate(batch)})
        args.output.write_text(json.dumps({"status": "pass", "kind": "CPU plan only", "cases": rows}, indent=2) + "\n")
        return

    import torch
    import ttnn
    from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import _local12_recurrent

    torch.set_num_threads(4)
    watcher = "TT_METAL_WATCHER" in os.environ
    profiler = os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0"
    tracking = os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0") == "1"
    assert not (watcher and profiler)
    names = ("custom_bf16_state", "custom_fp32_state", "native_fp32_ring")
    native_dir = ROOT.parents[2] / "ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule"
    sources = [
        Path(__file__),
        ROOT / "tt/native_recurrent_candidate.py",
        ROOT / "tt/multichip_decoder.py",
        ROOT / "tt/optimized_decoder.py",
        *sorted(native_dir.rglob("*.cpp")),
    ]
    report = {
        "status": "running",
        "mesh": [1, 4],
        "fabric": "FABRIC_1D",
        "watcher": watcher,
        "profiler": profiler,
        "allocation_tracking": tracking,
        "skip_program_cache": os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "1",
        "arguments": vars(args) | {"output": str(args.output)},
        "cases": [],
        "source_sha256": {
            str(p.relative_to(ROOT.parents[2])): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "boundary": "raw packed post-conv local QKV plus g/beta through preparation, recurrence, compact output and persistent state update",
        "precision": "Native FPU uses TF32-class inputs with FP32 state; custom control uses SFPU FP32 live arithmetic",
        "limitations": [
            "Synthetic rank/user-distinct inputs in actual local4QK/12V layout; actual-weight decoder gates remain separate.",
            "Public-native wave owner is isolated; no production public-state plumbing or defaults changed.",
            "No native prepared-input descriptor optimization yet; public wrapper movements are included.",
            "Memory figures are source estimates; measured allocation peaks remain separate.",
        ],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    mesh = None
    save()
    try:
        if report["allocation_tracking"]:
            assert not report["skip_program_cache"], "Quality evidence must track program-cache allocations"
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=64 * 1024 * 1024)
        grid = mesh.compute_with_storage_grid_size()
        report["worker_grid"] = [grid.x, grid.y]

        def upload(values, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                torch.cat(values, -1).contiguous(),
                device=mesh,
                dtype=dtype,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            )

        def read(tensor):
            return [ttnn.to_torch(value).contiguous() for value in ttnn.get_device_tensors(tensor)]

        def hashes(values):
            return [
                hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for value in values
            ]

        def metric(actual, expected, batch):
            scores = []
            errors = []
            for rank, (a, b) in enumerate(zip(actual, expected)):
                assert a.numel() == b.numel()
                a, b = a.reshape(batch, -1).double(), b.reshape(batch, -1).double()
                assert torch.isfinite(a).all() and torch.isfinite(b).all(), rank
                for user in range(batch):
                    x, y = a[user], b[user]
                    if torch.equal(x, y):
                        score = 1.0
                    else:
                        xc, yc = x - x.mean(), y - y.mean()
                        denominator = xc.norm() * yc.norm()
                        # Equal-zero cases passed above; unequal constant outputs fail closed.
                        score = float((xc @ yc) / denominator) if denominator else float("nan")
                    assert math.isfinite(score), (rank, user, "undefined PCC")
                    scores.append(score)
                    errors.append(float((x - y).abs().max() / y.abs().max().clamp_min(1e-12)))
            result = {
                "pcc_min": min(scores),
                "pcc_by_rank_user": scores,
                "maximum_absolute_relative_error": max(errors),
            }
            assert result["pcc_min"] >= 0.995, result
            return result

        def reference_step(host, states, state_dtype):
            outputs, updated = [], []
            for (packed, g, beta), state in zip(host, states):
                batch = packed.shape[0]
                q = packed[:, :, :512].reshape(batch, 4, 128).double()
                k = packed[:, :, 512:1024].reshape(batch, 4, 128).double()
                v = packed[:, :, 1024:2560].reshape(batch, 12, 128).double()
                q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
                k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
                q = q.repeat_interleave(3, 1) * 128**-0.5
                k = k.repeat_interleave(3, 1)
                decayed = state.double() * g.double().reshape(batch, 12, 1, 1).exp()
                prediction = (decayed * k.unsqueeze(-1)).sum(-2)
                delta = (v - prediction) * beta.double().reshape(batch, 12, 1)
                live = decayed + k.unsqueeze(-1) * delta.unsqueeze(-2)
                outputs.append((live * q.unsqueeze(-1)).sum(-2).reshape(1, batch * 12, 128).float())
                # Reference arithmetic uses FP64, with explicit persistence rounding only.
                updated.append(live.to(state_dtype))
            return outputs, updated

        def run_case(batch, mode):
            print(f"NATIVE_RECURRENT begin B{batch} {mode}", flush=True)
            entry = {
                "batch": batch,
                "mode": mode,
                "waves": wave_plan(batch, grid.x * grid.y),
                "memory_estimate": memory_estimate(batch, grid.x * grid.y),
                "eager": [],
                "trace": [],
            }
            report["cases"].append(entry)
            save()
            generators = [torch.Generator().manual_seed(52813 + 97 * batch + rank) for rank in range(4)]
            initial = [
                (torch.randn(batch, 12, 128, 128, generator=gen) * 0.015625 + rank * 0.001).bfloat16()
                for rank, gen in enumerate(generators)
            ]
            if mode == "zero_state":
                initial = [value.zero_() for value in initial]
            initial_bf16 = upload(initial, ttnn.bfloat16)
            initial_fp32 = upload([value.float() for value in initial], ttnn.float32)
            initial_hash = hashes(read(initial_fp32))
            states = [ttnn.empty_like(initial_bf16), ttnn.empty_like(initial_fp32)]
            controls = [
                ttnn.empty(
                    (1, batch * 12, 128),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            ]
            owner = NativeRecurrentCandidate(initial_fp32)
            # Materialize wave reset sources before the first trace. Later
            # phase resets only copy into existing ring addresses; no slicing
            # or new persistent allocation occurs between capture and replay.
            native_reset_sources = [
                ttnn.reshape(initial_fp32[start:end], ((end - start) * 12, 128, 128)) for start, end in owner.waves
            ]
            bank = []
            host_bank = []
            users = torch.arange(batch).reshape(batch, 1, 1).float()
            heads = torch.arange(12).reshape(1, 1, 12).float()
            for step in range(args.bank_size):
                packed = [
                    (
                        torch.randn(batch, 1, 2560, generator=gen) * 0.125 + users * 0.002 + rank * 0.01 + step * 0.02
                    ).bfloat16()
                    for rank, gen in enumerate(generators)
                ]
                if mode == "zero_qkv":
                    packed = [value.zero_() for value in packed]
                gs = [
                    (-0.04 - users * 0.001 - heads * 0.002 - rank * 0.003 - step * 0.004).to(
                        getattr(torch, args.g_dtype)
                    )
                    for rank in range(4)
                ]
                betas = [
                    (0.125 + users * 0.003 + heads * 0.015 + rank * 0.007 + step * 0.01).bfloat16() for rank in range(4)
                ]
                host_bank.append(list(zip(packed, gs, betas)))
                bank.append(
                    (
                        upload(packed, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                        upload(gs, getattr(ttnn, args.g_dtype)),
                        upload(betas, ttnn.bfloat16),
                    )
                )
            inputs = [ttnn.empty_like(t) for t in bank[0]]
            trace_ids = []
            trace_outputs = []
            eager_hashes = []

            def state_read(index, candidate=owner):
                if index < 2:
                    return read(states[index])
                chunks = [read(tensor) for tensor in candidate.state_tensors]
                return [
                    torch.cat(
                        [
                            part[rank].reshape(end - start, 12, 128, 128)
                            for part, (start, end) in zip(chunks, candidate.waves)
                        ],
                        0,
                    )
                    for rank in range(4)
                ]

            def reset():
                ttnn.copy(initial_bf16, states[0])
                ttnn.copy(initial_fp32, states[1])
                for source, destination in zip(native_reset_sources, owner.state_tensors):
                    ttnn.copy(source, destination)

            def set_inputs(step):
                for source, target in zip(bank[step % args.bank_size], inputs):
                    ttnn.copy(source, target)

            def invoke(index):
                if index == 2:
                    return owner.step(*inputs)
                qkv, g, beta = inputs
                return _local12_recurrent(
                    qkv,
                    qkv,
                    qkv,
                    g,
                    beta,
                    ttnn.reshape(states[index], (batch * 12, 128, 128)),
                    packed_qkv=True,
                    output=controls[index],
                    variant=args.custom_variant,
                )

            def check_step(step, outputs, refs, label):
                actual_y = [read(value) for value in outputs]
                actual_s = [state_read(index) for index in range(3)]
                record = {"step": step, "paths": {}}
                entry[label].append(record)
                save()
                for index, name in enumerate(names):
                    ref_y, ref_s = refs[0 if index == 0 else 1]
                    record["paths"][name] = {
                        "output": metric(actual_y[index], ref_y, batch),
                        "state": metric(actual_s[index], ref_s, batch),
                    }
                record["native_vs_controls"] = {
                    names[index]: {
                        "output": metric(actual_y[2], actual_y[index], batch),
                        "state": metric(actual_s[2], actual_s[index], batch),
                    }
                    for index in range(2)
                }
                save()
                return [(hashes(y), hashes(s)) for y, s in zip(actual_y, actual_s)]

            try:
                reset()
                ref_states = [[value.clone() for value in initial], [value.float() for value in initial]]
                for step in range(args.steps):
                    set_inputs(step)
                    refs = [
                        reference_step(host_bank[step % args.bank_size], ref_states[index], dtype)
                        for index, dtype in enumerate((torch.bfloat16, torch.float32))
                    ]
                    ref_states = [value[1] for value in refs]
                    outputs = [invoke(index) for index in range(3)]
                    ttnn.synchronize_device(mesh)
                    eager_hashes.append(check_step(step, outputs, refs, "eager"))

                # Fresh owner and fresh input bindings exercise native cached address
                # refresh, while the previous owner remains live and unchanged.
                original_hash = hashes(state_read(2))
                rebound = NativeRecurrentCandidate(initial_fp32)
                count_before = mesh.num_program_cache_entries()
                rebound_output = rebound.step(*bank[0])
                ttnn.synchronize_device(mesh)
                count_after = mesh.num_program_cache_entries()
                rebound_hash = (hashes(read(rebound_output)), hashes(state_read(2, rebound)))
                entry["rebound"] = {
                    "program_cache_before": count_before,
                    "program_cache_after": count_after,
                    "output_and_state_exact_to_first_eager": rebound_hash == eager_hashes[0][2],
                    "original_owner_unchanged": hashes(state_read(2)) == original_hash,
                    "distinct_state_addresses": all(
                        a != b for a, b in zip(owner.state_addresses, rebound.state_addresses)
                    ),
                }
                save()
                assert count_after == count_before, entry["rebound"]
                assert all(
                    entry["rebound"][key]
                    for key in (
                        "output_and_state_exact_to_first_eager",
                        "original_owner_unchanged",
                        "distinct_state_addresses",
                    )
                )

                set_inputs(0)
                for index in range(3):
                    for _ in range(args.warmup):
                        invoke(index)
                ttnn.synchronize_device(mesh)
                entry["persistent_output_addresses_before_capture"] = [
                    int(t.buffer_address()) for t in (*controls, owner.output)
                ]
                # All three paths are warm and all persistent buffers exist.
                # Returned handles are preallocated owner/control outputs;
                # native internal result/layout temporaries die inside step().
                for index in range(3):
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    captured_output = invoke(index)
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    trace_ids.append(trace)
                    trace_outputs.append(captured_output)
                    assert (
                        captured_output.buffer_address() == entry["persistent_output_addresses_before_capture"][index]
                    )
                if report["allocation_tracking"]:
                    from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker
                    from ttnn._ttnn.operations.trace import get_unsafe_tracked_ids

                    for trace in trace_ids:
                        TraceAllocationTracker.verify_before_replay(mesh, trace)
                    entry["trace_allocation_gate"] = {
                        "unsafe_buffers_after_capture": [
                            len(get_unsafe_tracked_ids(mesh, trace)) for trace in trace_ids
                        ],
                        "persistent_outputs_preallocated": True,
                    }
                    # Positive control: allocate a live buffer after capture,
                    # require rejection WITHOUT replaying, then free it and
                    # require all three independent traces to become clean.
                    negative = ttnn.empty_like(inputs[0])
                    negative_id = negative.buffer_unique_id()
                    unsafe = get_unsafe_tracked_ids(mesh, trace_ids[0])
                    assert negative_id in unsafe, (negative_id, unsafe)
                    rejected = False
                    try:
                        TraceAllocationTracker.verify_before_replay(mesh, trace_ids[0])
                    except RuntimeError as error:
                        assert "device buffer(s) still alive before trace replay" in str(error), str(error)
                        assert f"Buffer {negative_id}" in str(error), str(error)
                        rejected = True
                    assert rejected, "Allocation tracker did not reject the deliberate live buffer"
                    ttnn.deallocate(negative)
                    del negative
                    for trace in trace_ids:
                        TraceAllocationTracker.verify_before_replay(mesh, trace)
                    entry["trace_allocation_gate"].update(
                        {
                            "positive_control_rejected_before_replay": True,
                            "unsafe_buffers_after_positive_control_release": [
                                len(get_unsafe_tracked_ids(mesh, trace)) for trace in trace_ids
                            ],
                            "acknowledgements_or_corruptible_scopes": False,
                        }
                    )
                    save()
                reset()
                for step in range(args.steps):
                    set_inputs(step)
                    for trace in trace_ids:
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    actual = [
                        (hashes(read(output)), hashes(state_read(index))) for index, output in enumerate(trace_outputs)
                    ]
                    entry["trace"].append(
                        {"step": step, "exact_vs_own_eager": [a == b for a, b in zip(actual, eager_hashes[step])]}
                    )
                    save()
                    assert all(a == b for a, b in zip(actual, eager_hashes[step])), entry["trace"][-1]

                if args.queued:
                    reset()
                    for step in range(args.queued):
                        set_inputs(step)
                        for trace in trace_ids:
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    ref_states = [[value.clone() for value in initial], [value.float() for value in initial]]
                    for step in range(args.queued):
                        refs = [
                            reference_step(host_bank[step % args.bank_size], ref_states[index], dtype)
                            for index, dtype in enumerate((torch.bfloat16, torch.float32))
                        ]
                        ref_states = [value[1] for value in refs]
                    entry["queued_changed_inputs"] = []
                    check_step(args.queued, trace_outputs, refs, "queued_changed_inputs")
                    entry["queued_launch_contract"] = {
                        "steps_per_path": args.queued,
                        "input_bank_size": args.bank_size,
                        "per_step_synchronization": False,
                        "final_synchronization_and_readback": True,
                    }

                if not (args.skip_timing or watcher or profiler or tracking):
                    reset()
                    set_inputs(0)
                    for trace in trace_ids:
                        for _ in range(args.warmup):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    samples = [[] for _ in names]
                    for sample in range(args.samples):
                        for index in (range(3) if sample % 2 == 0 else reversed(range(3))):
                            ttnn.synchronize_device(mesh)
                            begin = time.perf_counter_ns()
                            for _ in range(args.replays):
                                ttnn.execute_trace(mesh, trace_ids[index], cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh)
                            samples[index].append((time.perf_counter_ns() - begin) / 1000 / args.replays)
                    entry["complete_boundary_trace_us"] = {
                        name: {"samples": values, "median": statistics.median(values)}
                        for name, values in zip(names, samples)
                    }
                    timed_steps = args.warmup + args.samples * args.replays
                    ref_states = [[value.clone() for value in initial], [value.float() for value in initial]]
                    for _ in range(timed_steps):
                        refs = [
                            reference_step(host_bank[0], ref_states[index], dtype)
                            for index, dtype in enumerate((torch.bfloat16, torch.float32))
                        ]
                        ref_states = [value[1] for value in refs]
                    entry["timed_final"] = []
                    check_step(timed_steps, trace_outputs, refs, "timed_final")
                else:
                    entry["timing_skipped"] = "requested or watcher/profiler/allocation instrumentation active"
                assert hashes(read(initial_fp32)) == initial_hash, "Candidate mutated reset source"
                entry["pass"] = True
                save()
                print(f"NATIVE_RECURRENT pass B{batch} {mode}", flush=True)
            finally:
                for trace in trace_ids:
                    ttnn.release_trace(mesh, trace)

        for batch in args.batches:
            for mode in args.modes:
                run_case(batch, mode)
        report["status"] = "pass"
    except BaseException:
        report["status"] = "fail"
        report["exception"] = traceback.format_exc()
        raise
    finally:
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
