# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Readout-only precision hypothesis: custom, original prepared, corrected prepared.

Timing covers packed raw QKV -> preparation -> recurrence -> compact FP32 output,
with persistent state updates included. It does not claim whole-decoder timing.
"""

import argparse
import ast
import sys
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
    parser.add_argument("--batches", nargs="+", type=int, default=[1])
    parser.add_argument("--all-batches", action="store_true")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--bank-size", type=int, default=3)
    parser.add_argument(
        "--modes", nargs="+", choices=("zero_state", "normal", "zero_qkv", "scale"), default=["zero_state"]
    )
    parser.add_argument("--g-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument(
        "--custom-variant", choices=("baseline", "cumulative_exp_scalar"), default="cumulative_exp_scalar"
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument(
        "--queued", type=int, default=50, help="Changed-input traced steps without per-step sync;0for initial smoke"
    )
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.all_batches:
        args.batches = list(range(1, 33))
    assert all(1 <= b <= 32 for b in args.batches)
    assert min(args.steps, args.bank_size) >= 2 and min(args.warmup, args.samples, args.replays) > 0
    assert args.queued == 0 or args.queued >= 50
    assert not args.output.exists(), "Preserve earlier evidence"
    assert args.output.resolve().is_relative_to((ROOT / "doc/optimized_multichip_decoder").resolve())
    from models.autoports.qwen_qwen3_6_27b.tt.prepared_native_precise_readout import PreparedNativePreciseReadout
    from models.autoports.qwen_qwen3_6_27b.tt.prepared_native_recurrent_candidate import (
        PreparedNativeRecurrentCandidate,
        memory_estimate,
        head_partition,
        cpu_geometry_check,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.cpu_only:
        args.output.write_text(json.dumps(cpu_source_check(), indent=2) + "\n")
        return

    import torch
    import ttnn
    from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import _local12_recurrent

    torch.set_num_threads(4)
    watcher = "TT_METAL_WATCHER" in os.environ
    profiler = os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0"
    assert not (watcher and profiler)
    names = (
        "custom_bf16_state",
        "custom_fp32_state",
        "prepared_native_fp32_state",
        "prepared_native_bf16_state",
        "precise_readout_fp32_state",
        "precise_readout_bf16_state",
    )
    required_indices = (0, 1, 4, 5)
    path_count = len(names)
    ref_index = (0, 1, 1, 0, 1, 0)
    native_dir = ROOT.parents[2] / "ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule"
    sources = [
        Path(__file__),
        ROOT / "tt/prepared_native_precise_readout.py",
        ROOT / "tests/probe_prepared_native_recurrent.py",
        ROOT / "tt/prepared_native_recurrent_candidate.py",
        ROOT / "tt/multichip_decoder.py",
        ROOT / "tt/optimized_decoder.py",
        ROOT / "tt/recurrent_variants.py",
        *sorted(native_dir.rglob("*.cpp")),
    ]
    report = {
        "status": "running",
        "mesh": [1, 4],
        "fabric": "FABRIC_1D",
        "watcher": watcher,
        "profiler": profiler,
        "allocation_tracking": os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0") == "1",
        "skip_program_cache": os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "1",
        "arguments": vars(args) | {"output": str(args.output)},
        "cases": [],
        "custom_control_variant": args.custom_variant,
        "source_sha256": {
            str(p.relative_to(ROOT.parents[2])): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "boundary": "raw packed post-conv local QKV plus g/beta through preparation, recurrence, compact output and persistent state update",
        "precision": "Only corrected prepared readout is SFPU/direct-FP32; native update remains FPU. Persistence dtype matched to custom/FP64 oracle.",
        "gate_policy": "Custom and corrected paths must pass .995 per rank/user/head output and state; corrected/original same-dtype states must be bitwise identical. Original prepared quality failures are recorded controls, not blockers to candidate scoring.",
        "limitations": [
            "Synthetic rank/user-distinct inputs in actual local4QK/12V layout; actual-weight decoder gates remain separate.",
            "All six paths are isolated; no production defaults changed; public native is omitted so its known quality failure cannot mask this hypothesis.",
            "Prepared path includes compact grouped prep, multihead native recurrence and direct compact output in two launches.",
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
            user_scores, head_scores, errors, failures = [], [], [], []
            assert len(actual) == len(expected) == 4, "Metric requires every TP4 rank"

            def pcc(x, y, location):
                if torch.equal(x, y):
                    return 1.0
                xc, yc = x - x.mean(), y - y.mean()
                denominator = xc.norm() * yc.norm()
                score = float((xc @ yc) / denominator) if denominator else float("nan")
                if not math.isfinite(score) or score < 0.995:
                    failures.append(
                        {
                            "location": list(location),
                            "pcc": score if math.isfinite(score) else None,
                            "minimum": 0.995,
                            "reason": "below_threshold" if math.isfinite(score) else "undefined_pcc",
                        }
                    )
                return score if math.isfinite(score) else None

            for rank, (a, b) in enumerate(zip(actual, expected)):
                assert a.numel() == b.numel()
                a, b = a.reshape(batch, 12, -1).double(), b.reshape(batch, 12, -1).double()
                assert torch.isfinite(a).all() and torch.isfinite(b).all(), rank
                for user in range(batch):
                    user_scores.append(pcc(a[user].flatten(), b[user].flatten(), (rank, user)))
                    for head in range(12):
                        head_scores.append(pcc(a[user, head], b[user, head], (rank, user, head)))
                    errors.append(float((a[user] - b[user]).abs().max() / b[user].abs().max().clamp_min(1e-12)))
            finite_scores = [value for value in user_scores + head_scores if value is not None]
            return {
                "pcc_min": min(finite_scores) if finite_scores else None,
                "pcc_by_rank_user": user_scores,
                "pcc_by_rank_user_head": head_scores,
                "maximum_absolute_relative_error": max(errors),
                "quality_pass": not failures,
                "failures": failures,
            }

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
            print(f"PREPARED_NATIVE_RECURRENT begin B{batch} {mode}", flush=True)
            entry = {
                "batch": batch,
                "mode": mode,
                "partition": head_partition(batch, grid.x * grid.y),
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
            if mode == "scale":
                initial = [(x.float() * 16).bfloat16() for x in initial]
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
            prepared_owners = [
                PreparedNativeRecurrentCandidate(initial_fp32),
                PreparedNativeRecurrentCandidate(initial_bf16),
                PreparedNativePreciseReadout(initial_fp32),
                PreparedNativePreciseReadout(initial_bf16),
            ]
            changed_initial = [
                (value.float() * 0.75 + (rank + 1) * 0.003).bfloat16() for rank, value in enumerate(initial)
            ]
            changed_bf16 = upload(changed_initial, ttnn.bfloat16)
            changed_fp32 = upload([value.float() for value in changed_initial], ttnn.float32)
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
                if mode == "scale":
                    factors = torch.where(torch.arange(2560) % 2 == 0, 0.03125, 8.0).reshape(1, 1, 2560)
                    packed = [(value.float() * factors).bfloat16() for value in packed]
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

            def state_read(index, candidate=None):
                if index < 2:
                    return read(states[index])
                return read((candidate or prepared_owners[index - 2]).state)

            def reset(changed=False):
                source_bf16, source_fp32 = (changed_bf16, changed_fp32) if changed else (initial_bf16, initial_fp32)
                ttnn.copy(source_bf16, states[0])
                ttnn.copy(source_fp32, states[1])
                for index, candidate in enumerate(prepared_owners):
                    ttnn.copy(source_fp32 if index % 2 == 0 else source_bf16, candidate.state)

            def set_inputs(step):
                for source, target in zip(bank[step % args.bank_size], inputs):
                    ttnn.copy(source, target)

            def invoke(index):
                if index >= 2:
                    return prepared_owners[index - 2].step(*inputs)
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
                actual_s = [state_read(index) for index in range(path_count)]
                record = {"step": step, "paths": {}, "failures": []}
                entry[label].append(record)
                save()

                def collect_metric(comparison, actual, expected, required=True):
                    # An accuracy assertion in one control must not hide the
                    # other five paths. Keep every gate and raise after saving
                    # all comparisons. Runtime/device errors are not swallowed.
                    try:
                        value = metric(actual, expected, batch)
                    except AssertionError as error:
                        value = {"quality_pass": False, "metric_assertion": str(error)}
                    if not value["quality_pass"]:
                        record["failures"].append({"comparison": comparison, "details": value, "required": required})
                    return value

                for index, name in enumerate(names):
                    ref_y, ref_s = refs[ref_index[index]]
                    record["paths"][name] = {
                        "output": collect_metric(
                            f"{name}/oracle/output", actual_y[index], ref_y, index in required_indices
                        ),
                        "state": collect_metric(
                            f"{name}/oracle/state", actual_s[index], ref_s, index in required_indices
                        ),
                    }
                record["prepared_vs_dtype_matched_custom"] = {
                    names[index]: {
                        "output": collect_metric(
                            f"{names[index]}/custom/output",
                            actual_y[index],
                            actual_y[ref_index[index]],
                            index in required_indices,
                        ),
                        "state": collect_metric(
                            f"{names[index]}/custom/state",
                            actual_s[index],
                            actual_s[ref_index[index]],
                            index in required_indices,
                        ),
                    }
                    for index in (2, 3, 4, 5)
                }
                record["same_native_update_state"] = {}
                for old, corrected in ((2, 4), (3, 5)):
                    equal = hashes(actual_s[old]) == hashes(actual_s[corrected])
                    record["same_native_update_state"][names[corrected]] = equal
                    if not equal:
                        record["failures"].append(
                            {
                                "comparison": f"{names[corrected]}/original_state_bits",
                                "required": True,
                                "details": "Readout-only state changed",
                            }
                        )
                record["quality_pass"] = not any(row["required"] for row in record["failures"])
                # Save every path before the case-level required gate. In particular,
                # a failing original prepared control cannot hide corrected metrics.
                save()
                return [(hashes(y), hashes(s)) for y, s in zip(actual_y, actual_s)]

            def check_prepared(step):
                records = []
                for candidate in prepared_owners:
                    for rank, (actual, host) in enumerate(
                        zip(read(candidate.prepared), host_bank[step % args.bank_size])
                    ):
                        packed, g, beta = host
                        actual = actual.reshape(batch, 2592).float()
                        assert torch.isfinite(actual).all()
                        expected_qk = []
                        for start, scale in ((0, 128**-0.5), (512, 1.0)):
                            raw = packed[:, :, start : start + 512].reshape(batch, 4, 128).double()
                            expected_qk.append(
                                (raw * torch.rsqrt((raw * raw).sum(-1, keepdim=True) + 1e-6) * scale).float()
                            )
                        expected_qk = torch.cat(expected_qk, 1).reshape(batch, 1024)
                        # Preparation is FP32 SFPU; bound its error separately
                        # from native TF32-class contractions and persistence.
                        assert torch.allclose(actual[:, :1024], expected_qk, rtol=0.001, atol=2e-6), (
                            rank,
                            "prepared QK",
                        )
                        assert torch.equal(actual[:, 1024:2560], packed.reshape(batch, 2560)[:, 1024:].float())
                        assert torch.allclose(
                            actual[:, 2560:2572], g.reshape(batch, 12).float().exp(), rtol=0.001, atol=2e-6
                        )
                        assert torch.equal(actual[:, 2572:2584], beta.reshape(batch, 12).float())
                        assert torch.count_nonzero(actual[:, 2584:]) == 0
                    records.append(hashes(read(candidate.prepared)))
                return records

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
                    outputs = [invoke(index) for index in range(path_count)]
                    ttnn.synchronize_device(mesh)
                    eager_hashes.append(check_step(step, outputs, refs, "eager"))
                    entry["eager"][-1]["prepared_record_hashes"] = check_prepared(step)
                    save()

                # Fresh owner and fresh input bindings exercise native cached address
                # refresh, while the previous owner remains live and unchanged.
                entry["rebound"] = {}
                for index in (2, 3, 4, 5):
                    original_hash = hashes(state_read(index))
                    rebound = (PreparedNativeRecurrentCandidate if index < 4 else PreparedNativePreciseReadout)(
                        initial_fp32 if index % 2 == 0 else initial_bf16
                    )
                    count_before = mesh.num_program_cache_entries()
                    rebound_output = rebound.step(*bank[0])
                    ttnn.synchronize_device(mesh)
                    count_after = mesh.num_program_cache_entries()
                    rebound_hash = (hashes(read(rebound_output)), hashes(state_read(index, rebound)))
                    original_addresses = prepared_owners[index - 2].addresses
                    rebound_addresses = rebound.addresses
                    row = {
                        "program_cache_before": count_before,
                        "program_cache_after": count_after,
                        "output_and_state_exact_to_first_eager": rebound_hash == eager_hashes[0][index],
                        "original_owner_unchanged": hashes(state_read(index)) == original_hash,
                        "distinct_state_scratch_output_addresses": all(
                            a != b for a, b in zip(original_addresses, rebound_addresses)
                        ),
                    }
                    entry["rebound"][names[index]] = row
                    save()
                    row["pass"] = count_after == count_before and all(
                        row[key]
                        for key in (
                            "output_and_state_exact_to_first_eager",
                            "original_owner_unchanged",
                            "distinct_state_scratch_output_addresses",
                        )
                    )
                    # Rebound owner is no longer needed; release retained buffers before capture.
                    for tensor in (rebound.state, rebound.prepared, rebound.output):
                        ttnn.deallocate(tensor)
                    del rebound, rebound_output
                reset(changed=True)
                set_inputs(1)
                changed_refs = [
                    reference_step(host_bank[1], [v.to(dtype) for v in changed_initial], dtype)
                    for dtype in (torch.bfloat16, torch.float32)
                ]
                changed_outputs = [invoke(index) for index in range(path_count)]
                ttnn.synchronize_device(mesh)
                entry["changed_state_eager"] = []
                changed_hashes = check_step(1, changed_outputs, changed_refs, "changed_state_eager")

                set_inputs(0)
                for index in range(path_count):
                    for _ in range(args.warmup):
                        invoke(index)
                ttnn.synchronize_device(mesh)
                entry["persistent_output_addresses_before_capture"] = [
                    int(t.buffer_address()) for t in (*controls, *(candidate.output for candidate in prepared_owners))
                ]
                # All six paths are warm and all persistent buffers exist.
                # Returned handles are preallocated owner/control outputs;
                # native internal result/layout temporaries die inside step().
                for index in range(path_count):
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
                    # require all six independent traces to become clean.
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
                    # All trace equality flags are retained and gated at case end.
                    assert check_prepared(step) == entry["eager"][step]["prepared_record_hashes"]

                reset(changed=True)
                set_inputs(1)
                for trace in trace_ids:
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                changed_actual = [
                    (hashes(read(output)), hashes(state_read(index))) for index, output in enumerate(trace_outputs)
                ]
                entry["trace_changed_state_and_input"] = [a == b for a, b in zip(changed_actual, changed_hashes)]
                # Check every candidate's equality after collecting remaining phases.
                save()

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

                if not (args.skip_timing or watcher or profiler or report["allocation_tracking"]):
                    reset()
                    set_inputs(0)
                    for trace in trace_ids:
                        for _ in range(args.warmup):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    samples = [[] for _ in names]
                    for sample in range(args.samples):
                        for index in (range(path_count) if sample % 2 == 0 else reversed(range(path_count))):
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
                    entry["timing_skipped"] = "requested or watcher/profiler instrumentation active"
                assert hashes(read(initial_fp32)) == initial_hash, "Candidate mutated reset source"
                quality_rows = [
                    row
                    for phase in ("eager", "changed_state_eager", "queued_changed_inputs", "timed_final")
                    for row in entry.get(phase, [])
                ]
                required_quality = all(row["quality_pass"] for row in quality_rows)
                required_trace = all(row["exact_vs_own_eager"][i] for row in entry["trace"] for i in required_indices)
                required_trace = required_trace and all(
                    entry["trace_changed_state_and_input"][i] for i in required_indices
                )
                required_rebind = all(entry["rebound"][names[i]]["pass"] for i in (4, 5))
                entry["control_failure_count"] = sum(
                    not failure["required"] for row in quality_rows for failure in row["failures"]
                )
                control_trace = all(row["exact_vs_own_eager"][i] for row in entry["trace"] for i in (2, 3))
                control_trace = control_trace and all(entry["trace_changed_state_and_input"][i] for i in (2, 3))
                entry["original_prepared_control_status"] = (
                    "pass"
                    if (
                        entry["control_failure_count"] == 0
                        and control_trace
                        and all(entry["rebound"][names[i]]["pass"] for i in (2, 3))
                    )
                    else "fail"
                )
                entry["pass"] = required_quality and required_trace and required_rebind
                save()
                assert entry["pass"], {
                    "complete_metrics": str(args.output),
                    "required_quality": required_quality,
                    "required_trace": required_trace,
                    "required_rebind": required_rebind,
                }
                print(
                    f"PRECISE_READOUT pass B{batch} {mode}; original control failures={entry['control_failure_count']}",
                    flush=True,
                )
            finally:
                for trace in trace_ids:
                    ttnn.release_trace(mesh, trace)

        for batch in args.batches:
            for mode in args.modes:
                run_case(batch, mode)
        report["candidate_status"] = "pass"
        report["original_prepared_control_status"] = (
            "pass" if all(c["original_prepared_control_status"] == "pass" for c in report["cases"]) else "fail"
        )
        report["status"] = "pass"
    except BaseException:
        report["candidate_status"] = "fail"
        report["status"] = "fail"
        report["exception"] = traceback.format_exc()
        raise
    finally:
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


def cpu_source_check():
    from models.autoports.qwen_qwen3_6_27b.tt import prepared_native_recurrent_candidate as base
    from models.autoports.qwen_qwen3_6_27b.tt import prepared_native_precise_readout as precise

    source = base.recurrence_source()
    reader, compute = precise.precise_readout_sources(base.REC_READER, source)
    assert reader.replace(precise.COLUMN_MAPPING, precise.ROW_MAPPING, 1) == base.REC_READER
    restored = compute.replace(precise.READOUT + "void kernel_main() {", "void kernel_main() {", 1)
    restored = restored.replace(precise.PRECISE_CALL, precise.READOUT_CALL, 1)
    restored = restored.replace(
        "compute_kernel_hw_startup(cb_S,cb_decay,cb_out);", "compute_kernel_hw_startup(cb_q,cb_v,cb_out);", 1
    )
    restored = restored.replace(
        "for(uint32_t head=0;head<count;head++) {\n        // Restore both ordinary FPU source formats before every head.\n        reconfig_data_format(cb_S,cb_decay);",
        "for(uint32_t head=0;head<count;head++) {",
        1,
    )
    restored = restored.replace("reconfig_data_format_srca(cb_S);", "reconfig_data_format_srca(cb_q);", 1)
    restored = restored.replace(
        "            reconfig_data_format_srca(cb_s2); copy_init(cb_s2);", "            copy_init(cb_s2);", 1
    )
    assert restored == source, "Unexpected update/state/math/source delta"
    assert source.split("void kernel_main()", 1)[0] in compute
    assert "mm(cb_q,cb_s2" not in compute
    assert "rank1_update<4,4>(cb_sd,cb_kcol,cb_u,cb_s2,!BF16);" in compute

    # Compare the two host step descriptors as syntax trees: only reader binding
    # and recurrence direct-FP32 CB indices may differ. No runtime AST rewrite.
    def method(path):
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        return ast.unparse(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step"))

    original_step = method(Path(base.__file__))
    candidate_step = method(Path(precise.__file__))
    candidate_step = candidate_step.replace("(self.reader,", "(REC_READER,").replace(
        "compute_config((0, 8))", "compute_config((8,) if self.bf16 else ())"
    )
    assert candidate_step == original_step, "Unexpected preparation/descriptor change"
    for i in range(128):
        tile, row = divmod(i, 32)
        offset = tile * 1024 + (row // 16) * 512 + (row % 16) * 16
        assert offset // 1024 == tile and (offset % 1024) % 16 == 0
        assert offset % 1024 // 256 in (0, 2)
    assert len({(i // 32) * 1024 + ((i % 32) // 16) * 512 + (i % 16) * 16 for i in range(128)}) == 128
    assert "torch" not in sys.modules and "ttnn" not in sys.modules
    paths = [Path(__file__), Path(precise.__file__), Path(base.__file__), base.NATIVE_COMPUTE]
    return {
        "status": "pass",
        "geometry": base.cpu_geometry_check(),
        "native_update_and_state_source_unchanged": True,
        "host_descriptor_changes_only": ["Q-column reader", "direct CB0+CB8 for both persistence dtypes"],
        "Q_column_unique_elements": 128,
        "readout_DST_tiles": 3,
        "CB_counts_unchanged": True,
        "generated_reader_sha256": hashlib.sha256(reader.encode()).hexdigest(),
        "generated_compute_sha256": hashlib.sha256(compute.encode()).hexdigest(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "limitations": [
            "No device/compiler/native arithmetic validation",
            "Prototype superclass reads native source at construction; static production form remains future work",
        ],
    }


if __name__ == "__main__":
    main()
