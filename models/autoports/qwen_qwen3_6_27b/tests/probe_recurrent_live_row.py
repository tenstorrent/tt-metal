# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact live-band Q/K probe, derived from probe_multichip_prediction.py.

Compare cumulative_exp_scalar versus reduced pointwise Q/K lane work.
All retained allocations and program/copy warmups precede both captures.
CPU checks import neither torch nor TTNN. Parent owns all hardware runs.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback
import sys

from models.autoports.qwen_qwen3_6_27b.tt.recurrent_variants import sources
from models.autoports.qwen_qwen3_6_27b.tt.recurrent_live_row_candidate import (
    BASELINE,
    VARIANT,
    WRAPPERS,
    WRAPPER_ANCHOR,
    HELPER_START,
    HELPER_END,
    PREP_START,
    PREP_END,
    REPLACEMENTS,
    live_row_sources,
    prepare_qk,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = "8502f42bca20dcad1507fbc8cebf7032220b7d8e"


def constant(text, name):
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing source constant {name}")


def frozen_sources(commit):
    files = {}
    for name in ("multichip_decoder", "optimized_decoder"):
        path = f"models/autoports/qwen_qwen3_6_27b/tt/{name}.py"
        files[name] = subprocess.check_output(["git", "show", f"{commit}:{path}"], text=True)
    return (
        constant(files["multichip_decoder"], "_LOCAL12_READER"),
        constant(files["optimized_decoder"], "_fused_delta_row_GDN_COMPACT_COMPUTE"),
        files,
    )


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def prepared_segment_map():
    # Blackhole FP32 SFPLOAD/SFPSTORE offsets0/2 select EVEN/ODD logical
    # columns within a16-column face (native sfpu_reduce.h load_face_data).
    # The four preparation slots are left-even,left-odd,right-even,right-odd.
    return [(column // 16) * 2 + column % 2 for column in range(32)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-check", action="store_true")
    ap.add_argument("--queued", type=int, default=50)
    ap.add_argument("--all-batches", action="store_true")
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 3, 17, 32])
    ap.add_argument("--state-dtypes", nargs="+", choices=["bf16", "fp32"], default=["bf16", "fp32"])
    ap.add_argument(
        "--modes",
        nargs="+",
        choices=["normal", "zero_qkv", "zero_state", "small", "large", "poison"],
        default=["normal", "zero_qkv", "zero_state", "small", "large", "poison"],
    )
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cores", type=int, help="Override regular-case core count")
    ap.add_argument("--low-cores", type=int, default=1, help="Additional B3 ring/user-wrap case; 0 disables")
    ap.add_argument("--grouped-qk", choices=["on", "off"], default="on")
    ap.add_argument("--poison-dead-padding", action="store_true", help="Correctness only; disables timings")
    ap.add_argument("--mesh-cols", type=int, choices=[4], default=4)
    ap.add_argument("--physical-device-ids", nargs="+", type=int)
    ap.add_argument("--reference-commit", default=FROZEN)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--timing-samples", type=int, default=7)
    ap.add_argument("--replays-per-sample", type=int, default=10)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.cpu_check:
        cpu_check()
        return
    assert args.output is not None and not args.output.exists(), "Use a fresh output file"
    assert args.queued >= 50
    if args.all_batches:
        args.batches = list(range(1, 33))
    tracking = os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0") == "1"
    watcher = "TT_METAL_WATCHER" in os.environ
    profiler = os.getenv("TT_METAL_DEVICE_PROFILER", "0") not in ("", "0")
    assert os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "1", "No tracker exemptions"
    timings_skipped = tracking or watcher or profiler or args.poison_dead_padding or "poison" in args.modes
    assert all(1 <= b <= 32 for b in args.batches)
    assert args.steps >= 2 and args.warmup >= 1 and args.timing_samples >= 1 and args.replays_per_sample >= 1
    assert args.cores is None or args.cores >= 1
    assert args.low_cores >= 0
    assert not args.physical_device_ids or len(args.physical_device_ids) == args.mesh_cols
    reader, compute, frozen = frozen_sources(args.reference_commit)
    assert reader == constant((ROOT / "tt/multichip_decoder.py").read_text(), "_LOCAL12_READER")
    assert compute == constant((ROOT / "tt/optimized_decoder.py").read_text(), "_fused_delta_row_GDN_COMPACT_COMPUTE")
    # Importing TTNN is deliberately deferred to the actual device entrypoint.
    import torch
    import ttnn
    from models.autoports.qwen_qwen3_6_27b.tt import multichip_decoder as local

    current_optimized = (ROOT / "tt/optimized_decoder.py").read_text()
    for name in (
        "_fused_delta_row_GDN_COMPACT_WRITER",
        "_fused_delta_row_GDN_FUSE_ACCUMULATOR",
        "_fused_delta_row_GDN_HOIST_DELTA",
        "_fused_delta_row_GDN_HOIST_SCALAR_INIT",
        "_fused_delta_row_GDN_HOIST_COL_INIT",
        "_fused_delta_row_GDN_FUSE_DECAY_PREDICT",
    ):
        assert constant(current_optimized, name) == constant(frozen["optimized_decoder"], name), name
    torch.set_num_threads(8)
    report = {
        "status": "running",
        "comparison": "cumulative_exp_scalar versus live-band pointwise Q/K preparation",
        "timings_skipped": timings_skipped,
        "watcher": watcher,
        "profiler": profiler,
        "allocation_tracking": tracking,
        "trace_contract": "All retained allocations and both warmups precede captures; positive control allocation is rejected without replay then freed. No exemptions.",
        "arguments": vars(args) | {"output": str(args.output)},
        "frozen_reader_sha256": digest(reader),
        "frozen_compute_sha256": digest(compute),
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p.read_bytes())
            for p in (
                Path(__file__),
                ROOT / "tests/probe_multichip_recurrent.py",
                ROOT / "tt/recurrent_live_row_candidate.py",
                ROOT / "tt/recurrent_variants.py",
                ROOT / "tt/multichip_decoder.py",
                ROOT / "tt/optimized_decoder.py",
            )
        },
        "environment": {
            k: os.environ.get(k)
            for k in ("TT_METAL_WATCHER", "TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_DEVICE_PROFILER")
        },
        "interface": "raw packed QKV only",
        "timing_kind": "serialized synchronized trace wall time",
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def upload(values, dtype, layout=None):
        return ttnn.from_torch(
            torch.cat(values, dim=-1).contiguous(),
            device=mesh,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT if layout is None else layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )

    def read(tensor):
        return [ttnn.to_torch(t).contiguous() for t in ttnn.get_device_tensors(tensor)]

    def check(left, right, batch, label, record):
        checks = []
        for rank, (a, b) in enumerate(zip(left, right)):
            assert a.shape == b.shape and a.dtype == b.dtype
            integer = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
            a, b = a.reshape(batch, -1), b.reshape(batch, -1)
            for user in range(batch):
                checks.append(
                    {
                        "rank": rank,
                        "user": user,
                        "bit_mismatches": int((a[user].view(integer) != b[user].view(integer)).sum()),
                        "finite": bool(torch.isfinite(a[user]).all() and torch.isfinite(b[user]).all()),
                    }
                )
        record[label] = checks
        save()
        assert all(c["bit_mismatches"] == 0 and c["finite"] for c in checks), label

    def diagnose_prepare(baseline, candidate, batch, mode, cores, entry):
        tasks = batch * 4
        rngs = [torch.Generator().manual_seed(8851 + batch * 71 + r) for r in range(4)]
        host_banks = []
        for phase in range(2):
            tensors = []
            for which in range(2):
                rank_values = []
                for rank, generator in enumerate(rngs):
                    values = torch.zeros(tasks, 32, 128)
                    values[:, 0, :] = (
                        (torch.randn(tasks, 128, generator=generator) * 0.125 + rank * 0.01 + phase * 0.02)
                        .bfloat16()
                        .float()
                    )
                    if mode == "zero_qkv":
                        values[:, 0, :] = 0
                    if mode in ("small", "large"):
                        values[:, 0, :] = (values[:, 0, :] * 2.0 ** (-18 if mode == "small" else 12)).bfloat16().float()
                    if mode == "poison" or args.poison_dead_padding:
                        values[:, 1:, :] = float("nan")
                    else:
                        # Nonzero finite dead rows disambiguate row0 from upper-face shortcuts.
                        values[:, 1:, :] = 19 + rank + phase + which
                    rank_values.append(values)
                tensors.append(rank_values)
            host_banks.append(tensors)
        banks = [[upload(values, ttnn.float32) for values in pair] for pair in host_banks]
        active = [[ttnn.empty_like(t) for t in banks[0]] for _ in range(2)]
        outs = [
            [upload([torch.zeros(tasks, 32, 32) for _ in range(4)], ttnn.float32) for _ in range(2)] for _ in range(3)
        ]
        retained = [t for pair in banks + active + outs for t in pair]
        evidence = {
            "tasks": tasks,
            "pointwise_band_rows": 4,
            "rows": [],
            "trace_scope": "complete recurrence below exercises preparation on trace",
        }
        entry["prepared_qk"] = evidence
        expected_banks, rebound_banks = {}, {}
        oracle_failures = []
        save()
        try:
            for phase in range(2):
                for src, dst in zip(banks[phase], active[0]):
                    ttnn.copy(src, dst)
                for src, dst in zip(banks[1 - phase], active[1]):
                    ttnn.copy(src, dst)
                for index, source in enumerate((baseline, candidate)):
                    prepare_qk(*active[0], *outs[index], source[1], cores=cores)
                prepare_qk(*active[1], *outs[2], candidate[1], cores=cores)
                ttnn.synchronize_device(mesh)
                row = {"phase": phase}
                evidence["rows"].append(row)
                for which in range(2):
                    expected_banks[phase, which] = read(outs[0][which])
                    rebound_banks[1 - phase, which] = read(outs[2][which])
                    check(read(outs[1][which]), expected_banks[phase, which], batch, f"prepared_{which}_exact", row)
                    # Distinct simultaneous owner compared with independent FP64 normalization;
                    # Each physical slot repeats one32-element segment across
                    # eight EVEN or ODD logical columns of its face.
                    for owner, bank_id in ((0, phase), (1, phase), (2, 1 - phase)):
                        actual = read(outs[owner][which])
                        for rank, got in enumerate(actual):
                            v = host_banks[bank_id][which][rank][:, 0, :].double()
                            norm = v * torch.rsqrt(v.square().sum(-1, keepdim=True) + 1e-6)
                            if which == 0:
                                norm = norm * (128**-0.5)
                            expected = norm.reshape(tasks, 4, 32).transpose(1, 2)[..., prepared_segment_map()]
                            from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc

                            scores = []
                            for head in range(tasks):
                                a, b = got[head].double(), expected[head]
                                if bool((b == 0).all()):
                                    score = 1.0 if bool((a == 0).all()) else -1.0
                                else:
                                    score = pcc(b, a)
                                scores.append(score)
                            row[f"oracle_owner{owner}_qk{which}_rank{rank}"] = scores
                            if min(scores) < 0.995 or not bool(torch.isfinite(got).all()):
                                failure = {
                                    "phase": phase,
                                    "owner": owner,
                                    "qk": which,
                                    "rank": rank,
                                    "scores": scores,
                                    "actual_first_rows": str(got[0, :2, :].tolist()),
                                    "expected_first_rows": str(expected[0, :2, :].tolist()),
                                }
                                oracle_failures.append(failure)
                                evidence["oracle_failures"] = oracle_failures
                            save()
            evidence["rebound_exact"] = {}
            for phase in range(2):
                for which in range(2):
                    check(
                        rebound_banks[phase, which],
                        expected_banks[phase, which],
                        batch,
                        f"bank{phase}_qk{which}",
                        evidence["rebound_exact"],
                    )
            assert not oracle_failures, oracle_failures
            evidence["pass"] = True
            return evidence
        finally:
            for tensor in retained:
                ttnn.deallocate(tensor)

    def run_case(batch, state_name, mode, cores, variant):
        group, partition = local._local12_head_partition(batch, grid.x * grid.y, cores, args.grouped_qk == "on")
        baseline = sources(
            reader, compute, BASELINE, grouped=group, poison_dead_padding=args.poison_dead_padding or mode == "poison"
        )
        if args.poison_dead_padding or mode == "poison":
            # Both Q/K double-buffer slots are initialized once; live row0
            # stores overwrite only their consumed elements on every head.
            old = "    noc.write_zeros_l1_barrier();"
            assert baseline[0].count(old) == 1
            poison = old + """
    for(uint32_t id=0;id<2;id++) {
        CircularBuffer cb(id);
        volatile tt_l1_ptr uint32_t* ptr=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb.get_write_ptr());
        for(uint32_t i=0;i<8*1024;i++) ptr[i]=0x7fc00000u;
    }
"""
            baseline = (baseline[0].replace(old, poison, 1), baseline[1])
        candidate = live_row_sources(*baseline)
        recurrent = local._local12_recurrent
        recurrent_options = {}
        entry = {
            "batch": batch,
            "state_dtype": state_name,
            "mode": mode,
            "cores": cores,
            "variant": variant,
            "grouped_qk": group,
            "partition": partition,
            "candidate_sha256": [digest(s) for s in candidate],
            "baseline_sha256": [digest(s) for s in baseline],
            "eager": [],
            "trace": [],
        }
        report["cases"].append(entry)
        save()
        print(f"RECURRENT begin B{batch} {state_name} {mode} cores={cores} {variant}", flush=True)
        dtype = ttnn.bfloat16 if state_name == "bf16" else ttnn.float32
        torch_dtype = torch.bfloat16 if state_name == "bf16" else torch.float32
        generators = [torch.Generator().manual_seed(20261113 + batch * 97 + r) for r in range(args.mesh_cols)]
        users = torch.arange(batch).reshape(batch, 1, 1).float()
        channels = torch.arange(2560).reshape(1, 1, 2560).float() % 131 / 1024
        heads = torch.arange(12).reshape(1, 1, 12).float()
        initial = [
            (torch.randn(batch * 12, 128, 128, generator=g) * 0.015625 + r * 0.001).to(torch_dtype)
            for r, g in enumerate(generators)
        ]
        if mode == "zero_state":
            initial = [x.zero_() for x in initial]
        diagnose_prepare(baseline, candidate, batch, mode, cores, entry)
        save()
        initial_t = upload(initial, dtype)
        states = [upload(initial, dtype) for _ in range(2)]
        outputs = [upload([torch.zeros(1, batch * 12, 128) for _ in generators], ttnn.float32) for _ in range(2)]
        bank = []
        for step in range(args.steps):
            packed = [
                (
                    torch.randn(batch, 1, 2560, generator=g) * 0.125 + users * 0.003 + channels + r * 0.01 + step * 0.02
                ).to(torch.bfloat16)
                for r, g in enumerate(generators)
            ]
            if mode == "zero_qkv":
                packed = [x.zero_() for x in packed]
            if mode in ("small", "large"):
                scale = 2.0 ** (-18 if mode == "small" else 12)
                # Only Q/K normalization is challenged; V/state remain bounded.
                for x in packed:
                    x[..., :1024] = (x[..., :1024].float() * scale).bfloat16()
            gates = [-0.02 - users * 0.001 - heads * 0.002 - r * 0.003 - step * 0.004 for r in range(args.mesh_cols)]
            betas = [
                (0.125 + users * 0.005 + heads * 0.015 + r * 0.007 + step * 0.01).to(torch.bfloat16)
                for r in range(args.mesh_cols)
            ]
            bank.append(
                (
                    upload(packed, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                    upload(gates, ttnn.float32),
                    upload(betas, ttnn.bfloat16),
                )
            )
        inputs = [ttnn.empty_like(t) for t in bank[0]]
        assert tuple(inputs[0].padded_shape) == (batch, 1, 2560)
        assert tuple(inputs[1].padded_shape) == tuple(inputs[2].padded_shape) == (batch, 32, 32)
        rebound_inputs = [ttnn.empty_like(t) for t in inputs]
        rebound_states = [ttnn.empty_like(t) for t in states]
        rebound_outputs = [ttnn.empty_like(t) for t in outputs]
        all_allocations = [
            initial_t,
            *states,
            *outputs,
            *inputs,
            *rebound_inputs,
            *rebound_states,
            *rebound_outputs,
            *(t for row in bank for t in row),
        ]
        traces = []

        def set_input(step):
            for src, dst in zip(bank[step], inputs):
                ttnn.copy(src, dst)

        def reset():
            for state in states:
                ttnn.copy(initial_t, state)

        def invoke(index, rebound=False):
            qkv, g, beta = rebound_inputs if rebound else inputs
            state_set, output_set = (rebound_states, rebound_outputs) if rebound else (states, outputs)
            return recurrent(
                qkv,
                qkv,
                qkv,
                g,
                beta,
                state_set[index],
                packed_qkv=True,
                output=output_set[index],
                cores=cores,
                grouped_qk=args.grouped_qk == "on",
                source_overrides=baseline if index == 0 else candidate,
                **recurrent_options,
            )

        try:
            eager = []
            for step in range(args.steps):
                set_input(step)
                invoke(0)
                invoke(1)
                ttnn.synchronize_device(mesh)
                yr, yc, sr, sc = map(read, (*outputs, *states))
                evidence = {"step": step}
                entry["eager"].append(evidence)
                check(yc, yr, batch, "output", evidence)
                check(sc, sr, batch, "state", evidence)
                eager.append((yr, sr))
            # Rebind all input, state and output addresses under the warmed cache.
            for src, dst in zip(bank[-1], rebound_inputs):
                ttnn.copy(src, dst)
            for state in rebound_states:
                ttnn.copy(initial_t, state)
            for index in range(2):
                invoke(index, rebound=True)
            ttnn.synchronize_device(mesh)
            entry["rebound"] = {}
            check(read(rebound_outputs[1]), read(rebound_outputs[0]), batch, "output", entry["rebound"])
            check(read(rebound_states[1]), read(rebound_states[0]), batch, "state", entry["rebound"])
            reset()
            for step in range(args.queued):
                set_input(step % args.steps)
                invoke(0)
                invoke(1)
            ttnn.synchronize_device(mesh)
            entry["queued"] = {"calls": args.queued}
            check(read(outputs[1]), read(outputs[0]), batch, "output", entry["queued"])
            check(read(states[1]), read(states[0]), batch, "state", entry["queued"])
            # All tensor allocations and both warmups precede captures. No allowlist.
            reset()
            set_input(0)
            for index in range(2):
                for _ in range(args.warmup):
                    invoke(index)
                ttnn.synchronize_device(mesh)
            cache_count = mesh.num_program_cache_entries()
            for index in range(2):
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                invoke(index)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                traces.append(tid)
            assert mesh.num_program_cache_entries() == cache_count, "Program allocated during capture"
            if tracking:
                from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker
                from ttnn._ttnn.operations.trace import get_unsafe_tracked_ids

                for tid in traces:
                    TraceAllocationTracker.verify_before_replay(mesh, tid)
                positive = {"unsafe_buffers_before": [len(get_unsafe_tracked_ids(mesh, t)) for t in traces]}
                unsafe = ttnn.empty_like(outputs[0])
                unsafe_id = unsafe.buffer_unique_id()
                try:
                    assert unsafe_id in get_unsafe_tracked_ids(mesh, traces[1])
                    rejected = False
                    try:
                        TraceAllocationTracker.verify_before_replay(mesh, traces[1])
                    except RuntimeError as error:
                        assert "device buffer(s) still alive before trace replay" in str(error), str(error)
                        assert f"Buffer {unsafe_id}" in str(error), str(error)
                        rejected = True
                    assert rejected, "Tracker failed deliberate live-allocation control"
                    positive["rejected_before_replay"] = True
                finally:
                    ttnn.deallocate(unsafe)
                    del unsafe
                for tid in traces:
                    TraceAllocationTracker.verify_before_replay(mesh, tid)
                positive["unsafe_buffers_after_release"] = [len(get_unsafe_tracked_ids(mesh, t)) for t in traces]
                positive["acknowledgements_or_exemptions"] = False
                entry["allocation_positive_control"] = positive
            reset()
            for step in range(args.steps):
                set_input(step)
                for tid in traces:
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                evidence = {"step": step}
                entry["trace"].append(evidence)
                for index, name in enumerate(("baseline", "candidate")):
                    check(read(outputs[index]), eager[step][0], batch, name + "_output_vs_eager", evidence)
                    check(read(states[index]), eager[step][1], batch, name + "_state_vs_eager", evidence)
            # Continue from changed recurrent state, alternate input bank, no reset.
            for step in range(args.queued):
                set_input((step + 1) % args.steps)
                for tid in traces:
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            entry["changed_trace"] = {"calls": args.queued}
            check(read(outputs[1]), read(outputs[0]), batch, "output", entry["changed_trace"])
            check(read(states[1]), read(states[0]), batch, "state", entry["changed_trace"])
            if not timings_skipped:
                reset()
                set_input(0)
                times = [[], []]
                for tid in traces:
                    for _ in range(args.warmup):
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                # Alternate order to reduce ordering bias; identical replay counts.
                for sample in range(args.timing_samples):
                    for index in ((0, 1) if sample % 2 == 0 else (1, 0)):
                        ttnn.synchronize_device(mesh)
                        begin = time.perf_counter_ns()
                        for _ in range(args.replays_per_sample):
                            ttnn.execute_trace(mesh, traces[index], cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        times[index].append((time.perf_counter_ns() - begin) / 1000 / args.replays_per_sample)
                entry["trace_us"] = {
                    name: {"samples": values, "median": statistics.median(values)}
                    for name, values in zip(("baseline", "candidate"), times)
                }
                check(read(outputs[1]), read(outputs[0]), batch, "timed_output", entry)
                check(read(states[1]), read(states[0]), batch, "timed_state", entry)
            # Eager calls under the existing captured signatures must not add
            # live device allocations or change later trace/state ownership.
            set_input(args.steps - 1)
            invoke(0)
            invoke(1)
            if tracking:
                for tid in traces:
                    TraceAllocationTracker.verify_before_replay(mesh, tid)
            for tid in traces:
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            entry["after_eager_interleave"] = {}
            check(read(outputs[1]), read(outputs[0]), batch, "output", entry["after_eager_interleave"])
            check(read(states[1]), read(states[0]), batch, "state", entry["after_eager_interleave"])
            entry["pass"] = True
            save()
            print(f"RECURRENT pass B{batch} {state_name} {mode} cores={cores} {variant}", flush=True)
        finally:
            for tid in traces:
                ttnn.release_trace(mesh, tid)
            # Avoid retaining hundreds of device buffers during the default sweep.
            for tensor in all_allocations:
                ttnn.deallocate(tensor)

    mesh = None
    save()
    try:
        if args.mesh_cols == 4:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        kwargs = {"trace_region_size": 8 * 1024 * 1024}
        if args.physical_device_ids:
            kwargs["physical_device_ids"] = args.physical_device_ids
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, args.mesh_cols), **kwargs)
        grid = mesh.compute_with_storage_grid_size()
        report["worker_grid"] = [grid.x, grid.y]
        cases = [(batch, args.cores) for batch in args.batches]
        if args.low_cores and (3, args.low_cores) not in cases:
            cases.append((3, args.low_cores))
        for variant in (VARIANT,):
            for batch, cores in cases:
                for dtype in args.state_dtypes:
                    for mode in args.modes:
                        run_case(batch, dtype, mode, cores, variant)
        report["status"] = "pass"
    except BaseException:
        report["status"] = "fail"
        report["exception"] = traceback.format_exc()
        raise
    finally:
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            report["mesh_closed"] = True
            save()


def cpu_check():
    reader, compute, _ = frozen_sources(FROZEN)
    assert reader == constant((ROOT / "tt/multichip_decoder.py").read_text(), "_LOCAL12_READER")
    assert compute == constant((ROOT / "tt/optimized_decoder.py").read_text(), "_fused_delta_row_GDN_COMPACT_COMPUTE")
    rows = []
    for group in (False, True):
        for poison in (False, True):
            baseline = sources(reader, compute, BASELINE, grouped=group, poison_dead_padding=poison)
            candidate = live_row_sources(*baseline)
            assert candidate[0] == baseline[0]
            # Undo only inserted helper, wrappers and six pointwise call substitutions.
            start = baseline[1].index(HELPER_START)
            end = baseline[1].index(HELPER_END)
            helper = baseline[1][start:end].replace("_gdn_mul_scalar_", "_gdn_mul_scalar_live_band_", 1)
            helper = helper.replace("half<2", "half<1", 1).replace("band<NUM_ROW_BANDS_PER_FACE_HALF", "band<1", 1)
            restored = candidate[1].replace("\n\n" + helper + HELPER_END, HELPER_END, 1)
            restored = restored.replace(WRAPPERS + WRAPPER_ANCHOR, WRAPPER_ANCHOR, 1)
            for before, after in REPLACEMENTS:
                assert restored.count(after) == 1
                restored = restored.replace(after, before, 1)
            assert restored == baseline[1]
            try:
                live_row_sources(*candidate)
            except ValueError:
                pass
            else:
                raise AssertionError("Repeated transform must fail")
            rows.append(
                {
                    "grouped": group,
                    "poison": poison,
                    "baseline": [digest(x) for x in baseline],
                    "candidate": [digest(x) for x in candidate],
                }
            )
    repo = ROOT.parents[2]
    paths = [
        "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_sfpu_common.h",
        "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_bcast.h",
        "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h",
        "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binop_with_unary.h",
        "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sqrt.h",
    ]
    src = [(repo / p).read_text() for p in paths]
    assert "face < 2" in src[0] and "VectorMode::R" in src[0]
    assert "FACE1_BASE = 16" in src[1]
    reduce_path = repo / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h"
    reduce_source = reduce_path.read_text()
    assert "even columns (0) or odd columns (2)" in reduce_source
    mapping = prepared_segment_map()
    assert [mapping.count(segment) for segment in range(4)] == [8, 8, 8, 8]
    for column in range(32):
        segment = mapping[column]
        assert (segment // 2) * 16 + segment % 2 + 2 * ((column % 16) // 2) == column
    assert all("d < ITERATIONS" in x and "dst_reg++" in x for x in src[2:])
    # Four native vector slots cover rows0..3, all32 columns; retained row0
    # occupies lanes0..7 in each slot. A row result's other columns are dead.
    live = {(row, col) for row in range(4) for col in range(32)}
    assert {(0, col) for col in range(32)} <= live
    assert len(live) == 128
    assert 4 * 32 == 128 and 4 * 8 == 32
    partition_source = (ROOT / "tt/multichip_decoder.py").read_text()
    partition_node = next(
        n
        for n in ast.parse(partition_source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_local12_head_partition"
    )
    namespace = {}
    exec(compile(ast.get_source_segment(partition_source, partition_node), "<CPU partition>", "exec"), namespace)
    geometry_cases = 0
    for batch in range(1, 33):
        for cores in (1, 7, 110):
            for grouped in (False, True):
                _, partition = namespace["_local12_head_partition"](batch, 110, cores, grouped)
                assert [h for start, count in partition for h in range(start, start + count)] == list(range(batch * 12))
                geometry_cases += 1
    assert "torch" not in sys.modules and "ttnn" not in sys.modules
    print(
        json.dumps(
            {
                "status": "pass",
                "scope": "CPU source/lane proof, no SFPU compile or execution",
                "controls": rows,
                "pointwise_band_rows": 4,
                "logical_live_rows": 1,
                "head_partition_geometry_cases": geometry_cases,
                "prepared_column_segment_map": mapping,
                "reduce_source_sha256": digest(reduce_source),
                "native_primitives": {p: digest(t) for p, t in zip(paths, src)},
                "unchanged": "reader, reduction tree, epsilon/Q-scale bits, all packs/transposes, recurrence, persistence",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
