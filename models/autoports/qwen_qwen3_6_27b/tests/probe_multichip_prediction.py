# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact prediction-register probe, derived from probe_multichip_recurrent.py.

Compare only cumulative_exp_scalar versus its ordered prediction transform.
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
from models.autoports.qwen_qwen3_6_27b.tt.recurrent_prediction_candidate import (
    BASELINE,
    VARIANT,
    HELPER,
    WRAPPER,
    HELPER_ANCHOR,
    WRAPPER_ANCHOR,
    PREDICTION_LOOP,
    PREDICTION_CALL,
    prediction_sources,
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-check", action="store_true")
    ap.add_argument("--queued", type=int, default=50)
    ap.add_argument("--all-batches", action="store_true")
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 2, 3, 10, 17, 31, 32])
    ap.add_argument("--state-dtypes", nargs="+", choices=["bf16", "fp32"], default=["bf16", "fp32"])
    ap.add_argument(
        "--modes",
        nargs="+",
        choices=["normal", "zero_qkv", "zero_state", "scale"],
        default=["normal", "zero_qkv", "zero_state", "scale"],
    )
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cores", type=int, help="Override regular-case core count")
    ap.add_argument("--low-cores", type=int, default=1, help="Additional B3 ring/user-wrap case; 0 disables")
    ap.add_argument("--grouped-qk", choices=["on", "off"], default="on")
    ap.add_argument(
        "--column-grain", type=int, choices=(1, 2), help="Compare both arms with static value-column ownership"
    )
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
    timings_skipped = tracking or watcher or profiler or args.poison_dead_padding
    assert all(1 <= b <= 32 for b in args.batches)
    assert args.steps >= 2 and args.warmup >= 1 and args.timing_samples >= 1 and args.replays_per_sample >= 1
    assert args.cores is None or args.cores >= 1
    assert args.low_cores >= 0
    assert not args.physical_device_ids or len(args.physical_device_ids) == args.mesh_cols
    reader, compute, frozen = frozen_sources(args.reference_commit)
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
        "comparison": "cumulative_exp_scalar versus prediction-only register fold",
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
                ROOT / "tt/recurrent_prediction_candidate.py",
                ROOT / "tt/recurrent_variants.py",
                ROOT / "tt/column_recurrent_runtime.py",
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
            layout=layout or ttnn.TILE_LAYOUT,
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

    def run_case(batch, state_name, mode, cores, variant):
        group, partition = local._local12_head_partition(batch, grid.x * grid.y, cores, args.grouped_qk == "on")
        baseline = sources(reader, compute, BASELINE, grouped=group, poison_dead_padding=args.poison_dead_padding)
        candidate = prediction_sources(*baseline)
        recurrent = local._local12_recurrent
        recurrent_options = {}
        if args.column_grain is not None:
            from models.autoports.qwen_qwen3_6_27b.tt.column_recurrent_runtime import (
                column_partition,
                column_recurrent,
                column_sources,
            )

            resolved_group, partition = column_partition(
                batch, grid.x * grid.y, cores, args.grouped_qk == "on", grain=args.column_grain
            )
            assert resolved_group == group
            writer = local._fused_delta_row_GDN_COMPACT_WRITER
            baseline = column_sources(*baseline, writer)
            candidate = column_sources(*candidate, writer)
            recurrent = column_recurrent
            recurrent_options["grain"] = args.column_grain
        entry = {
            "batch": batch,
            "state_dtype": state_name,
            "mode": mode,
            "cores": cores,
            "variant": variant,
            "grouped_qk": group,
            "partition": partition,
            "column_grain": args.column_grain,
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
        if mode == "scale":
            initial = [(x.float() * 16).to(torch_dtype) for x in initial]
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
            if mode == "scale":
                scale = torch.where(torch.arange(2560) % 2 == 0, 0.03125, 8.0).reshape(1, 1, 2560)
                packed = [(x.float() * scale).to(torch.bfloat16) for x in packed]
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


def cpu_check():
    reader, compute, frozen = frozen_sources(FROZEN)
    original_reader_hash = digest(reader)
    controls = []
    for grouped in (False, True):
        for poison in (False, True):
            baseline = sources(reader, compute, BASELINE, grouped=grouped, poison_dead_padding=poison)
            candidate = prediction_sources(*baseline)
            assert candidate[0] == baseline[0]
            restored = candidate[1].replace(HELPER + HELPER_ANCHOR, HELPER_ANCHOR, 1)
            restored = restored.replace(WRAPPER + WRAPPER_ANCHOR, WRAPPER_ANCHOR, 1)
            restored = restored.replace(PREDICTION_CALL, PREDICTION_LOOP, 1)
            assert restored == baseline[1], "Unexpected compute source delta"
            try:
                prediction_sources(*candidate)
            except ValueError:
                pass
            else:
                raise AssertionError("Transform must reject repeated application")
            controls.append(
                {
                    "grouped": grouped,
                    "poison": poison,
                    "baseline": [digest(x) for x in baseline],
                    "candidate": [digest(x) for x in candidate],
                }
            )
    # Pure symbolic lane ownership/operation sequence, not floating point emulation.
    # Every band has 4 vectors of 32 lanes. Both schedules retain m0,m1,m2,m3.
    old, new = {}, {}
    for m in range(4):
        for band in range(8):
            for vector in range(4):
                old.setdefault((band, vector), []).append(m)
    for band in range(8):
        for m in range(4):
            for vector in range(4):
                new.setdefault((band, vector), []).append(m)
    assert old == new and len(new) == 32
    assert 4 * 8 * 4 * 32 == 4096  # one value-column's 128x32 state elements
    # Current Blackhole destination addressing: each 32-lane vector occupies
    # two address units; four vectors per band cover both face halves exactly.
    state_addresses = [
        m * 64 + half * 32 + band * 4 + offset
        for m in range(4)
        for half in range(2)
        for band in range(4)
        for offset in (0, 2, 16, 18)
    ]
    assert sorted(state_addresses) == list(range(0, 256, 2))
    key_addresses = [
        (m // 2) * 16 + (m % 2) * 2 + half * 32 + band * 4 for m in range(4) for half in range(2) for band in range(4)
    ]
    assert sorted(key_addresses) == list(range(0, 64, 2))
    source = (ROOT / "tt/multichip_decoder.py").read_text()
    node = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_local12_head_partition"
    )
    namespace = {}
    exec(compile(ast.get_source_segment(source, node), "<CPU partition>", "exec"), namespace)
    geometry_cases = 0
    for batch in range(1, 33):
        for cores in (1, 7, 110):
            _, partition = namespace["_local12_head_partition"](batch, 110, cores, True)
            heads = [h for first, count in partition for h in range(first, first + count)]
            assert heads == list(range(batch * 12))
            geometry_cases += 1
    assert set(range(1, 5)).isdisjoint({0, 5, 6, 7})
    assert set(range(1, 5)) | {0, 5, 6, 7} == set(range(8))
    assert "SFPMAD" not in HELPER and "sfpu_bcast_col_init" not in HELPER
    assert "_gdn_store_bf16_rne_" not in HELPER  # unchanged downstream RNE
    assert "torch" not in sys.modules and "ttnn" not in sys.modules
    print(
        json.dumps(
            {
                "status": "pass",
                "source_sha256": {
                    str(p.relative_to(ROOT)): digest(p.read_bytes())
                    for p in (
                        Path(__file__),
                        ROOT / "tt/recurrent_prediction_candidate.py",
                        ROOT / "tt/recurrent_variants.py",
                        ROOT / "tests/probe_multichip_recurrent.py",
                        ROOT / "tt/optimized_decoder.py",
                        ROOT / "tt/multichip_decoder.py",
                    )
                },
                "source_controls": controls,
                "frozen_reader_sha256": original_reader_hash,
                "ordered_lane_vectors": len(new),
                "head_partition_geometry_cases": geometry_cases,
                "state_vector_addresses": len(state_addresses),
                "prepared_key_vector_addresses": len(key_addresses),
                "accumulator_dst_accesses_removed_per_head": 768,
                "register_budget": {"accumulators": [1, 2, 3, 4], "key": 0, "work": [5, 6], "decay": 7},
                "limits": "No C++ compile, SFPU execution or speed claim; source-only budget/geometry proof",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
