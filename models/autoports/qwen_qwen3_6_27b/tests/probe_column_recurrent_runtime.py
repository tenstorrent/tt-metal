# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compare current cumulative recurrence, generated columns and static runtime.

Parent owns devices. Default sweep covers B1..32, grains1/2, BF16/FP32 state,
changed-input eager/traced recurrence and a one-core B3 wrap case. This isolated
probe never changes decoder policy/functions. --cpu-check imports no TTNN/Torch.
"""

import argparse
import ast
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

from models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_recurrent import constant, digest
from models.autoports.qwen_qwen3_6_27b.tt import column_recurrent_runtime as runtime
from models.autoports.qwen_qwen3_6_27b.tt import recurrent_column_candidate as generated
from models.autoports.qwen_qwen3_6_27b.tt.recurrent_variants import sources

ROOT = Path(__file__).resolve().parents[1]
PATHS = ("cumulative_baseline", "generated_column", "static_explicit", "static_automatic")
VARIANT = "cumulative_exp_scalar"


def case_matrix(batches, grains, state_dtypes, modes, cores, low_cores):
    cases = [(batch, cores) for batch in batches]
    if low_cores and (3, low_cores) not in cases:
        cases.append((3, low_cores))
    return [
        (batch, grain, dtype, mode, cap)
        for batch, cap in cases
        for grain in grains
        for dtype in state_dtypes
        for mode in modes
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", nargs="+", type=int, default=list(range(1, 33)))
    ap.add_argument("--grains", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    ap.add_argument("--state-dtypes", nargs="+", choices=("bf16", "fp32"), default=["bf16", "fp32"])
    ap.add_argument("--modes", nargs="+", choices=("normal", "zero_qkv", "zero_state"), default=["normal"])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cores", type=int)
    ap.add_argument("--low-cores", type=int, default=1, help="Additional B3 wrap case; 0 disables")
    ap.add_argument("--grouped-qk", choices=("on", "off"), default="on")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--timing-samples", type=int, default=7)
    ap.add_argument("--replays-per-sample", type=int, default=50)
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--poison-dead-padding", action="store_true", help="Explicit source paths only; disables timing")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    assert all(1 <= b <= 32 for b in args.batches) and len(set(args.batches)) == len(args.batches)
    assert args.steps >= 2 and args.warmup >= 1 and args.timing_samples >= 1 and args.replays_per_sample >= 1
    assert args.cores is None or args.cores >= 1
    assert args.low_cores >= 0
    assert not args.output.exists()
    assert args.output.resolve().is_relative_to((ROOT / "doc/optimized_multichip_decoder").resolve())
    watcher = "TT_METAL_WATCHER" in os.environ
    profiler = os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0"
    tracking = os.getenv("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert not (watcher and profiler)
    timed = not (args.no_timing or args.poison_dead_padding or watcher or profiler or tracking)

    import torch
    import ttnn
    from models.autoports.qwen_qwen3_6_27b.tt import multichip_decoder as model

    torch.set_num_threads(4)
    model_source = (ROOT / "tt/multichip_decoder.py").read_text()
    reader = model._LOCAL12_READER
    compute = model._fused_delta_row_GDN_COMPACT_COMPUTE
    writer = model._fused_delta_row_GDN_COMPACT_WRITER
    # Probe-only legacy generation is intentional. The new runtime itself has
    # no AST/exec/source-read path. This binds fixed copies without model mutation.
    legacy = generated.bind_local(model, writer, model_source)
    assert reader == constant(model_source, "_LOCAL12_READER")
    paths = [
        Path(__file__),
        *[
            ROOT / "tt" / f
            for f in (
                "column_recurrent_runtime.py",
                "recurrent_column_candidate.py",
                "recurrent_variants.py",
                "multichip_decoder.py",
                "optimized_decoder.py",
            )
        ],
    ]
    report = dict(
        status="running",
        mesh=[1, 4],
        arguments=vars(args) | {"output": str(args.output)},
        paths=list(PATHS),
        variant=VARIANT,
        timed=timed,
        cases=[],
        source_sha256={str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in paths},
        generated_host_sha256=digest(generated.host_source(model_source)),
        base_sources_sha256=[digest(s) for s in (reader, compute, writer)],
        environment={
            k: os.getenv(k)
            for k in (
                "TT_METAL_WATCHER",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
            )
        },
        comparison="Current cumulative baseline, generated grain, static explicit and static automatic",
        gate="Exact finite per-rank/per-user output and complete recurrent state; no relaxed PCC",
        lifetime="All device tensors and all four kernels warmed before any capture; no tracker exemptions",
        timing_kind="Matched warmed whole-recurrence trace wall time in microseconds, excludes input/state copies",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    mesh = None
    save()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=16 * 1024 * 1024)
        grid = mesh.compute_with_storage_grid_size()
        report["worker_grid"] = [grid.x, grid.y]

        def upload(values, dtype, layout=None):
            return ttnn.from_torch(
                torch.cat(values, -1).contiguous(),
                device=mesh,
                dtype=dtype,
                layout=layout or ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            )

        def read(tensor):
            return [ttnn.to_torch(t).contiguous() for t in ttnn.get_device_tensors(tensor)]

        def check(left, right, batch, label, record):
            assert len(left) == len(right) == 4
            checks = []
            for rank, (a, b) in enumerate(zip(left, right)):
                assert a.shape == b.shape and a.dtype == b.dtype
                integer = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
                a, b = a.reshape(batch, -1), b.reshape(batch, -1)
                for user in range(batch):
                    checks.append(
                        dict(
                            rank=rank,
                            user=user,
                            bit_mismatches=int((a[user].view(integer) != b[user].view(integer)).sum()),
                            finite=bool(torch.isfinite(a[user]).all() and torch.isfinite(b[user]).all()),
                        )
                    )
            record[label] = checks
            save()
            assert all(c["bit_mismatches"] == 0 and c["finite"] for c in checks), label

        def check_padding(outputs, batch, record):
            expected = ((batch * 12 + 31) // 32) * 32
            checks = []
            for name, output in zip(PATHS, outputs):
                for rank, shard in enumerate(ttnn.get_device_tensors(output)):
                    physical = shard.cpu().to_torch_with_padded_shape()
                    assert tuple(physical.shape) == (1, expected, 128)
                    pad = physical[:, batch * 12 :]
                    checks.append(
                        dict(
                            path=name,
                            rank=rank,
                            padded_rows=expected - batch * 12,
                            finite=bool(torch.isfinite(pad).all()),
                            nonzero=int((pad != 0).sum()),
                        )
                    )
            record["output_padding"] = checks
            save()
            assert all(c["finite"] and c["nonzero"] == 0 for c in checks)

        matrix = case_matrix(args.batches, args.grains, args.state_dtypes, args.modes, args.cores, args.low_cores)
        report["planned_cases"] = len(matrix)
        for case_index, (batch, grain, state_name, mode, cores) in enumerate(matrix):
            grouped, plan = runtime.column_partition(
                batch, grid.x * grid.y, cores, args.grouped_qk == "on", grain=grain
            )
            baseline_group, baseline_plan = legacy._local12_head_partition(
                batch, grid.x * grid.y, cores, args.grouped_qk == "on"
            )
            assert grouped == baseline_group
            baseline = sources(reader, compute, VARIANT, grouped=grouped, poison_dead_padding=args.poison_dead_padding)
            selected = generated.column_sources(*baseline, writer)
            assert selected == runtime.column_sources(*baseline, writer)
            automatic = runtime.column_sources(*sources(reader, compute, VARIANT, grouped=grouped), writer)
            if not args.poison_dead_padding:
                assert selected == automatic
            entry = dict(
                batch=batch,
                grain=grain,
                state_dtype=state_name,
                mode=mode,
                cores=cores,
                grouped_qk=grouped,
                column_partition=plan,
                baseline_partition=baseline_plan,
                selected_sha256=[digest(s) for s in selected],
                automatic_sha256=[digest(s) for s in automatic],
                eager=[],
                trace=[],
            )
            report["cases"].append(entry)
            save()
            print(
                f"COLUMN_RUNTIME begin B{batch} grain={grain} state={state_name} mode={mode} cores={cores}", flush=True
            )
            dtype = ttnn.bfloat16 if state_name == "bf16" else ttnn.float32
            host_dtype = torch.bfloat16 if state_name == "bf16" else torch.float32
            generators = [torch.Generator().manual_seed(20261113 + batch * 97 + r) for r in range(4)]
            users = torch.arange(batch).reshape(batch, 1, 1).float()
            channels = torch.arange(2560).reshape(1, 1, 2560).float() % 131 / 1024
            heads = torch.arange(12).reshape(1, 1, 12).float()
            initial = [
                (torch.randn(batch * 12, 128, 128, generator=g) * 0.015625 + r * 0.001).to(host_dtype)
                for r, g in enumerate(generators)
            ]
            if mode == "zero_state":
                initial = [value.zero_() for value in initial]
            initial_t = upload(initial, dtype)
            states = [upload(initial, dtype) for _ in PATHS]
            outputs = [upload([torch.zeros(1, batch * 12, 128) for _ in generators], ttnn.float32) for _ in PATHS]
            bank = []
            for step in range(args.steps):
                packed = [
                    (
                        torch.randn(batch, 1, 2560, generator=g) * 0.125
                        + users * 0.003
                        + channels
                        + r * 0.01
                        + step * 0.02
                    ).bfloat16()
                    for r, g in enumerate(generators)
                ]
                if mode == "zero_qkv":
                    packed = [value.zero_() for value in packed]
                gates = [-0.02 - users * 0.001 - heads * 0.002 - r * 0.003 - step * 0.004 for r in range(4)]
                betas = [(0.125 + users * 0.005 + heads * 0.015 + r * 0.007 + step * 0.01).bfloat16() for r in range(4)]
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
            allocations = [initial_t, *states, *outputs, *inputs, *(t for row in bank for t in row)]
            assert len({t.buffer_address() for t in allocations}) == len(allocations)
            traces = []

            def set_input(step):
                for src, dst in zip(bank[step], inputs):
                    ttnn.copy(src, dst)

            def reset():
                for state in states:
                    ttnn.copy(initial_t, state)

            def invoke(index):
                qkv, g, beta = inputs
                kwargs = dict(
                    packed_qkv=True,
                    output=outputs[index],
                    cores=cores,
                    grouped_qk=args.grouped_qk == "on",
                    variant=VARIANT,
                )
                if index == 0:
                    return legacy._local12_recurrent(
                        qkv, qkv, qkv, g, beta, states[index], source_overrides=baseline, **kwargs
                    )
                if index == 1:
                    operation = legacy.column_functions["columns" if grain == 1 else "pairs"]
                    return operation(qkv, qkv, qkv, g, beta, states[index], source_overrides=selected, **kwargs)
                assert index in (2, 3)
                return runtime.column_recurrent(
                    qkv,
                    qkv,
                    qkv,
                    g,
                    beta,
                    states[index],
                    grain=grain,
                    source_overrides=selected if index == 2 else None,
                    **kwargs,
                )

            try:
                eager = []
                for step in range(args.steps):
                    set_input(step)
                    for index in range(len(PATHS)):
                        invoke(index)
                    ttnn.synchronize_device(mesh)
                    values = [read(t) for t in outputs]
                    next_states = [read(t) for t in states]
                    evidence = dict(step=step)
                    entry["eager"].append(evidence)
                    for index in range(1, len(PATHS)):
                        check(values[index], values[0], batch, PATHS[index] + "_output_vs_baseline", evidence)
                        check(next_states[index], next_states[0], batch, PATHS[index] + "_state_vs_baseline", evidence)
                    for index in (2, 3):
                        check(values[index], values[1], batch, PATHS[index] + "_output_vs_generated", evidence)
                        check(next_states[index], next_states[1], batch, PATHS[index] + "_state_vs_generated", evidence)
                    check_padding(outputs, batch, evidence)
                    eager.append((values[0], next_states[0]))
                    for live, original in zip(inputs, bank[step]):
                        check(
                            read(live), read(original), batch, "input_immutable_" + str(live.buffer_address()), evidence
                        )
                # Warm every fixed path before any trace is active. All device
                # buffers were allocated above; source assembly only uses CPU strings.
                reset()
                set_input(0)
                for index in range(len(PATHS)):
                    for _ in range(args.warmup):
                        invoke(index)
                    ttnn.synchronize_device(mesh)
                for index in range(len(PATHS)):
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    invoke(index)
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    traces.append(trace)
                if tracking and case_index == 0:
                    unsafe = ttnn.empty((1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
                    positive = False
                    try:
                        ttnn.execute_trace(mesh, traces[3], cq_id=0, blocking=True)
                    except RuntimeError as exc:
                        if not any(word in str(exc).lower() for word in ("unsafe", "allocation")):
                            raise
                        positive = True
                    finally:
                        ttnn.deallocate(unsafe)
                    entry["allocation_positive_control"] = positive
                    save()
                    assert positive
                # Two full reset/replay rounds reverse dispatch order; each
                # changed-input step must exactly match the eager state trajectory.
                for round_index, order in enumerate((range(4), range(3, -1, -1))):
                    reset()
                    for step in range(args.steps):
                        set_input(step)
                        for index in order:
                            ttnn.execute_trace(mesh, traces[index], cq_id=0, blocking=True)
                        evidence = dict(round=round_index, step=step)
                        entry["trace"].append(evidence)
                        for index, name in enumerate(PATHS):
                            check(read(outputs[index]), eager[step][0], batch, name + "_output_vs_eager", evidence)
                            check(read(states[index]), eager[step][1], batch, name + "_state_vs_eager", evidence)
                # Matched queued recurrence progression, independent of timing.
                reset()
                set_input(0)
                for index in range(4):
                    for _ in range(50):
                        ttnn.execute_trace(mesh, traces[index], cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                entry["queued_50"] = {}
                for index in range(1, 4):
                    check(read(outputs[index]), read(outputs[0]), batch, PATHS[index] + "_output", entry["queued_50"])
                    check(read(states[index]), read(states[0]), batch, PATHS[index] + "_state", entry["queued_50"])
                if timed:
                    reset()
                    set_input(0)
                    times = [[] for _ in PATHS]
                    for trace in traces:
                        for _ in range(args.warmup):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    for sample in range(args.timing_samples):
                        for index in (range(4) if sample % 2 == 0 else range(3, -1, -1)):
                            ttnn.synchronize_device(mesh)
                            start = time.perf_counter_ns()
                            for _ in range(args.replays_per_sample):
                                ttnn.execute_trace(mesh, traces[index], cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh)
                            times[index].append((time.perf_counter_ns() - start) / 1000 / args.replays_per_sample)
                    entry["trace_us"] = {
                        name: dict(samples=values, median=statistics.median(values))
                        for name, values in zip(PATHS, times)
                    }
                    for index in range(1, 4):
                        check(read(outputs[index]), read(outputs[0]), batch, PATHS[index] + "_timed_output", entry)
                        check(read(states[index]), read(states[0]), batch, PATHS[index] + "_timed_state", entry)
                if profiler:
                    ttnn.ReadDeviceProfiler(mesh)
                entry["pass"] = True
                save()
                print(
                    f"COLUMN_RUNTIME pass B{batch} grain={grain} state={state_name} mode={mode} cores={cores}",
                    flush=True,
                )
            finally:
                for trace in traces:
                    ttnn.release_trace(mesh, trace)
                for tensor in allocations:
                    ttnn.deallocate(tensor)
        report["status"] = "pass"
        report["pass"] = True
    except BaseException:
        report["status"] = "fail"
        report["exception"] = traceback.format_exc()
        raise
    finally:
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


def cpu_check():
    model_source = (ROOT / "tt/multichip_decoder.py").read_text()
    optimized_source = (ROOT / "tt/optimized_decoder.py").read_text()
    reader = constant(model_source, "_LOCAL12_READER")
    compute = constant(optimized_source, "_fused_delta_row_GDN_COMPACT_COMPUTE")
    writer = constant(optimized_source, "_fused_delta_row_GDN_COMPACT_WRITER")
    matrix = case_matrix(list(range(1, 33)), [1, 2], ["bf16", "fp32"], ["normal"], None, 1)
    assert len(matrix) == 132 and len(set(matrix)) == len(matrix)
    cases = 0
    for batch, grain, state_dtype, mode, cores in matrix:
        grouped, plan = runtime.column_partition(batch, 110, cores, True, grain=grain)
        assert (grouped, plan) == generated.column_partition(batch, 110, cores, True, grain=grain)
        for poison in (False, True):
            baseline = sources(reader, compute, VARIANT, grouped=grouped, poison_dead_padding=poison)
            assert runtime.column_sources(*baseline, writer) == generated.column_sources(*baseline, writer)
        assert state_dtype in ("bf16", "fp32") and mode == "normal"
        cases += 1
    assert {b for b, *_ in matrix} == set(range(1, 33))
    assert set(PATHS) == {"cumulative_baseline", "generated_column", "static_explicit", "static_automatic"}
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    ast.parse(Path(__file__).read_text())
    print(
        json.dumps(
            dict(
                status="pass",
                cases=cases,
                paths=list(PATHS),
                scope="CPU case/source equality only; device harness unexecuted",
                source_sha256=digest(Path(__file__).read_bytes()),
            )
        )
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--cpu-check"]:
        cpu_check()
    else:
        main()
