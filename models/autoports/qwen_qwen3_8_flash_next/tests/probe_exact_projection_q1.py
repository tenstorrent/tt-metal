# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, test-local exact Q1 fusion experiment. Run under external timeout.

Start: timeout 300s python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_exact_projection_q1 \
    --roles smoke --rows 8 --num-cores 1 --skip-trace --output /fresh/smoke.json
Then real shapes: --roles gr_down,gr_up,gdn_out,router --snapshot SNAPSHOT.
Optional --inputs PT provides {role: tensor[N,1,1,1,K]} captured inputs;
without it, inputs are deterministic controls and explicitly labeled synthetic.
No production module is patched. No devices are opened on import.
"""

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import torch

MODEL = Path(__file__).resolve().parents[1]
ROLE = {
    "gr_down": ("attn_hyper_connection.input_mix_weight_down.weight", 96, 10240, "float32"),
    "gr_up": ("attn_hyper_connection.input_mix_weight_up.weight", 2560, 320, "float32"),
    "gdn_out": ("linear_attn.out_proj.weight", 640, 6144, "bfloat16"),
    "router": ("mlp.gate.weight", 128, 2560, "bfloat16"),
}


def digest_tensor(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--roles", default="smoke")
    p.add_argument("--rows", default="8,4", help="Output rows per worker:2,4,8,16; R2 uses the R4 SFPU band")
    p.add_argument("--num-cores", type=int)
    p.add_argument(
        "--zero-once", action="store_true", help="Opt into once-per-invocation initialization of both input CB pages"
    )
    p.add_argument(
        "--precast-input",
        action="store_true",
        help="Candidate-only exact BF16-to-FP32 input widening when weights are FP32; included in trace/timing",
    )
    p.add_argument("--snapshot", type=Path)
    p.add_argument("--inputs", type=Path)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--skip-trace", action="store_true")
    p.add_argument("--trace-region-size", type=int, default=16 * 1024 * 1024)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--samples", type=int, default=3)
    a = p.parse_args()
    roles, row_choices = a.roles.split(","), list(map(int, a.rows.split(",")))
    if a.output.exists():
        raise ValueError("Refusing to overwrite an existing result")
    if any(role not in ("smoke", *ROLE) for role in roles) or any(r not in (2, 4, 8, 16) for r in row_choices):
        raise ValueError("Invalid role or row group")
    if any(role != "smoke" for role in roles) and a.snapshot is None:
        raise ValueError("Real roles require --snapshot")
    if a.iterations < 1 or a.samples < 1:
        raise ValueError("Timing repetitions must be positive")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    files = [
        Path(__file__).resolve(),
        MODEL / "tt/exact_projection_q1.py",
        MODEL / "tt/fused_decoder.py",
        *sorted((MODEL / "tt/kernels").glob("exact_projection_q1_*.cpp")),
    ]
    report = dict(
        passed=False,
        scope="component diagnostic; no full-model speed claim",
        command=sys.argv,
        phases=[],
        cases=[],
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        sources={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        environment={k: v for k, v in os.environ.items() if k.startswith(("TT_", "TTNN_"))},
        mesh=[1, 4],
        zero_once=a.zero_once,
        precast_input_requested=a.precast_input,
        reader_initialization="both physical CB pages once per invocation" if a.zero_once else "every consumed CB page",
        exactness_scope="finite model operands; nonfinite/NaN payload identity is not established",
        layer=a.layer,
        input_source=str(a.inputs) if a.inputs else "synthetic deterministic",
        padding_check="Writer source zeroes all cells; logical readback does not expose padded rows",
        physical_row_layout_control="smoke uses distinct row/rank ramp and changed inputs",
        runtime_guard="forbid torch ops, TTNN host conversions and fallback during selected/native calls",
    )

    def save():
        a.output.write_text(json.dumps(report, indent=2) + "\n")

    def phase(name, **details):
        report["phase"] = name
        event = dict(phase=name, time=time.time(), **details)
        report["phases"].append(event)
        save()
        print(json.dumps(event), flush=True)

    torch.set_num_threads(8)
    torch.manual_seed(3871)
    fixtures = torch.load(a.inputs, map_location="cpu", weights_only=True, mmap=True) if a.inputs else {}
    if a.inputs:
        report["input_fixture"] = dict(
            path=str(a.inputs.resolve()), bytes=a.inputs.stat().st_size, modified_ns=a.inputs.stat().st_mtime_ns
        )
    phase("HOST_SETUP_BEGIN")
    jobs = []
    for role in roles:
        if role == "smoke":
            for o, k in ((32, 32), (96, 320)):
                for wd in (torch.bfloat16, torch.float32):
                    for xd in (torch.bfloat16, torch.float32):
                        # Exactly representable row/rank ramp catches packing/order errors;
                        # small random FP32 terms also expose unintended input truncation.
                        w = torch.randn(4, o, k) * 0.007
                        w[:, :, 0] = torch.arange(4 * o).reshape(4, o).float() / 64
                        xs = torch.randn(3, 1, 1, 1, k) * 0.03
                        xs[0].zero_()
                        xs[0, ..., 0] = 1
                        xs[2] = -xs[1]
                        jobs.append(
                            (
                                f"smoke_{o}x{k}_{str(wd).split('.')[-1]}_{str(xd).split('.')[-1]}",
                                w.to(wd).clone(),
                                xs.to(xd).clone(),
                                None,
                            )
                        )
        else:
            from safetensors import safe_open

            suffix, o, k, wd = ROLE[role]
            key = f"model.language_model.layers.{a.layer}.{suffix}"
            index_path = a.snapshot / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())["weight_map"]
            with safe_open(a.snapshot / index[key], framework="pt") as f:
                host = f.get_tensor(key).to(getattr(torch, wd)).clone()
            parts = list(host.chunk(4, dim=0))
            w = torch.zeros((4, o, k), dtype=getattr(torch, wd))
            for rank, part in enumerate(parts):
                assert part.shape[-1] == k and part.shape[0] <= o, (key, part.shape)
                w[rank, : part.shape[0]] = part
            xs = fixtures.get(role)
            if xs is None:
                if a.inputs:
                    raise ValueError(f"Captured inputs missing role {role}")
                xs = (torch.randn(3, 1, 1, 1, k) * 0.03).bfloat16()
            xs = xs.clone()
            if xs.ndim != 5 or tuple(xs.shape[1:]) != (1, 1, 1, k) or len(xs) < 2:
                raise ValueError(f"{role} inputs must be[N>=2,1,1,1,{k}]")
            if xs.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError("Captured input dtype must be BF16/FP32, unchanged")
            jobs.append(
                (
                    role,
                    w,
                    xs,
                    dict(
                        key=key,
                        checkpoint=str(a.snapshot / index[key]),
                        index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
                    ),
                )
            )
    phase("HOST_SETUP_DONE", jobs=len(jobs))
    from ttnn.tools import trace_allocation_tracker

    import ttnn as t
    from models.autoports.qwen_qwen3_8_flash_next.tt.exact_projection_q1 import exact_projection_q1

    report["trace_allocation_tracking"] = bool(trace_allocation_tracker.TRACE_ALLOC_TRACKING)
    report[
        "capture_output_policy"
    ] = "Only returned raw/BF16 outputs acknowledged as intentionally replay-overwritten; no intermediate acknowledgement"

    class NoTorch(torch.overrides.TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            raise RuntimeError(f"Runtime torch operation forbidden: {func}")

    @contextlib.contextmanager
    def guard():
        def forbidden(*args, **kwargs):
            raise RuntimeError("Runtime host tensor conversion forbidden")

        with (
            mock.patch.object(t, "from_torch", forbidden),
            mock.patch.object(t, "to_torch", forbidden),
            t.manage_config("throw_exception_on_fallback", True),
            NoTorch(),
        ):
            yield

    report["runtime_guard_positive_controls"] = {}
    for name, operation in (
        ("torch", lambda: torch.zeros(1)),
        ("from_torch", lambda: t.from_torch(None)),
        ("to_torch", lambda: t.to_torch(None)),
    ):
        try:
            with guard():
                assert t.CONFIG.throw_exception_on_fallback is True
                operation()
        except RuntimeError as error:
            assert "Runtime torch operation forbidden" in str(
                error
            ) or "Runtime host tensor conversion forbidden" in str(error), str(error)
            report["runtime_guard_positive_controls"][name] = dict(rejected=True, error=str(error))
        else:
            raise AssertionError(f"Runtime guard did not reject {name}")
    phase("RUNTIME_GUARD_CONTROLS_PASSED")
    mesh, traces = None, []
    try:
        phase("MESH_OPEN_BEGIN")
        t.set_fabric_config(t.FabricConfig.FABRIC_1D_RING)
        mesh = t.open_mesh_device(
            t.MeshShape(1, 4), trace_region_size=0 if a.skip_trace else a.trace_region_size, l1_small_size=32768
        )
        t.synchronize_device(mesh)
        phase("MESH_OPEN_SYNCED")
        compute = t.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=t.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
            math_approx_mode=False,
        )

        def read(x):
            return [t.to_torch(v).clone() for v in t.get_device_tensors(x)]

        def same(expected, actual):
            checks = []
            for rank, (e, v) in enumerate(zip(expected, actual, strict=True)):
                bits = torch.int32 if e.dtype == torch.float32 else torch.int16
                changed = int(torch.count_nonzero(e.contiguous().view(bits) != v.contiguous().view(bits)))
                item = dict(
                    rank=rank,
                    changed_bits=changed,
                    max_abs=float((e.float() - v.float()).abs().max()),
                    finite=bool(torch.isfinite(v).all()),
                )
                if changed:
                    eb = e.contiguous().view(bits).flatten()
                    vb = v.contiguous().view(bits).flatten()
                    width = 32 if bits == torch.int32 else 16
                    mask = (1 << width) - 1
                    item.update(
                        first_mismatch_index=int(torch.nonzero(eb != vb)[0, 0]),
                        reference_first32=e.float().flatten()[:32].tolist(),
                        actual_first32=v.float().flatten()[:32].tolist(),
                        reference_bits_first32=[f"0x{int(b) & mask:0{width // 4}x}" for b in eb[:32]],
                        actual_bits_first32=[f"0x{int(b) & mask:0{width // 4}x}" for b in vb[:32]],
                    )
                checks.append(item)
            return dict(passed=all(c["changed_bits"] == 0 and c["finite"] for c in checks), ranks=checks)

        for label, host_weight, host_inputs, provenance in jobs:
            o, k = host_weight.shape[-2:]
            phase("UPLOAD_BEGIN", case=label)
            weight = t.from_torch(
                host_weight.reshape(1, 1, 4 * o, k).clone(),
                dtype=getattr(t, str(host_weight.dtype).split(".")[-1]),
                layout=t.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=t.ShardTensorToMesh(mesh, dim=2),
                memory_config=t.DRAM_MEMORY_CONFIG,
            )
            t.synchronize_device(mesh)
            phase("WEIGHT_UPLOAD_DONE", case=label)
            inputs = [
                t.from_torch(x.clone(), dtype=getattr(t, str(x.dtype).split(".")[-1]), layout=t.TILE_LAYOUT)
                for x in host_inputs
            ]
            value = t.from_torch(
                host_inputs[0].clone(),
                dtype=getattr(t, str(host_inputs.dtype).split(".")[-1]),
                layout=t.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=t.ReplicateTensorToMesh(mesh),
                memory_config=t.DRAM_MEMORY_CONFIG,
            )
            t.synchronize_device(mesh)
            phase("INPUT_UPLOAD_DONE", case=label)

            def reference():
                x = t.typecast(value, t.float32)
                product = t.multiply(x, weight)
                reduced = t.sum(
                    product, dim=-1, keepdim=True, compute_kernel_config=compute, fast_and_approximate_mode=False
                )
                raw = t.reshape(reduced, (1, 1, 1, o))
                return raw, t.typecast(raw, t.bfloat16)

            def invoke(fn):
                with guard():
                    return fn()

            precast_active = a.precast_input and value.dtype == t.bfloat16 and weight.dtype == t.float32
            for rows in row_choices:
                case = dict(
                    label=label,
                    shape=[o, k],
                    rows_per_worker=rows,
                    sfpu_iterations_per_face=2 if rows <= 4 else rows // 2,
                    sfpu_physical_rows=4 if rows <= 4 else rows,
                    zero_once=a.zero_once,
                    num_cores=a.num_cores,
                    weight_dtype=str(host_weight.dtype),
                    input_dtype=str(host_inputs.dtype),
                    precast_input_applied=precast_active,
                    native_input_dtype="float32" if precast_active else str(host_inputs.dtype).split(".")[-1],
                    weight_sha256=digest_tensor(host_weight),
                    inputs_sha256=digest_tensor(host_inputs),
                    provenance=provenance,
                    eager=[],
                    trace_checks=[],
                    timing={},
                )
                report["cases"].append(case)

                def candidate():
                    # This exact widening remains inside every warm/capture call;
                    # its device operation and temporary buffer are timed too.
                    native_value = t.typecast(value, t.float32) if precast_active else value
                    raw = exact_projection_q1(
                        native_value,
                        weight,
                        rows_per_worker=rows,
                        num_cores=a.num_cores,
                        raw_fp32=True,
                        zero_once=a.zero_once,
                    )
                    return raw, t.typecast(raw, t.bfloat16)

                # Native must be first execution, so its failure is unambiguous.
                phase("NATIVE_WARM_BEGIN", case=label, rows=rows)
                native = invoke(candidate)
                t.synchronize_device(mesh)
                phase("NATIVE_WARM_SYNCED", case=label, rows=rows)
                got = tuple(read(x) for x in native)
                phase("NATIVE_WARM_READ_DONE", case=label, rows=rows)
                phase("REFERENCE_WARM_BEGIN", case=label, rows=rows)
                baseline = invoke(reference)
                t.synchronize_device(mesh)
                expected = tuple(read(x) for x in baseline)
                case["warm"] = dict(raw=same(expected[0], got[0]), bf16=same(expected[1], got[1]))
                phase("REFERENCE_WARM_DONE", case=label, rows=rows)
                assert all(x["passed"] for x in case["warm"].values()), case["warm"]
                for i, host_input in enumerate(inputs):
                    t.copy_host_to_device_tensor(host_input, value)
                    t.synchronize_device(mesh)
                    actual = invoke(candidate)
                    expected = invoke(reference)
                    t.synchronize_device(mesh)
                    check = dict(
                        input=i,
                        raw=same(read(expected[0]), read(actual[0])),
                        bf16=same(read(expected[1]), read(actual[1])),
                    )
                    case["eager"].append(check)
                    save()
                    assert check["raw"]["passed"] and check["bf16"]["passed"], check
                if not a.skip_trace:
                    captures = {}
                    for name, fn in (("candidate", candidate), ("selected", reference)):
                        phase("CAPTURE_BEGIN", case=label, rows=rows, implementation=name)
                        trace = t.begin_trace_capture(mesh, cq_id=0)
                        traces.append(trace)
                        out = invoke(fn)
                        t.end_trace_capture(mesh, trace, cq_id=0)
                        # These are observable trace outputs deliberately overwritten
                        # by every replay, not retained scratch/intermediate state.
                        for tensor in out:
                            trace_allocation_tracker.acknowledge_corruptible(tensor)
                        captures[name] = (trace, out)
                        phase("CAPTURE_DONE", case=label, rows=rows, implementation=name)
                        if trace_allocation_tracker.TRACE_ALLOC_TRACKING:
                            phase("ALLOCATION_GUARD_CONTROL_BEGIN", case=label, rows=rows, implementation=name)
                            sentinel = t.empty_like(value)
                            try:
                                try:
                                    t.execute_trace(mesh, trace, cq_id=0, blocking=False)
                                except RuntimeError as error:
                                    assert "still alive before trace replay" in str(error), str(error)
                                    case.setdefault("allocation_guard_positive_controls", {})[name] = dict(
                                        rejected=True, error=str(error)
                                    )
                                else:
                                    raise AssertionError("Tracker did not reject retained post-capture sentinel")
                            finally:
                                t.deallocate(sentinel)
                                del sentinel
                            phase("ALLOCATION_GUARD_CONTROL_PASSED", case=label, rows=rows, implementation=name)
                    # Separate captured outputs remain live until both traces release.
                    for i in (0, len(inputs) - 1, 1, 0):
                        t.copy_host_to_device_tensor(inputs[i], value)
                        for trace, _ in captures.values():
                            t.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        t.synchronize_device(mesh)
                        actual = captures["candidate"][1]
                        expected = invoke(reference)  # fresh selected eager oracle after input refresh
                        t.synchronize_device(mesh)
                        checks = dict(
                            input=i,
                            raw=same(read(expected[0]), read(actual[0])),
                            bf16=same(read(expected[1]), read(actual[1])),
                            selected_trace=same(read(expected[0]), read(captures["selected"][1][0])),
                        )
                        case["trace_checks"].append(checks)
                        save()
                        assert all(v["passed"] for n, v in checks.items() if n != "input"), checks
                        del expected, actual
                    phase("TIMING_BEGIN", case=label, rows=rows)
                    for name, (trace, _) in captures.items():
                        for _ in range(3):
                            t.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        t.synchronize_device(mesh)
                        times = []
                        for _ in range(a.samples):
                            start = time.perf_counter()
                            for _ in range(a.iterations):
                                t.execute_trace(mesh, trace, cq_id=0, blocking=False)
                            t.synchronize_device(mesh)
                            times.append((time.perf_counter() - start) * 1000 / a.iterations)
                        case["timing"][name] = dict(
                            milliseconds_per_replay=times,
                            includes=(
                                "exact BF16-to-FP32 input widening + raw projection + explicit BF16 cast"
                                if name == "candidate" and precast_active
                                else "raw projection + explicit BF16 cast"
                            ),
                            iterations=a.iterations,
                            readback_in_timing=False,
                        )
                    for trace, _ in captures.values():
                        t.release_trace(mesh, trace)
                        traces.remove(trace)
                    captures.clear()
                    phase("TIMING_DONE", case=label, rows=rows)
                case["passed"] = True
                save()
        report["passed"] = True
        phase("ALL_CHECKS_PASSED")
    except BaseException as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
        save()
        raise
    finally:
        if mesh is not None:
            for trace in traces:
                t.release_trace(mesh, trace)
            phase("MESH_CLOSE_BEGIN")
            t.close_mesh_device(mesh)
            phase("MESH_CLOSE_DONE")
        save()


if __name__ == "__main__":
    main()
