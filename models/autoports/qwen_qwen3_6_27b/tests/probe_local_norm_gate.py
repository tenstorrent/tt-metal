# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Parent-serialized TP4 fused recurrent epilogue probe; CPU mode imports no TTNN."""

import argparse
import ast
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from models.autoports.qwen_qwen3_6_27b.tt import local_norm_gate_candidate as candidate

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
FROZEN_SHA = "315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22"


def provenance():
    paths = [
        Path(__file__),
        ROOT / "tt/local_norm_gate_candidate.py",
        ROOT / "tt/local_affine_gate_candidate.py",
        ROOT / "tt/optimized_decoder.py",
        ROOT / "tt/multichip_decoder.py",
    ]
    paths += [
        REPO / p
        for p in (
            "ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm.cpp",
            "ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op_multi_core.cpp",
            "ttnn/cpp/ttnn/operations/normalization/kernel_util/compute/numeric.h",
            "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp",
        )
    ]
    result = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert hashlib.sha256((ROOT / "tt/optimized_decoder.py").read_bytes()).hexdigest() == FROZEN_SHA
    return result


def cpu_check():
    from collections import defaultdict
    from types import SimpleNamespace as NS

    proof = candidate.cpu_check()
    node = next(
        n
        for n in ast.parse((ROOT / "tt/local_norm_gate_candidate.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "local_norm_gate"
    )
    node.body = [n for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    fake = NS(bfloat16="bf16", float32="fp32", TILE_LAYOUT="tile", DRAM_MEMORY_CONFIG="dram")
    fake.CoreCoord = lambda x, y: NS(x=x, y=y)
    fake.CoreRange = lambda a, b: (a, b)
    fake.CoreRangeSet = list
    fake.RuntimeArgs = lambda: defaultdict(dict)
    for name in (
        "ComputeConfigDescriptor",
        "ReaderConfigDescriptor",
        "WriterConfigDescriptor",
        "CBDescriptor",
        "CBFormatDescriptor",
        "ProgramDescriptor",
    ):
        setattr(fake, name, lambda **kw: NS(**kw))
    fake.KernelDescriptor = lambda **kw: NS(**kw)
    fake.KernelDescriptor.SourceType = NS(SOURCE_CODE="source")
    fake.MathFidelity = NS(HiFi4="HiFi4")
    fake.UnpackToDestMode = NS(Default="default", UnpackToDestFp32="fp32_direct")
    fake.TensorAccessorArgs = lambda t: NS(get_compile_time_args=lambda: [t.buffer_address(), t.shape[-1]])
    captured = []
    fake.generic_op = lambda ts, desc: captured.append((ts, desc))
    mesh = NS(compute_with_storage_grid_size=lambda: NS(x=11, y=10))

    def tensor(shape, address, dtype="bf16"):
        return NS(
            shape=shape,
            padded_shape=(1, math.ceil(shape[1] / 32) * 32, shape[2]),
            dtype=dtype,
            layout="tile",
            device=lambda: mesh,
            memory_config=lambda: "dram",
            buffer_address=lambda: address,
        )

    namespace = vars(candidate).copy() | {"ttnn": fake}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<norm-gate-descriptor>", "exec"), namespace)
    checks = 0
    for b in range(1, 33):
        for ownership in ("head", "tile"):
            for cores in (1, 7, 110):
                ts = [tensor((1, b * 12, 128), 100, "fp32"), tensor((1, 1, 128), 200), tensor((1, b, 4160), 300)]
                out = tensor((1, b, 1536), 400)
                assert namespace["local_norm_gate"](*ts, output=out, ownership=ownership, cores=cores) is out
                inputs, desc = captured.pop()
                reader, writer, compute = desc.kernels
                assert [t.buffer_address() for t in inputs] == [100, 200, 300, 400]
                assert reader.compile_time_args[:2] == [
                    int(ownership == "tile"),
                    candidate.struct.unpack("<I", candidate.struct.pack("<f", 1e-6))[0],
                ]
                tasks = [
                    task
                    for rows in writer.runtime_args.values()
                    for _, start, count in rows.values()
                    for task in range(start, start + count)
                ]
                assert sorted(tasks) == list(range(12 if ownership == "head" else 48))
                assert sum(cb.total_size for cb in desc.cbs) == proof["cb_bytes_per_core"]
                assert compute.config.unpack_to_dest_mode[0] == "fp32_direct"
                assert all(compute.config.unpack_to_dest_mode[i] == "default" for i in (2, 5, 6))
                assert compute.config.fp32_dest_acc_en and not compute.config.math_approx_mode
                checks += 1
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    return proof | {"descriptor_checks": checks, "source_sha256": provenance()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-check", action="store_true")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 3, 17, 32])
    ap.add_argument("--ownership", nargs="+", choices=["head", "tile"], default=["head", "tile"])
    ap.add_argument("--modes", nargs="+", choices=["normal", "zero", "large"], default=["normal", "zero", "large"])
    ap.add_argument("--cores", type=int)
    ap.add_argument("--low-cores", type=int, default=1, help="Extra B3 normal wrap case;0 disables")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--replays", type=int, default=50)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--epsilon", type=float, default=1e-6)
    ap.add_argument("--require-exact", action="store_true", help="Additional gate beyond unchanged PCC>=.995")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.cpu_check:
        print(json.dumps(cpu_check(), indent=2))
        return
    assert args.output and not args.output.exists()
    assert args.output.resolve().is_relative_to((ROOT / "doc/optimized_multichip_decoder").resolve())
    assert all(1 <= b <= 32 for b in args.batches) and args.steps >= 2 and args.low_cores >= 0
    assert args.cores is None or args.cores >= 1
    assert min(args.warmup, args.replays, args.samples) > 0 and math.isfinite(args.epsilon) and args.epsilon > 0
    watcher = "TT_METAL_WATCHER" in os.environ
    profiler = os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0"
    tracking = os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0") == "1"
    assert not (watcher and profiler)
    instrumented = watcher or profiler or tracking
    import torch
    import ttnn
    from models.autoports.qwen_qwen3_6_27b.tt.local_affine_gate_candidate import local_affine_gate
    from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc

    torch.set_num_threads(4)
    report = {
        "status": "running",
        "mesh": [1, 4],
        "arguments": vars(args) | {"output": str(args.output)},
        "source_sha256": provenance(),
        "watcher": watcher,
        "profiler": profiler,
        "allocation_tracking": tracking,
        "skip_program_cache": os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "1",
        "timings_skipped": instrumented,
        "cases": [],
        "trace_contract": "All external fixtures/outputs and program warmups precede first capture. Native RMS transient intermediate is scoped inside reference/capture. Native and both candidates receive matched traced timing; no allocation exemptions.",
        "input_contract": "Synthetic actual FP32 readout, never pre-rounded to BF16; recurrence/state independent.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    mesh = None
    save()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)
        ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        def upload(values, dtype):
            return ttnn.from_torch(
                torch.cat(values, -1).contiguous(),
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            )

        def read(t):
            return [ttnn.to_torch(v).contiguous() for v in ttnn.get_device_tensors(t)]

        def diagnostics(actual, expected, batch, *, exact=False):
            rows = []
            for rank, (a, b) in enumerate(zip(actual, expected)):
                assert a.dtype == b.dtype == torch.bfloat16 and a.shape == b.shape
                a, b = a.reshape(batch, 12, 128), b.reshape(batch, 12, 128)
                for user in range(batch):
                    for head in range(12):
                        x, y = a[user, head], b[user, head]
                        finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
                        score = pcc(x, y) if finite else -1.0
                        bits = int((x.view(torch.int16) != y.view(torch.int16)).sum())
                        rows.append(
                            {
                                "rank": rank,
                                "user": user,
                                "head": head,
                                "pcc": score,
                                "bits": bits,
                                "max_abs": float((x.float() - y.float()).abs().max()) if finite else None,
                                "finite": finite,
                            }
                        )
            result = {
                "minimum_pcc": min(r["pcc"] for r in rows),
                "bit_mismatches": sum(r["bits"] for r in rows),
                "maximum_abs": max((r["max_abs"] or 0) for r in rows),
                "finite": all(r["finite"] for r in rows),
                "details": [r for r in rows if r["bits"] or not r["finite"]],
            }
            return result

        def gate(result, *, exact=False):
            assert result["finite"] and result["minimum_pcc"] >= 0.995, result
            if exact:
                assert result["bit_mismatches"] == 0, result

        def run_case(batch, cores, mode):
            entry = {"batch": batch, "cores": cores, "mode": mode, "checks": [], "trace": []}
            report["cases"].append(entry)
            save()
            print(f"NORM_GATE B{batch} cores={cores} mode={mode}", flush=True)
            gs = [torch.Generator().manual_seed(137811 + batch * 19 + r) for r in range(4)]
            bank = []
            for step in range(args.steps):
                rs, ws, ps = [], [], []
                for rank, g in enumerate(gs):
                    r = torch.randn(1, batch * 12, 128, generator=g) * 0.5 + rank * 0.0625 + step * 0.0123
                    r += torch.arange(batch * 12).reshape(1, -1, 1) * 0.0013
                    w = (1 + torch.randn(1, 1, 128, generator=g) * 0.125).bfloat16()
                    p = (torch.randn(1, batch, 4160, generator=g) * 0.5 + rank * 0.03125 + step * 0.125).bfloat16()
                    if mode == "zero":
                        r.zero_()
                    if mode == "large":
                        scales = torch.tensor([1e-8, 1e-3, 1.0, 1e3, 1e8, 1e-5] * 2).repeat(batch).reshape(1, -1, 1)
                        r *= scales
                        p[:, :, 2560:4096] = ((torch.arange(1536) % 2) * 64 - 32).bfloat16()
                    rs.append(r.contiguous())
                    ws.append(w)
                    ps.append(p)
                bank.append([upload(rs, ttnn.float32), upload(ws, ttnn.bfloat16), upload(ps, ttnn.bfloat16)])
            inputs = [ttnn.empty_like(t) for t in bank[0]]
            output_shape = (1, batch, 1536)
            outputs = [
                ttnn.empty(
                    output_shape,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in args.ownership
            ]
            native_out = ttnn.empty_like(outputs[0])
            rounded = ttnn.empty_like(inputs[0], dtype=ttnn.bfloat16)
            # Every externally owned tensor and program is prepared before capture.
            allocations = [*inputs, *outputs, native_out, rounded, *(t for step in bank for t in step)]
            tids = []

            def update(step):
                for src, dst in zip(bank[step], inputs):
                    ttnn.copy(src, dst)

            def native():
                ttnn.typecast(inputs[0], ttnn.bfloat16, output_tensor=rounded)
                norm = ttnn.rms_norm(rounded, epsilon=args.epsilon, compute_kernel_config=ckc)
                return local_affine_gate(norm, inputs[1], inputs[2], output=native_out)

            def invoke(i):
                return candidate.local_norm_gate(
                    *inputs, epsilon=args.epsilon, ownership=args.ownership[i], output=outputs[i], cores=cores
                )

            eager = []
            try:
                for step in range(args.steps):
                    update(step)
                    native()
                    reference = read(native_out)
                    values = []
                    for i, name in enumerate(args.ownership):
                        invoke(i)
                        actual = read(outputs[i])
                        values.append(actual)
                        d = diagnostics(actual, reference, batch)
                        entry["checks"].append({"step": step, "ownership": name, **d})
                        save()
                        gate(d, exact=args.require_exact)
                    eager.append([reference, *values])
                    assert all(torch.equal(a, b) for t, s in zip(inputs, bank[step]) for a, b in zip(read(t), read(s)))
                # Same program, newly bound addresses; all old buffers stay alive.
                for i, name in enumerate(args.ownership):
                    candidate.local_norm_gate(
                        *bank[0], epsilon=args.epsilon, ownership=name, output=outputs[i], cores=cores
                    )
                    d = diagnostics(read(outputs[i]), eager[0][i + 1], batch)
                    entry.setdefault("rebind", []).append({"ownership": name, **d})
                    save()
                    gate(d, exact=True)
                update(0)
                for _ in range(args.warmup):
                    native()
                    for i in range(len(outputs)):
                        invoke(i)
                ttnn.synchronize_device(mesh)
                cache_entries = mesh.num_program_cache_entries()
                for i in range(len(outputs) + 1):
                    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                    native() if i == 0 else invoke(i - 1)
                    ttnn.end_trace_capture(mesh, tid, cq_id=0)
                    tids.append(tid)
                assert mesh.num_program_cache_entries() == cache_entries, "Program allocated during capture"
                for step in range(args.steps):
                    update(step)
                    for i, tid in enumerate(tids):
                        for _ in range(args.replays):
                            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        d = diagnostics(read([native_out, *outputs][i]), eager[step][i], batch)
                        entry["trace"].append({"step": step, "path": ["native", *args.ownership][i], **d})
                        save()
                        gate(d, exact=True)
                if not instrumented:
                    update(0)
                    times = [[] for _ in tids]
                    for sample in range(args.samples):
                        order = range(len(tids)) if sample % 2 == 0 else reversed(range(len(tids)))
                        for i in order:
                            ttnn.synchronize_device(mesh)
                            before = time.perf_counter_ns()
                            for _ in range(args.replays):
                                ttnn.execute_trace(mesh, tids[i], cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh)
                            times[i].append((time.perf_counter_ns() - before) / 1000 / args.replays)
                    entry["warmed_traced_us"] = {
                        name: {"samples": s, "median": statistics.median(s)}
                        for name, s in zip(["native", *args.ownership], times)
                    }
            finally:
                for tid in tids:
                    ttnn.release_trace(mesh, tid)
            assert len(allocations) > 0
            save()

        for b in args.batches:
            for mode in args.modes:
                run_case(b, args.cores, mode)
        if args.low_cores:
            run_case(3, args.low_cores, "normal")
        if profiler:
            ttnn.ReadDeviceProfiler(mesh)
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
